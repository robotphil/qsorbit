"""The receive path — Phase 2's vertical slice, wired into one object.

Everything this module needs already existed and was tested in isolation.
What was missing was the wiring, and the wiring is where the interesting
failures live, because it is the only place where the tracking side and
the receiving side have to agree about anything.

**A branch is one radio's whole chain.** :class:`Branch` owns a device's
stream, its Doppler tracker, its demodulator settings and its squelch;
:class:`ReceiveSession` owns a list of them, one speaker, and one
range-rate source. A single-device station is **one branch, not a
branchless special case** — two code paths would drift, and the one
exercised least would be the one every existing station uses.

Three things are per-branch and none of them look like it at first.
The **Doppler tracker**, because it is built against the centre
frequency *that* tuner actually reached, and two PLLs quantise a request
differently — the same predicted curve, two different corrections. The
**squelch**, because it is stateful, and one shared between two radios
would mix two signals' histories into one decision. And the **spectrum**,
because the waterfall shows what you are hearing.

**What runs where.** Two threads per branch plus one for the session,
and the division is not arbitrary:

``IqStream``'s reader
    One per branch, owned by :mod:`qsorbit.core.sdr.stream`. Reads that
    branch's device and fans each block out to its subscriptions.
    Nothing here touches it.

a demodulating thread per branch
    Owned by this module. Pulls :class:`~qsorbit.core.sdr.stream.TimedBlock`
    from that branch's ``"audio"`` subscription, asks its Doppler tracker
    where the downlink is *at that block's midpoint*, demodulates, and —
    if that branch currently holds the ear — writes the audio out. This
    is the closest thing here to a real-time path, so it shares a thread
    with nothing else: not with the rotor, whose serial round trips take
    0.15 s of RS-485 turnaround apiece, and **not with the other
    branch**, which would make each one's latency depend on the other's.

    Every branch runs the full chain whether or not anyone is listening.
    Skipping the work on a silent branch would save CPU and destroy the
    squelch measurements a combiner selects on — and would also make the
    cost of running two branches unmeasurable, which is the one thing a
    first dual-device run exists to find out.

the range-rate thread
    Feeds the Doppler tracker on its own cadence, from
    :class:`TargetRangeRate` -- the TLE and the observer's location,
    which is where a range rate actually comes from. One thread for the
    whole session, feeding every branch's tracker from the same sample:
    a range rate is a property of the pass, not of any radio. **No rotor
    is involved and none ever was**: for a while this thread also held the
    rotor's tick, by way of a range-rate source that ticked the loop to
    get its number, and the side effect was that the rotor was commanded
    at *this* cadence rather than at the one its tracking profile
    declared. The tick now lives in
    :class:`~qsorbit.core.tracking_thread.TrackingThread`, which is what
    makes "this module does not insist on owning the tracking side" true
    rather than aspirational.

A ``SpectrumStream``, when one is given, gets the listening branch's
``"waterfall"`` subscription and runs its own worker as it always has.

**The rotor is optional, and that is a design statement rather than a
convenience.** Doppler correction needs a range rate, and a range rate
comes from the TLE and the observer's location — not from the rotor. So
the entire radio job runs with nothing connected to COM5, and moving the
antenna is something added on top. On a bench day where several things
could be wrong at once, a rotor fault therefore does not cost you the
pass. It also matches ``point``'s standing asymmetry: computing is the
default, moving is opt-in.

**The trackers are primed before any reader starts.**
:meth:`~qsorbit.core.dsp.tuning.DopplerTracker.offset_at` raises if it
has never been given a range rate, and the demodulating thread can reach
its first block before the tracking side has produced anything. Rather
than skip those blocks and count them, :meth:`ReceiveSession.start`
takes one sample up front and gives it to every branch. Priming deletes
the race; counting would only have measured it.

**Nothing here counts what something else already counts.** Stale Doppler
queries live in :class:`~qsorbit.core.dsp.tuning.DopplerStats`, buffer
drops in :class:`~qsorbit.core.sdr.stream.StreamStats`, underruns in
:class:`~qsorbit.core.dsp.audio.AudioStats`. Duplicating any of them here
would only create two numbers with two chances to disagree — the same
reasoning :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStreamStats`
gives for not counting the IQ side's drops a second time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from qsorbit.core.combiner import (
    BranchReading,
    BranchSelector,
    SelectorStats,
    pair_simultaneous,
)
from qsorbit.core.dsp.audio import AudioOutput, AudioStats
from qsorbit.core.dsp.demod import NbfmConfig, demodulate_nbfm
from qsorbit.core.dsp.iq import unpack_uint8_iq
from qsorbit.core.dsp.spectrum_stream import SpectrumStream, SpectrumStreamStats
from qsorbit.core.dsp.squelch import NoiseSquelch, SquelchStats
from qsorbit.core.dsp.tuning import DopplerStats, DopplerTracker
from qsorbit.core.quieting_log import QuietingLog
from qsorbit.core.sdr.stream import IqStream, StreamStats, TimedBlock
from qsorbit.core.tracker.observer import ObserverLocation
from qsorbit.core.tracker.target import Target

if TYPE_CHECKING:
    from qsorbit.core.recorder import IqRecorder
    from qsorbit.core.sdr.stream import IqSubscription

#: Subscription name for the demodulating consumer.
AUDIO_SUBSCRIBER: Final = "audio"

#: Subscription name for the spectrum/waterfall consumer.
WATERFALL_SUBSCRIBER: Final = "waterfall"

#: Subscription name for the IQ-recording consumer. Made only when a run
#: asks to record, so an ordinary run pays nothing for a consumer that
#: does not exist -- the same conditional-subscription discipline the
#: waterfall follows (Session 24's phantom-drop lesson).
RECORDER_SUBSCRIBER: Final = "recorder"

#: Seconds between range-rate samples when this module drives the
#: tracking side itself. One second matches
#: :class:`~qsorbit.core.pointing.TrackingLoop`'s own default cadence and
#: :class:`~qsorbit.ui.readout_widget.ReadoutWidget`'s poll interval, so
#: headless and windowed runs feed the tracker at the same rate and their
#: measurements stay comparable.
DEFAULT_TRACKING_INTERVAL_S: Final = 1.0

#: How long a branch may go without producing a block before the
#: combiner stops treating its last measurement as usable. At 2.048 Msps
#: a block is about 64 ms, so two seconds is roughly thirty blocks --
#: long enough that an ordinary hiccup does not drop a branch out of
#: contention, short enough that a radio which has actually stopped
#: cannot hold the speaker for a meaningful part of a pass.
DEFAULT_STALE_AFTER_S: Final = 2.0

#: How close two branches' block midpoints must be to count as the same
#: moment, as a fraction of one block period. Half a block cleanly
#: separates aligned blocks (~1 ms apart, measured on the acceptance
#: capture) from the block behind (~one full period), which is the skew
#: that manufactured a switch on 2026-09-06. See
#: :func:`~qsorbit.core.combiner.pair_simultaneous`.
PAIRING_WINDOW_FRACTION: Final = 0.5

#: How long :meth:`ReceiveSession.stop` waits for each of its threads.
#: The demodulating thread checks for the stop signal between blocks, so
#: normal latency is one block — about 64 ms at 2.048 Msps.
DEFAULT_JOIN_TIMEOUT_S: Final = 5.0


def _utc_now() -> datetime:
    """The current instant, timezone-aware. Matches the rest of the project."""
    return datetime.now(UTC)


class RangeRateSource(Protocol):
    """Where the Doppler tracker's range-rate samples come from.

    Declared structurally rather than as a base class, matching
    :class:`~qsorbit.core.tracker.Target` and
    :class:`~qsorbit.ui.waterfall_widget.FrameSource`: a test double
    satisfies it by having the two methods, without importing anything.

    The split between the two methods is the whole reason this protocol
    exists. :meth:`prime` must always produce a sample, because it is
    what removes the race described in the module docstring.
    :meth:`sample` is allowed to say "nothing new yet", because a source
    that merely *follows* a loop somebody else is ticking genuinely has
    nothing to report between ticks.
    """

    def prime(self) -> tuple[datetime, float]:
        """Produce one sample now, before anything starts streaming.

        Returns:
            ``(time, range_rate_km_s)``, positive when receding.
        """
        ...

    def sample(self) -> tuple[datetime, float] | None:
        """The next sample, or ``None`` if none has arrived yet."""
        ...


class TargetRangeRate:
    """Range rates computed straight from the target. No rotor involved.

    What ``receive`` uses when the antenna is not being moved. Every
    sample is available on demand, so :meth:`prime` and :meth:`sample`
    are the same operation.

    Args:
        target: What is being received from.
        observer: The ground station's location.
        now: Clock, injected for tests.
    """

    def __init__(
        self,
        target: Target,
        observer: ObserverLocation,
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._target = target
        self._observer = observer
        self._now = now

    def prime(self) -> tuple[datetime, float]:
        """Compute a sample now."""
        return self.sample()

    def sample(self) -> tuple[datetime, float]:
        """Compute a sample now. Never ``None`` — nothing can be pending."""
        when = self._now()
        state = self._target.topocentric_state(self._observer, when)
        return when, state.range_rate_km_s


@dataclass(frozen=True)
class BranchStats:
    """What one receive branch did, in the pieces that have owners.

    Args:
        label: The branch's name, from station config. Names the
            antenna rather than the dongle, because that is what a
            reader of this report is comparing.
        listened: Whether this branch's audio reached the speaker.
            Recorded because a branch that was measured and a branch
            that was heard are different claims, and with no combiner
            yet only one of them can be both.
        blocks_demodulated: Blocks that went through
            :func:`~qsorbit.core.dsp.demod.demodulate_nbfm`.
        stream: The IQ side for this branch's own device, including
            per-consumer drop accounting.
        doppler: Correction range and stale-query count. One per
            branch and not one per session, because a tracker is built
            against the centre frequency *its own* tuner reached — two
            PLLs quantise differently, and two dongles have two
            crystals.
        squelch: Present only if this branch had a squelch.
    """

    label: str
    listened: bool
    blocks_demodulated: int
    stream: StreamStats
    doppler: DopplerStats
    squelch: SquelchStats | None

    def describe(self) -> str:
        """Summarise this branch, one owner per section."""
        ear = " (heard)" if self.listened else ""
        squelch = (
            self.squelch.describe()
            if self.squelch is not None
            else "squelch: off, so no quieting was measured this run."
        )
        return (
            f"branch {self.label}{ear}: {self.blocks_demodulated:,} block(s) demodulated\n"
            f"{self.stream.describe()}"
            f"{self.doppler.describe()}\n"
            f"{squelch}"
        )


@dataclass(frozen=True)
class ReceiveStats:
    """What one receive session did, in the pieces that have owners.

    Args:
        branches: One entry per receive branch, in the order they were
            declared. A single-device station has exactly one, which is
            why there is no separate single-device shape here: two code
            paths would drift, and the one that got exercised less
            would be the one every existing station uses.
        range_rate_updates: Samples handed to the Doppler trackers,
            including the priming one. Session-level because there is
            one predicted curve and one range-rate thread however many
            branches consume it.
        audio: Playback, including underruns. Session-level because
            there is one speaker.
        spectrum: Present only if a waterfall was being fed.
        combiner: Present only if a combiner was running. ``None`` means
            the branch was fixed for the whole run -- which is a
            different fact from "the combiner never switched", and the
            report says which.
        stopped_cleanly: Whether every thread this module started exited
            within its join timeout.
    """

    branches: tuple[BranchStats, ...]
    range_rate_updates: int
    audio: AudioStats
    spectrum: SpectrumStreamStats | None
    combiner: SelectorStats | None
    stopped_cleanly: bool

    @property
    def blocks_demodulated(self) -> int:
        """Blocks demodulated across every branch.

        A convenience over :attr:`branches`, kept because "did the
        receive path do any work at all" is a question worth answering
        without summing by hand. It is deliberately **not** the number
        that reached the speaker — with two branches running, most of
        this was demodulated and measured and never heard.
        """
        return sum(branch.blocks_demodulated for branch in self.branches)

    def describe(self) -> str:
        """Summarise the whole slice, one owner per section.

        Printed at the end of a bench run, so this *is* the measurement
        record. Sections that were not in use say so rather than being
        omitted: a missing line reads as zero, and "the squelch was off"
        and "the squelch never opened" are different facts. With more
        than one branch each gets its own section, because a combined
        number would hide exactly the per-branch difference a dual-SDR
        run exists to measure.
        """
        clean = "" if self.stopped_cleanly else "receive: threads DID NOT stop cleanly\n"
        spectrum = (
            self.spectrum.describe()
            if self.spectrum is not None
            else "spectrum: no waterfall was attached this run.\n"
        )
        branches = "".join(
            f"\n--- {branch.label} ---\n{branch.describe()}\n" for branch in self.branches
        )
        combiner = (
            self.combiner.describe()
            if self.combiner is not None
            else "combiner: off, so one branch held the speaker for the whole run."
        )
        return (
            f"{clean}"
            f"receive: {len(self.branches)} branch(es), "
            f"{self.range_rate_updates:,} range-rate update(s)\n"
            f"{branches}"
            f"\n--- audio ---\n{self.audio.describe()}\n"
            f"\n--- combiner ---\n{combiner}\n"
            f"\n--- spectrum ---\n{spectrum}"
        )


class Branch:
    """One complete receive chain: a device, a demodulator, and a name.

    A branch is what :class:`ReceiveSession` is a list of. It owns
    everything that is *per-radio* — the stream, the Doppler tracker
    built against that tuner's own centre frequency, the demodulator
    settings, and the squelch, which is stateful and cannot be shared.
    It deliberately owns no thread: threads, the stop signal, and error
    recording stay with the session, so that :meth:`demodulate` is a
    plain function of one block and can be tested without starting
    anything.

    **A single-device station is one branch, not a special case.** The
    alternative — a branch-free path alongside a branched one — would
    leave the route every existing station takes as the one exercised
    least.

    Args:
        label: What to call this branch in reports and on screen. From
            station config, where it names the antenna.
        stream: An :class:`~qsorbit.core.sdr.stream.IqStream` over this
            branch's configured device. **Must not have been started or
            subscribed to** — the subscription is made here, and
            subscriptions have to exist before the reader does.
        nbfm: Demodulation settings. ``channel_offset_hz`` is replaced
            per block with the Doppler-corrected offset, so whatever it
            holds here is ignored; everything else is used as given.
        doppler: This branch's tracker, built against the centre
            frequency **its own** tuner actually reached.
        squelch: Optional noise gate for this branch, off by default.
            One per branch: it is stateful, and sharing one between two
            radios would mix two signals' histories into one decision.
        mute_squelch: Whether a closed gate actually silences this
            branch's audio. Ignored when ``squelch`` is ``None``.
        spectrum_factory: Optional. When given, this branch also takes
            the ``"waterfall"`` subscription and drives a spectrum
            stream. Only the branch being listened to is given one, so
            a branch nobody is watching never pays for frames nobody
            sees — the same reasoning that made the subscription
            conditional in the first place.
        log: Optional :class:`~qsorbit.core.quieting_log.QuietingLog`,
            shared with every other branch in the session. When given,
            this branch writes one row per block. Nothing is written if
            it has no ``squelch``, because there is no measurement to
            write -- see :meth:`demodulate`.
        listened: Whether this branch's audio reaches the speaker.
            Mutable, and set through :meth:`ReceiveSession.listen_to`
            rather than directly, because exactly one branch may hold
            it at a time.
    """

    def __init__(
        self,
        *,
        label: str,
        stream: IqStream,
        nbfm: NbfmConfig,
        doppler: DopplerTracker,
        squelch: NoiseSquelch | None = None,
        mute_squelch: bool = True,
        spectrum_factory: Callable[[Iterable[bytes]], SpectrumStream] | None = None,
        log: QuietingLog | None = None,
        recorder_factory: Callable[[IqSubscription], IqRecorder] | None = None,
    ) -> None:
        self.label = label
        self.listened = False
        self._stream = stream
        self._nbfm = nbfm
        self._doppler = doppler
        self._squelch = squelch
        self._mute_squelch = mute_squelch
        self._log = log

        # Subscribed here rather than at start(), because subscriptions
        # must exist before the reader thread does and a caller is
        # entitled to hold the waterfall subscription before starting.
        self._audio_blocks = stream.subscribe(AUDIO_SUBSCRIBER)
        # The recorder is another consumer, made only when a run asks to
        # record -- built exactly like the waterfall, and for the same
        # reason conditional: an unused subscription would be offered every
        # block and evict them, reporting phantom loss (Session 24). The
        # branch makes the subscription (the stream is its own); the
        # session owns the thread that drains it, the same split as the
        # demodulating consumer.
        if recorder_factory is None:
            self._recorder = None
        else:
            self._recorder = recorder_factory(stream.subscribe(RECORDER_SUBSCRIBER))
        # The waterfall subscription is made only when something will
        # actually drain it. It used to be unconditional, so a headless
        # run offered every block to a consumer that did not exist and
        # the bounded deque evicted them in turn: a 60-second headless
        # `receive` reported "453 block(s) dropped (118,751,232 bytes)"
        # with nothing whatever wrong (Session 24). Harmless, and it
        # reads as catastrophic data loss -- and this project's own rule
        # is that "off" and "broken" must never look the same.
        if spectrum_factory is None:
            self._waterfall_blocks = None
            self._spectrum = None
        else:
            self._waterfall_blocks = stream.subscribe(WATERFALL_SUBSCRIBER)
            self._spectrum = spectrum_factory(self._waterfall_blocks.blocks())

        self._lock = threading.Lock()
        self._blocks_demodulated = 0
        # Monotonic, not wall clock: this is only ever used for an
        # elapsed comparison, and monotonic cannot be moved by an NTP
        # step in the middle of a pass.
        self._last_block_at: float | None = None
        # The most recent quieting measurement paired with the block
        # midpoint it describes -- the combiner needs both together to
        # tell whether two branches' readings are simultaneous. Stored
        # atomically under the lock so the value and its timestamp can
        # never be read torn apart.
        self._last_reading: BranchReading | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spectrum(self) -> SpectrumStream | None:
        """This branch's spectrum stream, or ``None`` if it has none."""
        return self._spectrum

    @property
    def recorder(self) -> IqRecorder | None:
        """This branch's IQ recorder, or ``None`` if this run records nothing.

        The session runs its :meth:`~qsorbit.core.recorder.IqRecorder.run`
        on a thread of its own -- the branch makes the subscription, the
        session owns the thread, the standing division here.
        """
        return self._recorder

    @property
    def doppler(self) -> DopplerTracker:
        """This branch's Doppler tracker, for the range-rate thread to feed."""
        return self._doppler

    @property
    def stats(self) -> BranchStats:
        """This branch's contribution to the run's report."""
        with self._lock:
            blocks = self._blocks_demodulated
        return BranchStats(
            label=self.label,
            listened=self.listened,
            blocks_demodulated=blocks,
            stream=self._stream.stats,
            doppler=self._doppler.stats,
            squelch=self._squelch.stats if self._squelch is not None else None,
        )

    @property
    def live_quieting_db(self) -> float | None:
        """This branch's most recent quieting measurement, or ``None``.

        See :attr:`ReceiveSession.live_quieting_db` for the polling
        contract and why this is not guarded by a lock.
        """
        if self._squelch is None:
            return None
        return self._squelch.stats.last_quieting_db

    @property
    def has_produced(self) -> bool:
        """Whether this branch has ever demodulated a block.

        The distinction the combiner needs at startup: a branch that has
        not produced yet is not a branch that has stopped. ``None`` from
        :meth:`latest_reading` cannot tell the two apart -- both read as
        "no opinion" -- so the session gates on this instead, holding the
        ear on the branch it started on until every branch is under way.
        Monotonic: once a block has arrived this is true for the rest of
        the run, so a branch that later goes silent is a *stale* branch,
        judged by :meth:`latest_reading` as before.
        """
        with self._lock:
            return self._last_block_at is not None

    def latest_reading(self, *, now: float, stale_after_s: float) -> BranchReading | None:
        """This branch's most recent reading, or ``None`` if it has stopped.

        Args:
            now: A ``time.monotonic()`` reading.
            stale_after_s: How long without a block counts as stopped.

        Returns:
            The most recent quieting measurement paired with the block
            midpoint it describes, or ``None``.

        **The staleness check is the point of this method existing at
        all.** A branch whose radio has died keeps its last measurement
        forever, and by value alone a stale reading is indistinguishable
        from a live one, so a combiner reading it directly could hold
        the speaker on a dead radio for the rest of a pass with a
        plausible number on the meter the whole time. ``None`` here is
        what lets the selector move the ear away instead.

        The value and its block midpoint are read together under the
        lock, so the combiner can never pair one branch's quieting with
        another branch's timestamp, which is exactly the mistake the
        simultaneity guard exists to prevent and would be a poor thing
        to reintroduce through a torn read.
        """
        with self._lock:
            last = self._last_block_at
            reading = self._last_reading
        if last is None or (now - last) > stale_after_s:
            return None
        return reading

    def fresh_quieting_db(self, *, now: float, stale_after_s: float) -> float | None:
        """This branch's quieting value alone, or ``None`` if it has stopped.

        A thin wrapper over :meth:`latest_reading` for callers that want
        the magnitude without the timestamp (the per-branch readout, and
        the tests that predate the combiner's simultaneity guard). The
        combiner itself takes the whole :class:`BranchReading`, because
        it needs the block midpoint to pair branches.
        """
        reading = self.latest_reading(now=now, stale_after_s=stale_after_s)
        return None if reading is None else reading.quieting_db

    @property
    def live_squelch_open(self) -> bool | None:
        """Whether this branch's gate is open right now, or ``None``."""
        if self._squelch is None:
            return None
        return self._squelch.is_open

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """Where this branch's downlink sits in RF right now, or ``None``."""
        offset_hz = self._doppler.stats.last_offset_hz
        if offset_hz is None:
            return None
        return self._doppler.center_hz + offset_hz

    # ------------------------------------------------------------------
    # Lifecycle and work
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start this branch's reader and, if it has one, its spectrum."""
        # Started here rather than left to whichever consumer reaches its
        # first block first. Both would call the same start-if-needed
        # path, and starting it explicitly means the reader is running
        # before either consumer exists rather than as a side effect of
        # one of them.
        self._stream.start()
        if self._spectrum is not None:
            self._spectrum.start()

    def stop_reading(self) -> None:
        """Stop the reader and the spectrum worker.

        Separate from joining the demodulating thread, and called for
        every branch before any thread is joined, so that no branch is
        still being fed while another is being waited on.

        Returns nothing, because neither owner forgets: an
        :class:`~qsorbit.core.sdr.stream.IqStream` keeps reporting the
        run's statistics after it has stopped, and so does a
        :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream`. A
        cached copy here would be a second place for the same numbers to
        live, with a second chance to disagree.
        """
        self._stream.stop()
        if self._spectrum is not None:
            self._spectrum.stop()

    def timed_blocks(self) -> Iterator[TimedBlock]:
        """This branch's audio-side blocks, as they arrive."""
        return self._audio_blocks.timed_blocks()

    def demodulate(self, block: TimedBlock) -> np.ndarray:
        """Demodulate one block at its own Doppler-corrected offset.

        Args:
            block: One block from :meth:`timed_blocks`.

        Returns:
            Mono audio at ``nbfm.audio_rate_hz``. Returned rather than
            written anywhere, because whether this branch is the one
            being heard is the session's decision and not this
            object's — and a branch nobody is listening to still has to
            do all of this, or its squelch metrics would mean nothing.

        Writes one row to the quieting log, if there is one, **after**
        demodulating rather than before: the squelch is updated by
        :func:`~qsorbit.core.dsp.demod.demodulate_nbfm` as it runs, so
        reading it first would record the previous block's measurement
        against this block's timestamp. The row is stamped with the
        block's own midpoint, which is the instant the measurement
        describes.

        A branch with no squelch writes nothing at all rather than a
        row of zeroes -- "not measured" and "measured as zero" are
        different facts, and a combiner sized from the second would be
        sized from nothing.
        """
        # The block's MIDPOINT, not either edge: it removes a
        # systematic half-block bias for free, and TimedBlock computes
        # it so no caller can get the sign wrong.
        offset_hz = self._doppler.offset_at(block.midpoint)
        config = replace(self._nbfm, channel_offset_hz=offset_hz)
        audio = demodulate_nbfm(
            unpack_uint8_iq(block.data),
            config,
            squelch=self._squelch,
            mute=self._mute_squelch,
        )
        with self._lock:
            self._blocks_demodulated += 1
            self._last_block_at = time.monotonic()
            if self._squelch is not None:
                quieting_db = self._squelch.stats.last_quieting_db
                if quieting_db is not None:
                    self._last_reading = BranchReading(quieting_db=quieting_db, at=block.midpoint)
        if self._log is not None and self._squelch is not None:
            quieting_db = self._squelch.stats.last_quieting_db
            if quieting_db is not None:
                self._log.record(
                    block.midpoint,
                    self.label,
                    quieting_db,
                    gate_open=self._squelch.is_open,
                )
        return audio


class ReceiveSession:
    """Runs the tracking side and the receive chains together.

    Usage::

        branch = Branch(
            label="A - Arrow V",
            stream=IqStream(sdr),
            nbfm=nbfm_config,
            doppler=DopplerTracker(downlink_hz, applied.center_hz),
        )
        session = ReceiveSession(
            branches=[branch],
            audio=AudioOutput(nbfm_config.audio_rate_hz),
            range_rate=TargetRangeRate(satellite, observer),
        )
        with session:
            time.sleep(300)
        print(session.stats.describe())

    Args:
        branches: The receive chains to run, in declaration order. One
            for a single-device station, two for dual-SDR. Must not be
            empty. Each gets its own demodulating thread, because the
            demodulating path is the closest thing here to real time and
            two branches sharing a thread would make each one's latency
            depend on the other's.
        audio: Where the recovered audio goes. **One per session**, not
            one per branch, because there is one speaker — which is the
            whole reason a branch has to be selected.
        range_rate: Where range-rate samples come from. See
            :class:`RangeRateSource`. One per session: a range rate
            comes from the TLE and the observer's location, so it is a
            property of the pass and not of any radio.
        listening: Index of the branch whose audio reaches the speaker.
            Defaults to the first. See :meth:`listen_to`. With a
            ``selector`` this is only the starting choice.
        selector: Optional
            :class:`~qsorbit.core.combiner.BranchSelector`. When given,
            the branch holding the speaker is chosen after every
            demodulated block instead of being fixed. ``None`` — the
            default — leaves ``listening`` in force for the whole run,
            which is what makes a single-branch control run possible.
        stale_after_s: How long a branch may go without producing a
            block before the selector stops trusting its last reading.
            Ignored without a ``selector``.
        tracking_interval_s: Seconds between range-rate samples.
        join_timeout_s: How long :meth:`stop` waits per thread.

    Raises:
        ValueError: If ``branches`` is empty, if ``listening`` is not a
            valid index into it, or if ``tracking_interval_s`` is not
            positive.
    """

    def __init__(
        self,
        *,
        branches: Sequence[Branch],
        audio: AudioOutput,
        range_rate: RangeRateSource,
        listening: int = 0,
        selector: BranchSelector | None = None,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        tracking_interval_s: float = DEFAULT_TRACKING_INTERVAL_S,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not branches:
            raise ValueError(
                "A receive session needs at least one branch. A single-device "
                "station is one branch, not a branchless special case."
            )
        if not 0 <= listening < len(branches):
            raise ValueError(
                f"listening must index one of the {len(branches)} branch(es), got {listening!r}."
            )
        if tracking_interval_s <= 0.0:
            raise ValueError(f"tracking_interval_s must be positive, got {tracking_interval_s!r}.")

        self._branches = tuple(branches)
        self._audio = audio
        self._range_rate = range_rate
        self._tracking_interval_s = tracking_interval_s
        self._join_timeout_s = join_timeout_s
        self._sleep = sleep

        self._listening = self._branches[listening]
        for branch in self._branches:
            branch.listened = branch is self._listening
        self._selector = selector
        self._stale_after_s = stale_after_s
        self._by_label = {branch.label: branch for branch in self._branches}
        # Guards the read-decide-switch sequence, which every
        # demodulating thread runs. Without it two threads can read the
        # same readings and both act on them, producing two switches
        # where the decision was one.
        self._select_lock = threading.Lock()

        self._stop = threading.Event()
        self._demod_threads: list[threading.Thread] = []
        self._tracking_thread: threading.Thread | None = None
        # One writer thread per branch that is recording. Empty on an
        # ordinary run, so the lifecycle below is a no-op unless --record-iq
        # gave a branch a recorder.
        self._recorder_threads: list[threading.Thread] = []
        self._started = False
        self._error: BaseException | None = None
        # Kept apart from _error deliberately: see tracking_error().
        self._tracking_error: BaseException | None = None
        # And apart again: a recorder that failed (a full disk, most
        # likely) is worth raising, but not ahead of a radio that died --
        # the demod error is the cause, a stalled recorder often the
        # symptom.
        self._recorder_error: BaseException | None = None
        self._stopped_cleanly = True

        self._lock = threading.Lock()
        self._range_rate_updates = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def branches(self) -> tuple[Branch, ...]:
        """The branches this session runs, in declaration order."""
        return self._branches

    @property
    def listening(self) -> Branch:
        """The branch whose audio is currently reaching the speaker."""
        return self._listening

    def listen_to(self, branch: Branch) -> None:
        """Send ``branch``'s audio to the speaker, and no other's.

        Args:
            branch: One of this session's own branches.

        Raises:
            ValueError: If ``branch`` does not belong to this session.

        Exactly one branch holds the ear at a time, enforced here rather
        than by whoever sets the flag, because two branches writing to
        one :class:`~qsorbit.core.dsp.audio.AudioOutput` would interleave
        two half-rate audio streams into something that sounds like a
        fault in the radio.

        Safe to call while running: the demodulating threads read the
        flag once per block, so a switch takes effect within one block
        without anything being torn down. Nothing calls it live yet —
        that is the combiner's job — but the flag is the mechanism it
        will use.
        """
        if branch not in self._branches:
            raise ValueError(f"{branch.label!r} is not a branch of this session.")
        for candidate in self._branches:
            candidate.listened = candidate is branch
        self._listening = branch

    @property
    def selector(self) -> BranchSelector | None:
        """The combiner, or ``None`` if this run has a fixed branch."""
        return self._selector

    def _reselect(self, tolerance_s: float) -> None:
        """Let the combiner move the speaker, if there is one.

        Called from every demodulating thread after that thread's branch
        has demodulated, so decisions happen at about twice the block
        rate on a two-branch station.

        **Simultaneity is decided here, not in the selector.** Each
        branch reports its most recent reading together with the block
        midpoint it describes; :func:`~qsorbit.core.combiner.pair_simultaneous`
        then withholds any challenger whose block is not within
        ``tolerance_s`` of the branch currently holding the ear, so the
        selector only ever compares readings that describe the same
        moment. Comparing a branch's reading against another branch's
        *neighbouring* block is what let a one-block skew cross the
        margin on 2026-09-06 and manufacture the run's only switch --
        the same class of error, one level deeper, as reading a level
        off the wrong branch.

        ``tolerance_s`` is passed in from the triggering block's own
        duration rather than stored, because it is a fact about the data
        rate and the caller is holding the block.
        """
        if self._selector is None:
            return
        # Startup: until every branch has produced a block, the ear stays
        # on the branch it started on (the declared first, or --listen's
        # branch). A branch that has not started yet is not one that has
        # died, and handing the ear to whichever demod thread wins the
        # race is the defect -- observed landing on the worse branch and
        # staying there. Once all have produced, has_produced stays true,
        # so a branch that later dies is a stale branch and choose()'s
        # rule 2 hands the ear off exactly as before. The cost is that a
        # branch which never produces a single block keeps the ear it
        # started with rather than yielding it; the declared first branch
        # is a rank choice, and a branch that never streams is a broken
        # run, not a branch to switch away from mid-startup.
        if not all(branch.has_produced for branch in self._branches):
            return
        with self._select_lock:
            now = time.monotonic()
            current_label = self._listening.label
            latest = {
                branch.label: branch.latest_reading(now=now, stale_after_s=self._stale_after_s)
                for branch in self._branches
            }
            readings = pair_simultaneous(current_label, latest, tolerance_s=tolerance_s)
            chosen = self._selector.choose(current_label, readings)
            if chosen != current_label:
                self.listen_to(self._by_label[chosen])

    @property
    def is_running(self) -> bool:
        """``True`` while any demodulating thread is alive."""
        return any(thread.is_alive() for thread in self._demod_threads)

    @property
    def spectrum(self) -> SpectrumStream | None:
        """The spectrum stream, for a widget that needs to be handed one.

        **Whichever branch holds one, not the branch being heard.** Those
        were the same thing until a combiner could move the ear, and
        reading it off the listening branch is a defect this found the
        hard way: with `--combine` the ear left the branch that had the
        spectrum, so this returned ``None`` and the end-of-run report
        said "no waterfall was attached this run" while a waterfall was
        visibly running -- contradicted, in the same report, by that
        branch's own ``consumer waterfall`` line.

        A subscription belongs to one branch's stream and cannot be
        moved, so **the waterfall is fixed at construction and does not
        follow the ear.** That is a real limitation rather than a
        choice: with the combiner running you may be watching one
        antenna's spectrum while hearing the other's audio. Saying so
        here is the cheapest honest option; making the display follow
        the ear would mean a second spectrum worker on every branch,
        computing frames nobody sees for whichever branch is currently
        silent.
        """
        for branch in self._branches:
            if branch.spectrum is not None:
                return branch.spectrum
        return None

    def tracking_error(self) -> BaseException | None:
        """What killed the range-rate thread, if anything killed it.

        Kept apart from the fault :meth:`stop` re-raises, because a
        tracking side that has stopped feeding is a *degradation* — the
        Doppler tracker extrapolates, then holds, and says how long it
        did so — while a demodulating fault means the radio job itself
        has stopped. A readout that conflated them would report a rotor
        problem as a receiver problem.
        """
        return self._tracking_error

    def start(self) -> None:
        """Prime the trackers, then start everything. Starting twice is an error.

        Order matters and is the opposite of the obvious one. Every
        branch's tracker is primed **before** any thread starts, so that
        by the time the first block can possibly arrive there is already
        a range rate to correct it with. Starting the readers first and
        priming after would reintroduce exactly the race the priming
        exists to remove, just with a smaller window.

        One sample primes every branch. They share a predicted curve —
        the same TLE, the same observer — and differ only in the centre
        frequency each tuner reached, which is baked into the tracker
        rather than into the sample.
        """
        if self._started:
            raise RuntimeError("This session has already been started; build a new one.")
        self._started = True

        when, range_rate_km_s = self._range_rate.prime()
        for branch in self._branches:
            branch.doppler.update(when, range_rate_km_s)
        self._range_rate_updates = 1

        self._audio.start()
        for branch in self._branches:
            branch.start()

        for branch in self._branches:
            thread = threading.Thread(
                target=self._demod_loop,
                args=(branch,),
                name=f"qsorbit-receive-demod-{branch.label}",
                daemon=True,
            )
            self._demod_threads.append(thread)
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, name="qsorbit-receive-tracking", daemon=True
        )
        # One recorder thread per branch that is recording. Built here,
        # after the readers exist, because the recorder drains a
        # subscription and a subscription without a reader behind it never
        # yields. Ordinary runs record nothing and add no threads.
        for branch in self._branches:
            if branch.recorder is not None:
                self._recorder_threads.append(
                    threading.Thread(
                        target=self._record_loop,
                        args=(branch,),
                        name=f"qsorbit-receive-record-{branch.label}",
                        daemon=True,
                    )
                )
        for thread in self._demod_threads:
            thread.start()
        self._tracking_thread.start()
        for thread in self._recorder_threads:
            thread.start()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until the demodulating thread ends, or until ``timeout_s``.

        A demodulating thread ends when its branch's blocks stop — the
        device was unplugged, the fake source ran out, or :meth:`stop`
        was called. So a caller that would otherwise sleep out a fixed
        duration can wait on this instead and **find out promptly that
        a radio died**, rather than sitting through the rest of a pass
        with nothing arriving. Whatever it died of is then raised by
        :meth:`stop`.

        Args:
            timeout_s: Seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            ``True`` if any branch's thread has ended, ``False`` if the
            timeout expired with all of them still running — which is
            the normal outcome of a run that lasted its full duration.

        Raises:
            RuntimeError: If the session was never started.
        """
        if not self._demod_threads:
            raise RuntimeError("This session has not been started, so there is nothing to wait on.")
        # Waits on the FIRST branch to finish, not the last. A branch
        # whose blocks stop has lost its radio, and with two of them a
        # caller wants to hear about that while the other is still
        # running -- waiting for both would turn "one dongle fell off
        # the bus" into a wait that only ends when the pass does.
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for thread in self._demod_threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
            if not thread.is_alive():
                return True
        return False

    def stop(self) -> ReceiveStats:
        """Stop everything this session started, and return the statistics.

        Stops in the order a consumer would want: the reader first, so
        no more blocks arrive, then this module's own threads, then the
        audio device. Whatever the demodulating thread died of, if
        anything, is re-raised here — a receive session that stops
        silently is the failure mode this whole project keeps meeting.

        The rotor is deliberately **not** stopped, matching
        :class:`~qsorbit.core.pointing.TrackingLoop` and
        :meth:`~qsorbit.core.rotor.Rotor.__exit__`: a move already in
        progress does not need us, and abandoning the antenna mid-slew
        is no improvement on letting it arrive.

        Raises:
            Whatever killed the demodulating thread.
        """
        self._stop.set()
        # Every reader stops before any thread is joined. Stopping and
        # joining one branch at a time would leave the other still
        # filling its buffers while nothing drained them, and the drops
        # that produced would be an artifact of the shutdown order
        # rather than of anything that happened during the run.
        for branch in self._branches:
            branch.stop_reading()

        clean = True
        # Recorder threads join with the rest: each ends when its
        # subscription drains, then writes its sidecar, so the files are
        # complete by the time stats are built.
        for thread in (*self._demod_threads, self._tracking_thread, *self._recorder_threads):
            if thread is not None:
                thread.join(self._join_timeout_s)
                clean = clean and not thread.is_alive()
        self._stopped_cleanly = clean

        audio_stats = self._audio.stop()
        stats = self._build_stats(audio_stats)

        # The demod error is the cause when both fired; a recorder error
        # is raised only when nothing more fundamental did.
        error = self._error or self._recorder_error
        if error is not None:
            raise error
        return stats

    @property
    def stats(self) -> ReceiveStats:
        """The run's statistics. Stable once :meth:`stop` has returned."""
        return self._build_stats(self._audio.stats)

    @property
    def live_quieting_db(self) -> float | None:
        """The listening branch's most recent quieting measurement.

        ``None`` when that branch has no squelch at all - there is
        nothing to show. Otherwise this is the number a live "quieting"
        readout polls, updating every block regardless of
        ``mute_squelch``: see :class:`Branch` for why a run with muting
        off still has a real, moving number here.

        **The listening branch's, not every branch's.** With two
        branches running, one number cannot describe both, and the one
        worth putting next to the audio is the one that made it. Each
        branch carries its own — see :attr:`Branch.live_quieting_db` —
        and the per-branch meters read those directly.

        Unlike :attr:`stats` (**not** safe to call this "live" - its own
        docstring says it is stable only once :meth:`stop` has
        returned), this property is meant to be polled *while the
        session is running*, from a different thread than the one
        updating the squelch. It is deliberately not guarded by a lock:
        the value it reads is a single float, reassigned as a whole on
        every block by
        :meth:`~qsorbit.core.dsp.squelch.NoiseSquelch.update`, so a
        concurrent read can only ever see the value from just before or
        just after an update, never a torn one - the CPython GIL makes a
        single attribute assignment atomic. A live gauge redrawn several
        times a second tolerates being one block stale; the session's
        lock exists for the counters that end up in a report and have to
        add up exactly, which this number does not.
        """
        return self._listening.live_quieting_db

    @property
    def live_squelch_open(self) -> bool | None:
        """Whether the listening branch's gate is open right now.

        ``None`` if that branch has no squelch. The gate's *decision*,
        exactly as :attr:`live_quieting_db` is its *measurement* - both
        real even when ``mute_squelch=False`` never lets that decision
        reach the speaker. Same polling contract as
        :attr:`live_quieting_db`: a single attribute read, safe enough
        for a live display, not for a report that has to add up.
        """
        return self._listening.live_squelch_open

    @property
    def live_tracked_frequency_hz(self) -> float | None:
        """The listening branch's downlink frequency in RF right now, or ``None``.

        ``None`` until the tracking loop has supplied that branch's
        Doppler tracker its first sample -
        :attr:`~qsorbit.core.dsp.tuning.DopplerTracker.stats` reports
        that as ``last_offset_hz is None``, and there is no honest
        frequency to report before then. Once a sample has landed, this
        is that tuner's own centre
        (:attr:`~qsorbit.core.dsp.tuning.DopplerTracker.center_hz`, fixed
        for the run) plus the most recent Doppler offset - the same two
        numbers :meth:`Branch.demodulate` combines every block to pick
        the demod's own ``channel_offset_hz``, so this property always
        matches where the audio the user is hearing actually sits, not a
        separately recomputed estimate.

        **The listening branch's**, though with two branches every branch
        answers the same here and that is correct: this is the
        downlink's RF frequency, which is a property of the pass and not
        of any receiver. What differs per branch is the *baseband*
        offset each tuner needs to put that frequency at zero, which is
        exactly why the tracker is per-branch even though this number is
        not.

        Same live-polling contract as :attr:`live_quieting_db`: meant to
        be read from a different thread than the one updating it, while
        the session runs. :attr:`~qsorbit.core.dsp.tuning.DopplerTracker.stats`
        already takes its own lock to hand back a consistent snapshot,
        so no additional locking is needed here.
        """
        return self._listening.live_tracked_frequency_hz

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ReceiveSession:
        """Start on entering a ``with`` block."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop on leaving, whether or not the body raised.

        A failure inside the ``with`` body wins: :meth:`stop` re-raises
        whatever killed the demodulating thread, and letting that replace
        the caller's own exception would hide the first fault behind a
        consequence of it.
        """
        try:
            self.stop()
        except BaseException:
            if exc_type is None:
                raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _demod_loop(self, branch: Branch) -> None:
        """Demodulate one branch's blocks, and play them if it has the ear.

        Every branch does the full chain whether or not anyone is
        listening to it. That is not waste: the squelch metrics are the
        measurement a combiner will eventually select on, and a branch
        that skipped demodulation to save CPU would have nothing to
        report and would also make the CPU cost of running two branches
        unmeasurable, which is the one thing a first dual-device run
        exists to find out.
        """
        try:
            for block in branch.timed_blocks():
                if self._stop.is_set():
                    break
                audio = branch.demodulate(block)
                # Decided BEFORE the listened check, so a switch takes
                # effect on the very block that justified it rather than
                # the next one. The block's own duration sets how close
                # two branches' readings must be to count as simultaneous.
                self._reselect(block.duration_s * PAIRING_WINDOW_FRACTION)
                # Read per block rather than captured once, so a live
                # switch takes effect within one block.
                if branch.listened:
                    self._audio.write(audio)
        except BaseException as exc:  # noqa: BLE001 - re-raised from stop()
            # First fault wins. With two branches a failing radio can
            # take its neighbour down a moment later, and the second
            # exception would be a consequence of the first.
            if self._error is None:
                self._error = exc

    def _record_loop(self, branch: Branch) -> None:
        """Run one branch's recorder until its stream finishes.

        The recorder returns of its own accord when the reader stops and
        its queue drains, so this loop has no stop check -- the same shape
        as a consumer that simply reads to the end. A failure (a full disk)
        is stored, not swallowed: a recorder that stopped writing silently
        would leave a short capture that looks whole, the exact defect
        class this project keeps naming.
        """
        try:
            recorder = branch.recorder
            if recorder is not None:
                recorder.run()
        except BaseException as exc:  # noqa: BLE001 - re-raised from stop()
            if self._recorder_error is None:
                self._recorder_error = exc

    def _tracking_loop(self) -> None:
        """Feed the Doppler tracker on a cadence until asked to stop.

        Failures here are recorded the same way the demodulating thread's
        are, rather than being allowed to kill the thread quietly. A
        tracking side that stops feeding does not stop the audio — the
        tracker extrapolates, then holds, and says how long it did so in
        :class:`~qsorbit.core.dsp.tuning.DopplerStats` — so this is a
        degradation to report rather than a reason to tear the session
        down mid-pass.
        """
        try:
            while not self._stop.wait(self._tracking_interval_s):
                pending = self._range_rate.sample()
                if pending is None:
                    continue
                when, range_rate_km_s = pending
                for branch in self._branches:
                    branch.doppler.update(when, range_rate_km_s)
                with self._lock:
                    self._range_rate_updates += 1
        except BaseException as exc:  # noqa: BLE001 - re-raised from stop()
            # Recorded twice, on purpose. _error is what stop() re-raises
            # so the run cannot end silently; _tracking_error is what a
            # following readout reads, and it must not be confused with a
            # demodulation fault, which says nothing about the rotor.
            self._tracking_error = exc
            if self._error is None:
                self._error = exc

    def _build_stats(self, audio_stats: AudioStats) -> ReceiveStats:
        """Assemble a snapshot from each owner's own accounting."""
        with self._lock:
            updates = self._range_rate_updates
        spectrum = self.spectrum
        return ReceiveStats(
            branches=tuple(branch.stats for branch in self._branches),
            range_rate_updates=updates,
            audio=audio_stats,
            spectrum=spectrum.stats if spectrum is not None else None,
            combiner=self._selector.stats if self._selector is not None else None,
            stopped_cleanly=self._stopped_cleanly,
        )
