"""Tests for the receive path — the vertical slice, asserted offline.

The whole chain runs here with no hardware of any kind: a fake device
handing out synthetic IQ, a stand-in for the tracking loop, a recording
stand-in for the audio device, and range rates the test dictates.

**What these tests are for, stated precisely so they are not read as
claiming more.** ``test_tuning.py`` already proves the Doppler
*arithmetic* against a real orbital profile. What was untested until now
is the **wiring** — that each block is corrected at its own midpoint,
that the correction changes as the pass does, and that the sign survives
the trip from a range rate through a tracker into a demodulator config.
So the carrier here sits at a known offset and the range rates are
scripted; the assertions are about which way the correction moves and
whether it moves at all, not about recovering a modulated signal. A real
pass is Chunk H's bench day, and nothing here substitutes for it.

**Two things have to be dated deliberately or these tests assert
nothing.** Block timestamps come from an injected clock, because a real
one would date the blocks at whatever instant the suite happened to run
while the range-rate samples sit in 2026 — making the extrapolation run
forwards or backwards depending on the hour. And the audio double is a
synchronisation barrier as well as a recorder: stepping the fake device
proves a block reached the *queue*, not that another thread demodulated
it.

**No sleeps anywhere.** The fake device parks on an event the test
controls, exactly as ``tests/unit/sdr/test_stream.py`` does, so "the
session has demodulated three blocks" is a state the test arranges rather
than one it waits for.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from qsorbit.core.dsp.demod import NbfmConfig
from qsorbit.core.dsp.spectrum import SpectrumConfig
from qsorbit.core.dsp.spectrum_stream import SpectrumStream
from qsorbit.core.dsp.squelch import NoiseSquelch
from qsorbit.core.dsp.tuning import DopplerTracker
from qsorbit.core.geometry import AzEl
from qsorbit.core.pointing import TravelGuardError
from qsorbit.core.quieting_log import QuietingLog
from qsorbit.core.receive import (
    AUDIO_SUBSCRIBER,
    RECORDER_SUBSCRIBER,
    WATERFALL_SUBSCRIBER,
    Branch,
    ReceiveSession,
    TargetRangeRate,
)
from qsorbit.core.recorder import IqRecorder
from qsorbit.core.sdr import AppliedSettings, DeviceError, IqStream, SdrConfig
from qsorbit.core.tracker.state import TopocentricState

#: Small rates, so a "block" is a few thousand samples rather than a
#: hundred thousand and the tests stay quick. The ratios are what the
#: chain cares about; the absolute numbers only have to be legal.
SAMPLE_RATE_HZ = 256_000.0
IF_RATE_HZ = 32_000.0
AUDIO_RATE_HZ = 32_000.0
BLOCK_SAMPLES = 8_192
BLOCK_BYTES = BLOCK_SAMPLES * 2

#: The downlink under test, and where the tuner sits relative to it.
DOWNLINK_HZ = 145_950_000.0
TUNING_OFFSET_HZ = 50_000.0
CENTER_HZ = DOWNLINK_HZ - TUNING_OFFSET_HZ

AN_INSTANT = datetime(2026, 8, 24, 18, 30, 0, tzinfo=UTC)


class BlockClock:
    """The wall clock ``IqStream`` stamps blocks with, advanced by hand.

    **Injected rather than left real, and the tests below do not work
    without it.** A block's timestamp is what the Doppler tracker
    extrapolates to, so a real clock would put block times at whatever
    instant the suite happened to run and the extrapolation would be
    against range-rate samples dated somewhere else entirely — sometimes
    forwards, sometimes backwards, depending on the hour. That is not a
    slow test or a flaky one, it is a test whose *direction* depends on
    when it runs.
    """

    def __init__(self, start: datetime = AN_INSTANT, step_s: float = 1.0) -> None:
        self._now = start
        self._step = timedelta(seconds=step_s)

    def __call__(self) -> datetime:
        self._now += self._step
        return self._now


def an_nbfm_config(**overrides) -> NbfmConfig:
    defaults = {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "if_rate_hz": IF_RATE_HZ,
        "audio_rate_hz": AUDIO_RATE_HZ,
    }
    return NbfmConfig(**{**defaults, **overrides})


def applied_settings() -> AppliedSettings:
    config = SdrConfig(center_hz=CENTER_HZ, sample_rate_hz=SAMPLE_RATE_HZ, gain_db=32.8)
    return AppliedSettings(
        requested=config,
        center_hz=CENTER_HZ,
        sample_rate_hz=SAMPLE_RATE_HZ,
        gain_db=32.8,
        manual_gain=True,
        ppm=0,
        agc_enabled=False,
    )


def carrier_block(offset_hz: float, index: int) -> bytes:
    """One block of uint8 IQ holding a carrier at ``offset_hz`` from centre.

    Phase is advanced across blocks by ``index`` so consecutive blocks
    join up rather than each restarting at zero — a discontinuity every
    block would be a broadband click the discriminator would faithfully
    reproduce, and would make any audio assertion meaningless.
    """
    start = index * BLOCK_SAMPLES
    n = np.arange(start, start + BLOCK_SAMPLES, dtype=np.float64)
    phase = 2.0 * np.pi * offset_hz * n / SAMPLE_RATE_HZ
    # 100 rather than 127 so the tone has headroom and nothing clips at
    # the ADC's rails, which would add harmonics we did not ask for.
    i = np.round(127.5 + 100.0 * np.cos(phase)).astype(np.uint8)
    q = np.round(127.5 + 100.0 * np.sin(phase)).astype(np.uint8)
    interleaved = np.empty(BLOCK_SAMPLES * 2, dtype=np.uint8)
    interleaved[0::2] = i
    interleaved[1::2] = q
    return interleaved.tobytes()


def noise_block(index: int, *, seed: int = 7) -> bytes:
    """One block of uint8 IQ holding broadband noise, no carrier at all.

    Where :func:`carrier_block` is a clean tone -- deliberately, so it
    reliably opens a squelch on the first block -- this is its opposite:
    a channel with nothing in it, for tests that need the gate to stay
    *closed*. ``index`` seeds the generator so consecutive blocks differ
    (a repeating block would not be noise), while staying reproducible
    run to run.
    """
    rng = np.random.default_rng(seed + index)
    return rng.integers(0, 256, size=BLOCK_SAMPLES * 2, dtype=np.uint8).tobytes()


class SteppedFakeDevice:
    """A device the test advances one block at a time.

    Each block carries a carrier at whatever offset the test asks for, so
    a sequence of steps can walk a signal across the passband the way a
    real Doppler shift does. ``block_fn``, when given, overrides *what*
    each block contains (its content, not its timing) -- used by the
    squelch tests below, which need a block that stays a closed channel
    rather than a carrier that would open the gate on the first read.
    """

    def __init__(
        self, offsets_hz: list[float], *, block_fn: Callable[[int], bytes] | None = None
    ) -> None:
        self.index = 0
        self.is_open = True
        self.applied = applied_settings()
        self.reads = 0
        self.done = False
        self._offsets = offsets_hz
        self._block_fn = block_fn
        self.allow = threading.Event()
        self.ready = threading.Event()

    def read_raw(self, length: int) -> bytes:
        self.ready.set()
        self.allow.wait(5.0)
        self.allow.clear()
        if self.done or self.reads >= len(self._offsets):
            raise DeviceError("stepped fake device exhausted")
        if self._block_fn is not None:
            block = self._block_fn(self.reads)
        else:
            block = carrier_block(self._offsets[self.reads], self.reads)
        self.reads += 1
        return block

    def step(self) -> None:
        self.ready.clear()
        self.allow.set()
        assert self.ready.wait(5.0), "the reader never came back for another block"

    def finish(self) -> None:
        self.done = True
        self.allow.set()


class RecordingAudio:
    """An AudioOutput-shaped double that keeps what was written to it.

    Also the tests' **synchronisation barrier**, which is the less
    obvious half of its job. ``SteppedFakeDevice.step()`` guarantees a
    block reached the queue, not that anything demodulated it — those
    are different threads — so a test that reads the Doppler statistics
    straight after a step is reading whatever the demodulating thread
    happened to have finished. :meth:`wait_for` waits on the audio
    actually arriving, which is the end of the chain and therefore the
    only honest place to say "that block is done".
    """

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []
        self.started = False
        self.stopped = False
        self._condition = threading.Condition()

    def start(self) -> None:
        self.started = True

    def write(self, samples: np.ndarray) -> None:
        with self._condition:
            self.blocks.append(samples)
            self._condition.notify_all()

    def wait_for(self, count: int, timeout_s: float = 5.0) -> bool:
        """Block until ``count`` blocks have been demodulated and written."""
        with self._condition:
            return self._condition.wait_for(lambda: len(self.blocks) >= count, timeout_s)

    def stop(self):
        self.stopped = True
        return self.stats

    @property
    def stats(self):
        from qsorbit.core.dsp.audio import AudioStats

        return AudioStats(
            blocks_written=len(self.blocks),
            blocks_played=len(self.blocks),
            blocks_dropped=0,
            frames_played=sum(block.size for block in self.blocks),
            underruns=0,
        )


class ScriptedRangeRate:
    """A range-rate source the test drives by hand."""

    def __init__(self, samples: list[tuple[datetime, float]]) -> None:
        self._samples = list(samples)
        self.primed = False
        self.pending: list[tuple[datetime, float]] = []

    def prime(self) -> tuple[datetime, float]:
        self.primed = True
        return self._samples[0]

    def sample(self) -> tuple[datetime, float] | None:
        if not self.pending:
            return None
        return self.pending.pop(0)


class FakeTarget:
    """A target whose range rate the test dictates. Satisfies ``Target``."""

    name = "FAKE-1"

    def __init__(self, range_rate_km_s: float = -3.0) -> None:
        self.range_rate_km_s = range_rate_km_s
        self.calls = 0

    def topocentric_state(self, observer: object, time: datetime) -> TopocentricState:
        self.calls += 1
        return TopocentricState(
            sky_position=AzEl(azimuth=120.0, elevation=30.0),
            range_km=1_000.0,
            range_rate_km_s=self.range_rate_km_s,
        )


def a_branch(
    device: SteppedFakeDevice,
    *,
    label: str = "A",
    clock: BlockClock | None = None,
    center_hz: float = CENTER_HZ,
    **overrides,
) -> Branch:
    """Build one receive branch over a stepped device."""
    stream = IqStream(device, block_bytes=BLOCK_BYTES, queue_blocks=8, now=clock or BlockClock())
    settings = {
        "label": label,
        "stream": stream,
        "nbfm": an_nbfm_config(),
        "doppler": DopplerTracker(DOWNLINK_HZ, center_hz),
    }
    return Branch(**{**settings, **overrides})


def a_session(
    device: SteppedFakeDevice, source, *, clock: BlockClock | None = None, **overrides
) -> tuple[ReceiveSession, RecordingAudio]:
    """Build a one-branch session over a stepped device, everything faked.

    Branch settings and session settings are told apart by name, so a
    test can pass ``squelch=`` or ``tracking_interval_s=`` without
    knowing which object ends up holding it.
    """
    branch_keys = {
        "label",
        "center_hz",
        "nbfm",
        "doppler",
        "squelch",
        "mute_squelch",
        "spectrum_factory",
        "log",
        "recorder_factory",
    }
    branch_overrides = {k: v for k, v in overrides.items() if k in branch_keys}
    session_overrides = {k: v for k, v in overrides.items() if k not in branch_keys}
    audio = RecordingAudio()
    branch = a_branch(device, clock=clock, **branch_overrides)
    session = ReceiveSession(
        branches=[branch],
        audio=audio,
        range_rate=source,
        **session_overrides,
    )
    return session, audio


def a_dual_session(
    devices: tuple[SteppedFakeDevice, SteppedFakeDevice], source, **overrides
) -> tuple[ReceiveSession, RecordingAudio]:
    """Build a two-branch session, labelled A and B, over two fake devices.

    Each branch gets its own clock, because two real dongles stamp their
    own blocks and a shared one here would hide a branch reading another
    branch's timestamps.
    """
    audio = RecordingAudio()
    branches = [
        a_branch(device, label=label, clock=BlockClock())
        for device, label in zip(devices, ("A", "B"), strict=True)
    ]
    session = ReceiveSession(branches=branches, audio=audio, range_rate=source, **overrides)
    return session, audio


def a_spectrum_factory():
    """A real SpectrumStream factory, for tests about the wiring."""
    config = SpectrumConfig(fft_size=64, sample_rate_hz=SAMPLE_RATE_HZ, center_freq_hz=CENTER_HZ)
    return lambda blocks: SpectrumStream(blocks, config)


def quietly_stop(session: ReceiveSession) -> None:
    """Stop a session whose fake device has run out, ignoring that fact.

    A session reports why its blocks stopped, which is the whole point
    of :meth:`ReceiveSession.stop` re-raising — but for a test about
    something else entirely, the fake running dry is the teardown rather
    than the subject.
    """
    try:
        session.stop()
    except DeviceError:
        pass


class TestTargetRangeRate:
    def test_it_computes_a_sample_from_the_target_with_no_rotor_anywhere(self):
        target = FakeTarget(range_rate_km_s=-4.5)
        source = TargetRangeRate(target, observer=object(), now=lambda: AN_INSTANT)

        when, range_rate = source.sample()

        assert when == AN_INSTANT
        assert range_rate == -4.5

    def test_prime_and_sample_are_the_same_operation(self):
        target = FakeTarget()
        source = TargetRangeRate(target, observer=object(), now=lambda: AN_INSTANT)

        assert source.prime() == source.sample()


class TestTrackingError:
    def test_it_is_none_until_something_goes_wrong(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)

        assert session.tracking_error() is None

    def test_a_tracking_fault_is_recorded_where_a_readout_can_find_it(self):
        """The readout follows the tracking thread, so it must be told.

        Before Chunk A PR2 a failing tick raised on the GUI thread and
        the readout caught it directly. Now it happens on the tracking
        thread, and without this the panel would keep showing the last
        plausible-looking numbers with a dead rotor underneath -- which
        is worse than showing nothing, because it looks fine.
        """

        class ExplodingRangeRate:
            def prime(self):
                return AN_INSTANT, -3.0

            def sample(self):
                raise TravelGuardError("elevation axis reads -8.2 degrees")

        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        session, _ = a_session(
            device, ExplodingRangeRate(), tracking_interval_s=0.01, join_timeout_s=2.0
        )
        session.start()
        deadline = time.monotonic() + 2.0
        while session.tracking_error() is None and time.monotonic() < deadline:
            time.sleep(0.01)

        error = session.tracking_error()
        # stop() still re-raises it as well: the fault is recorded twice
        # on purpose, so a run cannot end silently *and* the panel can
        # say what happened while the run is still going.
        with pytest.raises(TravelGuardError):
            session.stop()

        assert isinstance(error, TravelGuardError)
        assert "elevation axis" in str(error)

    def test_a_demodulation_fault_is_not_reported_as_a_tracking_fault(self):
        """Different faults, different consequences, different readers.

        A dead demodulator stops the audio and says so through stop().
        It says nothing whatever about the rotor, and a readout that
        greyed itself out on one would be lying about the antenna.
        """
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source, join_timeout_s=2.0)
        session.start()
        # Kill the radio, leave the rotor alone.
        device.done = True
        device.allow.set()
        session.wait(2.0)

        with pytest.raises(DeviceError):
            session.stop()

        assert session.tracking_error() is None


class TestSessionWiring:
    def test_it_subscribes_twice_under_the_documented_names(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source, spectrum_factory=a_spectrum_factory())

        names = [entry.name for entry in session.stats.branches[0].stream.subscribers]

        assert names == [AUDIO_SUBSCRIBER, WATERFALL_SUBSCRIBER]

    def test_it_does_not_subscribe_a_spectrum_consumer_that_will_never_drain(self):
        """A headless run reported 453 dropped blocks with nothing wrong.

        The waterfall subscription was made unconditionally, so with no
        window every block was offered to a consumer that did not exist
        and the bounded deque evicted it. The bytes figure that produced
        -- 118,751,232 on a 60-second run -- reads as catastrophic data
        loss. "Off" and "broken" must never look the same (Session 24).
        """
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)

        names = [entry.name for entry in session.stats.branches[0].stream.subscribers]

        assert names == [AUDIO_SUBSCRIBER]
        assert session.stats.branches[0].stream.blocks_dropped == 0

    def test_the_tracker_is_primed_before_anything_streams(self):
        # offset_at() raises if it has never been given a range rate, and
        # the demodulating thread can reach its first block before the
        # tracking side produces anything. Priming removes the race
        # rather than instrumenting it.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)

        session.start()
        try:
            assert source.primed
            assert session.stats.branches[0].doppler.updates == 1
        finally:
            device.finish()
            quietly_stop(session)

    def test_starting_twice_is_refused(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source)
        session.start()
        try:
            with pytest.raises(RuntimeError, match="already been started"):
                session.start()
        finally:
            device.finish()
            quietly_stop(session)

    def test_a_non_positive_tracking_interval_is_refused(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])

        with pytest.raises(ValueError, match="tracking_interval_s"):
            a_session(device, source, tracking_interval_s=0.0)


def a_recorder_factory(tmp_path, *, label: str = "A"):
    """A recorder factory writing to ``tmp_path/<label>.iq``, loss faked.

    The loss source is a stand-in: the file-writing is the subject here,
    not the loss value (that is asserted in ``test_iq_recorder``).
    """

    def factory(subscription):
        return IqRecorder(
            subscription=subscription,
            iq_path=tmp_path / f"{label}.iq",
            applied=applied_settings(),
            device_description="fake device",
            station_hz=None,
            loss_source=lambda: SimpleNamespace(lost_bytes=0.0, loss_fraction=0.0),
            label=label,
        )

    return factory


class TestRecorder:
    """``--record-iq`` rides on the branch as one more consumer; the
    session owns the thread that drains it to disk."""

    def test_a_branch_has_no_recorder_without_a_factory(self):
        branch = a_branch(SteppedFakeDevice([TUNING_OFFSET_HZ]))

        assert branch.recorder is None

    def test_a_branch_builds_its_recorder_on_the_recorder_subscription(self):
        captured = {}
        sentinel = object()

        def factory(subscription):
            captured["name"] = subscription.name
            return sentinel

        branch = a_branch(SteppedFakeDevice([TUNING_OFFSET_HZ]), recorder_factory=factory)

        assert branch.recorder is sentinel
        assert captured["name"] == RECORDER_SUBSCRIBER

    def test_the_recorder_subscription_is_made_only_when_recording(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, -3.0)])
        session, _ = a_session(device, source, recorder_factory=lambda subscription: object())

        names = [entry.name for entry in session.stats.branches[0].stream.subscribers]

        assert RECORDER_SUBSCRIBER in names

    def test_a_run_records_iq_and_writes_a_v2_sidecar(self, tmp_path):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 3)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, recorder_factory=a_recorder_factory(tmp_path))

        session.start()
        try:
            for _ in range(3):
                device.step()
            assert audio.wait_for(3), "not every block was demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        iq_path = tmp_path / "A.iq"
        sidecar_path = tmp_path / "A.json"
        assert iq_path.is_file() and iq_path.stat().st_size > 0
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert meta["sidecar_version"] == 2
        # Every byte that reached the recorder is on disk and counted: the
        # file and its own sidecar agree on the size.
        assert meta["bytes"] == iq_path.stat().st_size


class TestDemodulation:
    def test_every_block_is_demodulated_and_written_out(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 3)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            for _ in range(3):
                device.step()
            assert audio.wait_for(3), "not every block was demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert len(audio.blocks) == 3
        assert session.stats.blocks_demodulated == 3

    def test_the_recovered_audio_is_the_right_length_and_finite(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        recovered = audio.blocks[0]
        # 8192 samples at 256 kHz filtered to 32 kHz is 1024 IF samples,
        # and the audio decimation factor is 1 with these rates -- but
        # the result is 1023, not 1024. discriminate() works on adjacent
        # PAIRS, so it drops the one straddling the block boundary. That
        # is the same behaviour Session 20 measured as making phase
        # continuity across blocks unnecessary (-89 dBFS), so it is
        # asserted here rather than rounded past.
        assert recovered.shape == (BLOCK_SAMPLES // 8 - 1,)
        assert np.all(np.isfinite(recovered))

    def test_a_device_failure_reaches_the_caller_rather_than_dying_quietly(self):
        # A receive session that stops for a reason nobody is told is
        # the failure mode this project keeps meeting.
        device = SteppedFakeDevice([])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, _ = a_session(device, source)

        session.start()
        device.finish()
        assert session.wait(5.0), "the demodulating thread never noticed the stream ending"

        with pytest.raises(DeviceError, match="exhausted"):
            session.stop()


class TestDopplerFollowsThePass:
    def run_a_turnover(self) -> tuple[float, float, ReceiveSession]:
        """Demodulate one block approaching, then one receding.

        Blocks land at AN_INSTANT +1 s and +2 s; the range-rate samples
        sit at AN_INSTANT and +1.5 s, so the second block is genuinely
        extrapolating forward from a slope that has turned over. Both
        halves have to be dated deliberately or the test asserts nothing
        about direction.
        """
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 2)
        source = ScriptedRangeRate([(AN_INSTANT, -6.0)])
        session, audio = a_session(device, source, clock=BlockClock(step_s=1.0))

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the first block was never demodulated"
            first = session.stats.branches[0].doppler.last_offset_hz

            # The pass turns over: approaching becomes receding.
            session.branches[0].doppler.update(AN_INSTANT + timedelta(seconds=1.5), +6.0)
            device.step()
            assert audio.wait_for(2), "the second block was never demodulated"
            second = session.stats.branches[0].doppler.last_offset_hz
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()
        return first, second, session

    def test_the_correction_tracks_a_range_rate_that_changes_between_blocks(self):
        # The point of the whole chunk: the correction has to FOLLOW,
        # not merely be applied once. A pass goes from approaching to
        # receding, so the offset must sweep in the same direction.
        first, second, _ = self.run_a_turnover()

        # Approaching puts the downlink HIGH, receding puts it LOW. If
        # the sign were flipped this assertion is the one that catches
        # it, and a flipped Doppler sign is invisible in the audio.
        assert first > TUNING_OFFSET_HZ
        assert second < TUNING_OFFSET_HZ

    def test_the_offset_range_spans_the_pass_rather_than_one_instant(self):
        _, _, session = self.run_a_turnover()

        stats = session.stats.branches[0].doppler
        assert stats.min_offset_hz is not None
        assert stats.max_offset_hz is not None
        assert stats.max_offset_hz - stats.min_offset_hz > 1_000.0


class TestStatsPresentation:
    def test_describe_names_every_section_including_the_absent_ones(self):
        # "The squelch was off" and "the squelch never opened" are
        # different facts, and an omitted line reads as the second.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        text = session.stats.describe()

        assert "--- A ---" in text
        assert "--- audio ---" in text
        assert "blocks read" in text
        assert "doppler:" in text
        assert "squelch: off" in text
        assert "no waterfall was attached" in text


class TestLiveQuieting:
    """Chunk I: :attr:`ReceiveSession.live_quieting_db` and
    :attr:`live_squelch_open`, and the ``mute_squelch`` wiring they sit
    beside.

    ``test_demod.py`` already proves the decoupling arithmetic --
    ``mute=False`` still measures and decides, it just does not silence.
    What is untested until now is that :class:`ReceiveSession` actually
    threads its ``mute_squelch`` constructor argument down to that call,
    and that the two live properties read the real squelch rather than a
    stale or mismatched copy of it. So these tests are about the wiring,
    exactly as the rest of this module states its own scope.
    """

    def test_no_squelch_means_no_live_reading_at_all(self):
        # "No squelch" and "a squelch that has not opened yet" have to
        # read differently, or a caller cannot tell them apart.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        assert session.live_quieting_db is None
        assert session.live_squelch_open is None

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is None
        assert session.live_squelch_open is None

    def test_a_strong_signal_opens_the_gate_and_reports_a_live_measurement(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is True
        # mute_squelch defaults to True, but the gate opened on this very
        # block -- apply() runs after update() inside the same call, so an
        # opening block is never muted against itself.
        assert np.abs(audio.blocks[0]).max() > 0.0

    def test_an_empty_channel_stays_closed_and_mutes_by_default(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ], block_fn=noise_block)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is False
        assert not audio.blocks[0].any()

    def test_mute_squelch_false_still_reports_the_closed_decision_but_lets_audio_through(self):
        # The point of the whole item: the live readout has to be honest
        # about a gate that WOULD have muted, even on a run where
        # mute_squelch=False means nothing actually gets silenced.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ], block_fn=noise_block)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        squelch = NoiseSquelch()
        session, audio = a_session(device, source, squelch=squelch, mute_squelch=False)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        assert session.live_quieting_db is not None
        assert session.live_squelch_open is False
        assert np.abs(audio.blocks[0]).max() > 0.0


class TestLiveTrackedFrequency:
    """Chunk I: :attr:`ReceiveSession.live_tracked_frequency_hz`.

    ``TestDopplerFollowsThePass`` already proves the offset arithmetic
    itself follows a changing range rate. What is untested until now is
    that this property actually combines the tuner's fixed centre with
    that live offset -- the same two numbers :meth:`_demod_loop` uses to
    build the demodulator's own ``channel_offset_hz`` -- rather than
    reporting something that has quietly drifted from what the user is
    actually hearing.
    """

    def test_no_reading_before_the_session_starts(self):
        # start() is what primes the Doppler tracker with its first
        # sample; before that there is no honest frequency to report,
        # exactly as live_quieting_db has nothing to report before a
        # squelch has run at all.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, _audio = a_session(device, source)

        assert session.live_tracked_frequency_hz is None

    def test_a_stationary_range_rate_reports_the_nominal_downlink(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        # Zero range rate means no Doppler shift at all: the tracked
        # frequency lands exactly on the nominal downlink -- the
        # tuner's own centre plus the fixed offset it was primed with,
        # nothing else in the mix.
        assert session.live_tracked_frequency_hz == pytest.approx(DOWNLINK_HZ)

    def test_it_tracks_center_plus_the_most_recent_offset_as_it_moves(self):
        # Same two-block shape as TestDopplerFollowsThePass.run_a_turnover:
        # the manual _doppler.update() only appends a sample, it does not
        # itself recompute last_offset_hz -- that only happens the next
        # time offset_at() runs, inside _demod_loop for a real block. A
        # single-block test would silently keep asserting the priming
        # sample's offset no matter what update() was called with.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ] * 2)
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, clock=BlockClock(step_s=1.0))

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the first block was never demodulated"

            # Force a specific, non-trivial offset.
            session.branches[0].doppler.update(AN_INSTANT + timedelta(seconds=1.0), -6.0)
            device.step()
            assert audio.wait_for(2), "the second block was never demodulated"
        finally:
            device.finish()
            assert session.wait(5.0), "the demodulating thread never noticed the stream ending"
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()

        offset_hz = session.stats.branches[0].doppler.last_offset_hz
        assert offset_hz is not None
        assert session.live_tracked_frequency_hz == pytest.approx(CENTER_HZ + offset_hz)
        # Approaching (negative range rate) pushes the downlink above
        # its nominal tuning offset -- the same sign
        # TestDopplerFollowsThePass pins for the raw offset.
        assert session.live_tracked_frequency_hz > DOWNLINK_HZ


# ---------------------------------------------------------------------------
# Two branches
# ---------------------------------------------------------------------------


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    """Poll ``predicate`` until it holds or the deadline passes.

    The same shape ``TestTrackingError`` already uses. Needed here
    because a *silent* branch offers no barrier: ``RecordingAudio`` can
    say when the listening branch has finished a block, and nothing can
    say when the other one has. Stepping a device only proves a block
    reached its queue.

    It matters more than it looks. :meth:`ReceiveSession.stop` sets the
    stop flag and each demodulating loop checks it *before* taking the
    next block, so a block still queued when the session stops is
    dropped on purpose -- which is right, and which means a test that
    stops without waiting reads three demodulated blocks where it meant
    four, intermittently.
    """
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def step_both(devices, session, count):
    """Advance both fake devices one block and wait for BOTH to demodulate it.

    ``count`` is the cumulative number of blocks each branch should have
    demodulated once this returns -- not the number that reached the
    speaker, which is only ever the listening branch's.
    """
    for device in devices:
        device.step()
    assert wait_until(
        lambda: all(branch.stats.blocks_demodulated >= count for branch in session.branches)
    ), (
        "branches reached "
        f"{[branch.stats.blocks_demodulated for branch in session.branches]}, wanted {count}"
    )


class TestTwoBranches:
    """Two devices, both demodulating, with one speaker between them."""

    def a_pair(self, **overrides):
        devices = (
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
        )
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_dual_session(devices, source, **overrides)
        return devices, session, audio

    def test_both_branches_demodulate_every_block(self):
        # The point of the whole PR: a branch nobody is listening to
        # still runs the full chain, because its squelch metrics are
        # what a combiner will select on and because the CPU cost of
        # running two is the thing a dual-device run measures.
        devices, session, audio = self.a_pair()

        session.start()
        try:
            # Two blocks each. The audio barrier only proves the
            # LISTENING branch has finished a block, so the counts are
            # read after stop(), where both demodulating threads have
            # been joined and the numbers are final.
            step_both(devices, session, 1)
            step_both(devices, session, 2)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        counts = [branch.blocks_demodulated for branch in session.stats.branches]
        assert counts == [2, 2]

    def test_only_the_listening_branch_reaches_the_speaker(self):
        devices, session, audio = self.a_pair()

        session.start()
        try:
            step_both(devices, session, 1)
            step_both(devices, session, 2)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        # Two blocks demodulated per branch, four in total, and exactly
        # two of them played. If both branches wrote, the speaker would
        # have four blocks of two different signals interleaved.
        assert session.stats.blocks_demodulated == 4
        assert len(audio.blocks) == 2

    def test_exactly_one_branch_is_marked_as_heard(self):
        _, session, _ = self.a_pair()

        heard = [branch.listened for branch in session.stats.branches]

        assert heard == [True, False]

    def test_the_second_branch_can_be_the_one_heard(self):
        devices, session, audio = self.a_pair(listening=1)

        assert session.listening.label == "B"
        assert [branch.listened for branch in session.stats.branches] == [False, True]

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        assert len(audio.blocks) == 1

    def test_the_stats_name_every_branch(self):
        _, session, _ = self.a_pair()

        assert [branch.label for branch in session.stats.branches] == ["A", "B"]

    def test_describe_gives_each_branch_its_own_section(self):
        # A combined number would hide exactly the per-branch difference
        # a dual-SDR run exists to measure.
        _, session, _ = self.a_pair()

        text = session.stats.describe()

        assert "--- A ---" in text
        assert "--- B ---" in text
        assert "(heard)" in text

    def test_one_range_rate_sample_feeds_every_branch(self):
        # One predicted curve, one thread, two trackers. Priming counts
        # as an update, which is why this is 1 rather than 0.
        devices, session, audio = self.a_pair()

        session.start()
        try:
            step_both(devices, session, 1)
            updates = [branch.doppler.updates for branch in session.stats.branches]
            assert updates == [1, 1]
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_each_branch_corrects_against_its_own_tuner(self):
        # Not a detail: a DopplerTracker is built against the centre
        # frequency its own PLL reached, and two PLLs quantise the same
        # request differently. One shared tracker would put one dongle's
        # baseband offset on the other dongle's samples.
        #
        # The quantity that differs is the BASEBAND offset, not the RF
        # frequency: a tuner sitting 137 Hz higher needs the downlink
        # pushed 137 Hz further down to reach zero. This test asserted
        # the RF frequency first and found it identical on both
        # branches, which is correct and was the wrong thing to check --
        # the RF frequency is a property of the pass.
        devices = (SteppedFakeDevice([TUNING_OFFSET_HZ]), SteppedFakeDevice([TUNING_OFFSET_HZ]))
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        audio = RecordingAudio()
        branches = [
            a_branch(devices[0], label="A", center_hz=CENTER_HZ),
            a_branch(devices[1], label="B", center_hz=CENTER_HZ + 137.0),
        ]
        session = ReceiveSession(branches=branches, audio=audio, range_rate=source)

        session.start()
        try:
            step_both(devices, session, 1)
            at = AN_INSTANT + timedelta(seconds=1.0)
            first, second = (branch.doppler.offset_at(at) for branch in session.branches)
            assert second - first == pytest.approx(-137.0)

            # And the RF frequency they report is the same, because it
            # describes the satellite rather than either receiver.
            heard = [branch.live_tracked_frequency_hz for branch in session.branches]
            assert heard[0] == pytest.approx(heard[1])
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_only_the_listening_branch_gets_a_waterfall(self):
        # A branch nobody is watching never computes frames nobody sees,
        # which is the same reasoning that made the subscription
        # conditional in the first place (Session 24).
        devices = (SteppedFakeDevice([TUNING_OFFSET_HZ]), SteppedFakeDevice([TUNING_OFFSET_HZ]))
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        audio = RecordingAudio()
        branches = [
            a_branch(devices[0], label="A", spectrum_factory=a_spectrum_factory()),
            a_branch(devices[1], label="B"),
        ]
        session = ReceiveSession(branches=branches, audio=audio, range_rate=source)

        names = [
            [entry.name for entry in branch.stats.stream.subscribers] for branch in session.branches
        ]

        assert names == [[AUDIO_SUBSCRIBER, WATERFALL_SUBSCRIBER], [AUDIO_SUBSCRIBER]]
        assert session.spectrum is not None

    def test_a_failure_on_one_branch_reaches_the_caller(self):
        # A dead radio must not be something the run walks past. With
        # two of them the *first* fault is the real one; a second is
        # usually a consequence.
        devices, session, audio = self.a_pair()

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            devices[1].finish()
            devices[0].finish()
            with pytest.raises(DeviceError, match="exhausted"):
                session.stop()


class TestListeningTo:
    """Which branch holds the ear, and how that changes."""

    def a_pair(self):
        devices = (
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
        )
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_dual_session(devices, source)
        return devices, session, audio

    def test_switching_moves_the_ear_and_leaves_nobody_else_holding_it(self):
        # The mechanism PR2's combiner will drive. Enforced on the
        # session rather than by whoever sets the flag: two branches
        # writing to one AudioOutput would interleave two half-rate
        # streams into something that sounds like a broken radio.
        _, session, _ = self.a_pair()

        session.listen_to(session.branches[1])

        assert session.listening is session.branches[1]
        assert [branch.listened for branch in session.branches] == [False, True]

    def test_switching_takes_effect_while_running(self):
        devices, session, audio = self.a_pair()

        session.start()
        try:
            step_both(devices, session, 1)
            session.listen_to(session.branches[1])
            step_both(devices, session, 2)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        # Still exactly one block per step reaching the speaker -- the
        # ear moved, it did not multiply. The second of those came from
        # B, which had written nothing before the switch.
        assert len(audio.blocks) == 2
        assert session.stats.blocks_demodulated == 4

    def test_a_branch_from_somewhere_else_is_refused(self):
        _, session, _ = self.a_pair()
        stranger = a_branch(SteppedFakeDevice([TUNING_OFFSET_HZ]), label="C")

        with pytest.raises(ValueError, match="not a branch of this session"):
            session.listen_to(stranger)

    def test_the_waterfall_follows_the_ear(self):
        _, session, _ = self.a_pair()

        # Neither branch has a spectrum here, which is the honest answer
        # for a headless run -- the point is that it is read off the
        # listening branch rather than cached at construction.
        session.listen_to(session.branches[1])

        assert session.spectrum is session.branches[1].spectrum


class TestBranchListValidation:
    def test_a_session_with_no_branches_is_refused(self):
        with pytest.raises(ValueError, match="at least one branch"):
            ReceiveSession(
                branches=[],
                audio=RecordingAudio(),
                range_rate=ScriptedRangeRate([(AN_INSTANT, 0.0)]),
            )

    def test_a_listening_index_past_the_end_is_refused(self):
        branch = a_branch(SteppedFakeDevice([TUNING_OFFSET_HZ]))

        with pytest.raises(ValueError, match="listening"):
            ReceiveSession(
                branches=[branch],
                audio=RecordingAudio(),
                range_rate=ScriptedRangeRate([(AN_INSTANT, 0.0)]),
                listening=1,
            )

    def test_a_negative_listening_index_is_refused(self):
        # Python would happily read -1 as "the last one", which is a
        # plausible off-by-one arriving as a working program.
        branch = a_branch(SteppedFakeDevice([TUNING_OFFSET_HZ]))

        with pytest.raises(ValueError, match="listening"):
            ReceiveSession(
                branches=[branch],
                audio=RecordingAudio(),
                range_rate=ScriptedRangeRate([(AN_INSTANT, 0.0)]),
                listening=-1,
            )


# ---------------------------------------------------------------------------
# The quieting log
# ---------------------------------------------------------------------------


def logged_rows(path):
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestQuietingLogging:
    """What a branch writes, and when."""

    def a_logged_pair(self, tmp_path, **overrides):
        log = QuietingLog(tmp_path / "quieting.csv")
        log.open()
        devices = (
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 3),
        )
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        audio = RecordingAudio()
        branches = [
            a_branch(
                device,
                label=label,
                clock=BlockClock(),
                squelch=NoiseSquelch(),
                log=log,
                **overrides,
            )
            for device, label in zip(devices, ("A", "B"), strict=True)
        ]
        session = ReceiveSession(branches=branches, audio=audio, range_rate=source)
        return devices, session, audio, log

    def test_both_branches_write_their_own_rows(self, tmp_path):
        devices, session, audio, log = self.a_logged_pair(tmp_path)

        session.start()
        try:
            step_both(devices, session, 1)
            step_both(devices, session, 2)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)
            log.close()

        assert log.rows_per_branch == {"A": 2, "B": 2}

    def test_a_row_carries_the_branch_label_and_a_real_measurement(self, tmp_path):
        devices, session, audio, log = self.a_logged_pair(tmp_path)

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)
            log.close()

        rows = logged_rows(log.path)
        assert {row["branch"] for row in rows} == {"A", "B"}
        # A quieting figure that is present and finite, not a placeholder.
        assert all(float(row["quieting_db"]) == float(row["quieting_db"]) for row in rows)
        assert all(row["gate_open"] in {"0", "1"} for row in rows)

    def test_a_branch_with_no_squelch_writes_nothing(self, tmp_path):
        # "Not measured" and "measured as zero" are different facts, and
        # a margin sized from the second would be sized from nothing.
        log = QuietingLog(tmp_path / "quieting.csv")
        log.open()
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, log=log)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            quietly_stop(session)
            log.close()

        assert log.rows == 0

    def test_no_log_means_no_file_and_no_cost(self, tmp_path):
        # A run without the flag must behave exactly as it did before
        # the flag existed.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, squelch=NoiseSquelch())

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
        finally:
            device.finish()
            quietly_stop(session)

        assert not (tmp_path / "quieting.csv").exists()

    def test_the_row_is_written_after_the_squelch_has_seen_the_block(self, tmp_path):
        # Reading the squelch before demodulating would record the
        # PREVIOUS block's measurement against this block's timestamp -
        # a whole-block skew, invisible in any single run, and fatal to
        # a difference between two series.
        log = QuietingLog(tmp_path / "quieting.csv")
        log.open()
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, squelch=NoiseSquelch(), log=log)

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
            assert wait_until(lambda: log.rows >= 1)
            branch = session.branches[0]
            # The row matches what the squelch is reporting now, which
            # it could not if the read had happened first: before the
            # first block there was no measurement at all.
            assert float(logged_rows(log.path)[0]["quieting_db"]) == pytest.approx(
                branch.live_quieting_db, abs=0.005
            )
        finally:
            device.finish()
            quietly_stop(session)
            log.close()


# ---------------------------------------------------------------------------
# The combiner, wired in
# ---------------------------------------------------------------------------


class StubSelector:
    """A selector the test drives, standing in for BranchSelector.

    The selection *rules* are covered in test_combiner.py against the
    real thing. What is untestable there and testable here is the
    wiring: what readings the session hands over, and whether the answer
    actually moves the speaker.
    """

    def __init__(self, answers=None):
        self.seen: list[tuple[str, dict]] = []
        self._answers = list(answers or [])

    def choose(self, current, readings):
        self.seen.append((current, dict(readings)))
        return self._answers.pop(0) if self._answers else current

    @property
    def stats(self):
        from qsorbit.core.combiner import SelectorStats

        return SelectorStats(
            margin_db=3.0,
            evaluations=len(self.seen),
            switches=0,
            evaluations_by_branch=(),
        )


class TestCombinerWiring:
    def a_pair(self, **overrides):
        devices = (
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 4),
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 4),
        )
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        audio = RecordingAudio()
        branches = [
            a_branch(device, label=label, clock=BlockClock(), squelch=NoiseSquelch())
            for device, label in zip(devices, ("A", "B"), strict=True)
        ]
        session = ReceiveSession(branches=branches, audio=audio, range_rate=source, **overrides)
        return devices, session, audio

    def test_no_selector_leaves_the_branch_fixed(self):
        # The control run for the acceptance comparison: same command,
        # one flag apart.
        devices, session, audio = self.a_pair()

        session.start()
        try:
            step_both(devices, session, 1)
            step_both(devices, session, 2)
            assert session.listening.label == "A"
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        assert session.stats.combiner is None

    def test_the_selector_is_asked_after_every_block(self):
        stub = StubSelector()
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            # The first block per branch is the startup gate: the
            # selector is held off until both branches have produced one,
            # so decisions accrue from the second block onward -- one per
            # demodulated block per branch.
            step_both(devices, session, 1)
            step_both(devices, session, 2)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        assert len(stub.seen) >= 2

    def test_it_is_handed_a_reading_for_every_branch(self):
        stub = StubSelector()
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        _, readings = stub.seen[-1]
        assert set(readings) == {"A", "B"}

    def test_aligned_blocks_are_both_offered(self):
        # When both branches' latest blocks share a midpoint, the
        # selector may compare them -- both reach it as real numbers.
        stub = StubSelector()
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        assert any(r.get("A") is not None and r.get("B") is not None for _, r in stub.seen), (
            "aligned branches were not both offered to the selector"
        )

    def test_a_non_simultaneous_challenger_is_withheld(self):
        # The skew defect, at the wiring level. Advancing only branch A
        # puts its latest block one period ahead of B's, so the two are
        # no longer the same moment. B is still alive -- fresh by the
        # wall clock -- yet it must reach the selector as None, because a
        # difference against a non-simultaneous block is what crossed the
        # margin and switched on 2026-09-06.
        stub = StubSelector()
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
            devices[0].step()
            assert wait_until(
                lambda: any(r.get("A") is not None and r.get("B") is None for _, r in stub.seen)
            ), "a non-simultaneous challenger was not withheld"
            # ...and it was withheld for skew, not for staleness: B is
            # still a live branch at this instant.
            assert (
                session.branches[1].latest_reading(now=time.monotonic(), stale_after_s=2.0)
                is not None
            )
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_the_ear_holds_on_the_first_branch_until_both_have_started(self):
        # The startup race. Branch A is on the ear at start (declared
        # first). Before A has produced a block, B may already have one,
        # and a rule that reads "A has no reading" as "A has died" hands
        # B the ear -- landing on whichever demod thread wins, observed to
        # be the worse branch. The selector must not even be consulted
        # until every branch has produced a block.
        stub = StubSelector(answers=["B"] * 8)  # would grab the ear if asked
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            devices[1].step()  # only B produces a block
            assert wait_until(lambda: session.branches[1].stats.blocks_demodulated >= 1)
            # A has not started. The selector has not been asked, and the
            # ear has not moved off A.
            assert stub.seen == [], "the selector was consulted before both branches started"
            assert session.listening.label == "A"

            # Now A produces too; selection resumes and the stub is asked,
            # so the gate opened rather than latching the ear forever.
            devices[0].step()
            assert wait_until(lambda: session.branches[0].stats.blocks_demodulated >= 1)
            assert wait_until(lambda: session.listening.label == "B")
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_has_produced_is_false_before_a_block_and_true_after(self):
        devices, session, audio = self.a_pair(selector=StubSelector())

        session.start()
        try:
            assert session.branches[0].has_produced is False
            step_both(devices, session, 1)
            assert session.branches[0].has_produced is True
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_its_answer_moves_the_speaker(self):
        # The claim the pure-function tests cannot make: a decision has
        # to actually reach listen_to().
        stub = StubSelector(answers=["B"])
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
            assert wait_until(lambda: session.listening.label == "B")
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_exactly_one_branch_still_holds_the_ear_after_a_switch(self):
        stub = StubSelector(answers=["B"])
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
            assert wait_until(lambda: session.listening.label == "B")
            assert [branch.listened for branch in session.branches] == [False, True]
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

    def test_the_stats_reach_the_report(self):
        stub = StubSelector()
        devices, session, audio = self.a_pair(selector=stub)

        session.start()
        try:
            step_both(devices, session, 1)
        finally:
            for device in devices:
                device.finish()
            quietly_stop(session)

        assert session.stats.combiner is not None
        assert "combiner:" in session.stats.describe()

    def test_the_report_distinguishes_off_from_never_switched(self):
        # "off" and "broken" must never look the same, and neither must
        # "off" and "on but steady".
        _, session, _ = self.a_pair()

        assert "combiner: off" in session.stats.describe()


class TestStaleBranchGuard:
    """A branch that has stopped must not keep or take the speaker."""

    def a_branch_with_squelch(self, device, label):
        return a_branch(device, label=label, clock=BlockClock(), squelch=NoiseSquelch())

    def test_a_branch_that_never_produced_reads_as_stale(self):
        branch = self.a_branch_with_squelch(SteppedFakeDevice([TUNING_OFFSET_HZ]), "A")

        assert branch.fresh_quieting_db(now=time.monotonic(), stale_after_s=2.0) is None

    def test_a_branch_that_just_produced_reads_fresh(self):
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, squelch=NoiseSquelch())

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
            branch = session.branches[0]
            assert branch.fresh_quieting_db(now=time.monotonic(), stale_after_s=2.0) is not None
        finally:
            device.finish()
            quietly_stop(session)

    def test_a_long_enough_gap_makes_it_stale(self):
        # The value is still sitting there and still looks plausible --
        # which is exactly why the guard is a time check and not a value
        # check.
        device = SteppedFakeDevice([TUNING_OFFSET_HZ])
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        session, audio = a_session(device, source, squelch=NoiseSquelch())

        session.start()
        try:
            device.step()
            assert audio.wait_for(1), "the block was never demodulated"
            branch = session.branches[0]
            assert branch.live_quieting_db is not None
            # Same instant, judged against a threshold of zero: the
            # reading has not changed, only how long ago it arrived.
            assert branch.fresh_quieting_db(now=time.monotonic() + 60.0, stale_after_s=2.0) is None
        finally:
            device.finish()
            quietly_stop(session)


class TestSpectrumSurvivesASwitch:
    """The waterfall is fixed at construction; the ear is not.

    Found at the bench: with the combiner running, the ear left the
    branch holding the spectrum and the end-of-run report announced "no
    waterfall was attached this run" while one was visibly running --
    contradicted by that branch's own ``consumer waterfall`` line in the
    same report.
    """

    def a_pair_with_a_waterfall_on_a(self):
        devices = (
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 4),
            SteppedFakeDevice([TUNING_OFFSET_HZ] * 4),
        )
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        audio = RecordingAudio()
        branches = [
            a_branch(
                devices[0],
                label="A",
                clock=BlockClock(),
                squelch=NoiseSquelch(),
                spectrum_factory=a_spectrum_factory(),
            ),
            a_branch(devices[1], label="B", clock=BlockClock(), squelch=NoiseSquelch()),
        ]
        session = ReceiveSession(branches=branches, audio=audio, range_rate=source)
        return devices, session, audio

    def test_the_spectrum_is_found_on_whichever_branch_holds_it(self):
        _, session, _ = self.a_pair_with_a_waterfall_on_a()

        assert session.spectrum is session.branches[0].spectrum
        assert session.spectrum is not None

    def test_it_survives_the_ear_moving_to_a_branch_without_one(self):
        # The defect exactly: reading the spectrum off the LISTENING
        # branch returns None the moment the combiner switches away from
        # the branch that has it.
        _, session, _ = self.a_pair_with_a_waterfall_on_a()

        session.listen_to(session.branches[1])

        assert session.listening.label == "B"
        assert session.branches[1].spectrum is None
        assert session.spectrum is not None

    def test_the_report_still_says_a_waterfall_was_attached(self):
        # The number that lied. "off" and "running" must not read the
        # same, and here "running" was reading as "off".
        _, session, _ = self.a_pair_with_a_waterfall_on_a()

        session.listen_to(session.branches[1])

        assert session.stats.spectrum is not None
        assert "no waterfall was attached" not in session.stats.describe()

    def test_no_spectrum_anywhere_still_reports_none(self):
        # The guard must not turn "genuinely headless" into a false
        # positive by finding a spectrum that does not exist.
        devices = (SteppedFakeDevice([TUNING_OFFSET_HZ]), SteppedFakeDevice([TUNING_OFFSET_HZ]))
        source = ScriptedRangeRate([(AN_INSTANT, 0.0)])
        branches = [
            a_branch(device, label=label, clock=BlockClock())
            for device, label in zip(devices, ("A", "B"), strict=True)
        ]
        session = ReceiveSession(branches=branches, audio=RecordingAudio(), range_rate=source)

        assert session.spectrum is None
        assert "no waterfall was attached" in session.stats.describe()
