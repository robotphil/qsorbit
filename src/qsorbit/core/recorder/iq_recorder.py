"""Record one branch's raw IQ to a file, with a v2 sidecar, mid-session.

The live analogue of :func:`qsorbit.core.sdr.capture.capture_to_file`.
``capture_to_file`` owns a device and captures for a fixed duration; this
owns nothing -- it is a consumer on a branch's already-running
:class:`~qsorbit.core.sdr.stream.IqStream`, draining blocks to disk in a
thread the session owns, so a capture can run *inside* a receive session
rather than beside it fighting for the same dongle. That is what the
capture night needs: ``receive --record-iq DIR`` writes every branch's IQ
while the pass is also being demodulated, logged and heard.

**Gaps are recorded, never filled.** A dropped block is not padded with
zeros or a repeat; the loss is counted in the sidecar and the file is
exactly the bytes that arrived. A replay of a file this writes therefore
cannot come out cleaner than the night was -- the hazard the replan named
-- because the samples that were lost are simply not there, and the
sidecar says how many.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qsorbit.core.sdr.capture import build_sidecar
from qsorbit.core.sdr.device import AppliedSettings
from qsorbit.core.sdr.stream import IqSubscription, LossReport, byte_rate_for


def _utc_now() -> datetime:
    """Wall clock, injectable for tests. The capture *end* timestamp."""
    return datetime.now(UTC)


def safe_filename(label: str) -> str:
    """Turn a branch label into a filesystem-safe stem.

    A label names an antenna (``"A - Arrow V"``) and is not filename-safe.
    Runs of anything but ASCII letters, digits, dot or underscore collapse
    to a single ``-``; leading and trailing ``-`` are trimmed; and an
    empty result -- a single unnamed branch, or a label that was all
    punctuation -- becomes ``"branch"``. The true label is preserved
    verbatim inside the sidecar's ``device`` line and the run's own report,
    so nothing that identifies the antenna is lost by sanitising the name.
    """
    stem = re.sub(r"[^A-Za-z0-9._]+", "-", label).strip("-")
    return stem or "branch"


class IqRecorder:
    """Drains one branch's stream subscription to a ``.iq`` file + sidecar.

    A consumer, not an owner: it holds a subscription handed out by the
    branch's :class:`~qsorbit.core.sdr.stream.IqStream` and runs in a
    thread the :class:`~qsorbit.core.receive.ReceiveSession` owns, the
    same division of labour every other per-branch consumer follows --
    the branch makes the subscription, the session runs the thread.

    Args:
        subscription: This branch's recorder subscription, from
            :meth:`IqStream.subscribe`. Its ``timed_blocks`` drives the
            write loop and its ``stats.blocks_dropped`` reports this
            consumer's own buffer drops.
        iq_path: Where the raw ``.iq`` goes. Its parent is created; the
            sidecar is written beside it with a ``.json`` suffix.
        applied: What the device actually reached, for the sidecar -- the
            actual centre and sample rate, not what config asked for.
        device_description: The device string for the sidecar.
        station_hz: The downlink, if there is one, so the sidecar records
            where in the capture the signal sits. ``None`` omits it.
        loss_source: Returns the *stream's* USB loss when the sidecar is
            written. A callable rather than a stored value because the
            number is only final once the run has stopped, and rather than
            the whole stream so a test can supply a
            :class:`~qsorbit.core.sdr.stream.LossReport` directly.
        label: The branch's true label, for the sidecar and reports.
        now: Wall clock for the *end* timestamp, injectable for tests.
    """

    def __init__(
        self,
        *,
        subscription: IqSubscription,
        iq_path: Path,
        applied: AppliedSettings,
        device_description: str,
        station_hz: float | None,
        loss_source: Callable[[], LossReport],
        label: str,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._subscription = subscription
        self._iq_path = Path(iq_path)
        self._sidecar_path = self._iq_path.with_suffix(".json")
        self._applied = applied
        self._device_description = device_description
        self._station_hz = station_hz
        self._loss_source = loss_source
        self._label = label
        self._now = now
        self._lock = threading.Lock()
        self._written = 0
        self._done = False

    @property
    def label(self) -> str:
        """The branch this recorder is writing, by its true label."""
        return self._label

    @property
    def iq_path(self) -> Path:
        """The raw ``.iq`` file being written."""
        return self._iq_path

    @property
    def sidecar_path(self) -> Path:
        """The ``.json`` sidecar, written when the stream finishes."""
        return self._sidecar_path

    @property
    def bytes_written(self) -> int:
        """Bytes written so far. Safe to poll while running."""
        with self._lock:
            return self._written

    def run(self) -> None:
        """Write blocks until the reader finishes, then the sidecar.

        Intended as a thread target. Returns when the subscription's block
        iterator is exhausted -- the reader stopped and this consumer's
        queue drained. Any exception (a full disk, most likely) propagates
        to the caller, which records it the way a demod-thread failure is
        recorded; a recorder that failed silently would be the very
        "reports state it does not own" pattern this project keeps meeting.
        """
        first_started_at: datetime | None = None
        first_monotonic: float | None = None
        self._iq_path.parent.mkdir(parents=True, exist_ok=True)
        # The sidecar is written in ``finally`` so a capture that ends by a
        # fault -- a device unplugged mid-pass, a stream that raises the
        # exhaustion its reader hit -- still gets its metadata. A partial
        # capture is usable (the replan says so); a partial capture with no
        # sidecar is not, because a bare ``.iq`` cannot be read. The fault
        # is then re-raised for the session to record, exactly as a
        # demodulating thread's is.
        try:
            with self._iq_path.open("wb") as handle:
                for timed in self._subscription.timed_blocks():
                    if first_started_at is None:
                        # The first sample's instant. ``read_at`` is when
                        # the read *returned*, so the samples began one
                        # block duration earlier. Both branches use this
                        # same convention, so the pair aligns to the
                        # millisecond the live quieting log measured
                        # (<= 1.0 ms over 937 rows).
                        first_started_at = timed.read_at - timedelta(seconds=timed.duration_s)
                        first_monotonic = time.monotonic()
                    handle.write(timed.data)
                    with self._lock:
                        self._written += len(timed.data)
        finally:
            captured_at = self._now()
            started_at = first_started_at if first_started_at is not None else captured_at
            self._write_sidecar(
                started_at=started_at,
                captured_at=captured_at,
                first_block_monotonic_s=first_monotonic,
            )
            with self._lock:
                self._done = True

    def _write_sidecar(
        self,
        *,
        started_at: datetime,
        captured_at: datetime,
        first_block_monotonic_s: float | None,
    ) -> None:
        written = self.bytes_written
        loss = self._loss_source()
        rate = self._applied.sample_rate_hz
        # The honest recorded duration: bytes actually written over the
        # device's real byte rate, not a duration anybody asked for.
        seconds = written / byte_rate_for(rate) if rate > 0 else 0.0
        metadata = build_sidecar(
            applied=self._applied,
            written=written,
            seconds=seconds,
            station_hz=self._station_hz,
            device_description=self._device_description,
            captured_at=captured_at,
            started_at=started_at,
            first_block_monotonic_s=first_block_monotonic_s,
            # This consumer's own buffer drops -- not the whole stream's,
            # which would fold in a slow waterfall the recorder never felt.
            blocks_dropped=self._subscription.stats.blocks_dropped,
            estimated_lost_bytes=round(loss.lost_bytes),
            loss_fraction=loss.loss_fraction,
        )
        self._sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
