"""Capture IQ to a file, in the format the rest of the project reads.

One function does the work. It exists for three jobs that turn out to be
the same job: generating the test fixtures the DSP chunks need, giving
the hardware integration suite something real to exercise the streaming
layer with, and being the bring-up tool a user reaches for when they
want to know whether their dongle is receiving anything at all.

**The file format is not ours and that is the point.** Raw uint8
interleaved I/Q is exactly what the device delivers and exactly what
``rtl_sdr.exe`` writes, so a capture opens in GNU Radio or inspectrum
without conversion, and the uint8-to-complex step stays inside the code
under test rather than being baked into the fixture. See
``tests/fixtures/iq/README.md``.

**The sidecar records what the device did, not what it was asked to
do.** The tuner PLL and the sample clock both quantise. Bring-up found
them landing exactly on request on this particular V4, which is luck
and must not be relied on: every offset computed downstream uses the
actual values or it is quietly wrong.

**But know what "actual" is worth here, because it is not the truth.**
``actual_sample_rate_hz`` comes from ``rtlsdr_get_sample_rate()``, and
librtlsdr derives that from the *nominal* 28.8 MHz crystal rather than
from the physical one. It therefore reports the driver's quantisation
and is blind to the dongle's own crystal error. Measured on this V4 on
2026-08-22, across three streaming runs of two different lengths: the
sample clock is **about 57 ppm slow**, so a capture recorded as
2,048,000 sps was really running near 2,047,883. That is ~117 Hz, which
is nothing to an FM channel and is a second per five hours to a time
axis. Anything that integrates over a long capture, or that compares
timestamps against an external clock, wants a measured rate rather than
this field.

**It also records whether the capture is contiguous**, which the
bench-script sidecars had no way to say. A capture with a hole in it
looks entirely normal — the bytes either side are real samples — and
will produce a mysterious, wrong answer in whatever analyses it later.
So a capture that dropped anything says so in its own metadata.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qsorbit.core.sdr.config import SdrConfig
from qsorbit.core.sdr.device import DEFAULT_READ_BYTES, AppliedSettings, RtlSdr
from qsorbit.core.sdr.stream import (
    DEFAULT_QUEUE_BLOCKS,
    IqStream,
    StreamStats,
    byte_rate_for,
)

#: What the ``.iq`` files contain, recorded in every sidecar so a file
#: found on its own is still readable.
IQ_FORMAT_DESCRIPTION: str = "raw uint8 interleaved I/Q (I,Q,I,Q...), offset binary, 127.5 = zero"

#: Sidecar schema version. Bumped if a key ever changes meaning, so an
#: old file is rejected rather than silently misread.
#:
#: Version 1 differs from the bench scripts' hand-written sidecars in
#: one way that matters: those recorded ``tuning_offset_hz`` as the
#: centre relative to the station, and this records
#: ``station_offset_hz`` as the station relative to the centre — the
#: opposite sign. The key was renamed rather than reused precisely
#: because a same-named key with a flipped sign is worse than a new
#: one. The new sense matches
#: :meth:`~qsorbit.core.sdr.device.AppliedSettings.offset_from` and the
#: bench analysis code, both of which ask "where in this capture does
#: the station appear".
#:
#: Version 2 **adds** two keys and changes no existing one, so it is a
#: superset of v1 — a v1 reader ignores the additions and a v2 reader
#: fills them in from ``captured_utc`` when absent (a v1 file's start is
#: ``captured_utc − seconds``, the derivation Session 22 already relies
#: on). The additions exist for replay:
#:
#: - ``started_utc`` — the capture's **start**, to the millisecond.
#:   ``captured_utc`` is the *end* (stamped after the write loop), and a
#:   replay driving a Doppler curve needs the epoch of the first sample,
#:   not the last. Recorded rather than re-derived so the next reader is
#:   told the truth instead of computing ``end − seconds`` and hoping the
#:   duration was honest.
#: - ``first_block_monotonic_s`` — the monotonic-clock reading captured
#:   at the first block. Two dongles share no wall clock, so aligning a
#:   dual capture means pairing each branch's ``started_utc`` with a
#:   monotonic reference taken at the same instant. This is that
#:   reference; on a single capture it is informational.
SIDECAR_VERSION: int = 2


@dataclass(frozen=True)
class CaptureResult:
    """Where a capture went and how well it went.

    Args:
        iq_path: The raw IQ file written.
        sidecar_path: Its JSON sidecar.
        applied: What the device actually did when configured.
        stats: The streaming run's statistics, with device loss and
            buffer drops kept separate.
        metadata: Exactly what was written to the sidecar.
    """

    iq_path: Path
    sidecar_path: Path
    applied: AppliedSettings
    stats: StreamStats
    metadata: dict[str, object]

    @property
    def is_contiguous(self) -> bool:
        """``True`` if nothing was dropped at our buffer.

        Deliberately narrow: this speaks only to the loss we can count
        exactly. Samples that never crossed USB are in
        ``stats.loss`` and are a different question with a different
        fix — see :mod:`qsorbit.core.sdr.stream`.
        """
        return self.stats.blocks_dropped == 0

    def describe(self) -> str:
        """Return a short human-readable summary of the capture."""
        size = self.metadata["bytes"]
        hole = "" if self.is_contiguous else "  NOT CONTIGUOUS - blocks were dropped\n"
        return (
            f"{self.iq_path.name}: {size:,} bytes\n"
            f"  tuned {self.applied.center_hz:,.0f} Hz "
            f"(asked {self.applied.requested.center_hz:,.0f}, "
            f"error {self.applied.center_error_hz:+,.0f} Hz)\n"
            f"  {self.applied.sample_rate_hz:,.0f} sps, {self.applied.gain_db} dB\n"
            f"  {self.stats.describe()}" + hole
        )


def capture_to_file(
    device: RtlSdr,
    config: SdrConfig,
    path: str | os.PathLike[str],
    *,
    seconds: float,
    station_hz: float | None = None,
    block_bytes: int = DEFAULT_READ_BYTES,
    queue_blocks: int = DEFAULT_QUEUE_BLOCKS,
    captured_at: datetime | None = None,
    started_at: datetime | None = None,
) -> CaptureResult:
    """Configure ``device``, capture ``seconds`` of IQ, and write it out.

    Writes two files: ``path`` holding the raw samples, and ``path``
    with a ``.json`` suffix holding the metadata. Blocks are written as
    they arrive rather than accumulated, both because a long capture
    would otherwise sit in memory and because writing to disk is the
    realistic consumer whose ability to keep up is worth testing.

    Args:
        device: An open device. It is configured here, so the sidecar
            records the settings from this exact call.
        config: What to tune to.
        path: Where to write the ``.iq`` file. Parent directories are
            created.
        seconds: How much to capture. The file is truncated to exactly
            this much, computed from the device's **actual** sample
            rate, so a capture's duration does not depend on the clock
            having quantised the way it was asked to.
        station_hz: A signal of interest, if there is one. Recorded
            alongside its offset from the tuned centre, so an analysis
            later is told where to look rather than re-deriving it.
            Bring-up established that captures are made deliberately
            off-centre — a peak at DC cannot be told apart from the
            RTL-SDR's permanent DC spike — and this is where that
            intent gets written down.
        block_bytes: Bytes per read. See :class:`~qsorbit.core.sdr.stream.IqStream`.
        queue_blocks: Buffer depth in blocks.
        captured_at: End timestamp for the sidecar (``captured_utc``,
            stamped when the write loop finishes). Defaults to now;
            injectable so tests can assert exact metadata.
        started_at: Start timestamp for the sidecar (``started_utc``).
            Defaults to the wall clock at the first block — the epoch of
            the first sample, which is what a replay's Doppler curve
            needs. Injectable so tests can assert exact metadata.

    Returns:
        Where things went and how the run behaved. **Check
        :attr:`CaptureResult.is_contiguous`** — a capture with a hole in
        it is still written, because the samples either side are real
        and may be all a caller needs, but it is not fit to become a
        fixture.

    Raises:
        ValueError: If ``seconds`` is not positive.
        DeviceError: If the device is not open, or a read fails.
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds!r}.")

    applied = device.configure(config)
    target_bytes = int(round(seconds * byte_rate_for(applied.sample_rate_hz)))

    iq_path = Path(path)
    iq_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = iq_path.with_suffix(".json")

    written = 0
    first_block_monotonic: float | None = None
    stream = IqStream(device, block_bytes=block_bytes, queue_blocks=queue_blocks)
    with iq_path.open("wb") as handle, stream:
        for block in stream.blocks():
            if first_block_monotonic is None:
                # Stamp the start at the first real samples, so the two
                # references (wall + monotonic) mark the same instant —
                # sample zero — rather than the moment configure() returned.
                first_block_monotonic = time.monotonic()
                if started_at is None:
                    started_at = datetime.now(UTC)
            remaining = target_bytes - written
            if len(block) >= remaining:
                handle.write(block[:remaining])
                written += remaining
                break
            handle.write(block)
            written += len(block)
    stats = stream.stats

    # A capture that produced no blocks has no honest start instant; fall
    # back so the sidecar still carries a value rather than null.
    if started_at is None:
        started_at = captured_at or datetime.now(UTC)

    metadata = _build_metadata(
        applied=applied,
        stats=stats,
        written=written,
        seconds=seconds,
        station_hz=station_hz,
        device_description=device.info.describe() if device.info else "unknown device",
        captured_at=captured_at or datetime.now(UTC),
        started_at=started_at,
        first_block_monotonic_s=first_block_monotonic,
    )
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return CaptureResult(
        iq_path=iq_path,
        sidecar_path=sidecar_path,
        applied=applied,
        stats=stats,
        metadata=metadata,
    )


def _iso_ms(when: datetime) -> str:
    """Format an instant as ISO-8601 UTC to millisecond precision.

    ``captured_utc`` is second-precision; ``started_utc`` is not, because
    aligning two dongles that share no clock turns on sub-second honesty
    (the live pairing was measured at ≤1.0 ms). Assumes ``when`` is UTC,
    the same assumption ``captured_utc`` already makes.
    """
    return when.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_sidecar(
    *,
    applied: AppliedSettings,
    written: int,
    seconds: float,
    station_hz: float | None,
    device_description: str,
    captured_at: datetime,
    started_at: datetime,
    first_block_monotonic_s: float | None,
    blocks_dropped: int,
    estimated_lost_bytes: int,
    loss_fraction: float,
) -> dict[str, object]:
    """Assemble the sidecar contents from primitives.

    Shared by :func:`capture_to_file` and the in-session
    :class:`~qsorbit.core.recorder.IqRecorder`, so both write a
    byte-identical v2 sidecar and there is one format with one place to
    test it -- the ``_receive_shell_hub`` lesson, applied to a file
    format instead of a widget.

    Decoupled from :class:`~qsorbit.core.sdr.stream.StreamStats` on
    purpose: the recorder's own buffer drops come from its per-consumer
    subscription while a capture reads the whole stream, so each caller
    passes the loss numbers that are true for it rather than a stats
    object that means something different in the two cases.

    Separate from the writing, too, so its shape can be asserted in a
    unit test without touching a filesystem or a device.
    """
    requested = applied.requested
    metadata: dict[str, object] = {
        "sidecar_version": SIDECAR_VERSION,
        "format": IQ_FORMAT_DESCRIPTION,
        # started_utc (start) before captured_utc (end): a replay reads
        # the start to set its epoch, and the pair reads naturally in
        # order. first_block_monotonic_s is the alignment reference for a
        # dual capture; null only on a capture that produced no blocks.
        "started_utc": _iso_ms(started_at),
        "captured_utc": captured_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_block_monotonic_s": first_block_monotonic_s,
        "device": device_description,
        "requested_center_hz": requested.center_hz,
        "actual_center_hz": applied.center_hz,
        "requested_sample_rate_hz": requested.sample_rate_hz,
        "actual_sample_rate_hz": applied.sample_rate_hz,
        "gain_db": applied.gain_db,
        "gain_mode": "auto" if requested.uses_auto_gain else "manual",
        "ppm": applied.ppm,
        "agc_enabled": applied.agc_enabled,
        "seconds": seconds,
        "bytes": written,
        # Contiguity first among the quality keys: it is the one that
        # decides whether this file is fit to be a fixture.
        "contiguous": blocks_dropped == 0,
        "blocks_dropped": blocks_dropped,
        "estimated_lost_bytes": estimated_lost_bytes,
        "loss_fraction": loss_fraction,
    }
    if station_hz is not None:
        metadata["station_hz"] = station_hz
        # Positive means the station sits above the tuned centre. Note
        # this is the opposite sense from the bench scripts' old
        # 'tuning_offset_hz' key - see SIDECAR_VERSION.
        metadata["station_offset_hz"] = applied.offset_from(station_hz)
    return metadata


def _build_metadata(
    *,
    applied: AppliedSettings,
    stats: StreamStats,
    written: int,
    seconds: float,
    station_hz: float | None,
    device_description: str,
    captured_at: datetime,
    started_at: datetime,
    first_block_monotonic_s: float | None,
) -> dict[str, object]:
    """Adapt a capture's whole-stream :class:`StreamStats` to
    :func:`build_sidecar`.

    Kept as capture's own entry point so ``capture_to_file`` and its
    tests keep their ``StreamStats``-shaped call, while
    :func:`build_sidecar` carries the format the recorder shares.
    """
    return build_sidecar(
        applied=applied,
        written=written,
        seconds=seconds,
        station_hz=station_hz,
        device_description=device_description,
        captured_at=captured_at,
        started_at=started_at,
        first_block_monotonic_s=first_block_monotonic_s,
        blocks_dropped=stats.blocks_dropped,
        estimated_lost_bytes=round(stats.loss.lost_bytes),
        loss_fraction=stats.loss.loss_fraction,
    )
