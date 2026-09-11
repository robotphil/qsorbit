"""Replay a captured ``.iq`` back through the real stream layer.

The mirror image of :class:`~qsorbit.core.recorder.IqRecorder`: the
recorder writes a capture during a live run; this reads one back at the
:data:`~qsorbit.__main__.SdrFactory` seam, so a recorded pass runs through
:class:`~qsorbit.core.sdr.stream.IqStream`, the branch, the demod chain and
the combiner exactly as a live one would -- audio plays, the gate opens or
does not, the meters move, and a person can listen. That is what lets a
replay stand in for a pass the station could not spare a second evening to
re-fly.

Two honesty rules, both learned the hard way on this project:

* **It reports what the file says, never what config asks for.**
  ``configure`` returns :class:`~qsorbit.core.sdr.device.AppliedSettings`
  built from the sidecar, so a station configured for 2.048 Msps replaying
  a 1.024 Msps file gets a 1.024 session. A read-back that is quietly not
  compared to the request is the ``--downlink`` truncation bug (Session
  29); this refuses to repeat it.
* **It does not fill gaps.** The file holds exactly the bytes that
  arrived; a sample the device never delivered is simply absent, and the
  replay hands downstream only what is there. Nothing is zero-filled or
  repeated, so a replay cannot come out cleaner than the night was.

Real time, paced by sample count (replan §2.4): :meth:`read_raw` sleeps
until the wall clock has caught up with the samples it has handed out, so
the pass unfolds at its true rate. Faster-than-real-time is a later piece
with its own virtual clocks; this one moves a single real clock -- the
capture's date -- into the two places that already take one, through
:meth:`capture_clock`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from qsorbit.core.sdr.device import DEFAULT_READ_BYTES, AppliedSettings, DeviceInfo
from qsorbit.core.sdr.exceptions import DeviceError
from qsorbit.core.sdr.librtlsdr import TunerType
from qsorbit.core.sdr.stream import byte_rate_for

if TYPE_CHECKING:
    from qsorbit.core.sdr.config import SdrConfig

#: The zero point of offset-binary uint8 IQ: 127.5, halfway between 127
#: and 128. Attenuation scales samples toward this point.
_IQ_ZERO = 127.5


def _attenuate(block: bytes, gain: float) -> bytes:
    """Scale offset-binary uint8 IQ toward its zero point by ``gain``.

    ``gain`` is a linear amplitude factor (``10 ** (-dB / 20)``); a gain of
    1.0 (0 dB) is a no-op and returns the block untouched. A gain below 1
    pulls every sample toward 127.5 and re-quantises to uint8 -- so a
    strong attenuation crushes the signal into the quantisation floor,
    which is what makes one branch reliably the worse of an otherwise
    identical pair. A synthetic control instrument, not a physical model
    of a pad: FM demodulation is amplitude-invariant, so the degradation
    is the re-quantisation, and it bites only at large dB.
    """
    if gain >= 1.0:
        return block
    samples = np.frombuffer(block, dtype=np.uint8).astype(np.float64)
    scaled = _IQ_ZERO + (samples - _IQ_ZERO) * gain
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8).tobytes()


def _parse_iso_z(text: str) -> datetime:
    """Parse an ISO-8601 UTC instant that ends in ``Z``."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def sidecar_started_utc(meta: dict[str, object]) -> datetime:
    """The capture's start instant, from a v2 ``started_utc`` or a v1 fall-back.

    v2 sidecars record ``started_utc`` directly. v1 sidecars (and the old
    hand-written ones) have only ``captured_utc`` -- the *end* -- so the
    start is derived as ``end - seconds``, the same derivation Session 22
    settled and the one a v2 file exists to make unnecessary.
    """
    started = meta.get("started_utc")
    if isinstance(started, str) and started:
        return _parse_iso_z(started)
    captured = _parse_iso_z(str(meta["captured_utc"]))
    return captured - timedelta(seconds=float(meta.get("seconds", 0.0)))


class ReplaySdr:
    """A file-backed device satisfying the :class:`~qsorbit.core.sdr.device.RtlSdr` surface.

    Built from the ``.iq`` a capture wrote; its ``.json`` sidecar sits
    beside it (the recorder and ``sdr capture`` both write the pair). Drops
    in at the ``SdrFactory`` seam, so ``IqStream`` and everything above it
    cannot tell it from a radio -- which is the whole point.

    Args:
        iq_path: The captured ``.iq`` to replay. Its sidecar is
            ``iq_path`` with a ``.json`` suffix.
        index: The device index to report, for parity with a real device's
            :attr:`info`.
        sleep: Sleep function for real-time pacing, injectable so a test
            can no-op it and replay at disk speed rather than pass speed.
        monotonic: Monotonic clock for pacing and for
            :meth:`capture_clock`, injectable for tests.
    """

    def __init__(
        self,
        iq_path: str | Path,
        *,
        index: int = 0,
        attenuation_db: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._iq_path = Path(iq_path)
        self._sidecar_path = self._iq_path.with_suffix(".json")
        self._index = index
        # Stored as a linear amplitude factor; 0 dB is 1.0, a no-op. Used by
        # the known-answer control to make one branch of an otherwise
        # identical pair the definite loser -- see _attenuate.
        self._attenuation_db = attenuation_db
        self._gain = 10.0 ** (-attenuation_db / 20.0)
        self._sleep = sleep
        self._monotonic = monotonic
        self._meta: dict[str, object] | None = None
        self._handle: object | None = None
        self._info: DeviceInfo | None = None
        self._applied: AppliedSettings | None = None
        self._started_utc: datetime | None = None
        self._byte_rate: float = 0.0
        self._bytes_delivered = 0
        self._start_monotonic: float | None = None

    # ------------------------------------------------------------------
    # Properties (parity with RtlSdr)
    # ------------------------------------------------------------------

    @property
    def index(self) -> int:
        """The device index this instance reports."""
        return self._index

    @property
    def iq_path(self) -> Path:
        """The captured ``.iq`` this instance replays."""
        return self._iq_path

    @property
    def attenuation_db(self) -> float:
        """The attenuation applied to this branch, in dB (0.0 = none)."""
        return self._attenuation_db

    @property
    def is_open(self) -> bool:
        """``True`` while the ``.iq`` file is open for reading."""
        return self._handle is not None

    @property
    def info(self) -> DeviceInfo | None:
        """What the replay says about itself, or ``None`` if not open."""
        return self._info

    @property
    def applied(self) -> AppliedSettings | None:
        """The settings read from the sidecar, or ``None`` if not configured."""
        return self._applied

    @property
    def started_utc(self) -> datetime | None:
        """The capture's start instant, once opened -- the replay epoch."""
        return self._started_utc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> DeviceInfo:
        """Read the sidecar, open the ``.iq``, and identify as a replay.

        Raises:
            DeviceError: If the sidecar or ``.iq`` is missing or the
                sidecar cannot be read.
        """
        if self._info is not None:
            return self._info
        try:
            meta = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DeviceError(f"replay sidecar not found: {self._sidecar_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DeviceError(f"replay sidecar unreadable: {self._sidecar_path} ({exc})") from exc
        try:
            handle = self._iq_path.open("rb")
        except FileNotFoundError as exc:
            raise DeviceError(f"replay capture not found: {self._iq_path}") from exc
        self._meta = meta
        self._handle = handle
        self._started_utc = sidecar_started_utc(meta)
        self._byte_rate = byte_rate_for(float(meta["actual_sample_rate_hz"]))
        # A replay is not a Blog V4 and must not claim to be one: report an
        # UNKNOWN tuner and a name that reads plainly as a replay, so no
        # downstream check mistakes a file for the hardware.
        self._info = DeviceInfo(
            index=self._index,
            name=f"replay of {self._iq_path.name}",
            manufacturer="replay",
            product=str(meta.get("device", "")),
            serial="",
            tuner=TunerType.UNKNOWN,
        )
        return self._info

    def close(self) -> None:
        """Close the ``.iq`` file if open. Safe to call repeatedly."""
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._info = None
        self._applied = None

    # ------------------------------------------------------------------
    # Configuration -- reports the file, not the request
    # ------------------------------------------------------------------

    def configure(self, config: SdrConfig) -> AppliedSettings:
        """Report what the file was captured at, ignoring ``config``'s asks.

        ``config`` is kept as the *requested* settings so the gap between
        asked and delivered stays visible (a station at 2.048 Msps
        replaying a 1.024 file), but every actual value comes from the
        sidecar.
        """
        meta = self._require_meta()
        self._applied = AppliedSettings(
            requested=config,
            center_hz=float(meta["actual_center_hz"]),
            sample_rate_hz=float(meta["actual_sample_rate_hz"]),
            gain_db=meta["gain_db"],
            manual_gain=str(meta.get("gain_mode", "manual")) != "auto",
            ppm=int(meta.get("ppm", 0)),
            agc_enabled=bool(meta.get("agc_enabled", False)),
        )
        return self._applied

    # ------------------------------------------------------------------
    # Reading -- real time, paced by sample count
    # ------------------------------------------------------------------

    def read_raw(self, length: int = DEFAULT_READ_BYTES) -> bytes:
        """Return the next block from the file, paced to the capture's rate.

        Sleeps until the wall clock has caught up with the samples handed
        out so far, so blocks arrive at the true cadence -- the audio plays
        and the gate reacts as they did on the night.

        Raises:
            DeviceError: When the file has no more full block to give,
                which ends the run the way a real device stopping does.
        """
        handle = self._require_handle()
        block = handle.read(length)
        if len(block) < length:
            # A short tail or end of file. A real device blocks forever;
            # a file ends, and that end is the run's end -- raised, not
            # returned, so the session stops the same way an unplugged
            # dongle stops it.
            raise DeviceError(f"replay exhausted: {self._iq_path.name} has no more samples")
        if self._start_monotonic is None:
            self._start_monotonic = self._monotonic()
        self._bytes_delivered += len(block)
        target_elapsed = self._bytes_delivered / self._byte_rate if self._byte_rate else 0.0
        actual_elapsed = self._monotonic() - self._start_monotonic
        if target_elapsed > actual_elapsed:
            self._sleep(target_elapsed - actual_elapsed)
        return _attenuate(block, self._gain)

    # ------------------------------------------------------------------
    # The capture clock
    # ------------------------------------------------------------------

    def capture_clock(self) -> Callable[[], datetime]:
        """A wall clock reading the capture's date, advancing at real time.

        Injected into ``IqStream.now`` and ``TargetRangeRate.now`` so block
        timestamps and the Doppler curve run at the capture's date, not
        today's. One clock, taken from the listening branch's sidecar and
        shared across the session: two branches replaying the *same* file
        then stamp identical timestamps, which is exactly what the
        identical-input control needs (no skew to manufacture a switch).

        The origin is fixed on the clock's first call, so ``started_utc``
        lines up with the moment the run actually begins reading.
        """
        started = self._require_started_utc()
        monotonic = self._monotonic
        origin: float | None = None

        def now() -> datetime:
            nonlocal origin
            reading = monotonic()
            if origin is None:
                origin = reading
            return started + timedelta(seconds=reading - origin)

        return now

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ReplaySdr:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_meta(self) -> dict[str, object]:
        if self._meta is None:
            raise DeviceError("replay device is not open; call open() first.")
        return self._meta

    def _require_handle(self) -> object:
        if self._handle is None:
            raise DeviceError("replay device is not open; call open() first.")
        return self._handle

    def _require_started_utc(self) -> datetime:
        if self._started_utc is None:
            raise DeviceError("replay device is not open; call open() first.")
        return self._started_utc
