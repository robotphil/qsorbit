"""Tests for ReplaySdr -- the file-backed device at the SdrFactory seam.

Driven with tiny captures written to tmp_path and an injected sleep/clock,
so the real-time pacing and the capture clock can be asserted exactly
without a run taking a pass-length. The sidecar is a v2 (or v1) file on
disk, because the reader's whole job is to speak that format.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from qsorbit.core.sdr import DeviceError, SdrConfig
from qsorbit.core.sdr.librtlsdr import TunerType
from qsorbit.core.sdr.replay import ReplaySdr

RATE_HZ = 1_024_000.0
BYTE_RATE = 2_048_000.0  # uint8 I + Q
CENTER_HZ = 435_605_000.0


def a_capture(tmp_path, *, iq: bytes = b"\x00" * 4096, name: str = "cap.iq", **meta_overrides):
    """Write an .iq and its v2 sidecar to tmp_path; return the .iq path."""
    iq_path = tmp_path / name
    iq_path.write_bytes(iq)
    meta = {
        "sidecar_version": 2,
        "format": "raw uint8 interleaved I/Q",
        "started_utc": "2026-09-06T08:44:00.000Z",
        "captured_utc": "2026-09-06T08:44:02Z",
        "first_block_monotonic_s": 123.0,
        "device": "[1] RTL-SDR Blog V4 (R828D tuner, serial RIGHT)",
        "requested_center_hz": CENTER_HZ,
        "actual_center_hz": CENTER_HZ,
        "requested_sample_rate_hz": RATE_HZ,
        "actual_sample_rate_hz": RATE_HZ,
        "gain_db": 30.0,
        "gain_mode": "manual",
        "ppm": 0,
        "agc_enabled": False,
        "seconds": len(iq) / BYTE_RATE,
        "bytes": len(iq),
        "contiguous": True,
        "blocks_dropped": 0,
        "estimated_lost_bytes": 0,
        "loss_fraction": 0.0,
    }
    meta.update(meta_overrides)
    iq_path.with_suffix(".json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return iq_path


def a_config(center_hz=440_000_000.0, sample_rate_hz=2_048_000.0, gain_db=49.6) -> SdrConfig:
    return SdrConfig(center_hz=center_hz, sample_rate_hz=sample_rate_hz, gain_db=gain_db)


class ListClock:
    """A monotonic clock that returns a scripted sequence of readings."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = list(readings)
        self._last = self._readings[0] if self._readings else 0.0

    def __call__(self) -> float:
        if self._readings:
            self._last = self._readings.pop(0)
        return self._last


class TestReportsTheFile:
    def test_configure_reports_the_file_not_the_request(self, tmp_path):
        replay = ReplaySdr(a_capture(tmp_path))
        replay.open()

        applied = replay.configure(a_config(center_hz=440e6, sample_rate_hz=2.048e6))

        # The station asked for 2.048 Msps at 440 MHz; the file wins.
        assert applied.sample_rate_hz == RATE_HZ
        assert applied.center_hz == CENTER_HZ
        assert applied.gain_db == 30.0
        # The request is kept, so the gap between asked and delivered is visible.
        assert applied.requested.sample_rate_hz == 2.048e6

    def test_open_identifies_as_a_replay_not_a_radio(self, tmp_path):
        replay = ReplaySdr(a_capture(tmp_path))

        info = replay.open()

        assert "replay" in info.name.lower()
        assert info.tuner is TunerType.UNKNOWN
        assert replay.is_open

    def test_a_missing_sidecar_is_a_device_error(self, tmp_path):
        # An .iq with no sidecar cannot be interpreted.
        (tmp_path / "orphan.iq").write_bytes(b"\x00" * 16)
        with pytest.raises(DeviceError, match="sidecar"):
            ReplaySdr(tmp_path / "orphan.iq").open()

    def test_a_missing_capture_is_a_device_error(self, tmp_path):
        iq_path = a_capture(tmp_path)
        iq_path.unlink()  # leave the sidecar, remove the .iq
        with pytest.raises(DeviceError, match="capture not found"):
            ReplaySdr(iq_path).open()


class TestReading:
    def test_read_raw_returns_the_file_bytes(self, tmp_path):
        replay = ReplaySdr(a_capture(tmp_path, iq=bytes(range(256)) * 4), sleep=lambda s: None)
        replay.open()
        replay.configure(a_config())

        first = replay.read_raw(512)

        assert first == (bytes(range(256)) * 4)[:512]

    def test_read_raw_paces_by_sample_count(self, tmp_path):
        # Monotonic pinned at 0, so after handing out `length` bytes the
        # device must sleep length / byte_rate to let the wall catch up.
        slept: list[float] = []
        replay = ReplaySdr(
            a_capture(tmp_path, iq=b"\x00" * 8192),
            sleep=slept.append,
            monotonic=lambda: 0.0,
        )
        replay.open()
        replay.configure(a_config())

        replay.read_raw(4096)

        assert slept == [pytest.approx(4096 / BYTE_RATE)]

    def test_read_raw_raises_at_end_of_file(self, tmp_path):
        replay = ReplaySdr(a_capture(tmp_path, iq=b"\x00" * 512), sleep=lambda s: None)
        replay.open()
        replay.configure(a_config())

        replay.read_raw(512)  # consumes the whole file
        with pytest.raises(DeviceError, match="exhausted"):
            replay.read_raw(512)

    def test_a_short_final_block_ends_the_run(self, tmp_path):
        # 600 bytes, asked for 512: the first read is full, the second is a
        # 88-byte tail -> shorter than asked -> the run ends, no partial.
        replay = ReplaySdr(a_capture(tmp_path, iq=b"\x00" * 600), sleep=lambda s: None)
        replay.open()
        replay.configure(a_config())

        replay.read_raw(512)
        with pytest.raises(DeviceError, match="exhausted"):
            replay.read_raw(512)


class TestCaptureClock:
    def test_started_utc_from_a_v2_sidecar(self, tmp_path):
        replay = ReplaySdr(a_capture(tmp_path))
        replay.open()

        assert replay.started_utc == datetime(2026, 9, 6, 8, 44, 0, tzinfo=UTC)

    def test_started_utc_falls_back_for_a_v1_sidecar(self, tmp_path):
        # No started_utc; derive it as captured_utc - seconds. captured is
        # 08:44:02 and seconds is 2.0, so the start is 08:44:00.
        iq = b"\x00" * int(2.0 * BYTE_RATE)
        replay = ReplaySdr(
            a_capture(tmp_path, iq=iq, sidecar_version=1, started_utc=None, seconds=2.0)
        )
        # a_capture wrote started_utc=None into the JSON; open() must ignore it.
        replay.open()

        assert replay.started_utc == datetime(2026, 9, 6, 8, 44, 0, tzinfo=UTC)

    def test_capture_clock_reads_the_capture_date_and_advances(self, tmp_path):
        # Origin fixed on first call; a later reading advances the clock by
        # the real elapsed time, from the capture's date.
        replay = ReplaySdr(a_capture(tmp_path), monotonic=ListClock([100.0, 105.0]))
        replay.open()

        clock = replay.capture_clock()

        assert clock() == datetime(2026, 9, 6, 8, 44, 0, tzinfo=UTC)
        assert clock() == datetime(2026, 9, 6, 8, 44, 5, tzinfo=UTC)
