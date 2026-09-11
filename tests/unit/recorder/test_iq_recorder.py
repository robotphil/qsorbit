"""Tests for the in-session IQ recorder and its sidecar.

The recorder writes the same v2 sidecar ``sdr capture`` does, so the same
rule applies: the sidecar is the part worth testing hardest, because a
``.iq`` file is only interpretable through it. These drive
:meth:`IqRecorder.run` directly with a fake subscription -- no threads, no
device -- so the write loop and the sidecar can be asserted exactly. The
thread lifecycle is a ``ReceiveSession`` concern and is tested there.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from qsorbit.core.recorder import IqRecorder, safe_filename
from qsorbit.core.sdr import AppliedSettings, DeviceError, SdrConfig
from qsorbit.core.sdr.stream import SubscriberStats, TimedBlock

RATE_HZ = 250_000
BYTE_RATE = 500_000  # uint8 I + Q
CENTER_HZ = 99_649_900
DURATION_S = 0.01
AN_INSTANT = datetime(2026, 9, 6, 8, 44, 0, tzinfo=UTC)


def an_applied() -> AppliedSettings:
    config = SdrConfig(center_hz=99_650_000, sample_rate_hz=RATE_HZ, gain_db=32.8)
    return AppliedSettings(
        requested=config,
        center_hz=CENTER_HZ,
        sample_rate_hz=RATE_HZ,
        gain_db=32.8,
        manual_gain=True,
        ppm=0,
        agc_enabled=False,
    )


def a_loss(*, lost_bytes: float = 0.0, loss_fraction: float = 0.0):
    """A stand-in for the stream's LossReport, of which the recorder reads
    only the two fields the sidecar records."""
    return SimpleNamespace(lost_bytes=lost_bytes, loss_fraction=loss_fraction)


class FakeSubscription:
    """Yields a fixed list of blocks and reports a fixed drop count.

    Satisfies the slice of :class:`~qsorbit.core.sdr.stream.IqSubscription`
    the recorder uses: ``timed_blocks`` and ``stats.blocks_dropped``.
    """

    def __init__(self, blocks: list[TimedBlock], *, dropped: int = 0) -> None:
        self._blocks = blocks
        self._dropped = dropped

    def timed_blocks(self, poll_s: float = 0.0):
        yield from self._blocks

    @property
    def stats(self) -> SubscriberStats:
        return SubscriberStats(
            name="recorder",
            blocks_offered=len(self._blocks),
            blocks_dropped=self._dropped,
            queue_blocks=8,
        )


class RaisingSubscription:
    """Yields its blocks and then raises, the way the real subscription
    re-raises a reader's error (a device unplugged mid-pass) at the end of
    a stream. The recorder must still leave a readable capture behind."""

    def __init__(self, blocks: list[TimedBlock], exc: BaseException) -> None:
        self._blocks = blocks
        self._exc = exc

    def timed_blocks(self, poll_s: float = 0.0):
        yield from self._blocks
        raise self._exc

    @property
    def stats(self) -> SubscriberStats:
        return SubscriberStats(
            name="recorder", blocks_offered=len(self._blocks), blocks_dropped=0, queue_blocks=8
        )


def a_block(payload: bytes, *, read_at: datetime = AN_INSTANT) -> TimedBlock:
    return TimedBlock(data=payload, read_at=read_at, duration_s=DURATION_S)


def a_recorder(tmp_path, subscription, **overrides) -> IqRecorder:
    settings = {
        "subscription": subscription,
        "iq_path": tmp_path / "cap.iq",
        "applied": an_applied(),
        "device_description": "RTL-SDR Blog V4",
        "station_hz": None,
        "loss_source": a_loss,
        "label": "A - Arrow V",
        "now": lambda: AN_INSTANT + timedelta(seconds=1),
    }
    return IqRecorder(**{**settings, **overrides})


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("A - Arrow V", "A-Arrow-V"),
            ("B - Arrow H", "B-Arrow-H"),
            ("already-safe_1.0", "already-safe_1.0"),
            ("  spaces  ", "spaces"),
            ("", "branch"),
            ("///", "branch"),
            ("weird/\\:*name", "weird-name"),
        ],
    )
    def test_it_sanitises_a_label_to_a_stem(self, label, expected):
        assert safe_filename(label) == expected


class TestRecorderWritesTheSamples:
    def test_it_writes_the_blocks_verbatim(self, tmp_path):
        sub = FakeSubscription([a_block(b"\x01\x02"), a_block(b"\x03\x04\x05")])
        recorder = a_recorder(tmp_path, sub)

        recorder.run()

        assert recorder.iq_path.read_bytes() == b"\x01\x02\x03\x04\x05"
        assert recorder.bytes_written == 5

    def test_it_creates_the_parent_directory(self, tmp_path):
        sub = FakeSubscription([a_block(b"\x00" * 4)])
        recorder = a_recorder(tmp_path, sub, iq_path=tmp_path / "nested" / "deeper" / "cap.iq")

        recorder.run()

        assert recorder.iq_path.is_file()

    def test_the_sidecar_sits_beside_the_capture(self, tmp_path):
        sub = FakeSubscription([a_block(b"\x00" * 4)])
        recorder = a_recorder(tmp_path, sub)

        recorder.run()

        assert recorder.sidecar_path == tmp_path / "cap.json"
        assert json.loads(recorder.sidecar_path.read_text(encoding="utf-8"))


class TestRecorderSidecar:
    def sidecar(self, tmp_path, sub, **overrides) -> dict:
        recorder = a_recorder(tmp_path, sub, **overrides)
        recorder.run()
        return json.loads(recorder.sidecar_path.read_text(encoding="utf-8"))

    def test_it_is_a_v2_sidecar_with_the_actual_settings(self, tmp_path):
        meta = self.sidecar(tmp_path, FakeSubscription([a_block(b"\x00" * 8)]))

        assert meta["sidecar_version"] == 2
        assert meta["actual_center_hz"] == CENTER_HZ
        assert meta["actual_sample_rate_hz"] == RATE_HZ
        assert meta["device"] == "RTL-SDR Blog V4"

    def test_it_records_the_bytes_and_a_derived_duration(self, tmp_path):
        # 1000 bytes at 500,000 bytes/s is 0.002 s -- derived from what was
        # written over the device's real byte rate, not asked for.
        payload = b"\x00" * 1000
        meta = self.sidecar(tmp_path, FakeSubscription([a_block(payload)]))

        assert meta["bytes"] == 1000
        assert meta["seconds"] == pytest.approx(1000 / BYTE_RATE)

    def test_started_utc_is_the_first_blocks_first_sample(self, tmp_path):
        # read_at is when the read returned; the samples began one block
        # duration earlier, and that is the epoch a replay needs.
        read_at = datetime(2026, 9, 6, 8, 44, 10, 500_000, tzinfo=UTC)
        block = a_block(b"\x00" * 4, read_at=read_at)
        meta = self.sidecar(tmp_path, FakeSubscription([block]))

        expected = (read_at - timedelta(seconds=DURATION_S)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        assert meta["started_utc"] == expected + "Z"

    def test_it_records_a_first_block_monotonic_reference(self, tmp_path):
        meta = self.sidecar(tmp_path, FakeSubscription([a_block(b"\x00" * 4)]))

        assert isinstance(meta["first_block_monotonic_s"], float)

    def test_its_own_buffer_drops_are_recorded_not_the_streams(self, tmp_path):
        # The recorder reports the drops *it* felt at its own queue, which
        # is what SubscriberStats.blocks_dropped carries.
        sub = FakeSubscription([a_block(b"\x00" * 4)], dropped=3)
        meta = self.sidecar(tmp_path, sub)

        assert meta["contiguous"] is False
        assert meta["blocks_dropped"] == 3

    def test_usb_loss_comes_from_the_loss_source(self, tmp_path):
        meta = self.sidecar(
            tmp_path,
            FakeSubscription([a_block(b"\x00" * 4)]),
            loss_source=lambda: a_loss(lost_bytes=8192.4, loss_fraction=0.0012),
        )

        assert meta["estimated_lost_bytes"] == 8192
        assert meta["loss_fraction"] == pytest.approx(0.0012)

    def test_the_station_offset_is_recorded_when_a_downlink_is_given(self, tmp_path):
        meta = self.sidecar(
            tmp_path, FakeSubscription([a_block(b"\x00" * 4)]), station_hz=99_900_000
        )

        assert meta["station_hz"] == 99_900_000
        assert meta["station_offset_hz"] == pytest.approx(99_900_000 - CENTER_HZ)

    def test_no_station_keys_without_a_downlink(self, tmp_path):
        meta = self.sidecar(tmp_path, FakeSubscription([a_block(b"\x00" * 4)]))

        assert "station_hz" not in meta


class TestPartialCapture:
    def test_the_sidecar_is_written_even_when_the_stream_raises_at_the_end(self, tmp_path):
        # The real subscription re-raises the reader's DeviceError after its
        # last block. The bytes already written are a usable partial
        # capture -- but only if the sidecar lands too, since a bare .iq
        # cannot be read. The fault is re-raised for the session to record.
        sub = RaisingSubscription([a_block(b"\x11\x22\x33\x44")], DeviceError("exhausted"))
        recorder = a_recorder(tmp_path, sub)

        with pytest.raises(DeviceError, match="exhausted"):
            recorder.run()

        assert recorder.iq_path.read_bytes() == b"\x11\x22\x33\x44"
        meta = json.loads(recorder.sidecar_path.read_text(encoding="utf-8"))
        assert meta["sidecar_version"] == 2
        assert meta["bytes"] == 4


class TestEmptyCapture:
    def test_a_capture_with_no_blocks_still_writes_a_sidecar(self, tmp_path):
        # A branch that never produced a block: the file is empty, the
        # start falls back to the end timestamp rather than being null, and
        # the monotonic reference is null because there was no first block.
        recorder = a_recorder(tmp_path, FakeSubscription([]))

        recorder.run()

        assert recorder.iq_path.read_bytes() == b""
        meta = json.loads(recorder.sidecar_path.read_text(encoding="utf-8"))
        assert meta["bytes"] == 0
        assert meta["first_block_monotonic_s"] is None
        # started_utc falls back to captured_at (the pinned end), both the
        # same instant here.
        assert meta["started_utc"].startswith("2026-09-06T08:44:01")
        assert meta["captured_utc"] == "2026-09-06T08:44:01Z"
