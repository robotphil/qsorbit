"""Tests for the capture utility and the sidecar it writes.

The sidecar is the part worth testing hardest. A ``.iq`` file is only
interpretable through it — get a frequency or a sample rate wrong there
and every spectrum computed from that capture is wrong in a way nothing
downstream can detect, because the samples themselves look perfectly
healthy.

Deliberately small numbers throughout: a 250 ksps capture of a few
hundredths of a second. Nothing here is measuring performance, and a
test that writes megabytes to prove a byte count is a slow test proving
the same thing.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

from qsorbit.core.sdr import (
    SIDECAR_VERSION,
    AppliedSettings,
    AutoGain,
    DeviceError,
    DeviceInfo,
    LossReport,
    SdrConfig,
    StreamStats,
    TunerType,
    capture_to_file,
)
from qsorbit.core.sdr.capture import _build_metadata

RATE_HZ = 250_000
BYTE_RATE = 500_000
REQUESTED_CENTER_HZ = 99_650_000
#: What the PLL actually reached. Deliberately *not* what was asked for:
#: an offset computed against the requested centre is wrong by exactly
#: this, and nothing downstream would notice.
ACTUAL_CENTER_HZ = 99_649_900
STATION_HZ = 99_900_000


def a_config(**overrides) -> SdrConfig:
    defaults = {
        "center_hz": REQUESTED_CENTER_HZ,
        "sample_rate_hz": RATE_HZ,
        "gain_db": 32.8,
    }
    return SdrConfig(**{**defaults, **overrides})


def an_applied(config: SdrConfig | None = None) -> AppliedSettings:
    """Settings as a device would report them, with the PLL a touch low."""
    config = config or a_config()
    return AppliedSettings(
        requested=config,
        center_hz=ACTUAL_CENTER_HZ,
        sample_rate_hz=RATE_HZ,
        gain_db=32.8,
        manual_gain=True,
        ppm=0,
        agc_enabled=False,
    )


def stats_with(*, blocks_dropped: int = 0, lost_bytes: float = 0.0) -> StreamStats:
    """A statistics record with the two losses set independently.

    Constructed rather than measured, so the sidecar's reporting can be
    tested against each fault in isolation — including combinations a
    live capture would be lucky to produce on demand.
    """
    bytes_read = 10_000
    return StreamStats(
        blocks_read=5,
        bytes_read=bytes_read,
        blocks_dropped=blocks_dropped,
        block_bytes=4_096,
        queue_blocks=16,
        reader_stopped_cleanly=True,
        loss=LossReport(
            reads=5,
            bytes_read=bytes_read,
            elapsed_s=(bytes_read + lost_bytes) / BYTE_RATE,
            byte_rate=BYTE_RATE,
            deficits_s=(),
        ),
    )


class FakeDevice:
    """An open, configurable device that streams a counting pattern.

    Quantises the centre frequency the way a real PLL does, so a test
    asserting on "the actual centre" is asserting on something that
    genuinely differs from the request.

    **This fake is not rate-limited, and that is worth knowing before
    reading anything into a drop count measured against it.** Real
    hardware paces the reader — at 2.048 Msps a 256 KiB block takes
    64 ms to arrive, so a consumer has to stall for a very long time to
    overflow a sixteen-block buffer. This fake returns instantly, so the
    reader can lap a consumer arbitrarily and the buffer overflows for
    reasons that say nothing whatever about whether the real pipeline
    keeps up. Any test that must not see a drop therefore caps
    ``max_blocks`` at what the capture actually needs, and the honest
    version of the question is measured at the bench, not here.
    """

    def __init__(
        self,
        *,
        block_bytes: int = 4_096,
        actual_rate_hz: float = RATE_HZ,
        max_blocks: int = 1_000,
    ) -> None:
        self._max_blocks = max_blocks
        self.index = 0
        self.is_open = True
        self.applied: AppliedSettings | None = None
        self.info = DeviceInfo(
            index=0,
            name="Generic RTL2832U OEM",
            manufacturer="RTLSDRBlog",
            product="Blog V4",
            serial="00000001",
            tuner=TunerType.R828D,
        )
        self._block_bytes = block_bytes
        self._actual_rate_hz = actual_rate_hz
        self.reads = 0

    def configure(self, config: SdrConfig) -> AppliedSettings:
        self.applied = AppliedSettings(
            requested=config,
            center_hz=ACTUAL_CENTER_HZ,
            sample_rate_hz=self._actual_rate_hz,
            gain_db=0.0 if config.uses_auto_gain else float(config.gain_db),
            manual_gain=not config.uses_auto_gain,
            ppm=config.ppm,
            agc_enabled=config.enable_agc,
        )
        return self.applied

    def read_raw(self, length: int) -> bytes:
        if self.reads >= self._max_blocks:
            raise DeviceError("fake device exhausted")
        block = bytes([self.reads % 256]) * min(length, self._block_bytes)
        self.reads += 1
        return block


class TestCaptureFile:
    def test_it_writes_exactly_the_requested_duration(self, tmp_path):
        device = FakeDevice()

        result = capture_to_file(
            device, a_config(), tmp_path / "cap.iq", seconds=0.02, block_bytes=4_096
        )

        assert result.iq_path.read_bytes().__len__() == int(0.02 * BYTE_RATE)

    def test_the_duration_is_computed_from_the_actual_sample_rate(self, tmp_path):
        # The clock quantises. A file sized from the requested rate is
        # the wrong length, and every time axis derived from it is
        # stretched.
        device = FakeDevice(actual_rate_hz=249_984)

        result = capture_to_file(
            device, a_config(), tmp_path / "cap.iq", seconds=0.02, block_bytes=4_096
        )

        assert len(result.iq_path.read_bytes()) == int(round(0.02 * 249_984 * 2))

    def test_it_creates_the_parent_directory(self, tmp_path):
        device = FakeDevice()

        result = capture_to_file(
            device,
            a_config(),
            tmp_path / "nested" / "deeper" / "cap.iq",
            seconds=0.01,
            block_bytes=4_096,
        )

        assert result.iq_path.is_file()

    def test_the_sidecar_sits_beside_the_capture(self, tmp_path):
        device = FakeDevice()

        result = capture_to_file(
            device, a_config(), tmp_path / "cap.iq", seconds=0.01, block_bytes=4_096
        )

        assert result.sidecar_path == tmp_path / "cap.json"
        assert json.loads(result.sidecar_path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_a_non_positive_duration_is_refused(self, tmp_path, bad):
        with pytest.raises(ValueError, match="seconds"):
            capture_to_file(FakeDevice(), a_config(), tmp_path / "cap.iq", seconds=bad)


class TestSidecarRecordsWhatHappened:
    def sidecar(self, tmp_path, **kwargs) -> dict:
        device = kwargs.pop("device", None) or FakeDevice()
        config = kwargs.pop("config", None) or a_config()
        result = capture_to_file(
            device,
            config,
            tmp_path / "cap.iq",
            seconds=0.01,
            block_bytes=4_096,
            **kwargs,
        )
        return json.loads(result.sidecar_path.read_text(encoding="utf-8"))

    def test_it_records_both_the_requested_and_the_actual_centre(self, tmp_path):
        meta = self.sidecar(tmp_path)

        assert meta["requested_center_hz"] == REQUESTED_CENTER_HZ
        assert meta["actual_center_hz"] == ACTUAL_CENTER_HZ

    def test_it_records_both_the_requested_and_the_actual_sample_rate(self, tmp_path):
        meta = self.sidecar(tmp_path, device=FakeDevice(actual_rate_hz=249_984))

        assert meta["requested_sample_rate_hz"] == RATE_HZ
        assert meta["actual_sample_rate_hz"] == 249_984

    def test_the_station_offset_is_measured_from_the_actual_centre(self, tmp_path):
        # The regression guard for why offset_from() lives on
        # AppliedSettings rather than on SdrConfig. Measured from the
        # request this would be 250,000 - a plausible-looking number
        # that is wrong by the PLL's quantisation, and wrong silently.
        meta = self.sidecar(tmp_path, station_hz=STATION_HZ)

        assert meta["station_hz"] == STATION_HZ
        assert meta["station_offset_hz"] == pytest.approx(STATION_HZ - ACTUAL_CENTER_HZ)
        assert meta["station_offset_hz"] != pytest.approx(STATION_HZ - REQUESTED_CENTER_HZ)

    def test_a_station_below_centre_gets_a_negative_offset(self, tmp_path):
        meta = self.sidecar(tmp_path, station_hz=99_500_000)

        assert meta["station_offset_hz"] < 0

    def test_no_station_keys_appear_when_none_was_given(self, tmp_path):
        meta = self.sidecar(tmp_path)

        assert "station_hz" not in meta
        assert "station_offset_hz" not in meta

    def test_manual_and_automatic_gain_are_distinguishable(self, tmp_path):
        manual = self.sidecar(tmp_path)
        auto = self.sidecar(tmp_path, config=a_config(gain_db=AutoGain.AUTO))

        assert manual["gain_mode"] == "manual"
        assert auto["gain_mode"] == "auto"

    def test_it_records_the_device_it_came_from(self, tmp_path):
        assert "Blog V4" in self.sidecar(tmp_path)["device"]

    def test_it_carries_a_schema_version(self, tmp_path):
        assert self.sidecar(tmp_path)["sidecar_version"] == SIDECAR_VERSION

    def test_it_describes_the_sample_format(self, tmp_path):
        assert "uint8" in self.sidecar(tmp_path)["format"]

    def test_the_timestamp_can_be_pinned(self, tmp_path):
        meta = self.sidecar(tmp_path, captured_at=datetime(2026, 8, 22, 14, 30, 0, tzinfo=UTC))

        assert meta["captured_utc"] == "2026-08-22T14:30:00Z"

    def test_the_schema_version_is_two(self, tmp_path):
        # The v2 bump is deliberate (started_utc + first_block_monotonic_s);
        # pin the literal so an accidental change to the constant fails here
        # as well as the constant-based test above.
        assert SIDECAR_VERSION == 2
        assert self.sidecar(tmp_path)["sidecar_version"] == 2

    def test_it_records_a_start_time_distinct_from_the_end(self, tmp_path):
        # captured_utc is the end; started_utc is the start. On a real
        # (fast) capture they are close, but both keys must be present and
        # the start carries millisecond precision the end does not.
        meta = self.sidecar(tmp_path)

        assert "started_utc" in meta
        assert "captured_utc" in meta
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", meta["started_utc"])

    def test_the_start_time_can_be_pinned_to_the_millisecond(self, tmp_path):
        meta = self.sidecar(
            tmp_path, started_at=datetime(2026, 9, 6, 8, 44, 0, 123_000, tzinfo=UTC)
        )

        assert meta["started_utc"] == "2026-09-06T08:44:00.123Z"

    def test_it_records_a_first_block_monotonic_reference(self, tmp_path):
        # The alignment reference for a dual capture: a monotonic reading
        # taken at the first block. On a single capture it is only
        # informational, but it must be a real number, not null.
        meta = self.sidecar(tmp_path)

        assert isinstance(meta["first_block_monotonic_s"], float)


class TestContiguity:
    def test_an_uninterrupted_capture_is_marked_contiguous(self, tmp_path):
        # 0.01 s at 250 ksps is 5,000 bytes, which is two 4,096-byte
        # blocks. Capping the fake there is what makes this a test of
        # the contiguity reporting rather than a test of whether the
        # reader thread happened to outrun the writer - see FakeDevice.
        device = FakeDevice(max_blocks=2)

        result = capture_to_file(
            device, a_config(), tmp_path / "cap.iq", seconds=0.01, block_bytes=4_096
        )
        meta = json.loads(result.sidecar_path.read_text(encoding="utf-8"))

        assert result.is_contiguous
        assert meta["contiguous"] is True
        assert meta["blocks_dropped"] == 0

    def test_a_capture_with_a_hole_says_so_in_its_own_metadata(self):
        # Driven through _build_metadata rather than through a real
        # capture on purpose. Making a live capture drop a block means
        # arranging for the consumer to lose a race, and a test that
        # depends on losing a race passes for the wrong reason on a fast
        # machine - which is the same as not testing it at all.
        meta = _build_metadata(
            applied=an_applied(),
            stats=stats_with(blocks_dropped=3),
            written=10_000,
            seconds=0.02,
            station_hz=None,
            device_description="RTL-SDR Blog V4",
            captured_at=datetime(2026, 8, 22, 14, 30, 0, tzinfo=UTC),
            started_at=datetime(2026, 8, 22, 14, 29, 58, tzinfo=UTC),
            first_block_monotonic_s=123.5,
        )

        assert meta["contiguous"] is False
        assert meta["blocks_dropped"] == 3

    def test_device_loss_is_reported_separately_from_dropped_blocks(self):
        # The two faults stay apart all the way into the sidecar, so a
        # fixture carries enough to say which one damaged it.
        meta = _build_metadata(
            applied=an_applied(),
            stats=stats_with(blocks_dropped=0, lost_bytes=8_192),
            written=10_000,
            seconds=0.02,
            station_hz=None,
            device_description="RTL-SDR Blog V4",
            captured_at=datetime(2026, 8, 22, 14, 30, 0, tzinfo=UTC),
            started_at=datetime(2026, 8, 22, 14, 29, 58, tzinfo=UTC),
            first_block_monotonic_s=123.5,
        )

        assert meta["contiguous"] is True
        assert meta["blocks_dropped"] == 0
        assert meta["estimated_lost_bytes"] == 8_192

    def test_describe_names_the_file_and_the_tuning_error(self, tmp_path):
        device = FakeDevice()

        result = capture_to_file(
            device, a_config(), tmp_path / "cap.iq", seconds=0.01, block_bytes=4_096
        )

        text = result.describe()

        assert "cap.iq" in text
        assert "-100" in text  # the PLL landed 100 Hz low
