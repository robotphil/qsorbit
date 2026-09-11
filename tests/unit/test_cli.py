"""Unit tests for the command-line interface.

The rotor is a ``MagicMock(spec=Rotor)``: scripted like a stub, but
spec'd against the real class, so a test can't keep passing against a
method the facade no longer has.

Nothing here opens a serial port, and the tests that don't use --send
assert that no rotor is built at all.
"""

import argparse
import json
import signal
import textwrap
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qsorbit import __version__
from qsorbit.__main__ import (
    DEFAULT_PLAN_HOURS,
    DEFAULT_TUNING_OFFSET_KHZ,
    UNNAMED_BRANCH_LABEL,
    _below_horizon_note,
    _build_branches,
    _build_selector,
    _command_receive,
    _command_shell,
    _declared_branches,
    _describe_mechanics,
    _listening_index,
    _note_single_device,
    _offset_clock,
    _open_quieting_log,
    _open_sdr,
    _parse_audio_device,
    _parse_replay_map,
    _print_quieting_log,
    _print_recordings,
    _profile_pusher,
    _push_profile_gains,
    _quit_on_sigint,
    _range_rate_interval,
    _readout_poll_interval_ms,
    _receive_shell_hub,
    _replay_clock_for,
    _replay_sdr_factory,
    _sdr_ppm,
    _spectrum_factory,
    _squelch_status_line,
    _stall_guard,
    _window_title,
    build_parser,
    main,
)
from qsorbit.core.combiner import DEFAULT_MARGIN_DB
from qsorbit.core.dsp.spectrum import SpectrumConfig
from qsorbit.core.dsp.spectrum_stream import SpectrumStream
from qsorbit.core.profiles import CATALOG_MANIFEST_FILENAME
from qsorbit.core.quieting_log import QuietingLog
from qsorbit.core.receive import DEFAULT_TRACKING_INTERVAL_S
from qsorbit.core.rotor import Arrival, HomingError, Position, Rotor, RotorErrorCode, RotorStatus
from qsorbit.core.sdr import (
    AppliedSettings,
    DeviceError,
    DeviceInfo,
    DeviceNotFoundError,
    SdrError,
    TunerType,
)
from qsorbit.core.sdr.replay import ReplaySdr
from qsorbit.core.station import load_station_config
from qsorbit.ui.theme import DEFAULT_THEME_NAME, DEFAULT_THEMES_DIR, discover_themes

# Vallado AIAA 2006-6753 Appendix C example, the same TLE used throughout
# tests/unit/tracker/. Times below sit a few days past its epoch, inside
# the window where SGP4 is well behaved.
TEME_EXAMPLE_TLE = """\
TEME EXAMPLE
1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753
2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667
"""

IN_WINDOW_TIME = "2000-07-01T18:50:00+00:00"

CONFIG = """
    [observer]
    latitude = 40.5
    longitude = -83.25
    altitude_m = 250.0

    [rotor]
    port = "COM5"
    baudrate = 19200

    [rotor.capabilities]
    azimuth_min_deg = 0.0
    azimuth_max_deg = 360.0
    elevation_min_deg = 0.0
    elevation_max_deg = 180.0
    azimuth_wrap = "extra_rotation"
    acceptance_window_deg = 2.5
    rs485_turnaround_s = 0.15
    firmware_version = "SatNOGS-v2.2.1"
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "qsorbit.toml"
    path.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def aligned_config_path(tmp_path):
    # CONFIG plus a real [rotor.alignment] - a station that has actually
    # measured its offset, as distinct from every other fixture's
    # default "nobody has calibrated this one" state. A distinct
    # filename from config_path's, deliberately: a test using both
    # fixtures at once (the differential offset check) needs two real,
    # independent files, not one clobbering the other's write.
    text = (
        textwrap.dedent(CONFIG) + "\n[rotor.alignment]\nazimuth_deg = 5.0\nelevation_deg = -2.0\n"
    )
    path = tmp_path / "qsorbit_aligned.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tle_path(tmp_path):
    path = tmp_path / "example.tle"
    path.write_text(TEME_EXAMPLE_TLE, encoding="utf-8")
    return path


#: A profile for the TEME EXAMPLE satellite (NORAD 00005) -- fictional
#: as a real bird, but a real profile shape, matching
#: tests/unit/profiles/test_catalog.py's VALID_PROFILE.
PLAN_PROFILE = """
    norad_id = 5
    name = "TEME EXAMPLE"

    [alive]
    status = "active"
    as_of = 2026-08-25
    source = "test fixture"

    [[transmitters]]
    downlink_hz = 435600000.0
    mode = "cw"
    reliability = "unconditional"
    notes = "test beacon"
"""


@pytest.fixture
def tle_dir_path(tmp_path):
    """A directory of one *.tle file, for --tle-dir."""
    directory = tmp_path / "tles"
    directory.mkdir()
    (directory / "teme.tle").write_text(TEME_EXAMPLE_TLE, encoding="utf-8")
    return directory


@pytest.fixture
def profiles_dir_path(tmp_path):
    """A directory of one profile *.toml file, matching tle_dir_path's satellite."""
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "teme.toml").write_text(textwrap.dedent(PLAN_PROFILE), encoding="utf-8")
    return directory


@pytest.fixture
def profiles_dir_with_manifest_path(profiles_dir_path):
    """profiles_dir_path, plus a CATALOG.toml shipped 2026-08-25."""
    (profiles_dir_path / CATALOG_MANIFEST_FILENAME).write_text(
        "shipped = 2026-08-25\n", encoding="utf-8"
    )
    return profiles_dir_path


@pytest.fixture
def plan_config_path(tmp_path):
    # Same observer test_pass_prediction.py uses for this exact TLE, so
    # the pass geometry here is proven elsewhere rather than a fresh,
    # unverified claim -- CONFIG's own observer is close but not
    # identical, and "close" isn't good enough for a fixture other
    # tests' pass counts depend on.
    text = textwrap.dedent(CONFIG).replace(
        "latitude = 40.5\nlongitude = -83.25", "latitude = 40.0\nlongitude = -83.0"
    )
    path = tmp_path / "qsorbit_plan.toml"
    path.write_text(text, encoding="utf-8")
    return path


def run_plan(argv, config_path):
    return main(["--config", str(config_path), *argv])


#: What a freshly homed rotator reports: slightly past zero on both
#: axes, which is normal for an axis resting against its end-stop.
HOMED_POSITION = Position(-1.5, 2.0)


def make_rotor(
    *,
    position=HOMED_POSITION,
    error=RotorErrorCode.NO_ERROR,
    firmware="SatNOGS-v2.2.1",
    arrived=True,
    connect_raises=None,
):
    """A stand-in rotor, spec'd against the real facade."""
    rotor = MagicMock(spec=Rotor)
    rotor.firmware_version = firmware
    if connect_raises is not None:
        rotor.connect.side_effect = connect_raises
    status = RotorStatus(firmware_version=firmware, error=error, position=position)
    rotor.connect.return_value = status
    rotor.status.return_value = status
    rotor.stop.return_value = position
    rotor.wait_for_arrival.return_value = Arrival(arrived=arrived, position=position, elapsed_s=4.2)
    return rotor


@pytest.fixture
def factory():
    """A rotor factory that records its calls and hands back one mock."""
    rotor = make_rotor()
    calls = []

    def build(config, on_homing_wait):
        calls.append((config, on_homing_wait))
        return rotor

    build.rotor = rotor
    build.calls = calls
    return build


def run(argv, config_path, factory):
    return main([*argv[:0], "--config", str(config_path), *argv], rotor_factory=factory)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_no_command_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 2

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_point_requires_a_tle(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["point"])
        assert exit_info.value.code == 2

    def test_send_defaults_to_off(self):
        # The safety-relevant default, pinned so it can't be flipped by
        # accident: computing is free, moving is opt-in.
        assert build_parser().parse_args(["point", "--tle", "x"]).send is False

    def test_goto_requires_az_and_el(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["goto", "--az", "10"])
        assert exit_info.value.code == 2

    def test_goto_send_defaults_to_off(self):
        # Same safety-relevant default as point, and for the same
        # reason: goto's whole job is a command that reaches the
        # hardware, so it earns no looser a default than point's.
        args = build_parser().parse_args(["goto", "--az", "10", "--el", "20"])
        assert args.send is False

    def test_status_takes_no_arguments(self):
        assert build_parser().parse_args(["status"]).command == "status"


# ---------------------------------------------------------------------------
# point
# ---------------------------------------------------------------------------


class TestPoint:
    def test_prints_sky_rotor_and_command(self, config_path, tle_path, factory, capsys):
        code = run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "TEME EXAMPLE" in out
        assert "Sky:" in out
        assert "Rotor:" in out
        assert "Command:   AZ" in out
        assert " EL" in out

    def test_says_nothing_was_sent(self, config_path, tle_path, factory, capsys):
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "Nothing was sent" in capsys.readouterr().out

    def test_does_not_even_build_a_rotor_without_send(self, config_path, tle_path, factory):
        # Computing must not touch the serial port. On a rotator whose
        # adapter resets on open, merely connecting triggers a re-home.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert factory.calls == []

    def test_says_no_calibration_is_applied(self, config_path, tle_path, factory, capsys):
        # sky_to_rotor is identity when the station has no configured
        # offset. An interface that didn't say so would be implying an
        # accuracy the software doesn't have.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "No alignment calibration is applied" in capsys.readouterr().out

    def test_a_configured_offset_is_applied_and_reported(
        self, aligned_config_path, tle_path, factory, capsys
    ):
        # The other half: a station that HAS measured an offset must
        # both get it applied to the commanded position and be told so,
        # rather than reading the same "no calibration" note regardless.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], aligned_config_path, factory)

        out = capsys.readouterr().out
        assert "No alignment calibration is applied" not in out
        assert "Alignment offset applied" in out
        assert "+5.0" in out
        assert "-2.0" in out

    def test_the_offset_actually_shifts_the_commanded_position(
        self, config_path, aligned_config_path, tle_path, factory, capsys
    ):
        # Differential check against the same TLE and time with and
        # without the offset, so this doesn't need to know the sky
        # position in advance - only that the two runs must differ by
        # exactly the configured offset.
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)
        plain_rotor_line = next(
            line for line in capsys.readouterr().out.splitlines() if line.startswith("Rotor:")
        )

        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], aligned_config_path, factory)
        aligned_rotor_line = next(
            line for line in capsys.readouterr().out.splitlines() if line.startswith("Rotor:")
        )

        def axes(line):
            parts = line.replace("Rotor:", "").split()
            return float(parts[1]), float(parts[3])

        plain_az, plain_el = axes(plain_rotor_line)
        aligned_az, aligned_el = axes(aligned_rotor_line)
        assert aligned_az == pytest.approx(plain_az + 5.0)
        assert aligned_el == pytest.approx(plain_el - 2.0)

    def test_labels_the_rotor_line_as_an_axis_command(self, config_path, tle_path, factory, capsys):
        run(["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME], config_path, factory)

        assert "(axis command)" in capsys.readouterr().out

    def test_send_moves_and_reports_arrival(self, config_path, tle_path, factory, capsys):
        code = run(
            ["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME, "--send"],
            config_path,
            factory,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert factory.rotor.move_to.call_count == 1
        assert isinstance(factory.rotor.move_to.call_args.args[0], Position)
        assert "Arrived:" in out

    def test_send_closes_the_port(self, config_path, tle_path, factory):
        run(
            ["point", "--tle", str(tle_path), "--at", IN_WINDOW_TIME, "--send"],
            config_path,
            factory,
        )

        assert factory.rotor.close.call_count == 1

    def test_failure_to_settle_is_reported_and_non_zero(self, config_path, tle_path, capsys):
        rotor = make_rotor(arrived=False)
        factory = lambda config, on_wait: rotor  # noqa: E731

        code = main(
            [
                "--config",
                str(config_path),
                "point",
                "--tle",
                str(tle_path),
                "--at",
                IN_WINDOW_TIME,
                "--send",
            ],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Did not settle" in capsys.readouterr().err

    def test_out_of_range_target_is_refused_before_connecting(
        self, tmp_path, tle_path, factory, capsys
    ):
        # Elevation limits of 91-92 degrees can never be satisfied by a
        # sky position, because AzEl caps elevation at 90 by definition.
        # That makes this refusal deterministic without depending on
        # where the satellite happens to be.
        config = tmp_path / "narrow.toml"
        config.write_text(
            textwrap.dedent(CONFIG)
            .replace("elevation_min_deg = 0.0", "elevation_min_deg = 91.0")
            .replace("elevation_max_deg = 180.0", "elevation_max_deg = 92.0"),
            encoding="utf-8",
        )

        code = main(
            [
                "--config",
                str(config),
                "point",
                "--tle",
                str(tle_path),
                "--at",
                IN_WINDOW_TIME,
                "--send",
            ],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Out of range" in capsys.readouterr().err
        assert factory.calls == []

    def test_bad_time_is_rejected(self, config_path, tle_path, factory, capsys):
        code = run(
            ["point", "--tle", str(tle_path), "--at", "half past four"], config_path, factory
        )

        assert code == 1
        assert "ISO 8601" in capsys.readouterr().err

    def test_missing_tle_file(self, config_path, tmp_path, factory, capsys):
        code = run(["point", "--tle", str(tmp_path / "nope.tle")], config_path, factory)

        assert code == 1
        assert capsys.readouterr().err.startswith("Error:")


# ---------------------------------------------------------------------------
# goto
# ---------------------------------------------------------------------------


class TestGoto:
    def test_prints_rotor_and_command(self, config_path, factory, capsys):
        code = run(["goto", "--az", "151", "--el", "19"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "Rotor:     AZ 151.0  EL 19.0  (axis command)" in out
        assert "Command:   AZ151.0 EL19.0" in out

    def test_says_nothing_was_sent(self, config_path, factory, capsys):
        run(["goto", "--az", "151", "--el", "19"], config_path, factory)

        assert "Nothing was sent" in capsys.readouterr().out

    def test_does_not_even_build_a_rotor_without_send(self, config_path, factory):
        # Same reasoning as point: computing must not touch the serial
        # port, since merely connecting can trigger a re-home.
        run(["goto", "--az", "151", "--el", "19"], config_path, factory)

        assert factory.calls == []

    def test_send_moves_and_reports_arrival(self, config_path, factory, capsys):
        code = run(["goto", "--az", "151", "--el", "19", "--send"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert factory.rotor.move_to.call_count == 1
        sent = factory.rotor.move_to.call_args.args[0]
        assert isinstance(sent, Position)
        assert (sent.azimuth, sent.elevation) == (151.0, 19.0)
        assert "Arrived:" in out

    def test_send_closes_the_port(self, config_path, factory):
        run(["goto", "--az", "151", "--el", "19", "--send"], config_path, factory)

        assert factory.rotor.close.call_count == 1

    def test_failure_to_settle_is_reported_and_non_zero(self, config_path, capsys):
        rotor = make_rotor(arrived=False)
        factory = lambda config, on_wait: rotor  # noqa: E731

        code = main(
            ["--config", str(config_path), "goto", "--az", "151", "--el", "19", "--send"],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Did not settle" in capsys.readouterr().err

    def test_out_of_range_target_is_refused_before_connecting(self, tmp_path, factory, capsys):
        config = tmp_path / "narrow.toml"
        config.write_text(
            textwrap.dedent(CONFIG)
            .replace("elevation_min_deg = 0.0", "elevation_min_deg = 91.0")
            .replace("elevation_max_deg = 180.0", "elevation_max_deg = 92.0"),
            encoding="utf-8",
        )

        code = main(
            ["--config", str(config), "goto", "--az", "151", "--el", "19", "--send"],
            rotor_factory=factory,
        )

        assert code == 1
        assert "Out of range" in capsys.readouterr().err
        assert factory.calls == []

    def test_no_alignment_calibration_claim(self, config_path, factory, capsys):
        # Unlike point, goto is a raw axis command - it is not built from
        # a sky position at all, so there is nothing for
        # UNCALIBRATED_NOTE to say here that the "(axis command)" label
        # doesn't already say. Pinning the absence so a future change
        # that starts running goto's target through sky_to_rotor (it
        # shouldn't) gets caught by this test changing meaning.
        run(["goto", "--az", "151", "--el", "19"], config_path, factory)

        assert "No alignment calibration is applied" not in capsys.readouterr().out

    def test_bad_position_is_rejected(self, config_path, factory, capsys):
        # Position's own corruption filter, exercised through the CLI:
        # a garbled or absurd axis number should read as an ordinary
        # error, not a traceback.
        code = run(["goto", "--az", "99999", "--el", "19"], config_path, factory)

        assert code == 1
        assert capsys.readouterr().err.startswith("Error:")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


class TestPlanParser:
    def test_plan_requires_a_tle_dir(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["plan"])
        assert exit_info.value.code == 2

    def test_hours_defaults(self):
        args = build_parser().parse_args(["plan", "--tle-dir", "x"])
        assert args.hours == DEFAULT_PLAN_HOURS

    def test_visual_defaults_to_off(self):
        args = build_parser().parse_args(["plan", "--tle-dir", "x"])
        assert args.visual is False

    def test_at_defaults_to_none(self):
        # None is "now" to _parse_time -- same convention as point's --at.
        args = build_parser().parse_args(["plan", "--tle-dir", "x"])
        assert args.at is None

    def test_profiles_dir_defaults_to_none(self):
        # None means "use the curated starter set" -- load_profile_catalog()
        # with no argument -- not an empty directory.
        args = build_parser().parse_args(["plan", "--tle-dir", "x"])
        assert args.profiles_dir is None

    def test_refresh_catalogue_defaults_to_off(self):
        args = build_parser().parse_args(["plan", "--tle-dir", "x"])
        assert args.refresh_catalogue is False


class TestPlan:
    def test_lists_a_pass_with_its_transmitter(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "TEME EXAMPLE" in out
        assert "(unconditional)" in out
        assert "AOS" in out
        assert "TCA" in out
        assert "LOS" in out
        assert "435.6000 MHz down" in out
        assert "cw" in out
        assert "test beacon" in out

    def test_multiple_passes_come_out_in_chronological_order(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        # Same TLE, observer, and 48-hour window as
        # test_pass_prediction.py's "finds multiple passes in a two day
        # window" case -- that test is where ">5 passes" is proven; this
        # one only checks the CLI prints more than one of them, in order.
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        aos_lines = [line for line in out.splitlines() if line.strip().startswith("AOS")]
        assert len(aos_lines) > 1
        aos_times = [line.split()[1] for line in aos_lines]
        assert aos_times == sorted(aos_times)

    def test_no_passes_in_window_says_so(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "0.001",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "Nothing above the horizon in that window." in out

    def test_visual_flag_adds_an_illumination_line(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
                "--visual",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "naked-eye visible near TCA:" in out

    def test_without_visual_no_illumination_line(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "naked-eye visible" not in out

    def test_a_tle_with_no_matching_profile_is_skipped_not_fatal(
        self, plan_config_path, tle_dir_path, tmp_path, capsys
    ):
        empty_profiles_dir = tmp_path / "empty_profiles"
        empty_profiles_dir.mkdir()

        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(empty_profiles_dir),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        assert code == 0
        assert "No curated profile" in capsys.readouterr().err

    def test_an_unparseable_tle_file_is_skipped_not_fatal(
        self, plan_config_path, profiles_dir_path, tmp_path, capsys
    ):
        garbled_dir = tmp_path / "garbled"
        garbled_dir.mkdir()
        (garbled_dir / "garbage.tle").write_text("not a tle\nat all\n", encoding="utf-8")

        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(garbled_dir),
                "--profiles-dir",
                str(profiles_dir_path),
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        assert code == 0
        assert "Could not read" in capsys.readouterr().err

    def test_prints_catalogue_staleness_when_a_manifest_is_present(
        self, plan_config_path, tle_dir_path, profiles_dir_with_manifest_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_with_manifest_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "Curated catalogue shipped 2026-08-25 (3 d ago)." in out

    def test_no_staleness_line_when_no_manifest_present(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--at",
                "2026-08-28T00:00:00+00:00",
                "--hours",
                "48",
            ],
            plan_config_path,
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "shipped" not in out

    def test_refresh_catalogue_fails_with_a_clear_error(
        self, plan_config_path, tle_dir_path, profiles_dir_path, capsys
    ):
        code = run_plan(
            [
                "plan",
                "--tle-dir",
                str(tle_dir_path),
                "--profiles-dir",
                str(profiles_dir_path),
                "--refresh-catalogue",
            ],
            plan_config_path,
        )

        err = capsys.readouterr().err
        assert code == 1
        assert err.startswith("Error:")
        assert "no network source configured" in err

    def test_missing_tle_dir_is_an_error(self, plan_config_path, tmp_path, capsys):
        missing = tmp_path / "does-not-exist"

        code = run_plan(["plan", "--tle-dir", str(missing)], plan_config_path)

        assert code == 1
        assert capsys.readouterr().err.startswith("Error:")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_reports_the_essentials(self, config_path, factory, capsys):
        code = run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "SatNOGS-v2.2.1" in out
        assert "COM5" in out
        assert str(config_path) in out
        assert "Error:     none" in out

    def test_reports_the_step_the_cadence_will_actually_command(self, config_path, factory, capsys):
        # The deadband is 2.5 and the step is 3.0, because the loop can
        # only command in whole ticks. Measured on hardware in Session
        # 32; invisible from anywhere in the application until now.
        run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert "Cadence:" in out
        assert "2.5 deg deadband" in out
        assert "3 deg steps" in out

    def test_reports_when_no_profiles_are_declared(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert "Profiles:  none declared" in out
        assert "[rotor.profiles]" in out

    def test_reports_declared_profiles_and_the_active_one(self, tmp_path, factory, capsys):
        path = tmp_path / "profiled.toml"
        path.write_text(
            textwrap.dedent(CONFIG).replace('port = "COM5"', 'port = "COM5"\nprofile = "tracking"')
            + textwrap.dedent(
                """
                [rotor.profiles.stock]
                deadband_deg = 2.5
                interval_s = 1.0

                [rotor.profiles.tracking]
                deadband_deg = 0.25
                interval_s = 0.5
                arrival_window_deg = 1.0
                """
            ),
            encoding="utf-8",
        )
        run(["status"], path, factory)

        out = capsys.readouterr().out
        assert "Profiles:  stock, tracking" in out
        assert "tracking profile" in out
        assert "0.5 deg steps" in out

    def test_a_rotor_profile_does_not_move_the_doppler_sampler(self):
        # The two ticks look interchangeable and were the same value
        # before profiles existed. They are different concepts: the
        # profile drives the rotor, this paces the range-rate loop on
        # the receive path, which runs with no rotor attached at all.
        # A profile moving it would put an unmeasured variable inside
        # Chunk E's combined-versus-single comparison.
        args = argparse.Namespace(interval=None, rotor_profile="tracking")
        assert _range_rate_interval(args) == DEFAULT_TRACKING_INTERVAL_S

    def test_an_explicit_interval_still_moves_the_doppler_sampler(self):
        # An operator who asked for a specific tick asked for exactly
        # that, on both.
        args = argparse.Namespace(interval=0.25, rotor_profile=None)
        assert _range_rate_interval(args) == 0.25

    def test_the_readout_falls_back_to_its_own_default_interval(self):
        # `receive --window --send` with no --interval used to reach
        # int(None * 1000) and die building the window. --interval
        # defaults to None so a profile can decide the rotor's cadence;
        # a readout's repaint rate is a different quantity that merely
        # shared the flag, and the absence of one is not the absence of
        # the other.
        #
        # 4321 rather than the real 1000 on purpose: an assertion that
        # passes because some unrelated value in the fixture happens to
        # equal the answer is not evidence.
        args = argparse.Namespace(interval=None, rotor_profile="tracking")
        assert _readout_poll_interval_ms(args, 4321) == 4321

    def test_an_explicit_interval_still_moves_the_readout(self):
        # Unchanged behaviour where --interval was given: an operator
        # who named a tick asked for exactly that, here too.
        args = argparse.Namespace(interval=0.25, rotor_profile=None)
        assert _readout_poll_interval_ms(args, 4321) == 250

    def test_the_receive_parser_still_leaves_interval_unset(self):
        # The guard above only earns its keep while --interval really
        # can arrive as None. If a numeric default is ever restored,
        # this fails and points at the reason the guard exists rather
        # than leaving it looking like dead defensiveness.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.interval is None

    def test_reports_no_alignment_recorded_by_default(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert "Alignment: none recorded" in out
        assert "No alignment calibration is applied" in out

    def test_reports_a_configured_alignment_offset(self, aligned_config_path, factory, capsys):
        run(["status"], aligned_config_path, factory)

        out = capsys.readouterr().out
        assert "Alignment: AZ +5.0  EL -2.0" in out
        assert "Alignment offset applied" in out
        assert "No alignment calibration is applied" not in out

    def test_labels_position_as_an_axis_reading(self, config_path, factory, capsys):
        # The rotor reports -1.5 after homing. Printed without a label,
        # that reads like a compass bearing, which it is not.
        code = run(["status"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "AZ -1.5" in out
        assert "not compass bearings" in out

    def test_shows_travel_limits_as_axis_travel(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        assert "(axis travel)" in capsys.readouterr().out

    def test_never_moves_anything(self, config_path, factory, capsys):
        run(["status"], config_path, factory)

        assert factory.rotor.move_to.call_count == 0
        assert factory.rotor.stop.call_count == 0

    def test_warns_on_a_firmware_mismatch(self, config_path, capsys):
        rotor = make_rotor(firmware="SatNOGS-v2.2")
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        out = capsys.readouterr().out
        assert code == 0
        assert "Config declares SatNOGS-v2.2.1" in out

    def test_unhealthy_rotor_exits_non_zero(self, config_path, capsys):
        rotor = make_rotor(error=RotorErrorCode.OVER_TEMPERATURE)
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "over temperature" in capsys.readouterr().out

    def test_latched_homing_error_is_shown_with_the_fix(self, config_path, capsys):
        # status() reports it rather than raising, so the operator can
        # see the state - and the line has to say what to do about it.
        rotor = make_rotor(error=RotorErrorCode.HOMING_ERROR)
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "power-cycle the controller" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_reports_where_it_stopped(self, config_path, factory, capsys):
        code = run(["stop"], config_path, factory)

        assert code == 0
        assert factory.rotor.stop.call_count == 1
        assert "Stopped:   AZ -1.5" in capsys.readouterr().out

    def test_says_it_is_not_an_emergency_stop(self, config_path, factory, capsys):
        # It halts a converging loop by setting the setpoint to the
        # current position. It cannot stop a diverging one, and a user
        # who believes otherwise reaches for the wrong thing.
        run(["stop"], config_path, factory)

        out = capsys.readouterr().out
        assert "cannot stop a diverging one" in out
        assert "power switch" in out


# ---------------------------------------------------------------------------
# Failure paths shared by every command
# ---------------------------------------------------------------------------


class TestFailures:
    def test_homing_failure_is_its_own_message(self, config_path, capsys):
        rotor = make_rotor(connect_raises=HomingError("Power-cycle the controller."))
        code = main(["--config", str(config_path), "status"], rotor_factory=lambda c, w: rotor)

        assert code == 1
        assert "Homing failure" in capsys.readouterr().err

    def test_missing_config_lists_where_it_looked(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APPDATA", str(tmp_path / "empty"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

        code = main(["status"], rotor_factory=lambda c, w: make_rotor())

        assert code == 1
        assert "No station config file found" in capsys.readouterr().err

    def test_explicitly_named_config_that_is_missing(self, tmp_path, capsys):
        code = main(
            ["--config", str(tmp_path / "nope.toml"), "status"],
            rotor_factory=lambda c, w: make_rotor(),
        )

        assert code == 1
        assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# SDR
# ---------------------------------------------------------------------------

V4_GAIN_STEPS_DB = (0.0, 0.9, 12.5, 32.8, 49.6)


class FakeSdr:
    """A stand-in device, written by hand rather than mocked.

    ``capture_to_file`` drives a real streaming thread through this, so
    it has to behave like a device rather than answer every attribute
    truthily the way a ``MagicMock`` would — and it caps its output so
    the reader cannot lap the writer and manufacture a drop that says
    nothing about real hardware.
    """

    def __init__(self, *, max_blocks=8, gain_db=None, index=0, serial=""):
        self.index = index
        self.is_open = False
        self.applied = None
        self.configured = []
        self.reads = 0
        self._max_blocks = max_blocks
        self._forced_gain = gain_db
        self.info = DeviceInfo(
            index=index,
            name="Generic RTL2832U OEM",
            manufacturer="RTLSDRBlog",
            product="Blog V4",
            serial=serial,
            tuner=TunerType.R828D,
        )

    def __enter__(self):
        self.is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_open = False

    def supported_gains_db(self):
        return V4_GAIN_STEPS_DB

    def configure(self, config):
        self.configured.append(config)
        gain = self._forced_gain
        if gain is None:
            gain = 0.0 if config.uses_auto_gain else float(config.gain_db)
        self.applied = AppliedSettings(
            requested=config,
            center_hz=config.center_hz,
            sample_rate_hz=config.sample_rate_hz,
            gain_db=gain,
            manual_gain=not config.uses_auto_gain,
            ppm=config.ppm,
            agc_enabled=config.enable_agc,
        )
        return self.applied

    def read_raw(self, length):
        if self.reads >= self._max_blocks:
            raise DeviceError("fake device exhausted")
        self.reads += 1
        return bytes([self.reads % 256]) * length


def sdr_factory(device=None):
    """An SDR factory that records its calls and hands back one device."""
    device = device or FakeSdr()

    def build(config, branch=None):
        build.calls.append(config)
        build.branches.append(branch)
        return device

    build.calls = []
    build.branches = []
    build.device = device
    return build


def run_sdr(argv, config_path, factory):
    return main(["--config", str(config_path), *argv], sdr_factory=factory)


class TestSdrParser:
    def test_sdr_without_a_subcommand_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exit_info:
            main(["sdr"])

        assert exit_info.value.code == 2

    def test_capture_needs_somewhere_to_write(self):
        with pytest.raises(SystemExit):
            main(["sdr", "capture", "--station", "99.9", "--gain", "32.8"])

    def test_capture_needs_a_gain_decision(self):
        # No default gain, on purpose: a default is how a capture comes
        # back empty with nobody noticing.
        with pytest.raises(SystemExit):
            main(["sdr", "capture", "--station", "99.9", "--out", "x.iq"])

    def test_a_station_and_an_explicit_centre_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            main(
                [
                    "sdr",
                    "capture",
                    "--station",
                    "99.9",
                    "--center",
                    "99.65",
                    "--gain",
                    "32.8",
                    "--out",
                    "x.iq",
                ]
            )


class TestSdrInfo:
    def test_it_reports_the_device_and_its_gain_table(self, config_path, capsys):
        factory = sdr_factory()

        code = run_sdr(["sdr", "info"], config_path, factory)

        out = capsys.readouterr().out
        assert code == 0
        assert "Blog V4" in out
        assert "5 steps" in out
        assert "49.6" in out

    def test_it_closes_the_device_again(self, config_path):
        factory = sdr_factory()

        run_sdr(["sdr", "info"], config_path, factory)

        assert not factory.device.is_open


class TestSdrCapture:
    def test_it_tunes_below_the_station_by_default(self, config_path, tmp_path):
        # The off-centre discipline, made the default so it cannot be
        # forgotten: a peak at the centre of the passband is
        # indistinguishable from the receiver's own DC spike.
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].center_hz == pytest.approx(99_650_000.0)

    def test_the_offset_is_adjustable(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "162.55",
                "--offset",
                "150",
                "--gain",
                "49.6",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].center_hz == pytest.approx(162_400_000.0)

    def test_an_explicit_centre_records_no_station(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )
        meta = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))

        assert factory.device.configured[0].center_hz == pytest.approx(99_650_000.0)
        assert "station_hz" not in meta

    def test_it_writes_the_capture_and_its_sidecar(self, config_path, tmp_path, capsys):
        factory = sdr_factory()

        code = run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert code == 0
        assert (tmp_path / "cap.iq").is_file()
        assert (tmp_path / "cap.json").is_file()
        assert "Sidecar" in capsys.readouterr().out

    def test_a_zero_gain_report_is_called_out(self, config_path, tmp_path, capsys):
        # The bring-up failure in executable form: auto gain reported
        # 0.0 dB and captured a flat noise floor, with nothing raising.
        factory = sdr_factory(FakeSdr(gain_db=0.0))

        run_sdr(
            [
                "sdr",
                "capture",
                "--station",
                "99.9",
                "--auto-gain",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert "0.0 dB" in capsys.readouterr().out

    def test_the_station_config_ppm_is_applied(self, config_path, tmp_path):
        factory = sdr_factory()

        run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--seconds",
                "0.001",
                "--rate",
                "250000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert factory.device.configured[0].ppm == 0

    def test_a_bad_sample_rate_is_reported_not_traced(self, config_path, tmp_path, capsys):
        # 500 kHz falls in the RTL2832U's genuine gap between its two
        # rate windows. The operator gets a sentence, not a traceback.
        factory = sdr_factory()

        code = run_sdr(
            [
                "sdr",
                "capture",
                "--center",
                "99.65",
                "--gain",
                "32.8",
                "--rate",
                "500000",
                "--out",
                str(tmp_path / "cap.iq"),
            ],
            config_path,
            factory,
        )

        assert code == 1
        assert "sample_rate_hz" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------


class TestSquelchStatusLine:
    """``_squelch_status_line`` in isolation - a pure helper, not the session."""

    def test_muting_enabled_says_so(self):
        line = _squelch_status_line(mute=True, open_above_db=3.0, close_below_db=1.5)
        assert line == "Squelch:   muting enabled, open at/above 3.0 dB, close at/below 1.5 dB"

    def test_muting_off_still_names_the_thresholds(self):
        # The whole point of Chunk I's "always measure, optionally mute":
        # the gate is still configured and running even when nothing it
        # decides reaches the speaker, and the banner should say so
        # rather than reading like the squelch does not exist.
        line = _squelch_status_line(mute=False, open_above_db=3.0, close_below_db=1.5)
        assert "open at/above 3.0 dB" in line
        assert "close at/below 1.5 dB" in line
        assert "muting off" in line

    def test_muting_on_and_off_read_differently(self):
        on = _squelch_status_line(mute=True, open_above_db=3.0, close_below_db=1.5)
        off = _squelch_status_line(mute=False, open_above_db=3.0, close_below_db=1.5)
        assert on != off

    def test_thresholds_are_formatted_to_one_decimal(self):
        line = _squelch_status_line(mute=True, open_above_db=2.75, close_below_db=1.05)
        assert "2.8 dB" in line
        assert "1.1 dB" in line


class TestParseAudioDevice:
    """``_parse_audio_device`` in isolation."""

    def test_unset_stays_none(self):
        # None is sounddevice's own "system default" sentinel - the
        # pre-existing behaviour with no --audio-device given at all.
        assert _parse_audio_device(None) is None

    def test_a_numeric_string_becomes_an_int_index(self):
        assert _parse_audio_device("2") == 2
        assert isinstance(_parse_audio_device("2"), int)

    def test_a_name_substring_stays_a_string(self):
        assert _parse_audio_device("USB Audio") == "USB Audio"

    def test_a_negative_numeric_string_becomes_a_negative_int(self):
        # sounddevice uses negative indices for some default-device
        # sentinels of its own; this must not be mistaken for a name.
        assert _parse_audio_device("-1") == -1


class TestSpectrumFactory:
    """``_spectrum_factory`` in isolation - a pure helper, not the session."""

    def a_spectrum_config(self) -> SpectrumConfig:
        return SpectrumConfig(fft_size=2048, sample_rate_hz=2_048_000.0)

    def test_no_window_means_no_factory_at_all(self):
        # The whole point: ReceiveSession never even builds a
        # SpectrumStream, rather than building one and starving it of a
        # drain. Constructing a SpectrumStream starts no thread on its
        # own (that happens in .start()), so "returns None" is the
        # entire test - there is nothing further to spin up and tear
        # down here.
        assert _spectrum_factory(False, self.a_spectrum_config()) is None

    def test_a_window_means_a_working_factory(self):
        factory = _spectrum_factory(True, self.a_spectrum_config())

        assert factory is not None
        stream = factory([])
        assert isinstance(stream, SpectrumStream)


class FakeQuittableApp:
    """Stands in for QApplication - only the one method _quit_on_sigint uses."""

    def __init__(self):
        self.quit_calls = 0

    def quit(self):
        self.quit_calls += 1


class TestQuitOnSigint:
    """``_quit_on_sigint`` in isolation, with no real Qt or signal delivery.

    Signal handlers installed via ``signal.signal`` are plain callables -
    invoking the installed handler directly is the standard, safe way to
    test one without raising a real ``SIGINT`` at the test process (which
    would risk killing the test runner itself if anything went wrong).
    """

    def test_the_installed_handler_quits_the_app(self):
        app = FakeQuittableApp()

        with _quit_on_sigint(app):
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)

        assert app.quit_calls == 1

    def test_the_previous_handler_is_restored_on_the_way_out(self):
        app = FakeQuittableApp()
        sentinel = signal.getsignal(signal.SIGINT)

        with _quit_on_sigint(app):
            assert signal.getsignal(signal.SIGINT) is not sentinel

        assert signal.getsignal(signal.SIGINT) is sentinel

    def test_the_previous_handler_is_restored_even_if_the_body_raises(self):
        app = FakeQuittableApp()
        sentinel = signal.getsignal(signal.SIGINT)

        with pytest.raises(RuntimeError):
            with _quit_on_sigint(app):
                raise RuntimeError("the body failed, not the handler's business")

        assert signal.getsignal(signal.SIGINT) is sentinel

    def test_only_the_installed_handler_is_active_inside_the_block(self):
        # Two nested installs, restored in the right order - guards
        # against a bug where the "previous" captured is always the
        # original rather than whichever one was actually active.
        app = FakeQuittableApp()
        sentinel = signal.getsignal(signal.SIGINT)

        with _quit_on_sigint(app):
            first = signal.getsignal(signal.SIGINT)
            with _quit_on_sigint(app):
                assert signal.getsignal(signal.SIGINT) is not first
            assert signal.getsignal(signal.SIGINT) is first

        assert signal.getsignal(signal.SIGINT) is sentinel


class TestReceiveParser:
    """Parser-level checks only. The session itself is tested in test_receive.py."""

    def test_a_tle_is_required(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--downlink", "145.95", "--gain", "40"])
        assert exit_info.value.code == 2

    def test_a_downlink_is_required(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--tle", "x", "--gain", "40"])
        assert exit_info.value.code == 2

    def test_a_gain_choice_is_required(self):
        # Deliberately has no default, same as 'sdr capture': a default
        # gain is how a pass comes back silent with nobody noticing.
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["receive", "--tle", "x", "--downlink", "145.95"])
        assert exit_info.value.code == 2

    def test_gain_and_auto_gain_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(
                ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40", "--auto-gain"]
            )
        assert exit_info.value.code == 2

    def test_send_defaults_to_off(self):
        # The safety-relevant default again, and the one that makes a
        # receive possible with no rotor connected at all.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.send is False

    def test_the_window_and_the_squelch_both_default_to_off(self):
        # The squelch default is a Chunk G decision with a reason: a mute
        # set slightly too tight makes a working receiver sound exactly
        # like a broken one, which is the last thing wanted while
        # pointing at a weak downlink for the first time.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.window is False
        assert args.squelch is False

    def test_the_audio_device_defaults_to_unset(self):
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.audio_device is None

    def test_the_audio_device_accepts_a_name_or_an_index(self):
        args = build_parser().parse_args(
            [
                "receive",
                "--tle",
                "x",
                "--downlink",
                "145.95",
                "--gain",
                "40",
                "--audio-device",
                "USB Audio",
            ]
        )
        assert args.audio_device == "USB Audio"

    def test_the_tuning_offset_defaults_to_the_project_convention(self):
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.offset == DEFAULT_TUNING_OFFSET_KHZ


class TestTrackLogParser:
    """``--track-log`` is available wherever a rotor can be driven."""

    def test_it_is_unset_by_default(self):
        # Opt-in matters here beyond taste: the sampling adds serial
        # reads and a target computation to the path where CPU is
        # measured to turn into lost USB samples.
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "145.95", "--gain", "40"]
        )
        assert args.track_log is None

    def test_receive_accepts_it(self):
        args = build_parser().parse_args(
            [
                "receive",
                "--tle",
                "x",
                "--downlink",
                "145.95",
                "--gain",
                "40",
                "--send",
                "--track-log",
                "pass.csv",
            ]
        )
        assert args.track_log == "pass.csv"

    def test_shell_accepts_it_too(self):
        # Chunk E's acceptance runs on the receive path and Chunk H's
        # runs rotor-only, so both need the same instrument -- otherwise
        # the two halves of the evidence come from different tools.
        args = build_parser().parse_args(["shell", "--send", "--track-log", "pass.csv"])
        assert args.track_log == "pass.csv"

    def test_shell_refuses_it_without_a_rotor(self, config_path, factory, capsys):
        # Two of shell's three modes never reach the code that builds a
        # log at all, so the parser accepting the flag is not the same
        # claim as the command honouring it.
        #
        # This is the only --track-log refusal shell still has. The
        # rotor-only one was removed when that path moved onto
        # TrackingThread, and it is deliberately not replaced by a test
        # asserting the flag now works there: reaching that would need
        # Qt and a rotor, so a unit test could only get as far as an
        # import error, which asserts the sandbox's limitations rather
        # than the code's behaviour. That path is covered at the bench.
        assert run(["shell", "--track-log", "x.csv"], config_path, factory) == 1

        assert "--track-log needs --tle and --send" in capsys.readouterr().err


class TestReceiveTheme:
    """``receive --theme`` picks the instrument window's theme."""

    def test_it_defaults_to_the_shipped_default(self):
        args = build_parser().parse_args(
            ["receive", "--tle", "x", "--downlink", "435.605e6", "--gain", "30"]
        )
        assert args.theme == DEFAULT_THEME_NAME

    def test_it_accepts_a_slug(self):
        args = build_parser().parse_args(
            [
                "receive",
                "--tle",
                "x",
                "--downlink",
                "435.605e6",
                "--gain",
                "30",
                "--theme",
                "night-ops",
            ]
        )
        assert args.theme == "night-ops"

    def test_every_shipped_slug_is_accepted_by_the_parser(self):
        """The flag takes a stem rather than a choice list on purpose.

        Restricting it to ``choices`` would reject a user's own theme
        file, which is the whole point of themes being files -- so the
        validation is a lookup at apply time, not at parse time.
        """
        for slug in discover_themes((DEFAULT_THEMES_DIR,)):
            args = build_parser().parse_args(
                [
                    "receive",
                    "--tle",
                    "x",
                    "--downlink",
                    "435.605e6",
                    "--gain",
                    "30",
                    "--theme",
                    slug,
                ]
            )
            assert args.theme == slug

    def test_an_unknown_slug_still_parses(self):
        args = build_parser().parse_args(
            [
                "receive",
                "--tle",
                "x",
                "--downlink",
                "435.605e6",
                "--gain",
                "30",
                "--theme",
                "hologram",
            ]
        )
        assert args.theme == "hologram"


# ---------------------------------------------------------------------------
# Gain policy at the command line (Chunk H, PR2c)
# ---------------------------------------------------------------------------


class _GainRecordingRotor:
    """A rotor that records what gains were pushed to it.

    Deliberately minimal: nothing here needs a serial port, and the
    question under test is whether the CLI decides to push at all.

    ``in_force`` is what :meth:`read_gains` reports, and it defaults to
    something **other** than the firmware's compiled defaults on
    purpose. A double that answered 8.0 / 0.0 / 0.5 would let a test
    pass whether the value came from the hardware or from the old
    hard-coded sentence about compiled defaults.
    """

    def __init__(self, raises: Exception | None = None, in_force: dict | None = None):
        from qsorbit.core.rotor import GainRegister

        self.pushed: list[dict] = []
        self.reads = 0
        self._raises = raises
        self._in_force = in_force or dict(
            zip(GainRegister, [7.5, 0.42, 0.11, 9.25, 0.63, 0.07], strict=True)
        )

    def push_gains(self, gains):
        if self._raises is not None:
            raise self._raises
        self.pushed.append(dict(gains))
        return dict(gains)

    def read_gains(self):
        self.reads += 1
        return dict(self._in_force)


def _station(mechanics: bool = True):
    from qsorbit.core.rotor import AzimuthWrap, RotorCapabilities
    from qsorbit.core.station import StationConfig, TrackingSettings
    from qsorbit.core.tracker import ObserverLocation

    measured = (
        {
            "azimuth_free_play_deg": 2.95,
            "azimuth_breakaway_pwm": 17.0,
            "elevation_free_play_deg": 2.55,
            "elevation_breakaway_pwm": 21.0,
        }
        if mechanics
        else {}
    )
    return StationConfig(
        observer=ObserverLocation(latitude=40.5, longitude=-83.25, altitude_m=250.0),
        serial=__import__("qsorbit.core.station", fromlist=["SerialSettings"]).SerialSettings(
            port="COM5"
        ),
        capabilities=RotorCapabilities(
            azimuth_min_deg=0.0,
            azimuth_max_deg=360.0,
            elevation_min_deg=0.0,
            elevation_max_deg=180.0,
            azimuth_wrap=AzimuthWrap.EXTRA_ROTATION,
            acceptance_window_deg=2.5,
            rs485_turnaround_s=0.15,
            **measured,
        ),
        tracking=TrackingSettings(),
    )


def _profile(**overrides):
    from qsorbit.core.tracking_profile import TrackingProfile

    fields = {
        "name": "tracking",
        "deadband_deg": 0.25,
        "interval_s": 0.5,
        "azimuth_kp": 8.0,
        "azimuth_ki": 0.5,
        "azimuth_kd": 0.5,
        "elevation_kp": 10.0,
        "elevation_ki": 0.5,
        "elevation_kd": 0.3,
    }
    fields.update(overrides)
    return TrackingProfile(**fields)


class TestReceiveShellHub:
    """The receive-path feed hub, built by its own function.

    Extracted from ``_run_shell`` so the receive path and the rotor-only
    path cannot silently disagree about what the hub carries -- which
    they did, and it is why these tests exist.
    """

    def test_it_carries_the_station_tracking_profiles(self):
        # Regression. The receive-path hub was built inline without
        # tracking_profiles, so the Rotor tab reported "declare at least
        # two profiles" on a station that declared three, on the exact
        # configuration Chunk E's acceptance runs in (receive path, rotor
        # attached). The rotor-only path was wired correctly; this one
        # had diverged.
        from qsorbit.core.tracking_profile import TrackingProfile

        session = SimpleNamespace(spectrum=None, branches=())
        loop = SimpleNamespace(target=SimpleNamespace(name="RS-44"), latest_sample=None)
        profiles = (
            TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0),
            TrackingProfile(name="tracking", deadband_deg=0.25, interval_s=0.5),
            TrackingProfile(name="fast", deadband_deg=0.3, interval_s=0.5),
        )

        hub = _receive_shell_hub(session, loop, None, profiles)

        assert hub.rotor is not None
        assert hub.rotor.profiles == profiles

    def test_a_receive_run_without_a_rotor_still_builds_a_hub(self):
        # No loop, no rotor feed, and no profiles to show -- a plain
        # receive session, not an error.
        session = SimpleNamespace(spectrum=None, branches=())

        hub = _receive_shell_hub(session, None, None, ())

        assert hub.rotor is None


class TestPushProfileGains:
    def test_a_stock_profile_writes_nothing_at_all(self, capsys):
        from qsorbit.core.tracking_profile import TrackingProfile

        rotor = _GainRecordingRotor()
        stock = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
        _push_profile_gains(rotor, stock, _station())
        assert rotor.pushed == []
        assert "writes none" in capsys.readouterr().out

    def test_a_profile_that_writes_none_reports_what_is_actually_in_force(self, capsys):
        """The line used to be a claim about hardware that nothing checked.

        Gains are RAM-only and survive a disconnect, so "writes none"
        does not mean "compiled defaults" -- it means whatever was last
        written, which on this station was an aborted push that left Ki
        live for a 543-second track while the console said stock.
        """
        from qsorbit.core.tracking_profile import TrackingProfile

        rotor = _GainRecordingRotor()
        stock = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)

        _push_profile_gains(rotor, stock, _station())

        out = capsys.readouterr().out
        assert rotor.reads == 1
        assert "azimuth_ki 0.42" in out
        assert "elevation_kd 0.07" in out
        # The old sentence asserted a fact about the controller from a
        # string literal. It must not come back.
        assert "compiled defaults" not in out

    def test_a_live_toggle_does_not_read_the_gains_back(self, capsys):
        # Six round trips is about a second, and this path runs inside
        # tick() on the ticking thread. Paying it on the stock direction
        # only would make the toggle asymmetric during the very pass its
        # acceptance depends on.
        from qsorbit.core.tracking_profile import TrackingProfile

        rotor = _GainRecordingRotor()
        stock = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)

        _profile_pusher(rotor, _station())(stock)

        assert rotor.reads == 0
        assert "writes none" in capsys.readouterr().out

    def test_a_gain_profile_pushes_all_six(self, capsys):
        from qsorbit.core.rotor import GainRegister

        rotor = _GainRecordingRotor()
        _push_profile_gains(rotor, _profile(), _station())
        assert list(rotor.pushed[0]) == list(GainRegister)

    def test_the_clamp_is_rechecked_before_the_wire(self):
        # Config already checks this, but a profile can arrive here from
        # --rotor-profile or a direct construction. A check that only
        # guards the config file has a command-line-shaped hole in it.
        from qsorbit.core.tracking_profile import GainClampError

        rotor = _GainRecordingRotor()
        with pytest.raises(GainClampError):
            _push_profile_gains(rotor, _profile(azimuth_ki=1.0), _station())
        assert rotor.pushed == []

    def test_nothing_is_pushed_to_an_unmeasured_rotor(self):
        from qsorbit.core.tracking_profile import UnmeasuredMechanicsError

        rotor = _GainRecordingRotor()
        with pytest.raises(UnmeasuredMechanicsError):
            _push_profile_gains(rotor, _profile(), _station(mechanics=False))
        assert rotor.pushed == []

    def test_a_verification_failure_propagates(self):
        # Rather than warning and tracking on gains nobody chose, which
        # would attribute every later measurement to the wrong setup.
        from qsorbit.core.rotor import GainVerificationError

        rotor = _GainRecordingRotor(raises=GainVerificationError("register 2 disagreed"))
        with pytest.raises(GainVerificationError):
            _push_profile_gains(rotor, _profile(), _station())

    def test_the_console_output_is_ascii(self, capsys):
        # Session 34's em dash rendered as a stray glyph on Windows.
        _push_profile_gains(_GainRecordingRotor(), _profile(), _station())
        capsys.readouterr().out.encode("ascii")

    def test_the_output_says_the_gains_are_ram_only(self, capsys):
        # The one thing an operator has to remember about this feature:
        # a power cycle silently reverts it.
        _push_profile_gains(_GainRecordingRotor(), _profile(), _station())
        assert "RAM only" in capsys.readouterr().out


class TestStallGuardFromConfig:
    def test_falls_back_to_the_compiled_default_when_unmeasured(self):
        from qsorbit.core.stall_guard import DEFAULT_FREE_PLAY_DEG

        assert _stall_guard(_station(mechanics=False)).free_play_deg == DEFAULT_FREE_PLAY_DEG

    def test_uses_the_measured_value_when_present(self):
        # The detector's gate and the gain clamp are the same physical
        # number, so they read from the same place.
        assert _stall_guard(_station()).free_play_deg == pytest.approx(2.95)

    def test_the_larger_axis_wins(self):
        # One figure covers both axes, and a gate sized to the tighter
        # one would call the sloppier axis stalled every time it took up
        # its own slack.
        from dataclasses import replace

        config = _station()
        loose = replace(
            config,
            capabilities=replace(config.capabilities, elevation_free_play_deg=4.0),
        )
        assert _stall_guard(loose).free_play_deg == pytest.approx(4.0)


class TestDescribeMechanics:
    def test_says_so_when_nothing_is_measured(self):
        lines = _describe_mechanics(_station(mechanics=False).capabilities)
        assert len(lines) == 1
        assert "not measured" in lines[0]

    def test_reports_the_headroom_not_just_the_measurement(self):
        # "Free play 2.95 deg" is a fact; "max safe Ki 0.97" is the fact
        # an operator can act on. 0.97 and not 0.98: the figure is
        # floored, because a status line advertising a Ki the clamp
        # would then refuse is worse than printing nothing.
        lines = _describe_mechanics(_station().capabilities)
        assert any("0.97" in line for line in lines)
        assert not any("0.98" in line for line in lines)

    def test_reports_both_axes(self):
        lines = _describe_mechanics(_station().capabilities)
        assert len(lines) == 2
        assert "AZ" in lines[0]
        assert "EL" in lines[1]

    def test_the_lines_are_ascii(self):
        for line in _describe_mechanics(_station().capabilities):
            line.encode("ascii")


class TestProfilePusher:
    """The callback the tracking loop runs when a queued switch lands.

    It exists so the gain write happens on whichever thread ticks the
    loop -- the thread that already owns the serial port. That is what
    lets the switch be queued instead of locked.
    """

    def _profile(self, **overrides):
        from qsorbit.core.tracking_profile import TrackingProfile

        fields = {
            "name": "tracking",
            "deadband_deg": 0.25,
            "interval_s": 0.5,
            "azimuth_kp": 8.0,
            "azimuth_ki": 0.5,
            "azimuth_kd": 0.5,
            "elevation_kp": 10.0,
            "elevation_ki": 0.5,
            "elevation_kd": 0.3,
        }
        fields.update(overrides)
        return TrackingProfile(**fields)

    def test_pushing_reaches_the_rotor(self, capsys):
        rotor = _GainRecordingRotor()
        _profile_pusher(rotor, _station())(self._profile())
        assert len(rotor.pushed) == 1

    def test_a_stock_profile_writes_nothing(self, capsys):
        from qsorbit.core.tracking_profile import TrackingProfile

        rotor = _GainRecordingRotor()
        stock = TrackingProfile(name="stock", deadband_deg=2.5, interval_s=1.0)
        _profile_pusher(rotor, _station())(stock)
        assert rotor.pushed == []

    def test_a_verification_failure_is_not_swallowed(self):
        # Phil's call: a failed mid-pass switch stops the run. The
        # callback must therefore let the error out, so it propagates
        # through tick() to whatever is driving the loop.
        from qsorbit.core.rotor import GainVerificationError

        rotor = _GainRecordingRotor(raises=GainVerificationError("register 2 disagreed"))
        with pytest.raises(GainVerificationError):
            _profile_pusher(rotor, _station())(self._profile())

    def test_the_clamp_still_applies_to_a_switched_in_profile(self, capsys):
        # A profile can reach the loop from --rotor-profile or a toggle
        # press, not only from config. This is the last gate before the
        # wire in both cases.
        from qsorbit.core.tracking_profile import GainClampError

        rotor = _GainRecordingRotor()
        with pytest.raises(GainClampError):
            _profile_pusher(rotor, _station())(self._profile(azimuth_ki=1.0))
        assert rotor.pushed == []


# ---------------------------------------------------------------------------
# Opening the right dongle
# ---------------------------------------------------------------------------


class FakeBus:
    """Enough of a loaded binding to enumerate a bus, and nothing more.

    ``_open_sdr`` never opens what it builds -- the caller's ``with``
    does -- so this only has to answer the three handle-free calls that
    serial resolution makes.
    """

    def __init__(self, *serials: str) -> None:
        self.serials = serials

    def device_count(self) -> int:
        return len(self.serials)

    def device_name(self, index: int) -> str:
        return "Generic RTL2832U OEM"

    def usb_strings(self, index: int) -> tuple[str, str, str]:
        return ("RTLSDRBlog", "Blog V4", self.serials[index])


@pytest.fixture
def on_the_bus(monkeypatch):
    """Put a fake set of dongles on the bus that _open_sdr will walk."""

    def attach(*serials: str) -> FakeBus:
        bus = FakeBus(*serials)
        monkeypatch.setattr(
            "qsorbit.__main__.LibRtlSdr.load", staticmethod(lambda driver_dir=None: bus)
        )
        return bus

    return attach


def config_with(tmp_path, extra):
    path = tmp_path / "branched.toml"
    path.write_text(textwrap.dedent(CONFIG) + textwrap.dedent(extra), encoding="utf-8")
    return load_station_config(path)


TWO_BRANCHES = """
[[sdr.branch]]
serial = "RIGHT"
label = "A - Arrow V"
ppm = 4

[[sdr.branch]]
serial = "LEFT"
label = "B - Arrow H"
"""


class TestOpenSdr:
    def test_no_branches_addresses_the_configured_index(self, tmp_path):
        # The path every config written before branches existed takes.
        config = config_with(tmp_path, "\n[sdr]\ndevice_index = 1\n")

        assert _open_sdr(config).index == 1

    def test_no_branches_does_not_touch_the_bus(self, tmp_path, monkeypatch):
        # A single-device station must not need an enumeration to work,
        # or a broken driver load becomes a new failure mode for a
        # station that never asked for any of this.
        def explode(driver_dir=None):
            raise AssertionError("the library was loaded for an index-addressed open")

        monkeypatch.setattr("qsorbit.__main__.LibRtlSdr.load", staticmethod(explode))

        assert _open_sdr(config_with(tmp_path, "\n[sdr]\n")).index == 0

    def test_a_branch_is_addressed_by_serial(self, tmp_path, on_the_bus):
        # RIGHT is second on the bus on purpose. Asking for the device
        # that enumerates first would pass whether the serial was
        # honoured or quietly ignored.
        on_the_bus("LEFT", "RIGHT")
        config = config_with(tmp_path, '\n[[sdr.branch]]\nserial = "RIGHT"\nlabel = "B"\n')

        assert _open_sdr(config).index == 1

    def test_it_resolves_against_the_bus_as_it_is_now(self, tmp_path, on_the_bus):
        # The same config against a differently-populated bus resolves
        # to a different index, which is the entire point: indices move
        # and serials do not.
        on_the_bus("RIGHT", "LEFT")
        config = config_with(tmp_path, '\n[[sdr.branch]]\nserial = "RIGHT"\nlabel = "B"\n')

        assert _open_sdr(config).index == 0

    def test_a_serial_that_is_not_there_is_reported(self, tmp_path, on_the_bus):
        on_the_bus("LEFT")
        config = config_with(tmp_path, '\n[[sdr.branch]]\nserial = "RIGHT"\nlabel = "B"\n')

        with pytest.raises(DeviceNotFoundError, match="RIGHT"):
            _open_sdr(config)

    def test_two_branches_open_the_first_declared(self, tmp_path, on_the_bus):
        on_the_bus("LEFT", "RIGHT")
        config = config_with(tmp_path, TWO_BRANCHES)

        assert _open_sdr(config).index == 1

    def test_an_explicit_branch_overrides_the_default_one(self, tmp_path, on_the_bus):
        # What the receive path does: it opens every branch by name
        # rather than taking whichever the single-device commands would.
        on_the_bus("LEFT", "RIGHT")
        config = config_with(tmp_path, TWO_BRANCHES)

        opened = [_open_sdr(config, branch).index for branch in config.sdr.branches]

        assert opened == [1, 0]

    def test_it_prints_nothing(self, tmp_path, on_the_bus, capsys):
        # A factory has no business writing to the console, and the
        # receive path calls this once per branch, where a note about
        # opening one of two would be false.
        on_the_bus("LEFT", "RIGHT")

        _open_sdr(config_with(tmp_path, TWO_BRANCHES))

        assert capsys.readouterr().out == ""


class TestSingleDeviceNote:
    """What a one-device command says on a station cabled for two."""

    def test_it_names_the_branch_it_took(self, tmp_path, capsys):
        # Silence here would look exactly like both branches being used.
        _note_single_device(config_with(tmp_path, TWO_BRANCHES))

        out = capsys.readouterr().out
        assert "A - Arrow V" in out
        assert "RIGHT" in out

    def test_one_branch_says_nothing(self, tmp_path, capsys):
        _note_single_device(
            config_with(tmp_path, '\n[[sdr.branch]]\nserial = "LEFT"\nlabel = "B"\n')
        )

        assert capsys.readouterr().out == ""

    def test_no_branches_says_nothing(self, tmp_path, capsys):
        _note_single_device(config_with(tmp_path, "\n[sdr]\ndevice_index = 1\n"))

        assert capsys.readouterr().out == ""


class TestListeningIndex:
    """Which branch --listen selects."""

    def test_no_flag_takes_the_first_declared(self, tmp_path):
        config = config_with(tmp_path, TWO_BRANCHES)

        assert _listening_index(None, config.sdr.branches) == 0

    def test_a_label_selects_its_branch(self, tmp_path):
        config = config_with(tmp_path, TWO_BRANCHES)

        assert _listening_index("B - Arrow H", config.sdr.branches) == 1

    def test_an_unknown_label_lists_the_real_ones(self, tmp_path):
        config = config_with(tmp_path, TWO_BRANCHES)

        with pytest.raises(ValueError, match="A - Arrow V"):
            _listening_index("Arrow V", config.sdr.branches)

    def test_matching_is_exact(self, tmp_path):
        # Labels have spaces in them and come from a file nobody has
        # open at the time, so a near miss must not silently succeed.
        config = config_with(tmp_path, TWO_BRANCHES)

        with pytest.raises(ValueError):
            _listening_index("b - arrow h", config.sdr.branches)

    def test_a_station_with_no_branches_has_one_unnamed_one(self, tmp_path):
        config = config_with(tmp_path, "\n[sdr]\n")

        assert _listening_index(None, _declared_branches(config)) == 0
        assert _listening_index(UNNAMED_BRANCH_LABEL, _declared_branches(config)) == 0


class TestSdrPpm:
    def test_no_branches_uses_the_station_correction(self, tmp_path):
        assert _sdr_ppm(config_with(tmp_path, "\n[sdr]\nppm = -3\n")) == -3

    def test_the_first_branch_overrides_it(self, tmp_path):
        assert _sdr_ppm(config_with(tmp_path, "\n[sdr]\nppm = -3\n" + TWO_BRANCHES)) == 4

    def test_a_branch_without_one_falls_back(self, tmp_path):
        config = config_with(
            tmp_path, '\n[sdr]\nppm = -3\n\n[[sdr.branch]]\nserial = "L"\nlabel = "B"\n'
        )

        assert _sdr_ppm(config) == -3


# ---------------------------------------------------------------------------
# Opening a device per branch
# ---------------------------------------------------------------------------


def branch_sdr_factory():
    """A factory that hands out a distinct device per branch and records both."""

    def build(config, branch=None):
        device = FakeSdr(
            index=len(build.opened),
            serial="none" if branch is None else branch.serial,
        )
        build.opened.append(branch)
        build.devices.append(device)
        return device

    build.opened = []
    build.devices = []
    return build


def receive_args(tle_path, *extra):
    return build_parser().parse_args(
        ["receive", "--tle", str(tle_path), "--downlink", "145.95", "--gain", "32.8", *extra]
    )


def capturing_runner():
    """Stand in for _run_receive, keeping what it was handed."""

    def run(args, config, satellite, radios, listening, *, loop=None):
        run.radios = radios
        run.listening = listening
        return 0

    run.radios = None
    run.listening = None
    return run


class TestReceiveOpensEveryBranch:
    def test_a_station_with_no_branches_opens_one_unnamed_device(self, tmp_path, tle_path):
        # The path every config written before branches existed takes,
        # and it goes through the same code as a two-branch station
        # rather than around it.
        config = config_with(tmp_path, "\n[sdr]\n")
        factory = branch_sdr_factory()
        runner = capturing_runner()

        code = _command_receive(receive_args(tle_path), config, None, factory, runner=runner)

        assert code == 0
        assert factory.opened == [None]
        assert [radio.label for radio in runner.radios] == [UNNAMED_BRANCH_LABEL]

    def test_two_branches_open_two_devices_named_by_serial(self, tmp_path, tle_path):
        config = config_with(tmp_path, TWO_BRANCHES)
        factory = branch_sdr_factory()
        runner = capturing_runner()

        _command_receive(receive_args(tle_path), config, None, factory, runner=runner)

        assert [branch.serial for branch in factory.opened] == ["RIGHT", "LEFT"]
        assert [radio.label for radio in runner.radios] == ["A - Arrow V", "B - Arrow H"]

    def test_every_device_is_configured(self, tmp_path, tle_path):
        config = config_with(tmp_path, TWO_BRANCHES)
        factory = branch_sdr_factory()

        _command_receive(receive_args(tle_path), config, None, factory, runner=capturing_runner())

        assert [len(device.configured) for device in factory.devices] == [1, 1]

    def test_each_branch_is_tuned_with_its_own_ppm(self, tmp_path, tle_path):
        # A crystal correction belongs to one oscillator. The first
        # branch declares 4; the second declares none and falls back to
        # the station's -3.
        config = config_with(tmp_path, "\n[sdr]\nppm = -3\n" + TWO_BRANCHES)
        factory = branch_sdr_factory()

        _command_receive(receive_args(tle_path), config, None, factory, runner=capturing_runner())

        assert [device.configured[0].ppm for device in factory.devices] == [4, -3]

    def test_every_branch_gets_its_own_squelch_and_tracker(self, tmp_path, tle_path):
        # Both are stateful or tuner-specific, and sharing either would
        # mix two radios into one decision.
        config = config_with(tmp_path, TWO_BRANCHES)
        runner = capturing_runner()

        _command_receive(receive_args(tle_path), config, None, branch_sdr_factory(), runner=runner)

        first, second = runner.radios
        assert first.squelch is not second.squelch
        assert first.doppler is not second.doppler

    def test_it_closes_every_device_it_opened(self, tmp_path, tle_path):
        config = config_with(tmp_path, TWO_BRANCHES)
        factory = branch_sdr_factory()

        _command_receive(receive_args(tle_path), config, None, factory, runner=capturing_runner())

        assert [device.is_open for device in factory.devices] == [False, False]

    def test_the_first_branch_is_heard_by_default(self, tmp_path, tle_path):
        config = config_with(tmp_path, TWO_BRANCHES)
        runner = capturing_runner()

        _command_receive(receive_args(tle_path), config, None, branch_sdr_factory(), runner=runner)

        assert runner.listening == 0

    def test_listen_selects_a_branch_by_label(self, tmp_path, tle_path):
        config = config_with(tmp_path, TWO_BRANCHES)
        runner = capturing_runner()

        _command_receive(
            receive_args(tle_path, "--listen", "B - Arrow H"),
            config,
            None,
            branch_sdr_factory(),
            runner=runner,
        )

        assert runner.listening == 1

    def test_an_unknown_listen_label_fails_before_any_device_opens(
        self, tmp_path, tle_path, capsys
    ):
        # Checked before the hardware, so a mistyped label costs nothing
        # and the message can list the real ones.
        config = config_with(tmp_path, TWO_BRANCHES)
        factory = branch_sdr_factory()

        code = _command_receive(
            receive_args(tle_path, "--listen", "Arrow H"),
            config,
            None,
            factory,
            runner=capturing_runner(),
        )

        assert code == 2
        assert factory.opened == []
        assert "A - Arrow V" in capsys.readouterr().err

    def test_it_says_which_branch_is_being_heard(self, tmp_path, tle_path, capsys):
        config = config_with(tmp_path, TWO_BRANCHES)

        _command_receive(
            receive_args(tle_path, "--listen", "B - Arrow H"),
            config,
            None,
            branch_sdr_factory(),
            runner=capturing_runner(),
        )

        out = capsys.readouterr().out
        assert "B - Arrow H <- audio" in out
        assert "A - Arrow V\n" in out


def branch_capturing_runner(*, window):
    """Stand in for _run_receive, building the branches where it does.

    Inside the runner rather than after it, because that is the only
    place the devices are still open -- _command_receive holds them in
    an ExitStack, and an IqStream over a closed device refuses.
    """

    def run(args, config, satellite, radios, listening, *, loop=None):
        run.branches = _build_branches(args, radios, listening, window=window)
        run.listening = listening
        return 0

    run.branches = None
    run.listening = None
    return run


class TestBranchWiring:
    """What _build_branches hands each branch, which no bench run can see.

    Both branches are tuned to the same downlink, so their waterfalls are
    pixel-for-pixel identical -- looking at the window cannot tell you
    which one is feeding it. That makes this the wrong claim to check at
    the bench and the right one to check here.
    """

    def branches(self, tmp_path, tle_path, *extra, window=True):
        runner = branch_capturing_runner(window=window)
        _command_receive(
            receive_args(tle_path, *extra),
            config_with(tmp_path, TWO_BRANCHES),
            None,
            branch_sdr_factory(),
            runner=runner,
        )
        return runner.branches, runner.listening

    def test_the_waterfall_goes_to_the_branch_being_heard(self, tmp_path, tle_path):
        branches, listening = self.branches(
            tmp_path, tle_path, "--listen", "B - Arrow H", "--window"
        )

        assert listening == 1
        assert branches[1].spectrum is not None
        assert branches[0].spectrum is None

    def test_the_first_branch_gets_it_by_default(self, tmp_path, tle_path):
        branches, _ = self.branches(tmp_path, tle_path, "--window")

        assert branches[0].spectrum is not None
        assert branches[1].spectrum is None

    def test_no_window_means_no_spectrum_on_any_branch(self, tmp_path, tle_path):
        # A headless run must not compute frames nobody will see, on
        # either branch.
        branches, _ = self.branches(tmp_path, tle_path, window=False)

        assert [branch.spectrum for branch in branches] == [None, None]

    def test_every_branch_is_labelled_from_config(self, tmp_path, tle_path):
        branches, _ = self.branches(tmp_path, tle_path, window=False)

        assert [branch.label for branch in branches] == ["A - Arrow V", "B - Arrow H"]


class TestWindowTitle:
    """The title is the only thing on screen that names the branch."""

    def test_one_branch_is_not_labelled(self, tmp_path, tle_path):
        # A label would be noise on a station with one dongle.
        session = SimpleNamespace(branches=(object(),), listening=None, selector=None)

        title = _window_title(SimpleNamespace(name="AO-91"), session)

        assert title == "QSOrbit - receiving AO-91"

    def test_two_branches_name_the_one_being_heard(self, tmp_path, tle_path):
        # Without this, --listen has no visible effect whatever: both
        # branches are tuned to the same downlink, so their waterfalls
        # are identical and nothing on screen distinguishes them.
        session = SimpleNamespace(
            branches=(object(), object()),
            listening=SimpleNamespace(label="B - Arrow H"),
            selector=None,
        )

        title = _window_title(SimpleNamespace(name="AO-91"), session)

        assert title == "QSOrbit - receiving AO-91 on B - Arrow H"


class TestQuietingLogWiring:
    """The flag, and the one log every branch shares."""

    def branches_with(self, tmp_path, tle_path, *extra, log=None):
        def run(args, config, satellite, radios, listening, *, loop=None):
            run.branches = _build_branches(args, radios, listening, window=False, log=log)
            run.listening = listening
            return 0

        run.branches = None
        _command_receive(
            receive_args(tle_path, *extra),
            config_with(tmp_path, TWO_BRANCHES),
            None,
            branch_sdr_factory(),
            runner=run,
        )
        return run.branches

    def test_the_flag_defaults_to_off(self, tle_path):
        assert receive_args(tle_path).quieting_log is None

    def test_the_flag_takes_a_path(self, tle_path, tmp_path):
        target = tmp_path / "q.csv"

        assert receive_args(tle_path, "--quieting-log", str(target)).quieting_log == str(target)

    def test_the_shell_takes_it_too(self, tmp_path):
        # Both commands run the same receive path, so a flag on one and
        # not the other would be a difference with no reason behind it.
        args = build_parser().parse_args(
            [
                "shell",
                "--tle",
                "x",
                "--downlink",
                "145.95",
                "--gain",
                "32.8",
                "--quieting-log",
                str(tmp_path / "q.csv"),
            ]
        )

        assert args.quieting_log == str(tmp_path / "q.csv")

    def test_every_branch_shares_one_log(self, tmp_path, tle_path):
        # One log and not one per branch: the whole point is a
        # difference between two series, and two files would be two
        # time origins.
        log = QuietingLog(tmp_path / "q.csv")
        log.open()

        branches = self.branches_with(tmp_path, tle_path, log=log)

        assert [branch._log for branch in branches] == [log, log]
        log.close()

    def test_no_flag_means_no_log_on_any_branch(self, tmp_path, tle_path):
        branches = self.branches_with(tmp_path, tle_path)

        assert [branch._log for branch in branches] == [None, None]

    def test_open_quieting_log_returns_none_without_the_flag(self, tle_path):
        assert _open_quieting_log(receive_args(tle_path)) is None

    def test_open_quieting_log_creates_the_file_before_anything_streams(self, tmp_path, tle_path):
        # Opened early on purpose: its time origin has to be fixed
        # before a block can arrive, or the two branches would not
        # share one.
        target = tmp_path / "q.csv"

        log = _open_quieting_log(receive_args(tle_path, "--quieting-log", str(target)))

        assert target.exists()
        log.close()

    def test_the_report_names_the_file_and_says_nothing_without_one(self, tmp_path, capsys):
        _print_quieting_log(None)
        assert capsys.readouterr().out == ""

        log = QuietingLog(tmp_path / "q.csv")
        log.open()
        _print_quieting_log(log)

        assert "q.csv" in capsys.readouterr().out


class TestRecordIqWiring:
    """``--record-iq`` gives every branch a recorder writing into DIR."""

    def branches_with(self, tmp_path, tle_path, *extra, record_dir=None):
        def run(args, config, satellite, radios, listening, *, loop=None):
            run.branches = _build_branches(
                args, radios, listening, window=False, record_dir=record_dir
            )
            return 0

        run.branches = None
        _command_receive(
            receive_args(tle_path, *extra),
            config_with(tmp_path, TWO_BRANCHES),
            None,
            branch_sdr_factory(),
            runner=run,
        )
        return run.branches

    def test_the_flag_defaults_to_off(self, tle_path):
        assert receive_args(tle_path).record_iq is None

    def test_the_shell_takes_it_too(self, tmp_path):
        # Both commands run the same receive path, so a flag on one and not
        # the other would be a difference with no reason behind it.
        args = build_parser().parse_args(
            [
                "shell",
                "--tle",
                "x",
                "--downlink",
                "145.95",
                "--gain",
                "32.8",
                "--record-iq",
                str(tmp_path / "iq"),
            ]
        )

        assert args.record_iq == str(tmp_path / "iq")

    def test_no_record_dir_means_no_recorder_on_any_branch(self, tmp_path, tle_path):
        branches = self.branches_with(tmp_path, tle_path)

        assert [branch.recorder for branch in branches] == [None, None]

    def test_a_record_dir_gives_every_branch_a_recorder(self, tmp_path, tle_path):
        branches = self.branches_with(tmp_path, tle_path, record_dir=tmp_path / "iq")

        assert all(branch.recorder is not None for branch in branches)

    def test_each_recorder_is_named_for_its_antenna_filename_safe(self, tmp_path, tle_path):
        branches = self.branches_with(tmp_path, tle_path, record_dir=tmp_path / "iq")

        assert sorted(b.recorder.iq_path.name for b in branches) == [
            "A-Arrow-V.iq",
            "B-Arrow-H.iq",
        ]
        # The true label is preserved on the recorder for the report and
        # the sidecar, even though the filename is sanitised.
        assert sorted(b.recorder.label for b in branches) == ["A - Arrow V", "B - Arrow H"]

    def test_the_shell_refuses_record_iq_without_a_downlink(self, capsys):
        # --record-iq records a radio, and a rotor-only shell has none.
        args = SimpleNamespace(
            track_log=None, at=None, record_iq="iq", downlink=None, tle="x", send=True
        )

        code = _command_shell(args, MagicMock(), MagicMock(), MagicMock())

        assert code == 1
        assert "--record-iq needs --downlink" in capsys.readouterr().err

    def test_the_report_says_nothing_without_a_dir_and_names_files_with_one(self, tmp_path, capsys):
        _print_recordings(SimpleNamespace(branches=()), None)
        assert capsys.readouterr().out == ""

        recorder = SimpleNamespace(
            label="A - Arrow V",
            bytes_written=2048,
            iq_path=tmp_path / "A-Arrow-V.iq",
            sidecar_path=tmp_path / "A-Arrow-V.json",
        )
        session = SimpleNamespace(branches=(SimpleNamespace(recorder=recorder),))

        _print_recordings(session, tmp_path)

        out = capsys.readouterr().out
        assert "A - Arrow V" in out
        assert "A-Arrow-V.iq" in out
        assert "2,048 bytes" in out


class TestReplayWiring:
    """``--replay`` maps captured files to branches at the SdrFactory seam."""

    def test_a_bare_file_maps_the_unnamed_branch(self):
        assert _parse_replay_map("cap.iq") == {None: "cap.iq"}

    def test_labelled_entries_map_by_label(self):
        got = _parse_replay_map("A - Arrow V=a.iq, B - Arrow H=b.iq")

        assert got == {"A - Arrow V": "a.iq", "B - Arrow H": "b.iq"}

    def test_mixing_a_bare_file_and_labels_is_refused(self):
        with pytest.raises(ValueError, match="not both"):
            _parse_replay_map("a.iq, B - Arrow H=b.iq")

    def test_a_half_written_entry_is_refused(self):
        with pytest.raises(ValueError, match="LABEL=FILE"):
            _parse_replay_map("A - Arrow V=")

    def test_an_empty_spec_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            _parse_replay_map("   ")

    def test_the_factory_returns_a_replay_for_a_matching_branch(self):
        factory = _replay_sdr_factory({"A - Arrow V": "a.iq"})

        device = factory(object(), SimpleNamespace(label="A - Arrow V"))

        assert isinstance(device, ReplaySdr)
        assert device.iq_path.name == "a.iq"

    def test_the_factory_maps_the_unnamed_branch_from_a_bare_file(self):
        factory = _replay_sdr_factory({None: "cap.iq"})

        device = factory(object(), None)

        assert isinstance(device, ReplaySdr)
        assert device.iq_path.name == "cap.iq"

    def test_the_factory_refuses_an_unmapped_branch(self):
        # A typo'd label must not quietly fall through to opening a radio.
        factory = _replay_sdr_factory({"A - Arrow V": "a.iq"})

        with pytest.raises(SdrError, match="no file for branch"):
            factory(object(), SimpleNamespace(label="B - Arrow H"))

    def test_the_flag_defaults_to_off(self, tle_path):
        assert receive_args(tle_path).replay is None

    def test_the_shell_takes_it_too(self):
        # Both commands run the same receive path, so the flag lives on both.
        args = build_parser().parse_args(
            ["shell", "--tle", "x", "--downlink", "145.95", "--gain", "32.8", "--replay", "cap.iq"]
        )

        assert args.replay == "cap.iq"

    def test_a_live_run_has_no_replay_clock(self):
        # _replay_clock_for returns a clock only when the listening branch is
        # a replay; an ordinary radio run keeps IqStream's own wall clock.
        radios = [SimpleNamespace(sdr=object())]

        assert _replay_clock_for(radios, 0) is None


class TestCombinerFlags:
    def test_combining_is_off_by_default(self, tle_path):
        # Not timidity: "combined beats either branch alone" is a
        # comparison, and the control for it is the same command without
        # the flag.
        assert receive_args(tle_path).combine is False
        assert _build_selector(receive_args(tle_path)) is None

    def test_combine_builds_a_selector_at_the_measured_margin(self, tle_path):
        selector = _build_selector(receive_args(tle_path, "--combine"))

        assert selector is not None
        assert selector.margin_db == DEFAULT_MARGIN_DB

    def test_the_margin_is_overridable(self, tle_path):
        # So the acceptance pass can sweep it without a rebuild.
        selector = _build_selector(receive_args(tle_path, "--combine", "--combine-margin", "1.5"))

        assert selector.margin_db == 1.5

    def test_a_zero_margin_is_accepted(self, tle_path):
        # The deliberate no-hysteresis control.
        assert (
            _build_selector(receive_args(tle_path, "--combine", "--combine-margin", "0")).margin_db
            == 0.0
        )

    def test_a_negative_margin_is_refused(self, tle_path):
        with pytest.raises(ValueError, match="must not be negative"):
            _build_selector(receive_args(tle_path, "--combine", "--combine-margin", "-1"))

    def test_the_margin_is_ignored_without_combine(self, tle_path):
        # No selector at all, so nothing to carry the margin -- which is
        # the honest outcome rather than a selector nobody consults.
        assert _build_selector(receive_args(tle_path, "--combine-margin", "1.5")) is None

    def test_the_shell_takes_the_flags_too(self):
        args = build_parser().parse_args(
            ["shell", "--tle", "x", "--downlink", "145.95", "--gain", "32.8", "--combine"]
        )

        assert args.combine is True
        assert args.combine_margin == DEFAULT_MARGIN_DB


class TestWindowTitleWithACombiner:
    """A title is written once; the ear is not."""

    class FakeBranch:
        def __init__(self, label):
            self.label = label

    def a_session(self, *labels, selector=None, listening=0):
        branches = tuple(self.FakeBranch(label) for label in labels)
        return SimpleNamespace(
            branches=branches,
            listening=branches[listening],
            selector=selector,
        )

    def test_a_fixed_branch_is_still_named(self):
        # --listen has no other visible effect: both branches tune to
        # the same downlink, so their waterfalls are identical.
        session = self.a_session("A - Left Arrow", "B - Right Arrow", listening=1)

        assert _window_title(SimpleNamespace(name="AO-91"), session) == (
            "QSOrbit - receiving AO-91 on B - Right Arrow"
        )

    def test_a_combining_run_names_no_branch(self):
        # The defect: the title was evaluated once, the combiner moved
        # the ear inside a second, and the window went on asserting the
        # starting branch for the rest of the run. A confidently wrong
        # antenna is worse than the silence it replaced.
        session = self.a_session("A", "B", selector=object())

        assert _window_title(SimpleNamespace(name="AO-91"), session) == (
            "QSOrbit - receiving AO-91 (combining)"
        )

    def test_one_branch_is_never_named_combiner_or_not(self):
        for selector in (None, object()):
            session = self.a_session("A", selector=selector)

            assert _window_title(SimpleNamespace(name="AO-91"), session) == (
                "QSOrbit - receiving AO-91"
            )


class TestSimulatedPass:
    """`shell --at` simulates a pass for the rotor without waiting for one."""

    def test_the_parser_accepts_at_on_the_shell(self):
        args = build_parser().parse_args(
            ["shell", "--tle", "x", "--send", "--at", "2026-09-10T08:44:00+00:00"]
        )
        assert args.at == "2026-09-10T08:44:00+00:00"

    def test_at_defaults_to_none(self):
        args = build_parser().parse_args(["shell", "--tle", "x", "--send"])
        assert args.at is None

    def test_at_refuses_a_live_downlink(self, capsys):
        # Geometry is simulated by the clock; signal is simulated by
        # replay. A live radio at a simulated time would compute Doppler
        # for a time the signal is not at, so the combination is refused.
        args = SimpleNamespace(
            track_log=None, at="2026-09-10T08:44:00Z", downlink=145.9, tle="x", send=True
        )
        code = _command_shell(args, MagicMock(), MagicMock(), MagicMock())
        assert code == 1
        assert "rotor-only" in capsys.readouterr().err

    def test_at_needs_tle_and_send(self, capsys):
        args = SimpleNamespace(
            track_log=None, at="2026-09-10T08:44:00Z", downlink=None, tle="x", send=False
        )
        code = _command_shell(args, MagicMock(), MagicMock(), MagicMock())
        assert code == 1
        assert "--at needs --tle and --send" in capsys.readouterr().err

    def test_offset_clock_reads_the_simulated_time_and_advances(self):
        from datetime import UTC, datetime, timedelta

        future = datetime.now(UTC) + timedelta(hours=5)
        clock = _offset_clock(future)
        assert abs((clock() - future).total_seconds()) < 2.0

        first = clock()
        time.sleep(0.02)
        assert clock() > first

    def test_offset_clock_handles_a_past_time(self):
        from datetime import UTC, datetime, timedelta

        past = datetime.now(UTC) - timedelta(days=1)
        clock = _offset_clock(past)
        assert abs((clock() - past).total_seconds()) < 2.0

    # --- below-horizon note (the Qt-gated seam, extracted so it can be
    # unit-tested; the version this replaces read state.elevation, which
    # does not exist, and crashed only on a real --at run) ---

    @staticmethod
    def _fake_satellite(elevation_deg: float, *, name: str = "AO-73"):
        """A satellite stub whose topocentric elevation is fixed.

        Exercises _below_horizon_note without skyfield: it only touches
        ``.name`` and ``state.sky_position.elevation``, the exact surface
        the crash was on.
        """

        state = SimpleNamespace(sky_position=SimpleNamespace(elevation=elevation_deg))
        return SimpleNamespace(
            name=name,
            topocentric_state=lambda observer, when: state,
        )

    def test_below_horizon_note_is_none_when_the_target_is_up(self):
        from datetime import UTC, datetime

        sat = self._fake_satellite(12.3)
        note = _below_horizon_note(
            sat, MagicMock(), datetime(2026, 9, 10, 8, 44, tzinfo=UTC), "2026-09-10T08:44:00Z"
        )
        assert note is None

    def test_below_horizon_note_is_none_exactly_at_the_horizon(self):
        from datetime import UTC, datetime

        sat = self._fake_satellite(0.0)
        note = _below_horizon_note(
            sat, MagicMock(), datetime(2026, 9, 10, 8, 44, tzinfo=UTC), "2026-09-10T08:44:00Z"
        )
        assert note is None

    def test_below_horizon_note_warns_when_the_target_is_down(self):
        from datetime import UTC, datetime

        sat = self._fake_satellite(-7.5, name="AO-73")
        note = _below_horizon_note(
            sat, MagicMock(), datetime(2026, 9, 10, 8, 44, tzinfo=UTC), "2026-09-10T08:44:00Z"
        )
        assert note is not None
        assert "AO-73" in note
        assert "below the horizon" in note
        assert "-7.5 deg" in note
        # The suggested command echoes the raw --at value verbatim.
        assert "qsorbit plan --at 2026-09-10T08:44:00Z" in note
