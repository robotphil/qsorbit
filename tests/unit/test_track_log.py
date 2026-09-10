"""Tests for the CSV pass log.

No rotor, no thread, no clock -- this file is about the file format and
nothing else. How the log is *driven* (when a sample is taken, when one
defers to a tick) belongs to the thread that owns the port and is tested
in test_tracking_thread.py.
"""

from __future__ import annotations

import pytest

from qsorbit.core.rotor import Position
from qsorbit.core.track_log import CSV_COLUMNS, TrackLog

TARGET = Position(azimuth=123.456, elevation=45.678)
POSITION = Position(azimuth=122.0, elevation=44.5)


class TestFormat:
    def test_it_writes_the_header_on_open(self, tmp_path):
        path = tmp_path / "t.csv"
        with TrackLog(path):
            pass

        assert path.read_text(encoding="utf-8").splitlines() == [",".join(CSV_COLUMNS)]

    def test_a_row_carries_both_axes_with_target_beside_position(self, tmp_path):
        # Target and position adjacent per axis, so reading one axis out
        # is a column pick -- the Session 32 metrics are per-axis and
        # they are validated code worth feeding rather than rewriting.
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(1.5, TARGET, POSITION, "commanded")

        assert path.read_text(encoding="utf-8").splitlines()[1] == (
            "1.500,123.46,122.00,45.68,44.50,commanded"
        )

    def test_an_observed_row_names_no_outcome(self, tmp_path):
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(0.2, TARGET, POSITION)

        assert path.read_text(encoding="utf-8").splitlines()[1].endswith(",")

    def test_rows_are_readable_before_the_log_is_closed(self, tmp_path):
        # A pass that ends in a stall or somebody hitting the power
        # switch is exactly the run whose data is most worth having, and
        # a buffered final write is what loses it.
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(0.0, TARGET, POSITION, "commanded")

            assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_opening_replaces_an_existing_file(self, tmp_path):
        # Two runs appended into one file restart the time column in the
        # middle with nothing saying so, which is worse than no log.
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(9.0, TARGET, POSITION, "commanded")
        with TrackLog(path) as log:
            log.record(0.0, TARGET, POSITION, "commanded")

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert lines[1].startswith("0.000,")

    def test_it_does_not_write_blank_lines_between_rows(self, tmp_path):
        # csv writes \r\n itself; without newline="" the text layer
        # translates that again on Windows and every row ends \r\r\n.
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(0.0, TARGET, POSITION, "commanded")
            log.record(0.2, TARGET, POSITION)

        assert b"\r\r" not in path.read_bytes()


class TestReporting:
    def test_it_counts_the_rows_it_wrote(self, tmp_path):
        with TrackLog(tmp_path / "t.csv") as log:
            log.record(0.0, TARGET, POSITION, "commanded")
            log.record(0.2, TARGET, POSITION)

            assert log.rows == 2

    def test_the_description_names_the_file_and_the_count(self, tmp_path):
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(0.0, TARGET, POSITION, "commanded")

            assert "1 sample(s)" in log.describe()
            assert str(path) in log.describe()

    def test_recording_before_opening_is_refused(self, tmp_path):
        # Silently discarding a pass's data would be the worst outcome
        # available here.
        with pytest.raises(RuntimeError, match="not been opened"):
            TrackLog(tmp_path / "t.csv").record(0.0, TARGET, POSITION)

    def test_closing_twice_is_harmless(self, tmp_path):
        log = TrackLog(tmp_path / "t.csv")
        log.open()
        log.close()
        log.close()


class TestPreambleComment:
    """A run that must not be mistaken for a normal one marks itself."""

    def test_a_preamble_comment_precedes_the_header(self, tmp_path):
        path = tmp_path / "t.csv"
        with TrackLog(path, preamble_comment="SIMULATED pass: not live") as log:
            log.record(0.0, TARGET, POSITION, "commanded")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# SIMULATED pass: not live"
        assert lines[1] == ",".join(CSV_COLUMNS)

    def test_no_comment_by_default_leaves_the_format_unchanged(self, tmp_path):
        # A real run writes exactly what it always has -- the header
        # first, no leading comment -- so nothing downstream changes.
        path = tmp_path / "t.csv"
        with TrackLog(path) as log:
            log.record(0.0, TARGET, POSITION)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(CSV_COLUMNS)
