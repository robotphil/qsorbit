"""A CSV record of how the rotor actually moved during a pass.

**Why this exists, and why the app has to be the one writing it.** The
acceptance evidence for both the tracking profile (Chunk H) and dual-SDR
combining (Chunk E) needs rotor position sampled at around 5 Hz: the
stick-slip ring the profile exists to remove is about 1 Hz, and a
tracking cadence of 0.5 s cannot resolve it. Session 32 measured at
5.96 Hz with ``rotor-motion-log.py``, but that tool **drives the rotor
itself** -- and one process owns COM5. So it can measure the rotor or
the application can track with it, never both, and the run that matters
is the one the application is doing.

**One row per sample, both axes, and the columns are arranged so that
looking at one axis is a column pick.** Session 32's metrics --
burstiness, path ratio, reversals, mean lag -- are computed per axis,
and they are validated code worth feeding rather than reimplementing.

**The target is recomputed at every sample, not carried forward.** A
staircase target updating at 2 Hz, sampled against a position moving
continuously, produces artefacts that look like mechanism; see
:meth:`~qsorbit.core.pointing.TrackingLoop.observe`.

**Rows are flushed as they are written.** A pass that ends in a fault,
a stall, or somebody hitting the power switch is exactly the run whose
data is most worth having, and a buffered final write is what loses it.
At 5 Hz the cost is irrelevant.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import Final

from qsorbit.core.rotor import Position

#: The header row. ``outcome`` carries the tick's own outcome on rows
#: where the loop ticked, and is **empty on rows that were only
#: observed** -- an observation decided nothing, and naming a decision
#: there would let a later reader filter observations in as though they
#: were ticks.
CSV_COLUMNS: Final = (
    "t_s",
    "az_target_deg",
    "az_deg",
    "el_target_deg",
    "el_deg",
    "outcome",
)


class TrackLog:
    """Writes one CSV row per sample, flushing as it goes.

    Usage::

        with TrackLog(path) as log:
            log.record(0.0, target, position, "commanded")

    Args:
        path: Where to write. Opened on entering the context, and an
            existing file is **replaced** -- a log with two runs
            concatenated into it is worse than no log, because the time
            column restarts in the middle and nothing says so.
    """

    def __init__(self, path: Path | str, *, preamble_comment: str | None = None) -> None:
        self._path = Path(path)
        self._handle = None
        self._writer: csv.writer = None  # type: ignore[valid-type]
        self._rows = 0
        # Written as a leading ``# ...`` line before the header, for a run
        # that must not be mistaken for a normal one -- a simulated-clock
        # pass, above all. Default ``None`` means a real run writes exactly
        # what it always has, so nothing downstream changes; a reader that
        # wants the rows skips any leading ``#`` line.
        self._preamble_comment = preamble_comment

    @property
    def path(self) -> Path:
        """Where this log is being written."""
        return self._path

    @property
    def rows(self) -> int:
        """How many sample rows have been written."""
        return self._rows

    def open(self) -> None:
        """Create the file and write the header."""
        # newline="" is the csv module's documented requirement, not a
        # style choice: without it the writer's own \r\n meets the text
        # layer's newline translation on Windows and every row ends
        # \r\r\n.
        self._handle = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if self._preamble_comment is not None:
            self._handle.write(f"# {self._preamble_comment}\n")
        self._writer.writerow(CSV_COLUMNS)
        self._handle.flush()

    def record(
        self,
        elapsed_s: float,
        target: Position,
        position: Position,
        outcome: str = "",
    ) -> None:
        """Write one sample.

        Args:
            elapsed_s: Seconds since the log started. Relative rather
                than absolute so a run is self-contained, matching
                ``rotor-motion-log.py``'s ``t_s``.
            target: Where the axes would have to be to point at the
                target at this instant.
            position: Where the rotor reports they are.
            outcome: The tick's outcome, or ``""`` for a row that was
                only observed.
        """
        if self._writer is None:  # pragma: no cover - guards a misuse
            raise RuntimeError("This log has not been opened.")
        self._writer.writerow(
            [
                f"{elapsed_s:.3f}",
                f"{target.azimuth:.2f}",
                f"{position.azimuth:.2f}",
                f"{target.elevation:.2f}",
                f"{position.elevation:.2f}",
                outcome,
            ]
        )
        self._handle.flush()
        self._rows += 1

    def close(self) -> None:
        """Close the file. Safe to call twice."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None  # type: ignore[assignment]

    def describe(self) -> str:
        """One line for the end-of-run report."""
        return f"track log: {self._rows:,} sample(s) written to {self._path}"

    def __enter__(self) -> TrackLog:
        """Open on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on leaving, whether or not the body raised."""
        self.close()
