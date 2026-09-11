"""Command-line entry point, run as ``uv run qsorbit`` (or ``python -m qsorbit``).

Seven subcommands: ``point``, ``goto``, ``plan``, ``status``, ``stop``,
``sdr`` (itself split into ``info`` and ``capture``), and ``receive`` — the
whole vertical slice, tracking and receiving a pass together. ``goto`` is
``point``'s raw-axis sibling: it sends AZ/EL numbers you typed, rather
than working them out from a TLE, which is what calibration needs.
``plan`` answers a question upstream of all of them -- what's worth
pointing at in the first place -- by combining the curated satellite
catalogue, this station's own TLEs, and its horizon mask; nothing in it
touches the rotor or the SDR.

**Computing is the default; moving is opt-in.** ``point`` works out where
the rotor would have to go and prints it, without opening the serial port
at all. Only ``--send`` moves anything, and ``receive`` follows the same
rule — it demodulates a whole pass with nothing connected to the serial
port, because Doppler correction needs the TLE and the observer rather
than the rotor. That asymmetry is deliberate — the
controller firmware applies no position limits at any level, so a command
that reaches it is a command it attempts, and the cost of an accidental
slew is measured in cable and gearbox rather than in a retry.

Two things this interface has to be honest about, because both are easy
to misread:

* **Alignment calibration is applied only if your config has one.**
  ``[rotor.alignment]`` defaults to no correction, in which case the
  rotor is commanded to the raw computed sky position and any
  mechanical misalignment of the mast shows up directly as pointing
  error - true of every config file written before Chunk I. On a
  portable rig re-aimed by hand at every setup, that offset needs
  re-measuring and re-entering each time; there is still no
  auto-calibration routine to do the measuring for you.
* **Positions read back from the rotor are axis readings**, measured from
  wherever it homed — not compass bearings, and not sky elevation. They
  are labelled as such wherever they appear.

Output is deliberately plain ASCII. A degree symbol renders fine in most
terminals and then raises ``UnicodeEncodeError`` in the one legacy Windows
code page someone is actually using.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from qsorbit import __version__
from qsorbit.core.combiner import DEFAULT_MARGIN_DB, BranchSelector
from qsorbit.core.dsp import (
    DEFAULT_AUDIO_RATE_HZ,
    DEFAULT_CLOSE_BELOW_DB,
    DEFAULT_NBFM_DEVIATION_HZ,
    DEFAULT_NBFM_IF_RATE_HZ,
    DEFAULT_OPEN_ABOVE_DB,
    AudioOutput,
    DopplerTracker,
    NbfmConfig,
    NoiseSquelch,
    SpectrumConfig,
    SpectrumStream,
    SpectrumSubscription,
)
from qsorbit.core.pointing import AlignmentOffset, TrackingLoop, sky_to_rotor
from qsorbit.core.profiles import (
    CatalogManifest,
    NotConfiguredCatalogSource,
    ProfileCatalog,
    ProfileError,
    SatelliteProfile,
    Transmitter,
    load_catalog_manifest,
    load_profile_catalog,
)
from qsorbit.core.quieting_log import QuietingLog
from qsorbit.core.receive import (
    DEFAULT_TRACKING_INTERVAL_S,
    Branch,
    ReceiveSession,
    TargetRangeRate,
)
from qsorbit.core.recorder import IqRecorder, safe_filename
from qsorbit.core.rotor import (
    HomingError,
    Position,
    PositionLimitError,
    Rotor,
    RotorCapabilities,
    RotorError,
    RotorErrorCode,
    SerialPort,
    format_set_position,
)
from qsorbit.core.sdr import (
    AUTO_GAIN,
    AppliedSettings,
    IqStream,
    LibRtlSdr,
    RtlSdr,
    SdrConfig,
    SdrError,
    capture_to_file,
    index_for_serial,
)
from qsorbit.core.sdr.replay import ReplaySdr
from qsorbit.core.stall_guard import StallGuard
from qsorbit.core.station import ConfigError, SdrBranch, StationConfig, load_station_config
from qsorbit.core.track_log import TrackLog
from qsorbit.core.tracker import (
    ObserverLocation,
    Pass,
    Satellite,
    TrackerError,
    predict_passes,
)
from qsorbit.core.tracking_profile import (
    DESIGN_RATE_DEG_S,
    NOMINAL_TRACKING_RATE_DEG_S,
    TrackingProfile,
    max_safe_ki,
)
from qsorbit.core.tracking_thread import TrackingThread
from qsorbit.ui.theme import DEFAULT_THEME_NAME

if TYPE_CHECKING:
    from qsorbit.core.sdr.stream import IqSubscription
    from qsorbit.ui.feed_hub import FeedHub

#: How long ``point --send`` waits for the rotor to settle, in seconds.
DEFAULT_ARRIVAL_TIMEOUT_S = 90.0

#: How far ahead ``plan`` searches for passes by default, in hours.
#: Long enough to span a whole evening's worth of scheduling without
#: the operator having to think about it, short enough that a run
#: finishes quickly -- see the pass_prediction module's own docstring
#: for why this isn't performance-gated even so.
DEFAULT_PLAN_HOURS = 24.0

#: Default capture rate — what all of Phase 2's bring-up used.
DEFAULT_SAMPLE_RATE_HZ = 2_048_000

#: FFT size for ``receive``'s waterfall. Matches what Chunk F's bench
#: runs used, so a trace on screen during a pass is directly comparable
#: with the broadcast-FM runs the display was verified against.
RECEIVE_FFT_SIZE = 2048

#: Subscription names for the two spectrum panels. Distinct from
#: :data:`~qsorbit.core.receive.WATERFALL_SUBSCRIBER`, which names an
#: ``IqStream`` consumer one layer down -- these two consume the frames
#: that one's blocks are turned into, and they appear under these names
#: in the spectrum section of a run's report.
WATERFALL_FEED = "waterfall"
SPECTRUM_LINE_FEED = "spectrum-line"

#: How far below the signal of interest ``sdr capture`` tunes by default,
#: in kHz. Not a stylistic choice: the RTL-SDR has a permanent DC-offset
#: spike at the centre of its passband, so a signal tuned to exactly the
#: centre lands on top of an artifact and its presence proves nothing.
#: Tuning off-centre is what makes "did we receive it" answerable, and
#: making it the default is what stops it being forgotten.
DEFAULT_TUNING_OFFSET_KHZ = 250.0

#: Printed by ``sdr capture`` when the capture has a hole in it.
NON_CONTIGUOUS_WARNING = (
    "WARNING: blocks were dropped, so this capture is NOT contiguous. The "
    "samples are real but there is a gap in them, which will produce wrong "
    "and plausible-looking results in any analysis. Do not use it as a test "
    "fixture. See the sidecar for the count."
)

#: Printed wherever a commanded position is shown and the station's
#: config carries no alignment offset - which is every config file
#: written before Chunk I, and the honest default after it too, since
#: "uncalibrated" is what [rotor.alignment]'s identity value means.
UNCALIBRATED_NOTE = (
    "No alignment calibration is applied: the rotor is commanded to the raw "
    "computed sky position, so any mechanical misalignment shows up directly "
    "as pointing error."
)


def _alignment_offset(config: StationConfig) -> AlignmentOffset:
    """Build the pointing layer's :class:`AlignmentOffset` from station config.

    A one-line adapter, not a coincidence of naming: :mod:`core.station`
    deliberately does not import :mod:`core.pointing` (see that module's
    own dependency-direction rule), so nothing lower than this entry
    point can build the type :func:`~qsorbit.core.pointing.sky_to_rotor`
    actually wants. Every caller that needs one calls this rather than
    reaching into ``config.alignment`` themselves.
    """
    return AlignmentOffset(
        azimuth_deg=config.alignment.azimuth_deg,
        elevation_deg=config.alignment.elevation_deg,
    )


def _alignment_note(offset: AlignmentOffset) -> str:
    """Describe whether alignment correction is in effect, and what it is.

    :data:`UNCALIBRATED_NOTE` when ``offset`` corrects nothing - the
    default, and every config file written before Chunk I. Otherwise
    names the correction that was actually applied, so an operator who
    measured one sees it confirmed rather than told it doesn't exist -
    the same "say what's actually true" rule ``UNCALIBRATED_NOTE``
    itself exists to follow.
    """
    if offset.is_identity:
        return UNCALIBRATED_NOTE
    return (
        f"Alignment offset applied: AZ {offset.azimuth_deg:+.1f}  EL "
        f"{offset.elevation_deg:+.1f} (from [rotor.alignment] in station config). "
        "Flip mode and azimuth unwrapping are still not applied."
    )


def _squelch_status_line(*, mute: bool, open_above_db: float, close_below_db: float) -> str:
    """Describe the squelch's live configuration, for the startup banner.

    A NoiseSquelch is now built unconditionally (Chunk I: "always
    measure, optionally mute" - see demodulate_nbfm's own ``mute``
    argument and ReceiveSession's ``mute_squelch``) so the live quieting
    readout has something to show even on a run where ``--squelch`` was
    never given. This line is what tells an operator which of those two
    things is actually true here: the gate is always deciding, and
    ``--squelch`` is what lets that decision reach the speaker.
    """
    thresholds = f"open at/above {open_above_db:.1f} dB, close at/below {close_below_db:.1f} dB"
    if mute:
        return f"Squelch:   muting enabled, {thresholds}"
    return f"Squelch:   muting off - always measuring for the live quieting readout, {thresholds}"


def _parse_audio_device(value: str | None) -> int | str | None:
    """Interpret ``--audio-device`` into what ``AudioOutput`` wants.

    ``sounddevice`` itself accepts either a numeric index or a name
    substring for ``OutputStream(device=...)`` - see ``query_devices()``
    for what a string resolves against. The flag is always typed as
    ``str`` by argparse (a name like ``"2"`` and an index ``2`` need to
    stay distinguishable from each other only in *this* function, not in
    two different argparse types), so this is where that split actually
    happens: a purely numeric value becomes an ``int`` index, anything
    else is passed through as a name substring, and an unset flag stays
    ``None`` - PortAudio's own "system default" sentinel.
    """
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


class _Quittable(Protocol):
    """Anything with a ``quit()`` method - declared structurally so a test
    double stands in for a real ``QApplication`` without importing Qt.
    """

    def quit(self) -> None: ...


@contextlib.contextmanager
def _quit_on_sigint(app: _Quittable) -> Iterator[None]:
    """Make Ctrl-C during ``app.exec()`` call ``app.quit()`` instead of raising.

    Qt's ``exec()`` is a C++ loop that does not return to Python bytecode
    between events, so the interpreter's default SIGINT handler - which
    only runs *at* a bytecode boundary - cannot fire until whichever Qt
    callback happens to be executing next. Before this existed that was
    :meth:`~qsorbit.ui.waterfall_widget.WaterfallWidget._on_timer`, its
    50 ms poll being the shortest-period timer always running under
    ``_show_instruments``, and the ``KeyboardInterrupt`` raised inside it
    propagated out through Qt's C++ dispatch as an unhandled exception -
    PySide6's own exception hook printed a traceback, cosmetic since
    ``app.exec()`` still returned and the run still completed, but a real
    ugliness on every single Ctrl-C.

    **A corollary found the hard way (Session 25): this depends on some
    Python callback still running.** When a lifetime bug collected the
    window before the event loop started, every panel's timer went with
    it, ``app.exec()`` spun with no Python code to reach a bytecode
    boundary at all, and Ctrl-C stopped working entirely -- the terminal
    had to be closed. The handler was installed and correct; nothing ever
    got to run it. An event loop with no Python callbacks in it is
    uninterruptible, which is worth knowing before anything here is
    changed to poll less often.

    Installing a plain SIGINT handler sidesteps that path entirely: it
    still only runs at the next bytecode boundary, exactly the same
    timing an exception would have had, but it is an ordinary function
    call (``app.quit()``) rather than an exception unwinding through a
    callback that was never written to expect one. The previous handler
    is restored on the way out, in case something else in this process
    wants Ctrl-C's default behaviour afterward.
    """

    def handle_sigint(*_args: object) -> None:
        print()  # the ^C the terminal echoed deserves its own line
        app.quit()

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _spectrum_factory(
    window: bool, spectrum_config: SpectrumConfig
) -> Callable[[Iterable[bytes]], SpectrumStream] | None:
    """Build the spectrum pipeline's factory, or ``None`` to skip it entirely.

    Handing :class:`~qsorbit.core.receive.ReceiveSession` a factory
    commits to running :class:`~qsorbit.core.dsp.spectrum_stream.SpectrumStream`'s
    worker thread - unpacking IQ and computing FFTs continuously for as
    long as the session runs - so with no window open to show a frame to
    (``window=False``), returning ``None`` means that pipeline never
    starts at all, rather than starting it and discarding everything it
    computes. Measured at 98.2% of computed frames discarded on a
    headless run: this makes the honest headless figure 0%.
    """
    if not window:
        return None
    return lambda blocks: SpectrumStream(blocks, spectrum_config)


RotorFactory = Callable[[StationConfig, Callable[[float], None]], Rotor]
SdrFactory = Callable[[StationConfig, SdrBranch | None], RtlSdr]

#: What a single-device station's one branch is called in reports, when
#: there is no ``[[sdr.branch]]`` entry naming the antenna it is on.
UNNAMED_BRANCH_LABEL: Final = "radio"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser. Separate from :func:`main` so tests can
        inspect it without running a command.
    """
    parser = argparse.ArgumentParser(
        prog="qsorbit",
        description="Satellite tracking and rotor control for amateur radio.",
    )
    parser.add_argument("--version", action="version", version=f"QSOrbit {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Station config file. Defaults to ./qsorbit.toml, then the per-user "
            "config directory. See config.example.toml."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    point = subcommands.add_parser(
        "point",
        help="Work out where to point at a satellite, and optionally go there.",
        description=(
            "Computes the rotor position for a satellite at a given time and "
            "prints it. Nothing is transmitted, and the serial port is not "
            "opened, unless --send is given."
        ),
    )
    point.add_argument(
        "--tle",
        required=True,
        metavar="PATH",
        help="File holding one TLE: two element lines, optionally preceded by a name line.",
    )
    point.add_argument(
        "--at",
        metavar="TIME",
        help=(
            "ISO 8601 instant to compute for, e.g. 2026-08-20T18:30:00+00:00. "
            "Defaults to now. A time with no zone offset is read as UTC."
        ),
    )
    point.add_argument(
        "--send",
        action="store_true",
        help="Actually move the rotor. Without this, nothing is transmitted.",
    )
    point.add_argument(
        "--arrival-timeout",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"How long to wait for the move to settle (default {DEFAULT_ARRIVAL_TIMEOUT_S:.0f}).",
    )

    goto = subcommands.add_parser(
        "goto",
        help="Send the rotor straight to an axis position - no TLE, no satellite.",
        description=(
            "Command the rotor directly to AZ/EL axis coordinates, the same "
            "range-checked and --send-gated way 'point' commands a computed "
            "one. For calibration and manual pointing, where the position "
            "is typed rather than worked out from an orbit."
        ),
    )
    goto.add_argument(
        "--az",
        type=float,
        required=True,
        metavar="DEGREES",
        help="Azimuth axis position to send.",
    )
    goto.add_argument(
        "--el",
        type=float,
        required=True,
        metavar="DEGREES",
        help="Elevation axis position to send.",
    )
    goto.add_argument(
        "--send",
        action="store_true",
        help="Actually move the rotor. Without this, nothing is transmitted.",
    )
    goto.add_argument(
        "--arrival-timeout",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
        metavar="SECONDS",
        help=f"How long to wait for the move to settle (default {DEFAULT_ARRIVAL_TIMEOUT_S:.0f}).",
    )

    plan = subcommands.add_parser(
        "plan",
        help="List upcoming passes worth pointing at, from the curated satellite catalogue.",
        description=(
            "Predicts passes for every satellite that has both a curated profile "
            "and a TLE in --tle-dir, filtered by this station's horizon mask, and "
            "prints what each one transmits and how likely it is to actually be "
            "on. Read-only -- nothing is transmitted and no rotor is involved."
        ),
    )
    plan.add_argument(
        "--tle-dir",
        required=True,
        metavar="PATH",
        help=(
            "Directory of *.tle files, one satellite each. Matched to "
            "profiles by NORAD catalog number."
        ),
    )
    plan.add_argument(
        "--profiles-dir",
        metavar="PATH",
        help="Directory of profile *.toml files. Defaults to QSOrbit's own curated starter set.",
    )
    plan.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_PLAN_HOURS,
        metavar="HOURS",
        help=f"How far ahead to search for passes (default {DEFAULT_PLAN_HOURS:.0f}).",
    )
    plan.add_argument(
        "--at",
        metavar="TIME",
        help=(
            "ISO 8601 instant to start the search from, e.g. "
            "2026-08-20T18:30:00+00:00. Defaults to now. A time with no zone "
            "offset is read as UTC."
        ),
    )
    plan.add_argument(
        "--visual",
        action="store_true",
        help="Also report whether each pass is naked-eye visible (sunlit satellite, dark sky).",
    )
    plan.add_argument(
        "--refresh-catalogue",
        action="store_true",
        help=(
            "Fetch an updated profile catalogue over the network before planning, "
            "instead of using the shipped snapshot. Not yet wired to a real source "
            "-- fails with a clear error rather than silently doing nothing."
        ),
    )

    subcommands.add_parser(
        "status",
        help="Read the rotor's firmware version, error state, and position.",
        description=(
            "Read-only. Connects, reports what the rotator says about itself, "
            "and has no way to move anything."
        ),
    )

    subcommands.add_parser(
        "stop",
        help="Halt a converging move by setting the setpoint to the current position.",
        description=(
            "NOT an emergency stop. It halts a move that is converging; it "
            "cannot stop one that is diverging. The power switch is the real stop."
        ),
    )

    _add_sdr_commands(subcommands)
    _add_receive_command(subcommands)
    _add_shell_command(subcommands)
    return parser


def _add_shell_command(subcommands: argparse._SubParsersAction) -> None:
    """Add ``shell``: the tabbed application.

    **Why this is a command of its own rather than a flag on
    ``receive``.** Two of the shell's tabs need no SDR at all -- the
    rotor readout, and the pass planning that arrives in Chunk D -- and
    ``receive`` structurally cannot run without one. A shell reachable
    only through ``receive --window`` would be an application you had to
    start a radio to look at.

    ``receive --window`` is deliberately left exactly as it was. It is
    the instrument every USB-loss measurement in Sessions 24 through 27
    was taken through, including the 0.0175% that proved the theme
    system costs the read path nothing, and changing it in the same PR
    that adds the shell would leave the next bench number with nothing
    comparable to sit beside. It becomes the control run instead, which
    is worth more than the duplication costs for one chunk.
    """
    shell = subcommands.add_parser(
        "shell",
        help="Open the QSOrbit application window.",
        description=(
            "The tabbed shell: Radio, Rotor, Plan and Decode, with every panel "
            "fed by one hub. Every part of it is optional. With no --tle it "
            "opens with each tab saying what it is waiting for, which is what a "
            "sky-free evening looks like; with --tle and --downlink it receives; "
            "with --send as well it moves the antenna too."
        ),
    )
    _add_radio_arguments(shell, required=False)
    shell.add_argument(
        "--at",
        default=None,
        metavar="TIME",
        help=(
            "Simulate a pass at this ISO 8601 time instead of tracking in real "
            "time: the rotor follows the target's geometry as if it were then, so "
            "you can check rotor behaviour without waiting for a live pass. Past "
            "or future. Rotor-only -- cannot be combined with --downlink (simulate "
            "the signal with a replay instead)."
        ),
    )


def _add_radio_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    """Add the radio options shared by ``receive`` and ``shell``.

    One definition, two parsers, and that is the point rather than
    tidiness. Eighteen options copied into a second command would
    drift the first time one of them gained a default -- and a
    ``--gain`` that meant something slightly different depending on
    which command you typed is exactly the kind of quiet
    disagreement between a flag and its behaviour this project keeps
    finding the hard way.

    Args:
        parser: The subcommand parser to add them to.
        required: Whether the radio is mandatory. ``receive`` cannot
            run without a TLE, a downlink and a gain, so it passes
            ``True``. ``shell`` can: it opens with the Radio tab in
            placeholder and the rest of the application working,
            which is what a sky-free evening looks like.

    ``--window`` is deliberately **not** here. It is ``receive``'s
    own flag, and a shell that had to be told to open a window would
    be a strange thing to have typed.
    """
    parser.add_argument(
        "--tle",
        required=required,
        metavar="PATH",
        help="File holding one TLE: two element lines, optionally preceded by a name line.",
    )
    parser.add_argument(
        "--downlink",
        type=float,
        required=required,
        metavar="MHZ",
        help=(
            "The satellite's nominal downlink in MHz, as transmitted - not the "
            "frequency you expect to hear. Doppler is what makes those differ, "
            "and computing it is this command's job."
        ),
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_TUNING_OFFSET_KHZ,
        metavar="KHZ",
        help=(
            f"How far below the downlink to place the tuner, in kHz (default "
            f"{DEFAULT_TUNING_OFFSET_KHZ:.0f}). Same reason as 'sdr capture': the "
            "receiver's DC spike sits at the centre of its own passband."
        ),
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        metavar="SPS",
        help=f"IQ sample rate (default {DEFAULT_SAMPLE_RATE_HZ:,}).",
    )
    parser.add_argument(
        "--if-rate",
        type=float,
        default=DEFAULT_NBFM_IF_RATE_HZ,
        metavar="HZ",
        help=(
            f"Channel-filter output rate (default {DEFAULT_NBFM_IF_RATE_HZ:,.0f}). "
            "Must divide --rate evenly and be more than twice --deviation."
        ),
    )
    parser.add_argument(
        "--audio-rate",
        type=float,
        default=DEFAULT_AUDIO_RATE_HZ,
        metavar="HZ",
        help=f"Playback sample rate (default {DEFAULT_AUDIO_RATE_HZ:,.0f}).",
    )
    parser.add_argument(
        "--deviation",
        type=float,
        default=DEFAULT_NBFM_DEVIATION_HZ,
        metavar="HZ",
        help=(
            f"Transmitter peak deviation (default {DEFAULT_NBFM_DEVIATION_HZ:,.0f}, "
            "which suits amateur FM). Wider modes need both this and --if-rate raised."
        ),
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        metavar="S",
        help="Stop automatically after this long. Default: run until Ctrl-C.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Seconds between tracking updates. Overrides the active tracking "
            "profile's own interval; without it the profile decides, and a "
            f"station with no profiles uses {DEFAULT_TRACKING_INTERVAL_S:.0f}."
        ),
    )
    parser.add_argument(
        "--rotor-profile",
        default=None,
        metavar="NAME",
        help=(
            "Which [rotor.profiles.NAME] tracking profile to run. Overrides "
            "the config's own `profile` key for this run."
        ),
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help=(
            "Actually move the rotor to follow the pass. Without this nothing "
            "is transmitted to the controller and the serial port is not opened."
        ),
    )
    parser.add_argument(
        "--listen",
        default=None,
        metavar="LABEL",
        help=(
            "Which receive branch reaches the speaker, by the label it was "
            "given in [[sdr.branch]]. Defaults to the first declared. Every "
            "branch demodulates and measures either way -- this chooses what "
            "you hear, and the waterfall follows it. Means nothing on a "
            "station with one dongle, which has one branch."
        ),
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Let the combiner choose which branch you hear, switching when "
            "another beats it by the margin. Off by default, and deliberately: "
            "'combined beats either branch alone' is a comparison, and the "
            "control for it is this same command without this flag."
        ),
    )
    parser.add_argument(
        "--combine-margin",
        type=float,
        default=DEFAULT_MARGIN_DB,
        metavar="DB",
        help=(
            f"How far a branch must beat the current one before the speaker "
            f"moves (default {DEFAULT_MARGIN_DB:.1f} dB, measured rather than "
            "assumed - see core/combiner.py). Zero means no hysteresis at all, "
            "which is the control that shows why hysteresis is needed."
        ),
    )
    parser.add_argument(
        "--quieting-log",
        default=None,
        metavar="PATH",
        help=(
            "Write a CSV of what each branch was hearing: one row per block "
            "per branch, with the quieting measurement and the squelch's "
            "decision. This is what a combiner's switching margin is sized "
            "from - the wander in the difference between two branches "
            "hearing the same thing - which no summary of minima and maxima "
            "can give."
        ),
    )
    parser.add_argument(
        "--track-log",
        default=None,
        metavar="PATH",
        help=(
            "Write a CSV of how the rotor actually moved: one row per sample "
            "at about 5 Hz, both axes, target beside position. Needs --send. "
            "Sampling rides the thread that already owns the serial port, so "
            "it adds reads but no contention; a run without this flag pays "
            "nothing for it."
        ),
    )
    parser.add_argument(
        "--record-iq",
        default=None,
        metavar="DIR",
        help=(
            "Record every branch's raw IQ to DIR while the pass runs: one "
            ".iq file plus a sidecar per branch, named for the antenna. The "
            "recorder is a consumer on each branch's stream, not a second "
            "process, so it never contends with the demodulator for the "
            "dongle - which is what lets a capture be made during the same "
            "run that logs quieting and drives the rotor. Needs --downlink; "
            "about 4 MB/s per branch, so check the disk before a long pass."
        ),
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="LABEL=FILE.iq[,LABEL=FILE.iq]",
        help=(
            "Replay a captured .iq through the real receive path instead of "
            "opening a radio, matching each branch by its config label - or a "
            "bare FILE.iq for a single unnamed branch. The replay reports what "
            "the file's sidecar says (a 1.024 Msps capture replays as 1.024 "
            "even on a 2.048 station) and runs in real time, so audio plays "
            "and the gate reacts as on the night. A diagnostic: it needs a "
            "capture, which only --record-iq makes."
        ),
    )
    parser.add_argument(
        "--replay-attenuate",
        default=None,
        metavar="LABEL=DB[,LABEL=DB]",
        help=(
            "Attenuate a replayed branch by DB decibels (or a bare DB for a "
            "single unnamed branch). The known-answer control: feed one "
            "capture to both branches and pad one down, and a correct "
            "combiner must sit on the un-padded branch throughout. Needs "
            "--replay; the effect is a re-quantisation floor, so use a large "
            "value (20+ dB) for a decisive difference."
        ),
    )
    parser.add_argument(
        "--theme",
        metavar="SLUG",
        default=DEFAULT_THEME_NAME,
        help=(
            "Theme for the instrument window, by filename stem - one of the eight "
            "shipped in ui/themes/, or your own dropped into the themes/ directory "
            "beside config.toml. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--squelch",
        action="store_true",
        help=(
            "Mute the noise squelch's closed gate. Off by default: a mute set "
            "slightly too tight makes a working receiver indistinguishable from "
            "a broken one. The gate is always measured and shown live either way "
            "- see --window - this only controls whether it silences audio."
        ),
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Output device: a numeric index or a name substring, matching "
            "sounddevice's own OutputStream(device=...). Defaults to the system's "
            "configured default. To see what's available, run "
            "'python -c \"import sounddevice as sd; print(sd.query_devices())\"'."
        ),
    )
    parser.add_argument(
        "--squelch-open",
        type=float,
        default=DEFAULT_OPEN_ABOVE_DB,
        metavar="DB",
        help=f"Quieting at/above which the gate opens (default {DEFAULT_OPEN_ABOVE_DB:.1f}).",
    )
    parser.add_argument(
        "--squelch-close",
        type=float,
        default=DEFAULT_CLOSE_BELOW_DB,
        metavar="DB",
        help=f"Quieting at/below which it closes (default {DEFAULT_CLOSE_BELOW_DB:.1f}).",
    )
    gain = parser.add_mutually_exclusive_group(required=required)
    gain.add_argument(
        "--gain",
        type=float,
        metavar="DB",
        help=(
            "Tuner gain in dB. Required, and deliberately has no default: a "
            "default gain is how a pass comes back silent with nobody noticing."
        ),
    )
    gain.add_argument(
        "--auto-gain",
        action="store_true",
        help="Let the tuner choose. Rarely what you want - during bring-up it reported 0.0 dB.",
    )


def _add_receive_command(subcommands: argparse._SubParsersAction) -> None:
    """Add ``receive``: the whole slice, tracking and receiving together.

    Split out for the same reason the ``sdr`` group is: it has more
    options than everything else here put together, and inlining it
    would bury the commands above it.
    """
    receive = subcommands.add_parser(
        "receive",
        help="Follow a satellite and demodulate its FM downlink.",
        description=(
            "Runs the tracking loop and the receive chain together: the SDR "
            "streams, NBFM demodulates, and the Doppler correction follows the "
            "pass using range rates computed from the TLE. The rotor is NOT "
            "moved unless --send is given, and no window opens unless --window "
            "is. Doppler correction needs the TLE and your location, not the "
            "rotor, so the whole radio job works with nothing on the serial port."
        ),
    )
    _add_radio_arguments(receive, required=True)
    receive.add_argument(
        "--window",
        action="store_true",
        help=(
            "Open the instrument window - waterfall and live quieting, plus the "
            "readout when --send is given."
        ),
    )


def _add_sdr_commands(subcommands: argparse._SubParsersAction) -> None:
    """Add the ``sdr`` group: ``info`` and ``capture``.

    Split out because the group has its own nested subcommands and
    inlining it would bury the rotor commands.
    """
    sdr = subcommands.add_parser(
        "sdr",
        help="Inspect the SDR, or capture IQ from it.",
        description="Receiver-side commands. Nothing here moves the rotor.",
    )
    sdr_commands = sdr.add_subparsers(dest="sdr_command", required=True, metavar="COMMAND")

    sdr_commands.add_parser(
        "info",
        help="Report what SDR is attached and what it can do.",
        description=(
            "Read-only. Opens the device, reports how it identifies itself and "
            "which gain steps its tuner offers, and closes it again. The gain "
            "table doubles as a fingerprint of which librtlsdr got loaded."
        ),
    )

    capture = sdr_commands.add_parser(
        "capture",
        help="Record raw IQ to a file, with a JSON sidecar.",
        description=(
            "Captures raw uint8 interleaved I/Q - the same format rtl_sdr.exe "
            "writes - alongside a sidecar recording what the device actually "
            "did. By default the tuner is placed below the signal of interest "
            "rather than on it, because a peak at the centre of the passband "
            "cannot be told apart from the receiver's own DC spike."
        ),
    )
    target = capture.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--station",
        type=float,
        metavar="MHZ",
        help=(
            "Signal of interest in MHz. The tuner is placed --offset kHz below "
            "it, and both the frequency and its offset are recorded in the sidecar."
        ),
    )
    target.add_argument(
        "--center",
        type=float,
        metavar="MHZ",
        help="Tune here directly, in MHz. No signal of interest is recorded.",
    )
    capture.add_argument(
        "--offset",
        type=float,
        default=DEFAULT_TUNING_OFFSET_KHZ,
        metavar="KHZ",
        help=(
            f"How far below --station to tune, in kHz (default "
            f"{DEFAULT_TUNING_OFFSET_KHZ:.0f}). Ignored with --center."
        ),
    )
    capture.add_argument(
        "--seconds", type=float, default=2.0, metavar="S", help="Duration (default 2)."
    )
    capture.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        metavar="SPS",
        help=f"Sample rate in samples per second (default {DEFAULT_SAMPLE_RATE_HZ:,}).",
    )
    gain = capture.add_mutually_exclusive_group(required=True)
    gain.add_argument(
        "--gain",
        type=float,
        metavar="DB",
        help=(
            "Tuner gain in dB, snapped to the nearest step the device offers. "
            "Required, and deliberately has no default: a default gain is how a "
            "capture comes back empty with nobody noticing."
        ),
    )
    gain.add_argument(
        "--auto-gain",
        action="store_true",
        help=(
            "Let the tuner choose. Rarely what you want - during bring-up this "
            "reported 0.0 dB and captured a flat noise floor, with no error."
        ),
    )
    capture.add_argument(
        "--out", required=True, metavar="PATH", help="Where to write the .iq file."
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    rotor_factory: RotorFactory | None = None,
    sdr_factory: SdrFactory | None = None,
) -> int:
    """Run the CLI.

    Args:
        argv: Command-line arguments, defaulting to :data:`sys.argv`.
        rotor_factory: Builds the :class:`~qsorbit.core.rotor.Rotor` from
            a config and a homing-progress callback. Injected by tests so
            no serial port is ever opened.
        sdr_factory: Builds the :class:`~qsorbit.core.sdr.RtlSdr` from a
            config. Injected by tests so no device is ever opened.

    Returns:
        A process exit code: 0 on success, 1 on a failure the operator
        needs to read. Argparse exits with 2 on a usage error.
    """
    args = build_parser().parse_args(argv)
    factory = rotor_factory or _open_rotor

    try:
        config = load_station_config(args.config)
        if args.command == "point":
            return _command_point(args, config, factory)
        if args.command == "goto":
            return _command_goto(args, config, factory)
        if args.command == "plan":
            return _command_plan(args, config)
        if args.command == "status":
            return _command_status(config, factory)
        if args.command == "sdr":
            return _command_sdr(args, config, sdr_factory or _open_sdr)
        replay_factory: SdrFactory | None = None
        if args.command in ("receive", "shell") and getattr(args, "replay", None):
            try:
                attenuation = (
                    _parse_attenuation_map(args.replay_attenuate)
                    if getattr(args, "replay_attenuate", None)
                    else None
                )
                replay_factory = _replay_sdr_factory(_parse_replay_map(args.replay), attenuation)
            except ValueError as exc:
                print(f"{args.command}: {exc}", file=sys.stderr)
                return 1
        elif args.command in ("receive", "shell") and getattr(args, "replay_attenuate", None):
            print(f"{args.command}: --replay-attenuate needs --replay.", file=sys.stderr)
            return 1
        if args.command == "receive":
            return _command_receive(
                args, config, factory, sdr_factory or replay_factory or _open_sdr
            )
        if args.command == "shell":
            return _command_shell(args, config, factory, sdr_factory or replay_factory or _open_sdr)
        return _command_stop(config, factory)
    except HomingError as exc:
        # Its own state rather than a generic failure: nothing sent over
        # the link clears it, so "try again" would be the wrong advice.
        print(f"Homing failure: {exc}", file=sys.stderr)
        return 1
    except (
        ConfigError,
        ProfileError,
        RotorError,
        SdrError,
        TrackerError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _command_point(args: argparse.Namespace, config: StationConfig, factory: RotorFactory) -> int:
    # Time first: it is the cheapest thing to get wrong, and a typo in
    # --at should not wait behind reading and parsing a TLE.
    when = _parse_time(args.at)
    satellite = Satellite.from_file(args.tle)
    state = satellite.topocentric_state(config.observer, when)
    sky = state.sky_position
    offset = _alignment_offset(config)
    target = sky_to_rotor(sky, offset)

    print(f"Target:    {satellite.name}")
    print(f"Time:      {when.isoformat()}")
    print(f"Sky:       az {sky.azimuth:.1f}  el {sky.elevation:.1f}  (degrees)")
    print(f"Range:     {state.range_km:.0f} km, {_range_description(state.range_rate_km_s)}")
    print(f"Rotor:     {_format_position(target)}  (axis command)")
    print(f"Command:   {bytes(format_set_position(target)).decode('ascii').strip()}")
    print()
    print(_alignment_note(offset))

    if sky.elevation < 0.0:
        print(
            f"The target is below the horizon ({sky.elevation:.1f} degrees), so "
            "it is not observable from here at this time."
        )

    try:
        config.capabilities.check_setpoint(target)
    except PositionLimitError as exc:
        print(f"\nOut of range: {exc}", file=sys.stderr)
        return 1

    if not args.send:
        print("\nNothing was sent. Re-run with --send to move the rotor.")
        return 0

    return _send_and_wait(config, factory, target, args.arrival_timeout)


def _command_goto(args: argparse.Namespace, config: StationConfig, factory: RotorFactory) -> int:
    """``goto``: send the rotor straight to a typed AZ/EL, no TLE involved.

    ``point``'s raw-axis sibling. Everything past "here is the target
    position" is identical between the two - the range check, the
    --send gate, connecting, moving, and reporting arrival - which is
    why both call :func:`_send_and_wait` rather than each doing it
    themselves. What differs is only where ``target`` comes from: a
    TLE and a time there, typed numbers here.
    """
    target = Position(azimuth=args.az, elevation=args.el)

    print(f"Rotor:     {_format_position(target)}  (axis command)")
    print(f"Command:   {bytes(format_set_position(target)).decode('ascii').strip()}")

    try:
        config.capabilities.check_setpoint(target)
    except PositionLimitError as exc:
        print(f"\nOut of range: {exc}", file=sys.stderr)
        return 1

    if not args.send:
        print("\nNothing was sent. Re-run with --send to move the rotor.")
        return 0

    return _send_and_wait(config, factory, target, args.arrival_timeout)


def _command_plan(args: argparse.Namespace, config: StationConfig) -> int:
    """``plan``: what's worth pointing at, from the curated catalogue and this station's own TLEs.

    Answers the question every other subcommand assumes has already
    been answered -- *which* satellite, right now. Loads every TLE in
    ``--tle-dir``, matches each one to a curated profile by NORAD
    catalog number (:attr:`~qsorbit.core.tracker.satellite.Satellite.norad_id`),
    predicts passes against this station's own horizon mask rather than
    the bare geometric horizon, and prints AOS/TCA/LOS alongside what
    each satellite actually transmits and how reliably. Read-only --
    nothing is sent to the rotor and no SDR is opened, which is why it
    takes a ``StationConfig`` but no ``RotorFactory``, unlike ``point``
    and ``goto``.

    A TLE with no matching profile, or a file that doesn't parse as a
    TLE, is skipped with a note on stderr rather than aborting the
    whole search -- the catalogue is a curated subset by design (see
    ``core/profiles/``), so an unmatched TLE is the expected case, not
    an error.

    If the catalogue directory carries a manifest (:func:`~qsorbit.core.
    profiles.load_catalog_manifest`), prints how stale the shipped
    snapshot is. ``--refresh-catalogue`` asks for a network refresh
    first -- which currently always fails, loudly and specifically,
    because no real source is wired up yet (see ``catalog_source.py``).
    """
    catalog = (
        load_profile_catalog(args.profiles_dir) if args.profiles_dir else load_profile_catalog()
    )
    manifest = (
        load_catalog_manifest(args.profiles_dir) if args.profiles_dir else load_catalog_manifest()
    )

    if args.refresh_catalogue:
        # Always raises for now -- caught by main()'s ProfileError handler,
        # same path any other catalogue-loading failure takes.
        NotConfiguredCatalogSource().refresh()

    tle_dir = Path(args.tle_dir)
    if not tle_dir.is_dir():
        raise ValueError(f"TLE directory not found: {tle_dir}")

    now = _parse_time(args.at)
    end = now + timedelta(hours=args.hours)

    entries: list[tuple[SatelliteProfile, Pass]] = []
    for tle_path in sorted(tle_dir.glob("*.tle")):
        try:
            satellite = Satellite.from_file(tle_path)
        except TrackerError as exc:
            print(f"Could not read {tle_path} as a TLE: {exc}", file=sys.stderr)
            continue

        profile = catalog.by_norad_id(satellite.norad_id)
        if profile is None:
            print(
                f"No curated profile for {satellite.name} (NORAD {satellite.norad_id}) "
                f"in {tle_path} -- skipped.",
                file=sys.stderr,
            )
            continue

        passes = predict_passes(
            satellite,
            config.observer,
            now,
            end,
            horizon_mask=config.horizon,
            include_illumination=args.visual,
        )
        for one_pass in passes:
            entries.append((profile, one_pass))

    entries.sort(key=lambda entry: entry[1].aos.time)

    print(f"Searched {tle_dir} for the next {args.hours:.1f} hours, from {now.isoformat()}.")
    if manifest is not None:
        age_days = (now.date() - manifest.shipped).days
        print(f"Curated catalogue shipped {manifest.shipped.isoformat()} ({age_days} d ago).")
    print()

    if not entries:
        print("Nothing above the horizon in that window.")
        return 0

    for profile, one_pass in entries:
        _print_pass(profile, one_pass)

    return 0


def _print_pass(profile: SatelliteProfile, one_pass: Pass) -> None:
    """Print one pass -- AOS/TCA/LOS, visibility, and what it transmits."""
    reliability = profile.best_reliability()
    reliability_text = reliability.value if reliability is not None else "no known transmitter"

    print(f"{profile.name}  ({reliability_text})")
    print(f"  AOS {one_pass.aos.time.isoformat()}  az {one_pass.aos.sky_position.azimuth:5.1f}")
    print(f"  TCA {one_pass.tca.time.isoformat()}  el {one_pass.max_elevation_deg:5.1f}")
    print(f"  LOS {one_pass.los.time.isoformat()}  az {one_pass.los.sky_position.azimuth:5.1f}")
    if one_pass.illuminated is not None:
        print(f"  naked-eye visible near TCA: {'yes' if one_pass.illuminated else 'no'}")
    for transmitter in profile.transmitters:
        print(f"  {_format_transmitter(transmitter)}")
    print()


def _format_transmitter(transmitter: Transmitter) -> str:
    """One line (plus an optional indented note) describing a transmitter."""
    downlink_mhz = transmitter.downlink_hz / 1_000_000.0
    parts = [f"{downlink_mhz:.4f} MHz down"]
    if transmitter.uplink_hz is not None:
        parts.append(f"{transmitter.uplink_hz / 1_000_000.0:.4f} MHz up")
    parts.append(transmitter.mode.value)
    parts.append(transmitter.reliability.value)

    line = "    " + "  ".join(parts)
    if transmitter.notes:
        line += f"\n      {transmitter.notes}"
    return line


def _send_and_wait(
    config: StationConfig, factory: RotorFactory, target: Position, arrival_timeout_s: float
) -> int:
    """Connect, move to ``target``, wait for arrival, and report the outcome.

    Shared by ``point --send`` and ``goto --send``: once a target
    :class:`~qsorbit.core.rotor.Position` exists, sending it and
    reporting arrival is identical whether that position came from a
    TLE and a time, or was typed directly.
    """
    with _Connected(config, factory) as rotor:
        print(f"\nConnected: {rotor.firmware_version}")
        rotor.move_to(target)
        print(f"Sent:      {_format_position(target)}")
        arrival = rotor.wait_for_arrival(target, timeout_s=arrival_timeout_s)
        if arrival.arrived:
            print(f"Arrived:   {_format_position(arrival.position)} after {arrival.elapsed_s:.1f}s")
            return 0
        print(
            f"Did not settle within {arrival_timeout_s:.0f}s. Last reading "
            f"{_format_position(arrival.position)}, target {_format_position(target)}.",
            file=sys.stderr,
        )
        return 1


def _command_status(config: StationConfig, factory: RotorFactory) -> int:
    with _Connected(config, factory) as rotor:
        status = rotor.status()
        capabilities = config.capabilities

        print(f"QSOrbit {__version__}")
        print(f"Config:    {config.source_path}")
        print(f"Port:      {config.serial.port} at {config.serial.baudrate} baud")
        print(f"Firmware:  {status.firmware_version}")
        if (
            capabilities.firmware_version is not None
            and capabilities.firmware_version != status.firmware_version
        ):
            print(
                f"           Config declares {capabilities.firmware_version}. "
                "QSOrbit was verified against the declared version; behaviour on "
                "this one is untested."
            )
        print(f"Error:     {_describe_error(status.error)}")
        print(f"Position:  {_format_position(status.position)}")
        print(
            "           Axis readings, measured from where the rotor homed - "
            "not compass bearings, and not sky elevation."
        )
        print(
            f"Limits:    AZ {capabilities.azimuth_min_deg:.1f} to "
            f"{capabilities.azimuth_max_deg:.1f}, "
            f"EL {capabilities.elevation_min_deg:.1f} to "
            f"{capabilities.elevation_max_deg:.1f}  (axis travel)"
        )
        print(f"Wrap:      {capabilities.azimuth_wrap.value}")
        print(f"Window:    {capabilities.acceptance_window_deg:.1f} degrees acceptance")
        print(_describe_cadence(config.tracking_profile))
        declared = [entry.name for entry in config.tracking.profiles]
        if declared:
            print(f"Profiles:  {', '.join(declared)}")
        else:
            print("Profiles:  none declared (see [rotor.profiles] in config.example.toml)")
        for line in _describe_mechanics(capabilities):
            print(line)
        offset = _alignment_offset(config)
        if offset.is_identity:
            print("Alignment: none recorded (see [rotor.alignment] in config.example.toml)")
        else:
            print(
                f"Alignment: AZ {offset.azimuth_deg:+.1f}  EL {offset.elevation_deg:+.1f}  "
                "(from [rotor.alignment])"
            )
        print()
        print(_alignment_note(offset))
        return 0 if status.healthy else 1


def _command_sdr(args: argparse.Namespace, config: StationConfig, factory: SdrFactory) -> int:
    """Dispatch the ``sdr`` group."""
    if args.sdr_command == "info":
        return _command_sdr_info(config, factory)
    return _command_sdr_capture(args, config, factory)


def _command_sdr_info(config: StationConfig, factory: SdrFactory) -> int:
    _note_single_device(config)
    with factory(config, None) as sdr:
        info = sdr.info
        gains = sdr.supported_gains_db()

        print(f"QSOrbit {__version__}")
        print(f"Config:    {config.source_path}")
        print(f"Driver:    {config.sdr.driver_dir or 'system library search path'}")
        print(f"Device:    {info.describe()}")
        branch = _active_branch(config)
        if branch is not None:
            print(f"Branch:    {branch.label} (serial {branch.serial})")
        print(f"Gains:     {len(gains)} steps, {min(gains)} to {max(gains)} dB")
        print(f"           {' '.join(str(step) for step in gains)}")
        print(f"Ppm:       {_sdr_ppm(config)}")
        print()
        print(
            "The gain table belongs to the tuner chip, so it doubles as a "
            "fingerprint: a V4's R828D reports 29 steps topping out at 49.6 dB, "
            "and a different table means a different library was loaded than "
            "the one intended."
        )
        return 0


def _command_sdr_capture(
    args: argparse.Namespace, config: StationConfig, factory: SdrFactory
) -> int:
    station_hz = args.station * 1e6 if args.station is not None else None
    if station_hz is not None:
        center_hz = station_hz - args.offset * 1e3
    else:
        center_hz = args.center * 1e6

    sdr_config = SdrConfig(
        center_hz=center_hz,
        sample_rate_hz=args.rate,
        gain_db=AUTO_GAIN if args.auto_gain else args.gain,
        ppm=_sdr_ppm(config),
    )
    if sdr_config.may_drop_samples:
        print(
            f"Note: {args.rate:,.0f} sps is above what USB reliably sustains on "
            "many machines. If the capture comes back short, this is the first "
            "thing to look at."
        )

    _note_single_device(config)
    with factory(config, None) as sdr:
        print(f"Device:    {sdr.info.describe()}")
        result = capture_to_file(
            sdr,
            sdr_config,
            args.out,
            seconds=args.seconds,
            station_hz=station_hz,
        )

    print(result.describe())
    print(f"Sidecar:   {result.sidecar_path}")
    if result.applied.reports_zero_gain:
        print(
            "\nWARNING: the tuner reports 0.0 dB of gain, which nearly always "
            "means the capture is empty. Set a manual gain."
        )
    if not result.is_contiguous:
        print(f"\n{NON_CONTIGUOUS_WARNING}", file=sys.stderr)
        return 1
    return 0


def _command_receive(
    args: argparse.Namespace,
    config: StationConfig,
    rotor_factory: RotorFactory,
    sdr_factory: SdrFactory,
    *,
    runner: Callable[..., int] | None = None,
) -> int:
    """Run the vertical slice: track, stream, demodulate, correct, play.

    ``runner`` is what actually builds the window and runs the session,
    defaulting to :func:`_run_receive`. ``shell`` passes
    :func:`_run_shell` instead and gets every line of the setup above it
    unchanged -- the TLE read, the tuner configuration, the printed
    chain description, the gain warning, the squelch line, and above all
    **the order those happen in**, which is "fails cheapest first" and
    was arrived at deliberately. A second command that re-implemented
    that order would drift from it, and the drift would show up as a
    rotor that homes before a typo in ``--tle`` is noticed.

    Reads in the order that fails cheapest first — the TLE before the
    radio, the radio before the rotor — so a typo in ``--tle`` does not
    wait behind opening a USB device, and a rotor that will not home does
    not cost the SDR configuration that already succeeded.
    """
    run = runner if runner is not None else _run_receive
    satellite = Satellite.from_file(args.tle)
    downlink_hz = args.downlink * 1e6
    center_hz = downlink_hz - args.offset * 1e3

    declared = _declared_branches(config)
    try:
        listening = _listening_index(args.listen, declared)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    with contextlib.ExitStack() as devices:
        radios: list[_Radio] = []
        for index, branch in enumerate(declared):
            label = _branch_label(branch)
            # Its own ppm, because a crystal correction is a property of
            # one oscillator and two dongles have two of them.
            sdr_config = SdrConfig(
                center_hz=center_hz,
                sample_rate_hz=args.rate,
                gain_db=AUTO_GAIN if args.auto_gain else args.gain,
                ppm=_branch_ppm(config, branch),
            )
            sdr = devices.enter_context(sdr_factory(config, branch))
            applied = sdr.configure(sdr_config)
            heard = " <- audio" if index == listening else ""
            print(f"Target:    {satellite.name}" if index == 0 else "")
            print(f"Branch:    {label}{heard}")
            print(f"Device:    {sdr.info.describe()}")
            print(
                f"Tuned:     {applied.center_hz / 1e6:.4f} MHz, downlink "
                f"{applied.offset_from(downlink_hz) / 1e3:+.1f} kHz from centre "
                f"(before Doppler)"
            )
            if applied.reports_zero_gain:
                print(
                    f"\nWARNING: {label}'s tuner reports 0.0 dB of gain, which "
                    "nearly always means you will hear nothing. Set a manual gain."
                )

            # channel_offset_hz is left at its default here on purpose:
            # the branch replaces it on every block with the
            # Doppler-corrected value, and seeding it with a static
            # offset would only invite someone to believe the static one
            # mattered.
            nbfm = NbfmConfig(
                sample_rate_hz=applied.sample_rate_hz,
                if_rate_hz=args.if_rate,
                audio_rate_hz=args.audio_rate,
                deviation_hz=args.deviation,
            )
            if index == 0:
                print(
                    f"Chain:     {applied.sample_rate_hz:,.0f} -> {nbfm.if_rate_hz:,.0f} Hz IF "
                    f"(/{nbfm.channel_decimation_factor}) -> "
                    f"{nbfm.audio_rate_hz:,.0f} Hz audio "
                    f"(/{nbfm.audio_decimation_factor})"
                )
            radios.append(
                _Radio(
                    label=label,
                    sdr=sdr,
                    applied=applied,
                    nbfm=nbfm,
                    # Against the centre the tuner ACTUALLY reached,
                    # never the one it was asked for: the PLL quantises,
                    # and an offset computed from the requested
                    # frequency is wrong by exactly the amount nobody
                    # thinks to check. Per branch for the same reason -
                    # two PLLs quantise differently.
                    doppler=DopplerTracker(downlink_hz, applied.center_hz),
                    # Built unconditionally now - see
                    # _squelch_status_line's own docstring for why.
                    # --squelch only decides whether its decision reaches
                    # the speaker. One per branch: it is stateful, and a
                    # shared one would mix two signals' histories into
                    # one decision.
                    squelch=NoiseSquelch(
                        open_above_db=args.squelch_open, close_below_db=args.squelch_close
                    ),
                )
            )

        print(
            _squelch_status_line(
                mute=args.squelch,
                open_above_db=args.squelch_open,
                close_below_db=args.squelch_close,
            )
        )

        if not args.send:
            if args.track_log is not None:
                # Refused rather than ignored. A log that silently was
                # not written because --send was forgotten is the "off
                # and broken look the same" failure this project keeps
                # meeting, and it would be discovered after the pass.
                print(
                    "--track-log needs --send: without a rotor there is no motion to "
                    "sample, and an empty log discovered after the pass is worse than "
                    "an error before it.",
                    file=sys.stderr,
                )
                return 2
            print(
                "Rotor:     not being moved. Doppler correction needs the TLE and "
                "your location, not the rotor, so this is a complete receive."
            )
            return run(args, config, satellite, radios, listening)

        with _Connected(config, rotor_factory) as rotor:
            print(f"Rotor:     connected, {rotor.firmware_version}")
            profile = _tracking_profile(args, config)
            print(_describe_cadence(profile))
            _push_profile_gains(rotor, profile, config)
            loop = TrackingLoop(
                satellite,
                config.observer,
                rotor,
                interval_s=profile.interval_s,
                deadband_deg=profile.deadband_deg,
                alignment_offset=_alignment_offset(config),
                stall_guard=_stall_guard(config),
                profile=profile,
                on_stall=_report_stall,
                on_profile_change=_profile_pusher(rotor, config),
            )
            return run(args, config, satellite, radios, listening, loop=loop)


def _range_rate_interval(args: argparse.Namespace) -> float:
    """Seconds between Doppler range-rate samples.

    **Deliberately not the tracking profile's interval**, even though
    both were ``--interval`` before profiles existed and the two look
    interchangeable. A tracking profile says how hard to drive the
    *rotor*; this paces :class:`~qsorbit.core.receive.ReceiveSession`'s
    range-rate loop, which runs whether or not a rotor is connected and
    sits on the receive path — where Session 29 established that CPU
    turns directly into lost USB samples.

    Letting a profile move it would mean selecting the tracking profile
    silently doubled the Doppler sampling rate on a run with no rotor
    attached, and that the Rotor tab's live toggle changed the receive
    path mid-pass. Both would land inside Chunk E's combined-versus-
    single comparison as an unmeasured variable.

    ``--interval`` still moves both, because an operator asking for a
    specific tick has asked for exactly that.
    """
    if args.interval is not None:
        return args.interval
    return DEFAULT_TRACKING_INTERVAL_S


def _readout_poll_interval_ms(args: argparse.Namespace, default_ms: int) -> int:
    """Milliseconds between repaints of the live readout.

    The same guard as :func:`_range_rate_interval`, and it is here for
    the same reason: ``--interval`` defaults to ``None`` so that a
    tracking profile can decide the cadence, and every site that reads
    it has to cope with that. This one did not, and
    ``receive --window --send`` without an explicit ``--interval``
    therefore died building the window on ``int(None * 1000)``. Two of
    the three readers were given the guard when profiles landed; this
    was the third.

    **The default is passed in rather than imported.**
    :data:`~qsorbit.ui.readout_widget.DEFAULT_POLL_INTERVAL_MS` lives
    in a module that imports Qt at module scope, and importing it here
    would make the entire CLI unusable anywhere PySide6 is absent --
    the rule ``core/dsp/audio.py`` and ``core/sdr/librtlsdr.py``
    already follow for the same reason. Taking it as an argument also
    keeps this function testable with no Qt at all, and leaves exactly
    one copy of the number.

    Note what this is *not* saying. A readout's repaint rate and a
    rotor's tracking cadence are different quantities that happen to
    have shared a flag; ``--interval`` still moves both, because an
    operator who named a tick asked for exactly that, but the absence
    of one does not mean the absence of the other.

    Args:
        args: Parsed arguments, for ``--interval``.
        default_ms: What to use when ``--interval`` was not given.

    Returns:
        Milliseconds between repaints.
    """
    if args.interval is not None:
        return int(args.interval * 1000)
    return default_ms


def _build_selector(args: argparse.Namespace) -> BranchSelector | None:
    """The combiner, or ``None`` when this run holds one branch fixed."""
    if not args.combine:
        return None
    return BranchSelector(margin_db=args.combine_margin)


def _open_quieting_log(args: argparse.Namespace) -> QuietingLog | None:
    """Open the quieting log if one was asked for, else ``None``.

    Opened before the session is built, so its time origin is fixed
    before any block can arrive and both branches measure elapsed time
    from the same instant.
    """
    if args.quieting_log is None:
        return None
    log = QuietingLog(args.quieting_log)
    log.open()
    return log


def _print_quieting_log(log: QuietingLog | None) -> None:
    """Report the quieting log and close it, if there was one."""
    if log is None:
        return
    print(log.describe())
    log.close()


def _print_recordings(session: ReceiveSession, record_dir: Path | None) -> None:
    """Say where each branch's IQ went, if the run was recording.

    The recorders wrote their own files and sidecars as the run stopped;
    this only reports what landed, so a capture night ends with a visible
    confirmation of exactly which files to trim rather than a silent
    directory the operator has to go and check.
    """
    if record_dir is None:
        return
    recorders = [branch.recorder for branch in session.branches if branch.recorder is not None]
    if not recorders:
        return
    print(f"Recorded IQ to {record_dir}:")
    for recorder in recorders:
        print(
            f"  {recorder.label}: {recorder.bytes_written:,} bytes "
            f"-> {recorder.iq_path.name} (+ {recorder.sidecar_path.name})"
        )


def _print_track_log(ticker: TrackingThread, log: TrackLog | None) -> None:
    """Report the track log and close it, if there was one.

    The line carries the rate the run **achieved** rather than the one
    configured. They differ by design -- a sample due close to a tick
    defers to it -- and the achieved figure is the one that says whether
    the log can resolve the roughly 1 Hz mechanical ring it exists for.
    """
    if log is None:
        return
    line = ticker.describe_log()
    if line is not None:
        print(line)
    log.close()


def _no_tracking_fault() -> BaseException | None:
    """No rotor, so nothing can have stopped ticking one.

    Handed to a feed hub in place of a
    :meth:`~qsorbit.core.tracking_thread.TrackingThread.fault` when the
    run has no rotor attached. A hub asked for a fault source has to get
    a callable either way; "there is no rotor" and "the rotor is fine"
    are different facts, but neither of them is a fault, and the panel
    that would show one is not built at all in this configuration.
    """
    return None


def _report_stall(axes: tuple[str, ...]) -> None:
    """Tell the operator an axis has stopped following, while it matters.

    Printed rather than counted quietly, because unlike most of what a
    track reports this one is **actionable during the pass**: somebody
    can walk out and free the boom, and the track picks up by itself
    when the axis moves. A stall nobody is told about is the silent
    failure the guard exists to end.
    """
    which = " and ".join(axes) if axes else "an axis"
    verb = "are" if len(axes) > 1 else "is"
    # ASCII only. This reaches a Windows console, where the code page
    # turned an em dash into a stray "u" on the bench -- cosmetic, but
    # the one line an operator reads while deciding whether to walk out
    # to the rotor should not look like it has been corrupted.
    print(
        f"STALL:     {which} {verb} not following the commanded position. The "
        "setpoint is frozen, so the antenna is not being driven further from "
        "where it actually is. Check for an obstruction or a cable snag - "
        "tracking resumes on its own once the axis moves."
    )


def _describe_cadence(profile: TrackingProfile) -> str:
    """One line saying what this cadence will really command.

    The step is reported, not just the deadband, because they are not
    the same number and the difference is the whole point: the shipped
    2.5 deg deadband at a 1 s tick commands 3.0 deg steps against a
    1 deg/s target, which is a thing an operator should be able to read
    off the screen rather than off a logic analyser.
    """
    return (
        f"Cadence:   {profile.name} profile, {profile.deadband_deg:g} deg deadband "
        f"at {profile.interval_s:g} s -> {profile.commanded_step_deg:g} deg steps "
        f"at {NOMINAL_TRACKING_RATE_DEG_S:g} deg/s"
    )


def _describe_mechanics(capabilities: RotorCapabilities) -> list[str]:
    """What this rotor's measured mechanics allow, as status lines.

    Reports the **headroom** rather than only the measurements, because
    the measurements alone do not answer the question an operator has.
    "Azimuth free play 2.95 deg" is a fact; "the most integral gain this
    axis can safely take is 0.98" is the fact they can act on, and it is
    the difference between reading a number and knowing what it costs.

    Empty when nothing is measured, with a line saying so — an absent
    measurement is a state to report, not a blank to skip past.
    """
    if not capabilities.mechanics_measured:
        return [
            "Mechanics: not measured (see [rotor.capabilities] in config.example.toml). "
            "No profile may run a non-zero Ki until they are."
        ]
    lines = []
    for axis in ("azimuth", "elevation"):
        free_play, breakaway = capabilities.mechanics_for(axis)
        safe_ki = max_safe_ki(free_play, breakaway)
        label = "Mechanics:" if axis == "azimuth" else " " * 10
        lines.append(
            f"{label} {axis[:2].upper()} free play {free_play:.2f} deg, "
            f"breakaway {breakaway:g} PWM -> max safe Ki {safe_ki:.2f} "
            f"at {DESIGN_RATE_DEG_S:g} deg/s"
        )
    return lines


def _stall_guard(config: StationConfig) -> StallGuard:
    """The stall detector, sized to this rotor's measured free play.

    The detector's gate and the gain clamp are the same physical
    number, so it is read from the same place. Leaving the guard on its
    compiled default while the clamp read config would be two constants
    that must agree, written down twice — which
    :mod:`qsorbit.core.tracking_profile` already warns is two constants
    that will eventually disagree.

    **The larger axis wins.** The guard carries one figure for both
    axes, and a gate sized to the tighter axis would call the sloppier
    one stalled every time it took up its own slack.
    """
    capabilities = config.capabilities
    if not capabilities.mechanics_measured:
        return StallGuard()
    azimuth, _ = capabilities.mechanics_for("azimuth")
    elevation, _ = capabilities.mechanics_for("elevation")
    return StallGuard(free_play_deg=max(azimuth, elevation))


def _push_profile_gains(
    rotor: Rotor,
    profile: TrackingProfile,
    config: StationConfig,
    *,
    report_in_force: bool = True,
) -> None:
    """Write this profile's gains to the controller, verified, and say so.

    **Called when a track starts, not when the port opens.** Gains are
    RAM-only, so re-pushing at every connect is tempting — it would
    guarantee the controller always matched config. It would also mean
    ``qsorbit status`` writes tuning to a motor controller, and a
    read-only command that quietly changes the hardware is a thing
    nobody should have to remember. The accepted cost is that a power
    cycle mid-session reverts the controller to its compiled defaults
    with nothing noticing until the next track — which is exactly why
    :meth:`~qsorbit.core.rotor.Rotor.push_gains` verifies every register
    rather than writing blind.

    The clamp is re-checked here even though
    :func:`~qsorbit.core.station.load_station_config` already checked
    it, for the same reason the cadence check runs at both ends: a
    profile can reach this point from ``--rotor-profile`` or from a
    direct construction, and this is the last place before the wire.

    **A profile that writes nothing gets its gains read back and
    reported, rather than described.** This line used to say the
    controller "keeps its compiled defaults", which is a claim about
    hardware that nothing had checked -- and false on any controller
    written to since power-on, because gains are RAM-only and survive a
    disconnect. It was wrong for a whole 543-second track on this
    station: an aborted push had left Ki live while the console asserted
    stock. That matters beyond tidiness, since Chunk H's entire windup
    argument rests on ``stock`` meaning Ki = 0, and the acceptance pass
    is a stock-versus-tracking comparison whose baseline has to be real.

    Args:
        rotor: The controller to write to.
        profile: The profile whose gains to push.
        config: For the breakaway clamp.
        report_in_force: Whether to read the registers back and report
            them when the profile writes none. ``True`` at the start of
            a run, where a second of round trips buys a recorded
            baseline. **``False`` on a live profile toggle**, where the
            operator just chose the profile and a second of stalled
            ticking mid-pass costs more than the information is worth --
            and where paying it on only one of the two directions would
            make the switch asymmetric inside the very measurement the
            toggle exists to serve.
    """
    gains = profile.gains
    if gains is None:
        if not report_in_force:
            print(f"Gains:     {profile.name} profile writes none")
            return
        in_force = ", ".join(
            f"{register.name.lower()} {value:g}" for register, value in rotor.read_gains().items()
        )
        print(f"Gains:     {profile.name} profile writes none - controller is running {in_force}")
        return
    profile.check_against(config.capabilities)
    rotor.push_gains(gains)
    written = ", ".join(f"{register.name.lower()} {value:g}" for register, value in gains.items())
    print(f"Gains:     {profile.name} profile pushed and verified - {written}")
    print(
        f"           RAM only: a power cycle reverts these, and they are checked "
        f"against this rotor's breakaway at {DESIGN_RATE_DEG_S:g} deg/s"
    )


def _profile_pusher(rotor: Rotor, config: StationConfig) -> Callable[[TrackingProfile], None]:
    """The callback the tracking loop runs when a profile switch lands.

    Handed to :class:`~qsorbit.core.pointing.TrackingLoop` so the push
    happens **on whichever thread ticks the loop**, which is the thread
    that already owns the serial port. That is the whole reason the
    switch is queued rather than applied from the button: no lock, no
    second owner, and it works the same under the GUI ticker and under
    ``ReceiveSession``'s tracking thread.

    Nothing is caught. A :class:`~qsorbit.core.rotor.GainVerificationError`
    means the controller is running a gain mixture nobody chose, so it
    propagates out of ``tick()`` and stops the run -- Phil's call, over
    carrying on with an unknown tuning that would misattribute every
    measurement taken afterwards.
    """

    def push(profile: TrackingProfile) -> None:
        # No read-back here: see report_in_force. A live toggle must not
        # stall the tick for a second, and must not stall it on only one
        # of the two directions.
        _push_profile_gains(rotor, profile, config, report_in_force=False)

    return push


def _tracking_profile(args: argparse.Namespace, config: StationConfig) -> TrackingProfile:
    """The tracking profile in force, after command-line overrides.

    The single place cadence is resolved, so ``receive``, ``shell`` and
    ``status`` cannot disagree about what this run is actually doing.

    Both overrides go through :func:`dataclasses.replace`, which re-runs
    the value object's own validation — so ``--rotor-profile`` naming a
    profile that isn't declared, and an ``--interval`` that lands the
    cadence on the knife edge, are both refused here exactly as they
    would be in the config file. A check that only guards the config
    file is a check with a command-line-shaped hole in it.
    """
    tracking = config.tracking
    if args.rotor_profile is not None:
        tracking = replace(tracking, active=args.rotor_profile)
    profile = tracking.active_profile(config.capabilities.acceptance_window_deg)
    if args.interval is not None:
        profile = replace(profile, interval_s=args.interval)
    return profile


def _iq_recorder_factory(
    radio: _Radio,
    record_dir: Path,
    station_hz: float | None,
    stream: IqStream,
) -> Callable[[IqSubscription], IqRecorder]:
    """A recorder-building closure for one radio, for :class:`Branch`.

    Built here, where the device's actual settings and the downlink are
    in hand, and applied to the branch's own recorder subscription. The
    loss it reports at the end is the *stream's* USB loss, read when the
    sidecar is written -- a callable, not a stored value, because that
    number is only final once the run has stopped.
    """

    def factory(subscription: IqSubscription) -> IqRecorder:
        return IqRecorder(
            subscription=subscription,
            iq_path=record_dir / (safe_filename(radio.label) + ".iq"),
            applied=radio.applied,
            device_description=radio.sdr.info.describe() if radio.sdr.info else "unknown device",
            station_hz=station_hz,
            loss_source=lambda: stream.stats.loss,
            label=radio.label,
        )

    return factory


def _build_branches(
    args: argparse.Namespace,
    radios: Sequence[_Radio],
    listening: int,
    *,
    window: bool,
    log: QuietingLog | None = None,
    record_dir: Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> list[Branch]:
    """Turn configured radios into receive branches.

    Only the listening branch is given a spectrum factory, so the
    waterfall shows what is being heard and a branch nobody is watching
    never computes frames nobody sees.

    Args:
        args: The parsed command line, for the demod and squelch
            settings shared by every branch.
        radios: One per configured device, in declaration order.
        listening: Index of the branch whose audio reaches the speaker.
        window: Whether anything will drain spectrum frames. ``receive``
            passes ``--window``; the shell passes ``True`` always,
            because a shell has a Radio tab in every configuration.
        log: Optional quieting log, shared by every branch. One log and
            not one per branch, because the whole point is a difference
            between two series and two files would be two time origins.
        record_dir: Optional directory to record every branch's raw IQ
            into. When given, each branch takes a recorder consumer
            writing ``<antenna>.iq`` and its sidecar; ``None`` records
            nothing and adds no consumer, so an ordinary run is unchanged.
    """
    spectrum_config = SpectrumConfig(
        fft_size=RECEIVE_FFT_SIZE,
        sample_rate_hz=radios[listening].applied.sample_rate_hz,
        center_freq_hz=radios[listening].applied.center_hz,
    )
    factory = _spectrum_factory(window, spectrum_config)
    # The downlink, in Hz, so each sidecar records where in its own
    # capture the signal sits -- the same station_hz sdr capture records.
    station_hz = args.downlink * 1e6 if args.downlink is not None else None
    branches: list[Branch] = []
    for index, radio in enumerate(radios):
        # On a replay, every branch's stream reads the capture's clock, so
        # block timestamps run at the capture's date; live runs pass None
        # and IqStream keeps its own wall clock.
        stream = IqStream(radio.sdr, now=clock) if clock is not None else IqStream(radio.sdr)
        recorder_factory = (
            _iq_recorder_factory(radio, record_dir, station_hz, stream)
            if record_dir is not None
            else None
        )
        branches.append(
            Branch(
                label=radio.label,
                stream=stream,
                nbfm=radio.nbfm,
                doppler=radio.doppler,
                squelch=radio.squelch,
                # args.squelch is "let the gate's decision reach the
                # speaker" - the gate itself is always deciding now, see
                # _squelch_status_line.
                mute_squelch=args.squelch,
                spectrum_factory=factory if index == listening else None,
                log=log,
                recorder_factory=recorder_factory,
            )
        )
    return branches


def _run_receive(
    args: argparse.Namespace,
    config: StationConfig,
    satellite: Satellite,
    radios: Sequence[_Radio],
    listening: int,
    *,
    loop: TrackingLoop | None = None,
) -> int:
    """Build and run the session. Split out so the rotor's ``with`` stays thin."""
    track_log = TrackLog(args.track_log) if args.track_log is not None else None
    if track_log is not None:
        track_log.open()
    ticker = TrackingThread(loop, log=track_log) if loop is not None else None
    quieting_log = _open_quieting_log(args)
    record_dir = Path(args.record_iq) if args.record_iq is not None else None
    replay_clock = _replay_clock_for(radios, listening)
    _print_replay_banner(radios)

    session = ReceiveSession(
        branches=_build_branches(
            args,
            radios,
            listening,
            window=args.window,
            log=quieting_log,
            record_dir=record_dir,
            clock=replay_clock,
        ),
        audio=AudioOutput(
            radios[listening].nbfm.audio_rate_hz, device=_parse_audio_device(args.audio_device)
        ),
        # Always from the target, never from the rotor. A range rate
        # comes from the TLE and the observer's location; taking it from
        # a rotor tick was what made the rotor follow this cadence
        # instead of its own profile's. On a replay it reads the capture's
        # clock, so the Doppler curve runs at the capture's date.
        range_rate=_range_rate_source(satellite, config.observer, replay_clock),
        listening=listening,
        selector=_build_selector(args),
        tracking_interval_s=_range_rate_interval(args),
    )

    print(
        "Receiving - Ctrl-C to stop."
        if args.seconds is None
        else f"Receiving for {args.seconds:.0f}s."
    )

    # Subscribed before the worker starts, so neither panel misses a
    # frame and no implicit default is ever made. Two independent feeds:
    # before this, both panels drained one shared buffer and whichever
    # timer fired first took the batch, which is why they alternated on
    # the bench (verification #11, Session 24).
    #
    # The wiring lives here rather than in ReceiveSession because this is
    # where the widgets are built. The session stays ignorant of which
    # panels exist, and Chunk C's feed hub replaces these four lines
    # without touching either widget.
    feeds = (
        (
            session.spectrum.subscribe(WATERFALL_FEED),
            session.spectrum.subscribe(SPECTRUM_LINE_FEED),
        )
        if session.spectrum is not None
        else None
    )

    # Built before the session starts, and the ordering is the point of
    # the call rather than an accident of it. The window used to be built
    # after the session had already started, so QApplication and five
    # widgets were constructed while the SDR reader thread was already
    # streaming -- and Session 24 measured the cost of that
    # exactly once per run, in thirteen windowed runs across two commits,
    # indoors and out, from 31 s to 1261 s: a single ~1.03 s stall
    # accounting for essentially the whole of the reported USB loss,
    # against 0.0018% for the same code headless. Standing up the
    # graphics stack holds the GIL long enough to starve a thread that is
    # mid-read, and the fix is to have finished doing it before there is
    # a thread to starve.
    #
    # The session is therefore started *by* _show_instruments, through
    # the callback below, rather than here. Handing the window builder a
    # `start` rather than splitting it into build-then-run is deliberate:
    # an InstrumentWindow is a top-level widget with no Qt parent, so the
    # only thing keeping it alive is a Python reference, and a builder
    # that returns leaves that reference nowhere. Keeping `window` a
    # local of the frame that is still executing `app.exec()` makes the
    # lifetime structural instead of something a later edit can drop.
    def start_everything() -> None:
        # The rotor first, and synchronously: its first tick points the
        # antenna before anything is streaming, and a rotor that cannot
        # be reached at all should stop the run here rather than half a
        # second into the pass. Then the radio.
        if ticker is not None:
            ticker.start()
        session.start()

    try:
        if args.window:
            _show_instruments(
                args,
                satellite,
                session,
                loop,
                feeds,
                start=start_everything,
                tracking_fault=ticker.fault if ticker is not None else _no_tracking_fault,
            )
        else:
            start_everything()
            if session.wait(args.seconds):
                # The demodulating thread ended before the clock did,
                # which means the blocks stopped arriving. Sleeping out
                # the rest of the run would have hidden that behind a
                # normal-looking exit; session.stop() below raises
                # whatever caused it.
                print("The stream ended early - see the error below.", file=sys.stderr)
    except KeyboardInterrupt:
        print()  # the ^C the terminal echoed deserves its own line
    finally:
        if ticker is not None:
            ticker.stop()
        stats = session.stop()

    print()
    print(stats.describe())
    _print_quieting_log(quieting_log)
    _print_recordings(session, record_dir)
    if ticker is not None:
        print(ticker.describe())
        _print_track_log(ticker, track_log)
    return 0


def _window_title(satellite: Satellite, session: ReceiveSession) -> str:
    """The window's title, naming the branch being heard when it is fixed.

    Three cases, and the third was a defect before it was a case.

    One branch: no label, because there is no choice to report and
    naming the only antenna would imply one.

    Two branches with ``--listen``: named, because otherwise the flag
    has no visible effect at all -- both branches are tuned to the same
    downlink, so one's waterfall is pixel-for-pixel the other's and
    nothing on screen could say which radio you were hearing.

    **Two branches with the combiner running: not named.** A title is
    evaluated once when the window is built, and the combiner moves the
    ear within the first second; the first version of this function
    named the starting branch and then went on asserting it for the rest
    of the run, which is worse than the silence it replaced -- a wrong
    antenna stated confidently. Making the title follow the ear was the
    other option and is worse again: a window title that changes under
    you while you are reading it. The per-branch meters carry the live
    answer, and they carry it in the one place that can update.
    """
    if len(session.branches) < 2:
        return f"QSOrbit - receiving {satellite.name}"
    if session.selector is not None:
        return f"QSOrbit - receiving {satellite.name} (combining)"
    return f"QSOrbit - receiving {satellite.name} on {session.listening.label}"


def _show_instruments(
    args: argparse.Namespace,
    satellite: Satellite,
    session: ReceiveSession,
    loop: TrackingLoop | None,
    feeds: tuple[SpectrumSubscription, SpectrumSubscription] | None = None,
    *,
    start: Callable[[], None],
    tracking_fault: Callable[[], BaseException | None] = _no_tracking_fault,
) -> None:
    """Build the instrument window, start the session, then run Qt's loop.

    **The session is started from inside this function, after the window
    exists and before the event loop runs**, and that ordering is the
    point rather than an accident of it. Everything expensive about Qt --
    constructing QApplication, loading the platform plugin, building five
    widgets, realising a native window -- happens first, so the graphics
    stack is standing before there is a reader thread for it to starve.
    See :func:`_run_receive` for the measurement behind that.

    Taking a ``start`` callback rather than being split into a builder
    and a runner is also deliberate, and was learned the hard way. A
    builder that returns hands its caller a closure while ``window``
    itself goes out of scope -- and an :class:`InstrumentWindow` is a
    top-level widget with no Qt parent, so that Python reference is the
    only thing keeping it alive. Losing it collects the window the
    instant the builder returns: it flashes on screen, disappears,
    destroys every panel's ``QTimer`` with it, and leaves an event loop
    running with nothing in it. Keeping ``window`` a local of the frame
    that is still executing ``app.exec()`` makes that lifetime
    structural.

    **Qt is imported here and nowhere above**, because importing PySide6
    at module scope would make the entire CLI unusable anywhere it is not
    installed. Same reasoning as ``core/dsp/audio.py`` deferring
    ``sounddevice`` and ``core/sdr/librtlsdr.py`` deferring its
    ``CDLL()`` — a dependency that can fail merely by being imported
    belongs inside the function that actually needs it.

    The readout is present only when there is a loop to show. It
    **follows** that loop rather than ticking it, and the fault source it
    is handed comes from whatever *does* tick -- a
    :class:`~qsorbit.core.tracking_thread.TrackingThread` when a rotor is
    attached. That indirection is the point: a rotor fault has to reach
    the screen rather than freezing the last good numbers there, and the
    readout should not have to know which object owns the cadence to find
    out. It used to be ``session.tracking_error``, back when the receive
    session's own range-rate thread held the tick; that method still
    exists and now means something narrower.

    **Ctrl-C is handled explicitly** via :func:`_quit_on_sigint` wrapping
    ``app.exec()`` below — see that function's own docstring for why
    Qt's event loop needs this rather than Python's default handling.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from qsorbit.core.dsp.spectrum import frequency_axis_hz
    from qsorbit.ui.instrument_window import InstrumentWindow
    from qsorbit.ui.quieting_widget import QuietingWidget
    from qsorbit.ui.readout_widget import DEFAULT_POLL_INTERVAL_MS, ReadoutWidget
    from qsorbit.ui.spectrum_line_widget import SpectrumLineWidget
    from qsorbit.ui.theme_manager import ThemeManager
    from qsorbit.ui.waterfall_render import WaterfallScale
    from qsorbit.ui.waterfall_widget import WaterfallWidget
    from qsorbit.ui.zoom_controller import ZoomController
    from qsorbit.ui.zoom_controls_widget import ZoomControlsWidget

    app = QApplication.instance() or QApplication([])

    # Themed before a single widget is constructed. The order matters:
    # apply() sets the application-wide stylesheet and palette, and a
    # widget built beforehand would be styled on its first repaint
    # rather than on creation - which shows up as a visible flash of
    # unthemed chrome on a slow start.
    themes = ThemeManager.discover()
    try:
        themes.apply(getattr(args, "theme", None) or DEFAULT_THEME_NAME)
    except KeyError:
        # A bad --theme is worth a word rather than a traceback: the
        # window is still perfectly usable in the default theme, and
        # refusing to open a receiver over a misspelt colour scheme
        # would be the wrong trade during a pass.
        print(
            f"Unknown theme {args.theme!r}; using {DEFAULT_THEME_NAME}. "
            f"Available: {', '.join(themes.slugs)}.",
            file=sys.stderr,
        )
        themes.apply(DEFAULT_THEME_NAME)

    if feeds is None:  # pragma: no cover - guarded by args.window at the call site
        raise RuntimeError("The instrument window needs spectrum feeds to draw from.")
    waterfall_feed, line_feed = feeds

    # One ZoomController shared by the line-spectrum panel and the
    # waterfall, so a mouse or spinbox gesture on either moves both -
    # see that class's own docstring. `session` itself satisfies
    # TrackedFrequencySource (its live_tracked_frequency_hz property),
    # so the controller can poll it and drive the lock without either
    # spectrum widget needing to know tracking exists.
    # From a feed rather than from the stream: a subscription reports
    # the framing it is handing over, so the axis and the frames cannot
    # come from two objects that quietly disagree.
    axis = frequency_axis_hz(line_feed.config)
    zoom_controller = ZoomController(
        float(axis[0]), float(axis[-1]), tracked_frequency_source=session
    )
    # Shared too, so the line trace and the waterfall's colour ramp agree
    # on what "loud" means - defaults are the same numbers either way,
    # but a caller who ever customizes one should get both for free.
    scale = WaterfallScale()

    window = InstrumentWindow(
        readout=(
            ReadoutWidget(
                loop,
                fault=tracking_fault,
                poll_interval_ms=_readout_poll_interval_ms(args, DEFAULT_POLL_INTERVAL_MS),
            )
            if loop is not None
            else None
        ),
        # Always attached, regardless of --squelch: the session's squelch
        # is now unconditional (see _squelch_status_line), so there is
        # always a live reading to show, muted or not.
        quieting=QuietingWidget(session),
        zoom_controls=ZoomControlsWidget(zoom_controller),
        spectrum_line=SpectrumLineWidget(
            line_feed, themes=themes, zoom=zoom_controller, scale=scale
        ),
        waterfall=WaterfallWidget(waterfall_feed, themes=themes, zoom=zoom_controller, scale=scale),
        zoom_controller=zoom_controller,
        themes=themes,
        title=_window_title(satellite, session),
    )
    window.show()

    # Only now, with Qt fully up and the window realised, does anything
    # begin streaming.
    start()

    if args.seconds is not None:
        # Honoured rather than ignored: a --seconds that silently did
        # nothing under --window is precisely the sort of quiet
        # disagreement between a flag and its behaviour this project
        # keeps finding the hard way.
        #
        # Started after the session rather than before it so the countdown
        # measures the receiving, not the receiving plus however long the
        # session took to come up.
        QTimer.singleShot(int(args.seconds * 1000), app.quit)

    with _quit_on_sigint(app):
        app.exec()

    # After the loop, so the window still exists to be walked. The
    # control run reports this as well: a number from the shell with
    # nothing to compare it against would be half a measurement.
    _print_paint_stats(window)


def _print_paint_stats(window: object) -> None:
    """Report what every waterfall in ``window`` spent repainting.

    Walks the widget tree rather than being handed the panels, the same
    way the shell's own shutdown does, so a window with two waterfalls
    in it -- a Custom tab duplicating the Radio tab's, once PR3 lands --
    reports both without this function being told they exist.

    Printed beside the receive statistics because that is the only place
    the two can be compared. A paint cost with no USB-loss figure next
    to it, or the reverse, is what left a 28x regression invisible until
    somebody happened to maximize the window.
    """
    from qsorbit.ui.waterfall_widget import WaterfallWidget

    panels = window.findChildren(WaterfallWidget)
    for panel in panels:
        stats = panel.paint_stats
        if stats.paints:
            print()
            print(stats.describe())


def _command_shell(
    args: argparse.Namespace,
    config: StationConfig,
    rotor_factory: RotorFactory,
    sdr_factory: SdrFactory,
) -> int:
    """Open the shell, with as much hardware behind it as was asked for.

    Three modes, and the checks below are what keep them from blurring
    into each other:

    ``shell``
        No radio, no rotor. Every tab opens and says what it is waiting
        for. Useful for looking at themes, and it is what an evening
        with no sky looks like.
    ``shell --tle X --send``
        Rotor only. The antenna follows the target and the Radio tab
        stays in placeholder.
    ``shell --tle X --downlink M --gain G``
        The full receive chain, plus the rotor with ``--send``.

    **The gain check is here rather than in argparse**, because the
    option is only conditionally required: ``receive`` must have one and
    ``shell`` need not, so argparse cannot enforce it for both. The
    message is worth the special case -- a default gain is how a pass
    comes back silent with nobody noticing, which is the same reason
    ``sdr capture`` refuses to invent one.
    """
    # Checked before the mode dispatch below, because two of the three
    # modes never reach the code that builds a log -- and a flag that is
    # accepted, ignored, and discovered to have written nothing after
    # the pass is the failure this option exists to prevent.
    if args.track_log is not None:
        if args.tle is None or not args.send:
            print(
                "shell: --track-log needs --tle and --send. Without a rotor "
                "following something there is no motion to sample, and an empty "
                "log discovered after the pass is worse than an error before it.",
                file=sys.stderr,
            )
            return 1
    simulated_at = None
    if args.at is not None:
        if args.downlink is not None:
            print(
                "shell: --at is rotor-only and cannot be combined with --downlink. "
                "It simulates the pass geometry for the rotor to follow; a live "
                "radio would still be receiving now, so its Doppler would be "
                "computed for a time the signal is not at. Simulate the signal with "
                "a replay instead.",
                file=sys.stderr,
            )
            return 1
        if args.tle is None or not args.send:
            print(
                "shell: --at needs --tle and --send -- it simulates a pass for the "
                "rotor to follow, and without them there is nothing to point at.",
                file=sys.stderr,
            )
            return 1
        try:
            simulated_at = _parse_time(args.at)
        except ValueError as exc:
            print(f"shell: {exc}", file=sys.stderr)
            return 1
    if args.record_iq is not None and args.downlink is None:
        print(
            "shell: --record-iq needs --downlink. It records a radio's raw IQ, "
            "and a rotor-only shell has no radio to record.",
            file=sys.stderr,
        )
        return 1
    if args.tle is None:
        if args.downlink is not None or args.send:
            print(
                "shell: --downlink and --send both need --tle - they describe what "
                "to follow, and there is nothing to follow yet.",
                file=sys.stderr,
            )
            return 1
        return _run_shell_alone(args, config)

    if args.downlink is None:
        if not args.send:
            print(
                "shell: --tle with no --downlink means rotor-only, which needs "
                "--send. Add --downlink to receive instead.",
                file=sys.stderr,
            )
            return 1
        return _run_shell_tracking_only(args, config, rotor_factory, simulated_at=simulated_at)

    if args.gain is None and not args.auto_gain:
        print(
            "shell: --downlink needs --gain (or --auto-gain). There is "
            "deliberately no default: a default gain is how a pass comes back "
            "silent with nobody noticing.",
            file=sys.stderr,
        )
        return 1

    return _command_receive(args, config, rotor_factory, sdr_factory, runner=_run_shell)


def _shell_theme(args: argparse.Namespace) -> object:
    """Build a theme manager and apply the requested theme.

    **Applied before a single widget is constructed**, which is the same
    ordering PR1 settled: ``apply()`` sets the application-wide
    stylesheet and palette, and a widget built beforehand is styled on
    its first repaint rather than on creation -- visible as a flash of
    unthemed chrome on a slow start.

    A bad ``--theme`` is worth a word rather than a traceback. Refusing
    to open over a misspelt colour scheme would be the wrong trade at
    any time, and a spectacularly wrong one during a pass.
    """
    from qsorbit.ui.theme_manager import ThemeManager

    themes = ThemeManager.discover()
    try:
        themes.apply(getattr(args, "theme", None) or DEFAULT_THEME_NAME)
    except KeyError:
        print(
            f"Unknown theme {args.theme!r}; using {DEFAULT_THEME_NAME}. "
            f"Available: {', '.join(themes.slugs)}.",
            file=sys.stderr,
        )
        themes.apply(DEFAULT_THEME_NAME)
    return themes


def _shell_custom_tab(args: argparse.Namespace) -> tuple[object | None, str | None]:
    """Load ``custom_tab.toml`` if present, and say what a bad one broke.

    Mirrors :func:`_shell_theme`'s shape but not its severity. A bad
    ``--theme`` still opens the shell with a sane fallback; a bad
    Custom tab config still has to leave every *other* tab alone, so
    the failure is not raised here at all -- it is turned into the pair
    this returns and the caller hands straight to
    :class:`~qsorbit.ui.shell_window.ShellWindow`, which shows it only
    in the Custom tab. "Off" (no file yet) and "broken" (a file that
    failed to load) are kept distinguishable the whole way down, per
    this project's standing rule.

    Returns:
        ``(config, None)`` if a valid file was found, ``(None, None)``
        if there is no file at :func:`~qsorbit.ui.custom_tab.custom_tab_config_path`,
        or ``(None, message)`` if a file was found but failed to load.
    """
    from qsorbit.ui.custom_tab import (
        CustomTabConfigError,
        custom_tab_config_path,
        load_custom_tab_config,
    )

    path = custom_tab_config_path()
    if not path.is_file():
        return None, None
    try:
        return load_custom_tab_config(path), None
    except CustomTabConfigError as exc:
        print(f"shell: {exc}", file=sys.stderr)
        return None, str(exc)


def _exec_shell(window: object, app: object, args: argparse.Namespace) -> None:
    """Show the window, honour ``--seconds``, and run Qt's loop.

    **Ctrl-C works here even with nothing attached**, and that is a
    property worth naming rather than assuming. Qt's event loop can only
    be interrupted at a Python bytecode boundary, so it needs some
    Python callback still running -- Session 25 lost Ctrl-C entirely
    when a lifetime bug killed every panel timer. An empty shell has no
    panels at all, and what keeps it interruptible is the top bar's
    one-second clock, which runs in every configuration. Anything that
    ever stops that timer takes Ctrl-C with it.
    """
    from PySide6.QtCore import QTimer

    # show(), not showMaximized(), and the difference was measured
    # rather than debated. Maximizing this window costs 28x the USB
    # loss on the receive path -- 0.7444% against 0.0262% windowed,
    # reproduced across two maximized runs -- because
    # WaterfallWidget.paintEvent scales its whole history image to the
    # widget's size on every one of its 20 repaints a second, and that
    # scale is the one part of its cost that grows with the window.
    # The inefficiency predates the shell (`receive --window` pays the
    # same tax over fewer pixels), so the fix belongs to the waterfall
    # and gets its own before/after measurement. Opening maximized
    # comes back once it is free.
    window.show()
    if args.seconds is not None:
        QTimer.singleShot(int(args.seconds * 1000), app.quit)
    with _quit_on_sigint(app):
        app.exec()


def _shell_planning_catalog() -> tuple[ProfileCatalog, CatalogManifest | None]:
    """Load the curated catalogue and its manifest for the Plan tab.

    Same defaults ``qsorbit plan`` reads with no ``--profiles-dir`` --
    ``shell`` has no such flag of its own (yet); the shipped starter set
    is what every entry point without an override reads. Shared by all
    three shell-launching functions below so the load happens once per
    call path rather than being copied three times. A load failure
    propagates as :class:`~qsorbit.core.profiles.ProfileError`, caught
    by :func:`main`'s own handler exactly as it already is for ``plan``.
    """
    return load_profile_catalog(), load_catalog_manifest()


def _run_shell_alone(args: argparse.Namespace, config: StationConfig) -> int:
    """The shell with no hardware behind it.

    Radio and Rotor open in placeholder -- there is no SDR and no rotor
    connection at all in this mode. Plan does not: the target picker
    needs a curated catalogue and a TLE directory, neither of which is
    hardware, so it lights up here exactly as it would with a rotor and
    a radio attached, using whatever ``config.planning.tle_dir`` this
    station has set.
    """
    from PySide6.QtWidgets import QApplication

    from qsorbit.ui.feed_hub import FeedHub
    from qsorbit.ui.shell_window import ShellWindow

    app = QApplication.instance() or QApplication([])
    themes = _shell_theme(args)
    hub = FeedHub()
    print(hub.describe())
    print("Nothing attached. Pass --tle to track, and --downlink to receive.")
    custom_tab, custom_tab_error = _shell_custom_tab(args)
    catalog, manifest = _shell_planning_catalog()
    window = ShellWindow(
        hub,
        themes=themes,
        custom_tab=custom_tab,
        custom_tab_error=custom_tab_error,
        catalog=catalog,
        catalog_manifest=manifest,
        tle_dir=config.planning.tle_dir,
        observer=config.observer,
        horizon=config.horizon,
    )
    _exec_shell(window, app, args)
    return 0


def _run_shell_tracking_only(
    args: argparse.Namespace,
    config: StationConfig,
    rotor_factory: RotorFactory,
    *,
    simulated_at: datetime | None = None,
) -> int:
    """The shell with a rotor and no radio.

    The mirror image of ``receive``'s standing asymmetry: that command
    runs the whole radio job with nothing on the serial port, and this
    runs the whole pointing job with nothing on USB. On a bench evening
    where several things could be wrong at once, neither fault costs you
    the other half.

    Unlike ``receive``, **this ticks the loop from the GUI thread**, and
    that is a deliberate exception rather than a regression of Chunk A's
    fix. The fix moved the tick off the GUI thread because it was
    competing with a live SDR reader for the GIL and stalling it; with
    no reader running there is nothing to stall, and the rotor's own
    measured cost is 18-30 ms over a whole pass. Standing up a thread
    and a stop protocol to move a 30 ms budget off an idle event loop
    would be machinery for its own sake -- but if a radio is ever added
    to this path, the tick moves with it.
    """
    from PySide6.QtWidgets import QApplication

    from qsorbit.ui.feed_hub import FeedHub
    from qsorbit.ui.shell_window import ShellWindow

    satellite = Satellite.from_file(args.tle)
    app = QApplication.instance() or QApplication([])
    themes = _shell_theme(args)

    # --at: run the pointing job against a shifted clock so the rotor
    # follows a chosen pass without waiting for it. The clock drives the
    # loop's target computation only; TrackingThread still schedules in
    # real time, so the pass unfolds over its real duration. Marked
    # loudly here, in the track log, and in the closing report, because a
    # simulated run that reads as a live one is the "reports state it
    # does not own" failure at the level of the whole run.
    now_fn = None
    track_marker = None
    if simulated_at is not None:
        now_fn = _offset_clock(simulated_at)
        offset_s = (simulated_at - datetime.now(UTC)).total_seconds()
        print(
            f"*** SIMULATED PASS: clock set to {simulated_at.isoformat()} "
            f"({offset_s:+.0f} s from now). The geometry and the rotor motion are "
            "real; the time is not. This is NOT a live pass. ***"
        )
        note = _below_horizon_note(satellite, config.observer, simulated_at, args.at)
        if note is not None:
            print(note, file=sys.stderr)
        track_marker = (
            f"SIMULATED pass: clock set to {simulated_at.isoformat()} at launch; not a live pass"
        )

    with _Connected(config, rotor_factory) as rotor:
        print(f"Rotor:     connected, {rotor.firmware_version}")
        profile = _tracking_profile(args, config)
        print(_describe_cadence(profile))
        _push_profile_gains(rotor, profile, config)
        loop_clock = {"now": now_fn} if now_fn is not None else {}
        loop = TrackingLoop(
            satellite,
            config.observer,
            rotor,
            interval_s=profile.interval_s,
            deadband_deg=profile.deadband_deg,
            alignment_offset=_alignment_offset(config),
            stall_guard=_stall_guard(config),
            profile=profile,
            on_stall=_report_stall,
            on_profile_change=_profile_pusher(rotor, config),
            **loop_clock,
        )
        track_log = (
            TrackLog(args.track_log, preamble_comment=track_marker)
            if args.track_log is not None
            else None
        )
        if track_log is not None:
            track_log.open()
        # The ticker is built BEFORE the hub, because the hub needs its
        # fault callable. A readout left showing its last plausible
        # numbers under a dead rotor is the exact silent failure Chunk A
        # PR2 added `tracking_error` to prevent, and it would be a poor
        # joke to reintroduce it in the shell.
        #
        # No interval is passed: the loop already holds one, and giving
        # this object a second copy is two sources for a number that
        # must agree -- which a live profile switch would then have to
        # keep in step in two places.
        ticker = TrackingThread(loop, log=track_log)
        hub = FeedHub(
            tracking=loop,
            tracking_fault=ticker.fault,
            tracking_profiles=config.tracking.profiles,
        )
        print(hub.describe())
        custom_tab, custom_tab_error = _shell_custom_tab(args)
        catalog, manifest = _shell_planning_catalog()
        window = ShellWindow(
            hub,
            themes=themes,
            custom_tab=custom_tab,
            custom_tab_error=custom_tab_error,
            catalog=catalog,
            catalog_manifest=manifest,
            tle_dir=config.planning.tle_dir,
            observer=config.observer,
            horizon=config.horizon,
            title=f"QSOrbit - tracking {satellite.name}",
        )
        try:
            # Started before the window is shown, and unlike the receive
            # path that ordering carries no weight here: there is no SDR
            # reader thread for Qt's startup to starve, which is the
            # same reason this path was allowed a GUI-thread ticker for
            # as long as it was.
            ticker.start()
            _exec_shell(window, app, args)
        finally:
            ticker.stop()

    print()
    print(ticker.describe())
    _print_track_log(ticker, track_log)
    if simulated_at is not None:
        print("(SIMULATED pass -- the clock was offset; the timing above is not live.)")
    return 0


def _receive_shell_hub(
    session: ReceiveSession,
    loop: TrackingLoop | None,
    ticker: TrackingThread | None,
    tracking_profiles: tuple[TrackingProfile, ...],
) -> FeedHub:
    """Build the feed hub for the receive-path shell.

    Extracted so this path and the rotor-only path
    (:func:`_run_shell_tracking_only`) cannot silently disagree about
    what the hub carries -- which they did. This one was constructed
    inline without ``tracking_profiles``, so the Rotor tab reported
    "declare at least two profiles" on a station that declares three, on
    the one configuration Chunk E's acceptance runs in (receive path,
    rotor attached), while the rotor-only path was wired correctly. A
    named builder is one place to get the wiring right and one place to
    test it.
    """
    from qsorbit.ui.feed_hub import FeedHub

    return FeedHub(
        spectrum=session.spectrum,
        radio=session,
        tracking=loop,
        tracking_fault=ticker.fault if ticker is not None else _no_tracking_fault,
        tracking_profiles=tracking_profiles,
        # Every branch, not just the one being heard. `radio=session`
        # already publishes the listening branch's levels; a per-branch
        # display needs the ones nobody is listening to, because those
        # are where a fade shows up before the combiner acts on it.
        branches=session.branches,
    )


def _run_shell(
    args: argparse.Namespace,
    config: StationConfig,
    satellite: Satellite,
    radios: Sequence[_Radio],
    listening: int,
    *,
    loop: TrackingLoop | None = None,
) -> int:
    """Build the session, wrap it in a hub, and run the shell over it.

    The same shape as :func:`_run_receive`, and the ordering inside it is
    the same ordering for the same measured reason: **the window is
    built before anything streams.** Session 24 found a single ~1.03 s
    stall in every windowed run, in thirteen runs across two commits,
    caused by standing up QApplication and five widgets while the SDR
    reader thread was already going; Session 25 fixed it by swapping two
    lines and measured it down to 0.001-0.015 s. The session is
    therefore started *by* this function after the window exists, and
    ``window`` is kept as a local of a frame that is still executing
    ``app.exec()`` -- a top-level widget with no Qt parent is kept alive
    by exactly one Python reference, and a builder that returned would
    drop it.

    What is genuinely new is that **the widgets no longer subscribe to
    anything**. ``_run_receive`` claims two spectrum feeds by hand and
    passes them down; here the hub does it, from inside whichever tab
    wants one, and this function never learns how many feeds were
    claimed or by whom. That is the difference between a window that
    knows its panels and a shell that does not have to.
    """
    from PySide6.QtWidgets import QApplication

    from qsorbit.ui.shell_window import ShellWindow

    track_log = TrackLog(args.track_log) if args.track_log is not None else None
    if track_log is not None:
        track_log.open()
    ticker = TrackingThread(loop, log=track_log) if loop is not None else None
    quieting_log = _open_quieting_log(args)
    record_dir = Path(args.record_iq) if args.record_iq is not None else None
    replay_clock = _replay_clock_for(radios, listening)
    _print_replay_banner(radios)
    session = ReceiveSession(
        # window=True unconditionally here, unlike `receive`, where it
        # follows --window: a shell always has a Radio tab, so there is
        # always something that would drain the frames.
        branches=_build_branches(
            args,
            radios,
            listening,
            window=True,
            log=quieting_log,
            record_dir=record_dir,
            clock=replay_clock,
        ),
        audio=AudioOutput(
            radios[listening].nbfm.audio_rate_hz, device=_parse_audio_device(args.audio_device)
        ),
        range_rate=_range_rate_source(satellite, config.observer, replay_clock),
        listening=listening,
        selector=_build_selector(args),
        tracking_interval_s=_range_rate_interval(args),
    )

    app = QApplication.instance() or QApplication([])
    themes = _shell_theme(args)
    hub = _receive_shell_hub(session, loop, ticker, config.tracking.profiles)
    print(hub.describe())
    print(
        "Receiving - Ctrl-C to stop."
        if args.seconds is None
        else f"Receiving for {args.seconds:.0f}s."
    )

    custom_tab, custom_tab_error = _shell_custom_tab(args)
    catalog, manifest = _shell_planning_catalog()
    window = ShellWindow(
        hub,
        themes=themes,
        nominal_hz=args.downlink * 1e6,
        custom_tab=custom_tab,
        custom_tab_error=custom_tab_error,
        catalog=catalog,
        catalog_manifest=manifest,
        tle_dir=config.planning.tle_dir,
        observer=config.observer,
        horizon=config.horizon,
        title=_window_title(satellite, session),
    )
    # show(), not showMaximized(), and the difference was measured
    # rather than debated. Maximizing this window costs 28x the USB
    # loss on the receive path -- 0.7444% against 0.0262% windowed,
    # reproduced across two maximized runs -- because
    # WaterfallWidget.paintEvent scales its whole history image to the
    # widget's size on every one of its 20 repaints a second, and that
    # scale is the one part of its cost that grows with the window.
    # The inefficiency predates the shell (`receive --window` pays the
    # same tax over fewer pixels), so the fix belongs to the waterfall
    # and gets its own before/after measurement. Opening maximized
    # comes back once it is free.
    window.show()

    try:
        # Only now, with Qt fully up and the window realised, does
        # anything begin streaming. The rotor goes first, for the same
        # reason it does in _run_receive.
        if ticker is not None:
            ticker.start()
        session.start()
        if args.seconds is not None:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(int(args.seconds * 1000), app.quit)
        with _quit_on_sigint(app):
            app.exec()
    except KeyboardInterrupt:
        print()  # the ^C the terminal echoed deserves its own line
    finally:
        if ticker is not None:
            ticker.stop()
        stats = session.stop()

    print()
    print(stats.describe())
    _print_quieting_log(quieting_log)
    _print_recordings(session, record_dir)
    if ticker is not None:
        print(ticker.describe())
        _print_track_log(ticker, track_log)
    print(f"feeds: {', '.join(hub.claimed) or 'none claimed'}")
    _print_paint_stats(window)
    return 0


def _command_stop(config: StationConfig, factory: RotorFactory) -> int:
    with _Connected(config, factory) as rotor:
        position = rotor.stop()
        print(f"Stopped:   {_format_position(position)}")
        print(
            "The setpoint is now the current position. This halts a converging "
            "move only - it cannot stop a diverging one. Use the power switch "
            "for that."
        )
        return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Connected:
    """Connects a rotor on entry and closes it on exit.

    A thin wrapper rather than ``with Rotor(...)`` directly, so the
    factory stays injectable and homing progress reaches the terminal.
    """

    def __init__(self, config: StationConfig, factory: RotorFactory) -> None:
        self._config = config
        self._factory = factory
        self._rotor: Rotor | None = None

    def __enter__(self) -> Rotor:
        rotor = self._factory(self._config, _report_homing_wait)
        self._rotor = rotor
        rotor.connect()
        return rotor

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._rotor is not None:
            self._rotor.close()


def _open_rotor(config: StationConfig, on_homing_wait: Callable[[float], None]) -> Rotor:
    """Build a :class:`Rotor` over a real serial port."""
    port = SerialPort(
        config.serial.port,
        baudrate=config.serial.baudrate,
        timeout=config.serial.timeout_s,
    )
    return Rotor(port, config.capabilities, on_homing_wait=on_homing_wait)


@dataclass(frozen=True)
class _Radio:
    """One configured device and everything built against it.

    A staging record between :func:`_command_receive`, which opens and
    tunes the hardware, and the runner, which builds the
    :class:`~qsorbit.core.receive.Branch` objects. Every field here is
    per-device, and two of them look shareable and are not:
    ``doppler`` is built against the centre frequency *this* tuner
    reached, and ``squelch`` is stateful.
    """

    label: str
    sdr: RtlSdr
    applied: AppliedSettings
    nbfm: NbfmConfig
    doppler: DopplerTracker
    squelch: NoiseSquelch


def _declared_branches(config: StationConfig) -> tuple[SdrBranch | None, ...]:
    """The branches to open, or one ``None`` for a station with none declared.

    The ``None`` is what keeps a single-device station on the same code
    path as a dual-SDR one. A separate branchless route would leave the
    path every existing station takes as the one exercised least.
    """
    return config.sdr.branches or (None,)


def _branch_label(branch: SdrBranch | None) -> str:
    """What to call a branch in reports and on screen."""
    return UNNAMED_BRANCH_LABEL if branch is None else branch.label


def _branch_ppm(config: StationConfig, branch: SdrBranch | None) -> int:
    """The crystal correction for one branch's dongle."""
    return config.sdr.ppm if branch is None else config.sdr.ppm_for(branch)


def _listening_index(listen: str | None, branches: tuple[SdrBranch | None, ...]) -> int:
    """Which branch gets the speaker, from ``--listen``.

    Args:
        listen: The requested branch label, or ``None`` for the first.
        branches: What the station declared.

    Returns:
        An index into ``branches``.

    Raises:
        ValueError: If ``listen`` names no declared branch. The message
            lists the labels, because a label with a space in it is easy
            to mistype and the config file is the only place they exist.
    """
    if listen is None:
        return 0
    labels = [_branch_label(branch) for branch in branches]
    if listen in labels:
        return labels.index(listen)
    offered = ", ".join(repr(label) for label in labels)
    raise ValueError(
        f"--listen {listen!r} matches no declared branch. This station has: {offered}. "
        "Labels come from [[sdr.branch]] in your station config and are matched exactly."
    )


def _active_branch(config: StationConfig) -> SdrBranch | None:
    """The branch a single-device command works with, or ``None``.

    The first declared one. Everything in this module still opens
    exactly one device; the branch list is what says *which*, and the
    order it is written in is what ranks them.
    """
    return config.sdr.branches[0] if config.sdr.branches else None


def _sdr_ppm(config: StationConfig) -> int:
    """The crystal correction for whichever device this run will open.

    Asked as its own question rather than read off ``[sdr] ppm``
    directly, because with branches declared the two are not the same
    number, and a correction taken from the wrong dongle is a tuning
    error nothing downstream reports.
    """
    branch = _active_branch(config)
    return config.sdr.ppm if branch is None else config.sdr.ppm_for(branch)


def _note_single_device(config: StationConfig) -> None:
    """Say which branch a one-device command took, when there is a choice.

    Said out loud rather than assumed: on a station cabled for two,
    silence would look exactly like both being used. Printed by the
    commands that open one device, not by :func:`_open_sdr`, which is a
    factory and has no business writing to the console — and which the
    receive path calls once per branch, where the note would be false.
    """
    branch = _active_branch(config)
    if branch is None or len(config.sdr.branches) < 2:
        return
    print(
        f"Note: {len(config.sdr.branches)} branches are declared and this command "
        f"opens one. Using {branch.label} (serial {branch.serial})."
    )


def _parse_replay_map(spec: str) -> dict[str | None, str]:
    """Parse ``--replay`` into a map of branch label to ``.iq`` path.

    ``LABEL=FILE,LABEL=FILE`` maps by label; a bare ``FILE`` (no ``=``)
    maps the single unnamed branch, keyed by ``None``. Mixing the two, an
    empty spec, or a half-written entry is a usage error rather than a
    guess.
    """
    entries = [part.strip() for part in spec.split(",") if part.strip()]
    mapping: dict[str | None, str] = {}
    for entry in entries:
        if "=" in entry:
            label, _, path = entry.partition("=")
            if not label.strip() or not path.strip():
                raise ValueError(f"--replay entry {entry!r} must be LABEL=FILE.iq.")
            mapping[label.strip()] = path.strip()
        else:
            mapping[None] = entry
    if not mapping:
        raise ValueError("--replay needs at least one FILE.iq.")
    if None in mapping and len(mapping) > 1:
        raise ValueError(
            "--replay: give one bare FILE.iq for a single unnamed branch, or "
            "LABEL=FILE.iq per branch, but not both."
        )
    return mapping


def _parse_attenuation_map(spec: str) -> dict[str | None, float]:
    """Parse ``--replay-attenuate`` into a map of branch label to dB.

    Same shape as :func:`_parse_replay_map` -- ``LABEL=DB`` per branch, or a
    bare ``DB`` for a single unnamed branch -- but the values are decibels,
    and a value that is not a number is a usage error.
    """
    entries = [part.strip() for part in spec.split(",") if part.strip()]
    mapping: dict[str | None, float] = {}
    for entry in entries:
        label: str | None
        if "=" in entry:
            raw_label, _, raw_db = entry.partition("=")
            label = raw_label.strip()
            raw_db = raw_db.strip()
            if not label or not raw_db:
                raise ValueError(f"--replay-attenuate entry {entry!r} must be LABEL=DB.")
        else:
            label, raw_db = None, entry
        try:
            mapping[label] = float(raw_db)
        except ValueError:
            raise ValueError(f"--replay-attenuate: {raw_db!r} is not a number of dB.") from None
    if not mapping:
        raise ValueError("--replay-attenuate needs at least one LABEL=DB.")
    if None in mapping and len(mapping) > 1:
        raise ValueError("--replay-attenuate: give one bare DB or LABEL=DB per branch, not both.")
    return mapping


def _replay_sdr_factory(
    replay_map: dict[str | None, str],
    attenuation_map: dict[str | None, float] | None = None,
) -> SdrFactory:
    """An :data:`SdrFactory` that returns a :class:`ReplaySdr` per branch.

    Matches each branch by its config label (``None`` for a single
    unnamed branch); a branch with no file in the map is refused rather
    than opened as a live radio, so a typo can't quietly half-replay.
    ``attenuation_map`` pads a branch down by its dB for the known-answer
    control; a branch not in it is replayed at full amplitude.
    """
    attenuation = attenuation_map or {}

    def factory(config: StationConfig, branch: SdrBranch | None = None) -> RtlSdr:
        key = branch.label if branch is not None else None
        if key not in replay_map:
            declared = ", ".join(sorted(k for k in replay_map if k is not None)) or "a bare file"
            raise SdrError(f"--replay has no file for branch {key!r}; it maps: {declared}.")
        return ReplaySdr(replay_map[key], attenuation_db=attenuation.get(key, 0.0))

    return factory


def _replay_clock_for(radios: Sequence[_Radio], listening: int) -> Callable[[], datetime] | None:
    """The shared capture-epoch clock for a replay run, or ``None`` if live.

    Taken from the listening branch's :class:`ReplaySdr`, so block
    timestamps and the Doppler curve run at the capture's date. One clock
    for the whole session -- two branches replaying the same file then
    stamp identical timestamps, which is the identical-input control's
    whole premise.
    """
    device = radios[listening].sdr
    if isinstance(device, ReplaySdr):
        return device.capture_clock()
    return None


def _range_rate_source(
    satellite: Satellite, observer: ObserverLocation, clock: Callable[[], datetime] | None
) -> TargetRangeRate:
    """The range-rate source, reading a replay's clock when one is given.

    A live run uses the wall clock; a replay uses the capture's, so the
    Doppler curve is computed for the pass that was recorded rather than
    for whenever the replay happens to run.
    """
    if clock is not None:
        return TargetRangeRate(satellite, observer, now=clock)
    return TargetRangeRate(satellite, observer)


def _print_replay_banner(radios: Sequence[_Radio]) -> None:
    """Announce a replay run and what each file actually is, if replaying."""
    replays = [radio for radio in radios if isinstance(radio.sdr, ReplaySdr)]
    if not replays:
        return
    print(
        "*** REPLAY: reading captured IQ, not a live receive. Rate and tuning "
        "come from each file's own sidecar. ***"
    )
    for radio in replays:
        applied = radio.sdr.applied
        if applied is not None:
            print(
                f"    {radio.label}: {applied.sample_rate_hz / 1e6:.3f} Msps, "
                f"centre {applied.center_hz / 1e6:.4f} MHz"
            )


def _open_sdr(config: StationConfig, branch: SdrBranch | None = None) -> RtlSdr:
    """Build an :class:`RtlSdr` from the station's ``[sdr]`` settings.

    Args:
        config: The station.
        branch: Which declared branch to open. ``None`` means "whichever
            this station's single-device commands use" — the first
            declared branch if there are any, the configured index
            otherwise.

    Addresses the device by EEPROM serial when branches are declared,
    and by index otherwise — the second path is what every config
    written before branches existed still takes.

    The resolution happens here, at the composition root, rather than
    inside :class:`RtlSdr`: an index is only true for as long as the bus
    does not change, so it is worked out fresh on every open and never
    written down. The caller's ``with`` opens the device an instant
    later and nothing re-checks in between, so a dongle unplugged inside
    that window would be opened by a stale index. That gap is real and
    is left open deliberately — closing it means opening the device
    here, outside the context manager that owns its lifetime.
    """
    if branch is None:
        branch = _active_branch(config)
    if branch is None:
        return RtlSdr(config.sdr.device_index, driver_dir=config.sdr.driver_dir)
    lib = LibRtlSdr.load(config.sdr.driver_dir)
    return RtlSdr(index_for_serial(lib, branch.serial), driver_dir=config.sdr.driver_dir)


def _report_homing_wait(elapsed_s: float) -> None:
    """Say something while the controller is deaf, so it doesn't look hung."""
    print(f"Waiting for the controller ({elapsed_s:.0f}s) - it is deaf while homing.")


def _below_horizon_note(
    satellite: Satellite,
    observer: ObserverLocation,
    simulated_at: datetime,
    at_arg: str,
) -> str | None:
    """Warn if a simulated target starts below the horizon, else ``None``.

    Extracted from :func:`_run_shell_tracking_only` (which needs Qt) so
    this check runs under a plain unit test. The version this replaces
    read ``state.elevation`` and crashed: elevation lives on
    ``state.sky_position`` (an :class:`~qsorbit.core.geometry.AzEl`), the
    same access the ``describe`` command uses. Below the horizon the loop
    commands nothing until the satellite rises, so the operator gets told
    rather than watching a still rotor and wondering.

    Args:
        satellite: The target whose geometry is being simulated.
        observer: The station location the elevation is measured from.
        simulated_at: The clock instant the run is pinned to at launch.
        at_arg: The raw ``--at`` string, echoed into the suggested
            ``plan`` command so the hint is copy-pasteable.

    Returns:
        The warning to print to stderr, or ``None`` if the target is at
        or above the horizon.
    """
    elevation = satellite.topocentric_state(observer, simulated_at).sky_position.elevation
    if elevation >= 0.0:
        return None
    return (
        f"Note: {satellite.name} is below the horizon at the simulated time "
        f"(elevation {elevation:.1f} deg), so nothing is commanded until it "
        f"rises. Pick an --at during a pass; `qsorbit plan --at {at_arg}` shows "
        "what is up then."
    )


def _offset_clock(simulated_at: datetime) -> Callable[[], datetime]:
    """A wall clock shifted so it reads ``simulated_at`` at first call.

    The offset is fixed when this is built, so the returned clock
    advances at real time from the simulated instant -- a pass unfolds
    over its real duration, just shifted in when it happens. ``shell
    --at`` uses it to drive the tracking loop's target computation while
    the tick scheduling stays on the real clock, so the rotor follows a
    chosen pass without waiting for it to come round.
    """
    offset = simulated_at - datetime.now(UTC)

    def _now() -> datetime:
        return datetime.now(UTC) + offset

    return _now


def _parse_time(text: str | None) -> datetime:
    """Parse an ISO 8601 instant, defaulting to now and assuming UTC.

    Args:
        text: The ``--at`` value, or ``None`` for "now".

    Returns:
        A timezone-aware datetime.

    Raises:
        ValueError: If ``text`` isn't a recognizable ISO 8601 instant.
    """
    if text is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Could not read {text!r} as a time. Use ISO 8601, for example "
            "2026-08-20T18:30:00+00:00."
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _format_position(position: Position) -> str:
    """Format a rotor axis position for display."""
    return f"AZ {position.azimuth:.1f}  EL {position.elevation:.1f}"


def _range_description(range_rate_km_s: float) -> str:
    """Describe a range rate in words, since the sign alone is easy to misread."""
    if range_rate_km_s > 0:
        return f"receding at {range_rate_km_s:.3f} km/s"
    if range_rate_km_s < 0:
        return f"approaching at {abs(range_rate_km_s):.3f} km/s"
    return "range steady"


def _describe_error(error: RotorErrorCode) -> str:
    """Describe an error code, with the operator's next step where there is one."""
    if error is RotorErrorCode.NO_ERROR:
        return "none"
    if error is RotorErrorCode.HOMING_ERROR:
        return "homing failed - power-cycle the controller; nothing over serial clears it"
    return error.name.lower().replace("_", " ")


if __name__ == "__main__":
    sys.exit(main())
