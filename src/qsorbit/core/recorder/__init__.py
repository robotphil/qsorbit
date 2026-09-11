"""In-session IQ recording.

The package the roadmap reserved for the dual capture. Its one job is to
write a branch's raw samples to a file, in the same format
:mod:`qsorbit.core.sdr.capture` produces, while a receive session runs --
so a capture is a *consumer* on each branch's stream rather than a second
process fighting the first for the dongle.
"""

from __future__ import annotations

from qsorbit.core.recorder.iq_recorder import IqRecorder, safe_filename

__all__ = ["IqRecorder", "safe_filename"]
