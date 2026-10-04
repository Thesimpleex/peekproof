"""Exception types used across peekproof."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peekproof.report import OracleReport

__all__ = ["LeakError", "PeekproofError"]


class PeekproofError(Exception):
    """Base class for errors raised by peekproof itself (bad input, unsupported output)."""


class LeakError(AssertionError):
    """Raised by :func:`peekproof.assert_causal` when a function is not causal.

    Subclasses ``AssertionError`` so that it integrates with test runners. The
    full report is available as :attr:`report`; ``str(error)`` is the rendered
    report.
    """

    def __init__(self, report: OracleReport) -> None:
        self.report = report
        super().__init__(str(report))
