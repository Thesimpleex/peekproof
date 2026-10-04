"""peekproof: detect look-ahead leakage in backtests and research code.

Two complementary tools:

- the **oracle** (:func:`check`, :func:`check_label`, :func:`assert_causal`)
  re-evaluates a feature function on truncated and perturbed histories and
  reports whether its output at time *t* depends on rows after *t*;
- the **linter** (:func:`lint_path`, :func:`lint_source`) finds the usual
  leak patterns in pandas, polars, numpy and scikit-learn research code
  without executing it.
"""

from __future__ import annotations

from peekproof.errors import LeakError, PeekproofError
from peekproof.lint import RULES, lint_path, lint_source
from peekproof.oracle import assert_causal, check, check_label
from peekproof.report import Finding, LeakFinding, OracleReport

__version__ = "0.1.0"

__all__ = [
    "RULES",
    "Finding",
    "LeakError",
    "LeakFinding",
    "OracleReport",
    "PeekproofError",
    "__version__",
    "assert_causal",
    "check",
    "check_label",
    "lint_path",
    "lint_source",
]
