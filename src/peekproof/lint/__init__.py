"""Static linter for look-ahead leakage and selection bias in research code.

Use :func:`lint_path` for files and folders and :func:`lint_source` for a
string. Both return a list of :class:`peekproof.report.Finding`. ``RULES`` maps
rule ids (``"PP001"`` to ``"PP013"``) to the rule objects.
"""

from __future__ import annotations

from peekproof.lint.engine import lint_path, lint_source
from peekproof.lint.rules import RULES

__all__ = ["RULES", "lint_path", "lint_source"]
