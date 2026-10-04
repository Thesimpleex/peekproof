"""Result containers shared by the oracle and the linter.

Plain dataclasses with JSON rendering. Keeping them in one module lets the CLI
and the test-suite depend on a single, stable shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Confidence = Literal["low", "medium", "high"]
LeakKind = Literal["value_change", "missing_row"]
TestKind = Literal["truncate", "perturb"]

CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

__all__ = [
    "CONFIDENCE_ORDER",
    "Confidence",
    "Finding",
    "LeakFinding",
    "LeakKind",
    "OracleReport",
    "TestKind",
    "findings_to_json",
]


@dataclass(frozen=True)
class LeakFinding:
    """One piece of evidence that a function used information from after the cutoff."""

    column: str
    kind: LeakKind
    test: TestKind
    cutoff_pos: int
    cutoff_label: str
    n_rows_affected: int
    first_affected_label: str
    last_affected_label: str
    max_abs_diff: float | None = None


@dataclass
class OracleReport:
    """Outcome of :func:`peekproof.check` / :func:`peekproof.check_label`."""

    causal: bool | None
    deterministic: bool
    n_rows: int
    n_evaluations: int
    cutoff_positions: list[int] = field(default_factory=list)
    findings: list[LeakFinding] = field(default_factory=list)
    horizon_rows: int | None = None
    horizon_unbounded: bool = False
    notes: list[str] = field(default_factory=list)
    label_horizon: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def affected_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for f in self.findings:
            seen.setdefault(f.column, None)
        return list(seen)

    def __str__(self) -> str:
        if not self.deterministic:
            head = "peekproof: UNDETERMINED (function is not deterministic)"
            lines = [head, *(f"  note: {n}" for n in self.notes)]
            return "\n".join(lines)
        verdict = "CAUSAL" if self.causal else "NOT CAUSAL"
        head = (
            f"peekproof: {verdict}  (deterministic, {len(self.cutoff_positions)} cutoffs, "
            f"{self.n_evaluations} evaluations, {self.n_rows} rows)"
        )
        lines = [head]
        if self.label_horizon is not None:
            lines.append(f"  label mode: allowed horizon = {self.label_horizon} row(s)")
        for col in self.affected_columns:
            col_findings = [f for f in self.findings if f.column == col]
            kinds = sorted({f.kind for f in col_findings})
            cutoffs = sorted({f.cutoff_pos for f in col_findings})
            diffs = [f.max_abs_diff for f in col_findings if f.max_abs_diff is not None]
            first = min(col_findings, key=lambda f: f.cutoff_pos).first_affected_label
            diff_txt = f", max |diff| = {max(diffs):.3g}" if diffs else ""
            lines.append(
                f"  column '{col}': {len(cutoffs)} cutoff(s) affected, kind={'/'.join(kinds)}"
                f"{diff_txt}"
            )
            lines.append(f"    first affected row {first}")
        if self.causal is False:
            if self.horizon_unbounded:
                horizon_txt = "unbounded (whole-sample dependence)"
            elif self.horizon_rows is not None:
                horizon_txt = f"{self.horizon_rows} row(s)"
            else:
                horizon_txt = "unknown"
            lines.append(
                f"  verdict: output at t depends on rows after t, horizon = {horizon_txt}."
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Finding:
    """One static-analysis finding produced by the linter."""

    rule_id: str
    name: str
    message: str
    hint: str
    path: str
    line: int
    col: int
    confidence: Confidence
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def format(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.col}: {self.rule_id} [{self.confidence}] "
            f"{self.message}\n    hint: {self.hint}"
        )


def findings_to_json(findings: list[Finding], indent: int | None = 2) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=indent)
