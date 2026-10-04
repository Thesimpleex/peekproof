"""Command-line interface.

    peekproof lint PATH [PATH ...]
    peekproof check MODULE:FUNCTION --data FILE

Exit codes: 0 nothing to report, 1 findings at or above ``--fail-on`` or a
leak found, 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from peekproof import __version__
from peekproof.errors import PeekproofError
from peekproof.lint import lint_path
from peekproof.oracle import check, check_label
from peekproof.report import CONFIDENCE_ORDER, Finding, OracleReport, findings_to_json

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="peekproof",
        description="Detect look-ahead leakage in backtests and research code.",
    )
    parser.add_argument("--version", action="version", version=f"peekproof {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    lint = sub.add_parser("lint", help="static analysis of Python files and notebooks")
    lint.add_argument("paths", nargs="+", help="files or directories to lint")
    lint.add_argument("--select", help="comma-separated rule ids to enable (prefixes allowed)")
    lint.add_argument("--ignore", help="comma-separated rule ids to disable")
    lint.add_argument(
        "--min-confidence",
        choices=("low", "medium", "high"),
        default=None,
        help="report findings at or above this confidence (default: config value, else low)",
    )
    lint.add_argument("--format", choices=("text", "json"), default="text")
    lint.add_argument(
        "--fail-on",
        choices=("high", "medium", "low", "never"),
        default="high",
        help="exit with status 1 if any finding is at or above this confidence",
    )
    lint.add_argument("--config", help="explicit pyproject.toml to read [tool.peekproof] from")

    chk = sub.add_parser("check", help="run the dynamic causality oracle on a function")
    chk.add_argument("target", help="MODULE:FUNCTION, imported from the current directory")
    chk.add_argument("--data", required=True, help="CSV or Parquet file with a time index")
    index_group = chk.add_mutually_exclusive_group()
    index_group.add_argument("--index-col", help="column to use as the index")
    index_group.add_argument("--time-col", help="time column: parsed as dates, sorted, indexed")
    chk.add_argument("--cutoffs", type=int, default=None, help="number of cutoffs (default: 64)")
    chk.add_argument("--mode", choices=("truncate", "perturb", "both"), default="both")
    chk.add_argument(
        "--label-horizon",
        type=int,
        help="treat the output as a label allowed to look this many rows ahead",
    )
    chk.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _load_function(target: str) -> Callable[[Any], Any]:
    if ":" not in target:
        raise PeekproofError("target must be MODULE:FUNCTION, for example features:momentum")
    module_name, func_name = target.split(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise PeekproofError(
            f"could not import module '{module_name}': {type(exc).__name__}: {exc}"
        ) from exc
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise PeekproofError(f"'{func_name}' is not a callable in module '{module_name}'")
    return fn


def _read_table(file: Path, index_col: str | None, time_col: str | None) -> pd.DataFrame:
    suffix = file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file, index_col=index_col if time_col is None else None)
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(file)
        if index_col is not None and time_col is None:
            if index_col not in df.columns:
                raise PeekproofError(f"index column '{index_col}' not found in {file}")
            df = df.set_index(index_col)
        return df
    raise PeekproofError(f"unsupported data format '{suffix}'; use .csv or .parquet")


def _load_data(path: str, index_col: str | None, time_col: str | None) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        raise PeekproofError(f"data file not found: {path}")
    try:
        df = _read_table(file, index_col, time_col)
    except PeekproofError:
        raise
    except Exception as exc:
        raise PeekproofError(f"could not read {path}: {type(exc).__name__}: {exc}") from exc
    if time_col is not None:
        if time_col not in df.columns:
            raise PeekproofError(f"time column '{time_col}' not found in {path}")
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except (TypeError, ValueError) as exc:
            raise PeekproofError(f"time column '{time_col}' is not parseable as dates") from exc
        df = df.sort_values(time_col).set_index(time_col)
    elif not isinstance(df.index, pd.DatetimeIndex) and not pd.api.types.is_numeric_dtype(df.index):
        # Text labels that look like dates become a DatetimeIndex; integer row numbers stay.
        with contextlib.suppress(TypeError, ValueError):
            df.index = pd.to_datetime(df.index)
    return df


def _run_lint(args: argparse.Namespace) -> int:
    findings: list[Finding] = lint_path(
        args.paths,
        select=args.select,
        ignore=args.ignore,
        min_confidence=args.min_confidence,
        config_path=args.config,
    )
    if args.format == "json":
        print(findings_to_json(findings))
    else:
        for f in findings:
            print(f.format())
        n = len(findings)
        print(f"peekproof: {n} finding{'s' if n != 1 else ''}")
    if args.fail_on == "never":
        return EXIT_CLEAN
    threshold = CONFIDENCE_ORDER[args.fail_on]
    hit = any(CONFIDENCE_ORDER[f.confidence] >= threshold for f in findings)
    return EXIT_FINDINGS if hit else EXIT_CLEAN


def _run_check(args: argparse.Namespace) -> int:
    fn = _load_function(args.target)
    data = _load_data(args.data, args.index_col, args.time_col)
    options: dict[str, Any] = {"mode": args.mode}
    if args.cutoffs is not None:
        options["cutoffs"] = args.cutoffs
    report: OracleReport
    if args.label_horizon is not None:
        report = check_label(fn, data, horizon=args.label_horizon, **options)
    else:
        report = check(fn, data, **options)
    print(report.to_json() if args.format == "json" else str(report))
    return EXIT_CLEAN if report.causal is True else EXIT_FINDINGS


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "lint":
            return _run_lint(args)
        return _run_check(args)
    except PeekproofError as exc:
        print(f"peekproof: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
