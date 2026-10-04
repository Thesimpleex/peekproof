"""Dynamic causality oracle.

A function ``f`` that maps a time-indexed table to a time-indexed output is
*causal* when the output at time t never depends on rows after t. The oracle
tests this by running ``f`` on the full data, on shortened copies and on copies
whose later rows were replaced, and by comparing the output up to the cutoff
with the output on the full data.

The oracle tests; it does not prove. A report with ``causal=True`` means that no
difference was found at the cutoffs that were tried, not that none exists. A
leak that touches only a few rows can fall between the cutoffs. The default of
64 cutoffs per test makes that unlikely, and the report says so when the
cutoffs stop before the end of the data, but it stays possible.

The oracle also sees only the dependence on the data that it passes to the
function. Anything the function reaches by another route is invisible to it:
module-level variables and caches, frames captured from an enclosing scope,
files read inside the function, and objects fitted beforehand, such as a scaler
fitted on the whole sample outside the function. A function that uses such
state can be reported as causal although it leaks. It is reported as not
deterministic only if that state changes its output between two calls on the
same data.

The functions in this module are pure: they print nothing, keep no state
between calls, and give the same report for the same inputs and ``seed``.

Cost: ``check`` calls ``fn`` twice for the determinism check, then once per
cutoff of each test that runs. With the default of 64 cutoffs per test and
``mode="both"`` that is ``2 + 64 + 64`` calls. Every call gets its own copy of
the data, so a function that modifies its argument in place cannot affect the
caller's data or the next call.

Outputs: the index of the output must be a subset of the input index. For an
input with a ``DatetimeIndex`` it may instead be a monotonic ``DatetimeIndex``
of other labels, for example the bar labels of a resampled series; the output is
then compared by label, and rows that are labeled after the cutoff are ignored.

polars: for a polars input the function receives polars ``DataFrame`` objects
(with the time column restored) and must return a polars ``DataFrame`` that
keeps the time column.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from peekproof._compat import (
    PolarsAdapter,
    column_atol,
    compare_columns,
    format_label,
    frames_equal,
    label_at,
    polars_adapter,
    robust_scale,
    to_pandas,
)
from peekproof.errors import LeakError, PeekproofError
from peekproof.report import LeakFinding, LeakKind, OracleReport, TestKind

if TYPE_CHECKING:
    import polars as pl

__all__ = ["assert_causal", "check", "check_label"]

Mode = Literal["truncate", "perturb", "both"]
TimeUnit = Literal["s", "ms", "us", "ns"]
#: The function under test. It gets a pandas object (a polars ``DataFrame`` for polars input).
CausalFn = Callable[[Any], Any]

_TESTS: dict[str, tuple[TestKind, ...]] = {
    "truncate": ("truncate",),
    "perturb": ("perturb",),
    "both": ("truncate", "perturb"),
}
_ADJECTIVE: dict[TestKind, str] = {"truncate": "truncated", "perturb": "perturbed"}

_DEFAULT_CUTOFFS = 64  # cutoffs per test
_MIN_CUTOFF = 2  # shortest history that is ever tried, in rows
_LOW_FRACTION = 0.1  # default shortest truncated history, as a share of all rows
_EARLY_FRACTION = 0.05  # the "first 5 % of rows" of the unbounded-horizon rule
_T_DEGREES = 3  # degrees of freedom of the perturbation noise
_FLIP_PROBABILITY = 0.5  # share of boolean values that the perturbation flips
_MAX_SPAN = 2**62  # widest integer range that the perturbation shifts by modular arithmetic
_CUTOFF_STREAM = 1  # random streams: one per purpose, so that they do not share numbers
_PERTURB_STREAM = 2
_TEST_NUMBER: dict[TestKind, int] = {"truncate": 0, "perturb": 1}
_NOT_DETERMINISTIC = "function is not deterministic; seed it before testing"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def check(
    fn: CausalFn,
    data: pd.DataFrame | pd.Series | pl.DataFrame,
    *,
    cutoffs: int | Iterable[int] = _DEFAULT_CUTOFFS,
    mode: Mode = "both",
    atol: float = 1e-9,
    rtol: float = 1e-7,
    time_col: str | None = None,
    seed: int = 0,
    min_rows: int | None = None,
) -> OracleReport:
    """Test whether ``fn`` is causal on ``data``.

    The result is evidence, not a proof: the oracle looks for a difference at
    the cutoffs it tries and can miss a leak that lies between them. It only
    measures how the output depends on the data that it passes to ``fn``. State
    that ``fn`` reaches in other ways (module-level variables and caches, frames
    from an enclosing scope, files, a scaler or any object fitted beforehand) is
    invisible to it, so a function that leaks through such state can still be
    reported as causal.

    ``fn`` receives a pandas object like ``data`` (for polars input: a polars
    ``DataFrame`` that has the time column ``time_col``) and returns a
    ``DataFrame`` or a ``Series`` whose index is a subset of the index it
    received (for polars: a ``DataFrame`` that keeps the time column). A
    ``Series`` result is treated as a one-column frame named after the series,
    or ``"value"`` if it has no name. For an input with a ``DatetimeIndex`` the
    output index may also be a monotonic ``DatetimeIndex`` of labels that are not
    in the input, such as the bar labels of ``resample``. Such an output is
    matched by label and only the rows labeled at or before the cutoff are
    compared; rows that the shortened run labels after the cutoff are ignored.
    A bar labeled by its start (``label="left"``) holds rows after that label
    and is therefore found to leak; a bar labeled by its end is causal.

    The algorithm:

    1. Determinism. ``fn(data)`` is evaluated twice. If the outputs differ the
       report has ``deterministic=False`` and ``causal=None`` and no leak test
       runs.
    2. Cutoffs. A cutoff position ``p`` keeps rows ``0 .. p - 1``; the cutoff
       time is the label of row ``p - 1``.
    3. Truncation test. ``fn`` is evaluated on the first ``p`` rows. Its output
       is compared with the full output on the rows up to the cutoff.
    4. Perturbation test. Every row from ``p`` on is changed and ``fn`` is
       evaluated on the result, which has the full length. Floats get
       Student-t noise (3 degrees of freedom) scaled by a robust measure of the
       column's spread, so it is never zero; integers move to a different value
       within the observed range; booleans are flipped with probability one half
       and the first changed row is always flipped; categorical and string values
       are replaced by another observed value (permuted if there is only one);
       other dtypes are permuted. Sparse columns are made dense, changed and made
       sparse again. Dtypes and the index are kept and missing values stay
       missing. The output is compared on the rows up to the cutoff.

    A row of the full output up to the cutoff that is missing from the other
    output is a ``missing_row`` finding. So is a row that only the other output
    has, and every row of a column that the other output lacks. Rows present in
    both are compared per column: numeric columns with
    ``np.isclose(a, b, rtol=rtol, atol=atol_col, equal_nan=True)`` where NaN
    equals NaN, other columns with equality where missing equals missing; a
    difference is a ``value_change`` finding. The absolute tolerance
    ``atol_col`` is scaled to the column: ``atol * scale``, where ``scale`` is
    the median absolute value of that column in the full output (ignoring NaN
    and infinities; 1.0 if there is none or if the median is zero). A fixed
    absolute tolerance would hide every leak in a column of small numbers.

    Horizon. For each cutoff with findings, ``s_min`` is the position of the
    earliest affected row over all columns and tests, and the horizon at that
    cutoff is ``p - s_min``. ``horizon_rows`` is the maximum over cutoffs (so
    ``shift(-1)`` gives 1 and a centered window of 5 gives 2). If ``s_min`` lies
    in the first 5 % of the rows for more than half of the cutoffs that have
    findings, and the cutoff is far enough from the start that this means
    something (its horizon exceeds those first rows), the dependence reaches back
    to the start of the sample and ``horizon_unbounded`` is True;
    ``horizon_rows`` is then only the largest look-ahead that was measured.

    Cost. Each test calls ``fn`` once per cutoff, plus two calls for the
    determinism check. ``report.n_evaluations`` is the exact number of calls,
    including calls that raised.

    Args:
        fn: the function under test.
        data: pandas ``DataFrame`` or ``Series`` with a monotonic increasing,
            unique index, or a polars ``DataFrame`` together with ``time_col``.
        cutoffs: the number of cutoff positions per test (default 64), or
            explicit positions (any iterable of integers from 2 to
            ``len(data) - 1``; they are sorted, de-duplicated and used by both
            tests). With a count ``k`` each test gets its own ``k`` positions:
            ``len(data) - 1``, ``len(data) - 2`` and ``len(data) // 2``, and the
            others spread evenly over the allowed range, one at a random place in
            each equal stretch of it (seeded by ``seed``), so that they do not
            share a phase with periodic features such as month-ends. The
            perturbation test may start at position 2, because it always passes
            the full length to ``fn``. The truncation test starts at
            ``max(2, min_rows, ceil(0.1 * len(data)))``. Fewer positions are
            used if the range has fewer rows. A larger count finds sparser leaks
            at a linear cost; a leak that touches a single row is found only by a
            cutoff right behind it.
        mode: ``"truncate"``, ``"perturb"`` or ``"both"``.
        atol: absolute tolerance of the numeric comparison, relative to the
            scale of each column (see above).
        rtol: relative tolerance of the numeric comparison.
        time_col: name of the time column. Required for polars input. For a
            pandas ``DataFrame`` it makes that column the (sorted) index.
        seed: seed of the cutoff positions and of the perturbation. The same
            seed gives the same report.
        min_rows: shortest truncation length to try. Set it for functions that
            fail on short histories. It does not affect the perturbation test.

    Returns:
        The :class:`~peekproof.report.OracleReport`. ``causal`` is True when no
        test produced a finding. ``cutoff_positions`` lists every position that
        either test used. If the last cutoff lies before the last row, a note
        says which share of the rows the cutoffs cover (the last cutoff divided
        by the number of rows).

    Raises:
        PeekproofError: for unusable input or arguments (including a column
            whose dtype cannot be perturbed, which the message names), a function
            that raised on the full input, a test whose every evaluation raised
            while the other test found nothing, an output that is not a
            ``DataFrame`` or ``Series``, and an output whose index is neither a
            subset of the input index nor, for a ``DatetimeIndex`` input, a
            monotonic ``DatetimeIndex``; the message names the evaluation (full,
            truncated or perturbed) and the cutoff. If every evaluation of one
            test raised but the other test found leaks, the findings are kept and
            a note says so.
    """
    return _run(
        fn,
        data,
        cutoffs=cutoffs,
        mode=mode,
        atol=atol,
        rtol=rtol,
        time_col=time_col,
        seed=seed,
        min_rows=min_rows,
        lookahead=0,
        label_horizon=None,
    )


def check_label(
    fn: CausalFn,
    data: pd.DataFrame | pd.Series | pl.DataFrame,
    *,
    horizon: int,
    cutoffs: int | Iterable[int] = _DEFAULT_CUTOFFS,
    mode: Mode = "both",
    atol: float = 1e-9,
    rtol: float = 1e-7,
    time_col: str | None = None,
    seed: int = 0,
    min_rows: int | None = None,
) -> OracleReport:
    """Test a label function that may look exactly ``horizon`` rows ahead.

    A label such as the return over the next ``horizon`` bars uses later rows on
    purpose. It is acceptable when the label at row s depends on rows up to
    ``s + horizon`` and on nothing later.

    Construction. Let ``g(D)`` be the output of ``fn`` shifted back by
    ``horizon`` rows, so that ``g(D)[s] = fn(D)[s - horizon]``. Then ``g`` is
    causal exactly when the output of ``fn`` at row s depends only on rows up to
    ``s + horizon``. The check runs the tests of :func:`check` on ``g`` without
    building ``g``: with cutoff position ``p`` it compares only the rows
    ``r < p - horizon`` of the shortened or perturbed output of ``fn`` with the
    full output. The rows ``p - horizon .. p - 1`` may differ. With
    ``horizon=0`` this is :func:`check`.

    A cutoff ``p`` at or below ``horizon`` would leave nothing to compare, so no
    such cutoff is used. Counted cutoffs start above the horizon; explicit
    positions at or below it are dropped and a note lists them. If no explicit
    position remains, the call raises.

    The report is the report of that comparison, with ``label_horizon`` set to
    ``horizon``. If the label passes there are no findings. If it fails, the
    findings are the differences in rows ``r < p - horizon`` and ``horizon_rows``
    is the look-ahead measured on ``fn`` itself (not on ``g``), so it is larger
    than ``horizon``. A label that depends on the whole sample fails for every
    ``horizon`` and has ``horizon_unbounded=True``.

    All other arguments are those of :func:`check`, and so are its limits: the
    result is evidence and not a proof, and only dependence on the data that is
    passed to ``fn`` is seen.

    Raises:
        PeekproofError: as :func:`check`, for a negative ``horizon``, and when no
            cutoff lies above the horizon.
    """
    return _run(
        fn,
        data,
        cutoffs=cutoffs,
        mode=mode,
        atol=atol,
        rtol=rtol,
        time_col=time_col,
        seed=seed,
        min_rows=min_rows,
        lookahead=horizon,
        label_horizon=horizon,
    )


def assert_causal(
    fn: CausalFn,
    data: pd.DataFrame | pd.Series | pl.DataFrame,
    *,
    cutoffs: int | Iterable[int] = _DEFAULT_CUTOFFS,
    mode: Mode = "both",
    atol: float = 1e-9,
    rtol: float = 1e-7,
    time_col: str | None = None,
    seed: int = 0,
    min_rows: int | None = None,
) -> None:
    """Run :func:`check` and raise :class:`~peekproof.errors.LeakError` unless ``fn`` is causal.

    ``LeakError`` is an ``AssertionError``; its message is the rendered report
    and the report itself is available as ``error.report``. A function that is
    not deterministic also raises, because the leak tests could not run; the
    message then contains the note "function is not deterministic; seed it
    before testing". The arguments are those of :func:`check`, and so are its
    limits: passing the assertion means that no leak was found, not that none
    exists.
    """
    report = check(
        fn,
        data,
        cutoffs=cutoffs,
        mode=mode,
        atol=atol,
        rtol=rtol,
        time_col=time_col,
        seed=seed,
        min_rows=min_rows,
    )
    if report.causal is not True:
        raise LeakError(report)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _key_unit(index: pd.DatetimeIndex) -> TimeUnit:
    """The unit of the integer keys: nanoseconds, or microseconds for dates ``ns`` cannot hold.

    Output labels are converted to this unit. A coarser unit than the output's would
    round its labels together, so the finest one that fits is used for every index.
    """
    try:
        index.as_unit("ns")
    except (OverflowError, ValueError):  # OutOfBoundsDatetime is a ValueError
        return "us"
    return "ns"


def _time_keys(index: pd.DatetimeIndex, unit: TimeUnit) -> npt.NDArray[np.int64]:
    """The labels of a ``DatetimeIndex`` as integers in ``unit`` (counted from UTC if aware)."""
    return np.asarray(index.as_unit(unit).astype("int64"), dtype=np.int64)


@dataclass(frozen=True)
class _Axis:
    """The input index and an integer key for each of its rows.

    A key orders labels. For a ``DatetimeIndex`` it is the time as an integer, so
    that labels which are not in the index (bar labels) can be ordered against
    it. For any other index it is the row position.
    """

    index: pd.Index
    keys: npt.NDArray[np.int64]
    timed: bool
    unit: TimeUnit = "ns"
    tz_aware: bool = False  # only meaningful for a timed axis

    @classmethod
    def of(cls, index: pd.Index) -> _Axis:
        if isinstance(index, pd.DatetimeIndex):
            unit = _key_unit(index)
            return cls(index, _time_keys(index, unit), True, unit, index.tz is not None)
        return cls(index, np.arange(len(index), dtype=np.int64), False)


@dataclass(frozen=True)
class _Output:
    """A normalised function output: a frame with unique columns and labels.

    ``keys[i]`` orders the label of row ``i`` against the input (see
    :class:`_Axis`). ``positions[i]`` is the last input row at or before that
    label, which is the row itself when the label is in the input.
    """

    frame: pd.DataFrame
    keys: npt.NDArray[np.int64]
    positions: npt.NDArray[np.intp]


@dataclass(frozen=True)
class _Hit:
    """One finding before it is turned into a :class:`LeakFinding`.

    ``first_pos`` and ``last_pos`` are input positions (they give the horizon);
    the labels are those of the affected output rows.
    """

    test: TestKind
    cutoff: int
    column: str
    kind: LeakKind
    n_rows: int
    first_pos: int
    last_pos: int
    max_abs_diff: float | None
    first_label: str
    last_label: str


@dataclass(frozen=True)
class _Plan:
    """The cutoff positions of each test that runs, and the notes made while choosing them."""

    positions: dict[TestKind, list[int]]
    notes: list[str]

    @property
    def union(self) -> list[int]:
        return sorted({p for chosen in self.positions.values() for p in chosen})


@dataclass(frozen=True)
class _Evaluator:
    """Calls the function under test and normalises what it returns."""

    fn: CausalFn
    axis: _Axis
    adapter: PolarsAdapter | None = None

    def argument(self, frame: pd.DataFrame | pd.Series) -> object:
        """What the function receives for a pandas slice of the data."""
        if self.adapter is None or not isinstance(frame, pd.DataFrame):
            return frame
        return self.adapter.to_native(frame)

    def output(self, raw: object, where: str, given: int) -> _Output:
        """Normalise the result of an evaluation; ``given`` rows were passed in."""
        if self.adapter is not None:
            raw = self.adapter.to_pandas(raw, where)
        return _normalise(raw, self.axis, where, given, polars=self.adapter is not None)


def _run(
    fn: CausalFn,
    data: object,
    *,
    cutoffs: int | Iterable[int],
    mode: str,
    atol: float,
    rtol: float,
    time_col: str | None,
    seed: int,
    min_rows: int | None,
    lookahead: int,
    label_horizon: int | None,
) -> OracleReport:
    """Shared implementation of :func:`check` and :func:`check_label`."""
    _validate_options(
        mode=mode, atol=atol, rtol=rtol, seed=seed, min_rows=min_rows, lookahead=lookahead
    )
    # numpy integers are accepted; the report should only hold plain Python numbers
    seed = int(seed)
    lookahead = int(lookahead)
    min_rows = None if min_rows is None else int(min_rows)
    label_horizon = None if label_horizon is None else int(label_horizon)
    frame = to_pandas(data, time_col)
    index = frame.index
    n_rows = len(index)
    tests = _TESTS[mode]
    plan = _plan_cutoffs(n_rows, cutoffs, tests, min_rows=min_rows, lookahead=lookahead, seed=seed)
    evaluator = _Evaluator(fn, _Axis.of(index), polars_adapter(data, time_col))

    full = _evaluate_full(evaluator, frame)
    again = _evaluate_full(evaluator, frame)
    if not frames_equal(full.frame, again.frame, atol=atol, rtol=rtol, scaled=True):
        return OracleReport(
            causal=None,
            deterministic=False,
            n_rows=n_rows,
            n_evaluations=2,
            notes=[_NOT_DETERMINISTIC],
            label_horizon=label_horizon,
        )

    atols = [column_atol(atol, full.frame.iloc[:, j]) for j in range(full.frame.shape[1])]
    n_evaluations = 2
    hits: list[_Hit] = []
    notes = list(plan.notes)
    unable: dict[TestKind, tuple[int, str]] = {}
    for test in tests:
        positions = plan.positions[test]
        found, failures = _run_test(
            evaluator,
            frame,
            full,
            test,
            positions,
            lookahead=lookahead,
            seed=seed,
            atols=atols,
            rtol=rtol,
        )
        n_evaluations += len(positions)
        hits.extend(found)
        if len(failures) == len(positions):
            unable[test] = (len(positions), failures[0][1])
        elif failures:
            notes.append(_partial_failure_note(test, failures, len(positions)))
    if unable and not hits:
        raise PeekproofError(_all_failed_message(unable))
    notes.extend(_skipped_test_note(test, *failed) for test, failed in unable.items())

    last_cutoff = plan.union[-1]
    if last_cutoff < n_rows - 1:
        notes.append(_coverage_note(last_cutoff, n_rows))
    horizon_rows, unbounded = _estimate_horizon(hits, n_rows)
    findings = [_to_finding(hit, index) for hit in hits]
    return OracleReport(
        causal=not findings,
        deterministic=True,
        n_rows=n_rows,
        n_evaluations=n_evaluations,
        cutoff_positions=plan.union,
        findings=findings,
        horizon_rows=horizon_rows,
        horizon_unbounded=unbounded,
        notes=notes,
        label_horizon=label_horizon,
    )


def _run_test(
    evaluator: _Evaluator,
    frame: pd.DataFrame | pd.Series,
    full: _Output,
    test: TestKind,
    positions: Sequence[int],
    *,
    lookahead: int,
    seed: int,
    atols: Sequence[float],
    rtol: float,
) -> tuple[list[_Hit], list[tuple[int, str]]]:
    """Run one test over its cutoffs; return its hits and the cutoffs where ``fn`` raised."""
    n_rows = len(frame)
    keys = evaluator.axis.keys
    hits: list[_Hit] = []
    failures: list[tuple[int, str]] = []
    for cutoff in positions:
        if test == "truncate":
            argument = frame.iloc[:cutoff].copy()
            given = cutoff
        else:
            argument = _perturb(frame, cutoff, seed)
            given = n_rows
        native = evaluator.argument(argument)
        try:
            raw = evaluator.fn(native)
        except Exception as exc:
            failures.append((cutoff, _describe(exc)))
            continue
        part = evaluator.output(raw, _where(test, cutoff), given)
        hits.extend(
            _compare(
                full,
                part,
                bound=int(keys[cutoff - lookahead - 1]),
                test=test,
                cutoff=cutoff,
                atols=atols,
                rtol=rtol,
            )
        )
    return hits, failures


def _validate_options(
    *,
    mode: str,
    atol: float,
    rtol: float,
    seed: int,
    min_rows: int | None,
    lookahead: int,
) -> None:
    if mode not in _TESTS:
        raise PeekproofError(f"mode must be 'truncate', 'perturb' or 'both', got {mode!r}")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not (math.isfinite(value) and value >= 0):
            raise PeekproofError(f"{name} must be a finite number >= 0, got {value!r}")
    if not _is_int(seed) or seed < 0:
        raise PeekproofError(f"seed must be an integer >= 0, got {seed!r}")
    if min_rows is not None and (not _is_int(min_rows) or min_rows < 1):
        raise PeekproofError(f"min_rows must be an integer >= 1 or None, got {min_rows!r}")
    if not _is_int(lookahead) or lookahead < 0:
        raise PeekproofError(f"horizon must be an integer >= 0, got {lookahead!r}")


def _is_int(value: object) -> bool:
    """True for integers, including numpy integers, but not for booleans."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Cutoff selection
# --------------------------------------------------------------------------- #


def _plan_cutoffs(
    n_rows: int,
    cutoffs: int | Iterable[int],
    tests: Sequence[TestKind],
    *,
    min_rows: int | None,
    lookahead: int,
    seed: int,
) -> _Plan:
    """Choose the cutoff positions (numbers of rows to keep) of each test.

    A count gives every test its own positions: the perturbation test may start
    at row 2, the truncation test keeps a minimum history. Explicit positions are
    used as given by every test, except that those at or below ``lookahead`` are
    dropped: with such a cutoff the rows before ``p - lookahead`` are none.
    """
    if isinstance(cutoffs, int | np.integer) and not isinstance(cutoffs, bool):
        count = int(cutoffs)
        if count < 1:
            raise PeekproofError(f"cutoffs must be >= 1 when it is a count, got {count}")
        positions: dict[TestKind, list[int]] = {}
        for test in tests:
            lo = _first_cutoff(test, n_rows, min_rows, lookahead)
            hi = n_rows - 1
            if lo > hi:
                raise PeekproofError(
                    f"not enough rows: {n_rows} rows leave no {test} cutoff between {lo} and "
                    f"{hi}. Pass more data or lower min_rows / horizon."
                )
            rng = np.random.default_rng([seed, _CUTOFF_STREAM, _TEST_NUMBER[test]])
            positions[test] = _spread_cutoffs(n_rows, count, lo, rng)
        return _Plan(positions, [])

    explicit = _explicit_cutoffs(cutoffs, n_rows - 1)
    kept = [p for p in explicit if p > lookahead]
    notes: list[str] = []
    if len(kept) < len(explicit):
        dropped = [p for p in explicit if p <= lookahead]
        shown = ", ".join(str(p) for p in dropped)
        if not kept:
            raise PeekproofError(
                f"every cutoff ({shown}) is at or below the horizon of {lookahead} row(s); a "
                "cutoff p compares only the rows before p - horizon, so these compare nothing. "
                f"Use cutoffs above {lookahead}."
            )
        notes.append(
            f"dropped cutoff position(s) {shown}: at or below the horizon of {lookahead} "
            "row(s), so they would compare nothing"
        )
    return _Plan({test: list(kept) for test in tests}, notes)


def _first_cutoff(test: TestKind, n_rows: int, min_rows: int | None, lookahead: int) -> int:
    """Smallest cutoff position of a test.

    The perturbation test passes the full length to the function, so it has no
    minimum history. The truncation test needs a history that the function can
    handle: ``min_rows``, or a tenth of the rows.
    """
    lo = max(_MIN_CUTOFF, lookahead + 1)
    if test == "truncate":
        lo = max(lo, min_rows or 0, math.ceil(_LOW_FRACTION * n_rows))
    return lo


def _spread_cutoffs(n_rows: int, count: int, lo: int, rng: np.random.Generator) -> list[int]:
    """Choose about ``count`` positions in ``lo .. n_rows - 1``, spread evenly with jitter.

    ``n_rows - 1``, ``n_rows - 2`` and ``n_rows // 2`` are always included (those
    of them that are not below ``lo``). The remaining positions are drawn one from
    each of that many equal stretches of the other positions, at a random place
    inside the stretch. They cover the range evenly, yet they do not sit at a
    fixed distance from each other, so a feature with a period (month-ends, days
    of the week) is not systematically missed. If the range has no more room than
    ``count``, every position is used.
    """
    hi = n_rows - 1
    fixed = sorted({p for p in (hi, n_rows - 2, n_rows // 2) if lo <= p <= hi})
    free = hi - lo + 1 - len(fixed)
    wanted = count - len(fixed)
    if wanted <= 0 or free <= 0:
        return fixed
    if wanted >= free:
        return list(range(lo, hi + 1))
    # stretch i holds the free slots edges[i] .. edges[i + 1] - 1; each has at least one slot
    edges = [-(-i * free // wanted) for i in range(wanted + 1)]
    chosen = set(fixed)
    for start, stop in pairwise(edges):
        position = lo + int(rng.integers(start, stop))
        for taken in fixed:  # step over the fixed positions, in ascending order
            if position >= taken:
                position += 1
        chosen.add(position)
    return sorted(chosen)


def _explicit_cutoffs(cutoffs: object, hi: int) -> list[int]:
    """Validate explicit cutoff positions; return them sorted and de-duplicated."""
    if isinstance(cutoffs, str | bytes | bool) or not isinstance(cutoffs, Iterable):
        raise PeekproofError(
            f"cutoffs must be an integer count or a sequence of integers, got {cutoffs!r}"
        )
    positions: set[int] = set()
    for value in cutoffs:
        if not _is_int(value):
            raise PeekproofError(f"cutoff positions must be integers, got {value!r}")
        position = int(value)
        if not _MIN_CUTOFF <= position <= hi:
            raise PeekproofError(
                f"cutoff position {position} is outside the valid range {_MIN_CUTOFF}..{hi} "
                f"(a position is the number of rows kept, and the data has {hi + 1} rows)"
            )
        positions.add(position)
    if not positions:
        raise PeekproofError("cutoffs is empty")
    return sorted(positions)


def _coverage_note(last_cutoff: int, n_rows: int) -> str:
    share = 100 * last_cutoff // n_rows
    return (
        f"cutoffs cover {share}% of rows (the last cutoff is position {last_cutoff} of "
        f"{n_rows}); a leak that only affects later rows is not tested"
    )


# --------------------------------------------------------------------------- #
# Evaluation and output handling
# --------------------------------------------------------------------------- #


def _where(test: TestKind | None, cutoff: int | None = None) -> str:
    """Name an evaluation for error messages."""
    if test is None:
        return "the full evaluation"
    return f"the {_ADJECTIVE[test]} evaluation at cutoff {cutoff}"


def _evaluate_full(evaluator: _Evaluator, frame: pd.DataFrame | pd.Series) -> _Output:
    """Evaluate ``fn`` on a copy of the full data and normalise the result."""
    argument = evaluator.argument(frame.copy())
    try:
        raw = evaluator.fn(argument)
    except Exception as exc:
        raise PeekproofError(f"the function raised on the full input: {_describe(exc)}") from exc
    return evaluator.output(raw, _where(None), len(frame))


def _normalise(raw: object, axis: _Axis, where: str, given: int, *, polars: bool) -> _Output:
    """Turn a function result into a frame and locate its rows against the input.

    ``given`` is the number of input rows the function was passed. ``where`` names
    the evaluation for the error messages.
    """
    if isinstance(raw, pd.Series):
        frame = raw.to_frame(name=raw.name if raw.name is not None else "value")
    elif isinstance(raw, pd.DataFrame):
        frame = raw
    else:
        expected = "a polars DataFrame" if polars else "a pandas DataFrame or Series"
        raise PeekproofError(
            f"the function must return {expected}, got {type(raw).__name__} ({where})"
        )
    if not frame.columns.is_unique:
        repeated = sorted({str(c) for c in frame.columns[frame.columns.duplicated()]})
        raise PeekproofError(
            f"the function returned duplicate column names ({', '.join(repeated)}) in {where}; "
            "columns are matched between evaluations by name, so they must be unique"
        )
    labels = frame.index
    if not labels.is_unique:
        raise PeekproofError(
            f"the output of {where} contains duplicate labels; rows are matched between "
            "evaluations by label, so every label must be unique"
        )
    if labels.equals(axis.index[:given]):
        return _Output(frame, axis.keys[:given], np.arange(given, dtype=np.intp))
    try:
        located = np.asarray(axis.index.get_indexer(labels), dtype=np.intp)
    except (TypeError, ValueError, KeyError, pd.errors.InvalidIndexError):
        located = np.full(len(labels), -1, dtype=np.intp)
    if (located >= 0).all():
        return _Output(frame, axis.keys[located], located)
    return _relabeled(frame, axis, where, int((located < 0).sum()))


def _relabeled(frame: pd.DataFrame, axis: _Axis, where: str, n_unknown: int) -> _Output:
    """Locate an output whose labels are not all in the input index (bar labels).

    Only a ``DatetimeIndex`` output of a ``DatetimeIndex`` input is accepted, in
    increasing order: it can be compared with the cutoff by time.
    """
    labels = frame.index
    if not (axis.timed and isinstance(labels, pd.DatetimeIndex)):
        raise PeekproofError(
            f"the output of {where} has an index that is not a subset of the input index "
            f"({n_unknown} of {len(labels)} output rows have labels that are not in the input) "
            f"and is not a DatetimeIndex like the input's (it is a {type(labels).__name__}). "
            "Outputs may be labeled differently only by time, for example by resampling a "
            "DatetimeIndex; indexing by position or by anything else is not supported."
        )
    if (labels.tz is not None) != axis.tz_aware:
        raise PeekproofError(
            f"the output of {where} and the input differ in time-zone awareness, so their "
            "labels cannot be compared"
        )
    if not labels.is_monotonic_increasing:
        raise PeekproofError(
            f"the output of {where} has labels that are not in the input index and are not in "
            "increasing order; re-labeled outputs must have a monotonic increasing index"
        )
    try:
        keys = _time_keys(labels, axis.unit)
    except (OverflowError, ValueError, pd.errors.OutOfBoundsDatetime) as exc:
        raise PeekproofError(
            f"the labels of the output of {where} cannot be compared with the input labels: {exc}"
        ) from exc
    # the last input row at or before each label; the first row stands in for earlier labels
    positions = np.maximum(np.searchsorted(axis.keys, keys, side="right") - 1, 0)
    return _Output(frame, keys, positions.astype(np.intp))


def _describe(exc: Exception) -> str:
    """One short line describing an exception."""
    text = " ".join(str(exc).split())
    if len(text) > 120:
        text = text[:117] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _all_failed_message(unable: dict[TestKind, tuple[int, str]]) -> str:
    """Error for a run in which every evaluation of every test that ran raised.

    ``unable`` maps a test to its number of cutoffs and the first error.
    """
    if len(unable) == 2:
        (n_truncated, first_truncated), (n_perturbed, first_perturbed) = (
            unable["truncate"],
            unable["perturb"],
        )
        return (
            f"the function raised on all {n_truncated} truncated inputs (first error: "
            f"{first_truncated}) and on all {n_perturbed} perturbed inputs (first error: "
            f"{first_perturbed}). If it needs a minimum history, pass min_rows=<rows> so that "
            "only histories of at least that length are tried."
        )
    if "truncate" in unable:
        n_cutoffs, first = unable["truncate"]
        return (
            f"the function raised on all {n_cutoffs} truncated inputs (first error: {first}). "
            "It probably needs a minimum history; pass min_rows=<rows> so that only "
            "histories of at least that length are tried."
        )
    n_cutoffs, first = unable["perturb"]
    return (
        f"the function raised on all {n_cutoffs} perturbed inputs (first error: {first}). "
        "Use mode='truncate' to skip the perturbation test."
    )


def _skipped_test_note(test: TestKind, n_cutoffs: int, first_error: str) -> str:
    """Note for a test whose every evaluation raised while the other test found leaks."""
    if test == "truncate":
        return (
            f"the truncation test could not run: the function raised on all {n_cutoffs} "
            f"truncated inputs (first error: {first_error}). Pass min_rows to start from a "
            "longer history. Only the findings of the perturbation test are reported."
        )
    return (
        f"the perturbation test could not run: the function raised on all {n_cutoffs} "
        f"perturbed inputs (first error: {first_error}). Only the findings of the truncation "
        "test are reported."
    )


def _partial_failure_note(test: TestKind, failures: list[tuple[int, str]], n_cutoffs: int) -> str:
    shown = ", ".join(str(p) for p, _ in failures[:6]) + (", ..." if len(failures) > 6 else "")
    hint = " Pass min_rows to start from a longer history." if test == "truncate" else ""
    return (
        f"function raised on {len(failures)} of {n_cutoffs} {_ADJECTIVE[test]} inputs (cutoff "
        f"positions {shown}); first error: {failures[0][1]}. Those cutoffs were skipped.{hint}"
    )


# --------------------------------------------------------------------------- #
# Perturbation
# --------------------------------------------------------------------------- #


def _perturb(frame: pd.DataFrame | pd.Series, cutoff: int, seed: int) -> pd.DataFrame | pd.Series:
    """Copy of ``frame`` in which every row from ``cutoff`` on is changed.

    Each column is changed on its own, according to its dtype (see
    :func:`_perturb_column`); the rows before ``cutoff``, the index and the
    dtypes are untouched. The random numbers depend only on ``seed`` and
    ``cutoff``.
    """
    rng = np.random.default_rng([seed, _PERTURB_STREAM, cutoff])
    if isinstance(frame, pd.Series):
        return _perturb_checked(frame, str(frame.name), cutoff, rng)
    result = frame.copy()
    for j in range(frame.shape[1]):
        column = _perturb_checked(frame.iloc[:, j], str(frame.columns[j]), cutoff, rng)
        result.isetitem(j, column.array)
    return result


def _perturb_checked(
    column: pd.Series, name: str, start: int, rng: np.random.Generator
) -> pd.Series:
    """:func:`_perturb_column`, with a clear error for a column that cannot be changed."""
    try:
        return _perturb_column(column, start, rng)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise PeekproofError(
            f"cannot perturb column {name!r} of dtype {column.dtype}: {exc}. Convert the column "
            "to a plain dtype, or use mode='truncate', which does not change the data."
        ) from exc


def _perturb_column(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Change every row of ``column`` from position ``start`` on; keep the dtype.

    Sparse columns are made dense, changed and made sparse again. Then by dtype:

    - booleans: flipped with probability one half, the first row always;
    - integers: moved to a different value of the observed range;
    - floats: Student-t noise scaled by the robust spread of the column;
    - categoricals, strings and objects: replaced by another observed value;
    - anything else (dates, durations, complex numbers, ...): permuted.

    Missing values stay missing.
    """
    dtype = column.dtype
    if isinstance(dtype, pd.SparseDtype):
        dense = pd.Series(np.asarray(column.array), index=column.index, name=column.name)
        return _perturb_column(dense, start, rng).astype(dtype)
    if pd.api.types.is_bool_dtype(dtype):
        return _flip_booleans(column, start, rng)
    if pd.api.types.is_integer_dtype(dtype):
        return _offset_integers(column, start, rng)
    if pd.api.types.is_float_dtype(dtype):
        return _add_noise(column, start, rng)
    if (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    ):
        return _swap_categories(column, start, rng)
    return _permute_rows(column, start, rng)


def _with_future(column: pd.Series, start: int, values: pd.Series) -> pd.Series:
    """Copy of ``column`` whose rows from ``start`` on are ``values`` (of the same dtype)."""
    result = column.copy()
    result.iloc[start:] = values.array
    return result


def _add_noise(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Add Student-t noise to the rows from ``start`` on.

    The noise has 3 degrees of freedom, so it is heavy-tailed, and is scaled by
    the robust spread of the column (never zero, see
    :func:`~peekproof._compat.robust_scale`). It is added in float64 and cast
    back to the dtype of the column. Missing values stay missing.
    """
    scale = robust_scale(column)
    numbers = column.iloc[start:].to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(over="ignore", invalid="ignore"):  # a narrow dtype may overflow to inf
        moved = numbers + scale * rng.standard_t(_T_DEGREES, size=numbers.size)
        noisy = pd.Series(moved).astype(column.dtype)
    return _with_future(column, start, noisy)


def _offset_integers(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Move every integer from ``start`` on to a different value of the observed range.

    The offset is a random non-zero number of steps through the range
    ``min .. max`` of the whole column, wrapping around at the ends, so the new
    value is always different and always a value of the same kind as the others.
    A constant column has no range: its values move by one step, up or down. A
    range too wide for the arithmetic, or values outside the signed 64-bit
    range, are permuted instead.
    """
    observed = column.dropna()
    if observed.empty:
        return column.copy()
    low, high = int(observed.min()), int(observed.max())
    span = high - low + 1
    if span > _MAX_SPAN or low < -(2**63) or high >= 2**63:
        return _permute_rows(column, start, rng)
    future = column.iloc[start:]
    present = future.notna().to_numpy()
    numbers = future.to_numpy(dtype="int64", na_value=low)
    if span > 1:
        moved = low + (numbers - low + rng.integers(1, span, size=numbers.size)) % span
    else:
        limits = np.iinfo(getattr(column.dtype, "numpy_dtype", column.dtype))
        step = rng.choice(np.array([-1, 1], dtype="int64"), size=numbers.size)
        moved = numbers + step
        moved = np.where((moved < limits.min) | (moved > limits.max), numbers - step, moved)
    new = pd.Series(moved).astype(column.dtype)
    return _with_future(column, start, new.where(present))


def _flip_booleans(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Flip booleans from ``start`` on with probability one half; the first one always."""
    future = column.iloc[start:]
    present = future.notna().to_numpy()
    flags = future.to_numpy(dtype="bool", na_value=False)
    flip = rng.random(flags.size) < _FLIP_PROBABILITY
    flip[0] = True
    new = pd.Series(flags ^ flip).astype(column.dtype)
    return _with_future(column, start, new.where(present))


def _swap_categories(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Replace each value from ``start`` on by a different value that occurs in the column.

    With fewer than two distinct values (or values that cannot be hashed) there
    is nothing to swap to, and the rows are permuted instead.
    """
    try:
        codes, uniques = pd.factorize(column)
    except TypeError:
        return _permute_rows(column, start, rng)
    n_values = len(uniques)
    if n_values < 2:
        return _permute_rows(column, start, rng)
    old = codes[start:]
    present = old >= 0
    moved = (old + rng.integers(1, n_values, size=old.size)) % n_values
    values = np.asarray(uniques, dtype=object)[np.where(present, moved, 0)]
    values[~present] = None
    return _with_future(column, start, pd.Series(values, dtype=column.dtype))


def _permute_rows(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
    """Permute the values from ``start`` on among those rows."""
    order = rng.permutation(len(column) - start)
    return _with_future(column, start, column.iloc[start:].iloc[order])


# --------------------------------------------------------------------------- #
# Comparison and horizon
# --------------------------------------------------------------------------- #


def _compare(
    full: _Output,
    part: _Output,
    *,
    bound: int,
    test: TestKind,
    cutoff: int,
    atols: Sequence[float],
    rtol: float,
) -> list[_Hit]:
    """Compare the rows of two outputs whose label key is at most ``bound``.

    ``full`` is the output on the full data and ``part`` the output on the
    truncated or perturbed data. Rows are matched by label. Every column of
    ``full`` is checked: rows that only one output has are ``missing_row`` hits,
    rows in both with different values are ``value_change`` hits, and a column
    that ``part`` lacks is missing for every row of ``full``. Columns that only
    ``part`` has are ignored. Rows of either output after the bound are ignored.
    """
    rows_full = np.flatnonzero(full.keys <= bound)
    rows_part = np.flatnonzero(part.keys <= bound)
    keys_full = full.keys[rows_full]
    keys_part = part.keys[rows_part]
    nothing = np.zeros(0, dtype=np.intp)
    if np.array_equal(keys_full, keys_part):
        shared_full, shared_part, lone_full, lone_part = rows_full, rows_part, nothing, nothing
    else:
        _, at_full, at_part = np.intersect1d(
            keys_full, keys_part, assume_unique=True, return_indices=True
        )
        shared_full, shared_part = rows_full[at_full], rows_part[at_part]
        lone_full, lone_part = np.delete(rows_full, at_full), np.delete(rows_part, at_part)
    part_column = part.frame.columns.get_indexer(full.frame.columns)
    full_rows = full.frame.iloc[shared_full]
    part_rows = part.frame.iloc[shared_part]

    hits: list[_Hit] = []
    for j, label in enumerate(full.frame.columns):
        name = str(label)
        if part_column[j] < 0:
            if rows_full.size:
                hits.append(_hit(test, cutoff, name, "missing_row", [(full, rows_full)], None))
            continue
        if lone_full.size or lone_part.size:
            lone = [(full, lone_full), (part, lone_part)]
            hits.append(_hit(test, cutoff, name, "missing_row", lone, None))
        if shared_full.size:
            diff = compare_columns(
                full_rows.iloc[:, j], part_rows.iloc[:, part_column[j]], atol=atols[j], rtol=rtol
            )
            if diff.differs.any():
                changed = [(full, shared_full[diff.differs])]
                hits.append(_hit(test, cutoff, name, "value_change", changed, diff.max_abs_diff))
    return hits


def _hit(
    test: TestKind,
    cutoff: int,
    column: str,
    kind: LeakKind,
    sources: Sequence[tuple[_Output, npt.NDArray[np.intp]]],
    max_abs_diff: float | None,
) -> _Hit:
    """Build a hit from the affected rows of one or two outputs.

    The first and last affected rows are those with the smallest and largest key.
    """
    n_rows = 0
    first: tuple[int, int, str] | None = None
    last: tuple[int, int, str] | None = None
    for output, rows in sources:
        if rows.size == 0:
            continue
        n_rows += int(rows.size)
        keys = output.keys[rows]
        low, high = rows[int(np.argmin(keys))], rows[int(np.argmax(keys))]
        smallest = (
            int(output.keys[low]),
            int(output.positions[low]),
            format_label(output.frame.index[low]),
        )
        largest = (
            int(output.keys[high]),
            int(output.positions[high]),
            format_label(output.frame.index[high]),
        )
        first = smallest if first is None or smallest[0] < first[0] else first
        last = largest if last is None or largest[0] > last[0] else last
    assert first is not None and last is not None  # callers pass at least one affected row
    return _Hit(
        test=test,
        cutoff=cutoff,
        column=column,
        kind=kind,
        n_rows=n_rows,
        first_pos=first[1],
        last_pos=last[1],
        max_abs_diff=max_abs_diff,
        first_label=first[2],
        last_label=last[2],
    )


def _to_finding(hit: _Hit, index: pd.Index) -> LeakFinding:
    return LeakFinding(
        column=hit.column,
        kind=hit.kind,
        test=hit.test,
        cutoff_pos=hit.cutoff,
        cutoff_label=label_at(index, hit.cutoff - 1),
        n_rows_affected=hit.n_rows,
        first_affected_label=hit.first_label,
        last_affected_label=hit.last_label,
        max_abs_diff=hit.max_abs_diff,
    )


def _estimate_horizon(hits: Sequence[_Hit], n_rows: int) -> tuple[int | None, bool]:
    """Return ``(horizon_rows, horizon_unbounded)`` for the hits of a run.

    For each cutoff ``p`` with hits, ``s_min`` is the earliest affected row over
    all hits and the horizon at ``p`` is ``p - s_min``. The result is the largest
    horizon. It is unbounded if, for more than half of the cutoffs that have hits,
    ``s_min`` is within the first 5 % of the rows and the horizon at that cutoff
    is longer than those first rows. The second condition keeps a cutoff near the
    start from voting: there, any row that a leak touches is an early row.
    """
    if not hits:
        return None, False
    earliest: dict[int, int] = {}
    for hit in hits:
        earliest[hit.cutoff] = min(earliest.get(hit.cutoff, hit.first_pos), hit.first_pos)
    horizon = max(cutoff - first for cutoff, first in earliest.items())
    n_early_rows = max(1, math.ceil(_EARLY_FRACTION * n_rows))
    n_reaching = sum(
        1
        for cutoff, first in earliest.items()
        if first < n_early_rows and cutoff - first > n_early_rows
    )
    return horizon, 2 * n_reaching > len(earliest)
