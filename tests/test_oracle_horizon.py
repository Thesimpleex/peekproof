"""Tests for horizon estimation, label mode, cutoff selection and the detection of sparse leaks.

All data is synthetic and seeded, so every test is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from peekproof._compat import format_label
from peekproof.errors import PeekproofError
from peekproof.oracle import (
    Mode,
    _estimate_horizon,
    _Hit,
    _plan_cutoffs,
    _spread_cutoffs,
    check,
    check_label,
)

Fn = Callable[[pd.DataFrame], pd.DataFrame | pd.Series]

N_ROWS = 300
MODES: tuple[Mode, ...] = ("truncate", "perturb", "both")
BOTH = ("truncate", "perturb")
#: Cutoffs per test when a test runs a single mode: the default is kept for ``mode="both"``.
FEW = 16
#: ``(mode, cutoffs)``: every mode, with the default number of cutoffs only where both tests run.
MODE_CUTOFFS: list[tuple[Mode, int]] = [("both", 64), ("truncate", FEW), ("perturb", FEW)]


def make_prices(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Business-day frame with a float price column and an integer volume column."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    volume = rng.integers(1_000, 5_000, n)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices()


# --------------------------------------------------------------------------- #
# Functions under test
# --------------------------------------------------------------------------- #


def shift_lead_1(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1)


def shift_lead_3(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-3)


def lead_1_then_dropna(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1).dropna()


def centered_window_5(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(5, center=True).mean()


def pct_change_lead_2(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(-2)


def loop_reads_next_row(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    values = [
        close.iloc[i + 1] - close.iloc[i] if i + 1 < len(close) else np.nan
        for i in range(len(close))
    ]
    return pd.Series(values, index=df.index, name="step")


def global_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def global_minmax(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def rank_pct_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(pct=True)


def divide_by_last_row(df: pd.DataFrame) -> pd.DataFrame:
    return df / df.iloc[-1]


def forward_return(df: pd.DataFrame, bars: int) -> pd.Series:
    return (df["close"].shift(-bars) / df["close"] - 1.0).rename(f"fwd_ret_{bars}")


def forward_return_5(df: pd.DataFrame) -> pd.Series:
    return forward_return(df, 5)


def forward_return_5_dropna(df: pd.DataFrame) -> pd.Series:
    return forward_return(df, 5).dropna()


# --------------------------------------------------------------------------- #
# Horizon of leaky functions
# --------------------------------------------------------------------------- #

BOUNDED: dict[str, tuple[Fn, int]] = {
    "shift_lead_1": (shift_lead_1, 1),
    "shift_lead_3": (shift_lead_3, 3),
    "lead_1_then_dropna": (lead_1_then_dropna, 1),
    "centered_window_5": (centered_window_5, 2),
    "loop_reads_next_row": (loop_reads_next_row, 1),
    "pct_change_lead_2": (pct_change_lead_2, 2),
}


@pytest.mark.parametrize(("mode", "cutoffs"), MODE_CUTOFFS)
@pytest.mark.parametrize("name", list(BOUNDED))
def test_bounded_leaks_report_their_horizon(
    prices: pd.DataFrame, name: str, mode: Mode, cutoffs: int
) -> None:
    fn, expected = BOUNDED[name]
    report = check(fn, prices, mode=mode, cutoffs=cutoffs)
    assert report.deterministic is True
    assert report.causal is False
    assert report.horizon_rows == expected
    assert report.horizon_unbounded is False


@pytest.mark.parametrize("bars", [1, 2, 3, 5, 8, 12, 20])
def test_horizon_of_a_negative_shift_is_the_shift(prices: pd.DataFrame, bars: int) -> None:
    report = check(lambda df: df.shift(-bars), prices)
    assert report.causal is False
    assert report.horizon_rows == bars
    # Even 20 bars are bounded: only a few of the cutoffs lie close to the start.
    assert report.horizon_unbounded is False


@pytest.mark.parametrize(("window", "horizon"), [(3, 1), (5, 2), (7, 3), (9, 4)])
def test_horizon_of_a_centered_window_is_half_the_window(
    prices: pd.DataFrame, window: int, horizon: int
) -> None:
    report = check(lambda df: df.rolling(window, center=True).mean(), prices)
    assert report.causal is False
    assert report.horizon_rows == horizon
    assert report.horizon_unbounded is False


def test_horizon_of_a_backward_fill_is_the_length_of_the_gap_that_ends_at_a_cutoff() -> None:
    cutoffs = [50, 100, 150, 200, 250]
    data = make_prices().astype("float64")
    for p in cutoffs:
        data.iloc[[p - 2, p - 1], 0] = np.nan  # a gap of two rows right before each cutoff
    report = check(lambda df: df.bfill(), data, cutoffs=cutoffs)
    assert report.causal is False
    assert report.horizon_rows == 2
    assert report.horizon_unbounded is False


def test_horizon_is_the_largest_one_over_all_cutoffs(prices: pd.DataFrame) -> None:
    """Look-ahead of 1 row before position 200 and of 4 rows from there on."""

    def growing_lookahead(df: pd.DataFrame) -> pd.DataFrame:
        before_200 = np.broadcast_to((np.arange(len(df)) < 200)[:, None], df.shape)
        early = pd.DataFrame(before_200, index=df.index, columns=df.columns)
        return df.shift(-1).where(early, df.shift(-4))

    report = check(growing_lookahead, prices)
    per_cutoff = {f.cutoff_pos: 0 for f in report.findings}
    assert len(per_cutoff) == len(report.cutoff_positions)
    assert report.horizon_rows == 4
    assert report.horizon_unbounded is False


def test_horizon_uses_the_earliest_row_over_columns_and_tests(prices: pd.DataFrame) -> None:
    """One column leaks 1 row ahead, another 3 rows; the horizon is that of the worst column."""

    def mixed(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"near": df["close"].shift(-1), "far": df["close"].shift(-3)}, index=df.index
        )

    report = check(mixed, prices)
    assert report.horizon_rows == 3
    by_column = {
        c: check(lambda df, c=c: mixed(df)[[c]], prices).horizon_rows for c in mixed(prices)
    }
    assert by_column == {"near": 1, "far": 3}


# --------------------------------------------------------------------------- #
# Unbounded horizon
# --------------------------------------------------------------------------- #

UNBOUNDED: dict[str, Fn] = {
    "global_zscore": global_zscore,
    "global_minmax": global_minmax,
    "rank_pct_over_time": rank_pct_over_time,
    "divide_by_last_row": divide_by_last_row,
}


@pytest.mark.parametrize(("mode", "cutoffs"), MODE_CUTOFFS)
@pytest.mark.parametrize("name", list(UNBOUNDED))
def test_whole_sample_statistics_have_an_unbounded_horizon(
    prices: pd.DataFrame, name: str, mode: Mode, cutoffs: int
) -> None:
    report = check(UNBOUNDED[name], prices, mode=mode, cutoffs=cutoffs)
    assert report.causal is False
    assert report.horizon_unbounded is True
    assert report.horizon_rows is not None
    assert "unbounded" in str(report)


@pytest.mark.parametrize("mode", MODES)
def test_standard_scaler_fitted_on_the_full_frame_has_an_unbounded_horizon(
    prices: pd.DataFrame, mode: Mode
) -> None:
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    def scale(df: pd.DataFrame) -> pd.DataFrame:
        scaled = preprocessing.StandardScaler().fit_transform(df)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    report = check(scale, prices, mode=mode, cutoffs=FEW)
    assert report.causal is False
    assert report.horizon_unbounded is True


def test_unbounded_is_decided_among_the_cutoffs_that_have_findings() -> None:
    """Min-max scaling only changes while the extremes lie after the cutoff.

    The maximum sits at row 100 and the minimum at row 50, so the scaling of
    the early rows changes only for cutoffs up to 100. Those are about a third of
    the cutoffs, and each of them reaches back to row 0.
    """
    data = make_prices()[["close"]].copy()
    data["close"] = 100.0 + np.random.default_rng(2).normal(0.0, 0.1, N_ROWS)
    data.iloc[100, 0] = 150.0
    data.iloc[50, 0] = 50.0
    report = check(global_minmax, data)
    flagged = {f.cutoff_pos for f in report.findings}
    assert flagged
    assert max(flagged) <= 100
    assert len(flagged) < len(report.cutoff_positions)
    assert report.horizon_unbounded is True


def test_a_function_that_looks_ahead_only_a_little_is_not_unbounded_on_a_short_sample() -> None:
    short = make_prices(n=60)
    report = check(shift_lead_3, short)
    assert report.horizon_rows == 3
    assert report.horizon_unbounded is False


def hit(cutoff: int, first: int) -> _Hit:
    return _Hit(
        test="truncate",
        cutoff=cutoff,
        column="x",
        kind="value_change",
        n_rows=1,
        first_pos=first,
        last_pos=first,
        max_abs_diff=None,
        first_label=str(first),
        last_label=str(first),
    )


def test_estimate_horizon_without_findings() -> None:
    assert _estimate_horizon([], 300) == (None, False)


def test_estimate_horizon_is_the_largest_distance_to_the_earliest_affected_row() -> None:
    hits = [hit(100, 99), hit(100, 97), hit(150, 149), hit(200, 196)]
    assert _estimate_horizon(hits, 300) == (4, False)


def test_estimate_horizon_unbounded_needs_more_than_half_of_the_cutoffs() -> None:
    early = [hit(60, 0), hit(90, 3)]  # horizons 60 and 87, both reaching into the first rows
    late = [hit(120, 119), hit(150, 149)]  # horizon 1 each
    assert _estimate_horizon(early + late, 300) == (87, False)  # exactly half: not a majority
    assert _estimate_horizon(early + late[:1], 300) == (87, True)
    assert _estimate_horizon([hit(60, 0), hit(90, 89), hit(120, 119)], 300) == (60, False)


def test_estimate_horizon_early_rows_are_the_first_five_percent() -> None:
    assert _estimate_horizon([hit(100, 14)], 300) == (86, True)
    assert _estimate_horizon([hit(100, 15)], 300) == (85, False)
    # On a very short sample, five percent is rounded up to at least one row.
    assert _estimate_horizon([hit(5, 0)], 10) == (5, True)
    assert _estimate_horizon([hit(5, 1)], 10) == (4, False)


def test_a_cutoff_close_to_the_start_does_not_vote_for_an_unbounded_horizon() -> None:
    """Near the start every affected row is an early row; that says nothing about the leak.

    The perturbation test may use cutoffs from row 2 on. A look-ahead of 3 rows touches
    row 0 at cutoff 3 and row 9 at cutoff 12: all are within the first 15 rows (5 % of 300).
    """
    hits = [hit(cutoff, max(0, cutoff - 3)) for cutoff in (4, 8, 12, 14)]
    assert _estimate_horizon(hits, 300) == (3, False)
    # the same hits far from the start would be a bounded leak anyway
    assert _estimate_horizon([hit(cutoff + 100, cutoff + 97) for cutoff in (4, 8)], 300) == (
        3,
        False,
    )


def test_a_cutoff_votes_only_when_its_horizon_exceeds_the_early_rows() -> None:
    n_early = math.ceil(0.05 * 300)
    assert _estimate_horizon([hit(n_early, 0)], 300) == (n_early, False)
    assert _estimate_horizon([hit(n_early + 1, 0)], 300) == (n_early + 1, True)


# --------------------------------------------------------------------------- #
# Label mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("mode", "cutoffs"), MODE_CUTOFFS)
@pytest.mark.parametrize("horizon", [5, 6, 10])
def test_forward_return_passes_when_the_horizon_is_large_enough(
    prices: pd.DataFrame, horizon: int, mode: Mode, cutoffs: int
) -> None:
    report = check_label(forward_return_5, prices, horizon=horizon, mode=mode, cutoffs=cutoffs)
    assert report.deterministic is True
    assert report.causal is True
    assert report.findings == []
    assert report.horizon_rows is None
    assert report.horizon_unbounded is False
    assert report.label_horizon == horizon
    assert f"label mode: allowed horizon = {horizon} row(s)" in str(report)


@pytest.mark.parametrize(("mode", "cutoffs"), MODE_CUTOFFS)
@pytest.mark.parametrize("horizon", [4, 3, 1, 0])
def test_forward_return_fails_when_the_horizon_is_too_small(
    prices: pd.DataFrame, horizon: int, mode: Mode, cutoffs: int
) -> None:
    report = check_label(forward_return_5, prices, horizon=horizon, mode=mode, cutoffs=cutoffs)
    assert report.causal is False
    assert report.findings
    assert report.horizon_rows == 5  # measured on the label itself, not after the shift
    assert report.horizon_unbounded is False
    assert report.label_horizon == horizon
    assert f"label mode: allowed horizon = {horizon} row(s)" in str(report)


def test_the_same_label_fails_the_plain_check_with_its_full_horizon(prices: pd.DataFrame) -> None:
    report = check(forward_return_5, prices)
    assert report.causal is False
    assert report.horizon_rows == 5
    assert report.label_horizon is None


@pytest.mark.parametrize("bars", [1, 2, 7, 12])
def test_label_passes_with_its_own_look_ahead_and_fails_with_one_row_less(
    prices: pd.DataFrame, bars: int
) -> None:
    def label(df: pd.DataFrame) -> pd.Series:
        return forward_return(df, bars)

    assert check_label(label, prices, horizon=bars).causal is True
    report = check_label(label, prices, horizon=bars - 1)
    assert report.causal is False
    assert report.horizon_rows == bars


def test_label_findings_only_cover_the_rows_beyond_the_allowed_look_ahead(
    prices: pd.DataFrame,
) -> None:
    """Horizon 4 and a label looking 5 ahead: only row p - 5 of each cutoff p is compared."""
    report = check_label(forward_return_5, prices, horizon=4)
    assert report.findings
    for finding in report.findings:
        first_bad_row = format_label(prices.index[finding.cutoff_pos - 5])
        assert finding.n_rows_affected == 1
        assert finding.first_affected_label == first_bad_row
        assert finding.last_affected_label == first_bad_row


def test_label_that_drops_unknown_rows_is_checked_through_missing_rows(
    prices: pd.DataFrame,
) -> None:
    ok = check_label(forward_return_5_dropna, prices, horizon=5)
    assert ok.causal is True
    assert ok.findings == []

    bad = check_label(forward_return_5_dropna, prices, horizon=4, mode="truncate")
    assert bad.causal is False
    assert {f.kind for f in bad.findings} == {"missing_row"}
    assert bad.horizon_rows == 5


def test_label_with_a_whole_sample_dependence_fails_for_every_horizon(
    prices: pd.DataFrame,
) -> None:
    for horizon in (0, 5, 40):
        report = check_label(global_zscore, prices, horizon=horizon)
        assert report.causal is False
        assert report.horizon_unbounded is True
        assert report.label_horizon == horizon


def test_a_causal_function_passes_every_horizon(prices: pd.DataFrame) -> None:
    report = check_label(lambda df: df.rolling(10).mean(), prices, horizon=3)
    assert report.causal is True
    assert report.label_horizon == 3


def test_label_check_with_horizon_zero_is_the_plain_check(prices: pd.DataFrame) -> None:
    label = check_label(shift_lead_3, prices, horizon=0)
    plain = check(shift_lead_3, prices)
    assert label.findings == plain.findings
    assert label.horizon_rows == plain.horizon_rows == 3
    assert label.cutoff_positions == plain.cutoff_positions
    assert label.label_horizon == 0
    assert plain.label_horizon is None


def test_label_check_reports_a_function_that_is_not_deterministic(prices: pd.DataFrame) -> None:
    rng = np.random.default_rng(4)
    report = check_label(lambda df: df + rng.normal(size=df.shape), prices, horizon=2)
    assert report.deterministic is False
    assert report.causal is None
    assert report.label_horizon == 2


def test_label_check_accepts_the_options_of_check(prices: pd.DataFrame) -> None:
    report = check_label(
        forward_return_5,
        prices,
        horizon=5,
        cutoffs=[100, 200],
        mode="truncate",
        seed=3,
        min_rows=10,
    )
    assert report.causal is True
    assert report.cutoff_positions == [100, 200]
    assert report.n_evaluations == 2 + 2


def test_numpy_integers_are_accepted_and_reported_as_plain_integers(prices: pd.DataFrame) -> None:
    report = check_label(
        forward_return_5,
        prices,
        horizon=np.int64(5),
        cutoffs=np.int64(4),
        seed=np.int64(2),
        min_rows=np.int64(40),
    )
    assert report.causal is True
    assert type(report.label_horizon) is int
    assert all(type(p) is int for p in report.cutoff_positions)
    assert (
        min(_plan_cutoffs(N_ROWS, 4, ("truncate",), min_rows=40, lookahead=5, seed=2).union) >= 40
    )


@pytest.mark.parametrize("horizon", [-1, 2.5, True, "5"])
def test_invalid_label_horizon_is_rejected(prices: pd.DataFrame, horizon: Any) -> None:
    with pytest.raises(PeekproofError, match="horizon"):
        check_label(forward_return_5, prices, horizon=horizon)


def test_label_horizon_must_leave_room_for_a_cutoff(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="not enough rows"):
        check_label(forward_return_5, prices, horizon=N_ROWS)


def test_explicit_cutoffs_at_or_below_the_horizon_are_dropped_with_a_note(
    prices: pd.DataFrame,
) -> None:
    report = check_label(forward_return_5, prices, horizon=5, cutoffs=[3, 5, 50, 100])
    assert report.causal is True
    assert report.cutoff_positions == [50, 100]
    assert report.n_evaluations == 2 + 2 * 2
    assert report.notes[0] == (
        "dropped cutoff position(s) 3, 5: at or below the horizon of 5 row(s), so they would "
        "compare nothing"
    )
    assert "note: dropped cutoff position(s) 3, 5" in str(report)


def test_explicit_cutoffs_that_are_all_at_or_below_the_horizon_are_an_error(
    prices: pd.DataFrame,
) -> None:
    """They compare nothing, so the label used to pass without being tested at all."""
    with pytest.raises(PeekproofError, match="at or below the horizon of 5"):
        check_label(forward_return_5, prices, horizon=5, cutoffs=[3, 5])
    # A label that leaks beyond the allowed horizon, tested only at hopeless cutoffs:
    with pytest.raises(PeekproofError, match="compare nothing"):
        check_label(forward_return_5, prices, horizon=4, cutoffs=[3, 4])


def test_the_first_cutoff_above_the_horizon_does_compare_a_row(prices: pd.DataFrame) -> None:
    """Position horizon + 1 leaves exactly row 0 to compare."""
    leaky = check_label(forward_return_5, prices, horizon=4, cutoffs=[5])
    assert leaky.causal is False
    assert {f.cutoff_pos for f in leaky.findings} == {5}
    assert {f.first_affected_label for f in leaky.findings} == {format_label(prices.index[0])}
    assert check_label(forward_return_5, prices, horizon=5, cutoffs=[6]).causal is True


def test_a_note_for_dropped_cutoffs_comes_before_the_coverage_note(prices: pd.DataFrame) -> None:
    report = check_label(forward_return_5, prices, horizon=5, cutoffs=[2, 100])
    assert [note.split(" ")[0] for note in report.notes] == ["dropped", "cutoffs"]


# --------------------------------------------------------------------------- #
# Cutoff selection
# --------------------------------------------------------------------------- #


def plan_positions(
    n_rows: int,
    count: int,
    *,
    tests: tuple[str, ...] = BOTH,
    min_rows: int | None = None,
    lookahead: int = 0,
    seed: int = 0,
) -> dict[str, list[int]]:
    plan = _plan_cutoffs(
        n_rows,
        count,
        tests,  # type: ignore[arg-type]
        min_rows=min_rows,
        lookahead=lookahead,
        seed=seed,
    )
    return dict(plan.positions)


def test_default_plan_for_300_rows() -> None:
    positions = plan_positions(300, 64)
    assert set(positions) == {"truncate", "perturb"}
    for chosen in positions.values():
        assert len(chosen) == 64
        assert chosen == sorted(set(chosen))
        assert {299, 298, 150} <= set(chosen)
    assert min(positions["truncate"]) >= 30  # a tenth of the rows
    assert min(positions["perturb"]) <= 8  # starts at row 2, within the first stretch
    assert max(positions["perturb"]) == max(positions["truncate"]) == 299


def test_the_plan_of_a_single_test_has_only_that_test() -> None:
    assert set(plan_positions(300, 64, tests=("truncate",))) == {"truncate"}
    assert set(plan_positions(300, 64, tests=("perturb",))) == {"perturb"}


def test_the_union_of_the_plan_is_sorted_and_without_duplicates() -> None:
    plan = _plan_cutoffs(300, 64, BOTH, min_rows=None, lookahead=0, seed=0)  # type: ignore[arg-type]
    both = set(plan.positions["truncate"]) | set(plan.positions["perturb"])
    assert plan.union == sorted(both)
    assert 64 < len(plan.union) <= 2 * 64


@pytest.mark.parametrize("count", [1, 3, 4, 8, 20, 64])
@pytest.mark.parametrize("n_rows", [3, 4, 10, 20, 57, 100, 300, 1000])
def test_selected_cutoffs_follow_the_rules(n_rows: int, count: int) -> None:
    hi = n_rows - 1
    for test, lo in (("truncate", max(2, math.ceil(0.1 * n_rows))), ("perturb", 2)):
        positions = plan_positions(n_rows, count, tests=(test,))[test]
        always = {p for p in (hi, n_rows - 2, n_rows // 2) if lo <= p <= hi}
        assert positions == sorted(set(positions))
        assert all(lo <= p <= hi for p in positions)
        assert always <= set(positions)
        assert hi in positions
        assert len(positions) == min(max(count, len(always)), hi - lo + 1)


def test_cutoffs_are_spread_evenly_over_the_range() -> None:
    """One cutoff in each of 61 equal stretches of the 295 free positions: no large holes."""
    positions = plan_positions(300, 64, tests=("perturb",))["perturb"]
    gaps = np.diff(positions)
    stretch = math.ceil(295 / 61)
    assert gaps.max() <= 2 * stretch
    assert positions[0] <= 2 + stretch
    # every tenth of the range holds its share of the cutoffs
    counts, _ = np.histogram(positions, bins=10, range=(2, 300))
    assert counts.min() >= 4
    assert counts.max() <= 9


def test_cutoffs_are_jittered_so_that_they_do_not_share_a_phase() -> None:
    positions = plan_positions(1_000, 64, tests=("perturb",))["perturb"]
    assert len(set(np.diff(positions[:-3]).tolist())) >= 5  # not equally spaced
    # A feature that repeats every 21 rows (a month) meets cutoffs at many different offsets.
    assert len({p % 21 for p in positions}) >= 14
    assert len({p % 5 for p in positions}) == 5
    # and the positions are not a regular grid shifted by a constant
    spacing = np.diff(positions[:-3])
    assert spacing.std() > 1.0


def test_the_cutoffs_depend_on_the_seed_and_only_on_the_seed() -> None:
    first = plan_positions(300, 64, seed=1)
    assert first == plan_positions(300, 64, seed=1)
    other = plan_positions(300, 64, seed=2)
    assert first["truncate"] != other["truncate"]
    assert first["perturb"] != other["perturb"]
    # the fixed positions do not move
    assert {299, 298, 150} <= set(other["truncate"]) & set(other["perturb"])


def test_the_two_tests_draw_their_own_positions() -> None:
    positions = plan_positions(300, 64)
    assert positions["truncate"] != positions["perturb"]


def test_min_rows_raises_the_first_truncation_cutoff_only() -> None:
    positions = plan_positions(300, 8, min_rows=100)
    assert min(positions["truncate"]) >= 100
    assert {299, 298, 150} <= set(positions["truncate"])
    assert min(positions["perturb"]) < 100  # the perturbation test has no minimum history
    # Fixed positions below the shortest history are dropped.
    positions = plan_positions(300, 8, min_rows=200)
    assert min(positions["truncate"]) >= 200
    assert 150 not in positions["truncate"]
    assert {299, 298} <= set(positions["truncate"])
    assert 150 in positions["perturb"]


def test_a_large_min_rows_leaves_the_truncation_test_with_every_row_of_its_range() -> None:
    positions = plan_positions(300, 64, min_rows=250)["truncate"]
    assert positions == list(range(250, 300))


def test_label_mode_never_picks_a_cutoff_that_leaves_nothing_to_compare() -> None:
    positions = plan_positions(300, 8, lookahead=40)
    assert min(positions["truncate"]) >= 41
    assert min(positions["perturb"]) >= 41
    assert min(plan_positions(300, 64, tests=("perturb",), lookahead=40)["perturb"]) <= 41 + 5


def test_explicit_cutoffs_are_validated_and_sorted_and_used_by_every_test() -> None:
    plan = _plan_cutoffs(100, [60, 10, 60, 99], BOTH, min_rows=50, lookahead=0, seed=0)  # type: ignore[arg-type]
    assert plan.positions == {"truncate": [10, 60, 99], "perturb": [10, 60, 99]}
    assert plan.notes == []
    for bad in ([1], [100], [0, 50], [50.0], [True]):
        with pytest.raises(PeekproofError):
            _plan_cutoffs(100, bad, BOTH, min_rows=None, lookahead=0, seed=0)  # type: ignore[arg-type]


def test_explicit_cutoffs_at_or_below_the_look_ahead_are_dropped_in_the_plan() -> None:
    plan = _plan_cutoffs(100, [3, 5, 6, 60], BOTH, min_rows=None, lookahead=5, seed=0)  # type: ignore[arg-type]
    assert plan.union == [6, 60]
    assert len(plan.notes) == 1
    assert "3, 5" in plan.notes[0]
    with pytest.raises(PeekproofError, match="at or below the horizon"):
        _plan_cutoffs(100, [3, 5], BOTH, min_rows=None, lookahead=5, seed=0)  # type: ignore[arg-type]


def test_too_short_data_has_no_cutoffs() -> None:
    assert plan_positions(3, 8) == {"truncate": [2], "perturb": [2]}
    with pytest.raises(PeekproofError, match="not enough rows"):
        plan_positions(2, 8)
    with pytest.raises(PeekproofError, match=r"not enough rows.*truncate"):
        plan_positions(300, 8, min_rows=300)
    # min_rows only limits the truncation test, so the perturbation test can still run
    assert plan_positions(300, 8, tests=("perturb",), min_rows=300)["perturb"][-1] == 299


@pytest.mark.parametrize("bad", [0, -1])
def test_a_count_below_one_is_rejected(bad: int) -> None:
    with pytest.raises(PeekproofError, match="cutoffs must be >= 1"):
        plan_positions(300, bad)


def test_spread_cutoffs_gives_every_position_when_the_range_has_no_more_room() -> None:
    rng = np.random.default_rng(0)
    assert _spread_cutoffs(12, 64, 2, rng) == list(range(2, 12))
    assert _spread_cutoffs(12, 10, 2, rng) == list(range(2, 12))
    assert len(_spread_cutoffs(12, 9, 2, rng)) == 9


def test_spread_cutoffs_never_gives_fewer_than_the_fixed_positions() -> None:
    rng = np.random.default_rng(0)
    assert _spread_cutoffs(300, 1, 2, rng) == [150, 298, 299]
    assert _spread_cutoffs(300, 3, 2, rng) == [150, 298, 299]
    assert len(_spread_cutoffs(300, 4, 2, rng)) == 4


def test_spread_cutoffs_steps_over_the_fixed_positions() -> None:
    """The stretch that holds a fixed position still yields a distinct position next to it."""
    for seed in range(30):
        positions = _spread_cutoffs(60, 55, 2, np.random.default_rng(seed))
        assert len(positions) == 55
        assert positions == sorted(set(positions))
        assert {59, 58, 30} <= set(positions)


# --------------------------------------------------------------------------- #
# Sparse and early leaks are found
# --------------------------------------------------------------------------- #


def month_end_next_close(df: pd.DataFrame) -> pd.DataFrame:
    """The next day's close on the last trading day of each month, 0 elsewhere."""
    month = pd.Series(df.index.to_period("M"), index=df.index)
    last_of_month = month != month.shift(-1)
    leak = df["close"].shift(-1).where(last_of_month, 0.0).fillna(0.0)
    return leak.rename("feature").to_frame()


def make_t_returns(n: int = N_ROWS, seed: int = 3) -> pd.DataFrame:
    """Daily returns with heavy tails: Student-t with 3 degrees of freedom, sd of 1 %."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"ret": 0.01 * rng.standard_t(3, n) / math.sqrt(3.0)}, index=index)


def big_drop_next_day(df: pd.DataFrame) -> pd.DataFrame:
    """A flag that is true when the next return is below -2 %: about 2 % of the days."""
    return (df["ret"].shift(-1) < -0.02).rename("flag").to_frame()


def make_events(n: int = N_ROWS, seed: int = 5, dtype: str = "bool") -> pd.DataFrame:
    """A rare event column: about 3 % of the rows are one."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"event": (rng.random(n) < 0.03).astype(dtype)}, index=index)


def event_tomorrow(df: pd.DataFrame) -> pd.DataFrame:
    """The event of the next row; the last row gets 0 instead of a missing value."""
    column = df["event"]
    return column.shift(-1, fill_value=column.dtype.type(0)).rename("tomorrow").to_frame()


def short_leak_in_the_first_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows 0-24 take the close of 5 rows ahead; all other rows are causal."""
    out = df["close"].copy()
    out.iloc[:25] = df["close"].shift(-5).iloc[:25]
    return out.rename("x").to_frame()


def share_found(
    fn: Fn, make: Callable[[int], pd.DataFrame], lengths: range, **kwargs: Any
) -> float:
    """Share of the sample lengths for which the oracle reports the leak."""
    found = sum(check(fn, make(n), **kwargs).causal is False for n in lengths)
    return found / len(lengths)


def test_month_end_leak_on_the_next_days_close_is_found_with_the_defaults(
    prices: pd.DataFrame,
) -> None:
    report = check(month_end_next_close, prices)
    assert report.causal is False
    assert report.horizon_rows == 1
    # The leak sits on the last day of a month, and only the cutoff right behind it sees it.
    month = pd.Series(prices.index.to_period("M"), index=prices.index)
    last_of_month = month != month.shift(-1)
    for finding in report.findings:
        assert last_of_month.iloc[finding.cutoff_pos - 1]
    assert len({f.cutoff_pos for f in report.findings}) >= 2


def test_month_end_leak_is_found_for_almost_every_sample_length() -> None:
    """Eight cutoffs spread geometrically found it on about a third of the sample lengths."""
    lengths = range(230, 300, 7)
    assert share_found(month_end_next_close, make_prices, lengths) >= 0.9


def test_flag_on_the_next_return_is_found_with_the_defaults() -> None:
    report = check(big_drop_next_day, make_t_returns())
    assert report.causal is False
    assert report.horizon_rows == 1
    assert report.affected_columns == ["flag"]


def test_flag_on_the_next_return_is_found_for_most_sample_lengths() -> None:
    """The flag is true on 2 % of the days; eight cutoffs found it on a third of the lengths."""
    lengths = range(240, 300, 6)
    assert share_found(big_drop_next_day, make_t_returns, lengths) >= 0.8


def test_leak_confined_to_the_first_rows_is_found(prices: pd.DataFrame) -> None:
    """Rows 0-24 look 5 rows ahead. No cutoff of the old kind (10 % of 300 rows is 30) saw them."""
    report = check(short_leak_in_the_first_rows, prices)
    assert report.causal is False
    cutoffs = {f.cutoff_pos for f in report.findings}
    assert cutoffs
    assert max(cutoffs) <= 29  # the last row that looks ahead is row 24, which needs row 29
    assert report.horizon_rows == 5
    # The leak is confined to the start of the sample; that is not a whole-sample dependence.
    assert report.horizon_unbounded is False
    assert {f.test for f in report.findings} == {"perturb"}


def test_leak_confined_to_the_first_rows_is_not_seen_by_truncation_that_starts_at_a_tenth(
    prices: pd.DataFrame,
) -> None:
    """The truncation test keeps a minimum history; the perturbation test reaches in."""
    assert check(short_leak_in_the_first_rows, prices, mode="truncate").causal is True
    assert check(short_leak_in_the_first_rows, prices, mode="perturb").causal is False
    # Explicit cutoffs are used as given, so the truncation test finds it with short histories.
    short = check(short_leak_in_the_first_rows, prices, mode="truncate", cutoffs=[10, 20, 29])
    assert short.causal is False


def test_event_flag_shifted_by_one_row_is_found_with_the_defaults() -> None:
    report = check(event_tomorrow, make_events())
    assert report.causal is False
    assert report.horizon_rows == 1


def test_event_flag_shifted_by_one_row_is_found_for_almost_every_sample_length() -> None:
    lengths = range(230, 300, 10)
    assert share_found(event_tomorrow, make_events, lengths) >= 0.9
    assert share_found(event_tomorrow, lambda n: make_events(n, dtype="int8"), lengths) >= 0.9


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int64", "float64"])
def test_the_perturbation_changes_a_rare_event_at_every_cutoff(dtype: str) -> None:
    """Rounded noise left an int8 flag as it was; every perturbed cutoff now sees the shift."""
    report = check(event_tomorrow, make_events(dtype=dtype), mode="perturb")
    assert {f.cutoff_pos for f in report.findings} == set(report.cutoff_positions)


def test_few_cutoffs_miss_what_many_cutoffs_find(prices: pd.DataFrame) -> None:
    """A leak that touches one row in 21 is seen only by a cutoff right behind that row."""
    old_style = [30, 45, 67, 100, 150, 211, 298, 299]
    assert check(month_end_next_close, prices, cutoffs=old_style).causal is True
    assert check(month_end_next_close, prices).causal is False


def test_more_cutoffs_find_sparser_leaks() -> None:
    """A leak on one row in 61 needs about as many cutoffs as there are rows per leaky row."""
    n = 600
    data = make_prices(n)
    leaky_rows = np.arange(10, n - 1, 61)  # seen only by the cutoffs 11, 72, 133, ...

    def sparse_leak(df: pd.DataFrame) -> pd.DataFrame:
        position = np.arange(len(df))
        leak = df["close"].shift(-1).where(np.isin(position, leaky_rows), 0.0).fillna(0.0)
        return leak.rename("feature").to_frame()

    assert check(sparse_leak, data, cutoffs=4).causal is True
    assert check(sparse_leak, data, cutoffs=256).causal is False


def test_clean_functions_stay_clean_with_many_cutoffs(prices: pd.DataFrame) -> None:
    for fn in (
        lambda df: df.rolling(10).mean(),
        lambda df: (df - df.expanding().mean()) / df.expanding().std(),
    ):
        report = check(fn, prices, cutoffs=100)
        assert report.causal is True
        assert report.notes == []
