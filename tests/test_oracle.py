"""Tests for the dynamic causality oracle: verdicts, findings and robustness.

Horizon values, label mode, cutoff selection and the detection of sparse or early leaks are
covered in ``test_oracle_horizon.py``. All data is synthetic and seeded, so every test is
deterministic.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import peekproof.oracle as oracle
from peekproof import _compat
from peekproof._compat import format_label
from peekproof.errors import LeakError, PeekproofError
from peekproof.oracle import (
    Mode,
    _add_noise,
    _flip_booleans,
    _offset_integers,
    _perturb,
    _swap_categories,
    assert_causal,
    check,
)
from peekproof.report import OracleReport

Fn = Callable[[pd.DataFrame], pd.DataFrame | pd.Series]

N_ROWS = 300
#: Cutoffs per test for the tests whose outcome does not depend on how many there are.
FEW = 12
#: ``(mode, cutoffs)``: every mode, with the default number of cutoffs only where both tests run.
MODE_CUTOFFS: list[tuple[Mode, int]] = [("both", 64), ("truncate", FEW), ("perturb", FEW)]


# --------------------------------------------------------------------------- #
# Data and helpers
# --------------------------------------------------------------------------- #


def make_prices(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Business-day frame with a float price column and an integer volume column."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    volume = rng.integers(1_000, 5_000, n)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def make_wide(n: int = N_ROWS, assets: int = 10, seed: int = 1) -> pd.DataFrame:
    """Business-day frame with one price column per asset."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (n, assets)), axis=0))
    return pd.DataFrame(prices, index=index, columns=[f"A{i:02d}" for i in range(assets)])


def make_minute_bars(n: int = 400, seed: int = 2) -> pd.DataFrame:
    """One-minute bars with a price and a volume column."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-03-01 09:00", periods=n, freq="1min", name="t")
    price = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    return pd.DataFrame({"price": price, "volume": rng.integers(1, 100, n)}, index=index)


class Counted:
    """A function that counts how often it is called."""

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, df: Any) -> Any:
        self.calls += 1
        return self.fn(df)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide()


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_minute_bars()


def assert_clean(report: OracleReport, n_rows: int = N_ROWS) -> None:
    assert report.deterministic is True
    assert report.causal is True
    assert report.findings == []
    assert report.horizon_rows is None
    assert report.horizon_unbounded is False
    assert report.notes == []
    assert report.n_rows == n_rows


def assert_leaky(report: OracleReport) -> None:
    assert report.deterministic is True
    assert report.causal is False
    assert report.findings
    assert str(report).startswith("peekproof: NOT CAUSAL")


# --------------------------------------------------------------------------- #
# Clean functions: must be reported as causal
# --------------------------------------------------------------------------- #


def rolling_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(10).mean()


def ewm_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.ewm(span=10).mean()


def expanding_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.expanding().mean()) / df.expanding().std()


def lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change().shift(1)


def merge_asof_backward(df: pd.DataFrame) -> pd.DataFrame:
    """Join a weekly series built from the input onto the daily rows, looking back only."""
    weekly = df["close"].resample("W").last().rename("weekly_close").to_frame()
    return pd.merge_asof(df, weekly, left_index=True, right_index=True, direction="backward")


def column_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[["close"]].rolling(5).mean()


def drop_warmup_nans(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(20).mean().dropna()


CLEAN: dict[str, Fn] = {
    "rolling_mean": rolling_mean,
    "ewm_mean": ewm_mean,
    "expanding_zscore": expanding_zscore,
    "lagged_returns": lagged_returns,
    "merge_asof_backward": merge_asof_backward,
    "column_subset": column_subset,
    "drop_warmup_nans": drop_warmup_nans,
}


@pytest.mark.parametrize(("mode", "cutoffs"), MODE_CUTOFFS)
@pytest.mark.parametrize("name", list(CLEAN))
def test_clean_functions_are_causal(
    prices: pd.DataFrame, name: str, mode: Mode, cutoffs: int
) -> None:
    report = check(CLEAN[name], prices, mode=mode, cutoffs=cutoffs)
    assert_clean(report)


@pytest.mark.parametrize("mode", ["truncate", "perturb", "both"])
def test_cross_sectional_rank_on_a_wide_frame_is_causal(wide: pd.DataFrame, mode: Mode) -> None:
    assert wide.shape == (N_ROWS, 10)
    report = check(lambda df: df.rank(axis=1), wide, mode=mode, cutoffs=FEW)
    assert_clean(report)


def test_function_that_adds_columns_is_causal(prices: pd.DataFrame) -> None:
    """Extra output columns are fine as long as the index is a subset of the input index."""

    def with_extra(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(avg=df["close"].rolling(3).mean(), flag=df["close"] > df["close"].shift(1))

    assert_clean(check(with_extra, prices, cutoffs=FEW))


def test_clean_non_numeric_and_boolean_outputs_are_causal(prices: pd.DataFrame) -> None:
    def labels(df: pd.DataFrame) -> pd.DataFrame:
        up = df["close"] > df["close"].shift(1)
        return pd.DataFrame(
            {
                "regime": np.where(up, "up", "down"),
                "up": up,
                "seen": df.index.dayofweek.astype(str),
            },
            index=df.index,
        )

    assert_clean(check(labels, prices, cutoffs=FEW))


def test_function_with_missing_values_in_the_input_is_causal(prices: pd.DataFrame) -> None:
    gappy = prices.astype("float64")
    gappy.iloc[[10, 11, 50, 120, 121, 122, 250], 0] = np.nan
    assert_clean(check(lambda df: df.ffill().rolling(5, min_periods=1).mean(), gappy, cutoffs=FEW))


def test_seeded_randomness_inside_the_function_is_deterministic(prices: pd.DataFrame) -> None:
    """The remedy suggested for non-deterministic functions: seed them."""

    def seeded_noise(df: pd.DataFrame) -> pd.DataFrame:
        return df + np.random.default_rng(7).normal(size=df.shape)

    assert_clean(check(seeded_noise, prices, cutoffs=FEW))


# --------------------------------------------------------------------------- #
# Leaky functions: must be reported as not causal
# --------------------------------------------------------------------------- #


def shift_lead_1(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1)


def shift_lead_3(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-3)


def lead_1_then_dropna(df: pd.DataFrame) -> pd.DataFrame:
    return df.shift(-1).dropna()


def centered_window_5(df: pd.DataFrame) -> pd.DataFrame:
    return df.rolling(5, center=True).mean()


def global_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def global_minmax(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def rank_pct_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(pct=True)


def loop_reads_next_row(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    values = [
        close.iloc[i + 1] - close.iloc[i] if i + 1 < len(close) else np.nan
        for i in range(len(close))
    ]
    return pd.Series(values, index=df.index, name="step")


def pct_change_lead_2(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(-2)


def merge_asof_forward(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df["close"].resample("W").last().rename("weekly_close").to_frame()
    return pd.merge_asof(df, weekly, left_index=True, right_index=True, direction="forward")


LEAKY: dict[str, Fn] = {
    "shift_lead_1": shift_lead_1,
    "shift_lead_3": shift_lead_3,
    "lead_1_then_dropna": lead_1_then_dropna,
    "centered_window_5": centered_window_5,
    "global_zscore": global_zscore,
    "global_minmax": global_minmax,
    "rank_pct_over_time": rank_pct_over_time,
    "loop_reads_next_row": loop_reads_next_row,
    "pct_change_lead_2": pct_change_lead_2,
    "merge_asof_forward": merge_asof_forward,
}


@pytest.mark.parametrize("name", list(LEAKY))
def test_leaky_functions_are_not_causal(prices: pd.DataFrame, name: str) -> None:
    assert_leaky(check(LEAKY[name], prices))


@pytest.mark.parametrize("mode", ["truncate", "perturb"])
@pytest.mark.parametrize("name", ["shift_lead_1", "centered_window_5", "global_zscore"])
def test_each_test_alone_catches_value_leaks(prices: pd.DataFrame, name: str, mode: Mode) -> None:
    report = check(LEAKY[name], prices, mode=mode, cutoffs=FEW)
    assert_leaky(report)
    assert {f.test for f in report.findings} == {mode}


def test_standard_scaler_fitted_on_the_full_frame_is_not_causal(prices: pd.DataFrame) -> None:
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    def scale(df: pd.DataFrame) -> pd.DataFrame:
        scaled = preprocessing.StandardScaler().fit_transform(df)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    assert_leaky(check(scale, prices, cutoffs=FEW))


def test_bfill_is_not_causal() -> None:
    """A gap that ends right at a cutoff is filled from the future."""
    cutoffs = [50, 100, 150, 200, 250]
    data = make_prices().astype("float64")
    for p in cutoffs:
        data.iloc[[p - 2, p - 1], 0] = np.nan
    report = check(lambda df: df.bfill(), data, cutoffs=cutoffs)
    assert_leaky(report)
    assert {f.kind for f in report.findings} == {"value_change"}


def test_bfill_with_random_gaps_is_not_causal() -> None:
    data = make_prices().astype("float64")
    gaps = np.random.default_rng(5).random(N_ROWS) < 0.4
    data.loc[gaps, "close"] = np.nan
    assert_leaky(check(lambda df: df.bfill(), data, cutoffs=FEW))


def test_assert_causal_raises_leak_error_with_the_report(prices: pd.DataFrame) -> None:
    with pytest.raises(LeakError) as caught:
        assert_causal(shift_lead_1, prices, cutoffs=FEW)
    error = caught.value
    assert isinstance(error, AssertionError)
    assert error.report.causal is False
    assert str(error) == str(error.report)
    assert "NOT CAUSAL" in str(error)


def test_assert_causal_passes_silently_for_a_clean_function(prices: pd.DataFrame) -> None:
    assert assert_causal(rolling_mean, prices, cutoffs=FEW) is None


# --------------------------------------------------------------------------- #
# Kinds and details of findings
# --------------------------------------------------------------------------- #


def test_shift_then_dropna_is_found_through_missing_rows(prices: pd.DataFrame) -> None:
    report = check(lead_1_then_dropna, prices, mode="truncate")
    assert_leaky(report)
    missing = [f for f in report.findings if f.kind == "missing_row"]
    assert missing
    assert {f.kind for f in report.findings} == {"missing_row"}
    for finding in missing:
        last_kept = format_label(prices.index[finding.cutoff_pos - 1])
        assert finding.test == "truncate"
        assert finding.cutoff_label == last_kept
        assert finding.first_affected_label == last_kept
        assert finding.last_affected_label == last_kept
        assert finding.n_rows_affected == 1
        assert finding.max_abs_diff is None


def test_shift_then_dropna_also_found_with_both_tests(prices: pd.DataFrame) -> None:
    report = check(lead_1_then_dropna, prices, cutoffs=FEW)
    assert any(f.kind == "missing_row" and f.test == "truncate" for f in report.findings)
    # The perturbation keeps every row, so it can only see changed values.
    perturbed = [f for f in report.findings if f.test == "perturb"]
    assert perturbed
    assert {f.kind for f in perturbed} == {"value_change"}


def test_value_change_findings_carry_labels_and_differences(prices: pd.DataFrame) -> None:
    report = check(shift_lead_1, prices, mode="perturb")
    assert report.cutoff_positions
    for finding in report.findings:
        last_kept = format_label(prices.index[finding.cutoff_pos - 1])
        assert finding.kind == "value_change"
        assert finding.cutoff_label == last_kept
        assert finding.first_affected_label == last_kept
        assert finding.n_rows_affected == 1
        assert finding.max_abs_diff is not None
        assert finding.max_abs_diff > 0.0
    assert report.affected_columns == ["close", "volume"]


def test_truncation_reports_no_difference_when_a_value_meets_a_missing_value(
    prices: pd.DataFrame,
) -> None:
    """At the cutoff the leaky value is missing in the shortened output: no finite difference."""
    report = check(shift_lead_1, prices, mode="truncate")
    assert report.findings
    assert all(f.max_abs_diff is None for f in report.findings)


def test_columns_missing_from_the_shortened_output_count_as_missing_rows() -> None:
    data = make_prices().astype("float64")
    data["late"] = np.where(np.arange(N_ROWS) >= 200, 1.0, np.nan)
    report = check(lambda df: df.dropna(axis=1, how="all"), data, cutoffs=FEW)
    assert_leaky(report)
    late = [f for f in report.findings if f.column == "late"]
    assert late
    assert {(f.kind, f.test) for f in late} == {("missing_row", "truncate")}
    # The column is absent for every row that the full output has before the cutoff.
    assert all(f.n_rows_affected == f.cutoff_pos for f in late)
    assert all(f.first_affected_label == format_label(data.index[0]) for f in late)
    # The columns that exist in both outputs are not affected.
    assert {f.column for f in report.findings} == {"late"}


def test_row_that_only_the_shortened_output_has_is_a_missing_row_finding(
    prices: pd.DataFrame,
) -> None:
    """A filter on the next value keeps the last row of a short history but not of the full one."""

    def keep_unless_next_is_up(df: pd.DataFrame) -> pd.DataFrame:
        return df[~(df["close"].shift(-1) > df["close"])]

    close = prices["close"]
    ups = np.flatnonzero((close.shift(-1) > close).to_numpy())
    cutoffs = [int(u) + 1 for u in ups if 50 < u < 290][:3]
    report = check(keep_unless_next_is_up, prices, cutoffs=cutoffs, mode="truncate")
    assert_leaky(report)
    assert {f.kind for f in report.findings} == {"missing_row"}
    assert {f.cutoff_pos for f in report.findings} == set(cutoffs)


def test_non_numeric_leak_is_a_value_change_without_a_difference(prices: pd.DataFrame) -> None:
    def next_move(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"move": np.where(df["close"].shift(-1) > df["close"], "up", "down")}, index=df.index
        )

    report = check(next_move, prices)
    assert_leaky(report)
    assert {f.kind for f in report.findings} == {"value_change"}
    assert {f.max_abs_diff for f in report.findings} == {None}
    assert report.affected_columns == ["move"]


def test_boolean_leak_is_detected(prices: pd.DataFrame) -> None:
    def next_is_up(df: pd.DataFrame) -> pd.Series:
        return (df["close"].shift(-1) > df["close"]).rename("next_up")

    report = check(next_is_up, prices)
    assert_leaky(report)
    assert report.affected_columns == ["next_up"]


def test_findings_are_limited_to_the_columns_that_leak(prices: pd.DataFrame) -> None:
    def one_leaky_column(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"ok": df["close"].rolling(3).mean(), "bad": df["close"].shift(-1)}, index=df.index
        )

    report = check(one_leaky_column, prices, cutoffs=FEW)
    assert_leaky(report)
    assert report.affected_columns == ["bad"]


def test_tolerances_decide_what_counts_as_a_difference(prices: pd.DataFrame) -> None:
    """The output depends on the input length by a relative 1e-12 per row."""

    def scaled_by_length(df: pd.DataFrame) -> pd.DataFrame:
        return df * (1.0 + 1e-12 * len(df))

    assert_clean(check(scaled_by_length, prices, cutoffs=FEW))
    strict = check(scaled_by_length, prices, atol=0.0, rtol=0.0, cutoffs=FEW)
    assert_leaky(strict)


# --------------------------------------------------------------------------- #
# The absolute tolerance is relative to the scale of the column
# --------------------------------------------------------------------------- #


def make_small_returns(n: int = N_ROWS, seed: int = 4) -> pd.DataFrame:
    """Returns of the size of 1e-5, as for a very short bar."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"ret": rng.normal(0.0, 1e-5, n)}, index=index)


def centered_mean_of_squares(df: pd.DataFrame) -> pd.DataFrame:
    """A centered window of squared returns: values around 1e-10 that look 2 rows ahead."""
    squares = df["ret"] ** 2
    return squares.rolling(5, center=True, min_periods=1).mean().rename("var").to_frame()


@pytest.mark.parametrize("mode", ["truncate", "perturb", "both"])
def test_leak_in_a_column_of_tiny_numbers_is_found(mode: Mode) -> None:
    """The values (about 1e-10) are far below the default absolute tolerance of 1e-9."""
    returns = make_small_returns()
    report = check(centered_mean_of_squares, returns, mode=mode, cutoffs=FEW)
    assert_leaky(report)
    assert report.horizon_rows == 2
    assert report.affected_columns == ["var"]
    assert report.findings[0].max_abs_diff is not None
    assert report.findings[0].max_abs_diff < 1e-9


def test_tiny_numbers_that_do_not_leak_are_causal() -> None:
    returns = make_small_returns()
    clean = check(
        lambda df: (df["ret"] ** 2).rolling(5, min_periods=1).mean().to_frame("var"), returns
    )
    assert_clean(clean)


def test_absolute_tolerance_does_not_flag_rounding_noise_next_to_large_numbers() -> None:
    """Differences of 1e-7 near zero are noise next to values of 1000: scale 1e3 times atol 1e-9.

    The output depends on the input length by 1e-9 per row on the rows that are near zero. A
    fixed tolerance of 1e-9 would call that a leak; relative to the median of 1000 it is not.
    """
    data = make_prices()
    near_zero = np.arange(N_ROWS) % 4 == 0

    def large_with_dust(df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        values = np.where(near_zero[:n], 1e-9 * n, 1000.0)
        return pd.DataFrame({"x": values}, index=df.index)

    assert_clean(check(large_with_dust, data, cutoffs=FEW))
    # The tolerance is still a tolerance: with a scale of 1 the same dust is a difference.
    unit_scale = check(
        lambda df: large_with_dust(df).clip(upper=1.0), data, cutoffs=FEW, atol=1e-12
    )
    assert_leaky(unit_scale)


def test_column_scale_is_the_median_absolute_value() -> None:
    assert _compat.column_scale(pd.Series([1.0, -2.0, 3.0, -4.0, 50.0])) == 3.0
    assert _compat.column_scale(pd.Series([1e-5, -2e-5, 3e-5])) == pytest.approx(2e-5)
    # missing values and infinities are ignored
    assert _compat.column_scale(pd.Series([1.0, np.nan, np.inf, -2.0, 3.0])) == 2.0
    assert _compat.column_scale(pd.Series([1, 2, 3, 4], dtype="Int64")) == 2.5


def test_column_scale_is_one_when_there_is_no_scale() -> None:
    assert _compat.column_scale(pd.Series([np.nan, np.nan])) == 1.0
    assert _compat.column_scale(pd.Series([np.inf, -np.inf])) == 1.0
    assert _compat.column_scale(pd.Series([0.0, 0.0, 0.0, 7.0])) == 1.0  # median zero
    assert _compat.column_scale(pd.Series(dtype="float64")) == 1.0
    assert _compat.column_scale(pd.Series(["a", "b"])) == 1.0
    assert _compat.column_scale(pd.Series([True, False])) == 1.0


def test_column_atol_is_atol_times_the_scale() -> None:
    column = pd.Series([2e-5, -2e-5, 2e-5])
    assert _compat.column_atol(1e-9, column) == pytest.approx(2e-14)
    assert _compat.column_atol(0.0, column) == 0.0
    assert _compat.column_atol(1e-9, pd.Series([0.0, 0.0])) == pytest.approx(1e-9)


def test_scaled_frame_comparison_uses_the_tolerance_of_each_column() -> None:
    small = np.array([1e-10, 2e-10, 3e-10])
    large = np.array([1e3, 2e3, 3e3])
    a = pd.DataFrame({"small": small, "large": large})
    b = pd.DataFrame({"small": small * 1.5, "large": large + 1e-7})
    assert _compat.frames_equal(a, b, atol=1e-9, rtol=0.0) is False  # fixed: 1e-7 > 1e-9
    diffs = _compat.compare_frames(a, b, atol=1e-9, rtol=0.0, scaled=True)
    assert diffs[0].differs.all()  # 1e-10 vs 1.5e-10: far above 1e-9 * 2e-10
    assert not diffs[1].differs.any()  # 1e-7 is below 1e-9 * 2e3 = 2e-6
    assert _compat.frames_equal(a, b, atol=1e-9, rtol=0.0, scaled=True) is False
    assert _compat.frames_equal(a[["large"]], b[["large"]], atol=1e-9, rtol=0.0, scaled=True)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_random_function_is_reported_as_not_deterministic(prices: pd.DataFrame) -> None:
    rng = np.random.default_rng(1)

    def noisy(df: pd.DataFrame) -> pd.DataFrame:
        return df + rng.normal(size=df.shape)

    report = check(noisy, prices)
    assert report.deterministic is False
    assert report.causal is None
    assert report.findings == []
    assert report.cutoff_positions == []
    assert report.n_evaluations == 2
    assert report.horizon_rows is None
    assert any("not deterministic" in note for note in report.notes)
    assert "UNDETERMINED" in str(report)


def test_function_with_state_across_calls_is_reported_as_not_deterministic(
    prices: pd.DataFrame,
) -> None:
    calls = {"n": 0}

    def counter(df: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        return df + calls["n"]

    report = check(counter, prices)
    assert report.deterministic is False
    assert report.causal is None


def test_assert_causal_also_raises_for_a_function_that_is_not_deterministic(
    prices: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(LeakError) as caught:
        assert_causal(lambda df: df + rng.normal(size=df.shape), prices)
    assert caught.value.report.causal is None
    assert "not deterministic" in str(caught.value)


def test_random_row_selection_is_reported_as_not_deterministic(prices: pd.DataFrame) -> None:
    rng = np.random.default_rng(3)
    report = check(lambda df: df.iloc[np.sort(rng.permutation(len(df))[: len(df) // 2])], prices)
    assert report.deterministic is False


# --------------------------------------------------------------------------- #
# What the oracle cannot see: it tests, it does not prove
# --------------------------------------------------------------------------- #


def test_docstrings_say_that_the_oracle_tests_and_does_not_prove() -> None:
    module_doc = " ".join((oracle.__doc__ or "").split())
    check_doc = " ".join((oracle.check.__doc__ or "").split())
    assert "does not prove" in module_doc
    assert "not a proof" in check_doc
    for doc in (module_doc, check_doc):
        assert "the data that" in doc and "passes to" in doc
        assert "invisible" in doc
        assert "caches" in doc
    # state across calls is not reported as leaky: that claim was wrong
    assert "as not deterministic or as leaky" not in module_doc


def test_a_scaler_fitted_beforehand_is_invisible_to_the_oracle(prices: pd.DataFrame) -> None:
    """The scaler saw the whole sample outside the function; the oracle only sees the arguments."""
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    fitted = preprocessing.StandardScaler().fit(prices)

    def scale_with_fitted_scaler(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(fitted.transform(df), index=df.index, columns=df.columns)

    assert_clean(check(scale_with_fitted_scaler, prices, cutoffs=FEW))


def test_a_global_statistic_is_invisible_to_the_oracle(prices: pd.DataFrame) -> None:
    full_sample_mean = prices["close"].mean()
    report = check(lambda df: (df["close"] - full_sample_mean).to_frame("demeaned"), prices)
    assert_clean(report)
    # the same statistic computed inside the function is found
    assert_leaky(check(lambda df: (df["close"] - df["close"].mean()).to_frame("demeaned"), prices))


def test_a_cache_that_hides_a_leak_makes_the_function_look_causal(prices: pd.DataFrame) -> None:
    """State between calls is not reported as leaky: a cache can hide the leak completely.

    The first call computes the leaky feature on the full data and stores it. Every later call
    returns the stored frame, whose rows after the cutoff are ignored, so no difference shows.
    """
    cache: dict[object, pd.DataFrame] = {}

    def memoised(df: pd.DataFrame) -> pd.DataFrame:
        key = df.index[0]
        if key not in cache:
            cache[key] = df.shift(-1)
        return cache[key]

    assert_clean(check(memoised, prices))
    assert_leaky(check(shift_lead_1, prices, cutoffs=FEW))  # without the cache it is found


def test_rows_that_the_function_adds_after_the_cutoff_are_ignored(prices: pd.DataFrame) -> None:
    """A function that ignores its argument returns the same rows every time; none is compared."""
    frozen = prices.rolling(5).mean()
    assert_clean(check(lambda df: frozen, prices, cutoffs=FEW))


# --------------------------------------------------------------------------- #
# Functions that raise
# --------------------------------------------------------------------------- #


def needs_50_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        raise ValueError("need at least 50 rows")
    return df.rolling(50).mean()


def test_function_raising_on_short_histories_is_skipped_with_a_note(prices: pd.DataFrame) -> None:
    report = check(needs_50_rows, prices)
    assert report.causal is True
    assert len(report.notes) == 1
    assert re.search(r"raised on \d+ of 64 truncated inputs", report.notes[0])
    assert "need at least 50 rows" in report.notes[0]
    assert "min_rows" in report.notes[0]


def test_min_rows_avoids_the_short_histories(prices: pd.DataFrame) -> None:
    report = check(needs_50_rows, prices, min_rows=50)
    assert_clean(report)
    only_truncation = check(needs_50_rows, prices, min_rows=50, mode="truncate")
    assert min(only_truncation.cutoff_positions) >= 50
    # The perturbation test passes the full length, so min_rows does not limit it.
    only_perturbation = check(needs_50_rows, prices, min_rows=50, mode="perturb")
    assert min(only_perturbation.cutoff_positions) < 50


def test_function_raising_on_every_truncated_input_suggests_min_rows(prices: pd.DataFrame) -> None:
    def needs_all_rows(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < N_ROWS:
            raise ValueError("need all rows")
        return df

    with pytest.raises(PeekproofError, match="min_rows"):
        check(needs_all_rows, prices)
    # The perturbation keeps the length, so that test alone still works.
    assert_clean(check(needs_all_rows, prices, mode="perturb"))


def test_function_raising_on_the_full_input_is_a_peekproof_error(prices: pd.DataFrame) -> None:
    def broken(df: pd.DataFrame) -> pd.DataFrame:
        raise KeyError("no such column")

    with pytest.raises(PeekproofError, match="raised on the full input") as caught:
        check(broken, prices)
    assert isinstance(caught.value.__cause__, KeyError)


def test_function_raising_on_every_perturbed_input_suggests_truncate_mode(
    prices: pd.DataFrame,
) -> None:
    def only_the_real_data(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == N_ROWS and not df.equals(prices):
            raise ValueError("unexpected values")
        return df.rolling(5).mean()

    with pytest.raises(PeekproofError, match="perturbed"):
        check(only_the_real_data, prices)
    assert_clean(check(only_the_real_data, prices, mode="truncate"))


def test_truncation_findings_are_kept_when_every_perturbed_evaluation_raises(
    prices: pd.DataFrame,
) -> None:
    """A leak that the truncation test found must not be discarded by a failing perturbation."""

    def leaky_and_picky(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == N_ROWS and not df.equals(prices):
            raise ValueError("unexpected values")
        return df.shift(-1)

    report = check(leaky_and_picky, prices, cutoffs=FEW)
    assert_leaky(report)
    assert {f.test for f in report.findings} == {"truncate"}
    assert report.horizon_rows == 1
    assert len(report.notes) == 1
    assert "perturbation test could not run" in report.notes[0]
    assert f"all {FEW} perturbed inputs" in report.notes[0]
    assert "unexpected values" in report.notes[0]
    assert report.n_evaluations == 2 + 2 * FEW  # the failed evaluations count, too


def test_perturbation_findings_are_kept_when_every_truncated_evaluation_raises(
    prices: pd.DataFrame,
) -> None:
    def leaky_and_needs_all_rows(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < N_ROWS:
            raise ValueError("need all rows")
        return df.shift(-1)

    report = check(leaky_and_needs_all_rows, prices, cutoffs=FEW)
    assert_leaky(report)
    assert {f.test for f in report.findings} == {"perturb"}
    assert len(report.notes) == 1
    assert "truncation test could not run" in report.notes[0]
    assert "min_rows" in report.notes[0]


def test_function_that_raises_in_both_tests_is_an_error_naming_both(prices: pd.DataFrame) -> None:
    def only_the_real_data(df: pd.DataFrame) -> pd.DataFrame:
        if not df.equals(prices):
            raise ValueError("unexpected input")
        return df.rolling(5).mean()

    with pytest.raises(PeekproofError, match=r"truncated inputs.*perturbed inputs"):
        check(only_the_real_data, prices, cutoffs=FEW)


def test_a_test_that_raises_everywhere_but_whose_sibling_is_clean_is_still_an_error(
    prices: pd.DataFrame,
) -> None:
    """Without findings there is nothing to keep, and one half of the evidence is missing."""

    def picky_but_causal(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == N_ROWS and not df.equals(prices):
            raise ValueError("unexpected values")
        return df.rolling(5).mean()

    with pytest.raises(PeekproofError, match="perturbed"):
        check(picky_but_causal, prices, cutoffs=FEW)


# --------------------------------------------------------------------------- #
# Exact number of evaluations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["truncate", "perturb", "both"])
@pytest.mark.parametrize("cutoffs", [FEW, 64, [40, 90, 150, 280]], ids=["few", "default", "list"])
def test_n_evaluations_is_the_number_of_calls_of_the_function(
    prices: pd.DataFrame, mode: Mode, cutoffs: Any
) -> None:
    counted = Counted(rolling_mean)
    report = check(counted, prices, cutoffs=cutoffs, mode=mode)
    assert report.n_evaluations == counted.calls


def test_n_evaluations_counts_the_calls_that_raised(prices: pd.DataFrame) -> None:
    counted = Counted(needs_50_rows)
    report = check(counted, prices, cutoffs=FEW)
    assert report.notes  # some truncated inputs were too short
    assert report.n_evaluations == counted.calls == 2 + 2 * FEW


def test_n_evaluations_of_a_function_that_is_not_deterministic(prices: pd.DataFrame) -> None:
    calls = Counted(lambda df: df + np.random.default_rng().normal(size=df.shape))
    report = check(calls, prices)
    assert report.deterministic is False
    assert report.n_evaluations == calls.calls == 2


def test_header_of_the_text_report_names_the_cutoffs_and_the_evaluations(
    prices: pd.DataFrame,
) -> None:
    report = check(rolling_mean, prices, cutoffs=[50, 100, 299])
    assert str(report).startswith(
        "peekproof: CAUSAL  (deterministic, 3 cutoffs, 8 evaluations, 300 rows)"
    )


# --------------------------------------------------------------------------- #
# Input types
# --------------------------------------------------------------------------- #


def polars_modules() -> Any:
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    return pl


def test_polars_input_with_a_time_column(prices: pd.DataFrame) -> None:
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())

    def pl_rolling_mean(df: Any) -> Any:
        return df.with_columns(pl.col("close").rolling_mean(10).alias("ma"))

    def pl_lead(df: Any) -> Any:
        return df.with_columns(pl.col("close").shift(-1).alias("lead"))

    assert_clean(check(pl_rolling_mean, frame, time_col="date", cutoffs=FEW))

    leaky = check(pl_lead, frame, time_col="date", cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1
    assert leaky.affected_columns == ["lead"]
    # Labels come from the time column.
    assert leaky.findings[0].cutoff_label in {format_label(ts) for ts in prices.index}


def test_polars_function_receives_polars_frames_that_look_like_the_input(
    prices: pd.DataFrame,
) -> None:
    """Truncated and perturbed slices come back as polars with the schema of the input."""
    pl = polars_modules()
    flat = prices.reset_index()
    frame = pl.from_pandas(flat).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("volume").cast(pl.Int32),
        pl.Series("kind", ["a", "b", "c"] * (N_ROWS // 3)),
        pl.Series("regime", ["up", "down"] * (N_ROWS // 2)).cast(pl.Categorical),
    )
    seen: list[tuple[type, list[str], Any]] = []

    def inspect(df: Any) -> Any:
        seen.append((type(df), df.columns, df.schema))
        return df.with_columns(pl.col("close").rolling_mean(3).alias("ma"))

    report = check(inspect, frame, time_col="date", cutoffs=FEW)
    assert_clean(report)
    assert len(seen) == report.n_evaluations
    for kind, columns, schema in seen:
        assert kind is pl.DataFrame
        assert columns == frame.columns
        assert schema == frame.schema


def test_polars_function_may_return_a_lazy_frame(prices: pd.DataFrame) -> None:
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())

    def lazy_lead(df: Any) -> Any:
        return df.lazy().with_columns(pl.col("close").shift(-2).alias("lead"))

    report = check(lazy_lead, frame, time_col="date", cutoffs=FEW)
    assert_leaky(report)
    assert report.horizon_rows == 2


def test_polars_output_must_keep_the_time_column(prices: pd.DataFrame) -> None:
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())
    with pytest.raises(PeekproofError, match="time column 'date'"):
        check(lambda df: df.select("close"), frame, time_col="date", cutoffs=FEW)
    with pytest.raises(PeekproofError, match=r"polars Series.*time column"):
        check(lambda df: df["close"].shift(1), frame, time_col="date", cutoffs=FEW)
    with pytest.raises(PeekproofError, match="polars DataFrame"):
        check(lambda df: 3.14, frame, time_col="date", cutoffs=FEW)


def test_polars_output_with_missing_times_is_an_error(prices: pd.DataFrame) -> None:
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())

    def blank_time(df: Any) -> Any:
        return df.with_columns(
            pl.when(pl.col("close") > 100.0).then(None).otherwise(pl.col("date")).alias("date")
        )

    with pytest.raises(PeekproofError, match="missing values"):
        check(blank_time, frame, time_col="date", cutoffs=FEW)


def test_a_slice_that_cannot_be_turned_back_into_polars_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())

    def refuses(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("cannot convert")

    monkeypatch.setattr(pl, "from_pandas", refuses)
    with pytest.raises(PeekproofError, match="could not convert a pandas slice"):
        check(lambda df: df, frame, time_col="date", cutoffs=FEW)


def test_polars_rows_are_sorted_by_the_time_column(prices: pd.DataFrame) -> None:
    pl = polars_modules()
    shuffled = pl.from_pandas(prices.reset_index()).sample(fraction=1.0, shuffle=True, seed=3)

    def pl_lead(df: Any) -> Any:
        return df.with_columns(pl.col("close").shift(-1).alias("lead"))

    def pl_rolling_mean(df: Any) -> Any:
        return df.with_columns(pl.col("close").rolling_mean(10).alias("ma"))

    assert_clean(check(pl_rolling_mean, shuffled, time_col="date", cutoffs=FEW))
    assert check(pl_lead, shuffled, time_col="date", cutoffs=FEW).horizon_rows == 1


def test_polars_input_needs_a_valid_time_column(prices: pd.DataFrame) -> None:
    pl = pytest.importorskip("polars")
    frame = pl.from_pandas(prices.reset_index())
    with pytest.raises(PeekproofError, match="time_col"):
        check(rolling_mean, frame)
    with pytest.raises(PeekproofError, match="not a column"):
        check(rolling_mean, frame, time_col="nope")


def test_a_pandas_function_on_polars_input_fails_with_a_clear_error(prices: pd.DataFrame) -> None:
    """The function now gets what it was given: polars. A pandas function cannot use that."""
    pl = polars_modules()
    frame = pl.from_pandas(prices.reset_index())
    with pytest.raises(PeekproofError, match="raised on the full input"):
        check(lambda df: df.rolling(10).mean(), frame, time_col="date")


def test_polars_is_only_imported_when_a_polars_object_is_passed() -> None:
    code = (
        "import sys\n"
        "import numpy as np, pandas as pd\n"
        "import peekproof.oracle as oracle\n"
        "df = pd.DataFrame({'a': np.arange(40.0)}, index=pd.RangeIndex(40))\n"
        "assert oracle.check(lambda d: d.rolling(3).mean(), df).causal is True\n"
        "assert 'polars' not in sys.modules, 'polars was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_polars_adapter_restores_columns_and_dtypes() -> None:
    pl = polars_modules()
    frame = pl.DataFrame(
        {
            "qty": pl.Series([1, 2, None, 4], dtype=pl.Int32),
            "day": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 4), eager=True),
            "price": [1.5, 2.5, 3.5, 4.5],
            "tag": ["a", "b", "a", "b"],
        }
    )
    adapter = _compat.polars_adapter(frame, "day")
    assert adapter is not None
    as_pandas = _compat.to_pandas(frame, "day")
    assert isinstance(as_pandas, pd.DataFrame)
    restored = adapter.to_native(as_pandas)
    assert restored.columns == frame.columns  # the time column is back in its place
    assert restored.schema == frame.schema
    assert restored.equals(frame)


def test_polars_adapter_is_not_made_for_other_data(prices: pd.DataFrame) -> None:
    assert _compat.polars_adapter(prices, "date") is None
    assert _compat.polars_adapter(prices, None) is None


def test_pandas_time_column_becomes_the_sorted_index(prices: pd.DataFrame) -> None:
    flat = prices.reset_index().sample(frac=1.0, random_state=4)
    report = check(rolling_mean, flat, time_col="date", cutoffs=FEW)
    assert_clean(report)
    assert check(shift_lead_1, flat, time_col="date", cutoffs=FEW).horizon_rows == 1


def test_series_output_is_treated_as_a_one_column_frame(prices: pd.DataFrame) -> None:
    named = check(lambda df: df["close"].rolling(5).mean(), prices, cutoffs=FEW)
    assert_clean(named)

    def unnamed_leak(df: pd.DataFrame) -> pd.Series:
        return pd.Series(df["close"].shift(-1).to_numpy(), index=df.index)

    leaky = check(unnamed_leak, prices, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.affected_columns == ["value"]

    def named_leak(df: pd.DataFrame) -> pd.Series:
        return df["close"].shift(-1).rename("lead")

    assert check(named_leak, prices, cutoffs=FEW).affected_columns == ["lead"]


def test_series_input(prices: pd.DataFrame) -> None:
    close = prices["close"]
    assert_clean(check(lambda s: s.rolling(5).mean(), close, cutoffs=FEW))
    leaky = check(lambda s: s.shift(-1), close, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1


def test_integer_index_is_accepted(prices: pd.DataFrame) -> None:
    plain = prices.reset_index(drop=True)
    assert_clean(check(rolling_mean, plain, cutoffs=FEW))
    report = check(shift_lead_1, plain, cutoffs=FEW)
    assert_leaky(report)
    assert report.findings[0].cutoff_label == str(report.findings[0].cutoff_pos - 1)


def test_output_indexed_by_something_other_than_time_is_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="subset"):
        check(lambda df: df.reset_index(), prices)
    with pytest.raises(PeekproofError, match="RangeIndex"):
        check(lambda df: df.reset_index(drop=True), prices)
    with pytest.raises(PeekproofError, match="subset"):
        check(lambda df: df.set_axis(np.arange(len(df)) + 0.5), prices)


def test_output_labels_outside_the_input_are_rejected_for_a_position_index(
    prices: pd.DataFrame,
) -> None:
    plain = prices.reset_index(drop=True)
    with pytest.raises(PeekproofError, match=r"not a subset.*not a DatetimeIndex"):
        check(lambda df: df.set_axis(df.index + 1000), plain)


def test_output_with_extra_columns_is_only_rejected_when_the_index_is_not_a_subset(
    prices: pd.DataFrame,
) -> None:
    extra = check(lambda df: df.assign(extra=1.0), prices, cutoffs=FEW)
    assert_clean(extra)
    with pytest.raises(PeekproofError, match="subset"):
        check(lambda df: df.assign(extra=1.0).set_axis(range(len(df))), prices)


def test_output_that_is_not_a_pandas_object_is_rejected(prices: pd.DataFrame) -> None:
    def returns_array(df: pd.DataFrame) -> Any:
        return df.to_numpy()

    with pytest.raises(PeekproofError, match="DataFrame or Series"):
        check(returns_array, prices)


def test_output_with_duplicate_column_names_is_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="duplicate column names"):
        check(lambda df: pd.concat([df, df], axis=1), prices)


def test_output_with_duplicate_index_labels_is_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="duplicate labels"):
        check(lambda df: pd.concat([df, df]), prices)


def test_output_rows_may_come_in_any_order(prices: pd.DataFrame) -> None:
    assert_clean(check(lambda df: rolling_mean(df).iloc[::-1], prices, cutoffs=FEW))


def test_output_may_use_another_time_unit(prices: pd.DataFrame) -> None:
    """The labels are the same instants, only stored in milliseconds."""

    def in_milliseconds(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rolling(5).mean()
        out.index = out.index.as_unit("ms")
        return out

    assert_clean(check(in_milliseconds, prices, cutoffs=FEW))
    leaky = check(lambda df: in_milliseconds(df.shift(-1)), prices, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1


def test_data_must_have_a_monotonic_increasing_index(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="monotonic"):
        check(rolling_mean, prices.iloc[::-1])


def test_data_must_not_have_duplicate_labels(prices: pd.DataFrame) -> None:
    doubled = prices.iloc[[0, 0, *range(1, 50)]]
    with pytest.raises(PeekproofError, match="duplicate"):
        check(rolling_mean, doubled)


def test_data_must_not_have_missing_labels(prices: pd.DataFrame) -> None:
    index = prices.index.to_series().reset_index(drop=True)
    index.iloc[5] = pd.NaT
    broken = prices.copy()
    broken.index = pd.DatetimeIndex(index)
    with pytest.raises(PeekproofError, match="missing labels"):
        check(rolling_mean, broken)


def test_data_of_an_unsupported_type_is_rejected() -> None:
    not_a_frame: Any = [1.0, 2.0, 3.0]
    with pytest.raises(PeekproofError, match="unsupported data type"):
        check(rolling_mean, not_a_frame)


def test_too_few_rows_are_rejected(prices: pd.DataFrame) -> None:
    with pytest.raises(PeekproofError, match="not enough rows"):
        check(rolling_mean, prices.iloc[:2])
    short = check(rolling_mean, prices.iloc[:3])
    assert short.cutoff_positions == [2]
    assert short.n_evaluations == 2 + 1 + 1  # one cutoff for each of the two tests


# --------------------------------------------------------------------------- #
# Outputs labeled by time: bars made by resampling
# --------------------------------------------------------------------------- #


def right_labeled_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Five-minute bars labeled by their end: the bar 09:05 holds 09:01 .. 09:05."""
    return df.resample("5min", label="right", closed="right").last()


def left_labeled_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Five-minute bars labeled by their start: the bar 09:00 holds 09:00 .. 09:04."""
    return df.resample("5min", label="left", closed="left").last()


@pytest.mark.parametrize("mode", ["truncate", "perturb", "both"])
def test_right_labeled_bars_are_causal(bars: pd.DataFrame, mode: Mode) -> None:
    """The last bar of a shortened history is labeled after the cutoff and is ignored."""
    report = check(right_labeled_bars, bars, mode=mode, cutoffs=FEW)
    assert_clean(report, n_rows=len(bars))


@pytest.mark.parametrize("mode", ["truncate", "perturb", "both"])
def test_left_labeled_bars_are_not_causal(bars: pd.DataFrame, mode: Mode) -> None:
    """The bar 09:00 holds 09:04, so its value changes when the data ends at 09:02."""
    report = check(left_labeled_bars, bars, mode=mode, cutoffs=FEW)
    assert_leaky(report)
    assert report.horizon_rows == 4
    assert report.horizon_unbounded is False
    assert {f.kind for f in report.findings} == {"value_change"}
    # The affected row is the bar, named by its own label.
    bar_labels = {format_label(ts) for ts in left_labeled_bars(bars).index}
    assert {f.first_affected_label for f in report.findings} <= bar_labels


def test_default_resample_labels_the_bar_by_its_start_and_leaks(bars: pd.DataFrame) -> None:
    report = check(lambda df: df.resample("5min").last(), bars, cutoffs=FEW)
    assert_leaky(report)
    assert report.horizon_rows == 4


def test_sums_over_right_labeled_bars_are_causal(bars: pd.DataFrame) -> None:
    summed = check(
        lambda df: df.resample("15min", label="right", closed="right").sum(), bars, cutoffs=FEW
    )
    assert_clean(summed, n_rows=len(bars))


@pytest.mark.parametrize("rule", ["W-FRI", "W", "ME", "D"])
def test_resampled_series_labeled_by_the_end_of_the_period_are_causal(
    prices: pd.DataFrame, rule: str
) -> None:
    """Weeks end on a Friday or a Sunday, months on their last day; a day is one row."""
    report = check(lambda df: df.resample(rule).last(), prices)
    assert_clean(report)


def test_monthly_bars_labeled_by_the_start_of_the_month_leak(prices: pd.DataFrame) -> None:
    report = check(lambda df: df.resample("MS").last(), prices, cutoffs=FEW)
    assert_leaky(report)
    assert 15 <= report.horizon_rows <= 23  # a month holds about 21 trading days


def test_a_leaky_bar_is_named_by_its_own_label_even_if_the_input_has_no_such_row(
    prices: pd.DataFrame,
) -> None:
    """Weeks labeled by the Sunday before them: the input has business days only."""
    report = check(lambda df: df.resample("W", label="left", closed="left").last(), prices)
    assert_leaky(report)
    sundays = {format_label(ts) for ts in pd.date_range(prices.index[0], periods=80, freq="W-SUN")}
    assert {f.first_affected_label for f in report.findings} <= sundays
    assert not {f.first_affected_label for f in report.findings} & {
        format_label(ts) for ts in prices.index
    }
    assert report.horizon_rows is not None
    assert 1 <= report.horizon_rows <= 5  # a week holds five trading days


def test_labels_shifted_forward_in_time_are_causal_and_backward_are_not(
    prices: pd.DataFrame,
) -> None:
    """The label t + 1 day holds the row of t; the label t - 1 day holds the row of t + 1 day."""
    assert_clean(check(lambda df: df.shift(freq="1D"), prices, cutoffs=FEW))
    leaky = check(lambda df: df.shift(freq="-1D"), prices, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1


def test_output_with_extra_timestamps_between_the_input_rows_is_causal(
    prices: pd.DataFrame,
) -> None:
    def with_midday_rows(df: pd.DataFrame) -> pd.DataFrame:
        midday = df.index + pd.Timedelta("12h")
        return df.rolling(3).mean().reindex(df.index.union(midday)).ffill()

    assert_clean(check(with_midday_rows, prices, cutoffs=FEW))


def test_sub_second_labels_are_not_rounded_together() -> None:
    """The input is stored in seconds; the output has rows at every half second."""
    index = pd.date_range("2021-03-01 09:00:00", periods=200, freq="1s", name="t").as_unit("s")
    data = pd.DataFrame({"x": np.arange(200.0)}, index=index)

    def with_half_seconds(df: pd.DataFrame) -> pd.DataFrame:
        half_seconds = df.index + pd.Timedelta("500ms")
        return df.reindex(df.index.union(half_seconds))

    assert_clean(check(with_half_seconds, data, cutoffs=FEW), n_rows=200)
    # the row at second t takes the value at second t + 1: one input row ahead
    leaky = check(lambda df: with_half_seconds(df).shift(-2), data, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1
    assert_clean(check(lambda df: with_half_seconds(df).shift(2), data, cutoffs=FEW), n_rows=200)


def test_dates_beyond_the_range_of_nanoseconds_are_checked_too() -> None:
    far = pd.date_range("3000-01-01", periods=120, freq="D", name="day", unit="us")
    data = pd.DataFrame({"x": np.random.default_rng(0).normal(size=120)}, index=far)
    assert_clean(check(lambda df: df.rolling(5).mean(), data, cutoffs=FEW), n_rows=120)
    assert_clean(check(lambda df: df.resample("W-FRI").last(), data, cutoffs=FEW), n_rows=120)
    assert_leaky(check(lambda df: df.shift(-1), data, cutoffs=FEW))


@pytest.mark.parametrize(
    ("mode", "word"),
    [("truncate", "truncated"), ("perturb", "perturbed")],
)
def test_an_output_that_is_not_indexed_by_time_names_the_evaluation_and_the_cutoff(
    prices: pd.DataFrame, mode: Mode, word: str
) -> None:
    """Only the evaluations of one test return an index by position."""

    def broken_in_one_test(df: pd.DataFrame) -> pd.DataFrame:
        shortened = len(df) < N_ROWS
        perturbed = len(df) == N_ROWS and not df.equals(prices)
        if (mode == "truncate" and shortened) or (mode == "perturb" and perturbed):
            return df.reset_index(drop=True)
        return df

    with pytest.raises(PeekproofError) as caught:
        check(broken_in_one_test, prices, cutoffs=[100, 200], mode=mode)
    message = str(caught.value)
    assert f"the {word} evaluation at cutoff 100" in message
    assert "RangeIndex" in message
    assert "subset" in message


def test_an_output_that_is_not_indexed_by_time_names_the_full_evaluation(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises(PeekproofError, match="the full evaluation has an index"):
        check(lambda df: df.reset_index(drop=True), prices)


def test_relabeled_output_must_be_in_increasing_order(prices: pd.DataFrame) -> None:
    def newest_first(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample("W").last().iloc[::-1]

    with pytest.raises(PeekproofError, match="increasing order"):
        check(newest_first, prices, cutoffs=FEW)


def test_relabeled_output_must_have_the_time_zone_awareness_of_the_input(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises(PeekproofError, match="time-zone"):
        check(lambda df: df.tz_localize("UTC"), prices, cutoffs=FEW)


def test_time_zone_aware_input_is_checked_like_any_other(prices: pd.DataFrame) -> None:
    aware = prices.tz_localize("Europe/Berlin")
    assert_clean(check(lambda df: df.resample("W-FRI").last(), aware, cutoffs=FEW))
    assert_clean(check(rolling_mean, aware, cutoffs=FEW))
    assert_leaky(check(lambda df: df.resample("MS").last(), aware, cutoffs=FEW))


# --------------------------------------------------------------------------- #
# The caller's data is never modified
# --------------------------------------------------------------------------- #


def test_function_that_modifies_its_argument_does_not_change_the_callers_data() -> None:
    data = make_prices()
    before = data.copy(deep=True)

    def modifies_in_place(df: pd.DataFrame) -> pd.DataFrame:
        df["double"] = df["close"] * 2
        df.iloc[0, 0] = -1.0
        return df

    report = check(modifies_in_place, data, cutoffs=FEW)
    pd.testing.assert_frame_equal(data, before)
    assert_clean(report)


def test_check_leaves_data_untouched_in_every_mode(prices: pd.DataFrame) -> None:
    before = prices.copy(deep=True)
    modes: tuple[Mode, ...] = ("truncate", "perturb", "both")
    for mode in modes:
        check(shift_lead_1, prices, mode=mode, cutoffs=FEW)
    pd.testing.assert_frame_equal(prices, before)


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #


def test_default_cutoffs_and_evaluation_count(prices: pd.DataFrame) -> None:
    report = check(rolling_mean, prices)
    positions = report.cutoff_positions
    assert positions == sorted(set(positions))
    assert {N_ROWS - 1, N_ROWS - 2, N_ROWS // 2} <= set(positions)
    assert max(positions) == N_ROWS - 1
    # The perturbation test starts near row 2; the truncation test could not.
    assert min(positions) <= 8
    assert 64 < len(positions) <= 2 * 64
    assert report.n_evaluations == 2 + 2 * 64


@pytest.mark.parametrize(
    ("mode", "evaluations", "positions"),
    [("truncate", 2 + 64, 64), ("perturb", 2 + 64, 64), ("both", 2 + 2 * 64, None)],
)
def test_number_of_evaluations_per_mode(
    prices: pd.DataFrame, mode: Mode, evaluations: int, positions: int | None
) -> None:
    report = check(rolling_mean, prices, mode=mode)
    assert report.n_evaluations == evaluations
    if positions is not None:
        assert len(report.cutoff_positions) == positions


def test_each_test_has_its_own_first_cutoff(prices: pd.DataFrame) -> None:
    truncation = check(rolling_mean, prices, mode="truncate")
    perturbation = check(rolling_mean, prices, mode="perturb")
    assert min(truncation.cutoff_positions) >= 30  # a tenth of the rows
    assert min(perturbation.cutoff_positions) <= 8
    assert max(truncation.cutoff_positions) == max(perturbation.cutoff_positions) == N_ROWS - 1


def test_explicit_cutoffs_are_used_as_given(prices: pd.DataFrame) -> None:
    report = check(rolling_mean, prices, cutoffs=[200, 50, 50, 120])
    assert report.cutoff_positions == [50, 120, 200]
    assert report.n_evaluations == 2 + 2 * 3
    numpy_cutoffs = check(rolling_mean, prices, cutoffs=np.array([60, 90]))
    assert numpy_cutoffs.cutoff_positions == [60, 90]
    # Explicit positions are not subject to min_rows or to the tenth of the rows.
    early = check(rolling_mean, prices, cutoffs=[3, 10], min_rows=100)
    assert early.cutoff_positions == [3, 10]
    assert early.n_evaluations == 2 + 2 * 2


def test_cutoffs_that_stop_early_are_reported_with_the_share_of_rows_they_cover(
    prices: pd.DataFrame,
) -> None:
    report = check(rolling_mean, prices, cutoffs=[50, 100, 276])
    assert report.causal is True
    assert report.notes == [
        "cutoffs cover 92% of rows (the last cutoff is position 276 of 300); "
        "a leak that only affects later rows is not tested"
    ]
    assert "note: cutoffs cover 92% of rows" in str(report)


def test_the_coverage_is_the_last_cutoff_over_the_number_of_rows(prices: pd.DataFrame) -> None:
    def share(last: int) -> str:
        report = check(rolling_mean, prices, cutoffs=[20, last], mode="truncate")
        (note,) = report.notes
        return note.split("cover ")[1].split("%")[0]

    assert share(30) == "10"
    assert share(150) == "50"
    assert share(299 - 1) == "99"  # 298 of 300 is 99.33 %: never rounded up to 100


def test_no_coverage_note_when_the_last_cutoff_is_the_last_row_that_can_be_tested(
    prices: pd.DataFrame,
) -> None:
    assert check(rolling_mean, prices, cutoffs=[50, N_ROWS - 1]).notes == []
    assert check(rolling_mean, prices, cutoffs=FEW).notes == []


@pytest.mark.parametrize("bad", [[0], [1], [N_ROWS], [N_ROWS + 5], [-3], [10.5], ["a"], [], 0, "8"])
def test_invalid_cutoffs_are_rejected(prices: pd.DataFrame, bad: Any) -> None:
    with pytest.raises(PeekproofError):
        check(rolling_mean, prices, cutoffs=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "everything"},
        {"atol": -1.0},
        {"rtol": float("nan")},
        {"seed": -1},
        {"seed": 1.5},
        {"min_rows": 0},
        {"min_rows": 2.5},
    ],
)
def test_invalid_options_are_rejected(prices: pd.DataFrame, kwargs: dict[str, Any]) -> None:
    with pytest.raises(PeekproofError):
        check(rolling_mean, prices, **kwargs)


def test_same_seed_gives_the_same_report(prices: pd.DataFrame) -> None:
    first = check(shift_lead_1, prices, seed=11, cutoffs=FEW)
    second = check(shift_lead_1, prices, seed=11, cutoffs=FEW)
    assert first.to_dict() == second.to_dict()


def test_the_seed_changes_the_cutoffs(prices: pd.DataFrame) -> None:
    first = check(rolling_mean, prices, seed=1, cutoffs=FEW)
    second = check(rolling_mean, prices, seed=2, cutoffs=FEW)
    assert first.cutoff_positions != second.cutoff_positions
    assert first.n_evaluations == second.n_evaluations


def test_verdict_does_not_depend_on_the_seed(prices: pd.DataFrame) -> None:
    for seed in range(5):
        assert check(rolling_mean, prices, seed=seed, cutoffs=FEW).causal is True
        assert check(shift_lead_1, prices, seed=seed, mode="perturb", cutoffs=FEW).causal is False
        assert check(global_zscore, prices, seed=seed, mode="perturb", cutoffs=FEW).causal is False


def test_report_can_be_rendered_as_json(prices: pd.DataFrame) -> None:
    report = check(lead_1_then_dropna, prices, cutoffs=FEW)
    payload = json.loads(report.to_json())
    assert payload["causal"] is False
    assert payload["deterministic"] is True
    assert payload["findings"][0]["column"] in {"close", "volume"}
    assert payload["horizon_rows"] == 1
    assert payload["n_evaluations"] == 2 + 2 * FEW
    assert all(type(p) is int for p in payload["cutoff_positions"])


# --------------------------------------------------------------------------- #
# The perturbation
# --------------------------------------------------------------------------- #


def make_mixed_frame(n: int = 200, seed: int = 6) -> pd.DataFrame:
    """One column of each dtype that the perturbation handles."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f64": rng.normal(size=n),
            "f32": rng.normal(size=n).astype("float32"),
            "i64": rng.integers(0, 50, n),
            "u8": rng.integers(0, 200, n).astype("uint8"),
            "flat": np.full(n, 7),
            "Int64": pd.array(rng.integers(0, 10, n), dtype="Int64"),
            "Float64": pd.array(rng.normal(size=n), dtype="Float64"),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "text": rng.choice(["x", "y", "z"], n),
        },
        index=pd.bdate_range("2020-01-01", periods=n, name="date"),
    )


def n_changed(old: pd.Series, new: pd.Series) -> int:
    """Number of rows whose value differs; two missing values count as equal."""
    return sum(
        1
        for a, b in zip(old.tolist(), new.tolist(), strict=True)
        if not (pd.isna(a) and pd.isna(b)) and bool(a != b)
    )


def test_perturbation_changes_only_the_rows_from_the_cutoff_on() -> None:
    data = make_mixed_frame()
    before = data.copy(deep=True)

    perturbed = _perturb(data, 80, seed=0)

    pd.testing.assert_frame_equal(data, before)  # the input is not modified
    assert isinstance(perturbed, pd.DataFrame)
    assert perturbed.index.equals(data.index)
    assert perturbed.dtypes.equals(data.dtypes)
    pd.testing.assert_frame_equal(perturbed.iloc[:80], data.iloc[:80])
    assert list(perturbed.columns) == list(data.columns)


@pytest.mark.parametrize("column", list(make_mixed_frame().columns))
def test_perturbation_changes_every_future_row_of_a_column(column: str) -> None:
    """Rounded noise or a permutation can leave a row as it was; this must not."""
    data = make_mixed_frame()
    perturbed = _perturb(data, 80, seed=0)
    assert isinstance(perturbed, pd.DataFrame)
    future = len(data) - 80
    assert n_changed(data[column].iloc[80:], perturbed[column].iloc[80:]) == future


def test_perturbation_depends_only_on_seed_and_cutoff() -> None:
    data = make_mixed_frame()
    first = _perturb(data, 100, seed=3)
    pd.testing.assert_frame_equal(first, _perturb(data, 100, seed=3))
    assert isinstance(first, pd.DataFrame)
    assert not first.equals(_perturb(data, 100, seed=4))
    assert not first.equals(_perturb(data, 101, seed=3))


def test_perturbation_of_a_series() -> None:
    close = make_prices()["close"]
    perturbed = _perturb(close, 100, seed=0)
    assert isinstance(perturbed, pd.Series)
    assert perturbed.index.equals(close.index)
    pd.testing.assert_series_equal(perturbed.iloc[:100], close.iloc[:100])
    assert n_changed(close.iloc[100:], perturbed.iloc[100:]) == N_ROWS - 100


def test_perturbation_of_the_last_row_only() -> None:
    data = make_mixed_frame()
    perturbed = _perturb(data, len(data) - 1, seed=0)
    assert isinstance(perturbed, pd.DataFrame)
    for column in ("f64", "i64", "Int64", "cat", "text"):
        assert n_changed(data[column].iloc[-1:], perturbed[column].iloc[-1:]) == 1


def test_integer_columns_keep_their_dtype_when_the_function_sees_them(prices: pd.DataFrame) -> None:
    def needs_integer_volume(df: pd.DataFrame) -> pd.DataFrame:
        assert pd.api.types.is_integer_dtype(df["volume"]), "volume must stay an integer column"
        return df.rolling(5).mean()

    assert_clean(check(needs_integer_volume, prices, cutoffs=FEW))


def test_missing_values_stay_missing_in_every_dtype() -> None:
    rng = np.random.default_rng(8)
    n = 120
    holes = rng.random(n) < 0.2
    data = pd.DataFrame(
        {
            "f64": np.where(holes, np.nan, rng.normal(size=n)),
            "Int64": pd.array(np.where(holes, None, rng.integers(0, 10, n)), dtype="Int64"),
            "boolean": pd.array(np.where(holes, None, rng.random(n) < 0.5), dtype="boolean"),
            "cat": pd.Categorical(np.where(holes, None, rng.choice(["a", "b"], n))),
            "text": pd.array(np.where(holes, None, rng.choice(["x", "y"], n)), dtype="string"),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    perturbed = _perturb(data, 40, seed=1)
    assert isinstance(perturbed, pd.DataFrame)
    assert perturbed.dtypes.equals(data.dtypes)
    for column in data.columns:
        assert perturbed[column].isna().tolist() == data[column].isna().tolist()
        present = ~data[column].iloc[40:].isna()
        assert n_changed(data[column].iloc[40:][present], perturbed[column].iloc[40:][present]) > 0


# -- floats


def test_noise_is_heavy_tailed_and_scaled_by_a_robust_measure_of_spread() -> None:
    column = pd.Series(np.random.default_rng(0).normal(10.0, 4.0, 20_000))
    noisy = _add_noise(column, 5_000, np.random.default_rng(1))
    assert (noisy.iloc[:5_000] == column.iloc[:5_000]).all()
    added = (noisy - column).iloc[5_000:]
    scale = _compat.robust_scale(column)
    assert scale == pytest.approx(4.0, rel=0.05)
    # Student-t with 3 degrees of freedom: its median absolute value is 0.765 times the scale
    assert float(added.abs().median()) == pytest.approx(0.765 * scale, rel=0.05)
    # and its tails are far heavier than a normal's: a normal has no values beyond 4.5 scales
    assert added.abs().max() > 8 * scale
    assert (added.abs() > 4.5 * scale).mean() > 0.003


def test_noise_is_not_inflated_by_outliers() -> None:
    values = np.random.default_rng(0).normal(0.0, 1.0, 20_000)
    values[:50] = 1e9
    noisy = _add_noise(pd.Series(values), 0, np.random.default_rng(1))
    added = noisy - pd.Series(values)
    assert float(added.abs().median()) < 2.0


def test_noise_scale_falls_back_to_one_for_a_constant_column() -> None:
    column = pd.Series(np.full(20_000, 3.0))
    noisy = _add_noise(column, 0, np.random.default_rng(1))
    assert (noisy != column).all()
    assert float((noisy - column).abs().median()) == pytest.approx(0.765, rel=0.05)


def test_noise_scale_for_a_column_that_is_mostly_one_value_is_its_standard_deviation() -> None:
    """A rare event flag stored as floats has no spread by median or quartiles, but it has some."""
    flag = pd.Series((np.random.default_rng(0).random(20_000) < 0.03).astype("float64"))
    assert _compat.robust_scale(flag) == pytest.approx(float(np.std(flag.to_numpy())))
    noisy = _add_noise(flag, 0, np.random.default_rng(1))
    assert (noisy != flag).all()


def test_noise_scale_ignores_values_that_are_not_finite() -> None:
    values = np.random.default_rng(0).normal(0.0, 4.0, 20_000)
    values[0] = np.inf
    values[1] = np.nan
    column = pd.Series(values)
    noisy = _add_noise(column, 0, np.random.default_rng(1))
    assert np.isinf(noisy.iloc[0])
    assert np.isnan(noisy.iloc[1])
    assert (noisy - column).iloc[2:].abs().median() == pytest.approx(0.765 * 4.0, rel=0.05)


def test_robust_scale_is_never_zero() -> None:
    assert _compat.robust_scale(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])) == pytest.approx(1.4826)
    assert _compat.robust_scale(pd.Series([5.0, 5.0, 5.0])) == 1.0
    assert _compat.robust_scale(pd.Series([np.nan, np.inf])) == 1.0
    assert _compat.robust_scale(pd.Series(dtype="float64")) == 1.0
    assert _compat.robust_scale(pd.Series([0.0] * 97 + [1.0] * 3)) == pytest.approx(
        0.1714, abs=1e-3
    )


def test_noise_keeps_the_float_dtypes() -> None:
    rng = np.random.default_rng(0)
    floats32 = pd.Series(rng.normal(size=500), dtype="float32")
    assert _add_noise(floats32, 100, np.random.default_rng(1)).dtype == floats32.dtype
    nullable = pd.Series(rng.normal(size=500), dtype="Float64")
    nullable.iloc[[3, 400]] = pd.NA
    noisy = _add_noise(nullable, 100, np.random.default_rng(1))
    assert noisy.dtype == nullable.dtype
    assert noisy.isna().tolist() == nullable.isna().tolist()


def test_noise_that_overflows_a_narrow_float_dtype_becomes_infinite_without_a_warning() -> None:
    column = pd.Series(np.full(2_000, 60_000.0), dtype="float16") + pd.Series(
        np.random.default_rng(0).uniform(0.0, 4_000.0, 2_000), dtype="float16"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        noisy = _add_noise(column, 0, np.random.default_rng(1))
    assert noisy.dtype == column.dtype
    assert np.isinf(noisy.to_numpy(dtype="float64")).any()


# -- integers


def test_integers_move_to_a_different_value_of_the_observed_range() -> None:
    rng = np.random.default_rng(0)
    column = pd.Series(rng.integers(1_000, 5_000, 2_000))
    moved = _offset_integers(column, 500, np.random.default_rng(1))
    assert moved.dtype == column.dtype
    assert (moved.iloc[:500] == column.iloc[:500]).all()
    assert (moved.iloc[500:] != column.iloc[500:]).all()
    assert moved.min() >= column.min()
    assert moved.max() <= column.max()
    # The offsets are not small: the new values are spread over the whole range.
    assert (moved.iloc[500:] - column.iloc[500:]).abs().median() > 500


def test_a_two_valued_integer_column_toggles() -> None:
    """The case of an event column stored as int8: every future row flips."""
    column = pd.Series((np.random.default_rng(0).random(1_000) < 0.03).astype("int8"))
    moved = _offset_integers(column, 100, np.random.default_rng(1))
    assert moved.dtype == column.dtype
    assert (moved.iloc[100:] == 1 - column.iloc[100:]).all()


def test_a_constant_integer_column_moves_by_one_step_inside_the_dtype() -> None:
    constant = pd.Series(np.full(2_000, 7))
    moved = _offset_integers(constant, 0, np.random.default_rng(1))
    assert set(moved.unique()) == {6, 8}

    at_bottom = pd.Series(np.zeros(500, dtype="uint8"))
    assert set(_offset_integers(at_bottom, 0, np.random.default_rng(1)).unique()) == {1}
    at_top = pd.Series(np.full(500, 255, dtype="uint8"))
    assert set(_offset_integers(at_top, 0, np.random.default_rng(1)).unique()) == {254}


def test_integers_stay_inside_their_dtype() -> None:
    rng = np.random.default_rng(0)
    small = pd.Series(rng.integers(0, 200, 500), dtype="uint8")
    moved = _offset_integers(small, 0, np.random.default_rng(1))
    assert moved.dtype == small.dtype
    assert (moved != small).all()


def test_nullable_integers_keep_their_missing_values() -> None:
    rng = np.random.default_rng(0)
    nullable = pd.Series(rng.integers(0, 10_000, 500), dtype="Int64")
    nullable.iloc[[3, 400]] = pd.NA
    moved = _offset_integers(nullable, 100, np.random.default_rng(1))
    assert moved.dtype == nullable.dtype
    assert moved.isna().tolist() == nullable.isna().tolist()
    present = ~nullable.iloc[100:].isna()
    assert (moved.iloc[100:][present] != nullable.iloc[100:][present]).all()


def test_integers_with_an_enormous_range_are_permuted_instead() -> None:
    wide_range = pd.Series(np.array([-(2**62), 2**62, 5, 17, -3, 2**61] * 50, dtype="int64"))
    moved = _offset_integers(wide_range, 60, np.random.default_rng(1))
    assert moved.dtype == wide_range.dtype
    assert sorted(moved.iloc[60:]) == sorted(wide_range.iloc[60:])


def test_a_column_without_values_is_left_alone() -> None:
    empty = pd.Series([pd.NA, pd.NA, pd.NA], dtype="Int64")
    assert _offset_integers(empty, 1, np.random.default_rng(1)).isna().all()


# -- booleans


def test_the_first_future_boolean_is_always_flipped_and_about_half_of_the_others() -> None:
    column = pd.Series(np.random.default_rng(0).random(10_000) < 0.03)
    for seed in range(40):
        flipped = _flip_booleans(column, 9_000, np.random.default_rng(seed))
        assert flipped.dtype == column.dtype
        assert flipped.iloc[9_000] != column.iloc[9_000]
        assert (flipped.iloc[:9_000] == column.iloc[:9_000]).all()
    changed = (_flip_booleans(column, 0, np.random.default_rng(0)) != column).mean()
    assert changed == pytest.approx(0.5, abs=0.03)


def test_flipping_nullable_booleans_keeps_the_missing_values() -> None:
    column = pd.Series([True, None, False, True, None, False] * 20, dtype="boolean")
    flipped = _flip_booleans(column, 6, np.random.default_rng(2))
    assert flipped.dtype == column.dtype
    assert flipped.isna().tolist() == column.isna().tolist()


# -- categories and strings


def test_categories_are_replaced_by_another_observed_value() -> None:
    rng = np.random.default_rng(0)
    values = rng.choice(["buy", "sell", "hold"], 3_000)
    for column in (
        pd.Series(values),
        pd.Series(pd.Categorical(values, categories=["buy", "sell", "hold", "never_seen"])),
        pd.Series(values, dtype=object),
    ):
        swapped = _swap_categories(column, 1_000, np.random.default_rng(1))
        assert swapped.dtype == column.dtype
        assert (swapped.iloc[:1_000] == column.iloc[:1_000]).all()
        assert (swapped.iloc[1_000:] != column.iloc[1_000:]).all()
        assert set(swapped.dropna().astype(object)) <= {"buy", "sell", "hold"}


def test_a_column_with_a_single_value_is_permuted_and_so_stays_as_it_is() -> None:
    column = pd.Series(["only"] * 50)
    swapped = _swap_categories(column, 10, np.random.default_rng(1))
    assert swapped.equals(column)


def test_values_that_cannot_be_hashed_are_permuted() -> None:
    column = pd.Series([[1, 2], [3], [4, 5, 6], [7]] * 25, dtype=object)
    swapped = _swap_categories(column, 20, np.random.default_rng(1))
    assert swapped.dtype == object
    assert swapped.iloc[:20].tolist() == column.iloc[:20].tolist()
    assert sorted(map(tuple, swapped.iloc[20:])) == sorted(map(tuple, column.iloc[20:]))


def test_dates_are_permuted_among_the_future_rows() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=100, freq="D"))
    data = pd.DataFrame({"filed": dates, "x": np.arange(100.0)})
    perturbed = _perturb(data, 40, seed=0)
    assert isinstance(perturbed, pd.DataFrame)
    assert perturbed["filed"].dtype == dates.dtype
    assert perturbed["filed"].iloc[:40].equals(dates.iloc[:40])
    assert sorted(perturbed["filed"].iloc[40:]) == sorted(dates.iloc[40:])
    assert n_changed(dates.iloc[40:], perturbed["filed"].iloc[40:]) > 50


# -- sparse columns


def make_sparse_frame(n: int = N_ROWS, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.where(rng.random(n) < 0.1, rng.normal(size=n), 0.0)
    return pd.DataFrame(
        {
            "x": pd.arrays.SparseArray(values, fill_value=0.0),
            "event": pd.arrays.SparseArray(rng.random(n) < 0.05, fill_value=False),
            "count": pd.arrays.SparseArray(rng.poisson(0.1, n), fill_value=0),
        },
        index=pd.bdate_range("2020-01-01", periods=n, name="date"),
    )


def test_sparse_columns_are_perturbed_and_stay_sparse() -> None:
    """The perturbation used to fail with a TypeError on the std of a sparse column."""
    data = make_sparse_frame()
    perturbed = _perturb(data, 100, seed=0)
    assert isinstance(perturbed, pd.DataFrame)
    assert perturbed.dtypes.equals(data.dtypes)
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in perturbed.dtypes)
    pd.testing.assert_frame_equal(perturbed.iloc[:100], data.iloc[:100])
    dense_before = data.sparse.to_dense()
    dense_after = perturbed.sparse.to_dense()
    assert (dense_after["x"].iloc[100:] != dense_before["x"].iloc[100:]).all()
    assert (dense_after["count"].iloc[100:] != dense_before["count"].iloc[100:]).all()
    assert dense_after["event"].iloc[100] != dense_before["event"].iloc[100]


def test_a_sparse_series_is_perturbed_and_stays_sparse() -> None:
    series = make_sparse_frame()["x"]
    perturbed = _perturb(series, 100, seed=0)
    assert isinstance(perturbed, pd.Series)
    assert perturbed.dtype == series.dtype


def test_sparse_input_can_be_checked() -> None:
    data = make_sparse_frame()
    clean = check(lambda df: df.astype("float64").rolling(3).mean(), data, cutoffs=FEW)
    assert_clean(clean)
    leaky = check(lambda df: df.astype("float64").shift(-1), data, cutoffs=FEW)
    assert_leaky(leaky)
    assert leaky.horizon_rows == 1
    # A function that keeps the sparse dtype in its output is compared like any other.
    assert_leaky(check(lambda df: df[["x"]].shift(-1), data, cutoffs=FEW))


def test_a_column_that_cannot_be_perturbed_is_a_clear_error_that_truncation_avoids(
    monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    """An error raised while a column is changed becomes one that names the column."""

    def refuses(column: pd.Series, start: int, rng: np.random.Generator) -> pd.Series:
        raise TypeError("this dtype does not support that")

    monkeypatch.setattr(oracle, "_perturb_column", refuses)
    with pytest.raises(PeekproofError, match=r"cannot perturb column 'close' of dtype float64"):
        check(rolling_mean, prices, cutoffs=FEW)
    with pytest.raises(PeekproofError, match="mode='truncate'"):
        check(rolling_mean, prices, cutoffs=FEW, mode="perturb")
    # The truncation test never changes the data, so it is not affected.
    assert_clean(check(rolling_mean, prices, cutoffs=FEW, mode="truncate"))


# --------------------------------------------------------------------------- #
# Comparison helpers in _compat
# --------------------------------------------------------------------------- #


def test_numeric_comparison_uses_tolerances_and_treats_nan_as_equal() -> None:
    a = pd.Series([1.0, np.nan, 2.0, 3.0, np.inf])
    b = pd.Series([1.0 + 1e-12, np.nan, 2.1, np.nan, np.inf])
    diff = _compat.compare_columns(a, b, atol=1e-9, rtol=1e-7)
    assert diff.differs.tolist() == [False, False, True, True, False]
    assert diff.max_abs_diff == pytest.approx(0.1)


def test_difference_between_a_value_and_a_missing_value_has_no_size() -> None:
    diff = _compat.compare_columns(
        pd.Series([1.0, 2.0]), pd.Series([1.0, np.nan]), atol=1e-9, rtol=1e-7
    )
    assert diff.differs.tolist() == [False, True]
    assert diff.max_abs_diff is None


def test_non_numeric_comparison_uses_equality_and_treats_missing_as_equal() -> None:
    a = pd.Series(["x", None, "y", "z", None])
    b = pd.Series(["x", None, "q", None, "w"])
    diff = _compat.compare_columns(a, b, atol=1e-9, rtol=1e-7)
    assert diff.differs.tolist() == [False, False, True, True, True]
    assert diff.max_abs_diff is None


def test_boolean_categorical_and_nullable_columns_are_compared_for_equality() -> None:
    flags = _compat.compare_columns(
        pd.Series([True, False, True]), pd.Series([True, True, True]), atol=0.0, rtol=0.0
    )
    assert flags.differs.tolist() == [False, True, False]

    left = pd.Series(pd.Categorical(["a", None, "b"]))
    right = pd.Series(pd.Categorical(["a", None, "c"], categories=["a", "b", "c"]))
    cats = _compat.compare_columns(left, right, atol=0.0, rtol=0.0)
    assert cats.differs.tolist() == [False, False, True]

    nullable = _compat.compare_columns(
        pd.Series([1, None, 3], dtype="Int64"),
        pd.Series([1, None, 4], dtype="Int64"),
        atol=0.0,
        rtol=0.0,
    )
    assert nullable.differs.tolist() == [False, False, True]
    assert nullable.max_abs_diff == pytest.approx(1.0)

    objects = _compat.compare_columns(
        pd.Series([pd.NA, "a", 1], dtype=object),
        pd.Series([pd.NA, "a", 2], dtype=object),
        atol=0.0,
        rtol=0.0,
    )
    assert objects.differs.tolist() == [False, False, True]


def test_sparse_columns_are_compared_like_dense_ones() -> None:
    left = pd.Series(pd.arrays.SparseArray([0.0, 1.5, 0.0, np.nan], fill_value=0.0))
    right = pd.Series(pd.arrays.SparseArray([0.0, 1.5, 2.0, np.nan], fill_value=0.0))
    assert _compat.compare_columns(left, right, atol=1e-9, rtol=1e-7).differs.tolist() == [
        False,
        False,
        True,
        False,
    ]
    flags = pd.Series(pd.arrays.SparseArray([True, False, False], fill_value=False))
    other = pd.Series(pd.arrays.SparseArray([True, True, False], fill_value=False))
    assert _compat.compare_columns(flags, other, atol=0.0, rtol=0.0).differs.tolist() == [
        False,
        True,
        False,
    ]


def test_frames_equal_needs_same_index_columns_and_values() -> None:
    base = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]}, index=[10, 20])
    assert _compat.frames_equal(base, base.copy(), atol=0.0, rtol=0.0)
    assert not _compat.frames_equal(base, base.rename(columns={"a": "c"}), atol=0.0, rtol=0.0)
    assert not _compat.frames_equal(base, base.set_axis([10, 21]), atol=0.0, rtol=0.0)
    changed = base.copy()
    changed.iloc[0, 0] = 2.0
    assert not _compat.frames_equal(base, changed, atol=0.0, rtol=0.0)


def test_labels_are_strings_of_the_index_labels(prices: pd.DataFrame) -> None:
    assert _compat.label_at(prices.index, 3) == format_label(prices.index[3])
    assert _compat.labels_at(prices.index, np.array([0, 2])) == [
        format_label(prices.index[0]),
        format_label(prices.index[2]),
    ]
    assert _compat.label_at(pd.RangeIndex(5), 4) == "4"


def test_to_pandas_passes_pandas_objects_through(prices: pd.DataFrame) -> None:
    assert _compat.to_pandas(prices) is prices
    close = prices["close"]
    assert _compat.to_pandas(close) is close
    with pytest.raises(PeekproofError, match="neither a column nor"):
        _compat.to_pandas(prices, time_col="nope")
    assert _compat.to_pandas(prices, time_col="date") is prices


def test_a_long_format_panel_is_rejected_with_a_hint() -> None:
    """Cutoffs by row would split the assets of one date, so a MultiIndex is refused."""
    dates = pd.bdate_range("2021-01-04", periods=50, name="date")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "asset"])
    long = pd.DataFrame({"close": np.random.default_rng(0).normal(size=150)}, index=index)
    with pytest.raises(PeekproofError, match="one wide frame per feature"):
        check(lambda d: d.groupby(level="date")["close"].rank(), long)
