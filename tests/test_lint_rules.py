"""Tests for the lint rules PP001 to PP013.

Three layers:

1. The registry: every rule has the metadata that docs/DESIGN.md promises.
2. One leaky and one clean fixture file per rule. The leaky file fires exactly its own
   rule, the clean file fires nothing.
3. Small inline snippets that pin down each trigger and each exemption of a rule.
"""

from __future__ import annotations

import ast
import textwrap
import time
import types
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import peekproof.lint.rules as rules_module
from peekproof.lint import RULES, lint_source
from peekproof.lint.engine import LintContext
from peekproof.report import Finding

FIXTURES = Path(__file__).parent / "fixtures"
RULE_IDS = [f"PP{n:03d}" for n in range(1, 14)]
CONFIRM = "Confirm with peekproof.check()."

# Name and default confidence of every rule, as listed in docs/DESIGN.md section 5.2.
RULE_TABLE = {
    "PP001": ("negative-shift", "high"),
    "PP002": ("backward-fill", "high"),
    "PP003": ("centered-window", "high"),
    "PP004": ("fit-on-full-sample", "medium"),
    "PP005": ("shuffled-split", "high"),
    "PP006": ("full-sample-statistic", "medium"),
    "PP007": ("forward-asof-join", "high"),
    "PP008": ("bar-label-leak", "low"),
    "PP009": ("forward-index-in-loop", "high"),
    "PP010": ("same-bar-execution", "medium"),
    "PP011": ("target-as-feature", "medium"),
    "PP012": ("current-constituents", "medium"),
    "PP013": ("latest-fundamentals", "low"),
}


def lint_snippet(source: str, **kwargs: Any) -> list[Finding]:
    """Lint an indented snippet and make sure that it parses."""
    findings = lint_source(textwrap.dedent(source) + "\n", **kwargs)
    assert all(f.rule_id != "PP000" for f in findings), f"snippet does not parse: {source!r}"
    return findings


def lint_fixture(kind: str, rule_id: str) -> tuple[Path, list[Finding]]:
    path = FIXTURES / kind / f"{rule_id.lower()}.py"
    return path, lint_source(path.read_text(encoding="utf-8"), str(path))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_lists_exactly_the_thirteen_rules_in_order() -> None:
    assert list(RULES) == RULE_IDS


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_rule_metadata_matches_the_spec_table(rule_id: str) -> None:
    rule = RULES[rule_id]
    name, confidence = RULE_TABLE[rule_id]
    assert rule.id == rule_id
    assert rule.name == name
    assert rule.confidence == confidence
    assert rule.message
    assert rule.hint
    assert "\n" not in rule.message
    assert "\n" not in rule.hint
    assert callable(rule.visit)


def test_rule_names_are_unique() -> None:
    names = [rule.name for rule in RULES.values()]
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------- #
# Fixture files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["leaky", "clean"])
def test_every_rule_has_a_fixture_file(kind: str) -> None:
    present = sorted(p.name for p in (FIXTURES / kind).glob("*.py"))
    assert present == [f"{rule_id.lower()}.py" for rule_id in RULE_IDS]


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_leaky_fixture_fires_exactly_its_rule(rule_id: str) -> None:
    path, findings = lint_fixture("leaky", rule_id)
    assert findings, f"{path.name} produced no findings"
    assert {f.rule_id for f in findings} == {rule_id}


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_clean_fixture_is_silent(rule_id: str) -> None:
    _path, findings = lint_fixture("clean", rule_id)
    assert findings == []


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_findings_carry_complete_metadata(rule_id: str) -> None:
    path, findings = lint_fixture("leaky", rule_id)
    source_lines = path.read_text(encoding="utf-8").splitlines()
    for finding in findings:
        assert finding.name == RULES[rule_id].name
        assert finding.path == str(path)
        assert 1 <= finding.line <= len(source_lines)
        assert finding.col >= 1
        assert finding.confidence in {"low", "medium", "high"}
        assert finding.message
        assert finding.hint
        assert "\n" not in finding.message
        assert "\n" not in finding.hint
        assert finding.snippet == source_lines[finding.line - 1].strip()


# --------------------------------------------------------------------------- #
# Inline snippets: (code, expected number of findings of that rule)
# --------------------------------------------------------------------------- #

CASES: dict[str, list[tuple[str, int]]] = {
    "PP001": [
        ("z = x.shift(-1)", 1),
        ("z = x.shift(periods=-1)", 1),
        ("z = x.shift(- 3)", 1),
        ("z = x.shift(-(1))", 1),
        ("z = x.shift(-1.5)", 1),
        ("z = x.pct_change(-5)", 1),
        ("z = x.pct_change(periods=-5)", 1),
        ("z = x.diff(-1)", 1),
        ("z = np.roll(x, -1)", 1),
        ("z = np.roll(x, shift=-1)", 1),
        ("import numpy as numpy_lib\nz = numpy_lib.roll(x, -2, axis=0)", 1),
        ("from numpy import roll\nz = roll(x, -1)", 1),
        ("z = np.roll(x, (-1, 2), axis=(0, 1))", 1),
        ("z = x.shift(n=-1)", 1),
        ("z = df.groupby('a')['b'].shift(-1)", 1),
        ("z = x.shift(-1).shift(-2)", 2),
        ("z = x.shift(1)", 0),
        ("z = x.shift(0)", 0),
        ("z = x.shift(k)", 0),
        ("z = x.shift()", 0),
        ("z = x.shift(fill_value=-1)", 0),
        ("z = np.roll(x, 1)", 0),
        ("z = np.roll(x, s)", 0),
        ("z = shift(-1)", 0),
        ("z = roll(x, -1)", 0),
        ("z = other.roll(x, -1)", 0),
        ("z = x.diff()", 0),
        ("def forward_return(x):\n    return x.shift(-5)", 0),
        ("def make_label(x):\n    return x.shift(-5)", 0),
        ("def f(x):  # peekproof: label\n    return x.shift(-5)", 0),
        ("def f(\n    x,\n):  # peekproof: label\n    return x.shift(-5)", 0),
        ("def f(x):\n    # peekproof: label\n    return x.shift(-5)", 0),
        (
            (
                "def f(x):\n    def inner(y):\n        return y.shift(-1)\n    return inner(x)\n\n"
                "def make_target(x):\n    return x"
            ),
            1,
        ),
        (
            (
                "def make_target(x):\n    def inner(y):\n        return y.shift(-1)\n"
                "    return inner(x)"
            ),
            0,
        ),
        ("class A:\n    def target(self, x):\n        return x.shift(-1)", 0),
        ("async def get_label(x):\n    return x.shift(-1)", 0),
        ("def label(x):\n    return x\n\ndef features(x):\n    return x.shift(-1)", 1),
        ("z = x.shift(-1)  # peekproof: ignore", 0),
        ("z = x.shift(-1)  # peekproof: ignore[PP001]", 0),
        ("z = x.shift(-1)  # peekproof: ignore[PP002]", 1),
        ("z = x.shift(-1)  # peekproof: ignore[PP002, PP001]", 0),
        ("z = x.shift(-1)  # peekproof: ignore[pp001]", 0),
        ("z = x.shift(-1)  # noqa  # peekproof: ignore", 0),
        ("s = '# peekproof: ignore'; z = x.shift(-1)", 1),
        # a shift across columns (maturities, assets) does not move values in time
        ("z = curve.shift(-1, axis=1)", 0),
        ("z = curve.shift(-1, axis='columns')", 0),
        ("z = curve.shift(-1, None, 1)", 0),
        ("z = curve.shift(-1, axis=0)", 1),
        ("z = curve.shift(-1, axis='index')", 1),
        ("z = x.diff(-1, axis=1)", 0),
        ("z = x.diff(-1, 1)", 0),
        ("z = x.diff(-1, axis=0)", 1),
        ("z = x.pct_change(-1, axis=1)", 0),
        ("z = np.roll(w, -1, axis=1)", 0),
        ("z = np.roll(w, -1, 1)", 0),
        ("z = np.roll(w, -1, axis=0)", 1),
        ("z = np.roll(w, -1, axis=(0, 1))", 1),
        # a negated variable may be negative or positive: medium confidence, still reported
        ("z = x.shift(-k)", 1),
        ("z = x.shift(-HORIZON)", 1),
        ("z = x.shift(periods=-n)", 1),
        ("z = x.shift(-self.horizon)", 1),
        ("z = x.shift(-(h + 1))", 1),
        ("z = x.shift(-1 * h)", 1),
        ("z = x.shift(h * -2)", 1),
        ("z = x.shift(h * 2)", 0),
        # a function of an imported module: the negated thing is the array, not the periods
        ("z = np.diff(-np.log(x))", 0),
        ("z = np.diff(-x)", 0),
        ("import numpy as xp\nz = xp.diff(-x)", 0),
        ("from scipy import ndimage\nz = ndimage.shift(-img, 3)", 0),
        ("z = jnp.diff(-x)", 1),
        ("z = df.diff(-n)", 1),
        ("z = self.prices.shift(-n)", 1),
        ("z = pd.Series(x).shift(-n)", 1),
        ("z = x.pct_change(-n)", 1),
        ("z = x.diff(-n)", 1),
        ("z = np.roll(x, -n)", 1),
        ("z = np.roll(x, (-n, 1), axis=(0, 1))", 1),
        ("z = x.shift(-n, axis=1)", 0),
        ("z = x.shift(+k)", 0),
        ("z = x.shift(-0)", 0),
        # a next-bar fill or entry price is medium confidence
        ("fill_price = df['open'].shift(-1)", 1),
        ("entry = open.shift(-1)", 1),
        ("exit_px = open.shift(-1)", 1),
        ("df['exec_price'] = df['open'].shift(-1)", 1),
        ("df['next_fill_price'] = df['open'].shift(-1)", 1),
        ("df = df.assign(execution_price=lambda d: d['open'].shift(-1))", 1),
        ("entry_signal = df['open'].shift(-1)", 1),
        # columns and variables that hold labels are label code, wherever they are built
        ("px['target'] = px['close'].shift(-1) / px['close'] - 1", 0),
        ("px['label'] = (px['close'].shift(-5) > px['close']).astype(int)", 0),
        ("px['fwd_ret'] = px['close'].pct_change(-5)", 0),
        ("px['fwd_ret_5d'] = px['close'].pct_change(-5)", 0),
        ("px['next_target'] = px['close'].shift(-1)", 0),
        ("px.loc[:, 'target'] = px['close'].shift(-1)", 0),
        ("y = x.shift(-5)", 0),
        ("Y = x.shift(-5)", 0),
        ("y_true = x.shift(-5)", 0),
        ("labels = x.shift(-5)", 0),
        ("forward_return = prices.shift(-5) / prices - 1", 0),
        ("forward_returns_5d = prices.shift(-5) / prices - 1", 0),
        ("y: pd.Series = x.shift(-1)", 0),
        ("df = df.assign(target=lambda d: d['close'].shift(-1))", 0),
        ("df = pd.DataFrame({'y': x.shift(-1)})", 0),
        ("a, y = x, x.shift(-1)", 0),
        ("a, y = x.shift(-1), x", 1),
        ("y = f(x.shift(-1))", 0),
        ("y = x.shift(-1)\nz = x.shift(-2)", 1),
        # ... but a name that merely starts with, or contains, a label word is not a label
        ("target_position = signal.shift(-1)", 1),
        ("df['target_position'] = signal.shift(-1)", 1),
        ("labeled_rows = x.shift(-1)", 1),
        ("feature = x.shift(-1)", 1),
        ("df['feature'] = df['close'].shift(-1)", 1),
        ("fwd = x.shift(-5)", 1),
        ("df['target'] = f(x)\nz = x.shift(-1)", 1),
        ("df.loc[x.shift(-1) > 0, 'a'] = 1", 1),
        # label functions are matched by name (the name has to END with a label word, or hold a
        # forward-return word) or by the marker comment
        ("def FORWARD_RETURN(x):\n    return x.shift(-5)", 0),
        ("def fwd_ret_5d(x):\n    return x.shift(-5)", 0),
        ("def forward_returns(x):\n    return x.shift(-5)", 0),
        ("def get_y(x):\n    return x.shift(-5)", 0),
        ("def build_targets(x):\n    return x.shift(-5)", 0),
        ("def target_position(x):\n    return x.shift(-1)", 1),
        ("def forward_fill_features(x):\n    return x.shift(-1)", 1),
        ("def labeled_features(x):\n    return x.shift(-1)", 1),
        ("def targets_first(x):\n    return x.shift(-1)", 1),
        ("def forward_looking(x):\n    return x.shift(-1)", 1),
    ],
    "PP002": [
        ("y = x.bfill()", 1),
        ("y = x.backfill()", 1),
        ("y = x.fillna(method='bfill')", 1),
        ("y = x.fillna(method='backfill', limit=2)", 1),
        ("y = x.interpolate(limit_direction='backward')", 1),
        ("y = x.interpolate(method='linear', limit_direction='both')", 1),
        ("y = x.groupby('k').bfill()", 1),
        ("y = pl.col('a').backward_fill()", 1),
        ("y = pl.col('a').fill_null(strategy='backward')", 1),
        ("y = x.ffill().bfill()", 1),
        ("y = x.ffill()", 0),
        ("y = x.fillna(0)", 0),
        ("y = x.fillna(method='ffill')", 0),
        ("y = x.fillna(method=m)", 0),
        ("y = bfill(x)", 0),
        # axis=1 fills across columns (maturities), not across time
        ("y = yields.bfill(axis=1)", 0),
        ("y = yields.bfill(axis='columns')", 0),
        ("y = yields.backfill(axis=1)", 0),
        ("y = yields.fillna(method='bfill', axis=1)", 0),
        ("y = yields.interpolate(axis=1)", 0),
        ("y = yields.bfill(axis=0)", 1),
        ("y = yields.bfill(axis='index')", 1),
        # linear interpolation bridges a gap with the next observation (medium confidence)
        ("y = x.interpolate()", 1),
        ("y = x.interpolate(method='linear')", 1),
        ("y = x.interpolate('time')", 1),
        ("y = x.interpolate(method='time')", 1),
        ("y = x.interpolate(method='cubic', limit=3)", 1),
        ("from numpy.polynomial import Chebyshev\ny = Chebyshev.interpolate(f, 3)", 0),
        ("import numpy as np\ny = np.polynomial.Chebyshev.interpolate(f, 3)", 0),
        ("y = pd.Series(x).interpolate()", 1),
        ("y = self.frame.interpolate()", 1),
        ("y = x.interpolate(limit_direction=None)", 1),
        # limit_direction only selects which edge gaps are filled; interior gaps still use
        # the next observation
        ("y = x.interpolate(limit_direction='forward')", 1),
        ("y = x.interpolate(method='time', limit_direction='forward')", 1),
        ("y = x.interpolate(method='pad')", 0),
        ("y = x.interpolate(method='ffill')", 0),
        ("y = x.interpolate(limit_direction=direction)", 1),
        ("y = x.interpolate(**options)", 0),
        ("y = pl.col('a').interpolate()", 1),
        ("y = df.select(pl.all().interpolate())", 1),
        ("y = pl.col('a').interpolate(method='nearest')", 1),
        # reindex, asfreq and pct_change can fill backwards too
        ("y = x.reindex(idx, method='bfill')", 1),
        ("y = x.reindex(idx, method='backfill')", 1),
        ("y = x.reindex(idx, method='nearest')", 1),
        ("y = x.reindex_like(other, method='bfill')", 1),
        ("y = x.reindex_like(other, method='ffill')", 0),
        ("y = x.reindex_like(other)", 0),
        ("y = x.reindex(idx, method='nearest', tolerance=1)", 1),
        ("y = x.reindex(idx, method='ffill')", 0),
        ("y = x.reindex(idx, method='pad')", 0),
        ("y = x.reindex(idx)", 0),
        ("y = x.reindex(idx, method=m)", 0),
        ("y = x.asfreq('D', method='bfill')", 1),
        ("y = x.asfreq('D', method='backfill')", 1),
        ("y = x.asfreq('D', 'bfill')", 1),
        ("y = x.asfreq('D', method='ffill')", 0),
        ("y = x.asfreq('D')", 0),
        ("y = x.pct_change(fill_method='bfill')", 1),
        ("y = x.pct_change(fill_method='backfill')", 1),
        ("y = x.pct_change(1, 'bfill')", 1),
        ("y = x.pct_change(fill_method='ffill')", 0),
        ("y = x.pct_change(fill_method=None)", 0),
        ("y = x.pct_change()", 0),
        ("y = x.reindex(idx, method='bfill', axis=1)", 0),
    ],
    "PP003": [
        ("y = x.rolling(5, center=True).mean()", 1),
        ("y = x.rolling(window=5, min_periods=1, center=True).std()", 1),
        ("y = x.rolling(5, 1, True).mean()", 1),
        ("y = pl.col('a').rolling_mean(window_size=5, center=True)", 1),
        ("y = x.rolling(5, center=False).mean()", 0),
        ("y = x.rolling(5).mean()", 0),
        ("y = x.rolling(5, center=flag).mean()", 0),
        ("y = x.ewm(span=5).mean()", 0),
        ("from scipy.signal import savgol_filter\ny = savgol_filter(x, 11, 3)", 1),
        ("from scipy import signal\ny = signal.savgol_filter(x, 11, 3)", 1),
        ("import scipy.signal\ny = scipy.signal.savgol_filter(x, 11, 3)", 1),
        ("y = savgol_filter(x, 11, 3)", 1),
        ("from mylib import savgol_filter\ny = savgol_filter(x, 11, 3)", 0),
        ("def savgol_filter(x):\n    return x\ny = savgol_filter(x)", 0),
        ("y = np.convolve(x, k, mode='same')", 1),
        ("y = np.convolve(x, k, 'same')", 1),
        ("y = np.convolve(x, k, mode='full')", 1),
        ("y = np.convolve(x, k)", 1),
        ("y = np.convolve(x, k, mode='valid')", 0),
        ("y = np.convolve(x, k, mode=m)", 0),
        ("from scipy.signal import convolve\ny = convolve(x, k)", 0),
        ("from scipy import ndimage\ny = ndimage.gaussian_filter1d(x, 2)", 1),
        ("import scipy.ndimage as ndi\ny = ndi.uniform_filter1d(x, 3)", 1),
        ("from scipy.ndimage import median_filter\ny = median_filter(x, 3)", 1),
        ("import scipy.ndimage\ny = scipy.ndimage.maximum_filter(x, 3)", 1),
        ("y = ndi.gaussian_filter(x, 2)", 1),
        ("y = gaussian_filter1d(x, 2)", 1),
        ("import skimage.ndimage as ndi\ny = ndi.gaussian_filter(x, 2)", 0),
        ("from scipy import ndimage\ny = ndimage.convolve1d(x, k)", 0),
        ("y = other.gaussian_filter(x, 2)", 0),
        # a centered window of one or two rows still ends on the current row (pandas looks
        # ahead from three rows on); a window of unknown size is reported
        ("y = x.rolling(2, center=True).mean()", 0),
        ("y = x.rolling(window=2, center=True).mean()", 0),
        ("y = x.rolling(1, center=True).mean()", 0),
        ("y = x.rolling(2, 1, True).mean()", 0),
        ("y = pl.col('a').rolling_mean(window_size=2, center=True)", 0),
        ("y = x.rolling(3, center=True).mean()", 1),
        ("y = x.rolling(4, center=True).mean()", 1),
        ("y = x.rolling(window=9, center=True).mean()", 1),
        ("y = x.rolling('5D', center=True).mean()", 1),
        ("y = x.rolling(w, center=True).mean()", 1),
        ("y = x.rolling(center=True, window=3).mean()", 1),
        ("y = pl.col('a').rolling_mean(window_size=3, center=True)", 1),
        # two-sided filters
        ("from scipy.signal import filtfilt\ny = filtfilt(b, a, x)", 1),
        ("from scipy import signal\ny = signal.filtfilt(b, a, x)", 1),
        ("from scipy.signal import sosfiltfilt\ny = sosfiltfilt(sos, x)", 1),
        ("y = filtfilt(b, a, x)", 1),
        ("from mylib import filtfilt\ny = filtfilt(b, a, x)", 0),
        ("from scipy.signal import lfilter\ny = lfilter(b, a, x)", 0),
        ("from statsmodels.tsa.filters.hp_filter import hpfilter\ncycle, trend = hpfilter(x)", 1),
        ("import statsmodels.api as sm\ncycle, trend = sm.tsa.filters.hpfilter(x, 1600)", 1),
        ("cycle, trend = sm.tsa.filters.hpfilter(x, 1600)", 1),
        ("cycle, trend = hpfilter(x, lamb=1600)", 1),
        ("from mylib import hpfilter\ncycle, trend = hpfilter(x)", 0),
        ("from statsmodels.tsa.seasonal import seasonal_decompose\ny = seasonal_decompose(x)", 1),
        ("y = seasonal_decompose(x, model='additive', two_sided=False)", 1),
        ("import statsmodels.api as sm\ny = sm.tsa.seasonal_decompose(x, period=12)", 1),
        ("from mylib import seasonal_decompose\ny = seasonal_decompose(x)", 0),
        ("y = np.gradient(x)", 1),
        ("y = np.gradient(x, 2.0)", 1),
        ("import numpy\ny = numpy.gradient(x)", 1),
        ("from numpy import gradient\ny = gradient(x)", 1),
        ("y = torch.gradient(x)", 0),
        ("y = x.diff()", 0),
        ("y = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)", 1),
        (
            "from pandas.api import indexers\ny = indexers.FixedForwardWindowIndexer(3)",
            1,
        ),
        ("y = FixedForwardWindowIndexer(window_size=n)", 1),
        ("y = FixedForwardWindowIndexer(window_size=1)", 0),
        (
            "from mylib import FixedForwardWindowIndexer\ny = FixedForwardWindowIndexer(3)",
            0,
        ),
        # np.convolve in mode 'full' is causal when it is cut to the input length
        ("y = np.convolve(x, k)[:n]", 0),
        ("y = np.convolve(x, k)[:len(x)]", 0),
        ("y = np.convolve(x, k, mode='full')[:n]", 0),
        ("y = np.convolve(x, k, 'full')[: -(len(k) - 1)]", 0),
        ("y = np.convolve(x, k)[0:len(x)]", 0),
        ("y = np.convolve(x, k)[:-4]", 0),
        ("y = np.convolve(x, k)[len(k) - 1 : len(x)]", 1),
        ("y = np.convolve(x, k)[2:-2]", 1),
        ("y = np.convolve(x, k, mode='same')[:n]", 1),
        ("y = np.convolve(x, k)[2:]", 1),
        ("y = np.convolve(x, k)[::2]", 1),
        ("c = np.convolve(x, k)\ny = c[:n]", 1),
    ],
    "PP004": [
        (
            (
                "s = StandardScaler()\nxs = s.fit_transform(X)\n"
                "Xtr, Xte, ytr, yte = train_test_split(xs, y, shuffle=False)"
            ),
            1,
        ),
        ("s = StandardScaler()\ns.fit(X)\nXtr = X.iloc[:100]\nXte = X.iloc[100:]", 1),
        ("s.fit(X)\nXtr = X[:n]", 1),
        ("X = s.fit_transform(X)\nXtr = X[:n]", 1),
        ("for tr, te in tscv.split(X):\n    pass\n", 0),
        ("xs = s.fit_transform(X)\nfor tr, te in tscv.split(xs):\n    a = xs[tr]", 1),
        ("est.fit(X, y)\nsc = cross_val_score(est, X, y, cv=tscv)", 1),
        ("s.fit(X)\nXtr = X[train_idx]", 1),
        ("s.fit(X)\nz = X['close']", 0),
        ("s.fit(X)\nz = X[['a', 'b']]", 0),
        ("s.fit(X)\nz = X.loc[:, 'a']", 0),
        ("s.fit(X)\nz = X[:, 1:3]", 0),
        ("s.fit(X)\nz = X.values", 0),
        ("Xtr, Xte = train_test_split(X, shuffle=False)\ns.fit(Xtr)\nz = Xtr[:5]", 0),
        ("s.fit(X[:n])\nz = X[n:]", 0),
        ("s.fit(X.iloc[:n])\nz = X.iloc[n:]", 0),
        ("s.fit(X.head(100))\nz = X.iloc[100:]", 0),
        ("s.fit(X)\nXtr = X[value_mask]", 0),
        ("def f(X):\n    s.fit(X)\n\ndef g(X):\n    return X[:5]", 0),
        ("def f(X):\n    s.fit(X)\n    return X[:5]", 1),
        ("s.fit()\nz = X[:5]", 0),
        ("Xs = pd.DataFrame(s.fit_transform(X), columns=c)\nz = Xs.iloc[:5]", 1),
        ("z = X[:5]\ns.fit(X)", 0),
    ],
    "PP005": [
        ("from sklearn.model_selection import train_test_split\na = train_test_split(X, y)", 1),
        ("a = train_test_split(X, y, test_size=0.2, random_state=1)", 1),
        ("a = train_test_split(X, y, shuffle=True)", 1),
        ("a = train_test_split(X, y, shuffle=False)", 0),
        ("a = train_test_split(X, y, shuffle=flag)", 0),
        ("a = train_test_split(X, y, **kw)", 0),
        ("import sklearn.model_selection as ms\na = ms.train_test_split(X, y)", 1),
        ("from sklearn import model_selection\na = model_selection.train_test_split(X, y)", 1),
        ("from sklearn.model_selection import train_test_split as tts\na = tts(X, y)", 1),
        ("from mylib import train_test_split\na = train_test_split(X, y)", 0),
        ("def train_test_split(df, f):\n    return df\na = train_test_split(X, 0.2)", 0),
        ("a = self.train_test_split(X, y)", 0),
        ("a = KFold(n_splits=5, shuffle=True, random_state=0)", 1),
        ("a = KFold(n_splits=5)", 0),
        ("a = KFold(5, shuffle=False)", 0),
        ("a = ShuffleSplit(n_splits=3)", 1),
        ("a = StratifiedShuffleSplit(n_splits=3)", 1),
        ("a = GroupShuffleSplit(n_splits=3)", 1),
        ("a = StratifiedKFold(5)", 1),
        ("a = PurgedKFold(5, shuffle=True)", 0),
        ("a = cross_val_score(m, X, y)", 1),
        ("a = cross_val_score(m, X, y, cv=5)", 1),
        ("a = cross_validate(m, X, y, cv=KFold(5))", 1),
        ("a = cross_val_predict(m, X, y, cv=kf)", 1),
        ("a = cross_val_score(m, X, y, cv=TimeSeriesSplit(5))", 0),
        ("a = cross_val_score(m, X, y, cv=PurgedKFold(5))", 0),
        ("a = cross_val_score(m, X, y, cv=tscv)", 0),
        ("splitter = TimeSeriesSplit(5)\na = cross_val_score(m, X, y, cv=splitter)", 0),
        ("splitter = KFold(5)\na = cross_val_score(m, X, y, cv=splitter)", 1),
        ("a = cross_val_score(m, X, y, **kw)", 0),
        # hyper-parameter searches: an integer or missing cv= is not time-aware (medium)
        ("gs = GridSearchCV(est, grid, cv=5)", 1),
        ("gs = GridSearchCV(est, grid)", 1),
        ("gs = GridSearchCV(est, grid, cv=None)", 1),
        ("gs = GridSearchCV(est, param_grid=grid, scoring='r2', n_jobs=-1)", 1),
        ("gs = RandomizedSearchCV(est, dist, n_iter=10)", 1),
        ("gs = RandomizedSearchCV(est, dist, cv=3)", 1),
        ("gs = HalvingGridSearchCV(est, grid)", 1),
        ("gs = HalvingRandomSearchCV(est, dist, cv=4)", 1),
        ("gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(5))", 0),
        ("gs = GridSearchCV(est, grid, cv=PurgedKFold(5))", 0),
        ("gs = GridSearchCV(est, grid, cv=tscv)", 0),
        ("gs = GridSearchCV(est, grid, cv=n_folds)", 0),
        ("gs = GridSearchCV(est, grid, **options)", 0),
        ("gs = GridSearchCV(est, grid, cv=5, **options)", 0),
        ("from sklearn.model_selection import GridSearchCV\ngs = GridSearchCV(est, grid, cv=5)", 1),
        ("from mylib import GridSearchCV\ngs = GridSearchCV(est, grid)", 0),
        ("gs = self.GridSearchCV(est, grid)", 0),
        # KFold without shuffle keeps the rows in order, so it is not reported (not even as the cv)
        ("a = KFold()", 0),
        ("a = KFold(5, shuffle=False)", 0),
        ("gs = GridSearchCV(est, grid, cv=KFold(5))", 0),
        ("gs = GridSearchCV(est, grid, cv=KFold(5, shuffle=True))", 1),
    ],
    "PP006": [
        ("df['z'] = (df['c'] - df['c'].mean()) / df['c'].std()", 1),
        ("df['z'] = (df[\"c\"] - df['c'].mean()) / df['c'].std()", 1),
        ("z = (x - x.mean()) / x.std()", 1),
        ("z = x / x.max()", 1),
        ("z = (x - x.min()) / (x.max() - x.min())", 1),
        ("z = x - x.median()", 1),
        ("z = x > x.quantile(0.9)", 1),
        ("z = x.sub(x.mean())", 1),
        ("z = x.div(x.std())", 1),
        ("x -= x.mean()", 1),
        ("df['x'] -= df['x'].mean()", 1),
        ("z = df.assign(z=lambda d: (d.x - d.x.mean()) / d.x.std())", 1),
        ("z = df.groupby('sym')['x'].transform(lambda s: (s - s.mean()) / s.std())", 1),
        ("z = x.rank(pct=True)", 1),
        ("df['r'] = df['c'].rank()", 1),
        ("z = scipy.stats.zscore(x)", 1),
        ("from scipy.stats import zscore\nz = zscore(x)", 1),
        ("z = x - x.mean() + y - y.mean()", 1),
        ("z = (x - x.expanding().mean()) / x.expanding().std()", 0),
        ("z = (x - x.rolling(20).mean()) / x.rolling(20).std()", 0),
        ("z = x.mean() / x.std()", 0),
        ("z = x.mean() * 252", 0),
        ("z = x.mean() / x.shift(1).std()", 0),
        ("z = x - y.mean()", 0),
        ("z = x.rank(axis=1)", 0),
        ("z = df.rank(axis=1, pct=True)", 0),
        ("z = x.rolling(20).rank()", 0),
        ("z = x.expanding().rank()", 0),
        ("z = df.groupby('date')['x'].rank()", 0),
        ("z = df.groupby('sym')['x'].rank()", 0),
        ("z = df.groupby('date')['x'].transform(lambda s: (s - s.mean()) / s.std())", 0),
        ("z = df.groupby(level=0)['x'].transform(lambda s: (s - s.mean()) / s.std())", 0),
        ("z = scipy.stats.zscore(x, axis=1)", 0),
        ("print(x - x.mean())", 0),
        ("def f(x):\n    return (x - x.mean()) / x.std()", 1),
        ("def f(x):\n    return (x - x.min()) / (x.max() - x.min())", 1),
        ("def f(df):\n    return df.rank(pct=True)", 1),
        ("def f(df):\n    return df.assign(z=lambda d: (d.x - d.x.mean()) / d.x.std())", 1),
        ("f = lambda x: (x - x.mean()) / x.std()", 1),
        ("def f(x):\n    return x.mean()", 0),
        ("def f(x):\n    return x.mean() / x.std()", 0),
        ("def f(x):\n    return (x - x.expanding().mean()) / x.expanding().std()", 0),
        ("def f(df):\n    return df.rank(axis=1)", 0),
        ("def f(df):\n    return", 0),
        ("z = (x - x.mean(axis=1)) / x.std(axis=1)", 0),
        ("z = x.max() - x.min()", 0),
        # a statistic that is stored first and combined with the series on a later line
        ("mu = x.mean()\nz = x - mu", 1),
        ("mu = x.mean()\nsd = x.std()\nz = (x - mu) / sd", 1),
        ("mu = df['c'].mean()\ndf['z'] = df['c'] - mu", 1),
        ("mu = x.mean()\nx -= mu", 1),
        ("cut = x.median()\nflag = x > cut", 1),
        ("lo = x.quantile(0.01)\nhi = x.quantile(0.99)\nz = x.clip(lo, hi)", 1),
        ("hi = x.quantile(0.99)\nz = x.clip(upper=hi)", 1),
        ("mu = x.mean()\nsd = x.std()\nhi = mu + 3 * sd\nz = x.clip(upper=hi)", 1),
        ("lo = x.min()\nhi = x.max()\nz = (x - lo) / (hi - lo)", 1),
        ("mu, sd = x.mean(), x.std()\nz = (x - mu) / sd", 1),
        ("def f(x):\n    mu = x.mean()\n    sd = x.std()\n    return (x - mu) / sd", 1),
        ("def f(x):\n    mu = x.mean()\n    return x.sub(mu)", 1),
        ("mu = x.mean()\nz = x - mu\nw = x + mu", 2),
        ("mu = x.mean()\nz = (x - mu) / x.std()", 1),
        ("mu = x.mean()\nz = y - mu", 0),
        ("mu = x.mean()\nsd = x.std()\nt = mu / sd", 0),
        ("mu = x.mean()\nz = mu * 252", 0),
        ("mu = x.mean()\nz = mu * len(x)", 0),
        ("mu = x.mean()\nprint(x - mu)", 0),
        ("mu = x.mean()\nmu = 0.0\nz = x - mu", 0),
        ("mu = x.mean()\nfor mu in mus:\n    z = x - mu", 0),
        ("mu = x.mean()\nz = x - other", 0),
        ("mu = x.mean()\nz = df.apply(lambda mu: x - mu)", 0),
        ("mu = x.mean()\ndef f(x):\n    return x - mu", 0),
        ("z = x - mu\nmu = x.mean()", 0),
        ("mu = x.expanding().mean()\nz = x - mu", 0),
        ("mu = x.rolling(20).mean()\nz = x - mu", 0),
        ("mu = x.shift(1).mean()\nz = x - mu", 0),
        ("mu = x.iloc[:split].mean()\nz = x - mu", 0),
        ("mu = x[:split].mean()\nz = x - mu", 0),
        ("mu = x_train.mean()\nz = x_train - mu", 0),
        ("mu = train['c'].mean()\nz = train['c'] - mu", 0),
        ("mu = df.groupby('date')['x'].mean()\nz = df['x'] - mu", 0),
        ("mu = x.mean(axis=1)\nz = x.sub(mu, axis=0)", 0),
        ("mu = x.mean()\nz = x.groupby('date').transform(lambda s: s - mu)", 0),
        ("c = x - x.mean()\nz = c / x.std()", 1),
        # the series has to appear itself: a t-statistic only uses its length or its count
        ("t = x.mean() / x.std() * np.sqrt(len(x))", 0),
        ("t = x.mean() / (x.std() / np.sqrt(len(x)))", 0),
        ("t = np.sqrt(len(x)) * x.mean() / x.std()", 0),
        ("t = x.mean() / (x.std() / np.sqrt(x.count()))", 0),
        ("t = x.mean() / x.std() * np.sqrt(x.shape[0])", 0),
        ("t = x.mean() / x.std() * np.sqrt(x.size)", 0),
        ("t = (x.mean() - y.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y))", 0),
        ("z = x.mean() + x.sum()", 0),
        ("z = x.mean() * x.index.year", 0),
        ("z = x.mean() - x", 1),
        ("z = x.mean() * x.shift(1)", 1),
        ("z = x.mean() + len(x) * x", 1),
        # whole-series scaling and binning
        ("from sklearn.preprocessing import scale\nz = scale(x)", 1),
        ("from sklearn import preprocessing\nz = preprocessing.scale(x)", 1),
        ("from sklearn.preprocessing import minmax_scale\nz = minmax_scale(x)", 1),
        ("from sklearn.preprocessing import robust_scale\nz = robust_scale(x)", 1),
        ("z = scale(x)", 1),
        ("from mylib import scale\nz = scale(x)", 0),
        ("def scale(x):\n    return x\nz = scale(x)", 0),
        ("from sklearn.preprocessing import scale\nz = scale(x, axis=1)", 0),
        (
            "from sklearn.preprocessing import scale\nz = df.groupby('date')['x'].transform(scale)",
            0,
        ),
        ("z = pd.qcut(x, 5)", 1),
        ("z = pd.qcut(df['x'], q=[0, 0.5, 1])", 1),
        ("from pandas import qcut\nz = qcut(x, 10)", 1),
        ("z = pd.qcut(x.rank(method='first'), 5)", 1),
        ("z = df.groupby('date')['x'].transform(lambda s: pd.qcut(s, 5))", 0),
        ("z = pd.cut(x, [0, 1, 2])", 0),
        ("print(pd.qcut(x, 5))", 0),
        ("z = x.clip(x.quantile(0.01), x.quantile(0.99))", 1),
        ("z = x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))", 1),
        (
            (
                "z = df['c'].clip(df['c'].mean() - 3 * df['c'].std(), "
                "df['c'].mean() + 3 * df['c'].std())"
            ),
            1,
        ),
        ("z = np.clip(x, x.quantile(0.01), x.quantile(0.99))", 1),
        ("z = x.between(x.quantile(0.05), x.quantile(0.95))", 1),
        ("z = x.clip(-1, 1)", 0),
        ("z = x.clip(lower=0)", 0),
        ("z = x.clip(y.quantile(0.01), y.quantile(0.99))", 0),
        ("z = x.clip(x.expanding().quantile(0.01), x.expanding().quantile(0.99))", 0),
    ],
    "PP007": [
        ("d = pd.merge_asof(a, b, on='t', direction='forward')", 1),
        ("d = pd.merge_asof(a, b, on='t', direction='nearest')", 1),
        ("d = pd.merge_asof(a, b, on='t', direction='backward')", 0),
        ("d = pd.merge_asof(a, b, on='t')", 0),
        ("d = pd.merge_asof(a, b, on='t', direction=dirn)", 0),
        ("from pandas import merge_asof\nd = merge_asof(a, b, on='t', direction='forward')", 1),
        ("d = a.join_asof(b, on='t', strategy='forward')", 1),
        ("d = a.join_asof(b, on='t', strategy='backward')", 0),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='period_end')", 1),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='fiscal_date')", 1),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='report_date')", 1),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='quarter_end')", 1),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='FY')", 1),
        ("d = pd.merge_asof(a, b, left_on='d', right_on='filing_date')", 0),
        ("d = pd.merge_asof(a, b, left_on='period_end', right_on='accepted')", 0),
        ("d = pd.merge_asof(a, b, on=['sym', 'period_end'])", 1),
        ("d = a.merge(b, left_on='date', right_on='period_end')", 1),
        ("d = pd.merge(a, b, on='period_end')", 1),
        ("d = a.merge(b, left_on='date', right_on='filed_at')", 0),
        ("d = a.merge(b, on='date')", 0),
        ("d = a.merge(b, on=key)", 0),
        ("d = pd.merge_asof(a, b, on='period_end', direction='forward')", 1),
    ],
    "PP008": [
        ("bars = t.resample('5min').last()\nr = bars.pct_change()", 1),
        ("bars = t['p'].resample('1h').ohlc()\nr = bars['close']", 1),
        ("bars = t.resample('1D').agg({'p': 'last'})\nr = bars.p", 1),
        ("r = t.resample('5min').last().pct_change()", 1),
        ("feat['bar'] = t.resample('5min').last()", 1),
        (
            (
                "bars = t.resample('5min', label='right', closed='right').last()\n"
                "r = bars.pct_change()"
            ),
            0,
        ),
        ("bars = t.resample('5min', label='right').last()\nr = bars.pct_change()", 0),
        ("bars = t.resample('5min', closed='right').last()\nr = bars.pct_change()", 0),
        ("bars = t.resample('5min').first()\nr = bars.pct_change()", 0),
        ("bars = t.resample('5min').last()", 0),
        ("bars = t.resample('5min').last()\nprint(1)", 0),
        ("def f(t):\n    return t.resample('5min').last()", 0),
        ("t.resample('5min').last()", 0),
        ("bars = t.resample('M').last()\nr = bars.pct_change()", 0),
        ("bars = t.resample('ME').last()\nr = bars.pct_change()", 0),
        ("bars = t.resample('W-FRI').last()\nr = bars.pct_change()", 0),
        ("bars = t.resample('MS').last()\nr = bars.pct_change()", 1),
        ("bars = t.resample('ME', label='left').last()\nr = bars.pct_change()", 1),
        ("bars = t.resample(freq).last()\nr = bars.pct_change()", 1),
        ("def f(t):\n    bars = t.resample('5min').last()\n    return bars.pct_change()", 1),
        (
            (
                "def f(t):\n    bars = t.resample('5min').last()\n    return 1\n\n"
                "def g(bars):\n    return bars.pct_change()"
            ),
            0,
        ),
    ],
    "PP009": [
        ("for i in range(n):\n    out[i] = x[i + 1] - x[i]", 1),
        ("for i in range(n):\n    out[i] = x[1 + i]", 1),
        ("for i in range(n):\n    out[i] = x.iloc[i + 2]", 1),
        ("for i in range(n):\n    out[i] = x.loc[i + 1]", 1),
        ("for i in range(n):\n    df.iloc[i, 3] = df.iloc[i + 1, 2]", 1),
        ("for i in range(n):\n    df.loc[i, 'a'] = df['b'][i + 1]", 1),
        ("for i in range(n):\n    out[i] += x[i + 1]", 1),
        ("for i in range(n):\n    for j in range(m):\n        out[i, j] = x[i + 1, j]", 1),
        ("for i in range(n):\n    if i > 0:\n        out[i] = f(x[i + 1])", 1),
        ("for i in range(n):\n    out[i] = x[i - 1]", 0),
        ("for i in range(n):\n    out[i] = x[i]", 0),
        ("for i in range(n):\n    out[i] = x[i + h]", 0),
        ("for i in range(n):\n    out[i + 1] = x[i + 1]", 0),
        ("for i in range(n):\n    out[i + 2] = x[i + 1]", 0),
        ("for i in range(n):\n    out[i + 1] = x[i + 2]", 1),
        ("for i in range(n):\n    total = x[i + 1]", 0),
        ("for i in range(n):\n    out[i] = x[i - 5:i + 1].mean()", 0),
        ("for i in x:\n    out[i] = x[i + 1]", 0),
        ("for i, v in enumerate(x):\n    out[i] = x[i + 1]", 0),
        ("for i in range(n):\n    for i in other:\n        out[i] = x[i + 1]", 0),
        ("for i in range(n):\n    def g(i):\n        out[i] = x[i + 1]", 0),
        ("for k in range(n):\n    out[k] = x[k + 1]", 1),
        ("for i in range(n):\n    out[i] = x[i + 1.0]", 0),
        ("out = [x[i + 1] - x[i] for i in range(n)]", 1),
        ("out = [x.iloc[i + 2] for i in range(n)]", 1),
        ("out = {i: x[i + 1] for i in range(n)}", 1),
        ("out = {x[i + 1]: i for i in range(n)}", 1),
        ("out = {x[i + 1] for i in range(n)}", 0),  # a set has no row order
        ("out = sum(x[i + 1] for i in range(n))", 0),  # reduced to one value, not one per row
        ("ok = all(x[i] <= x[i + 1] for i in range(n - 1))", 0),
        ("out = [[x[i + 1, j] for j in range(m)] for i in range(n)]", 1),
        ("out = [x[i - 1] for i in range(1, n)]", 0),
        ("out = [x[i] for i in range(n)]", 0),
        ("out = [x[i + h] for i in range(n)]", 0),
        ("out = [x[i - 5 : i + 1].mean() for i in range(n)]", 0),
        ("out = [x[i + 1] for i in items]", 0),
        ("out = [x[j + 1] for i in range(n)]", 0),
        ("out = [x[i + 1, j] for j in range(m)]", 0),
        # ``append`` builds one element per row: element i must not be made from row i + k
        ("out = []\nfor i in range(n):\n    out.append(x[i + 1])", 1),
        ("out = []\nfor i in range(n):\n    out.append(x.iloc[i + 1] - x.iloc[i])", 1),
        ("out = []\nfor i in range(n):\n    out.append(f(x.iloc[i + 2]))", 1),
        ("out = []\nfor i in range(n):\n    out.append(x[i - 1])", 0),
        ("out = []\nfor i in range(n):\n    out.append(x[i])", 0),
        ("out = []\nfor i in range(n):\n    out.append(x[i + h])", 0),
        ("out = []\nfor i in range(n):\n    out.append(x[i + 1.0])", 0),
        ("out = []\nfor i in range(n):\n    if x[i + 1] > x[i]:\n        out.append(1)", 0),
        ("out = []\nfor i in items:\n    out.append(x[i + 1])", 0),
        ("out = []\nfor i in range(n):\n    out.append(x[i - 5 : i + 1].mean())", 0),
        # a window that starts at row i and runs on into later rows: medium confidence
        ("for i in range(n):\n    out[i] = x[i : i + 5].mean()", 1),
        ("for i in range(n):\n    out[i] = x.iloc[i : i + 5].mean()", 1),
        ("for i in range(n):\n    out[i] = x.loc[i : i + 5].mean()", 1),
        ("for i in range(n):\n    out[i] = x[i : i + 2].sum()", 1),
        ("for i in range(n):\n    out[i] = x[: i + 3].mean()", 1),
        ("for i in range(n):\n    out[i] = x.iloc[i : i + 3, 0].mean()", 1),
        ("for i in range(n):\n    window = x[i : i + 5]", 1),
        ("for i in range(n):\n    total += x[i : i + 3].sum()", 1),
        ("for i in range(n):\n    out.append(x[i : i + 3].mean())", 1),
        ("out = [x[i : i + 3].mean() for i in range(n)]", 1),
        ("out = {i: x[i : i + 3].sum() for i in range(n)}", 1),
        ("for i in range(n):\n    out[i] = x[i : i + 1]", 0),
        ("for i in range(n):\n    out[i] = x.iloc[i : i + 1]", 0),
        ("for i in range(n):\n    out[i] = x.loc[i:i].sum()", 0),
        ("for i in range(n):\n    out[i] = x.loc[i : i + 1].sum()", 1),
        ("for i in range(n):\n    out[i] = x[i - 5 : i].mean()", 0),
        ("for i in range(n):\n    out[i] = x[: i + 1].mean()", 0),
        ("for i in range(n):\n    out[i] = x[:i].mean()", 0),
        ("for i in range(n):\n    out[i] = x[i - 2 : i + 1].mean()", 0),
        ("for i in range(n):\n    out[i] = x[i : i + h].mean()", 0),
        ("for i in range(n):\n    out[i] = x[i:].mean()", 0),
        ("for i in range(n):\n    out[i + 2] = x[i : i + 3].mean()", 0),
        ("for i in range(n):\n    out[i] = x.iloc[:, i : i + 3].mean()", 0),
        ("for i in items:\n    out[i] = x[i : i + 3].mean()", 0),
        ("out = [x[i - 2 : i + 1].mean() for i in range(n)]", 0),
        ("out = [x[j : j + 3].mean() for i in range(n)]", 0),
        # a loop over blocks (step other than 1) walks over batches, not over rows
        ("for i in range(0, n, 32):\n    batch = x[i : i + 32]", 0),
        ("for i in range(0, n, step):\n    out[i] = x[i : i + 5].mean()", 0),
        ("out = [x[i : i + 32] for i in range(0, n, 32)]", 0),
        ("for i in range(0, n, 1):\n    out[i] = x[i : i + 5].mean()", 1),
        ("for i in range(0, n, 2):\n    out[i] = x[i + 1]", 1),
        ("out = []\nfor i in range(0, n, 2):\n    out.append(x[i + 1])", 1),
        # an element read beats a window read in the same statement: one finding either way
        ("for i in range(n):\n    out[i] = x[i + 1] + x[i : i + 3].mean()", 1),
        # label functions look ahead on purpose
        (
            (
                "def make_label(x):\n    out = []\n    for i in range(n):\n"
                "        out.append(x[i + 1])\n    return out"
            ),
            0,
        ),
        ("def make_label(x):\n    out = [x[i + 1] for i in range(n)]\n    return out", 0),
        ("def features(x):\n    out = [x[i + 1] for i in range(n)]\n    return out", 1),
        (
            (
                "def build(x):  # peekproof: label\n"
                "    out = [x[i : i + 3].mean() for i in range(n)]\n    return out"
            ),
            0,
        ),
        ("def target_position(x):\n    out = [x[i + 1] for i in range(n)]\n    return out", 1),
    ],
    "PP010": [
        ("p = signal * returns", 1),
        ("p = returns * signal", 1),
        ("p = df['position'] * df['ret']", 1),
        ("p = df.position * df.ret", 1),
        ("p = positions * rets", 1),
        ("p = weights * df.pct_change()", 1),
        ("p = side * log_returns", 1),
        ("p = np.sign(pred) * returns", 1),
        ("p = np.sign(pred) * df['ret']", 1),
        ("p = signal * (returns - cost)", 1),
        ("p = (signal * returns).sum()", 1),
        ("p = pos * ret_1d", 1),
        ("p = exposure * daily_return", 1),
        ("p = signal.shift(1) * returns", 0),
        ("p = returns * signal.shift(1)", 0),
        ("p = df['position'].shift(1) * df['ret']", 0),
        ("p = df.shift(1)['position'] * df['ret']", 0),
        ("p = np.sign(pred).shift(1) * returns", 0),
        ("position = signal.shift(1)\np = position * returns", 0),
        ("df['position'] = df['signal'].shift(1)\np = df['position'] * df['ret']", 0),
        ("p = signal * returns.shift(-1)", 0),
        ("p = signal * fwd_ret", 0),
        ("p = signal * next_return", 0),
        ("p = signal * forward_returns", 0),
        ("p = signal * volume", 0),
        ("p = 0.5 * returns", 0),
        ("p = returns * 100", 0),
        ("p = weights * prices", 0),
        ("p = retail * position", 0),
        ("p = interpret * signal", 0),
        ("p = positive * returns", 0),
        ("p = weighted * returns", 0),
        ("p = signal + returns", 0),
        ("p = signal @ returns", 0),
        ("p = signal() * returns", 0),
        # the difference of prices (or of log prices) is a return, although no name says so
        ("p = signal * np.log(close).diff()", 1),
        ("p = positions * df['close'].diff()", 1),
        ("p = signal * prices.diff()", 1),
        ("p = np.log(prices).diff() * weights", 1),
        ("p = signal * np.diff(np.log(prices))", 1),
        ("p = signal * df['Adj Close'].diff()", 1),
        ("p = signal * px.diff(1)", 1),
        ("p = signal.shift(1) * np.log(close).diff()", 0),
        ("p = signal * np.log(close).diff().shift(-1)", 0),
        ("p = signal * close.diff(-1)", 0),
        ("p = signal * np.log(close).diff(periods=-1)", 0),
        ("p = signal * volume.diff()", 0),
        ("p = signal * x.diff()", 0),
        ("p = signal * catalog.diff()", 0),
        ("p = signal * close.mean()", 0),
        # a name that holds forward-shifted returns is label code; so is a shifted operand
        ("returns = prices.pct_change().shift(-1)\np = positions * returns", 0),
        ("p = positions * returns.shift(-1)", 0),
        ("p = positions * df['ret'].shift(periods=-1)", 0),
        ("p = positions * (returns.shift(-1) - cost)", 0),
        ("p = positions * prices.pct_change(-1)", 0),
        ("p = positions * prices.pct_change(periods=-5)", 0),
        ("p = positions * prices.pct_change(1)", 1),
        ("returns = prices.pct_change(-1)\np = positions * returns", 0),
        ("returns = prices.pct_change()\np = positions * returns", 1),
        ("returns = prices.pct_change().shift(1)\np = positions * returns", 0),
        # forecasts and expected returns are not realised returns
        ("p = weights * expected_returns", 0),
        ("p = weights * exp_ret", 0),
        ("p = weights * df['expected_return']", 0),
        ("p = signal * forecast_returns", 0),
        ("p = alpha_signal * returns", 0),
        ("p = mu_weights * returns", 0),
        ("p = signal * (returns - alpha)", 0),
        ("p = exposure * df['returns']", 1),
        ("p = expiry_weight * returns", 1),
    ],
    "PP011": [
        ("df['f'] = df['target'].rolling(3).mean()", 1),
        ("df['f'] = df['y'] * 2", 1),
        ("df['f'] = df['label'] + 1", 1),
        ("df['f'] = df['y_true'] - df['pred']", 1),
        ("df['f'] = df['fwd_ret'] > 0", 1),
        ("df['f'] = df['forward_return']", 1),
        ("df['f'] = df['fwd_ret_5d']", 1),
        ("df['f'] = df['next_target']", 1),
        ("df['f'] = df['binary_label']", 1),
        ("X = df[['a', 'target']]", 1),
        ("X = df.loc[:, ['a', 'label']]", 1),
        ("feat = df.loc[mask, 'target']", 1),
        ("df2 = df.assign(f=lambda d: d['target'] * 2)", 1),
        ("df['f'] += df['target']", 1),
        ("a, b = df['x'], df['target']", 1),
        ("y = df['target']", 0),
        ("y_train = df['target'].iloc[:n]", 0),
        ("labels = df['label']", 0),
        ("df['target'] = df['fwd_ret'] > 0", 0),
        ("df['label'] = (df['fwd_ret'] > 0).astype(int)", 0),
        ("df['f'] = df['target'].shift(1)", 0),
        ("df['f'] = df['target'].shift(1).rolling(3).mean()", 0),
        ("df['f'] = df['target'].rolling(3).mean().shift(2)", 0),
        ("df['f'] = df['target'].shift(-1)", 1),
        ("X = df.drop(columns=['target'])", 0),
        ("X = df.drop('target', axis=1)", 0),
        ("df['pnl'] = df['position'] * df['fwd_ret']", 0),
        ("acc = (pred == df['target']).mean()", 0),
        ("df['f'] = df['city']", 0),
        ("df['f'] = df['entry']", 0),
        ("df['f'] = df['target_lag']", 0),
        ("df['f'] = df.target", 0),
        ("df['f'] = obj.label", 0),
        ("print(df['target'])", 0),
        ("df['f'] = df[col]", 0),
        # a merge suffix is no label, and neither is a year column called "Y"
        ("df['f'] = df['close_y'] * 2", 0),
        ("df['f'] = df['volume_y'] + df['volume_x']", 0),
        ("df['f'] = df['Y'] * 2", 0),
        ("df['f'] = df['target_y']", 0),
        ("df['f'] = df['y_pred']", 0),
        ("df['f'] = df['y']", 1),
        ("df['f'] = df['y_true']", 1),
        ("df['f'] = df['Y_true']", 1),
        ("df['f'] = df['x_target']", 1),
        ("df['f'] = df['next_label']", 1),
        ("df['f'] = df['fwd_y_true']", 1),
        ("df['f'] = df['Target']", 1),
        ("df['f'] = df['LABEL']", 1),
        ("df['f'] = df['forward_return_5']", 1),
        ("df['f'] = df['label_encoded']", 0),
        # label functions may read the target: building labels is their job
        ("def make_label(df):\n    df['f'] = df['target'] * 2\n    return df", 0),
        ("def get_y(df):\n    df['f'] = df['target'] * 2\n    return df", 0),
        ("def features(df):\n    df['f'] = df['target'] * 2\n    return df", 1),
        ("def build(df):  # peekproof: label\n    df['f'] = df['target'] * 2\n    return df", 0),
        ("def target_encoding(df):\n    df['f'] = df['target'] * 2\n    return df", 1),
        ("def forward_fill_features(df):\n    df['f'] = df['target'] * 2\n    return df", 1),
        (
            "def make_label(df):\n    df['f'] = 1\n\ndef features(df):\n    df['g'] = df['target']",
            1,
        ),
    ],
    "PP012": [
        ("t = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')", 1),
        ("t = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')", 1),
        ("t = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')", 1),
        ("t = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')", 1),
        ("t = pd.read_html('https://en.wikipedia.org/wiki/DAX')", 1),
        ("URL = 'https://en.wikipedia.org/wiki/DAX'\nt = pd.read_html(URL)", 1),
        ("t = pd.read_html(io='https://en.wikipedia.org/wiki/DAX')", 1),
        ("t = pd.read_html('https://example.com/table')", 0),
        ("t = pd.read_html('https://en.wikipedia.org/wiki/List_of_largest_banks')", 0),
        ("t = pd.read_html(url)", 0),
        ("sp500_tickers = ['AAPL']", 1),
        ("SP500_TICKERS = ['AAPL']", 1),
        ("sp500tickers = []", 1),
        ("nasdaq100_symbols = []", 1),
        ("ftse100_constituents = []", 1),
        ("dax_tickers = []", 1),
        ("dax40_symbols = []", 1),
        ("def get_sp500_tickers():\n    return []", 1),
        ("sp500_tickers = []\nprint(sp500_tickers)\nfor t in sp500_tickers:\n    pass", 1),
        ("sp500_tickers = []\nnasdaq100_tickers = []", 2),
        ("x = cfg.sp500_tickers", 1),
        ("def f(sp500_tickers):\n    return 1", 1),
        ("u = yf.Tickers('AAPL MSFT')", 1),
        ("import yfinance\nu = yfinance.Tickers(['AAPL', 'MSFT'])", 1),
        ("from yfinance import Tickers\nu = Tickers('AAPL MSFT')", 1),
        ("u = yf.Ticker('AAPL')", 0),
        ("u = Tickers('AAPL')", 0),
        ("sp500_constituents_asof = []", 0),
        ("def sp500_constituents_history(d):\n    return []", 0),
        ("members = []", 0),
        ("sp500 = []", 0),
        ("sp500_returns = []", 0),
    ],
    "PP013": [
        ("p = yf.download('A')\ni = yf.Ticker('A').info", 1),
        ("p = yf.download('A')\nf = yf.Ticker('A').financials", 1),
        ("p = yf.download('A')\nf = yf.Ticker('A').balance_sheet\ng = yf.Ticker('A').cashflow", 2),
        ("p = yf.download('A')\nf = yf.Ticker('A').income_stmt", 1),
        ("p = yf.download('A')\nt = yf.Ticker('A')\nf = t.financials", 1),
        ("t = yf.Ticker('A')\np = t.history(period='max')\ni = t.info", 1),
        ("t = yf.Ticker('A')\ni = t.info", 0),
        ("p = yf.download('A')\ni = other.info", 0),
        ("p = yf.download('A')\nt = something()\ni = t.info", 0),
        ("i = yf.Ticker('A').info", 0),
        ("p = yf.download('A')", 0),
        ("t = yf.Ticker('A')\nh = t.history()", 0),
        ("import yfinance\np = yfinance.download('A')\ni = yfinance.Ticker('A').info", 1),
        ("from yfinance import Ticker, download\np = download('A')\ni = Ticker('A').info", 1),
        ("p = yf.download('A')\ni = yf.Ticker('A').info['x']\nj = yf.Ticker('A').info['y']", 2),
        ("p = yf.download('A')\ni = (yf.Ticker('A').info['x'], yf.Ticker('B').info['y'])", 1),
    ],
}

SNIPPET_PARAMS = [
    pytest.param(rule_id, code, expected, id=f"{rule_id}-{index:02d}")
    for rule_id, cases in CASES.items()
    for index, (code, expected) in enumerate(cases)
]


@pytest.mark.parametrize(("rule_id", "code", "expected"), SNIPPET_PARAMS)
def test_snippet(rule_id: str, code: str, expected: int) -> None:
    found = [f for f in lint_snippet(code) if f.rule_id == rule_id]
    assert len(found) == expected, f"{rule_id} expected {expected}, got {found!r} for:\n{code}"


def test_every_rule_has_snippet_cases_that_fire_and_that_stay_silent() -> None:
    for rule_id in RULE_IDS:
        counts = [expected for code, expected in CASES[rule_id]]
        assert any(c > 0 for c in counts), rule_id
        assert any(c == 0 for c in counts), rule_id


# --------------------------------------------------------------------------- #
# Confidence per trigger (PP003, PP005 and PP007 have two levels)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rule_id", "code", "confidence"),
    [
        ("PP001", "z = x.shift(-1)", "high"),
        ("PP001", "z = x.pct_change(-5)", "high"),
        ("PP001", "z = np.roll(x, -1)", "high"),
        ("PP001", "z = x.shift(-HORIZON)", "medium"),
        ("PP001", "z = x.shift(periods=-n)", "medium"),
        ("PP001", "z = x.pct_change(-n)", "medium"),
        ("PP001", "z = np.roll(x, -n)", "medium"),
        ("PP001", "fill_price = df['open'].shift(-1)", "medium"),
        ("PP001", "entry = open.shift(-1)", "medium"),
        ("PP001", "exit_px = open.shift(-1)", "medium"),
        ("PP001", "df['execution_price'] = df['open'].shift(-1)", "medium"),
        ("PP001", "df = df.assign(fill=lambda d: d['open'].shift(-1))", "medium"),
        ("PP001", "entry_signal = df['open'].shift(-1)", "high"),
        ("PP001", "fillprice = df['open'].shift(-1)", "medium"),
        ("PP001", "fillPx = df['open'].shift(-1)", "medium"),
        ("PP001", "FILL_PRICE = df['open'].shift(-1)", "medium"),
        ("PP001", "fill_price_next = df['open'].shift(-1)", "high"),
        ("PP001", "target_position = signal.shift(-1)", "high"),
        ("PP001", "df['fill_price'] = df['open'].shift(-HORIZON)", "medium"),
        ("PP002", "y = x.bfill()", "high"),
        ("PP002", "y = x.interpolate(limit_direction='both')", "high"),
        ("PP002", "y = x.interpolate()", "medium"),
        ("PP002", "y = x.interpolate(method='time')", "medium"),
        ("PP002", "y = pl.col('a').interpolate()", "medium"),
        ("PP002", "y = x.reindex(idx, method='bfill')", "high"),
        ("PP002", "y = x.reindex(idx, method='nearest')", "high"),
        ("PP002", "y = x.asfreq('D', method='backfill')", "high"),
        ("PP002", "y = x.pct_change(fill_method='bfill')", "high"),
        ("PP003", "y = x.rolling(5, center=True).mean()", "high"),
        ("PP003", "y = savgol_filter(x, 11, 3)", "high"),
        ("PP003", "y = ndi.gaussian_filter(x, 2)", "high"),
        ("PP003", "y = filtfilt(b, a, x)", "high"),
        ("PP003", "y = hpfilter(x)", "high"),
        ("PP003", "y = seasonal_decompose(x)", "high"),
        ("PP003", "y = np.gradient(x)", "high"),
        ("PP003", "y = FixedForwardWindowIndexer(window_size=3)", "high"),
        ("PP003", "y = np.convolve(x, k, mode='same')", "medium"),
        ("PP003", "y = np.convolve(x, k)", "medium"),
        ("PP003", "y = np.convolve(x, k)[2:]", "medium"),
        ("PP005", "a = train_test_split(X, y)", "high"),
        ("PP005", "a = KFold(5, shuffle=True)", "high"),
        ("PP005", "a = ShuffleSplit(3)", "high"),
        ("PP005", "a = StratifiedKFold(5)", "medium"),
        ("PP005", "a = cross_val_score(m, X, y)", "medium"),
        ("PP005", "a = GridSearchCV(m, grid)", "medium"),
        ("PP005", "a = GridSearchCV(m, grid, cv=5)", "medium"),
        ("PP005", "a = RandomizedSearchCV(m, grid, cv=5)", "medium"),
        ("PP006", "z = (x - x.mean()) / x.std()", "medium"),
        ("PP006", "mu = x.mean()\nz = x - mu", "medium"),
        ("PP006", "z = scale(x)", "medium"),
        ("PP006", "z = pd.qcut(x, 5)", "medium"),
        ("PP006", "z = x.clip(x.quantile(0.01), x.quantile(0.99))", "medium"),
        ("PP007", "d = pd.merge_asof(a, b, on='t', direction='forward')", "high"),
        ("PP007", "d = pd.merge_asof(a, b, on='t', direction='nearest')", "high"),
        ("PP007", "d = a.merge(b, left_on='d', right_on='period_end')", "low"),
        ("PP007", "d = pd.merge_asof(a, b, on='period_end')", "low"),
        ("PP009", "for i in range(n):\n    out[i] = x[i + 1]", "high"),
        ("PP009", "for i in range(n):\n    out.append(x[i + 1])", "high"),
        ("PP009", "out = [x[i + 1] for i in range(n)]", "high"),
        ("PP009", "for i in range(n):\n    out[i] = x[i : i + 5].mean()", "medium"),
        ("PP009", "out = [x[i : i + 5].mean() for i in range(n)]", "medium"),
        ("PP009", "for i in range(n):\n    out[i] = x[i + 1] + x[i : i + 5].mean()", "high"),
    ],
)
def test_confidence_follows_the_trigger(rule_id: str, code: str, confidence: str) -> None:
    (finding,) = [f for f in lint_snippet(code) if f.rule_id == rule_id]
    assert finding.confidence == confidence


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("fy", 1),
        ("FY", 1),
        ("fy2023", 1),
        ("fy_end", 1),
        ("Fiscal_Year", 1),
        ("verify_id", 0),
        ("classify", 0),
        ("notify_date", 0),
    ],
)
def test_pp007_period_key_fy_needs_word_boundaries(key: str, expected: int) -> None:
    findings = lint_snippet(f"d = a.merge(b, on={key!r})")
    assert len([f for f in findings if f.rule_id == "PP007"]) == expected


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("M", 0),
        ("ME", 0),
        ("2ME", 0),
        ("BME", 0),
        ("Q", 0),
        ("QE-DEC", 0),
        ("A", 0),
        ("YE", 0),
        ("W", 0),
        ("w-sun", 0),
        ("W-FRI", 0),
        ("MS", 1),
        ("QS", 1),
        ("YS", 1),
        ("D", 1),
        ("B", 1),
        ("h", 1),
        ("1H", 1),
        ("5min", 1),
        ("min", 1),
    ],
)
def test_pp008_end_labeled_frequencies_are_exempt(rule: str, expected: int) -> None:
    source = f"bars = t.resample({rule!r}).last()\nr = bars.pct_change()"
    assert len([f for f in lint_snippet(source) if f.rule_id == "PP008"]) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("from sklearn.model_selection import train_test_split\na = train_test_split(X, y)", 1),
        ("from sklearn.model_selection import *\na = train_test_split(X, y)", 1),
        ("a = train_test_split(X, y)", 1),  # no import in this cell: assume scikit-learn
        ("from .splits import train_test_split\na = train_test_split(X, y)", 0),
        ("from ..utils import train_test_split\na = train_test_split(X, y)", 0),
        ("from myproject.splits import train_test_split\na = train_test_split(X, y)", 0),
        ("import myproject.splits as s\na = s.train_test_split(X, y)", 0),
        (
            "class Splitter:\n    def train_test_split(self, X):\n"
            "        return self.train_test_split(X)",
            0,
        ),
    ],
)
def test_pp005_only_fires_for_scikit_learn_functions(code: str, expected: int) -> None:
    found = [f for f in lint_snippet(code) if f.rule_id == "PP005"]
    assert len(found) == expected


def test_a_forward_asof_join_on_period_keys_is_reported_once_at_high() -> None:
    source = "d = pd.merge_asof(a, b, on='period_end', direction='forward')"
    (finding,) = [f for f in lint_snippet(source) if f.rule_id == "PP007"]
    assert finding.confidence == "high"


# --------------------------------------------------------------------------- #
# Positions and messages
# --------------------------------------------------------------------------- #


def test_finding_for_a_method_call_points_at_the_method_name() -> None:
    (finding,) = lint_snippet("z = (\n    x\n    .shift(-1)\n)\n")
    assert (finding.line, finding.col) == (3, 6)


def test_two_methods_in_one_chain_are_reported_on_their_own_lines() -> None:
    source = "z = (\n    x\n    .shift(-1)\n    .rolling(5, center=True)\n    .mean()\n)\n"
    found = [(f.rule_id, f.line) for f in lint_snippet(source)]
    assert found == [("PP001", 3), ("PP003", 4)]


def test_finding_for_a_function_call_points_at_the_start_of_the_call() -> None:
    (finding,) = lint_snippet("a = 1\nb = np.roll(x, -1)\n")
    assert (finding.line, finding.col) == (2, 5)


def test_columns_count_characters_not_bytes() -> None:
    source = 's = "héllo wörld"; z = x.shift(-1)\n'
    (finding,) = lint_snippet(source)
    assert finding.col == source.index("shift") + 1


def test_messages_name_what_was_found() -> None:
    assert "shift(-3)" in lint_snippet("z = x.shift(periods=-3)")[0].message
    assert "mean" in lint_snippet("z = (x - x.mean()) / x.std()")[0].message
    assert "period_end" in lint_snippet("d = a.merge(b, on='period_end')")[0].message
    assert "sp500_tickers" in lint_snippet("sp500_tickers = []")[0].message


def test_pp006_reports_one_finding_per_assignment() -> None:
    findings = lint_snippet("z = (x - x.mean()) / x.std()\nw = (y - y.mean()) / y.std()")
    assert [f.line for f in findings] == [1, 2]


def test_pp012_reports_each_identifier_once() -> None:
    source = "sp500_tickers = []\nfor t in sp500_tickers:\n    print(sp500_tickers)"
    assert len(lint_snippet(source)) == 1


# --------------------------------------------------------------------------- #
# The linter agrees with the oracle on the functions that docs/DESIGN.md section 8 lists
# --------------------------------------------------------------------------- #

WEEKLY = "w = df.close.resample('W').last().to_frame()\n    "
DESIGN_LEAKY_FUNCTIONS = [
    ("PP001", "def f(df):\n    return df.shift(-1)"),
    ("PP001", "def f(df):\n    return df.shift(-3)"),
    ("PP001", "def f(df):\n    return df.shift(-1).dropna()"),
    ("PP001", "def f(df):\n    return df.pct_change(-2)"),
    ("PP002", "def f(df):\n    return df.bfill()"),
    ("PP003", "def f(df):\n    return df.rolling(5, center=True).mean()"),
    ("PP006", "def f(df):\n    return (df - df.mean()) / df.std()"),
    ("PP006", "def f(df):\n    return (df - df.min()) / (df.max() - df.min())"),
    ("PP006", "def f(df):\n    return df.rank(pct=True)"),
    (
        "PP009",
        "def f(df):\n    return pd.Series([df.close.iloc[i + 1] - df.close.iloc[i] "
        "for i in range(len(df))])",
    ),
    (
        "PP007",
        "def f(df):\n    "
        + WEEKLY
        + "return pd.merge_asof(df, w, left_index=True, right_index=True, direction='forward')",
    ),
]
DESIGN_CLEAN_FUNCTIONS = [
    "def f(df):\n    return df.rolling(5).mean()",
    "def f(df):\n    return df.ewm(span=10).mean()",
    "def f(df):\n    return (df - df.expanding().mean()) / df.expanding().std()",
    "def f(df):\n    return df.pct_change().shift(1)",
    "def f(df):\n    "
    + WEEKLY
    + "return pd.merge_asof(df, w, left_index=True, right_index=True, direction='backward')",
    "def f(df):\n    return df[['close']].rolling(5).mean()",
    "def f(df):\n    return df.rolling(20).mean().dropna()",
    "def f(df):\n    return df.rank(axis=1)",
]


@pytest.mark.parametrize(("rule_id", "code"), DESIGN_LEAKY_FUNCTIONS)
def test_functions_that_the_oracle_calls_leaky_are_flagged(rule_id: str, code: str) -> None:
    assert rule_id in {f.rule_id for f in lint_snippet(code)}


@pytest.mark.parametrize("code", DESIGN_CLEAN_FUNCTIONS)
def test_functions_that_the_oracle_calls_causal_are_not_flagged(code: str) -> None:
    assert lint_snippet(code) == []


# --------------------------------------------------------------------------- #
# Rules are plain objects that can be run on their own
# --------------------------------------------------------------------------- #


def test_negative_constants_count_like_a_unary_minus() -> None:
    # ``ast.parse`` produces UnaryOp(USub, Constant(1)) for ``-1``; a tree built by a
    # tool may hold Constant(-1) instead. Both must be recognised.
    ctx = LintContext.build("z = x.shift(1)\nz = np.roll(x, 2)\n")
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Constant) and node.value in {1, 2}:
            node.value = -node.value
    findings = RULES["PP001"].visit(ctx.tree, ctx)
    assert [f.line for f in findings] == [1, 2]


def test_a_rule_can_be_run_on_its_own() -> None:
    ctx = LintContext.build("z = x.shift(-1)\nz = x.bfill()\n")
    findings = RULES["PP001"].visit(ctx.tree, ctx)
    assert [f.rule_id for f in findings] == ["PP001"]
    assert findings[0].line == 1


def test_a_rule_instance_gives_the_same_result_for_repeated_calls() -> None:
    first = LintContext.build("z = x.shift(-1)\n")
    second = LintContext.build("z = x.shift(1)\nz = x.shift(-2)\n")
    rule = RULES["PP001"]
    assert len(rule.visit(first.tree, first)) == 1
    assert len(rule.visit(second.tree, second)) == 1
    assert len(rule.visit(first.tree, first)) == 1


def test_text_inside_strings_and_comments_is_never_matched() -> None:
    source = '"""x.shift(-1) and x.bfill()"""\n# z = x.shift(-1)\ns = "x.rolling(5, center=True)"\n'
    assert lint_snippet(source) == []


MODERN_SYNTAX = """
import asyncio

type Series[T] = list[T]


def generic[T: (int, float)](x: T, *rest: T, key=lambda v, w=2: v * w, **kw) -> T:
    global counter
    counter = 0
    mu = x.mean()
    match x:
        case [a, b, *others] if a > mu:
            z = a - mu
        case {"k": v, **extra}:
            z = v - mu
        case Point(x=px, y=py) | Other(px, py):
            z = px
        case _:
            z = f"{x!r:>{width}} {mu=}"
    sd = x.std()

    async def inner():
        async with ctx() as c:
            return [y async for y in aiter(c) if (w := y) > mu]

    for mu in range(3):
        del sd
    return (x - mu) / x.std() if (n := len(x)) else {k: v for k, v in kw.items()} | {1, 2, *rest}
"""


def test_modern_syntax_runs_through_every_rule_without_failing() -> None:
    findings = lint_source(MODERN_SYNTAX)
    assert [f for f in findings if f.rule_id == "PP000"] == []


def test_empty_source_has_no_findings() -> None:
    assert lint_source("") == []
    assert lint_source("\n\n# only a comment\n") == []


# --------------------------------------------------------------------------- #
# What each leaky fixture reports (so that a lost trigger shows up as a changed count)
# --------------------------------------------------------------------------- #

LEAKY_FIXTURE_CONFIDENCES = {
    "PP001": {"high": 7, "medium": 3},
    "PP002": {"high": 7, "medium": 3},
    "PP003": {"high": 8, "medium": 1},
    "PP004": {"medium": 2},
    "PP005": {"high": 3, "medium": 4},
    "PP006": {"medium": 8},
    "PP007": {"high": 1, "low": 1},
    "PP008": {"low": 1},
    "PP009": {"high": 3, "medium": 1},
    "PP010": {"medium": 3},
    "PP011": {"medium": 2},
    "PP012": {"medium": 2},
    "PP013": {"low": 1},
}


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_leaky_fixture_reports_every_trigger_at_the_expected_confidence(rule_id: str) -> None:
    _path, findings = lint_fixture("leaky", rule_id)
    assert dict(Counter(f.confidence for f in findings)) == LEAKY_FIXTURE_CONFIDENCES[rule_id]


# --------------------------------------------------------------------------- #
# Hints: a medium or low confidence finding asks the user to confirm it with the oracle
# --------------------------------------------------------------------------- #


def reported_findings() -> Iterator[Finding]:
    """Yield what the snippets that are meant to fire report, and what the leaky fixtures report."""
    for cases in CASES.values():
        for code, expected in cases:
            if expected:
                yield from lint_snippet(code)
    for rule_id in RULE_IDS:
        yield from lint_fixture("leaky", rule_id)[1]


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_the_default_hint_of_a_medium_or_low_rule_ends_with_the_confirmation(rule_id: str) -> None:
    rule = RULES[rule_id]
    if rule.confidence != "high":
        assert rule.hint.endswith(CONFIRM)


def test_every_medium_or_low_finding_ends_its_hint_with_the_confirmation() -> None:
    seen: Counter[str] = Counter()
    for finding in reported_findings():
        seen[finding.confidence] += 1
        if finding.confidence != "high":
            assert finding.hint.endswith(CONFIRM), (finding.rule_id, finding.message)
            assert len(finding.hint) > len(CONFIRM) + 20  # there is advice before the sentence
    assert seen["high"] > 20
    assert seen["medium"] > 50
    assert seen["low"] > 3


@pytest.mark.parametrize(
    ("rule_id", "code"),
    [
        ("PP001", "z = x.shift(-n)"),
        ("PP001", "fill_price = df['open'].shift(-1)"),
        ("PP002", "z = x.interpolate()"),
        ("PP003", "z = np.convolve(x, k, mode='same')"),
        ("PP005", "z = StratifiedKFold(5)"),
        ("PP005", "z = GridSearchCV(est, grid)"),
        ("PP007", "z = a.merge(b, on='period_end')"),
        ("PP009", "for i in range(n):\n    out[i] = x[i : i + 5].mean()"),
    ],
)
def test_a_lower_confidence_trigger_of_a_high_rule_has_its_own_hint(
    rule_id: str, code: str
) -> None:
    (finding,) = [f for f in lint_snippet(code) if f.rule_id == rule_id]
    assert finding.confidence != "high"
    assert finding.hint != RULES[rule_id].hint
    assert finding.hint.endswith(CONFIRM)


def test_a_high_confidence_finding_keeps_the_hint_of_its_rule() -> None:
    (finding,) = lint_snippet("z = x.shift(-1)")
    assert finding.hint == RULES["PP001"].hint


# --------------------------------------------------------------------------- #
# Label functions are exempt from PP001, PP009 and PP011 only
# --------------------------------------------------------------------------- #

LABEL_CODE = {
    "PP001": "return x.shift(-5)",
    "PP009": "out = [x[i + 1] for i in range(n)]\n    return out",
    "PP011": "df['f'] = df['target'] * 2\n    return df",
}


@pytest.mark.parametrize("rule_id", sorted(LABEL_CODE))
@pytest.mark.parametrize(
    ("name", "is_label_function"),
    [
        ("make_label", True),
        ("get_target", True),
        ("forward_return", True),
        ("fwd_ret_5d", True),
        ("compute_y", True),
        ("target_position", False),
        ("forward_fill_features", False),
        ("target_encoding", False),
        ("features", False),
    ],
)
def test_label_functions_are_exempt_from_pp001_pp009_and_pp011(
    rule_id: str, name: str, is_label_function: bool
) -> None:
    code = f"def {name}(x, df=None, n=3):\n    {LABEL_CODE[rule_id]}"
    found = [f for f in lint_snippet(code) if f.rule_id == rule_id]
    assert len(found) == (0 if is_label_function else 1)


@pytest.mark.parametrize("rule_id", sorted(LABEL_CODE))
def test_the_label_marker_comment_exempts_pp001_pp009_and_pp011(rule_id: str) -> None:
    code = f"def build(x, df=None, n=3):  # peekproof: label\n    {LABEL_CODE[rule_id]}"
    assert [f for f in lint_snippet(code) if f.rule_id == rule_id] == []


@pytest.mark.parametrize(
    ("rule_id", "body"),
    [
        ("PP002", "return x.bfill()"),
        ("PP003", "return x.rolling(5, center=True).mean()"),
        ("PP006", "return (x - x.mean()) / x.std()"),
        ("PP010", "return signal * returns"),
    ],
)
def test_label_functions_are_not_exempt_from_the_other_rules(rule_id: str, body: str) -> None:
    code = f"def make_label(x, signal=None, returns=None):\n    {body}"
    assert [f.rule_id for f in lint_snippet(code) if f.rule_id == rule_id] == [rule_id]


# --------------------------------------------------------------------------- #
# Messages and positions of the newer triggers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "words"),
    [
        ("z = x.shift(-HORIZON)", ["shift(-HORIZON)", "negative shift by a variable"]),
        ("z = np.roll(x, -n)", ["np.roll(..., -n)", "negative shift by a variable"]),
        ("fill_price = df['open'].shift(-1)", ["shift(-1)", "fill_price", "fill price"]),
        ("z = x.interpolate()", ["interpolate()", "next valid observation"]),
        ("z = x.interpolate(method='time')", ['interpolate(method="time")']),
        ("z = x.reindex(idx, method='nearest')", ['reindex(method="nearest")']),
        ("z = x.asfreq('D', method='bfill')", ['asfreq(method="bfill")']),
        ("z = x.pct_change(fill_method='bfill')", ['pct_change(fill_method="bfill")']),
        ("z = filtfilt(b, a, x)", ["filtfilt", "later rows"]),
        ("z = hpfilter(x)", ["hpfilter", "whole sample"]),
        ("z = seasonal_decompose(x)", ["seasonal_decompose"]),
        ("z = np.gradient(x)", ["np.gradient", "central differences"]),
        ("z = FixedForwardWindowIndexer(window_size=3)", ["FixedForwardWindowIndexer"]),
        ("z = np.convolve(x, k)", ["mode 'full'", "[:n]"]),
        ("z = np.convolve(x, k, mode='same')", ["mode='same'"]),
        ("z = GridSearchCV(est, grid)", ["GridSearchCV", "cv="]),
        ("mu = x.mean()\nz = x - mu", ["'mu'", "mean()", "same series"]),
        ("z = scale(x)", ["scale()", "whole series"]),
        ("z = pd.qcut(x, 5)", ["qcut()", "quantiles of the whole series"]),
        ("out = [x[i : i + 3].mean() for i in range(n)]", ["x[i : i + 3]", "i+2"]),
        ("out = []\nfor i in range(n):\n    out.append(x[i + 1])", ["x[i + 1]", "row i+1"]),
        ("p = signal * np.log(close).diff()", ["np.log(close).diff()"]),
    ],
)
def test_messages_name_what_was_found_for_the_newer_triggers(code: str, words: list[str]) -> None:
    message = lint_snippet(code)[0].message
    for word in words:
        assert word in message, (word, message)


def test_a_stored_statistic_is_reported_where_it_meets_the_series() -> None:
    (finding,) = lint_snippet("mu = x.mean()\nsd = x.std()\nz = (x - mu) / sd\n")
    assert (finding.rule_id, finding.line, finding.col) == ("PP006", 3, 10)
    assert finding.snippet == "z = (x - mu) / sd"


def test_a_stored_statistic_is_reported_once_per_result() -> None:
    findings = lint_snippet("mu = x.mean()\nsd = x.std()\na = (x - mu) / sd\nb = x / sd\n")
    assert [f.line for f in findings] == [3, 4]


def test_a_direct_aggregate_and_a_stored_one_in_one_result_are_one_finding() -> None:
    assert len(lint_snippet("mu = x.mean()\nz = (x - mu) / x.std()\n")) == 1


# --------------------------------------------------------------------------- #
# Speed: the cost of a file grows with its size, not with its square
# --------------------------------------------------------------------------- #


def resample_calls(count: int) -> str:
    return "".join(
        f"bars{i} = t{i}.resample('5min').last()\nr{i} = bars{i}.pct_change()\n"
        for i in range(count)
    )


def zscore_terms(count: int) -> str:
    return "z = " + " + ".join(f"(a{i} - a{i}.mean()) / a{i}.std()" for i in range(count)) + "\n"


def test_a_thousand_resample_calls_and_a_400_term_expression_lint_in_under_three_seconds() -> None:
    source = resample_calls(1000) + zscore_terms(400)
    started = time.perf_counter()
    findings = lint_source(source)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"linting took {elapsed:.1f} s"
    assert Counter(f.rule_id for f in findings) == {"PP008": 1000, "PP006": 1}


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("z = " + " + ".join(["(x - x.mean())"] * 400) + "\n", id="one-series"),
        pytest.param(
            "z = " + " + ".join(f"(df['c{i}'] - df['c{i}'].mean())" for i in range(400)) + "\n",
            id="frame-columns",
        ),
        pytest.param(
            "z = " + " & ".join(f"(a{i} > a{i}.quantile(0.5))" for i in range(400)) + "\n",
            id="comparisons",
        ),
        pytest.param(
            "z = " + " + ".join(f"x.mean() * y{i}" for i in range(400)) + " + x\n", id="many-uses"
        ),
        pytest.param(
            "".join(f"mu{i} = x{i}.mean()\nz{i} = x{i} - mu{i}\n" for i in range(400)),
            id="stored-statistics",
        ),
    ],
)
def test_pp006_is_fast_on_long_expressions_of_any_shape(source: str) -> None:
    started = time.perf_counter()
    findings = lint_source(source, select=["PP006"])
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"PP006 took {elapsed:.1f} s"
    assert findings


def test_pp008_is_fast_with_many_resample_calls_in_one_function() -> None:
    body = "".join(
        f"    bars{i} = t{i}.resample('5min').last()\n    r{i} = bars{i}.pct_change()\n"
        for i in range(1000)
    )
    started = time.perf_counter()
    findings = lint_source("def f(" + ", ".join(f"t{i}" for i in range(1000)) + "):\n" + body)
    assert time.perf_counter() - started < 3.0
    assert len(findings) == 1000


# --------------------------------------------------------------------------- #
# The module keeps its helpers private
# --------------------------------------------------------------------------- #


def test_messages_and_fixtures_use_american_spelling() -> None:
    sources = [Path(rules_module.__file__), FIXTURES / "leaky" / "pp003.py"]
    for path in sources:
        assert "centred" not in path.read_text(encoding="utf-8").lower(), path.name


def test_only_expr_key_is_a_public_function_of_the_rules_module() -> None:
    public = {
        name
        for name, value in vars(rules_module).items()
        if isinstance(value, types.FunctionType)
        and value.__module__ == rules_module.__name__
        and not name.startswith("_")
    }
    assert public == {"expr_key"}


def test_the_rules_module_lists_what_it_exports() -> None:
    exported = set(rules_module.__all__)
    assert exported == {*RULES, "RULES", "Rule", "expr_key"}
    assert all(hasattr(rules_module, name) for name in exported)
    classes = {
        name
        for name, value in vars(rules_module).items()
        if isinstance(value, type) and value.__module__ == rules_module.__name__
    }
    assert {name for name in classes if not name.startswith("_")} == exported - {
        "RULES",
        "expr_key",
    }
