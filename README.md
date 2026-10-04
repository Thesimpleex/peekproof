# peekproof

**Detect look-ahead leakage in backtests.** A dynamic causality oracle that
tests whether a feature function uses information from the future, and a
static linter that catches the usual leak patterns in pandas, polars, NumPy and
scikit-learn research code before anyone runs a backtest.

[![CI](https://github.com/Thesimpleex/peekproof/actions/workflows/ci.yml/badge.svg)](https://github.com/Thesimpleex/peekproof/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<img alt="Animated terminal session: peekproof check reports a centered rolling mean as NOT CAUSAL with a horizon of 9 rows; peekproof lint reports six findings in a strategy file" src="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/terminal-session.svg" width="100%">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/leak-vs-causal-dark.png">
  <img alt="Cumulative return of the same mean-reversion rule computed with a centered rolling mean (Sharpe 5.1) and with a trailing mean (Sharpe 0.1) on a synthetic random walk" src="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/leak-vs-causal-light.png" width="100%">
</picture>

The two curves above come from the same rule on the same synthetic random
walk. The only difference is `rolling(20, center=True)` versus `rolling(20)`.
A centered window reads nine rows into the future, and that is enough to turn
a strategy with no edge by construction into a Sharpe ratio above 5 (seed 0;
other seeds land between 3.7 and 5.3). peekproof reports exactly that:
**NOT CAUSAL, horizon = 9 rows**.

This is not a lucky seed. Across fifty independent random walks, none of which
has any structure to exploit, the centered window manufactures a Sharpe ratio
between 3.7 and 5.3 every single time, while the honest version scatters
around zero.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/leak-sharpe-worlds-dark.png">
  <img alt="Sharpe ratio of the leaky and the causal variant across 50 synthetic random walks: the leaky variant clusters between 3.7 and 5.3 with median 4.5, the causal variant scatters around zero" src="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/leak-sharpe-worlds-light.png" width="100%">
</picture>

## Why

Look-ahead bias goes by many names: lookahead bias, future leakage, temporal
leakage, peeking, or simply data leakage in a time series. Whatever it is
called, it is the most common reason a backtest, a cross-validated model or a
trading strategy looks far better in research than in production.

Most backtest failures are not subtle statistics. They are a `shift(-1)`
where a `shift(1)` was meant, a `bfill()` on a gap, a scaler fit on the whole
sample, a `train_test_split` that shuffles time away, a position multiplied by
the same bar's return, or a universe built from today's index constituents.
Each of these is easy to write, hard to see in a notebook, and invisible in a
tear sheet, where it shows up only as performance that is too good.

peekproof attacks the problem from two sides:

- The **oracle** does not read your code. It runs your function on the full
  history and on histories truncated or scrambled after a cutoff, then checks
  whether any output at or before the cutoff changed. If the past depends on
  the future, the function is not causal, and the report says where, by how
  much and how far ahead it looked.
- The **linter** does not run your code. It reads it and flags the patterns
  that produce leaks, with a one-line explanation and a concrete fix for each.

## Install

```
pip install peekproof
```

Until the first PyPI release:

```
pip install git+https://github.com/Thesimpleex/peekproof
```

Requires Python 3.12 or later. Runtime dependencies are pandas and NumPy only.
With the `peekproof[polars]` extra, polars DataFrames are accepted and the
function under test receives polars frames.

## Quick start

```python
import peekproof

report = peekproof.check(build_features, prices)  # prices: DataFrame with a DatetimeIndex
print(report)  # CAUSAL / NOT CAUSAL, affected rows, horizon
```

```
peekproof lint src/ notebooks/        # static rules, exit 1 on high-confidence findings
peekproof check features:build_features --data prices.parquet
```

## The oracle

```python
import pandas as pd
import peekproof


def momentum(prices: pd.DataFrame) -> pd.Series:
    return prices["close"].pct_change(20)


def smooth(prices: pd.DataFrame) -> pd.Series:
    return prices["close"].rolling(20, center=True).mean()


report = peekproof.check(momentum, prices)
report.causal  # True

report = peekproof.check(smooth, prices)
report.causal  # False
report.horizon_rows  # 9
print(report)  # the text shown in the terminal above
```

Labels are allowed to look ahead, but only as far as they claim:

```python
def forward_return_5(prices):
    return prices["close"].pct_change(5).shift(-5)


peekproof.check_label(forward_return_5, prices, horizon=5).causal  # True
peekproof.check_label(forward_return_5, prices, horizon=4).causal  # False
```

And the whole thing fits in a test:

```python
def test_features_are_causal(prices):
    peekproof.assert_causal(build_features, prices)  # raises LeakError with the report
```

### How it decides

<img alt="Diagram: the function is evaluated on the full history and on the history truncated at a cutoff; rows at or before the cutoff must agree, otherwise the difference is a leak and its distance to the cutoff is the horizon" src="https://raw.githubusercontent.com/Thesimpleex/peekproof/main/docs/assets/how-it-works.svg" width="100%">

A function `f` from a time-indexed table to a time-indexed output is
**causal** if, for every cutoff `t`, the output restricted to times `≤ t` is
identical whether `f` saw the rows after `t` or not, and every row that exists
in the full output at a time `≤ t` also exists when the future is removed.

That second clause is what catches `shift(-1).dropna()`: the values on the
overlap agree, but the last row only exists when the next row is present. In
production that row would never exist, so it is reported as a leak of kind
`missing_row`.

peekproof runs two tests at 64 cutoffs each by default (the report header counts
the distinct cutoff positions across both tests):

- **truncation** removes the rows after the cutoff;
- **perturbation** keeps the rows but permutes them and adds noise, which
  also covers functions that need a minimum length and exposes dependence on
  future *values*, not just on their existence.

For each leaking column it reports the affected rows, the largest difference
and the **horizon**: the distance from the cutoff to the earliest row that
changed. `shift(-3)` gives 3, a centered window of 20 gives 9, a whole-sample
z-score gives *unbounded*. A function that is not deterministic is reported as
such and not judged; seed it first.

What the oracle cannot do:

- It only sees the data it passes to the function. A module-level mean, a
  scaler fitted on the full sample outside the function, or an external frame
  joined inside it are invisible to the test. Pass everything the function
  needs as part of the input frame, or the leak stays hidden.
- It samples cutoffs (64 per test by default, 130 evaluations in total), so
  it provides evidence rather than a proof. A leak that touches only a handful
  of rows can still slip between cutoffs; raise `cutoffs` for rare-event
  features.
- Comparisons use a relative tolerance plus an absolute tolerance scaled to
  each column, so very small-scale features are still compared meaningfully;
  differences below `rtol` are not reported.
- Long-format panels with a `MultiIndex` of (date, asset) are not supported in
  this version; use one wide frame per feature instead.
- It cannot tell you whether a causal signal is any good.

## The linter

```
peekproof lint src/ notebooks/
peekproof lint strategy.py --format json --fail-on medium
```

Thirteen rules, each with a message and a fix. Confidence is `high` when the
pattern is a leak in nearly every time-series context the rule can recognise
from the source alone, `medium` or `low` when it depends on context. Static
rules are heuristics: confirm a finding with the oracle before acting on it.

| Rule  | Name                  | Fires on                                                                                                                                                                  | Confidence    |
|-------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| PP001 | negative-shift        | `shift(-k)`, `pct_change(-k)`, `diff(-k)`, `np.roll(x, -k)` outside label code; a negated variable such as `shift(-h)`, or a result stored as an execution price, at medium | high / medium |
| PP002 | backward-fill         | `bfill()`, `fillna(method="bfill")`, `reindex(method="bfill"/"nearest")`, `asfreq(method="bfill")`, `pct_change(fill_method="bfill")`; `interpolate()` at medium          | high / medium |
| PP003 | centered-window       | `rolling(n, center=True)` for n ≥ 3, `savgol_filter`, `filtfilt`, `hpfilter`, `seasonal_decompose`, `np.gradient`, `FixedForwardWindowIndexer`; `np.convolve` at medium  | high / medium |
| PP004 | fit-on-full-sample    | a scaler, decomposer or model fit on data that is split afterwards                                                                                                        | medium        |
| PP005 | shuffled-split        | `train_test_split` without `shuffle=False`, `KFold(shuffle=True)`, `ShuffleSplit`, `StratifiedKFold`; cross-validation or `GridSearchCV` without a time-aware `cv` at medium | high / medium |
| PP006 | full-sample-statistic | a whole-sample `mean()`, `std()`, `min()`, `max()`, `quantile()` or `rank()` combined with the same series, also via a stored value; `pd.qcut`, `preprocessing.scale`      | medium        |
| PP007 | forward-asof-join     | `merge_asof(direction="forward"/"nearest")`; joins on period-end keys without a filing or availability date at low                                                       | high / low    |
| PP008 | bar-label-leak        | `resample()` aggregates used as features without right-labeled bars                                                                                                      | low           |
| PP009 | forward-index-in-loop | `x[i + k]`, `x.iloc[i + k]` or `.append(x[i + k])` read while writing row `i` in a loop; forward slices `x[i:i + k]` at medium                                           | high / medium |
| PP010 | same-bar-execution    | a position or signal multiplied by the same bar's return, including `diff()` of prices, without a one-bar lag                                                            | medium        |
| PP011 | target-as-feature     | a feature computed from a label column (`y`, `label`, `target`, `fwd_ret*`, `forward_return*`)                                                                           | medium        |
| PP012 | current-constituents  | a universe built from today's index membership (Wikipedia lists, `yf.Tickers`)                                                                                           | medium        |
| PP013 | latest-fundamentals   | yfinance fundamentals used next to historical prices (not point-in-time)                                                                                                 | low           |

Polars spellings (`shift(n=-1)`, `backward_fill()`, `fill_null(strategy="backward")`,
`rolling_*(center=True)`, `join_asof(strategy=...)`) are covered by the same
rules. Notebooks are linted cell by cell; findings are reported as
`notebook.ipynb#cell3:2`.

Suppress a finding on one line with `# peekproof: ignore` or
`# peekproof: ignore[PP001]`. Label code is allowed to look ahead: mark a label
function with `# peekproof: label` on its `def` line, or give it a name that
ends in `label`, `labels`, `target`, `targets` or `y`, or that contains a
forward-return token such as `fwd_ret` or `forward_return_5`. Inside label
functions PP001, PP009 and PP011 stay quiet, and so does PP001 for assignments
to label-named targets such as `y = ...` or `df["target"] = ...`. Cross-sectional
operations (`axis=1`) are never reported as time shifts.

Project-wide settings live in `pyproject.toml`:

```toml
[tool.peekproof]
ignore = ["PP008"]
min_confidence = "medium"
label_functions = "(?:^|_)(?:label|target|y|outcome)$"
exclude = ["notebooks/scratch/*"]
```

Exit codes: `0` nothing to report, `1` findings at or above `--fail-on`
(default `high`), a leak found by `check`, or a function `check` could not
judge because it is not deterministic, `2` usage error. A file that does not
parse is reported as `PP000` at low confidence. The repository ships a
`.pre-commit-hooks.yaml`, so `peekproof lint` runs as a pre-commit hook, and
both commands work as CI steps.

## Command line

```
peekproof lint PATH [PATH ...] [--select IDS] [--ignore IDS]
               [--min-confidence low|medium|high] [--format text|json]
               [--fail-on high|medium|low|never] [--config pyproject.toml]

peekproof check MODULE:FUNCTION --data FILE.csv|FILE.parquet
               [--index-col NAME | --time-col NAME] [--cutoffs N]
               [--mode truncate|perturb|both] [--label-horizon H]
               [--format text|json]
```

`check` imports `MODULE` from the current directory, loads the data with a
time index, and prints the report.

## Validation

The test-suite contains more than 1,700 tests. The oracle is checked against planted
leaks with known answers: `shift(-k)` must report a horizon of `k`,
`shift(-1).dropna()` must be caught through missing rows, a centered window of
20 must report 9, a whole-sample z-score must report an unbounded horizon,
and a dozen causal constructions (trailing windows, exponential averages,
expanding statistics, lagged returns, cross-sectional ranks per date,
backward as-of joins) must pass. Each linter rule has a leaky fixture that
fires exactly that rule and a clean counterpart that fires nothing.

The chart and the terminal session in this README are produced by
`docs/make_assets.py` from seeded synthetic data and real command output;
nothing in them is tuned by hand.

## Scope and roadmap

peekproof finds leaks. It does not make a strategy profitable, it does not
detect survivorship bias in a dataset (PP012 only flags the most common way of
introducing it), and its static rules are heuristics: a `medium` or `low`
finding is a reason to run the oracle, not a verdict.

Planned next:

- a **trial ledger** that records every backtest evaluation so that
  search-adjusted statistics (deflated Sharpe ratio, probability of backtest
  overfitting) use the true number of trials instead of a guess;
- **pricing a leak**: the performance difference between a leaky signal and its
  causal reconstruction;
- a public, labeled corpus of leaky and clean research snippets to measure
  detection rates, this tool's included;
- a GitHub Action, and a machine-readable interface so that automated
  strategy generators can be held to the same check.

## Related work and references

Static look-ahead linters with a handful of rules exist (for example
`lookahead-lint` and `leakguard-mcp`), as do research scripts that re-run a
function on truncated data. peekproof combines both approaches, adds the
missing-row and perturbation semantics, measures the horizon, and ships a
test-suite with known answers.

- X. Fonseca, *Look-Ahead-Freedom as Temporal Non-Interference*, 2026
  (arXiv:2607.04958) formalises the property the oracle tests.
- Yin, Miki, Lesnichenko and Gural, *Implementation Risk in Portfolio
  Backtesting*, 2026 (arXiv:2603.20319) documents how far backtesting engines
  diverge once costs are involved.
- Zhang, Li, Peng and Chen, *When Alpha Disappears: A One-Switch Benchmark for
  Decision-Time Leakage in Financial Backtests*, 2026 (arXiv:2605.23959) shows
  which leaks inflate results most: centered features and same-bar execution.
- Kong et al., *Evaluating LLMs in Finance Requires Explicit Bias
  Consideration*, 2026 (arXiv:2602.14233) catalogues look-ahead, survivorship
  and cost biases across 164 studies.
- Bailey and López de Prado, *The Deflated Sharpe Ratio*, Journal of Portfolio
  Management, 2014, is the basis for the planned trial ledger.

## Development

Design notes on the oracle's semantics and the rule catalogue are in
[docs/DESIGN.md](docs/DESIGN.md); where they and the code disagree, the code
and its tests are authoritative.

```
uv sync --all-extras --dev
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run python docs/make_assets.py   # regenerates the figures
```

## Contributing

Bug reports with a minimal reproduction, new linter rules with a leaky and a
clean fixture, and additional oracle test cases are welcome. Run the checks
listed under Development before opening a pull request.

## License

MIT. See [LICENSE](LICENSE).
