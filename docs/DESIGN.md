# peekproof design notes (version 0.1)

## 1. Problem

Backtests lie most often for one of two reasons:

1. **Look-ahead leakage.** A feature or signal computed at time *t* silently uses
   information that only became available after *t* (negative shifts, backward
   fills, centered windows, whole-sample normalisation, forward as-of joins,
   same-bar execution, scalers fit on the full sample, current index
   constituents, non-point-in-time fundamentals).
2. **Selection bias from search.** Dozens or hundreds of variants are tried and
   only the best one is reported, so the reported Sharpe ratio is a maximum of
   noisy draws, not an estimate.

Existing tools treat both problems as fragments: a handful of single-file AST
linters with a few rules each, a few research scripts that re-run a function on
truncated data, and several packages that compute a deflated Sharpe ratio once
someone types in the number of trials by hand. None of them is engine-agnostic,
none ships a public labeled corpus, none is a de facto CI check, and none
counts the trials automatically.

`peekproof` is a small, well-tested Python library and CLI that makes a
research pipeline **peek-proof**:

- a **dynamic oracle** that proves or disproves causality of any feature
  function by re-evaluating it on truncated and perturbed histories
  (temporal non-interference), reports the affected rows and the leak horizon;
- a **static linter** for pandas / polars / numpy / scikit-learn research code
  with a rule set that is the union of what is known plus several rules nobody
  implements (same-bar execution, current constituents, period-end joins);
- (phase 2) a **trial ledger** that records every backtest evaluation and feeds
  the true trial count into search-adjusted statistics (deflated Sharpe ratio,
  probability of backtest overfitting);
- (phase 3) a **public labeled corpus** of leaky and clean research snippets to
  benchmark this and other detectors.

Everything runs on the user's own code and data. No proprietary data is needed.

## 2. Scope of version 0.1

Version 0.1 contains the oracle, the linter with the rule set in §5, the CLI
and the test-suite. The trial ledger, the labeled corpus and the CI
integrations are later phases (§9).

Supported inputs: pandas `DataFrame`/`Series` with a monotonic index
(`DatetimeIndex` preferred, any monotonic increasing index accepted), and
polars `DataFrame` with a declared time column (converted to pandas internally
in v0.1). Python ≥ 3.12. Runtime dependencies: `pandas`, `numpy` only.
Tested against pandas 3.x (installed) and must not rely on APIs removed in
pandas 3 (`fillna(method=...)`, silent downcasting, etc.).

## 3. Package layout

```
src/peekproof/
  __init__.py      # public API re-exports + __version__
  oracle.py        # dynamic causality oracle
  report.py        # dataclasses: OracleReport, LeakFinding, Finding; JSON + text rendering
  errors.py        # LeakError, PeekproofError
  _compat.py       # pandas/polars adapters, index helpers
  lint/
    __init__.py    # lint_path, lint_source
    engine.py      # file discovery (.py, .ipynb), AST walking, suppression comments, config
    rules.py       # rule implementations, one class per rule, registry
  cli.py           # argparse entry point: `peekproof lint`, `peekproof check`
tests/
  test_oracle.py
  test_oracle_horizon.py
  test_lint_rules.py
  test_lint_engine.py
  test_cli.py
  fixtures/
    leaky/*.py     # one file per rule, must fire
    clean/*.py     # counterpart, must not fire
    notebook_leaky.ipynb
```

The project is built with uv (`uv_build`); the console script is
`peekproof = "peekproof.cli:main"`.

## 4. Dynamic oracle (`peekproof.oracle`)

### 4.1 Definition

Let `f` map a time-indexed table `D` to a time-indexed output `f(D)` whose
index is a subset of `D`'s index. `f` is **causal** if for every cutoff time
`t` the restriction of `f(D)` to times ≤ `t` equals `f(D restricted to ≤ t)`
on those times, and every row that exists in `f(D)` at a time ≤ `t` also exists
in `f(D restricted to ≤ t)`.

The second clause matters: a function that computes `shift(-1)` and then
`dropna()` produces identical values on the overlap but *loses* the last row
whenever the future is removed. In production that row would never exist, so
"output at `s` exists only when rows after `s` are present" is a leak and is
reported as such (`kind="missing_row"`).

### 4.2 Public API

```python
def check(
    fn: Callable[[pd.DataFrame], pd.DataFrame | pd.Series],
    data: pd.DataFrame | "pl.DataFrame",
    *,
    cutoffs: int | Sequence[int] = 64,    # count per test, or explicit row positions
    mode: Literal["truncate", "perturb", "both"] = "both",
    atol: float = 1e-9,
    rtol: float = 1e-7,
    time_col: str | None = None,          # required for polars input
    seed: int = 0,                        # for perturbation noise
    min_rows: int | None = None,          # smallest truncation length to try
) -> OracleReport

def check_label(fn, data, *, horizon: int, **kwargs) -> OracleReport
    # A label (e.g. forward return over h bars) is allowed to look exactly
    # `horizon` rows ahead: output at s may depend on rows ≤ s + horizon and on
    # nothing later. Fails if the measured horizon exceeds `horizon`. Explicit
    # cutoffs at or below the horizon compare nothing and are dropped with a
    # note; if none remain the call raises.

def assert_causal(fn, data, **kwargs) -> None
    # raises LeakError (subclass of AssertionError) whose message is str(report)
```

### 4.3 Algorithm

1. **Determinism pre-check.** Evaluate `fn(data)` twice. If the two outputs
   differ (same tolerance rules as below), set `deterministic=False`, add a
   note ("function is not deterministic; seed it before testing") and return
   without running leak tests (`causal=None`).
2. **Normalise output.** Accept `Series` or `DataFrame`; convert `Series` to a
   one-column frame named after the series or `"value"`. The output index must
   be of the same kind as the input index (a time index for a time index);
   rows are matched by label, and rows of a truncated or perturbed output that
   are labeled after the cutoff are ignored, which makes right-labeled
   resampled bars checkable. An output with an index of another kind raises
   `PeekproofError` naming the evaluation and the cutoff. polars input is
   passed to the function as polars frames; the output must keep the time
   column.
3. **Choose cutoff positions.** If `cutoffs` is an int `k` (default 64), each
   test gets `k` positions: `T-1`, `T-2` and `T//2` plus the rest spread
   uniformly over the admissible range with a seeded jitter, one per stretch.
   The truncation test starts at `max(2, min_rows, ceil(0.1·T))`; the
   perturbation test starts at row 2, because the function still sees the full
   length. If a sequence is given it is used as is (validated, sorted,
   de-duplicated). The report lists the union of the positions used.
4. **Truncation test** (mode `truncate` or `both`). For each position `p`:
   `part = fn(data.iloc[:p])`. Compare `part` with the rows of `full` at or
   before the cutoff:
   - rows present in `full` (≤ p) but absent in `part`, and columns of `full`
     that `part` lacks → `missing_row` findings;
   - rows present in both: numeric columns compared with
     `np.isclose(a, b, rtol=rtol, atol=atol · median|column|, equal_nan=True)`
     so that the absolute tolerance follows the scale of each column;
     non-numeric compared with equality treating missing values as equal →
     `value_change` findings with `max_abs_diff` (numeric) or
     `n_rows_affected` only.
   - If `fn` raises on the truncated input, record a note and continue. If it
     raises on every cutoff and the other test found nothing either, raise
     `PeekproofError` suggesting `min_rows`; if the other test found leaks,
     keep them and add a note.
5. **Perturbation test** (mode `perturb` or `both`). For each position `p`:
   build `pert = data.copy()` and change every row ≥ p: floats receive
   heavy-tailed noise (Student-t with 3 degrees of freedom, scaled by a robust
   spread of the column, never zero), integers move to another value in the
   observed range, booleans flip (the first future row always), categoricals
   and strings take another observed value, other dtypes are permuted. Dtypes
   and missing values are preserved. Evaluate `fn(pert)` and compare rows < p
   with `full` exactly as in step 4. The perturbation test catches leaks in
   functions that need a minimum length and detects dependence on future
   *values*, not only on their existence.
6. **Horizon.** For each cutoff `p` with findings, let `s_min(p)` be the
   earliest affected row position. `horizon_rows(p) = p - s_min(p)`
   (so `shift(-1)` gives 1, `shift(-3)` gives 3, a centered window of 20 gives
   9). Report `horizon_rows = max over p`. If for the majority of cutoffs with
   findings the horizon reaches back into the first 5 % of rows, set
   `horizon_unbounded=True` (whole-sample statistics such as a global z-score).
7. **Report.** Return `OracleReport` (see §6). `causal` is `True` iff no
   findings were produced in any test that ran.

### 4.4 Complexity and limits

`k` cutoffs cost `k` evaluations per mode plus 2 for determinism; the default
is therefore 130 evaluations. The oracle samples cutoffs, so it provides
evidence rather than a proof, and it only sees the data it passes to the
function: module-level state, pre-fitted scalers and external frames are
invisible to it, so a leak through them is not detected. The README lists
this as a limitation. Outputs indexed by time may carry labels that are not in
the input (resampled bars); outputs indexed by anything else must keep the
input labels. Long-format panels with a `MultiIndex` are rejected in this
version.

## 5. Static linter (`peekproof.lint`)

### 5.1 Engine

- Discover files: paths given on the CLI; directories are walked for `*.py`
  and `*.ipynb` (skip `.venv`, `site-packages`, `__pycache__`, `.git`,
  configured `exclude` globs).
- Notebooks: read JSON, lint each code cell separately, report
  `path#cell<N>` with cell-relative line numbers. Skip IPython magics
  (lines starting with `%` or `!`) by blanking them before parsing.
- Parse with `ast`; on `SyntaxError` emit one finding `PP000` (confidence
  `low`, "could not parse") and continue.
- Suppression: a trailing comment `# peekproof: ignore` suppresses all rules
  on that line; `# peekproof: ignore[PP001, PP005]` suppresses the listed
  rules. A function whose `def` line carries `# peekproof: label`, or whose
  name matches the configured `label_functions` regex (default: names ending
  in `label`, `labels`, `target`, `targets` or `y`, or containing a
  forward-return token such as `fwd_ret` or `forward_return_5`), is a **label
  function**: PP001, PP009 and PP011 are not reported inside it.
- Configuration from the nearest `pyproject.toml` `[tool.peekproof]` table:
  `select`, `ignore`, `min_confidence`, `label_functions`, `exclude`. CLI
  flags override.
- Each rule is a class with `id`, `name`, `confidence` default, and a
  `visit(tree, ctx) -> list[Finding]`; register in a dict. Keep rules
  independent and side-effect free so they are easy to test.

### 5.2 Rules for v0.1

| ID | Name | Fires on | Confidence |
|----|------|----------|------------|
| PP001 | negative-shift | `.shift(-k)`, `.shift(periods=-k)`, `.pct_change(-k)`, `.diff(-k)`, `np.roll(x, -k)` with a negative literal; polars `shift(n=-k)`; not inside label functions or for label-named targets; skipped for `axis=1`. A negated variable (`shift(-h)`) or a result stored under an execution-price name (`fill_price`, `entry_px`) | high; medium for the last two |
| PP002 | backward-fill | `.bfill(`, `.backfill(`, `.fillna(method="bfill"|"backfill")`, `.reindex(method="bfill"|"backfill"|"nearest")`, `.asfreq(method="bfill")`, `.pct_change(fill_method="bfill")`, `.interpolate(limit_direction="backward"|"both")`, polars `backward_fill`/`fill_null(strategy="backward")`; skipped for `axis=1`. Any other `.interpolate(` except `method="pad"|"ffill"` | high; medium for plain `interpolate` |
| PP003 | centered-window | `.rolling(n, center=True)` with n ≥ 3, polars `rolling_*(center=True)`, `savgol_filter`, `filtfilt`/`sosfiltfilt`, statsmodels `hpfilter`, `seasonal_decompose`, `np.gradient`, `FixedForwardWindowIndexer`, `scipy.ndimage` filters; `np.convolve(` unless the result is cut with `[:n]` | high; medium for convolve |
| PP004 | fit-on-full-sample | `.fit(...)`/`.fit_transform(...)` of a scaler, decomposer or model whose argument is later split, sliced or indexed by a split variable in the same scope | medium |
| PP005 | shuffled-split | `train_test_split(...)` without `shuffle=False`; `KFold(shuffle=True)`; `ShuffleSplit(`; `StratifiedKFold(`; `cross_val_*` and `GridSearchCV`/`RandomizedSearchCV` (and halving variants) without a time-aware `cv=` | high (first three), medium (rest) |
| PP006 | full-sample-statistic | a series aggregate (`mean`, `std`, `var`, `min`, `max`, `median`, `quantile`, `rank`, `zscore`) combined with the same series, directly or through a name assigned in the same scope; `pd.qcut(`; `sklearn.preprocessing.scale` and siblings; clipping at full-sample quantiles. Uses that only feed `len()`, `.count()`, `.shape` or `.size` do not count | medium |
| PP007 | forward-asof-join | `merge_asof(direction="forward"|"nearest")`, polars `join_asof(strategy="forward"|"nearest")`; `merge_asof`/`merge` on period-end keys without a filing or availability key | high / low |
| PP008 | bar-label-leak | `.resample(rule)` aggregated with `.last()`, `.ohlc()` or `.agg(` without right labels when the result feeds further feature code in the same scope; end-labeled frequencies are skipped | low |
| PP009 | forward-index-in-loop | inside `for i in range(...)` with step 1: `x[i + k]`, `x.iloc[i + k]`, `.append(x[i + k])` beyond the row being written; forward slices `x[i:i + k]` | high; medium for slices |
| PP010 | same-bar-execution | a product of a signal-like operand (`signal`, `position`, `weight`, `side`, `exposure`) and a returns-like operand (`ret`, `return`, `pct_change(`, `diff()` of prices) where neither side is lagged; forecast names are skipped | medium |
| PP011 | target-as-feature | an assignment whose right-hand side reads a label column (`y`, `label`, `target`, `y_true`, suffixes `_label`, `_target`, `_y_true`, `fwd_ret*`, `forward_return*`) outside label functions | medium |
| PP012 | current-constituents | `pd.read_html(` with a Wikipedia constituents URL; names such as `sp500_tickers`; `yf.Tickers(` used to build a universe | medium |
| PP013 | latest-fundamentals | yfinance statement attributes used in a file that also downloads historical prices | low |

Each finding has a one-line `message` and a `hint` telling the user what to do
instead (shift by one bar, use `expanding()`/`rolling()`, fit inside the
training window, use filing dates, use historical constituents, etc.).

### 5.3 Things the linter must not do

No network access, no importing of the linted code, no execution. Pure AST and
text matching. False positives are acceptable at `low`/`medium` confidence
only if the hint explains how to confirm with the oracle; `high` rules must
have a false-positive rate near zero on the clean fixtures.

## 6. Reports

```python
@dataclass
class LeakFinding:
    column: str
    kind: Literal["value_change", "missing_row"]
    test: Literal["truncate", "perturb"]
    cutoff_pos: int
    cutoff_label: str           # str() of the index label at cutoff
    n_rows_affected: int
    first_affected_label: str
    last_affected_label: str
    max_abs_diff: float | None

@dataclass
class OracleReport:
    causal: bool | None          # None when not deterministic
    deterministic: bool
    n_rows: int
    n_evaluations: int
    cutoff_positions: list[int]
    findings: list[LeakFinding]
    horizon_rows: int | None
    horizon_unbounded: bool
    notes: list[str]
    def to_dict(self) -> dict ...
    def to_json(self, indent=2) -> str ...
    def __str__(self) -> str ...  # compact human-readable summary, no colours

@dataclass
class Finding:                   # linter
    rule_id: str
    name: str
    message: str
    hint: str
    path: str
    line: int
    col: int
    confidence: Literal["low", "medium", "high"]
    snippet: str
```

Text rendering of the oracle report, for a leaky function, looks like:

```
peekproof: NOT CAUSAL  (deterministic, 8 cutoffs, 18 evaluations)
  column 'signal': 8 cutoffs affected, kind=value_change, max |diff| = 2.31e-01
    first affected row 2021-03-11, horizon = 1 row(s)
  verdict: output at t depends on rows after t (horizon 1). Shift the feature or fix the label.
```

## 7. CLI

```
peekproof lint PATH [PATH ...] [--select IDS] [--ignore IDS]
               [--min-confidence low|medium|high] [--format text|json]
               [--fail-on high|medium|low|never] [--config pyproject.toml]

peekproof check MODULE:FUNCTION --data FILE.csv|FILE.parquet
               [--index-col NAME | --time-col NAME] [--cutoffs N]
               [--mode truncate|perturb|both] [--label-horizon H]
               [--format text|json]
```

Exit codes: `0` nothing to report, `1` findings at or above `--fail-on`
(default `high`), a leak found, or a function that could not be judged because
it is not deterministic; `2` usage error. A file that does not parse is a
`PP000` finding at low confidence, not a usage error. `check` imports the
module from the current working directory (which is put first on `sys.path`).

## 8. Test-suite

Oracle, clean functions (must report `causal=True`):
rolling mean, `ewm().mean()`, `expanding()` z-score, lagged returns
(`pct_change().shift(1)`), cross-sectional rank per date on a wide frame
(`rank(axis=1)`), `merge_asof(direction="backward")` against a second frame
built from the input, a function that returns a strict subset of columns, a
function that drops warm-up NaNs at the *start* only.

Oracle, leaky functions (must report `causal=False`, with the stated horizon):
`shift(-1)` (horizon 1), `shift(-3)` (3), `shift(-1)` followed by `dropna()`
(found via `missing_row`), `bfill()`, centered rolling window 5 (2), global
z-score (unbounded), global min-max scaling (unbounded), `rank(pct=True)` over
time (unbounded), `StandardScaler().fit_transform` on the full frame
(unbounded), a loop reading `iloc[i+1]`, `pct_change(-2)`.

Label mode: forward 5-bar return passes with `horizon=5`, fails with
`horizon=4`.

Robustness: non-deterministic function → `deterministic=False`, `causal=None`;
function raising below 50 rows handled with `min_rows`; polars input with
`time_col`; `Series` output; output with extra columns raises a clear error
only when its index is not a subset.

Linter: for every rule one leaky fixture that fires exactly that rule and one
clean fixture that fires nothing; suppression comments; label-function
exemption for PP001; notebook linting; `pyproject.toml` config; CLI exit
codes and JSON output.

Quality gates: `uv run pytest -q` green; `uv run ruff check .` and
`uv run ruff format --check .` clean; `uv run mypy src` clean with
`strict = true` relaxed only where pandas typing makes it unreasonable
(document each relaxation in `pyproject.toml`).

## 9. Non-goals for v0.1

Trial ledger and deflated statistics (phase 2), labeled corpus and benchmark
against other linters (phase 3), CI integrations (phase 3),
"pricing" a leak in Sharpe-ratio terms (phase 2), polars-native evaluation
without conversion (later), inter-procedural dataflow in the linter (later).

## 10. Writing style for code and docs

Plain English, short docstrings, no marketing language, no claims about
results that have not been measured. The README states what the tool does, how
to install and run it, what it cannot do, and how it was validated.

Where the implementation deviates from these notes, the code and its tests are
authoritative; the notes record the intent.
