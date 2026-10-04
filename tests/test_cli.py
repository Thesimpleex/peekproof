"""Tests for the command-line interface: ``peekproof lint`` and ``peekproof check``.

``main`` is called in-process with an argument list and its output is read with ``capsys``.
The lint tests use the fixtures in ``tests/fixtures`` or small files in ``tmp_path``. ``check``
imports its target from the working directory, so those tests write a module and a data file
into ``tmp_path`` and change into it. All data is synthetic and seeded.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import pytest

from peekproof.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
LEAKY_PP003 = FIXTURES / "leaky" / "pp003.py"
CLEAN_PP003 = FIXTURES / "clean" / "pp003.py"
NOTEBOOK = FIXTURES / "notebook_leaky.ipynb"

N_ROWS = 240
#: Data arguments for the CSV that the ``project`` fixture writes (``date`` is its index column).
CSV = ("--data", "prices.csv", "--index-col", "date")

FINDING_KEYS = {
    "rule_id",
    "name",
    "message",
    "hint",
    "path",
    "line",
    "col",
    "confidence",
    "snippet",
}
LEAK_FINDING_KEYS = {
    "column",
    "kind",
    "test",
    "cutoff_pos",
    "cutoff_label",
    "n_rows_affected",
    "first_affected_label",
    "last_affected_label",
    "max_abs_diff",
}
#: One line of a text report: ``path:line:col: PP003 [high] message``.
FINDING_LINE = re.compile(
    r"^(?P<path>.+):(?P<line>\d+):(?P<col>\d+): (?P<rule>PP\d{3}) "
    r"\[(?P<confidence>low|medium|high)\] .+$",
    re.MULTILINE,
)

# Lint sources. Each of the first three has exactly one finding, of the confidence in its name.
HIGH = """\
import pandas as pd


def build(prices: pd.Series) -> pd.Series:
    return prices.shift(-1)
"""
MEDIUM = """\
import pandas as pd


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["z"] = (df["close"] - df["close"].mean()) / df["close"].std()
    return df
"""
LOW = """\
bars = ticks.resample("5min").last()
returns = bars.pct_change()
"""
# Two high findings from different rules: PP001 on line 5 and PP003 on line 6.
TWO_LEAKS = """\
import pandas as pd


def build(prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    ahead = prices.shift(-1)
    smooth = prices.rolling(5, center=True).mean()
    return ahead, smooth
"""
SOURCE_BY_CONFIDENCE = {"high": HIGH, "medium": MEDIUM, "low": LOW}

# The module that the ``check`` tests import from their working directory.
FEATURES = """\
import itertools

THRESHOLD = 3  # not callable
_calls = itertools.count(1)


def causal(df):
    return df[["close"]].rolling(20).mean()


def leaky(df):
    # the window reaches 9 rows ahead
    return df[["close"]].rolling(20, center=True).mean()


def forward_return(df):
    # a label: the return over the next 5 rows
    return (df["close"].shift(-5) / df["close"] - 1.0).rename("fwd_ret")


def flaky(df):
    # two calls on the same data give different results
    return df[["close"]] * next(_calls)


def broken(df):
    raise ValueError("bad input")


def not_a_frame(df):
    return 3.14
"""


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


class Result(NamedTuple):
    code: int
    out: str
    err: str


class Cli(Protocol):
    def __call__(self, *argv: str | Path) -> Result: ...


def make_prices(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Business-day frame named ``date`` with a random-walk ``close`` and an integer ``volume``."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n, name="date")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    volume = rng.integers(1_000, 5_000, n)
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_module(root: Path, name: str, source: str) -> Path:
    path = write(root / f"{name}.py", source)
    importlib.invalidate_caches()
    return path


def rules_in(out: str) -> list[str]:
    """Rule ids of the finding lines of a text report, in output order."""
    return [match["rule"] for match in FINDING_LINE.finditer(out)]


def reported_count(out: str) -> int:
    """The N of the closing ``peekproof: N finding(s)`` line of a text report."""
    last = out.rstrip("\n").splitlines()[-1]
    match = re.fullmatch(r"peekproof: (\d+) findings?", last)
    assert match, f"unexpected closing line: {last!r}"
    return int(match[1])


def assert_usage_error(result: Result, fragment: str = "") -> None:
    """Exit status 2, nothing on stdout, and an error message on stderr without a traceback."""
    assert result.code == 2
    assert result.out == ""
    assert result.err.startswith("peekproof: error:")
    assert "Traceback" not in result.err
    assert fragment in result.err


def assert_verdicts(cli: Cli, *data_args: str) -> None:
    """With these data arguments the causal function passes and the leaky one is found."""
    causal = cli("check", "features:causal", *data_args)
    assert (causal.code, causal.err) == (0, "")
    assert causal.out.startswith("peekproof: CAUSAL")
    leaky = cli("check", "features:leaky", *data_args)
    assert (leaky.code, leaky.err) == (1, "")
    assert leaky.out.startswith("peekproof: NOT CAUSAL")
    assert "horizon = 9 row(s)" in leaky.out
    assert "first affected row 2021-" in leaky.out  # the date column became the index


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    """Run ``peekproof.cli.main`` in-process; return its exit code and captured output."""

    def run(*argv: str | Path) -> Result:
        code = main([str(arg) for arg in argv])
        captured = capsys.readouterr()
        return Result(code, captured.out, captured.err)

    return run


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> Iterator[Path]:
    """Working directory with ``features.py`` and ``prices.csv``; ``check`` runs from here.

    ``sys.path`` is restored after the test and the modules that were imported from the
    directory are forgotten, so that no test sees the leftovers of another.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    known = set(sys.modules)
    write_module(tmp_path, "features", FEATURES)
    prices.to_csv(tmp_path / "prices.csv")
    yield tmp_path
    root = tmp_path.resolve()
    for name in set(sys.modules) - known:
        origin = getattr(sys.modules[name], "__file__", None)
        if origin and Path(origin).resolve().is_relative_to(root):
            del sys.modules[name]


# --------------------------------------------------------------------------- #
# Options of the program itself
# --------------------------------------------------------------------------- #


def test_version_flag_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == "peekproof 0.1.0"


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["frobnicate"],
        ["lint"],
        ["lint", "a.py", "--fail-on", "sometimes"],
        ["lint", "a.py", "--format", "xml"],
        ["lint", "a.py", "--min-confidence", "extreme"],
        ["check"],
        ["check", "features:causal"],
        ["check", "features:causal", "--data", "prices.csv", "--mode", "sideways"],
        ["check", "features:causal", "--data", "prices.csv", "--cutoffs", "many"],
        ["check", "features:causal", "--data", "prices.csv", "--label-horizon", "far"],
    ],
    ids=lambda argv: " ".join(argv) or "no arguments",
)
def test_malformed_command_line_exits_with_status_two(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    captured = capsys.readouterr()
    assert excinfo.value.code == 2
    assert captured.out == ""
    assert captured.err.startswith("usage: peekproof")


# --------------------------------------------------------------------------- #
# lint: reports
# --------------------------------------------------------------------------- #


def test_lint_prints_findings_and_exits_one_for_a_leaky_file(cli: Cli) -> None:
    result = cli("lint", LEAKY_PP003)
    found = list(FINDING_LINE.finditer(result.out))
    assert result.code == 1
    assert result.err == ""
    assert len(found) >= 2
    assert {match["rule"] for match in found} == {"PP003"}
    assert {match["path"] for match in found} == {str(LEAKY_PP003)}
    assert result.out.count("\n    hint: ") == len(found)
    assert result.out.splitlines()[-1] == f"peekproof: {len(found)} findings"


def test_lint_prints_zero_findings_and_exits_zero_for_a_clean_file(cli: Cli) -> None:
    result = cli("lint", CLEAN_PP003)
    assert (result.code, result.out, result.err) == (0, "peekproof: 0 findings\n", "")


def test_lint_summary_uses_the_singular_for_one_finding(cli: Cli, tmp_path: Path) -> None:
    result = cli("lint", write(tmp_path / "one.py", HIGH))
    assert result.code == 1
    assert rules_in(result.out) == ["PP001"]
    assert result.out.splitlines()[-1] == "peekproof: 1 finding"


def test_lint_accepts_several_paths(cli: Cli) -> None:
    result = cli("lint", CLEAN_PP003, LEAKY_PP003)
    assert result.code == 1
    assert {match["path"] for match in FINDING_LINE.finditer(result.out)} == {str(LEAKY_PP003)}


def test_lint_walks_a_directory_for_python_files(cli: Cli, tmp_path: Path) -> None:
    leaky = write(tmp_path / "src" / "leaky.py", HIGH)
    write(tmp_path / "src" / "pkg" / "clean.py", "x = 1\n")
    write(tmp_path / "src" / "notes.txt", "y = x.shift(-1)\n")
    result = cli("lint", tmp_path / "src")
    assert result.code == 1
    assert [match["path"] for match in FINDING_LINE.finditer(result.out)] == [str(leaky)]


def test_lint_json_is_a_list_of_findings_with_every_key(cli: Cli) -> None:
    result = cli("lint", LEAKY_PP003, "--format", "json")
    items = json.loads(result.out)
    assert result.code == 1
    assert isinstance(items, list)
    assert len(items) >= 2
    for item in items:
        assert set(item) == FINDING_KEYS
        assert (item["rule_id"], item["name"]) == ("PP003", "centered-window")
        assert item["path"] == str(LEAKY_PP003)
        assert isinstance(item["line"], int) and item["line"] >= 1
        assert isinstance(item["col"], int) and item["col"] >= 1
        assert item["confidence"] in {"low", "medium", "high"}
        assert item["message"] and item["hint"] and item["snippet"]
    assert len(items) == reported_count(cli("lint", LEAKY_PP003).out)


def test_lint_json_is_an_empty_list_for_a_clean_file(cli: Cli) -> None:
    result = cli("lint", CLEAN_PP003, "--format", "json")
    assert result.code == 0
    assert json.loads(result.out) == []


def test_lint_text_is_the_default_format(cli: Cli) -> None:
    assert cli("lint", LEAKY_PP003, "--format", "text") == cli("lint", LEAKY_PP003)


def test_lint_reports_the_cell_of_a_notebook_finding(cli: Cli) -> None:
    result = cli("lint", NOTEBOOK)
    paths = [match["path"] for match in FINDING_LINE.finditer(result.out)]
    assert result.code == 1
    assert paths
    assert all(re.fullmatch(r".+\.ipynb#cell\d+", path) for path in paths)


def test_lint_reports_a_file_that_does_not_parse_and_goes_on(cli: Cli, tmp_path: Path) -> None:
    broken = write(tmp_path / "broken.py", "def (:\n")
    leaky = write(tmp_path / "leaky.py", HIGH)
    both = cli("lint", broken, leaky)
    assert sorted(rules_in(both.out)) == ["PP000", "PP001"]
    # DESIGN section 5.1 gives PP000 the confidence "low", so it fails a run only at --fail-on low
    assert cli("lint", broken).code == 0
    assert cli("lint", broken, "--fail-on", "low").code == 1


# --------------------------------------------------------------------------- #
# lint: exit status
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("confidence", "fail_on", "code"),
    [
        pytest.param(confidence, fail_on, code, id=f"{confidence}-finding-fail-on-{fail_on}")
        for confidence, fail_on, code in [
            ("high", "high", 1),
            ("high", "medium", 1),
            ("high", "low", 1),
            ("medium", "high", 0),
            ("medium", "medium", 1),
            ("medium", "low", 1),
            ("low", "high", 0),
            ("low", "medium", 0),
            ("low", "low", 1),
        ]
    ],
)
def test_fail_on_is_the_lowest_confidence_that_gives_exit_status_one(
    cli: Cli, tmp_path: Path, confidence: str, fail_on: str, code: int
) -> None:
    path = write(tmp_path / "sample.py", SOURCE_BY_CONFIDENCE[confidence])
    result = cli("lint", path, "--fail-on", fail_on)
    assert f"[{confidence}]" in result.out  # the finding is reported whatever the exit status
    assert result.code == code


def test_fail_on_defaults_to_high(cli: Cli, tmp_path: Path) -> None:
    medium = write(tmp_path / "medium.py", MEDIUM)
    high = write(tmp_path / "high.py", HIGH)
    assert cli("lint", medium).code == 0
    assert cli("lint", high).code == 1


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_fail_on_never_returns_zero_even_with_findings(cli: Cli, output_format: str) -> None:
    result = cli("lint", LEAKY_PP003, "--fail-on", "never", "--format", output_format)
    assert result.code == 0
    if output_format == "json":
        assert json.loads(result.out)
    else:
        assert rules_in(result.out)


# --------------------------------------------------------------------------- #
# lint: filters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("select", "ignore", "expected"),
    [
        pytest.param(None, None, ["PP001", "PP003"], id="no-filter"),
        pytest.param("PP001", None, ["PP001"], id="select-one-rule"),
        pytest.param("PP003", None, ["PP003"], id="select-the-other-rule"),
        pytest.param("PP001,PP003", None, ["PP001", "PP003"], id="select-a-list"),
        pytest.param(None, "PP003", ["PP001"], id="ignore-one-rule"),
        pytest.param(None, "PP001,PP003", [], id="ignore-a-list"),
        pytest.param("PP001,PP003", "PP003", ["PP001"], id="ignore-wins-over-select"),
        pytest.param("PP002", None, [], id="select-a-rule-that-does-not-fire"),
    ],
)
def test_select_and_ignore_filter_by_rule(
    cli: Cli, tmp_path: Path, select: str | None, ignore: str | None, expected: list[str]
) -> None:
    path = write(tmp_path / "leaks.py", TWO_LEAKS)
    flags: list[str] = []
    if select:
        flags += ["--select", select]
    if ignore:
        flags += ["--ignore", ignore]
    result = cli("lint", path, *flags)
    assert sorted(rules_in(result.out)) == expected
    assert reported_count(result.out) == len(expected)
    assert result.code == (1 if expected else 0)


@pytest.mark.parametrize(
    ("min_confidence", "shown"),
    [("low", True), ("medium", True), ("high", False)],
)
def test_min_confidence_hides_findings_below_it(
    cli: Cli, tmp_path: Path, min_confidence: str, shown: bool
) -> None:
    path = write(tmp_path / "zscore.py", MEDIUM)
    result = cli("lint", path, "--min-confidence", min_confidence, "--fail-on", "low")
    assert rules_in(result.out) == (["PP006"] if shown else [])
    assert result.code == (1 if shown else 0)


def test_min_confidence_high_keeps_the_high_findings_of_a_mixed_file(cli: Cli) -> None:
    everything = cli("lint", LEAKY_PP003)
    high_only = cli("lint", LEAKY_PP003, "--min-confidence", "high")
    assert "[medium]" in everything.out
    assert "[high]" in high_only.out
    assert "[medium]" not in high_only.out
    assert reported_count(high_only.out) < reported_count(everything.out)


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--select", "PP999"), ("--ignore", "PP999"), ("--select", "PP001,PP999")],
)
def test_unknown_rule_id_is_a_usage_error(cli: Cli, flag: str, value: str) -> None:
    result = cli("lint", CLEAN_PP003, flag, value)
    assert_usage_error(result, "PP999")
    assert len(result.err.splitlines()) == 1


def test_lint_of_a_missing_path_is_a_usage_error(cli: Cli, tmp_path: Path) -> None:
    assert_usage_error(cli("lint", tmp_path / "missing.py"), "missing.py")


def test_config_flag_reads_the_given_pyproject_and_flags_override_it(
    cli: Cli, tmp_path: Path
) -> None:
    source = write(tmp_path / "src" / "shift.py", HIGH)
    config = write(tmp_path / "cfg" / "pyproject.toml", '[tool.peekproof]\nignore = ["PP001"]\n')
    assert rules_in(cli("lint", source).out) == ["PP001"]  # no configuration on the way up
    configured = cli("lint", source, "--config", config)
    assert (configured.code, configured.out) == (0, "peekproof: 0 findings\n")
    overridden = cli("lint", source, "--config", config, "--ignore", "PP003")
    assert rules_in(overridden.out) == ["PP001"]  # the flag replaces the configured value


def test_config_flag_with_a_missing_file_is_a_usage_error(cli: Cli, tmp_path: Path) -> None:
    result = cli("lint", CLEAN_PP003, "--config", tmp_path / "nope" / "pyproject.toml")
    assert_usage_error(result, "pyproject.toml")


# --------------------------------------------------------------------------- #
# check: verdicts and reports
# --------------------------------------------------------------------------- #


def test_check_reports_a_causal_function_and_exits_zero(cli: Cli, project: Path) -> None:
    result = cli("check", "features:causal", *CSV)
    assert result.code == 0
    assert result.err == ""
    assert result.out.startswith("peekproof: CAUSAL")
    assert "NOT CAUSAL" not in result.out


def test_check_reports_a_leaky_function_and_exits_one(cli: Cli, project: Path) -> None:
    result = cli("check", "features:leaky", *CSV)
    assert result.code == 1
    assert result.err == ""
    assert result.out.startswith("peekproof: NOT CAUSAL")
    assert "column 'close'" in result.out
    assert "horizon = 9 row(s)" in result.out


def test_check_text_is_the_default_format(cli: Cli, project: Path) -> None:
    text = cli("check", "features:leaky", *CSV, "--format", "text")
    assert text == cli("check", "features:leaky", *CSV)


def test_check_imports_the_target_from_the_working_directory(cli: Cli, project: Path) -> None:
    cli("check", "features:causal", *CSV)
    assert Path(sys.modules["features"].__file__).samefile(project / "features.py")


def test_check_json_for_a_leaky_function(cli: Cli, project: Path) -> None:
    result = cli("check", "features:leaky", *CSV, "--format", "json")
    report = json.loads(result.out)
    assert result.code == 1
    assert isinstance(report, dict)
    assert {"causal", "deterministic", "horizon_rows", "findings"} <= set(report)
    assert report["causal"] is False
    assert report["deterministic"] is True
    assert report["horizon_rows"] == 9
    assert report["horizon_unbounded"] is False
    assert report["findings"]
    for finding in report["findings"]:
        assert set(finding) == LEAK_FINDING_KEYS
        assert finding["column"] == "close"
        assert finding["cutoff_label"].startswith("2021-")
        assert finding["first_affected_label"].startswith("2021-")


def test_check_json_for_a_causal_function(cli: Cli, project: Path) -> None:
    result = cli("check", "features:causal", *CSV, "--format", "json")
    report = json.loads(result.out)
    assert result.code == 0
    assert report["causal"] is True
    assert report["deterministic"] is True
    assert report["findings"] == []
    assert report["notes"] == []
    assert report["horizon_rows"] is None
    assert report["n_rows"] == N_ROWS
    # 64 cutoffs per test by default; both tests choose their own positions
    assert len(report["cutoff_positions"]) >= 64
    assert report["n_evaluations"] == 2 + 2 * 64  # determinism check, then both tests per cutoff


def test_check_exits_one_for_a_function_that_is_not_deterministic(cli: Cli, project: Path) -> None:
    # DESIGN section 7 does not say; a run that could not show causality must not exit 0
    text = cli("check", "features:flaky", *CSV)
    assert text.code == 1
    assert not text.out.startswith("peekproof: CAUSAL")
    assert "function is not deterministic" in text.out
    as_json = cli("check", "features:flaky", *CSV, "--format", "json")
    report = json.loads(as_json.out)
    assert as_json.code == 1
    assert (report["causal"], report["deterministic"]) == (None, False)


@pytest.mark.parametrize(
    ("target", "label_flags"),
    [
        pytest.param("features:leaky", [], id="plain"),
        # the label looks 5 rows ahead, so a horizon of 4 is a leak
        pytest.param("features:forward_return", ["--label-horizon", "4"], id="label-mode"),
    ],
)
@pytest.mark.parametrize(
    ("mode", "tests", "evaluations"),
    [
        pytest.param("truncate", {"truncate"}, 2 + 4, id="truncate"),
        pytest.param("perturb", {"perturb"}, 2 + 4, id="perturb"),
        pytest.param("both", {"truncate", "perturb"}, 2 + 2 * 4, id="both"),
    ],
)
def test_cutoffs_and_mode_reach_the_oracle(
    cli: Cli,
    project: Path,
    target: str,
    label_flags: list[str],
    mode: str,
    tests: set[str],
    evaluations: int,
) -> None:
    result = cli(
        "check", target, *CSV, *label_flags, "--cutoffs", "4", "--mode", mode, "--format", "json"
    )
    report = json.loads(result.out)
    assert result.code == 1
    assert len(report["cutoff_positions"]) >= 4  # union of the positions of the tests that ran
    assert report["n_evaluations"] == evaluations
    assert {finding["test"] for finding in report["findings"]} == tests


# --------------------------------------------------------------------------- #
# check: label mode
# --------------------------------------------------------------------------- #


def test_forward_label_is_a_leak_without_a_label_horizon(cli: Cli, project: Path) -> None:
    result = cli("check", "features:forward_return", *CSV)
    assert result.code == 1
    assert "horizon = 5 row(s)" in result.out


def test_label_horizon_equal_to_the_look_ahead_passes(cli: Cli, project: Path) -> None:
    result = cli("check", "features:forward_return", *CSV, "--label-horizon", "5")
    assert result.code == 0
    assert result.out.startswith("peekproof: CAUSAL")


def test_label_horizon_below_the_look_ahead_fails(cli: Cli, project: Path) -> None:
    result = cli("check", "features:forward_return", *CSV, "--label-horizon", "4")
    assert result.code == 1
    assert result.out.startswith("peekproof: NOT CAUSAL")
    assert "horizon = 5 row(s)" in result.out


# --------------------------------------------------------------------------- #
# check: data files
# --------------------------------------------------------------------------- #


def test_time_col_reads_a_csv_where_the_date_is_a_regular_column(
    cli: Cli, project: Path, prices: pd.DataFrame
) -> None:
    prices.reset_index().to_csv(project / "flat.csv", index=False)
    assert_verdicts(cli, "--data", "flat.csv", "--time-col", "date")


def test_time_col_sorts_the_rows_by_time(cli: Cli, project: Path, prices: pd.DataFrame) -> None:
    flat = prices.reset_index()
    flat.to_csv(project / "ordered.csv", index=False)
    flat.sample(frac=1.0, random_state=0).to_csv(project / "shuffled.csv", index=False)
    ordered = cli("check", "features:leaky", "--data", "ordered.csv", "--time-col", "date")
    shuffled = cli("check", "features:leaky", "--data", "shuffled.csv", "--time-col", "date")
    assert shuffled.code == 1
    assert "horizon = 9 row(s)" in shuffled.out
    assert shuffled.out == ordered.out


def test_check_reads_a_parquet_file_with_a_time_index(
    cli: Cli, project: Path, prices: pd.DataFrame
) -> None:
    prices.to_parquet(project / "prices.parquet")
    assert_verdicts(cli, "--data", "prices.parquet")


@pytest.mark.parametrize("flag", ["--time-col", "--index-col"])
def test_check_reads_a_parquet_file_with_a_date_column(
    cli: Cli, project: Path, prices: pd.DataFrame, flag: str
) -> None:
    prices.reset_index().to_parquet(project / "flat.parquet", index=False)
    assert_verdicts(cli, "--data", "flat.parquet", flag, "date")


# --------------------------------------------------------------------------- #
# check: errors that exit with status 2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        pytest.param(["features", *CSV], "MODULE:FUNCTION", id="target-without-colon"),
        pytest.param(["no_such_module:causal", *CSV], "no_such_module", id="missing-module"),
        pytest.param(["features:THRESHOLD", *CSV], "THRESHOLD", id="attribute-not-callable"),
        pytest.param(["features:no_such_name", *CSV], "no_such_name", id="missing-attribute"),
        pytest.param(
            ["features:causal", "--data", "missing.csv"], "missing.csv", id="missing-data-file"
        ),
        pytest.param(
            ["features:causal", "--data", "notes.txt"], ".txt", id="unsupported-extension"
        ),
        pytest.param(
            ["features:causal", "--data", "prices.csv", "--time-col", "when"],
            "when",
            id="missing-time-col",
        ),
        pytest.param(["features:broken", *CSV], "bad input", id="function-raises"),
        pytest.param(["features:not_a_frame", *CSV], "DataFrame", id="function-returns-a-number"),
        pytest.param(["features:causal", *CSV, "--cutoffs", "0"], "cutoffs", id="zero-cutoffs"),
        pytest.param(
            ["features:causal", *CSV, "--label-horizon", "-1"], "horizon", id="negative-horizon"
        ),
    ],
)
def test_check_errors_exit_with_status_two(
    cli: Cli, project: Path, argv: list[str], fragment: str
) -> None:
    write(project / "notes.txt", "not data\n")
    assert_usage_error(cli("check", *argv), fragment)


# --------------------------------------------------------------------------- #
# check: unreadable data and failing imports are usage errors (status 2)
# --------------------------------------------------------------------------- #


def test_unknown_index_col_in_a_csv_is_a_usage_error(cli: Cli, project: Path) -> None:
    result = cli("check", "features:causal", "--data", "prices.csv", "--index-col", "when")
    assert_usage_error(result, "when")


def test_unknown_index_col_in_a_parquet_file_is_a_usage_error(
    cli: Cli, project: Path, prices: pd.DataFrame
) -> None:
    prices.reset_index().to_parquet(project / "flat.parquet", index=False)
    result = cli("check", "features:causal", "--data", "flat.parquet", "--index-col", "when")
    assert_usage_error(result, "when")


def test_empty_csv_file_is_a_usage_error(cli: Cli, project: Path) -> None:
    write(project / "empty.csv", "")
    assert_usage_error(cli("check", "features:causal", "--data", "empty.csv"))


def test_file_that_is_not_parquet_is_a_usage_error(cli: Cli, project: Path) -> None:
    write(project / "broken.parquet", "this is not a parquet file\n")
    assert_usage_error(cli("check", "features:causal", "--data", "broken.parquet"))


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "def run(:\n",
            id="syntax-error",
        ),
        pytest.param(
            "raise RuntimeError('boom')\n",
            id="exception-on-import",
        ),
    ],
)
def test_module_that_fails_to_import_is_a_usage_error(cli: Cli, project: Path, source: str) -> None:
    write_module(project, "faulty", source)
    assert_usage_error(cli("check", "faulty:run", *CSV), "faulty")
