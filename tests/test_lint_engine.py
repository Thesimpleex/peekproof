"""Tests for the lint engine.

Covers suppression comments, label functions, notebooks, configuration from
``pyproject.toml``, filtering by rule and confidence, parse failures and file discovery.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import peekproof.lint
from peekproof.errors import PeekproofError
from peekproof.lint import RULES, lint_path, lint_source
from peekproof.lint.engine import (
    DEFAULT_LABEL_FUNCTIONS,
    LintConfig,
    LintContext,
    find_config,
    load_config,
)
from peekproof.report import Finding

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_FIXTURE = FIXTURES / "pyproject_config"

LEAK = "z = x.shift(-1)\n"  # PP001, high
BFILL = "z = x.bfill()\n"  # PP002, high
CONVOLVE = "w = np.convolve(x, k, mode='same')\n"  # PP003, medium
BARS = "bars = t.resample('5min').last()\nr = bars.pct_change()\n"  # PP008, low


def ids(findings: list[Finding]) -> list[str]:
    return [f.rule_id for f in findings]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def project(root: Path, options: str = "") -> Path:
    """Create a pyproject.toml with a [tool.peekproof] table so that no outer file is found."""
    write(root / "pyproject.toml", "[tool.peekproof]\n" + textwrap.dedent(options))
    return root


def notebook(*cells: tuple[str, str | list[str]]) -> str:
    """Build the JSON text of an nbformat 4 notebook from ``(cell_type, source)`` pairs."""
    built = []
    for cell_type, source in cells:
        cell: dict[str, object] = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell.update(execution_count=None, outputs=[])
        built.append(cell)
    return json.dumps({"cells": built, "metadata": {}, "nbformat": 4, "nbformat_minor": 4})


# --------------------------------------------------------------------------- #
# Suppression comments
# --------------------------------------------------------------------------- #


def test_bare_ignore_silences_every_rule_on_its_line() -> None:
    assert lint_source("z = x.shift(-1).bfill()  # peekproof: ignore\n") == []


def test_ignore_with_a_rule_list_silences_only_the_listed_rules() -> None:
    source = "z = x.shift(-1).bfill()  # peekproof: ignore[PP001]\n"
    assert ids(lint_source(source)) == ["PP002"]


def test_ignore_list_takes_several_ids_in_any_case() -> None:
    assert lint_source("z = x.shift(-1).bfill()  # peekproof: ignore[PP001, PP002]\n") == []
    assert lint_source("z = x.shift(-1).bfill()  # peekproof: ignore[pp001,pp002]\n") == []
    assert ids(lint_source("z = x.shift(-1)  # peekproof: ignore[PP002]\n")) == ["PP001"]


def test_ignore_comment_covers_its_own_line_only() -> None:
    source = "z = x.shift(-1)  # peekproof: ignore\nz = x.shift(-1)\n"
    assert [f.line for f in lint_source(source)] == [2]


def test_ignore_comment_must_sit_on_the_line_of_the_finding() -> None:
    elsewhere = "z = (  # peekproof: ignore\n    x\n    .shift(-1)\n)\n"
    assert [f.line for f in lint_source(elsewhere)] == [3]
    same_line = "z = (\n    x\n    .shift(-1)  # peekproof: ignore\n)\n"
    assert lint_source(same_line) == []


def test_directive_text_inside_a_string_is_not_a_comment() -> None:
    source = 's = "# peekproof: ignore"; z = x.shift(-1)\n'
    assert ids(lint_source(source)) == ["PP001"]


def test_ignore_comment_may_share_the_comment_with_other_text() -> None:
    source = "z = x.shift(-1)  # noqa: E501  # peekproof: ignore  (known, see ticket)\n"
    assert lint_source(source) == []


def test_ignore_comment_is_case_insensitive_in_its_keyword() -> None:
    assert lint_source("z = x.shift(-1)  # Peekproof: Ignore\n") == []


# --------------------------------------------------------------------------- #
# Label functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "forward_return",
        "forward_returns",
        "forward_return_5d",
        "FORWARD_RETURN",
        "fwd_ret",
        "fwd_ret_5",
        "make_label",
        "make_labels",
        "get_target",
        "build_targets",
        "label",
        "target",
        "get_y",
        "y",
    ],
)
def test_function_names_matching_the_default_pattern_are_label_functions(name: str) -> None:
    source = f"def {name}(prices):\n    return prices.shift(-5)\n"
    assert lint_source(source) == []


@pytest.mark.parametrize(
    "name",
    [
        "target_position",
        "target_weights",
        "forward_fill_features",
        "forward_looking_features",
        "fwd_features",
        "labeled_rows",
        "label_encoder",
        "targets_first",
        "labels_5d",
        "yearly",
        "make_features",
    ],
)
def test_a_label_word_inside_a_name_does_not_make_a_label_function(name: str) -> None:
    source = f"def {name}(prices):\n    return prices.shift(-5)\n"
    assert ids(lint_source(source)) == ["PP001"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("make_label", True),
        ("get_Target", True),
        ("forward_return_20", True),
        ("fwd_rets", True),
        ("compute_y", True),
        ("target_position", False),
        ("forward_fill_features", False),
        ("forward_returns5", False),
        ("backward_return", False),
    ],
)
def test_default_label_pattern_is_the_documented_regex(name: str, expected: bool) -> None:
    assert bool(re.search(DEFAULT_LABEL_FUNCTIONS, name, re.IGNORECASE)) is expected


def test_default_label_pattern_is_what_lint_source_and_the_config_start_from() -> None:
    assert LintConfig().label_functions == DEFAULT_LABEL_FUNCTIONS
    assert load_config(CONFIG_FIXTURE / "pyproject.toml").label_functions == DEFAULT_LABEL_FUNCTIONS


def test_marker_comment_on_the_def_line_makes_a_label_function() -> None:
    source = "def future_move(prices):  # peekproof: label\n    return prices.shift(-5)\n"
    assert lint_source(source) == []


def test_marker_comment_may_sit_on_the_last_line_of_a_multi_line_signature() -> None:
    source = (
        "def future_move(\n    prices,\n    horizon=5,\n):  # peekproof: label\n"
        "    return prices.shift(-5)\n"
    )
    assert lint_source(source) == []


def test_functions_without_marker_or_matching_name_are_not_exempt() -> None:
    source = "def make_features(prices):\n    return prices.shift(-5)\n"
    assert ids(lint_source(source)) == ["PP001"]


def test_marker_comment_elsewhere_in_the_body_does_not_count() -> None:
    source = "def make_features(prices):\n    return prices.shift(-5)  # peekproof: label\n"
    assert ids(lint_source(source)) == ["PP001"]


def test_exemption_covers_nested_functions_and_methods() -> None:
    nested = (
        "def make_target(x):\n    def inner(y):\n        return y.shift(-1)\n    return inner(x)\n"
    )
    assert lint_source(nested) == []
    method = "class Builder:\n    def target(self, x):\n        return x.shift(-1)\n"
    assert lint_source(method) == []
    coroutine = "async def get_label(x):\n    return x.shift(-1)\n"
    assert lint_source(coroutine) == []


def test_exemption_ends_with_the_function() -> None:
    source = (
        "def forward_return(x):\n    return x.shift(-5)\n\n\n"
        "def features(x):\n    return x.shift(-5)\n"
    )
    assert [f.line for f in lint_source(source)] == [6]


def test_label_functions_are_exempt_from_pp001_pp009_and_pp011_only() -> None:
    source = "def forward_return(x):\n    a = x.shift(-5)\n    return a.bfill()\n"
    assert ids(lint_source(source)) == ["PP002"]
    source = (
        "def make_label(x, df, n):\n"
        "    a = x.shift(-5)\n"
        "    out = [x[i + 1] for i in range(n)]\n"
        "    df['f'] = df['target'] * 2\n"
        "    return a.bfill(), out\n"
    )
    assert ids(lint_source(source)) == ["PP002"]
    unmarked = source.replace("make_label", "make_features")
    assert sorted(ids(lint_source(unmarked))) == ["PP001", "PP002", "PP009", "PP011"]


def test_assigning_to_a_label_name_is_label_code_in_any_function() -> None:
    source = (
        "def make_features(prices):\n"
        "    out = {}\n"
        "    out['target'] = prices.shift(-1) / prices - 1\n"
        "    out['next'] = prices.shift(-1)\n"
        "    return out\n"
    )
    assert [f.line for f in lint_source(source)] == [4]


def test_label_function_pattern_can_be_changed() -> None:
    source = (
        "def make_y(x):\n    return x.shift(-1)\n\n\n"
        "def forward_return(x):\n    return x.shift(-5)\n"
    )
    assert [f.line for f in lint_source(source, label_functions=r"^make_y$")] == [6]


def test_invalid_label_function_pattern_is_rejected() -> None:
    with pytest.raises(PeekproofError, match="label_functions"):
        lint_source(LEAK, label_functions="(unclosed")


def test_context_exposes_label_function_spans() -> None:
    source = "a = 1\n\ndef forward_return(x):\n    y = x\n    return y.shift(-5)\n\nb = 2\n"
    ctx = LintContext.build(source)
    assert ctx.label_spans == [(3, 5)]
    assert ctx.in_label_function(4)
    assert not ctx.in_label_function(1)
    assert not ctx.in_label_function(7)


# --------------------------------------------------------------------------- #
# Notebooks
# --------------------------------------------------------------------------- #


def test_notebook_fixture_reports_cell_path_and_cell_relative_line() -> None:
    path = FIXTURES / "notebook_leaky.ipynb"
    (finding,) = lint_path(path)
    assert finding.rule_id == "PP002"
    assert finding.path == f"{path}#cell2"
    # Line 1 of the cell is the %matplotlib magic, so the call is on line 2 of the cell.
    assert finding.line == 2
    assert finding.snippet == "filled = prices.bfill()"


def test_notebook_fixture_has_two_code_cells_with_the_magic_in_the_second() -> None:
    data = json.loads((FIXTURES / "notebook_leaky.ipynb").read_text(encoding="utf-8"))
    cells = data["cells"]
    assert [c["cell_type"] for c in cells] == ["code", "code"]
    assert "".join(cells[1]["source"]).startswith("%matplotlib inline")


def test_lint_source_reads_a_notebook_when_the_path_ends_in_ipynb() -> None:
    text = (FIXTURES / "notebook_leaky.ipynb").read_text(encoding="utf-8")
    (finding,) = lint_source(text, "demo.ipynb")
    assert (finding.path, finding.line) == ("demo.ipynb#cell2", 2)


def test_cell_numbers_count_all_cells_and_lines_are_relative_to_the_cell() -> None:
    text = notebook(
        ("markdown", "# Title\n"),
        ("code", "import pandas as pd\n\nprices = pd.Series([1.0])\n"),
        ("raw", "z = x.shift(-1)"),
        ("code", "a = 1\nb = 2\nc = a\nd = x.shift(-1)\n"),
    )
    findings = lint_source(text, "nb.ipynb")
    assert [(f.path, f.line) for f in findings] == [("nb.ipynb#cell4", 4)]


def test_markdown_and_raw_cells_are_never_linted() -> None:
    text = notebook(("markdown", "z = x.shift(-1)"), ("raw", "z = x.bfill()"))
    assert lint_source(text, "nb.ipynb") == []


def test_cell_source_may_be_a_list_of_lines_or_one_string() -> None:
    as_list = notebook(("code", ["a = 1\n", "z = x.shift(-1)\n"]))
    as_string = notebook(("code", "a = 1\nz = x.shift(-1)\n"))
    assert [f.line for f in lint_source(as_list, "nb.ipynb")] == [2]
    assert [f.line for f in lint_source(as_string, "nb.ipynb")] == [2]


@pytest.mark.parametrize(
    "source",
    [
        "%matplotlib inline\n{leak}",
        "!pip install pandas\n{leak}",
        "%%time\n{leak}",
        "%load_ext autoreload\n%autoreload 2\n{leak}",
        "!pip install \\\n    pandas\n{leak}",
        "files = !ls data\n{leak}",
        "df.describe?\n{leak}",
        "?df\n{leak}",
    ],
)
def test_ipython_syntax_is_blanked_without_moving_lines(source: str) -> None:
    code = source.format(leak="z = x.shift(-1)\n")
    findings = lint_source(notebook(("code", code)), "nb.ipynb")
    assert ids(findings) == ["PP001"]
    assert findings[0].line == code.count("\n")  # the leak is on the last line of the cell


def test_indented_magic_keeps_its_block_valid() -> None:
    code = "if True:\n    %time fit()\nz = x.shift(-1)\n"
    assert [f.line for f in lint_source(notebook(("code", code)), "nb.ipynb")] == [3]


def test_cells_that_parse_as_plain_python_are_left_alone() -> None:
    # The second line starts with % and with != but continues an expression: not a magic.
    code = "a = (b\n     % c)\nd = (e\n     != f)\nz = x.shift(-1)\n"
    findings = lint_source(notebook(("code", code)), "nb.ipynb")
    assert [(f.rule_id, f.line) for f in findings] == [("PP001", 5)]


def test_cells_with_a_non_python_cell_magic_are_skipped() -> None:
    text = notebook(("code", "%%bash\nls -l | wc -l\necho 'hello world'\n"))
    assert lint_source(text, "nb.ipynb") == []
    text = notebook(("code", "%%sql\nSELECT * FROM prices;\n"))
    assert lint_source(text, "nb.ipynb") == []


def test_cells_with_a_python_cell_magic_are_linted() -> None:
    text = notebook(("code", "%%timeit\nz = x.shift(-1)\n"))
    assert [(f.rule_id, f.line) for f in lint_source(text, "nb.ipynb")] == [("PP001", 2)]


def test_a_cell_with_a_syntax_error_is_reported_and_the_others_are_still_linted() -> None:
    text = notebook(("code", "def broken(:\n"), ("code", "a = 1\nz = x.shift(-1)\n"))
    findings = lint_source(text, "nb.ipynb")
    assert [(f.rule_id, f.path, f.line) for f in findings] == [
        ("PP000", "nb.ipynb#cell1", 1),
        ("PP001", "nb.ipynb#cell2", 2),
    ]


def test_suppression_and_label_functions_work_inside_notebook_cells() -> None:
    text = notebook(
        ("code", "z = x.shift(-1)  # peekproof: ignore\n"),
        ("code", "def forward_return(x):\n    return x.shift(-5)\n"),
    )
    assert lint_source(text, "nb.ipynb") == []


@pytest.mark.parametrize(
    "metadata",
    [
        {"language_info": {"name": "R"}},
        {"kernelspec": {"language": "julia", "name": "julia-1.10", "display_name": "Julia"}},
    ],
)
def test_notebooks_in_another_language_are_skipped(metadata: dict[str, object]) -> None:
    text = json.loads(notebook(("code", "df %>% filter(x > 1)\n")))
    text["metadata"] = metadata
    assert lint_source(json.dumps(text), "nb.ipynb") == []


def test_notebooks_that_name_python_or_nothing_are_linted() -> None:
    base = json.loads(notebook(("code", LEAK)))
    for metadata in (
        {},
        {"language_info": {"name": "python"}},
        {"kernelspec": {"language": "Python"}},
    ):
        base["metadata"] = metadata
        assert ids(lint_source(json.dumps(base), "nb.ipynb")) == ["PP001"]


def test_notebook_that_is_not_json_gives_one_pp000() -> None:
    (finding,) = lint_source("{not json", "nb.ipynb")
    assert (finding.rule_id, finding.confidence, finding.path) == ("PP000", "low", "nb.ipynb")


@pytest.mark.parametrize("text", ['{"cells": 3}', '{"worksheets": []}', "[]", '"text"'])
def test_notebook_without_a_cell_list_gives_one_pp000(text: str) -> None:
    assert ids(lint_source(text, "nb.ipynb")) == ["PP000"]


def test_notebook_files_are_found_and_linted_by_lint_path(tmp_path: Path) -> None:
    project(tmp_path)
    (tmp_path / "nb.ipynb").write_text(notebook(("code", BFILL)), encoding="utf-8")
    (finding,) = lint_path(tmp_path)
    assert finding.path == f"{tmp_path / 'nb.ipynb'}#cell1"


def test_notebook_file_with_a_utf8_bom_is_read(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "nb.ipynb"
    path.write_bytes(b"\xef\xbb\xbf" + notebook(("code", BFILL)).encode("utf-8"))
    assert ids(lint_path(path)) == ["PP002"]


# --------------------------------------------------------------------------- #
# Configuration from pyproject.toml
# --------------------------------------------------------------------------- #

CONFIG_SAMPLE = CONFIG_FIXTURE / "sample.py"


def test_config_fixture_would_trigger_pp001_and_pp006_without_a_config() -> None:
    source = CONFIG_SAMPLE.read_text(encoding="utf-8")
    assert sorted(set(ids(lint_source(source)))) == ["PP001", "PP006"]


def test_ignore_is_read_from_the_nearest_pyproject_toml() -> None:
    assert find_config(CONFIG_FIXTURE) == (CONFIG_FIXTURE / "pyproject.toml").resolve()
    findings = lint_path(CONFIG_FIXTURE)
    assert ids(findings) == ["PP001"]


def test_config_is_found_from_a_file_path_too() -> None:
    assert ids(lint_path(CONFIG_SAMPLE)) == ["PP001"]


def test_arguments_replace_the_configured_ignore_list() -> None:
    assert ids(lint_path(CONFIG_SAMPLE, ignore=["PP001"])) == ["PP006"]
    assert sorted(ids(lint_path(CONFIG_SAMPLE, ignore=["PP013"]))) == ["PP001", "PP006"]


def test_explicit_config_path_accepts_a_file_or_a_folder(tmp_path: Path) -> None:
    config = project(tmp_path / "cfg", 'ignore = ["PP001"]\n')
    target = write(tmp_path / "elsewhere" / "a.py", LEAK + BFILL)
    assert ids(lint_path(target, config_path=config / "pyproject.toml")) == ["PP002"]
    assert ids(lint_path(target, config_path=config)) == ["PP002"]


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PeekproofError, match="not found"):
        lint_path(tmp_path, config_path=tmp_path / "nope" / "pyproject.toml")


def test_the_nearest_config_wins(tmp_path: Path) -> None:
    project(tmp_path, 'ignore = ["PP001"]\n')
    project(tmp_path / "sub", 'ignore = ["PP002"]\n')
    target = write(tmp_path / "sub" / "a.py", LEAK + BFILL)
    assert ids(lint_path(target)) == ["PP001"]
    outer = write(tmp_path / "b.py", LEAK + BFILL)
    assert ids(lint_path(outer)) == ["PP002"]


def test_a_pyproject_without_a_peekproof_table_is_skipped(tmp_path: Path) -> None:
    project(tmp_path, 'ignore = ["PP001"]\n')
    write(tmp_path / "sub" / "pyproject.toml", '[project]\nname = "sub"\n')
    target = write(tmp_path / "sub" / "a.py", LEAK + BFILL)
    assert find_config(target) == (tmp_path / "pyproject.toml").resolve()
    assert ids(lint_path(target)) == ["PP002"]


def test_no_config_means_defaults(tmp_path: Path) -> None:
    config = write(tmp_path / "pyproject.toml", '[project]\nname = "x"\n')
    assert find_config(config) != config  # a pyproject.toml without the table is not a config
    cfg = load_config(config)
    assert cfg.select is None
    assert cfg.ignore == ()
    assert cfg.min_confidence == "low"
    assert cfg.exclude == ()


def test_config_select(tmp_path: Path) -> None:
    project(tmp_path, 'select = ["PP002"]\n')
    target = write(tmp_path / "a.py", LEAK + BFILL)
    assert ids(lint_path(target)) == ["PP002"]
    assert ids(lint_path(target, select=["PP001"])) == ["PP001"]


def test_config_min_confidence(tmp_path: Path) -> None:
    project(tmp_path, 'min_confidence = "high"\n')
    target = write(tmp_path / "a.py", LEAK + CONVOLVE + BARS)
    assert ids(lint_path(target)) == ["PP001"]
    assert sorted(ids(lint_path(target, min_confidence="low"))) == ["PP001", "PP003", "PP008"]


def test_config_label_functions(tmp_path: Path) -> None:
    project(tmp_path, 'label_functions = "^outcome_"\n')
    target = write(
        tmp_path / "a.py",
        """
        def outcome_5d(x):
            return x.shift(-5)


        def forward_return(x):
            return x.shift(-5)
        """,
    )
    assert [f.line for f in lint_path(target)] == [7]


def test_config_accepts_kebab_case_keys(tmp_path: Path) -> None:
    project(tmp_path, 'min-confidence = "high"\nlabel-functions = "^outcome_"\n')
    target = write(tmp_path / "a.py", LEAK + CONVOLVE)
    assert ids(lint_path(target)) == ["PP001"]


def test_config_values_may_be_comma_separated_strings_and_prefixes(tmp_path: Path) -> None:
    project(tmp_path, 'ignore = "PP001, PP00"\n')
    target = write(tmp_path / "a.py", LEAK + BFILL)
    assert lint_path(target) == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ('ignor = ["PP001"]', "unknown option 'ignor'"),
        ('min_confidence = "certain"', "min_confidence"),
        ("min_confidence = 3", "min_confidence"),
        ('ignore = ["PP999"]', "unknown rule selector"),
        ("select = [1]", "strings"),
        ('label_functions = "(oops"', "label_functions"),
        ("label_functions = 5", "label_functions"),
        ("exclude = [1, 2]", "exclude"),
    ],
)
def test_invalid_config_is_reported_with_the_file_and_the_option(
    tmp_path: Path, options: str, message: str
) -> None:
    project(tmp_path, options + "\n")
    target = write(tmp_path / "a.py", LEAK)
    with pytest.raises(PeekproofError, match=message) as excinfo:
        lint_path(target)
    assert "pyproject.toml" in str(excinfo.value)


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    write(tmp_path / "pyproject.toml", "[tool.peekproof\nignore = ")
    target = write(tmp_path / "a.py", LEAK)
    with pytest.raises(PeekproofError, match="invalid TOML"):
        lint_path(target)


def test_peekproof_entry_that_is_not_a_table_is_reported(tmp_path: Path) -> None:
    write(tmp_path / "pyproject.toml", "[tool]\npeekproof = 3\n")
    with pytest.raises(PeekproofError, match="must be a table"):
        load_config(tmp_path / "pyproject.toml")


def test_exclude_globs_skip_walked_files_but_not_files_named_by_the_caller(tmp_path: Path) -> None:
    project(tmp_path, 'exclude = ["extern", "legacy/*.py", "*_pb2.py", "build/"]\n')
    for name in (
        "keep.py",
        "extern/a.py",
        "legacy/old.py",
        "legacy/sub/deep.py",
        "api_pb2.py",
        "pkg/extern/b.py",
        "build/c.py",
        "pkg/ok.py",
    ):
        write(tmp_path / name, LEAK)
    walked = sorted(Path(f.path).relative_to(tmp_path).as_posix() for f in lint_path(tmp_path))
    assert walked == ["keep.py", "pkg/ok.py"]
    assert len(lint_path(tmp_path / "extern" / "a.py")) == 1


def test_a_folder_named_by_the_caller_is_linted_even_if_a_glob_matches_it(tmp_path: Path) -> None:
    project(tmp_path, 'exclude = ["extern", "extern/vendor", "*_pb2.py"]\n')
    write(tmp_path / "extern" / "a.py", LEAK)
    write(tmp_path / "extern" / "sub" / "b.py", LEAK)
    write(tmp_path / "extern" / "vendor" / "c.py", LEAK)
    write(tmp_path / "extern" / "api_pb2.py", LEAK)
    walked = sorted(
        Path(f.path).relative_to(tmp_path).as_posix() for f in lint_path(tmp_path / "extern")
    )
    assert walked == ["extern/a.py", "extern/sub/b.py"]
    assert lint_path(tmp_path) == []


def test_exclude_understands_double_star_and_a_leading_dot_slash(tmp_path: Path) -> None:
    project(tmp_path, 'exclude = ["tests/**/fixtures", "./scratch", "**/skip_*.py"]\n')
    for name in (
        "keep.py",
        "tests/fixtures/a.py",
        "tests/unit/fixtures/b.py",
        "tests/unit/test_c.py",
        "scratch/d.py",
        "pkg/skip_e.py",
        "skip_f.py",
    ):
        write(tmp_path / name, LEAK)
    walked = sorted(Path(f.path).relative_to(tmp_path).as_posix() for f in lint_path(tmp_path))
    assert walked == ["keep.py", "tests/unit/test_c.py"]


# --------------------------------------------------------------------------- #
# Filtering by confidence and by rule
# --------------------------------------------------------------------------- #

MIXED = LEAK + CONVOLVE + BARS


def test_min_confidence_filters_findings_by_level() -> None:
    assert sorted(ids(lint_source(MIXED))) == ["PP001", "PP003", "PP008"]
    assert sorted(ids(lint_source(MIXED, min_confidence="low"))) == ["PP001", "PP003", "PP008"]
    assert sorted(ids(lint_source(MIXED, min_confidence="medium"))) == ["PP001", "PP003"]
    assert ids(lint_source(MIXED, min_confidence="high")) == ["PP001"]


def test_min_confidence_uses_the_confidence_of_each_finding_not_of_the_rule() -> None:
    source = "a = np.convolve(x, k, mode='same')\nb = x.rolling(5, center=True).mean()\n"
    assert ids(lint_source(source, min_confidence="high")) == ["PP003"]  # PP003 is high by default
    assert [f.line for f in lint_source(source, min_confidence="high")] == [2]


def test_unknown_confidence_level_is_rejected() -> None:
    with pytest.raises(PeekproofError, match="min_confidence"):
        lint_source(LEAK, min_confidence="certain")


def test_select_keeps_only_the_chosen_rules() -> None:
    source = LEAK + BFILL + CONVOLVE
    assert ids(lint_source(source, select=["PP002"])) == ["PP002"]
    assert sorted(ids(lint_source(source, select=["PP001", "PP003"]))) == ["PP001", "PP003"]


def test_ignore_drops_the_chosen_rules() -> None:
    source = LEAK + BFILL + CONVOLVE
    assert sorted(ids(lint_source(source, ignore=["PP001"]))) == ["PP002", "PP003"]
    assert lint_source(source, ignore=["PP001", "PP002", "PP003"]) == []


def test_select_and_ignore_accept_comma_strings_prefixes_and_any_case() -> None:
    source = LEAK + BFILL + CONVOLVE
    assert sorted(ids(lint_source(source, select="pp001, PP002"))) == ["PP001", "PP002"]
    assert ids(lint_source(source, select="PP00", ignore="PP002,PP003")) == ["PP001"]


def test_ignore_wins_over_select_for_the_same_rule() -> None:
    assert lint_source(LEAK, select=["PP001"], ignore=["PP001"]) == []


def test_empty_select_means_no_restriction() -> None:
    assert ids(lint_source(LEAK, select=[])) == ["PP001"]
    assert ids(lint_source(LEAK, select="")) == ["PP001"]


def test_unknown_rule_selector_is_rejected() -> None:
    with pytest.raises(PeekproofError, match="PP999"):
        lint_source(LEAK, select=["PP999"])
    with pytest.raises(PeekproofError, match="PP999"):
        lint_source(LEAK, ignore="PP999")


def test_select_and_ignore_apply_to_parse_failures_like_to_any_rule() -> None:
    broken = "def (:\n"
    assert ids(lint_source(broken)) == ["PP000"]
    assert lint_source(broken, select=["PP001"]) == []
    assert ids(lint_source(broken, select=["PP000"])) == ["PP000"]
    assert lint_source(broken, ignore=["PP000"]) == []
    assert lint_source(broken, min_confidence="medium") == []


def test_every_registered_rule_can_be_selected_alone() -> None:
    for rule_id in RULES:
        assert lint_source("x = 1\n", select=[rule_id]) == []


# --------------------------------------------------------------------------- #
# Syntax errors become PP000
# --------------------------------------------------------------------------- #


def test_syntax_error_gives_one_low_confidence_pp000() -> None:
    (finding,) = lint_source("a = 1\ndef (:\n    pass\n", "broken.py")
    assert finding.rule_id == "PP000"
    assert finding.name == "parse-error"
    assert finding.confidence == "low"
    assert finding.path == "broken.py"
    assert finding.line == 2
    assert finding.col >= 1
    assert finding.message.startswith("could not parse")
    assert finding.hint
    assert finding.snippet == "def (:"


def test_null_bytes_are_a_parse_failure_not_a_crash() -> None:
    assert ids(lint_source("x = 1\n\x00y = 2\n")) == ["PP000"]


def test_a_broken_file_does_not_stop_the_run(tmp_path: Path) -> None:
    project(tmp_path)
    write(tmp_path / "a_broken.py", "def (:\n")
    write(tmp_path / "b_good.py", LEAK)
    findings = lint_path(tmp_path)
    assert [(Path(f.path).name, f.rule_id) for f in findings] == [
        ("a_broken.py", "PP000"),
        ("b_good.py", "PP001"),
    ]


def test_warnings_about_the_checked_code_are_not_shown_or_raised() -> None:
    source = 'pattern = "\\d+"\nz = x.shift(-1)\n'  # an invalid escape sequence
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ids(lint_source(source)) == ["PP001"]
    assert caught == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ids(lint_source(source)) == ["PP001"]


def test_pathologically_deep_nesting_does_not_crash() -> None:
    code = "z = (x" + " + 0" * 3000 + ").mean() - x\nz = a.shift(-1)\n"
    found = lint_source(code)
    assert "PP001" in ids(found)  # the other rules still ran


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #


def test_walk_skips_virtualenvs_caches_and_version_control(tmp_path: Path) -> None:
    project(tmp_path)
    write(tmp_path / "src" / "main.py", LEAK)
    for skipped in (
        ".venv/lib/python3.12/site-packages/pkg",
        "site-packages",
        "__pycache__",
        ".git/hooks",
        ".ipynb_checkpoints",
        "node_modules/pkg",
        "tool/.mypy_cache",
    ):
        write(tmp_path / skipped / "bad.py", LEAK)
    write(tmp_path / "myenv" / "pyvenv.cfg", "home = /usr/bin\n")
    write(tmp_path / "myenv" / "lib" / "mod.py", LEAK)
    findings = lint_path(tmp_path)
    assert [Path(f.path).relative_to(tmp_path).as_posix() for f in findings] == ["src/main.py"]


def test_walk_only_reads_python_files_and_notebooks_in_a_stable_order(tmp_path: Path) -> None:
    project(tmp_path)
    write(tmp_path / "b.py", LEAK)
    write(tmp_path / "a.py", LEAK)
    write(tmp_path / "sub" / "c.py", LEAK)
    write(tmp_path / "notes.txt", LEAK)
    write(tmp_path / "data.csv", "z = x.shift(-1)\n")
    write(tmp_path / "sub" / "d.ipynb", notebook(("code", LEAK)))
    order = [
        Path(f.path.split("#")[0]).relative_to(tmp_path).as_posix() for f in lint_path(tmp_path)
    ]
    assert order == ["a.py", "b.py", "sub/c.py", "sub/d.ipynb"]


def test_a_file_named_by_the_caller_is_linted_whatever_its_suffix(tmp_path: Path) -> None:
    project(tmp_path)
    script = write(tmp_path / "run_backtest", LEAK)
    assert ids(lint_path(script)) == ["PP001"]


def test_paths_may_be_a_single_path_or_a_list_and_duplicates_are_dropped(tmp_path: Path) -> None:
    project(tmp_path)
    one = write(tmp_path / "one.py", LEAK)
    two = write(tmp_path / "two.py", BFILL)
    assert [Path(f.path).name for f in lint_path(str(one))] == ["one.py"]
    assert [Path(f.path).name for f in lint_path([one, two])] == ["one.py", "two.py"]
    assert [Path(f.path).name for f in lint_path([tmp_path, one, str(two)])] == ["one.py", "two.py"]


def test_missing_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PeekproofError, match="does not exist"):
        lint_path(tmp_path / "missing.py")


def test_findings_of_a_directory_walk_match_those_of_each_file(tmp_path: Path) -> None:
    project(tmp_path)
    write(tmp_path / "a.py", LEAK)
    write(tmp_path / "b.py", BFILL + CONVOLVE)
    together = lint_path(tmp_path)
    separately = lint_path(tmp_path / "a.py") + lint_path(tmp_path / "b.py")
    assert together == separately


def test_fixture_folders_behave_as_the_rule_tests_expect() -> None:
    leaky = lint_path(FIXTURES / "leaky")
    by_file: dict[str, set[str]] = {}
    for finding in leaky:
        by_file.setdefault(Path(finding.path).stem.upper(), set()).add(finding.rule_id)
    assert by_file == {rule_id: {rule_id} for rule_id in RULES}
    assert lint_path(FIXTURES / "clean") == []


# --------------------------------------------------------------------------- #
# Reading files
# --------------------------------------------------------------------------- #


def test_encoding_declaration_is_honoured(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "latin.py"
    path.write_bytes(b"# -*- coding: latin-1 -*-\nname = '\xe9'\nz = x.shift(-1)\n")
    (finding,) = lint_path(path)
    assert (finding.rule_id, finding.line) == ("PP001", 3)


def test_undecodable_file_gives_pp000(tmp_path: Path) -> None:
    project(tmp_path)
    path = tmp_path / "bad.py"
    path.write_bytes(b"y = '\xff\xfe'\n")
    (finding,) = lint_path(path)
    assert finding.rule_id == "PP000"
    assert "decode" in finding.message


def test_windows_line_endings_and_a_byte_order_mark_are_handled() -> None:
    bom = chr(0xFEFF)
    findings = lint_source(f"{bom}a = 1\r\nb = 2\r\nz = x.shift(-1)\r\n")
    assert [(f.rule_id, f.line) for f in findings] == [("PP001", 3)]
    assert findings[0].snippet == "z = x.shift(-1)"


# --------------------------------------------------------------------------- #
# The linter only reads code
# --------------------------------------------------------------------------- #


def test_linted_code_is_never_imported_or_executed(tmp_path: Path) -> None:
    project(tmp_path)
    sentinel = tmp_path / "ran.txt"
    code = (
        "import this_package_does_not_exist\n"
        f"open({str(sentinel)!r}, 'w').write('x')\n"
        "raise SystemExit(3)\n"
        "z = x.shift(-1)\n"
    )
    assert ids(lint_source(code)) == ["PP001"]
    script = write(tmp_path / "script.py", code)
    assert ids(lint_path(script)) == ["PP001"]
    assert not sentinel.exists()
    assert "this_package_does_not_exist" not in sys.modules


def test_linter_modules_import_only_the_standard_library_and_peekproof() -> None:
    package = Path(peekproof.lint.__file__).parent
    sources = sorted(package.glob("*.py"))
    assert {p.name for p in sources} >= {"__init__.py", "engine.py", "rules.py"}
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top in sys.stdlib_module_names or top == "peekproof", (path.name, name)


# --------------------------------------------------------------------------- #
# LintContext
# --------------------------------------------------------------------------- #

SEGMENT_SOURCE = textwrap.dedent(
    """
    import pandas as pd
    s = "héllo wörld"
    total = (
        df["a"]
        + df["b"]  # comment
    )


    def f(x, *, y=1):
        return [i ** 2 for i in range(x) if i % 2]


    value = d["ключ"].shift(-1)
    """
)


def test_segment_gives_the_same_text_as_ast_get_source_segment() -> None:
    ctx = LintContext.build(SEGMENT_SOURCE)
    checked = 0
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.expr, ast.stmt)):
            assert ctx.segment(node) == ast.get_source_segment(SEGMENT_SOURCE, node)
            checked += 1
    assert checked > 30


def test_context_carries_path_lines_and_source() -> None:
    ctx = LintContext.build("a = 1\nb = 2\n", "demo.py")
    assert ctx.path == "demo.py"
    assert ctx.lines[:2] == ["a = 1", "b = 2"]
    assert ctx.source == "a = 1\nb = 2\n"


def test_context_resolves_import_aliases() -> None:
    source = textwrap.dedent(
        """
        import numpy as np
        import scipy.ndimage as ndi
        from sklearn.model_selection import train_test_split as tts
        a = np.roll(x, 1)
        b = ndi.gaussian_filter(x, 1)
        c = tts(x)
        d = pd.read_html(u)
        e = unknown.thing(x)
        """
    )
    ctx = LintContext.build(source)
    values = [n.value for n in ctx.tree.body[3:] if isinstance(n, ast.Assign)]
    callees = [v.func for v in values if isinstance(v, ast.Call)]
    assert [ctx.qualname(c) for c in callees] == [
        "numpy.roll",
        "scipy.ndimage.gaussian_filter",
        "sklearn.model_selection.train_test_split",
        "pandas.read_html",
        "unknown.thing",
    ]


def test_context_parents_and_enclosing_scope() -> None:
    ctx = LintContext.build("def f(x):\n    y = x.shift(1)\n    return y\n")
    call = next(n for n in ast.walk(ctx.tree) if isinstance(n, ast.Call))
    assert isinstance(ctx.parent(call), ast.Assign)
    assert isinstance(ctx.enclosing_scope(call), ast.FunctionDef)
    assert ctx.parent(ctx.tree) is None


# --------------------------------------------------------------------------- #
# A broken pyproject.toml only matters if it holds the peekproof configuration
# --------------------------------------------------------------------------- #

BROKEN_TOML = "[project\nname = 'oops'\n= 3\n"


def test_a_broken_pyproject_in_a_parent_folder_is_ignored(tmp_path: Path) -> None:
    write(tmp_path / "pyproject.toml", BROKEN_TOML)
    target = write(tmp_path / "pkg" / "sub" / "a.py", LEAK)
    assert find_config(target) is None
    assert ids(lint_path(target)) == ["PP001"]
    assert ids(lint_path(tmp_path)) == ["PP001"]  # the folder that holds the broken file
    assert ids(lint_path(tmp_path / "pkg")) == ["PP001"]


def test_a_broken_pyproject_next_to_the_file_is_ignored(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "pyproject.toml", BROKEN_TOML)
    target = write(tmp_path / "pkg" / "a.py", LEAK)
    assert ids(lint_path(target)) == ["PP001"]


def test_the_search_goes_on_past_a_broken_pyproject_to_a_valid_config(tmp_path: Path) -> None:
    project(tmp_path, 'ignore = ["PP001"]\n')
    write(tmp_path / "pkg" / "pyproject.toml", BROKEN_TOML)
    target = write(tmp_path / "pkg" / "a.py", LEAK + BFILL)
    assert find_config(target) == (tmp_path / "pyproject.toml").resolve()
    assert ids(lint_path(target)) == ["PP002"]


def test_a_broken_pyproject_that_only_names_peekproof_is_ignored(tmp_path: Path) -> None:
    text = (
        '[project]\nname = "peekproof"\n[project.scripts]\npeekproof = "peekproof.cli:main"\n'
        "[tool.ruff\n"
    )
    write(tmp_path / "pyproject.toml", text)
    write(tmp_path / "pkg" / "pyproject.toml", "# [tool.peekproof]\n[tool.other\n")
    target = write(tmp_path / "pkg" / "a.py", LEAK)
    assert ids(lint_path(target)) == ["PP001"]


@pytest.mark.parametrize(
    "text",
    [
        "[tool.peekproof\nignore = ",
        "[tool.peekproof]\nignore = [",
        "[tool.peekproof]\nignore = ['PP001'\n",
        '[tool."peekproof"]\nignore = [',
        "[ tool . peekproof ]\nignore = ",
        "[[tool.peekproof.extra]]\nname = ",
        "tool.peekproof.ignore = [",
        "[project]\nname = 'x'\n\n[tool.peekproof]\nignore = [\n",
    ],
)
def test_a_broken_pyproject_that_holds_the_peekproof_table_is_an_error(
    tmp_path: Path, text: str
) -> None:
    config = write(tmp_path / "pyproject.toml", text)
    target = write(tmp_path / "pkg" / "a.py", LEAK)
    with pytest.raises(PeekproofError, match="invalid TOML") as excinfo:
        lint_path(target)
    assert str(config) in str(excinfo.value)
    with pytest.raises(PeekproofError, match="invalid TOML"):
        find_config(target)


def test_the_nearest_pyproject_with_the_table_counts_even_if_it_is_broken(tmp_path: Path) -> None:
    project(tmp_path, 'ignore = ["PP001"]\n')
    write(tmp_path / "pkg" / "pyproject.toml", "[tool.peekproof]\nignore = [")
    target = write(tmp_path / "pkg" / "a.py", LEAK)
    with pytest.raises(PeekproofError, match="invalid TOML"):
        lint_path(target)
    outer = write(tmp_path / "b.py", LEAK + BFILL)
    assert ids(lint_path(outer)) == ["PP002"]


def test_a_pyproject_that_is_not_utf8_is_ignored_unless_it_holds_the_table(tmp_path: Path) -> None:
    config = tmp_path / "pyproject.toml"
    config.write_bytes(b'[project]\nname = "caf\xe9"\n')
    target = write(tmp_path / "a.py", LEAK)
    assert ids(lint_path(target)) == ["PP001"]
    config.write_bytes(b'[tool.peekproof]\nignore = ["caf\xe9"]\n')
    with pytest.raises(PeekproofError, match="not valid UTF-8"):
        lint_path(target)


def test_a_config_file_that_is_named_and_broken_is_always_an_error(tmp_path: Path) -> None:
    broken = write(tmp_path / "cfg" / "pyproject.toml", BROKEN_TOML)
    target = write(tmp_path / "a.py", LEAK)
    with pytest.raises(PeekproofError, match="invalid TOML"):
        lint_path(target, config_path=broken)
    with pytest.raises(PeekproofError, match="invalid TOML"):
        load_config(broken)


def test_a_folder_called_pyproject_toml_is_not_a_config(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").mkdir()
    target = write(tmp_path / "a.py", LEAK)
    assert find_config(target) is None
    assert ids(lint_path(target)) == ["PP001"]


# --------------------------------------------------------------------------- #
# Smaller engine details
# --------------------------------------------------------------------------- #


def test_the_lint_package_exports_what_it_lists() -> None:
    assert {"RULES", "lint_path", "lint_source"} <= set(peekproof.lint.__all__)
    assert len(set(peekproof.lint.__all__)) == len(peekproof.lint.__all__)
    assert all(hasattr(peekproof.lint, name) for name in peekproof.lint.__all__)
    namespace: dict[str, object] = {}
    exec("from peekproof.lint import *", namespace)  # what a star import brings in
    assert set(namespace) - {"__builtins__"} == set(peekproof.lint.__all__)


def test_segment_is_remembered_for_the_nodes_of_the_tree() -> None:
    ctx = LintContext.build("z = (a +\n     b)\n")
    node = ctx.tree.body[0]
    assert isinstance(node, ast.Assign)
    first = ctx.segment(node.value)
    assert first == "a +\n     b"
    assert ctx.segment(node.value) is first
    assert ctx.segment(ast.Name(id="q", ctx=ast.Load())) == ""  # a node without a position


def test_a_failing_rule_is_reported_as_pp000_and_the_other_rules_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Failing(RULES["PP002"].__class__):
        def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
            raise KeyError("boom")

    monkeypatch.setitem(RULES, "PP002", Failing())
    found = lint_source(LEAK + BFILL)
    assert [(f.rule_id, f.confidence) for f in found] == [("PP000", "low"), ("PP001", "high")]
    assert found[0].message == "could not analyse: PP002 failed (KeyError)"
