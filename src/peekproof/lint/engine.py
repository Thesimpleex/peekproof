"""Engine of the static linter.

The engine finds files, parses them with :mod:`ast`, hands the tree and a
:class:`LintContext` to every rule, applies suppression comments and the
configuration filters, and returns plain :class:`~peekproof.report.Finding`
objects. It never imports or runs the code it checks.

Notebooks (``.ipynb``) are linted cell by cell. A finding in a notebook is
reported with the path ``notebook.ipynb#cell<N>`` and a line number that is
relative to the cell. ``N`` is the 1-based position of the cell among *all*
cells of the notebook (markdown cells count).

Columns in findings are 1-based character columns.

Label functions: code that builds labels may look ahead on purpose, so PP001,
PP009 and PP011 are not reported inside a *label function*. That is a function
whose ``def`` line carries ``# peekproof: label`` or whose name matches the
``label_functions`` option of ``[tool.peekproof]`` (a regular expression, searched
in the name without regard to case; default :data:`DEFAULT_LABEL_FUNCTIONS`). The
default matches names that end in a label word (``make_label``, ``get_target``,
``y``) and names with a forward-return word (``fwd_ret``, ``forward_returns_5d``).
It does not match ``target_position`` or ``forward_fill_features``.

Configuration files: the nearest ``pyproject.toml`` with a ``[tool.peekproof]``
table is used. A ``pyproject.toml`` that is not valid TOML is skipped unless it
holds that table, because then the broken file is the configuration.
"""

from __future__ import annotations

import ast
import fnmatch
import io
import json
import os
import re
import tokenize
import tomllib
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from peekproof.errors import PeekproofError
from peekproof.lint.rules import RULES, Rule, expr_key
from peekproof.report import CONFIDENCE_ORDER, Confidence, Finding

__all__ = [
    "DEFAULT_LABEL_FUNCTIONS",
    "PARSE_ERROR_ID",
    "SKIP_DIRS",
    "LintConfig",
    "LintContext",
    "find_config",
    "lint_path",
    "lint_source",
    "load_config",
]

PARSE_ERROR_ID = "PP000"
PARSE_ERROR_NAME = "parse-error"
#: Default ``label_functions`` pattern, searched in a function name without regard to case:
#: a name that ends in a label word, or that holds a forward-return word.
DEFAULT_LABEL_FUNCTIONS = (
    r"(?:^|_)(?:label|labels|target|targets|y)$"
    r"|(?:^|_)(?:forward|fwd)_(?:return|returns|ret|rets)(?:_\d+)?(?:$|_)"
)

#: Directories that are never walked (virtual environments are also found by ``pyvenv.cfg``).
SKIP_DIRS = frozenset(
    {
        ".venv",
        "site-packages",
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        ".nox",
        "node_modules",
    }
)
_SOURCE_SUFFIXES = frozenset({".py", ".ipynb"})
_CONFIDENCE_LEVELS: dict[str, Confidence] = {"low": "low", "medium": "medium", "high": "high"}
_CONFIG_KEYS = {
    "select": "select",
    "ignore": "ignore",
    "min_confidence": "min_confidence",
    "min-confidence": "min_confidence",
    "label_functions": "label_functions",
    "label-functions": "label_functions",
    "exclude": "exclude",
}
_DIRECTIVE_RE = re.compile(
    r"peekproof:\s*(?P<kind>ignore|label)\b(?:\s*\[(?P<ids>[^\]]*)\])?", re.IGNORECASE
)
#: Notebook cell magics whose body is ordinary Python; any other ``%%magic`` cell is skipped.
_PYTHON_CELL_MAGICS = frozenset({"time", "timeit", "capture", "prun", "debug", "python", "python3"})
_ASSIGNED_MAGIC_RE = re.compile(r"^(?P<head>\s*[A-Za-z_][\w.\[\]'\", ]*=\s*)[!%].*$")
_HELP_LINE_RE = re.compile(r"^\s*(?:\?\??\s*[\w.]+|[\w.]+\s*\?\??)\s*$")
_DEFAULT_ALIASES = {"np": "numpy", "pd": "pandas", "sm": "statsmodels.api", "yf": "yfinance"}
_SNIPPET_LIMIT = 160
_BOM = chr(0xFEFF)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LintConfig:
    """Settings that decide which rules run and which findings are kept.

    ``label_functions`` is a regular expression for the names of label functions (see the
    module docstring); ``exclude`` holds glob patterns relative to the folder of the config file.
    """

    select: tuple[str, ...] | None = None
    ignore: tuple[str, ...] = ()
    min_confidence: Confidence = "low"
    label_functions: str = DEFAULT_LABEL_FUNCTIONS
    exclude: tuple[str, ...] = ()
    root: Path | None = None

    def rule_enabled(self, rule_id: str) -> bool:
        """Return True if ``rule_id`` is selected and not ignored."""
        if self.select is not None and not any(rule_id.startswith(s) for s in self.select):
            return False
        return not any(rule_id.startswith(i) for i in self.ignore)

    def keeps(self, finding: Finding) -> bool:
        """Return True if a finding passes the rule and confidence filters."""
        if not self.rule_enabled(finding.rule_id):
            return False
        return CONFIDENCE_ORDER[finding.confidence] >= CONFIDENCE_ORDER[self.min_confidence]


def _known_rule_ids() -> list[str]:
    return [PARSE_ERROR_ID, *RULES]


def _normalise_selectors(value: str | Iterable[str] | None, where: str) -> tuple[str, ...] | None:
    """Return upper-case rule selectors (ids or id prefixes) from a list or a comma string.

    An empty selection is returned as ``None`` (meaning "no restriction").
    """
    if value is None:
        return None
    items = [value] if isinstance(value, str) else list(value)
    selectors: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise PeekproofError(f"{where}: rule selectors must be strings, got {item!r}")
        selectors.extend(part.upper() for part in re.split(r"[,\s]+", item.strip()) if part)
    known = _known_rule_ids()
    for selector in selectors:
        if not any(rule_id.startswith(selector) for rule_id in known):
            raise PeekproofError(
                f"{where}: unknown rule selector {selector!r} (known rules: {', '.join(known)})"
            )
    return tuple(selectors) or None


def _check_confidence(value: object, where: str) -> Confidence:
    if isinstance(value, str) and value in _CONFIDENCE_LEVELS:
        return _CONFIDENCE_LEVELS[value]
    raise PeekproofError(
        f"{where}: min_confidence must be one of {', '.join(_CONFIDENCE_LEVELS)}, got {value!r}"
    )


def _check_regex(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise PeekproofError(f"{where}: label_functions must be a regular expression string")
    try:
        re.compile(value)
    except re.error as exc:
        raise PeekproofError(f"{where}: label_functions is not a valid regex ({exc})") from exc
    return value


def _check_globs(value: object, where: str) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
        raise PeekproofError(f"{where}: exclude must be a list of glob strings")
    return tuple(items)


#: Marks a file that (probably) holds the peekproof configuration: a ``[tool.peekproof]``
#: or ``[tool.peekproof.x]`` header, or a ``tool.peekproof.x = ...`` key. It is only used to
#: decide whether a file that is not valid TOML matters, so a loose match is fine.
_TOOL_TABLE_RE = re.compile(
    r"^[ \t]*(?:\[\[?[ \t]*)?tool[ \t]*\.[ \t]*[\"']?peekproof(?![\w-])", re.MULTILINE
)


def _read_toml(path: Path) -> tuple[dict[str, Any] | None, str, str]:
    """Read a TOML file and return ``(data, problem, text)``.

    ``data`` is ``None`` if the file is not valid TOML (``problem`` then says why).
    ``text`` is the file content; bytes that are not UTF-8 are replaced.
    Raises :class:`OSError` if the file cannot be read.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    try:
        return tomllib.loads(raw.decode("utf-8")), "", text
    except UnicodeDecodeError as exc:
        return None, f"not valid UTF-8: {exc.reason}", text
    except tomllib.TOMLDecodeError as exc:
        return None, str(exc), text


def load_config(path: Path) -> LintConfig:
    """Read the ``[tool.peekproof]`` table of a ``pyproject.toml`` file.

    A file without that table gives the default configuration. Unknown keys,
    wrong value types, invalid patterns and a file that is not valid TOML raise
    :class:`PeekproofError`.
    """
    try:
        data, problem, _text = _read_toml(path)
    except OSError as exc:
        raise PeekproofError(f"{path}: cannot read file ({exc.strerror or exc})") from exc
    if data is None:
        raise PeekproofError(f"{path}: invalid TOML ({problem})")
    table = _peekproof_table(data, path)
    root = path.resolve().parent
    if table is None:
        return LintConfig(root=root)
    where = f"{path} [tool.peekproof]"
    values: dict[str, Any] = {}
    for key, value in table.items():
        if key not in _CONFIG_KEYS:
            allowed = ", ".join(
                sorted({"select", "ignore", "min_confidence", "label_functions", "exclude"})
            )
            raise PeekproofError(f"{where}: unknown option {key!r} (allowed: {allowed})")
        values[_CONFIG_KEYS[key]] = value
    return LintConfig(
        select=_normalise_selectors(values.get("select"), where),
        ignore=_normalise_selectors(values.get("ignore"), where) or (),
        min_confidence=_check_confidence(values.get("min_confidence", "low"), where),
        label_functions=_check_regex(values.get("label_functions", DEFAULT_LABEL_FUNCTIONS), where),
        exclude=_check_globs(values.get("exclude", []), where),
        root=root,
    )


def _peekproof_table(data: dict[str, Any], path: Path) -> dict[str, Any] | None:
    tool = data.get("tool")
    if not isinstance(tool, dict) or "peekproof" not in tool:
        return None
    table = tool["peekproof"]
    if not isinstance(table, dict):
        raise PeekproofError(f"{path}: [tool.peekproof] must be a table")
    return table


def find_config(start: Path) -> Path | None:
    """Return the nearest ``pyproject.toml`` that has a ``[tool.peekproof]`` table.

    The search starts in ``start`` (or its folder, for a file) and goes upwards. A
    ``pyproject.toml`` that cannot be read or is not valid TOML is skipped, as it
    belongs to some other tool, unless it holds a ``[tool.peekproof]`` table: a broken
    peekproof configuration raises :class:`PeekproofError`.
    """
    start = start.resolve()
    directory = start if start.is_dir() else start.parent
    for candidate_dir in (directory, *directory.parents):
        candidate = candidate_dir / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            data, problem, text = _read_toml(candidate)
        except OSError:
            continue
        if data is None:
            if _TOOL_TABLE_RE.search(text):
                raise PeekproofError(f"{candidate}: invalid TOML ({problem})")
            continue
        if _peekproof_table(data, candidate) is not None:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Per-file context handed to the rules
# --------------------------------------------------------------------------- #


def _normalise_newlines(text: str) -> str:
    """Use ``\\n`` everywhere (the parser does the same) and drop a leading BOM."""
    return text.removeprefix(_BOM).replace("\r\n", "\n").replace("\r", "\n")


def _parse(source: str, filename: str) -> ast.Module:
    """Parse ``source``; warnings about the checked code (``'\\d'`` escapes, ...) are not shown."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(source, filename=filename)


def _comment_tokens(source: str) -> dict[int, str]:
    """Return ``{line: comment text}`` for every comment in ``source``."""
    comments: dict[int, str] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                comments[token.start[0]] = token.string
    except (tokenize.TokenError, SyntaxError):
        # The parser accepted the source, so this is rare; fall back to a plain scan.
        for number, line in enumerate(source.split("\n"), start=1):
            if "#" in line:
                comments[number] = line[line.index("#") :]
    return comments


class LintContext:
    """Information about one source file (or notebook cell) for the rules.

    Attributes:
        path: Path shown in findings (``name.ipynb#cell3`` for notebook cells).
        source: Source text with ``\\n`` line endings.
        lines: Source split into lines; ``lines[0]`` is line 1.
        tree: The parsed module.
        label_spans: ``(first_line, last_line)`` of every label function.
        aliases: Names bound by imports, mapped to the dotted name they stand for.
        bound_names: Names bound by imports or by ``def``/``class`` in the file.
    """

    def __init__(
        self,
        source: str,
        path: str = "<string>",
        tree: ast.Module | None = None,
        *,
        label_functions: str = DEFAULT_LABEL_FUNCTIONS,
    ) -> None:
        self.path = path
        self.source = _normalise_newlines(source)
        self.tree: ast.Module = tree if tree is not None else _parse(self.source, path)
        self.lines = self.source.split("\n")
        self._byte_lines = [line.encode("utf-8") for line in self.lines]
        self._segments: dict[ast.AST, str] = {}
        self.aliases: dict[str, str] = {}
        self.bound_names: set[str] = set()
        self._parents: dict[ast.AST, ast.AST] = {}
        self._assignments: dict[str, list[ast.expr]] | None = None
        self._index_tree()
        self._ignores: dict[int, frozenset[str] | None] = {}
        self._label_lines: set[int] = set()
        self._read_directives()
        self.label_spans: list[tuple[int, int]] = self._find_label_functions(label_functions)

    # -- construction helpers ------------------------------------------------

    @classmethod
    def build(
        cls,
        source: str,
        path: str = "<string>",
        *,
        label_functions: str = DEFAULT_LABEL_FUNCTIONS,
    ) -> LintContext:
        """Parse ``source`` and return its context (raises ``SyntaxError`` if it does not parse)."""
        return cls(source, path, label_functions=label_functions)

    def _index_tree(self) -> None:
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self._parents[child] = parent
            if isinstance(parent, ast.Import):
                for alias in parent.names:
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
                        self.bound_names.add(alias.asname)
                    else:
                        root = alias.name.split(".")[0]
                        self.aliases.setdefault(root, root)
                        self.bound_names.add(root)
            elif isinstance(parent, ast.ImportFrom):
                for alias in parent.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    self.bound_names.add(local)  # also for relative imports (project code)
                    if parent.module and parent.level == 0:
                        self.aliases[local] = f"{parent.module}.{alias.name}"
            elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bound_names.add(parent.name)

    def _read_directives(self) -> None:
        """Collect ``# peekproof: ignore[...]`` and ``# peekproof: label`` comments by line."""
        for line, text in _comment_tokens(self.source).items():
            ignore_all = False
            listed_ids: set[str] = set()
            has_ignore = False
            for match in _DIRECTIVE_RE.finditer(text):
                if match.group("kind").lower() == "label":
                    self._label_lines.add(line)
                    continue
                has_ignore = True
                ids = {i.upper() for i in re.split(r"[,\s]+", match.group("ids") or "") if i}
                if ids:
                    listed_ids |= ids
                else:
                    ignore_all = True  # a bare ``ignore`` (or an empty list) covers every rule
            if has_ignore:
                self._ignores[line] = None if ignore_all else frozenset(listed_ids)

    def _find_label_functions(self, pattern_text: str) -> list[tuple[int, int]]:
        pattern = re.compile(pattern_text, re.IGNORECASE)
        spans: list[tuple[int, int]] = []
        for node in ast.walk(self.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            first_body = node.body[0].lineno if node.body else node.lineno
            header = range(node.lineno, max(node.lineno, first_body - 1) + 1)
            if pattern.search(node.name) or any(line in self._label_lines for line in header):
                spans.append((node.lineno, node.end_lineno or node.lineno))
        return spans

    # -- queries used by the rules ---------------------------------------------

    def parent(self, node: ast.AST) -> ast.AST | None:
        """Return the parent of ``node`` in the tree, or ``None`` for the module."""
        return self._parents.get(node)

    def enclosing_scope(self, node: ast.AST) -> ast.AST:
        """Return the function (or the module) that contains ``node``."""
        current = self._parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                return current
            current = self._parents.get(current)
        return self.tree

    @property
    def assignments(self) -> dict[str, list[ast.expr]]:
        """Values assigned to each simple target, keyed by :func:`expr_key` of the target."""
        if self._assignments is None:
            found: dict[str, list[ast.expr]] = {}
            for node in ast.walk(self.tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self._record(found, target, node.value)
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    self._record(found, node.target, node.value)
            self._assignments = found
        return self._assignments

    def _record(self, found: dict[str, list[ast.expr]], target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, (ast.Name, ast.Attribute, ast.Subscript)):
            found.setdefault(expr_key(target), []).append(value)
        elif (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            for sub_target, sub_value in zip(target.elts, value.elts, strict=True):
                self._record(found, sub_target, sub_value)

    def qualname(self, node: ast.expr) -> str | None:
        """Return the dotted name of ``a.b.c`` with its first part resolved through imports.

        ``np.roll`` becomes ``numpy.roll`` after ``import numpy as np`` (``np``,
        ``pd`` and ``yf`` are resolved even without an import, which keeps
        single notebook cells working). Returns ``None`` if the expression is
        not a plain dotted name.
        """
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        head = self.aliases.get(current.id) or _DEFAULT_ALIASES.get(current.id, current.id)
        parts.append(head)
        return ".".join(reversed(parts))

    def head_is_bound(self, node: ast.expr) -> bool:
        """Return True if the first name of a dotted expression is bound by an import or a def."""
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        return isinstance(current, ast.Name) and current.id in self.bound_names

    def is_module_name(self, name: str) -> bool:
        """Return True if ``name`` stands for something imported rather than for a local variable.

        That is a name bound by an import, or one of the conventional aliases (``np``, ``pd``,
        ...) that the file does not bind itself. ``np.diff(x)`` calls a function of a module;
        ``prices.diff()`` calls a method of data.
        """
        if name in self.aliases:
            return True
        return name in _DEFAULT_ALIASES and name not in self.bound_names

    def is_from(self, func: ast.expr, *roots: str) -> bool:
        """Return True if a callee plausibly comes from one of the top-level packages ``roots``.

        A bare name that the file never imports or defines is accepted, because
        the import may live in another notebook cell.
        """
        qualified = self.qualname(func)
        if qualified is None:
            return False
        if qualified.split(".", 1)[0] in roots:
            return True
        return isinstance(func, ast.Name) and func.id not in self.bound_names

    def in_label_function(self, line: int) -> bool:
        """Return True if ``line`` lies inside a label function."""
        return any(first <= line <= last for first, last in self.label_spans)

    def is_suppressed(self, rule_id: str, line: int) -> bool:
        """Return True if a ``# peekproof: ignore`` comment on ``line`` covers ``rule_id``."""
        if line not in self._ignores:
            return False
        ids = self._ignores[line]
        return ids is None or rule_id.upper() in ids

    # -- text helpers ----------------------------------------------------------

    def segment(self, node: ast.AST) -> str:
        """Return the source text of ``node``.

        Same result as :func:`ast.get_source_segment`, but it works on lines that
        were split once and remembers each answer, so calling it many times on a
        large file stays cheap.
        """
        cached = self._segments.get(node)
        if cached is None:
            cached = self._segments[node] = self._cut(node)
        return cached

    def _cut(self, node: ast.AST) -> str:
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        if lineno is None or end_lineno is None or col is None or end_col is None:
            return ""
        lines = self._byte_lines
        if lineno == end_lineno:
            raw = lines[lineno - 1][col:end_col]
        else:
            parts = [
                lines[lineno - 1][col:],
                *lines[lineno : end_lineno - 1],
                lines[end_lineno - 1][:end_col],
            ]
            raw = b"\n".join(parts)
        return raw.decode("utf-8")

    def snippet(self, line: int) -> str:
        """Return the stripped source line ``line`` (shortened if it is very long)."""
        if not 1 <= line <= len(self.lines):
            return ""
        text = self.lines[line - 1].strip()
        return text if len(text) <= _SNIPPET_LIMIT else text[: _SNIPPET_LIMIT - 3] + "..."

    def position(self, node: ast.AST, *, method: bool = False) -> tuple[int, int]:
        """Return the 1-based ``(line, column)`` that a finding for ``node`` points at.

        With ``method=True`` and a call such as ``a.b.shift(-1)`` (or an attribute
        node), the position is that of the method name ``shift``.
        """
        target = node.func if method and isinstance(node, ast.Call) else node
        if method and isinstance(target, ast.Attribute) and target.end_col_offset is not None:
            line = target.end_lineno or target.lineno
            byte_col = target.end_col_offset - len(target.attr.encode("utf-8"))
        else:
            line = int(getattr(node, "lineno", 1))
            byte_col = int(getattr(node, "col_offset", 0))
        raw = self._byte_lines[line - 1] if 1 <= line <= len(self._byte_lines) else b""
        return line, len(raw[:byte_col].decode("utf-8", errors="ignore")) + 1

    def finding(
        self,
        rule: Rule,
        node: ast.AST,
        *,
        message: str | None = None,
        hint: str | None = None,
        confidence: Confidence | None = None,
        method: bool = False,
    ) -> Finding:
        """Build a finding for ``rule`` located at ``node``."""
        line, col = self.position(node, method=method)
        return Finding(
            rule_id=rule.id,
            name=rule.name,
            message=message or rule.message,
            hint=hint or rule.hint,
            path=self.path,
            line=line,
            col=col,
            confidence=confidence or rule.confidence,
            snippet=self.snippet(line),
        )


# --------------------------------------------------------------------------- #
# Parsing and linting one source text
# --------------------------------------------------------------------------- #


def _failure(
    path: str, line: int, col: int, text: str, source: str = "", *, action: str = "parse"
) -> Finding:
    """Return the PP000 finding that reports a file (or a rule) the linter had to give up on."""
    lines = source.split("\n")
    snippet = lines[line - 1].strip()[:_SNIPPET_LIMIT] if 1 <= line <= len(lines) else ""
    return Finding(
        rule_id=PARSE_ERROR_ID,
        name=PARSE_ERROR_NAME,
        message=f"could not {action}: {text}",
        hint="Fix the problem or exclude the file; peekproof skipped what it could not read.",
        path=path,
        line=line,
        col=col,
        confidence="low",
        snippet=snippet,
    )


def _blank_ipython_lines(source: str) -> str:
    """Remove IPython-only syntax so the cell can be parsed as plain Python.

    - Lines whose first non-blank character is ``%`` or ``!`` (magics and shell
      escapes) are blanked. An indented one becomes ``pass`` so that its block
      stays valid. A trailing backslash continues the magic onto the next line.
    - ``name = !command`` and ``name = %magic`` keep the assignment and use ``None``.
    - Help lines such as ``df.describe?`` or ``?df`` are blanked.

    Line numbers do not change.
    """
    out: list[str] = []
    continued = False
    for line in source.split("\n"):
        stripped = line.lstrip()
        assigned = None if continued else _ASSIGNED_MAGIC_RE.match(line)
        escape = stripped.startswith(("%", "!")) and not stripped.startswith("!=")
        is_magic = continued or escape or _HELP_LINE_RE.match(line) is not None
        if assigned is not None:
            out.append(assigned.group("head") + "None")
        elif is_magic:
            indent = line[: len(line) - len(stripped)]
            out.append("" if continued or not indent else indent + "pass")
        else:
            out.append(line)
            continue
        continued = line.rstrip().endswith("\\")
    return "\n".join(out)


def _is_foreign_cell_magic(source: str) -> bool:
    """Return True for a ``%%name`` cell whose body is not Python (``%%bash``, ``%%sql``...)."""
    for line in source.split("\n"):
        if line.strip():
            if not line.lstrip().startswith("%%"):
                return False
            words = line.lstrip()[2:].split()
            return not words or words[0].lower() not in _PYTHON_CELL_MAGICS
    return False


def _lint_python(
    text: str,
    path: str,
    config: LintConfig,
    rules: Sequence[Rule],
    *,
    notebook_cell: bool = False,
) -> list[Finding]:
    source = _normalise_newlines(text)
    if notebook_cell and _is_foreign_cell_magic(source):
        return []
    tree: ast.Module | None = None
    error: Exception | None = None
    candidates = [source]
    if notebook_cell:
        candidates.append(_blank_ipython_lines(source))
    for candidate in candidates:
        try:
            tree = _parse(candidate, path)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            error = exc
            continue
        source = candidate
        break
    if tree is None:
        return [_syntax_failure(path, error, source)]
    ctx = LintContext(source, path, tree, label_functions=config.label_functions)
    raw: list[Finding] = []
    for rule in rules:
        try:
            raw.extend(rule.visit(tree, ctx))
        except RecursionError:
            # Extremely deep nesting (a huge data table written as one expression, say):
            # skip this rule for the file instead of crashing.
            raw.append(
                _failure(
                    path, 1, 1, f"{rule.id} gave up (nesting too deep)", source, action="analyse"
                )
            )
        except Exception as exc:
            # A bug in one rule must not stop the whole run: report it and go on.
            raw.append(
                _failure(
                    path, 1, 1, f"{rule.id} failed ({type(exc).__name__})", source, action="analyse"
                )
            )
    seen: set[tuple[str, int, int]] = set()
    kept: list[Finding] = []
    for finding in sorted(raw, key=lambda f: (f.line, f.col, f.rule_id)):
        key = (finding.rule_id, finding.line, finding.col)
        if key in seen or ctx.is_suppressed(finding.rule_id, finding.line):
            continue
        seen.add(key)
        kept.append(finding)
    return kept


def _syntax_failure(path: str, error: Exception | None, source: str) -> Finding:
    if isinstance(error, SyntaxError):
        line = error.lineno or 1
        col = error.offset or 1
        return _failure(path, line, col, error.msg, source)
    return _failure(path, 1, 1, str(error) or "invalid source", source)


def _notebook_language(notebook: dict[str, Any]) -> str:
    """Return the lower-case language named in the notebook metadata ("" if it names none)."""
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    for section, key in (("language_info", "name"), ("kernelspec", "language")):
        entry = metadata.get(section)
        if isinstance(entry, dict) and isinstance(entry.get(key), str):
            return str(entry[key]).strip().lower()
    return ""


def _lint_notebook(
    text: str, path: str, config: LintConfig, rules: Sequence[Rule]
) -> list[Finding]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        return [_failure(path, exc.lineno, exc.colno, f"notebook is not valid JSON ({exc.msg})")]
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return [_failure(path, 1, 1, "notebook has no 'cells' list (only nbformat 4 is supported)")]
    if _notebook_language(notebook) not in {"", "python"}:
        return []  # an R or Julia notebook is not Python code
    findings: list[Finding] = []
    for number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        elif not isinstance(source, str):
            continue
        findings.extend(
            _lint_python(source, f"{path}#cell{number}", config, rules, notebook_cell=True)
        )
    return findings


def _lint_text(text: str, path: str, config: LintConfig, rules: Sequence[Rule]) -> list[Finding]:
    if path.lower().endswith(".ipynb"):
        findings = _lint_notebook(text, path, config, rules)
    else:
        findings = _lint_python(text, path, config, rules)
    return [finding for finding in findings if config.keeps(finding)]


def _enabled_rules(config: LintConfig) -> list[Rule]:
    return [rule for rule in RULES.values() if config.rule_enabled(rule.id)]


def lint_source(
    source: str,
    path: str = "<string>",
    *,
    select: str | Iterable[str] | None = None,
    ignore: str | Iterable[str] | None = None,
    min_confidence: str | None = None,
    label_functions: str | None = None,
) -> list[Finding]:
    """Lint a source text and return its findings.

    ``path`` is only used in the findings, except that a path ending in
    ``.ipynb`` makes ``source`` be read as notebook JSON. No configuration
    file is read; pass the options explicitly. ``None`` means the default
    (all rules, minimum confidence ``"low"``, the default label-function
    pattern).

    Raises:
        PeekproofError: for an unknown rule selector, confidence level or
            invalid ``label_functions`` pattern.
    """
    config = LintConfig(
        select=_normalise_selectors(select, "select"),
        ignore=_normalise_selectors(ignore, "ignore") or (),
        min_confidence=_check_confidence(
            "low" if min_confidence is None else min_confidence, "lint_source"
        ),
        label_functions=(
            DEFAULT_LABEL_FUNCTIONS
            if label_functions is None
            else _check_regex(label_functions, "lint_source")
        ),
    )
    return _lint_text(source, path, config, _enabled_rules(config))


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #


def _relative_posix(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _depth(path: Path, root: Path | None) -> int:
    """Return how many path components ``path`` has below the folder of the config file."""
    relative = _relative_posix(path, root)
    return 0 if relative in {"", "."} else len(relative.split("/"))


def _is_excluded(path: Path, config: LintConfig, walk_depth: int = 0) -> bool:
    """Return True if ``path`` matches an ``exclude`` glob (relative to the config file's folder).

    A pattern matches the path itself or any parent folder of it. A pattern
    without ``/`` also matches a single folder or file name anywhere. ``*``
    matches across ``/`` (plain :mod:`fnmatch` rules), and ``**/`` may also
    stand for no folder at all, so ``tests/**/data`` matches ``tests/data``.

    The folder that the caller asked to walk (``walk_depth`` components deep) and
    its parents are never tested: naming a folder explicitly lints it, as naming a
    file does.
    """
    if not config.exclude:
        return False
    parts = _relative_posix(path, config.root).split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(walk_depth, len(parts))]
    names = parts[walk_depth:]
    for raw in config.exclude:
        pattern = raw.removeprefix("./").rstrip("/")
        for variant in {pattern, pattern.replace("**/", "")}:
            if any(fnmatch.fnmatchcase(prefix, variant) for prefix in prefixes):
                return True
            if "/" not in variant and any(fnmatch.fnmatchcase(name, variant) for name in names):
                return True
    return False


def _discover(path: Path, config: LintConfig) -> Iterator[Path]:
    """Yield the files to lint below ``path``; an explicit file is always yielded."""
    if path.is_file():
        yield path
        return
    depth = _depth(path, config.root) if config.exclude else 0
    for dirpath, dirnames, filenames in os.walk(path):
        base = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not (base / d / "pyvenv.cfg").is_file()
            and not _is_excluded(base / d, config, depth)
        )
        for name in sorted(filenames):
            candidate = base / name
            suffix_ok = candidate.suffix.lower() in _SOURCE_SUFFIXES
            if suffix_ok and not _is_excluded(candidate, config, depth):
                yield candidate


def _read_source(path: Path) -> str | Finding:
    """Read a source file, honouring its encoding declaration; return a PP000 finding on failure."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _failure(str(path), 1, 1, f"cannot read file ({exc.strerror or exc})")
    try:
        if path.suffix.lower() == ".ipynb":
            return data.decode("utf-8-sig")
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as exc:
        return _failure(str(path), 1, 1, f"cannot decode file ({exc})")


def _resolve_config_file(config_path: str | os.PathLike[str]) -> Path:
    path = Path(config_path)
    if path.is_dir():
        path = path / "pyproject.toml"
    if not path.is_file():
        raise PeekproofError(f"configuration file not found: {path}")
    return path


def lint_path(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    select: str | Iterable[str] | None = None,
    ignore: str | Iterable[str] | None = None,
    min_confidence: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> list[Finding]:
    """Lint files and directories and return all findings.

    Directories are walked for ``*.py`` and ``*.ipynb`` files (skipping
    ``.venv``, ``site-packages``, ``__pycache__``, ``.git``, virtual
    environments and the configured ``exclude`` globs). A file given by name is
    always linted.

    Settings come from the nearest ``pyproject.toml`` with a ``[tool.peekproof]``
    table above each given path, or from ``config_path`` (a ``pyproject.toml``
    file or the folder that holds it) if given. The keyword arguments override
    the file; ``None`` means "not given". ``select`` and ``ignore`` take rule
    ids or id prefixes, as a list or a comma-separated string.

    Raises:
        PeekproofError: for a path that does not exist, an unreadable or
            invalid configuration, an unknown rule selector or confidence level.
    """
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    overrides: dict[str, Any] = {}
    if select is not None:
        overrides["select"] = _normalise_selectors(select, "select")
    if ignore is not None:
        overrides["ignore"] = _normalise_selectors(ignore, "ignore") or ()
    if min_confidence is not None:
        overrides["min_confidence"] = _check_confidence(min_confidence, "min_confidence")
    explicit = load_config(_resolve_config_file(config_path)) if config_path is not None else None
    cache: dict[Path, LintConfig] = {}
    findings: list[Finding] = []
    seen: set[Path] = set()
    for item in items:
        path = Path(item)
        if not path.exists():
            raise PeekproofError(f"path does not exist: {path}")
        config = explicit
        if config is None:
            found = find_config(path)
            if found is None:
                config = LintConfig()
            else:
                if found not in cache:
                    cache[found] = load_config(found)
                config = cache[found]
        config = replace(config, **overrides)
        rules = _enabled_rules(config)
        for file in _discover(path, config):
            key = file.resolve()
            if key in seen:
                continue
            seen.add(key)
            source = _read_source(file)
            if isinstance(source, Finding):
                findings.extend(f for f in [source] if config.keeps(f))
            else:
                findings.extend(_lint_text(source, str(file), config, rules))
    return findings
