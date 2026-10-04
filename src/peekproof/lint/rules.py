"""Rules of the static linter (PP001 to PP013).

Every rule is a small class with an ``id``, a ``name``, a default
``confidence``, a one-line ``message``, a ``hint`` and a ``visit(tree, ctx)``
method that returns findings. Rules only look at the syntax tree and at source
text. They never import, run or fetch anything, and they keep no state between
calls, so one rule instance can be reused for any number of files.

A finding may carry its own confidence, message and hint. The hint of every
medium or low confidence finding ends with "Confirm with peekproof.check().",
because only the oracle can settle those cases.

The registry :data:`RULES` maps each rule id to its (stateless) instance.
"""

from __future__ import annotations

import ast
import re
from bisect import bisect_left
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from peekproof.report import Confidence, Finding

if TYPE_CHECKING:
    from peekproof.lint.engine import LintContext

__all__ = [
    "PP001",
    "PP002",
    "PP003",
    "PP004",
    "PP005",
    "PP006",
    "PP007",
    "PP008",
    "PP009",
    "PP010",
    "PP011",
    "PP012",
    "PP013",
    "RULES",
    "Rule",
    "expr_key",
]


# --------------------------------------------------------------------------- #
# Small AST helpers shared by the rules (only ``expr_key`` is used by the engine)
# --------------------------------------------------------------------------- #

_SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_CONFIRM = "Confirm with peekproof.check()."
_MAX_CLIMB = 100  # levels that _stored_under climbs from an expression to its statement


def _confirm(text: str) -> str:
    """Return ``text`` closed by the sentence that every medium or low hint ends with."""
    return f"{text} {_CONFIRM}"


def expr_key(node: ast.AST) -> str:
    """Return a key that is equal for expressions that differ only in formatting.

    Quotes, spacing and the load/store context do not change the key, so
    ``df["a"]`` on the right-hand side matches ``df['a'] = ...`` on the left.
    """
    return ast.dump(node).replace("ctx=Store()", "ctx=Load()").replace("ctx=Del()", "ctx=Load()")


def _node_position(node: ast.AST) -> tuple[int, int]:
    """Return ``(line, column)`` of a node, for sorting."""
    return int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))


def _method_name(call: ast.Call) -> str | None:
    """Return ``m`` for a call of the form ``obj.m(...)``, otherwise ``None``."""
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def _last_segment(dotted: str) -> str:
    """Return the part after the last dot of a dotted name."""
    return dotted.rsplit(".", 1)[-1]


def _last_name(call: ast.Call) -> str | None:
    """Return the final name of the callee: ``f`` for ``f(...)`` and ``m.f(...)``."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _name_tokens(name: str) -> set[str]:
    """Split an identifier or a source snippet into lower-case alphabetic words."""
    return set(re.findall(r"[a-z]+", name.lower()))


def _get_argument(call: ast.Call, position: int | None, *names: str) -> ast.expr | None:
    """Return an argument given by keyword name, or by position, if present."""
    for keyword in call.keywords:
        if keyword.arg is not None and keyword.arg in names:
            return keyword.value
    if position is not None and position < len(call.args):
        leading = call.args[: position + 1]
        if not any(isinstance(arg, ast.Starred) for arg in leading):
            return call.args[position]
    return None


def _has_star_kwargs(call: ast.Call) -> bool:
    """Return True if the call passes ``**mapping``, so its keywords are unknown."""
    return any(keyword.arg is None for keyword in call.keywords)


def _numeric_literal(node: ast.expr | None) -> float | None:
    """Return the value of an int/float literal, including a leading ``-`` or ``+``."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _numeric_literal(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _is_negative_literal(node: ast.expr | None) -> bool:
    """Return True for ``-3``, ``-0.5`` and for a constant node holding a negative number."""
    value = _numeric_literal(node)
    return value is not None and value < 0


def _int_literal(node: ast.expr | None) -> int | None:
    """Return the value of an integer literal (with an optional sign), otherwise ``None``."""
    if isinstance(node, ast.Constant):
        value = node.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _int_literal(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _is_positive_int_literal(node: ast.expr | None) -> bool:
    """Return True for a literal integer greater than zero."""
    value = _int_literal(node)
    return value is not None and value > 0


def _string_value(node: ast.expr | None) -> str | None:
    """Return the value of a string literal, otherwise ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _string_values(node: ast.expr | None) -> list[str]:
    """Return the strings in a string literal or in a list/tuple/set of them."""
    single = _string_value(node)
    if single is not None:
        return [single]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [s for s in (_string_value(elt) for elt in node.elts) if s is not None]
    return []


def _is_truthy_constant(node: ast.expr | None) -> bool:
    """Return True for a literal that is true in a boolean context (``True``, ``1``, ...)."""
    return isinstance(node, ast.Constant) and bool(node.value)


def _axis_is_columns(call: ast.Call, position: int | None = None) -> bool:
    """Return True for ``axis=1`` / ``axis="columns"``: the call works across columns, not time."""
    axis = _get_argument(call, position, "axis")
    return (
        isinstance(axis, ast.Constant)
        and not isinstance(axis.value, bool)
        and axis.value in (1, "columns")
    )


def _root_name(node: ast.expr | None) -> str | None:
    """Return the name at the start of an attribute / subscript / method-call chain."""
    current = node
    while current is not None:
        if isinstance(current, ast.Name):
            return current.id
        if isinstance(current, (ast.Attribute, ast.Subscript, ast.Starred)):
            current = current.value
        elif isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            current = current.func.value
        else:
            return None
    return None


def _target_names(target: ast.AST) -> list[str]:
    """Return the plain variable names bound by an assignment or loop target."""
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes of one scope without entering nested functions or classes."""
    stack = list(ast.iter_child_nodes(scope))
    stack.reverse()
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_TYPES):
            continue
        children = list(ast.iter_child_nodes(node))
        children.reverse()
        stack.extend(children)


def _scopes(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield the module and every function definition in it."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _result_values(tree: ast.AST) -> Iterator[tuple[ast.expr, ast.expr | None]]:
    """Yield ``(value, partner)`` for every expression whose result is kept.

    A result is kept when it is assigned to a variable or a column, or returned
    from a function. ``partner`` is the target of an augmented assignment
    (``x -= ...`` reads ``x`` as well); it is ``None`` otherwise. Keyword values
    of ``DataFrame.assign(...)`` count as assignments to a column.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
            if node.value is not None:
                yield node.value, None
        elif isinstance(node, ast.AugAssign):
            yield node.value, node.target
        elif isinstance(node, ast.Call) and _method_name(node) == "assign":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    yield keyword.value, None


def _called_on_import(call: ast.Call, ctx: LintContext) -> bool:
    """Return True for ``np.diff(x)`` or ``Chebyshev.interpolate(f)``: called on an import.

    The receiver has to be a plain dotted name. A call on data (``df.diff()``, ``pl.col("a")
    .interpolate()``) is a method call, and so is a call on a local variable.
    """
    current = call.func
    if not isinstance(current, ast.Attribute):
        return False
    current = current.value
    while isinstance(current, ast.Attribute):
        current = current.value
    return isinstance(current, ast.Name) and ctx.is_module_name(current.id)


def _is_within(node: ast.AST, ancestor: ast.AST, ctx: LintContext) -> bool:
    """Return True if ``node`` is ``ancestor`` or lies somewhere below it."""
    current: ast.AST | None = node
    while current is not None:
        if current is ancestor:
            return True
        current = ctx.parent(current)
    return False


def _column_names(index: ast.expr) -> list[str]:
    """Return string column keys in a subscript index (``"a"``, ``["a", "b"]``, ``:, "a"``)."""
    if isinstance(index, ast.Tuple):
        names: list[str] = []
        for element in index.elts:
            names.extend(_column_names(element))
        return names
    return _string_values(index)


def _target_labels(target: ast.expr) -> list[str]:
    """Return names that describe an assignment target (variable or column key)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, ast.Subscript):
        return _column_names(target.slice) or _target_names(target.value)[:1]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_labels(element)]
    return []


def _assignment_pairs(target: ast.expr, value: ast.expr) -> Iterator[tuple[ast.expr, ast.expr]]:
    """Pair the parts of ``a, b = x, y`` one by one; any other assignment is one pair."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))
    ):
        for sub_target, sub_value in zip(target.elts, value.elts, strict=True):
            yield from _assignment_pairs(sub_target, sub_value)
    else:
        yield target, value


def _stored_under(node: ast.AST, ctx: LintContext) -> list[str]:
    """Return the names (variables, columns, ``assign`` keywords, dict keys) that ``node`` feeds.

    The walk goes up from ``node`` to its statement. ``df["fill_price"] = a.shift(-1)``
    gives ``["fill_price"]``; so do ``df.assign(fill_price=...)`` and ``{"fill_price": ...}``.
    It gives up after ``_MAX_CLIMB`` levels, so a very deeply nested expression stays cheap.
    """
    names: list[str] = []
    child: ast.AST = node
    parent = ctx.parent(child)
    for _level in range(_MAX_CLIMB):
        if parent is None:
            break
        if isinstance(parent, ast.keyword):
            call = ctx.parent(parent)
            if (
                parent.arg is not None
                and isinstance(call, ast.Call)
                and _method_name(call) == "assign"
            ):
                names.append(parent.arg)
        elif isinstance(parent, ast.Dict):
            for key, value in zip(parent.keys, parent.values, strict=True):
                if value is child and key is not None and _string_value(key) is not None:
                    names.append(str(_string_value(key)))
        elif isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assigned = parent.value
            if assigned is not None and child is assigned:
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                for target in targets:
                    for sub_target, sub_value in _assignment_pairs(target, assigned):
                        if _is_within(node, sub_value, ctx):
                            names.extend(_target_labels(sub_target))
            break
        elif isinstance(parent, ast.stmt):
            break
        child, parent = parent, ctx.parent(parent)
    return names


# --------------------------------------------------------------------------- #
# Rule base class
# --------------------------------------------------------------------------- #


class Rule:
    """Base class: a rule has metadata and a ``visit`` method that returns findings."""

    id: ClassVar[str]
    name: ClassVar[str]
    confidence: ClassVar[Confidence]
    message: ClassVar[str]
    hint: ClassVar[str]

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        """Return all findings of this rule in ``tree``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# PP001 negative shift
# --------------------------------------------------------------------------- #

# Names of columns and variables that hold labels (forward returns, targets). A negative shift
# that builds one of them is label code, which PP001 leaves alone.
_LABEL_TARGET_RE = re.compile(
    r"(?i:^(?:y|labels?|targets?|y_true)$|_(?:labels?|targets?|y_true)$"
    r"|(?:^|_)(?:forward|fwd)_(?:returns?|rets?)(?:_\d+)?(?:$|_))"
)
# Names of execution prices: ``fill_price = open.shift(-1)`` is how a next-bar fill is written.
_EXECUTION_NAME_RE = re.compile(
    r"(?:fill|exec|execution|entry|exit)_?(?:price|px)?$", re.IGNORECASE
)


class PP001(Rule):
    """``shift``, ``pct_change``, ``diff`` or ``np.roll`` with a negative amount.

    A negative literal is reported at high confidence. A negated variable (``shift(-h)``) and a
    next-bar fill or entry price (``fill_price = open.shift(-1)``) are reported at medium
    confidence. Not reported: label functions, a result stored under a label name
    (``df["target"] = ...``, ``y = ...``) and calls that work across columns (``axis=1``).
    """

    id = "PP001"
    name = "negative-shift"
    confidence = "high"
    message = "negative shift pulls future values into the current row"
    hint = (
        "Shift by a positive number of rows (look back). Forward-looking label code "
        "belongs in a function marked '# peekproof: label'."
    )

    _METHODS = frozenset({"shift", "pct_change", "diff"})
    _AXIS_POSITION = {"shift": 2, "diff": 1}
    _VARIABLE_HINT = _confirm(
        "A shift by a negated variable is a leak if the variable is positive. Shift by a "
        "positive number of rows (look back); label code belongs in a function marked "
        "'# peekproof: label'."
    )
    _EXECUTION_HINT = _confirm(
        "The next bar is a fair fill or entry price only for valuing a trade, never as an "
        "input that decides it. Keep it out of features, or mark label code "
        "'# peekproof: label'."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            found = self._describe(node, ctx)
            if found is None:
                continue
            described, is_variable = found
            as_method = _method_name(node) in self._METHODS
            line, _column = ctx.position(node, method=as_method)
            if ctx.in_label_function(line):
                continue
            names = _stored_under(node, ctx)
            if any(_LABEL_TARGET_RE.search(name) for name in names):
                continue  # ``df["target"] = df["close"].shift(-1) / df["close"] - 1``
            execution = next((n for n in names if _EXECUTION_NAME_RE.search(n)), None)
            if is_variable:
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=(
                            f"{described} is a negative shift by a variable and may pull "
                            "future values into the current row"
                        ),
                        hint=self._VARIABLE_HINT,
                        confidence="medium",
                        method=as_method,
                    )
                )
            elif execution is not None:
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=(
                            f"{described} reads the next row into '{execution}'; that is "
                            "safe for a fill price and a leak for a feature"
                        ),
                        hint=self._EXECUTION_HINT,
                        confidence="medium",
                        method=as_method,
                    )
                )
            else:
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=f"{described} pulls future values into the current row",
                        method=as_method,
                    )
                )
        return findings

    @staticmethod
    def _negative_kind(node: ast.expr | None) -> str | None:
        """Return ``"literal"`` for ``-3``, ``"variable"`` for ``-k`` and ``None`` otherwise."""
        if node is None:
            return None
        if _is_negative_literal(node):
            return "literal"
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and _numeric_literal(node.operand) is None
        ):
            return "variable"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):  # ``-1 * h``
            for factor, other in ((node.left, node.right), (node.right, node.left)):
                if _is_negative_literal(factor) and _numeric_literal(other) is None:
                    return "variable"
        return None

    def _describe(self, call: ast.Call, ctx: LintContext) -> tuple[str, bool] | None:
        """Return ``(text, negated_variable)`` for a negative shift, else ``None``."""
        method = _method_name(call)
        if method in self._METHODS:
            if _axis_is_columns(call, self._AXIS_POSITION.get(method or "")):
                return None  # ``curve.shift(-1, axis=1)`` moves across columns, not time
            periods = _get_argument(call, 0, "periods", "n")
            kind = self._negative_kind(periods)
            if periods is None or kind is None:
                return None
            if kind == "variable" and _called_on_import(call, ctx):
                return None  # ``np.diff(-x)`` negates the array, not the number of periods
            return f"{method}({ctx.segment(periods)})", kind == "variable"
        if ctx.qualname(call.func) == "numpy.roll":
            if _axis_is_columns(call, 2):
                return None
            shift = _get_argument(call, 1, "shift")
            if shift is None:
                return None
            parts = shift.elts if isinstance(shift, (ast.Tuple, ast.List)) else [shift]
            kinds = [self._negative_kind(part) for part in parts]
            if "literal" in kinds:
                return f"np.roll(..., {ctx.segment(shift)})", False
            if "variable" in kinds:
                return f"np.roll(..., {ctx.segment(shift)})", True
        return None


# --------------------------------------------------------------------------- #
# PP002 backward fill
# --------------------------------------------------------------------------- #

_BACKWARD_METHODS = frozenset({"bfill", "backfill"})


class PP002(Rule):
    """Backward fills: ``bfill``, ``fillna(method="bfill")``, ``reindex``, ``interpolate``...

    High confidence: an explicit backward fill, or ``reindex``, ``asfreq`` and ``pct_change``
    with a backward ``method``. Medium confidence: ``interpolate()`` with any method other than
    ``pad``/``ffill``, because it bridges an interior gap with the next observation;
    ``limit_direction`` only controls which leading or trailing gaps are filled, so
    ``limit_direction="forward"`` does not make it causal. Calls with ``axis=1`` fill across
    columns, not across time, and are skipped.
    """

    id = "PP002"
    name = "backward-fill"
    confidence = "high"
    message = "backward fill copies later observations into earlier rows"
    hint = "Use ffill() so a gap is filled from the past only, or leave the gap and drop it."

    _INTERPOLATE_HINT = _confirm(
        "Interpolation bridges a gap with the next valid observation. Use ffill() so a gap is "
        "filled from the past only, or leave the gap and drop it."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            found = self._describe(node)
            if found is None:
                continue
            described, confidence = found
            if confidence != "high" and _called_on_import(node, ctx):
                continue  # ``Chebyshev.interpolate(f, deg)`` fits a function; it fills no gap
            if confidence == "high":
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=f"{described} copies later observations into earlier rows",
                        method=True,
                    )
                )
            else:
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=f"{described} fills a gap using the next valid observation",
                        hint=self._INTERPOLATE_HINT,
                        confidence=confidence,
                        method=True,
                    )
                )
        return findings

    @staticmethod
    def _describe(call: ast.Call) -> tuple[str, Confidence] | None:
        method = _method_name(call)
        if method is None or _axis_is_columns(call):
            return None  # ``yields.bfill(axis=1)`` fills across maturities, not across time
        if method in {*_BACKWARD_METHODS, "backward_fill"}:
            return f"{method}()", "high"
        if method == "fillna":
            value = _string_value(_get_argument(call, None, "method"))
            if value in _BACKWARD_METHODS:
                return f'fillna(method="{value}")', "high"
        elif method == "fill_null":
            value = _string_value(_get_argument(call, 1, "strategy"))
            if value == "backward":
                return 'fill_null(strategy="backward")', "high"
        elif method in {"reindex", "reindex_like"}:
            value = _string_value(_get_argument(call, None, "method"))
            if value in {*_BACKWARD_METHODS, "nearest"}:
                return f'{method}(method="{value}")', "high"
        elif method == "asfreq":
            value = _string_value(_get_argument(call, 1, "method"))
            if value in _BACKWARD_METHODS:
                return f'asfreq(method="{value}")', "high"
        elif method == "pct_change":
            value = _string_value(_get_argument(call, 1, "fill_method"))
            if value in _BACKWARD_METHODS:
                return f'pct_change(fill_method="{value}")', "high"
        elif method == "interpolate":
            return PP002._describe_interpolate(call)
        return None

    @staticmethod
    def _describe_interpolate(call: ast.Call) -> tuple[str, Confidence] | None:
        direction = _get_argument(call, None, "limit_direction")
        if isinstance(direction, ast.Constant) and direction.value is None:
            direction = None  # ``limit_direction=None`` is the default
        direction_text = _string_value(direction)
        if direction_text in {"backward", "both"}:
            return f'interpolate(limit_direction="{direction_text}")', "high"
        if _has_star_kwargs(call):
            return None  # the method is only known at run time
        how = _string_value(_get_argument(call, 0, "method"))
        if how in {"pad", "ffill"}:
            return None
        return (f'interpolate(method="{how}")' if how else "interpolate()"), "medium"


# --------------------------------------------------------------------------- #
# PP003 centered window
# --------------------------------------------------------------------------- #

_NDIMAGE_FILTERS = frozenset(
    {
        "gaussian_filter",
        "gaussian_filter1d",
        "uniform_filter",
        "uniform_filter1d",
        "median_filter",
        "minimum_filter",
        "minimum_filter1d",
        "maximum_filter",
        "maximum_filter1d",
        "percentile_filter",
        "rank_filter",
        "generic_filter",
        "generic_filter1d",
    }
)
_TWO_SIDED_HINTS = {
    "filtfilt": (
        "filtfilt runs the filter forward and backward, so every output uses later rows",
        "Use a causal filter such as scipy.signal.lfilter.",
    ),
    "hpfilter": (
        "hpfilter is a two-sided filter: every trend value is fitted on the whole sample",
        "Refit the filter on the rows up to each date, or use a trailing average.",
    ),
    "seasonal_decompose": (
        "seasonal_decompose centers its moving average and averages the seasonal part "
        "over the whole sample",
        "Decompose the rows up to each date, or use a trailing window.",
    ),
    "gradient": (
        "np.gradient uses central differences, so every value reads the next row",
        "Use a backward difference: x.diff() or x - x.shift(1).",
    ),
    "FixedForwardWindowIndexer": (
        "FixedForwardWindowIndexer makes every window start at the current row and run "
        "into later rows",
        "Use a trailing window: rolling(n). Forward-looking labels belong in a label function.",
    ),
}
_CONVOLVE_HINT = _confirm(
    "Use mode='valid', or keep only the causal part of mode='full' with [:n]; "
    "or use a causal filter such as scipy.signal.lfilter."
)


class PP003(Rule):
    """Centered rolling windows and two-sided filters.

    High confidence: ``rolling(..., center=True)`` over three rows or more, ``savgol_filter``,
    ``filtfilt``, ``hpfilter``, ``seasonal_decompose``, ``np.gradient``,
    ``FixedForwardWindowIndexer`` and the ``scipy.ndimage`` filters. Medium confidence:
    ``np.convolve`` in mode ``'same'`` or ``'full'``; ``np.convolve(x, k)[:n]`` is causal.
    """

    id = "PP003"
    name = "centered-window"
    confidence = "high"
    message = "centered window uses rows after the current one"
    hint = (
        "Use a trailing window: rolling(n) without center=True, ewm(...), "
        "or a causal filter such as scipy.signal.lfilter."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            method = _method_name(node)
            if method is not None and (method == "rolling" or method.startswith("rolling_")):
                if self._is_centered_window(node, method):
                    findings.append(
                        ctx.finding(
                            self,
                            node,
                            message=f"{method}(center=True) uses rows after the current one",
                            method=True,
                        )
                    )
                continue
            qualified = ctx.qualname(node.func)
            if qualified is None:
                continue
            found = self._check(node, qualified, ctx)
            if found is not None:
                message, hint, confidence = found
                findings.append(
                    ctx.finding(self, node, message=message, hint=hint, confidence=confidence)
                )
        return findings

    @staticmethod
    def _is_centered_window(call: ast.Call, method: str) -> bool:
        """Return True for ``rolling(..., center=True)`` over a window that reads later rows.

        Windows of one or two rows are causal even when centered: pandas and polars only
        look ahead from a window of three rows on.
        """
        position = 2 if method == "rolling" else None
        if not _is_truthy_constant(_get_argument(call, position, "center")):
            return False
        size = _int_literal(
            _get_argument(call, 0, "window" if method == "rolling" else "window_size")
        )
        return size is None or size > 2

    def _check(
        self, call: ast.Call, qualified: str, ctx: LintContext
    ) -> tuple[str, str | None, Confidence | None] | None:
        """Return ``(message, hint, confidence)`` for a two-sided filter call, else ``None``."""
        last = _last_segment(qualified)
        if last == "savgol_filter" and ctx.is_from(call.func, "scipy"):
            return "savgol_filter smooths with a centered window (uses later rows)", None, None
        if last in {"filtfilt", "sosfiltfilt"} and ctx.is_from(call.func, "scipy"):
            return self._two_sided("filtfilt")
        if last == "hpfilter" and ctx.is_from(call.func, "statsmodels"):
            return self._two_sided("hpfilter")
        if last == "seasonal_decompose" and ctx.is_from(call.func, "statsmodels"):
            return self._two_sided("seasonal_decompose")
        if qualified == "numpy.gradient":
            return self._two_sided("gradient")
        if last == "FixedForwardWindowIndexer" and ctx.is_from(call.func, "pandas"):
            window = _int_literal(_get_argument(call, 1, "window_size"))
            if window is not None and window <= 1:
                return None  # a forward window of one row is just the current row
            return self._two_sided("FixedForwardWindowIndexer")
        if qualified == "numpy.convolve":
            return self._check_convolve(call, ctx)
        if self._is_ndimage_filter(call, qualified, ctx):
            return f"scipy.ndimage.{last} filters with a window centered on each row", None, None
        return None

    @staticmethod
    def _two_sided(key: str) -> tuple[str, str, None]:
        message, advice = _TWO_SIDED_HINTS[key]
        return message, advice, None

    @staticmethod
    def _check_convolve(call: ast.Call, ctx: LintContext) -> tuple[str, str, Confidence] | None:
        mode = _get_argument(call, 2, "mode")
        mode_text = "full" if mode is None else _string_value(mode)
        if mode is not None and mode_text not in {"same", "full"}:
            return None
        if mode_text == "full" and PP003._is_cut_to_length(call, ctx):
            return None  # ``np.convolve(x, w)[:n]`` is the trailing weighted sum: causal
        if mode_text == "same":
            message = "np.convolve(mode='same') centers the kernel on each row and uses later rows"
        else:
            message = (
                "np.convolve in mode 'full' is causal only if the result is cut with [:n]; "
                "any other alignment can use later rows"
            )
        return message, _CONVOLVE_HINT, "medium"

    @staticmethod
    def _is_cut_to_length(call: ast.Call, ctx: LintContext) -> bool:
        """Return True if the call is cut right away to its first rows (``[:n]``, ``[:-k]``).

        A slice that starts later (``[2:-2]``) centers the result on the input, which is
        what mode ``'same'`` does.
        """
        parent = ctx.parent(call)
        if not (isinstance(parent, ast.Subscript) and parent.value is call):
            return False
        cut = parent.slice
        return (
            isinstance(cut, ast.Slice)
            and cut.upper is not None
            and cut.step is None
            and (cut.lower is None or _int_literal(cut.lower) == 0)
        )

    @staticmethod
    def _is_ndimage_filter(call: ast.Call, qualified: str, ctx: LintContext) -> bool:
        """Return True for ``scipy.ndimage.<something>filter<something>(...)``."""
        parts = qualified.split(".")
        last = parts[-1]
        if "filter" not in last:
            return False
        if qualified.startswith("scipy.ndimage."):
            return True
        if isinstance(call.func, ast.Name):
            # A bare name that is never imported here (the import may be in another cell).
            return last in _NDIMAGE_FILTERS and last not in ctx.bound_names
        return (
            len(parts) >= 2 and parts[-2] in {"ndimage", "ndi"} and not ctx.head_is_bound(call.func)
        )


# --------------------------------------------------------------------------- #
# PP004 fit on the full sample
# --------------------------------------------------------------------------- #

_SPLIT_FUNCTIONS = frozenset(
    {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict"}
)
_SPLIT_NAME_TOKENS = frozenset(
    {"train", "training", "test", "testing", "valid", "validation", "val", "split", "fold"}
    | {"cut", "cutoff", "boundary"}
)
_TRAINING_NAME_RE = re.compile(r"train|trn|insample|in_sample|(?:^|_)[xy]?tr(?:$|_)", re.IGNORECASE)
_SUBSET_METHODS = frozenset({"head", "tail", "truncate"})


def _is_row_split(index: ast.expr, split_vars: set[str]) -> bool:
    """Return True if a subscript index selects a block of rows (a train/test cut)."""
    if isinstance(index, ast.Tuple):
        return bool(index.elts) and _is_row_split(index.elts[0], split_vars)
    if isinstance(index, ast.Slice):
        return not (index.lower is None and index.upper is None and index.step is None)
    if isinstance(index, ast.Name):
        return index.id in split_vars or bool(_name_tokens(index.id) & _SPLIT_NAME_TOKENS)
    return False


def _split_variables(nodes: list[ast.AST]) -> set[str]:
    """Return names bound by ``for train_idx, test_idx in splitter.split(...)`` loops."""
    names: set[str] = set()
    for node in nodes:
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and _method_name(node.iter) == "split"
        ):
            names.update(_target_names(node.target))
    return names


def _is_subset_expression(expr: ast.expr, split_vars: set[str]) -> bool:
    """Return True if ``expr`` already selects a part of the rows (``X[:n]``, ``X.head(n)``)."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Subscript) and _is_row_split(node.slice, split_vars):
            return True
        if isinstance(node, ast.Call) and _method_name(node) in _SUBSET_METHODS:
            return True
    return False


class PP004(Rule):
    """A scaler, decomposer or other estimator fitted on data that is split afterwards."""

    id = "PP004"
    name = "fit-on-full-sample"
    confidence = "medium"
    message = "estimator is fitted on the full sample and the data is split afterwards"
    hint = (
        "Split first and fit on the training window only, then transform both parts "
        "(for example fit on X[:split]). Confirm with peekproof.check()."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for scope in _scopes(tree):
            nodes = list(_scope_nodes(scope))
            fits = [n for n in nodes if isinstance(n, ast.Call) and self._is_fit(n)]
            if not fits:
                continue
            split_vars = _split_variables(nodes)
            last_split = self._last_split_lines(nodes, split_vars)
            for call in fits:
                names = self._fitted_names(call, ctx, split_vars)
                if names and self._split_later(last_split, call, names):
                    findings.append(
                        ctx.finding(
                            self,
                            call,
                            message=(
                                f"{_method_name(call)}() runs on the full sample "
                                f"({', '.join(sorted(names))}) that is split afterwards"
                            ),
                            method=True,
                        )
                    )
        return findings

    @staticmethod
    def _is_fit(call: ast.Call) -> bool:
        return _method_name(call) in {"fit", "fit_transform"} and bool(call.args or call.keywords)

    @staticmethod
    def _fitted_names(call: ast.Call, ctx: LintContext, split_vars: set[str]) -> set[str]:
        """Return the names that carry the fitted data, or an empty set to skip the call."""
        fitted = _get_argument(call, 0, "X")
        if fitted is None or isinstance(fitted, ast.Starred):
            return set()
        root = _root_name(fitted)
        if root is None or _TRAINING_NAME_RE.search(root):
            return set()
        if _is_subset_expression(fitted, split_vars):
            return set()
        names = {root}
        if _method_name(call) != "fit_transform":
            return names  # fit() returns the estimator, not transformed data
        node: ast.AST = call
        while True:
            parent = ctx.parent(node)
            if isinstance(parent, (ast.Assign, ast.AnnAssign)):
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                for target in targets:
                    names.update(_target_names(target))
                break
            if parent is None or isinstance(parent, ast.stmt):
                break
            node = parent
        return names

    @staticmethod
    def _last_split_lines(nodes: list[ast.AST], split_vars: set[str]) -> dict[str, int]:
        """Return, per name, the last line where it goes into a split or is cut by rows.

        Collected once per scope, so that a file with many ``fit`` calls is not scanned
        again for each of them.
        """
        last: dict[str, int] = {}
        for node in nodes:
            roots: list[str | None] = []
            if isinstance(node, ast.Call):
                if _last_name(node) in _SPLIT_FUNCTIONS or _method_name(node) == "split":
                    roots = [
                        _root_name(arg) for arg in (*node.args, *(k.value for k in node.keywords))
                    ]
            elif isinstance(node, ast.Subscript) and _is_row_split(node.slice, split_vars):
                roots = [_root_name(node.value)]
            line = int(getattr(node, "lineno", 0))
            for root in roots:
                if root is not None and line > last.get(root, 0):
                    last[root] = line
        return last

    @staticmethod
    def _split_later(last_split: dict[str, int], call: ast.Call, names: set[str]) -> bool:
        after = call.end_lineno or call.lineno
        return any(last_split.get(name, 0) > after for name in names)


# --------------------------------------------------------------------------- #
# PP005 shuffled split
# --------------------------------------------------------------------------- #

_TIME_AWARE_CV_RE = re.compile(
    r"TimeSeries|Purged|Embargo|WalkForward|Walk_Forward|Blocking|Blocked|Combinatorial"
    r"|Temporal|Rolling|Expanding|tscv|ts_cv|time_?split",
    re.IGNORECASE,
)
_CROSS_VALIDATION_FUNCTIONS = frozenset({"cross_val_score", "cross_validate", "cross_val_predict"})
_SEARCH_ESTIMATORS = frozenset(
    {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
)


class PP005(Rule):
    """Shuffled or time-blind splitting of data for evaluation.

    High confidence: ``train_test_split`` without ``shuffle=False``, ``KFold(shuffle=True)`` and
    ``ShuffleSplit``. Medium confidence: ``StratifiedKFold``, ``cross_val_*`` without a
    time-aware ``cv=``, and ``GridSearchCV`` / ``RandomizedSearchCV`` with an integer or missing
    ``cv=``. ``KFold`` without shuffling keeps the rows in order and is not reported.
    """

    id = "PP005"
    name = "shuffled-split"
    confidence = "high"
    message = "rows are split without regard to time order"
    hint = (
        "Split by time: train_test_split(..., shuffle=False), TimeSeriesSplit, "
        "or a purged/embargoed splitter."
    )
    _MEDIUM_HINT = _confirm(
        "Split by time: TimeSeriesSplit, a purged/embargoed splitter, or "
        "train_test_split(..., shuffle=False)."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            qualified = ctx.qualname(node.func)
            if qualified is None or not ctx.is_from(node.func, "sklearn"):
                continue
            last = _last_segment(qualified)
            found = self._check(node, last, ctx)
            if found is not None:
                message, confidence = found
                hint = None if confidence == "high" else self._MEDIUM_HINT
                findings.append(
                    ctx.finding(self, node, message=message, hint=hint, confidence=confidence)
                )
        return findings

    def _check(self, call: ast.Call, last: str, ctx: LintContext) -> tuple[str, Confidence] | None:
        """Return ``(message, confidence)`` if this call splits data without regard to time."""
        if last == "train_test_split":
            shuffle = _get_argument(call, None, "shuffle")
            if _has_star_kwargs(call) or not (shuffle is None or _is_truthy_constant(shuffle)):
                return None
            return "train_test_split shuffles rows by default (future rows reach training)", "high"
        if last == "KFold":
            if _is_truthy_constant(_get_argument(call, None, "shuffle")):
                return "KFold(shuffle=True) mixes future rows into every training fold", "high"
            return None
        if last.endswith("ShuffleSplit"):
            return f"{last} draws random rows and ignores time order", "high"
        if last == "StratifiedKFold":
            return "StratifiedKFold ignores time order", "medium"
        if last in _CROSS_VALIDATION_FUNCTIONS and not _has_star_kwargs(call):
            cv = _get_argument(call, None, "cv")
            if cv is None or not self._is_time_aware(cv, ctx):
                return (
                    f"{last} without a time-aware cv= trains on rows after the test rows",
                    "medium",
                )
        if last in _SEARCH_ESTIMATORS and not _has_star_kwargs(call):
            cv = _get_argument(call, None, "cv")
            default = cv is None or (isinstance(cv, ast.Constant) and cv.value is None)
            if default or _int_literal(cv) is not None:
                return (
                    f"{last} with an integer or missing cv= validates on folds that train "
                    "on rows after the validation rows",
                    "medium",
                )
        return None

    def _is_time_aware(self, cv: ast.expr, ctx: LintContext) -> bool:
        if _TIME_AWARE_CV_RE.search(ctx.segment(cv)):
            return True
        if isinstance(cv, (ast.Name, ast.Attribute)):
            return any(
                _TIME_AWARE_CV_RE.search(ctx.segment(value))
                for value in ctx.assignments.get(expr_key(cv), [])
            )
        return False


# --------------------------------------------------------------------------- #
# PP006 full-sample statistic
# --------------------------------------------------------------------------- #

_AGGREGATES = frozenset({"mean", "std", "var", "min", "max", "median", "quantile"})
_ARITHMETIC_METHODS = frozenset(
    {"add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"}
    | {"radd", "rsub", "rmul", "rdiv", "rtruediv", "gt", "ge", "lt", "le", "eq", "ne"}
    | {"clip", "between"}
)
_WINDOW_METHODS = frozenset({"rolling", "expanding", "ewm"})
_TIME_KEY_TOKENS = frozenset(
    {"date", "dates", "datetime", "time", "timestamp", "ts", "day", "dt", "period", "month"}
    | {"week", "year", "bar", "index", "level"}
)
# What turns a series into a single number or a length: ``len(x)``, ``x.count()``, ``x.shape``.
# An ``x`` that only appears like this is not the series itself, as in the t-statistic
# ``x.mean() / x.std() * np.sqrt(len(x))``.
_SCALAR_METHODS = _AGGREGATES | {"sum", "prod", "count", "nunique", "sem", "skew", "kurt"}
_SCALAR_METHODS |= {"kurtosis", "idxmax", "idxmin", "any", "all"}
_SCALAR_ATTRIBUTES = frozenset(
    {"size", "shape", "ndim", "dtype", "dtypes", "name", "nbytes", "empty", "index", "columns"}
)
_SIZE_FUNCTIONS = frozenset({"len", "size", "shape"})
_SCALE_FUNCTIONS = frozenset(
    {"scale", "minmax_scale", "robust_scale", "maxabs_scale", "power_transform"}
    | {"quantile_transform"}
)

_KeyTable = dict[tuple[object, ...], int]


def _plain(value: object) -> object:
    """Return a hashable stand-in for a field of a node that is not itself a node."""
    if value is None or isinstance(value, (str, bytes, int, float, complex)):
        return (type(value).__name__, value)
    return repr(value)


def _count_between(positions: list[int], low: int, high: int) -> int:
    """Return how many of the sorted ``positions`` lie in ``[low, high)``."""
    return bisect_left(positions, high) - bisect_left(positions, low)


class _Anchor(NamedTuple):
    """A full-sample statistic in an expression and the series it was computed from."""

    position: int  # pre-order position of ``node`` in the expression
    node: ast.AST  # where a finding points: the aggregate call, or the name that holds it
    keys: frozenset[int]  # structural keys of the series that the statistic summarises
    label: str  # "mean()", "std()", ...
    name: str | None  # the variable that holds the statistic; ``None`` for a direct call
    root: str | None  # the variable at the start of the summarised series, if it has one


class _ExprIndex:
    """Pre-order index of one expression tree.

    For every node it records the extent of its subtree, its parent and a *structural
    key*: two nodes get the same key exactly when :func:`expr_key` would give them the
    same text (quotes, spacing and the load/store context are ignored). Keys are numbers
    handed out by a table that is shared between expressions, and every node is visited
    once, so building an index takes time proportional to the size of the expression.
    Nothing here is recursive, so deeply nested expressions are fine.
    """

    def __init__(self, root: ast.AST, table: _KeyTable) -> None:
        nodes: list[ast.AST] = []
        parents: list[int] = []
        stack: list[tuple[ast.AST, int]] = [(root, -1)]
        while stack:
            node, parent = stack.pop()
            position = len(nodes)
            nodes.append(node)
            parents.append(parent)
            stack.extend((child, position) for child in reversed(list(ast.iter_child_nodes(node))))
        sizes = [1] * len(nodes)
        for position in range(len(nodes) - 1, 0, -1):
            sizes[parents[position]] += sizes[position]
        key_of: dict[ast.AST, int] = {}
        keys = [0] * len(nodes)
        for position in range(len(nodes) - 1, -1, -1):
            node = nodes[position]
            parts: list[object] = [type(node).__name__]
            for field, value in ast.iter_fields(node):
                if field in {"ctx", "kind"}:
                    continue
                if isinstance(value, ast.AST):
                    parts.append(key_of[value])
                elif isinstance(value, list):
                    parts.append(
                        tuple(key_of[v] if isinstance(v, ast.AST) else _plain(v) for v in value)
                    )
                else:
                    parts.append(_plain(value))
            keys[position] = key_of[node] = table.setdefault(tuple(parts), len(table))
        self.nodes = nodes
        self.parents = parents
        self.sizes = sizes
        self.keys = keys

    def aggregate_anchors(self) -> list[_Anchor]:
        """Return an anchor for every ``x.mean()``-style call that works down the rows."""
        anchors: list[_Anchor] = []
        for position, node in enumerate(self.nodes):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _AGGREGATES
                and not _axis_is_columns(node)
            ):
                # Pre-order: the call, then ``func``, then the receiver (``func.value``).
                receiver = position + 2
                if receiver < len(self.nodes) and self.nodes[receiver] is node.func.value:
                    anchors.append(
                        _Anchor(
                            position,
                            node,
                            frozenset({self.keys[receiver]}),
                            f"{node.func.attr}()",
                            None,
                            _root_name(node.func.value),
                        )
                    )
        return anchors

    def stat_anchors(self, stats: _ScopeStats) -> list[_Anchor]:
        """Return an anchor for every name in the expression that holds a full-sample statistic."""
        if not stats.names:
            return []
        shadowed: set[str] | None = None
        anchors: list[_Anchor] = []
        for position, node in enumerate(self.nodes):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
                continue
            if node.id not in stats.names:
                continue
            stat = stats.lookup(node.id, node)
            if stat is None:
                continue
            if shadowed is None:
                shadowed = self._bound_names()
            if node.id not in shadowed:
                anchors.append(_Anchor(position, node, stat.keys, stat.label, node.id, None))
        return anchors

    def _bound_names(self) -> set[str]:
        """Return names that the expression itself binds (lambda arguments, loop variables)."""
        bound: set[str] = set()
        for node in self.nodes:
            if isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.comprehension):
                bound.update(_target_names(node.target))
            elif isinstance(node, ast.NamedExpr):
                bound.add(node.target.id)
        return bound

    def is_scalarised(self, position: int) -> bool:
        """Return True if the node only feeds a length or a single statistic."""
        current = position
        while True:
            up = self.parents[current]
            if up < 0:
                return False
            node, parent = self.nodes[current], self.nodes[up]
            if isinstance(parent, ast.Attribute) and parent.value is node:
                if parent.attr in _SCALAR_ATTRIBUTES:
                    return True
                above = self.parents[up]
                call = self.nodes[above] if above >= 0 else None
                if isinstance(call, ast.Call) and call.func is parent:
                    if parent.attr in _SCALAR_METHODS:
                        return True
                    current = above
                else:
                    current = up
            elif isinstance(parent, ast.Subscript) and parent.value is node:
                current = up
            elif isinstance(parent, ast.Call) and any(arg is node for arg in parent.args):
                if _last_name(parent) in _SIZE_FUNCTIONS:
                    return True
                current = up
            else:
                return False

    def series_uses(self, wanted: set[int], anchors: list[_Anchor]) -> dict[int, list[int]]:
        """Return, per key, the sorted positions where that series itself is used.

        Uses inside the receiver of a statistic (the ``x`` of ``x.mean()``) and uses that
        only feed a length (``len(x)``) do not count.
        """
        covered = [0] * (len(self.nodes) + 1)
        for anchor in anchors:
            if anchor.name is None:
                receiver = anchor.position + 2
                covered[receiver] += 1
                covered[receiver + self.sizes[receiver]] -= 1
        uses: dict[int, list[int]] = {}
        inside = 0
        for position, key in enumerate(self.keys):
            inside += covered[position]
            if key in wanted and inside <= 0 and not self.is_scalarised(position):
                uses.setdefault(key, []).append(position)
        return uses


def _may_hold_statistic(value: ast.AST, names: set[str]) -> bool:
    """Cheap test: does ``value`` call ``.mean()``-style aggregates or read one of ``names``?"""
    for node in ast.walk(value):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _AGGREGATES:
                return True
        elif isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load):
            return True
    return False


def _is_group(node: ast.AST, ctx: LintContext) -> bool:
    """Return True for an expression that combines its operands arithmetically."""
    if isinstance(node, (ast.BinOp, ast.Compare)):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in _ARITHMETIC_METHODS
        return ctx.qualname(node.func) == "numpy.clip"
    return False


def _combined_anchors(
    index: _ExprIndex, anchors: list[_Anchor], partner_keys: frozenset[int], ctx: LintContext
) -> list[_Anchor]:
    """Return the anchors whose series is used in another operand of an arithmetic expression.

    ``(x - x.mean()) / x.std()`` qualifies: ``x.mean()`` sits in an expression whose other
    side contains ``x`` again. ``x.mean() / x.std()`` does not, because only scalars are
    combined. A name that holds a statistic (``mu = x.mean()`` earlier in the scope) counts
    like the call itself. Series are compared by structure, so quoting and spacing do not
    matter. The partner of an augmented assignment (``x -= ...``) is one more operand.
    """
    wanted: set[int] = set()
    for anchor in anchors:
        wanted |= anchor.keys
    uses = index.series_uses(wanted, anchors)
    hits: list[_Anchor] = []
    for anchor in anchors:
        keys = [key for key in anchor.keys if key in uses]
        if anchor.keys & partner_keys or (keys and _meets_series(index, anchor, keys, uses, ctx)):
            hits.append(anchor)
    return hits


def _meets_series(
    index: _ExprIndex,
    anchor: _Anchor,
    keys: list[int],
    uses: dict[int, list[int]],
    ctx: LintContext,
) -> bool:
    """Return True if an expression above the anchor combines it with the series (other operand)."""
    child = anchor.position
    parent = index.parents[child]
    while parent >= 0:
        owned = {key: _count_between(uses[key], child, child + index.sizes[child]) for key in keys}
        if all(owned[key] == len(uses[key]) for key in keys):
            return False  # every use lies on the anchor's own side, here and further up
        if _is_group(index.nodes[parent], ctx) and any(
            _count_between(uses[key], parent, parent + index.sizes[parent]) > owned[key]
            for key in keys
        ):
            return True
        child, parent = parent, index.parents[parent]
    return False


class _Stat(NamedTuple):
    keys: frozenset[int]
    label: str


class _ScopeStats:
    """Names bound to full-sample statistics in one scope, such as ``mu = x.mean()``.

    A statistic is an expression that holds aggregates of a series (or of other
    statistics) but never the series itself. A statistic of a training set
    (``train.mean()``) is left out: using it on that same set is the correct way.
    """

    def __init__(self, scope: ast.AST, table: _KeyTable) -> None:
        self.names: set[str] = set()  # names that hold a statistic somewhere in the scope
        self._where: dict[str, list[tuple[int, int]]] = {}
        self._what: dict[str, list[_Stat | None]] = {}
        self._table = table
        for node in _scope_nodes(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._bind(target, node.value, node)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._bind(node.target, node.value, node)
            elif isinstance(node, ast.AugAssign):
                self._record(node.target, None, node)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._record(node.target, None, node.target)

    def _bind(self, target: ast.expr, value: ast.expr, statement: ast.AST) -> None:
        for sub_target, sub_value in _assignment_pairs(target, value):
            stat = self._statistic(sub_value) if isinstance(sub_target, ast.Name) else None
            self._record(sub_target, stat, statement)

    def _record(self, target: ast.expr, stat: _Stat | None, ends_at: ast.AST) -> None:
        """Note that ``target`` is bound (to ``stat``) from the end of ``ends_at`` on."""
        if isinstance(target, ast.Name):
            end = (
                int(getattr(ends_at, "end_lineno", None) or 0),
                int(getattr(ends_at, "end_col_offset", None) or 0),
            )
            self._where.setdefault(target.id, []).append(end)
            self._what.setdefault(target.id, []).append(stat)
            if stat is not None:
                self.names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List, ast.Starred)):
            for element in ast.iter_child_nodes(target):
                if isinstance(element, ast.expr):
                    self._record(element, None, ends_at)

    def lookup(self, name: str, node: ast.AST) -> _Stat | None:
        """Return the statistic that ``name`` holds where ``node`` reads it, if it holds one."""
        where = self._where.get(name)
        if not where:
            return None
        at = (int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0)))
        found = bisect_left(where, at) - 1
        return self._what[name][found] if found >= 0 else None

    def _statistic(self, value: ast.expr) -> _Stat | None:
        if not _may_hold_statistic(value, self.names):
            return None
        index = _ExprIndex(value, self._table)
        anchors = index.aggregate_anchors()
        if any(a.root is not None and _TRAINING_NAME_RE.search(a.root) for a in anchors):
            return None
        anchors += index.stat_anchors(self)
        if not anchors:
            return None
        keys: set[int] = set()
        for anchor in anchors:
            keys |= anchor.keys
        if index.series_uses(keys, anchors):
            return None  # the series itself is in there, so this is a series, not a statistic
        return _Stat(frozenset(keys), anchors[0].label)


def _is_windowed(expr: ast.expr) -> bool:
    """Return True if a rolling / expanding / ewm call sits in the receiver chain of ``expr``."""
    current: ast.expr | None = expr
    while current is not None:
        if isinstance(current, ast.Call):
            if _method_name(current) in _WINDOW_METHODS:
                return True
            func = current.func
            current = func.value if isinstance(func, ast.Attribute) else None
        elif isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        else:
            return False
    return False


def _has_groupby(value: ast.AST, ctx: LintContext, *, time_key_only: bool) -> bool:
    """Return True if ``value`` calls ``groupby`` (with a date/time-like key, if asked)."""
    for node in ast.walk(value):
        if not (isinstance(node, ast.Call) and _method_name(node) == "groupby"):
            continue
        if not time_key_only:
            return True
        keys = [ctx.segment(arg) for arg in node.args]
        keys += [f"{kw.arg} {ctx.segment(kw.value)}" for kw in node.keywords]
        if _name_tokens(" ".join(keys)) & _TIME_KEY_TOKENS:
            return True
    return False


def _whole_series_transform(value: ast.expr, ctx: LintContext) -> tuple[ast.Call, str, bool] | None:
    """Find a rank, z-score, scaling or quantile binning that looks at the whole series."""
    if _has_groupby(value, ctx, time_key_only=False):
        return None
    best: tuple[ast.Call, str, bool] | None = None
    for node in ast.walk(value):
        if not isinstance(node, ast.Call) or _axis_is_columns(node):
            continue
        found = _transform_of(node, ctx)
        if found is not None and (best is None or _node_position(node) < _node_position(best[0])):
            best = found
    return best


def _transform_of(node: ast.Call, ctx: LintContext) -> tuple[ast.Call, str, bool] | None:
    """Return ``(call, message, anchor_on_method)`` if ``node`` is a whole-series transform."""
    if (
        _method_name(node) == "rank"
        and isinstance(node.func, ast.Attribute)
        and not _is_windowed(node.func.value)
    ):
        return node, "rank() over the whole series uses every row, including later ones", True
    qualified = ctx.qualname(node.func)
    if qualified is None:
        return None
    last = _last_segment(qualified)
    everything = "uses every row, including later ones"
    if last == "zscore" and ctx.is_from(node.func, "scipy"):
        return node, f"zscore() over the whole series {everything}", False
    if last in _SCALE_FUNCTIONS and ctx.is_from(node.func, "sklearn"):
        return (
            node,
            f"{last}() standardizes with statistics of the whole series; it {everything}",
            False,
        )
    if last == "qcut" and ctx.is_from(node.func, "pandas"):
        return node, f"qcut() bins by quantiles of the whole series; it {everything}", False
    return None


class PP006(Rule):
    """A whole-series statistic combined with the series itself, or a whole-series transform.

    The result must be kept: assigned to a variable or column, or returned. Reported are
    ``(x - x.mean()) / x.std()``, the same with the statistics stored first (``mu = x.mean()``
    earlier in the scope), ``x.clip(x.quantile(...), ...)``, and a whole-series ``rank()``,
    ``zscore()``, ``sklearn.preprocessing.scale()`` or ``pd.qcut()``. A statistic that is not
    combined with the series itself is not: a t-statistic only uses ``len(x)``.
    """

    id = "PP006"
    name = "full-sample-statistic"
    confidence = "medium"
    message = "full-sample statistic uses every row, including later ones"
    hint = (
        "Use expanding()/rolling() statistics (shifted by one bar if needed) so each row "
        "sees only the past. Confirm with peekproof.check()."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        table: _KeyTable = {}
        stats: dict[ast.AST, _ScopeStats] = {}
        for value, partner in _result_values(tree):
            if _has_groupby(value, ctx, time_key_only=True):
                continue  # a per-date (cross-sectional) statistic does not look into the future
            candidates: list[tuple[ast.AST, str, bool]] = []
            combined = self._self_combined(value, partner, ctx, table, stats)
            if combined is not None:
                candidates.append(combined)
            ranked = _whole_series_transform(value, ctx)
            if ranked is not None:
                candidates.append(ranked)
            if candidates:
                node, text, as_method = min(candidates, key=lambda c: _node_position(c[0]))
                findings.append(ctx.finding(self, node, message=text, method=as_method))
        return findings

    @staticmethod
    def _self_combined(
        value: ast.expr,
        partner: ast.expr | None,
        ctx: LintContext,
        table: _KeyTable,
        stats: dict[ast.AST, _ScopeStats],
    ) -> tuple[ast.AST, str, bool] | None:
        scope = ctx.enclosing_scope(value)
        if scope not in stats:
            stats[scope] = _ScopeStats(scope, table)
        if not _may_hold_statistic(value, stats[scope].names):
            return None
        index = _ExprIndex(value, table)
        anchors = index.aggregate_anchors()
        anchors += index.stat_anchors(stats[scope])
        if not anchors:
            return None
        partner_keys = (
            frozenset(_ExprIndex(partner, table).keys) if partner is not None else frozenset()
        )
        hits = _combined_anchors(index, anchors, partner_keys, ctx)
        if not hits:
            return None
        first = min(hits, key=lambda anchor: _node_position(anchor.node))
        if first.name is None:
            return first.node, f"full-sample {first.label} is combined with the same series", True
        text = (
            f"'{first.name}' holds a full-sample {first.label} and is combined with the same series"
        )
        return first.node, text, False


# --------------------------------------------------------------------------- #
# PP007 forward as-of join / period-end join keys
# --------------------------------------------------------------------------- #

# "fy" only counts as a word of its own, so that "verify_id" is not a fiscal-year key.
_PERIOD_KEY_RE = re.compile(
    r"period_end|fiscal|report_date|quarter_end|(?<![a-z])fy(?![a-z])", re.IGNORECASE
)
_FILING_KEY_RE = re.compile(r"filed|filing|accepted|available|publish", re.IGNORECASE)
_DIRECTION_KEYWORDS = {"merge_asof": "direction", "join_asof": "strategy"}


class PP007(Rule):
    """Forward as-of joins (high) and joins on period-end keys (low)."""

    id = "PP007"
    name = "forward-asof-join"
    confidence = "high"
    message = "as-of join can attach rows from after the join date"
    hint = (
        "Use direction='backward' and join on the date the data became available "
        "(filing or publication date), not on the period it describes."
    )
    _PERIOD_HINT = _confirm(
        "Join on the date the data became available (filing or publication date), "
        "not on the period it describes."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            found = self._kind(node, ctx)
            if found is None:
                continue
            kind, on_method = found
            direction_keyword = _DIRECTION_KEYWORDS.get(kind)
            if direction_keyword is not None:
                direction = _string_value(_get_argument(node, None, direction_keyword))
                if direction in {"forward", "nearest"}:
                    findings.append(
                        ctx.finding(
                            self,
                            node,
                            message=(
                                f"{kind} with {direction_keyword}={direction!r} "
                                "can attach rows from after the join date"
                            ),
                            method=on_method,
                        )
                    )
                    continue
            keys: list[str] = []
            for keyword in ("on", "left_on", "right_on"):
                keys.extend(_string_values(_get_argument(node, None, keyword)))
            period_keys = [k for k in keys if _PERIOD_KEY_RE.search(k)]
            if period_keys and not any(_FILING_KEY_RE.search(k) for k in keys):
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=(
                            f"{kind} on period key {period_keys[0]!r} without a filing date: "
                            "the data may not have been public at that date"
                        ),
                        hint=self._PERIOD_HINT,
                        confidence="low",
                        method=on_method,
                    )
                )
        return findings

    @staticmethod
    def _kind(call: ast.Call, ctx: LintContext) -> tuple[str, bool] | None:
        """Return ``(kind, anchor_on_method_name)`` for a merge-like call, else ``None``."""
        qualified = ctx.qualname(call.func)
        method = _method_name(call)
        if qualified is not None and _last_segment(qualified) == "merge_asof":
            return ("merge_asof", False) if ctx.is_from(call.func, "pandas") else None
        if method == "join_asof":
            return "join_asof", True
        if qualified == "pandas.merge":
            return "merge", False
        if method == "merge":
            return "merge", True
        return None


# --------------------------------------------------------------------------- #
# PP008 bar label leak
# --------------------------------------------------------------------------- #

# Frequencies that pandas labels with the end of the period by default (month, quarter,
# year and week ends); "MS", "QS", "YS" and the intraday frequencies are labeled at the start.
_RIGHT_LABELLED_RULE_RE = re.compile(
    r"\d*(?:B?(?:ME?|QE?|AE?|YE?)|W)(?:-[A-Za-z]{3})?", re.IGNORECASE
)
_BAR_AGGREGATIONS = frozenset({"last", "ohlc", "agg", "aggregate"})


def _last_load_lines(scope: ast.AST) -> dict[str, int]:
    """Return, for every name that one scope reads, the last line on which it is read."""
    last: dict[str, int] = {}
    for node in _scope_nodes(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            last[node.id] = max(last.get(node.id, 0), node.lineno)
    return last


class PP008(Rule):
    """``resample(...).last()/ohlc()/agg()`` whose result feeds later code."""

    id = "PP008"
    name = "bar-label-leak"
    confidence = "low"
    message = "resampled bar is labeled with its start time"
    hint = (
        "Stamp bars with their close time (label='right', closed='right') and shift "
        "the result before using it as a feature. Confirm with peekproof.check()."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        loads: dict[ast.AST, dict[str, int]] = {}  # per scope, computed once however many bars
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _BAR_AGGREGATIONS
            ):
                continue
            resample = node.func.value
            if not (isinstance(resample, ast.Call) and _method_name(resample) == "resample"):
                continue
            if self._is_right_labeled(resample):
                continue
            if not self._feeds_further_code(node, ctx, loads):
                continue
            findings.append(
                ctx.finding(
                    self,
                    node,
                    message=(
                        f"resample(...).{_method_name(node)}() labels each bar with its start "
                        "time, so the bar's last value appears too early"
                    ),
                    method=True,
                )
            )
        return findings

    @staticmethod
    def _is_right_labeled(resample: ast.Call) -> bool:
        keywords = {kw.arg: _string_value(kw.value) for kw in resample.keywords if kw.arg}
        if "right" in (keywords.get("label"), keywords.get("closed")):
            return True
        if "left" in (keywords.get("label"), keywords.get("closed")):
            return False
        rule = _string_value(_get_argument(resample, 0, "rule"))
        return rule is not None and _RIGHT_LABELLED_RULE_RE.fullmatch(rule) is not None

    @staticmethod
    def _feeds_further_code(
        node: ast.Call, ctx: LintContext, loads: dict[ast.AST, dict[str, int]]
    ) -> bool:
        parent = ctx.parent(node)
        if isinstance(parent, (ast.Expr, ast.Return)):
            return False
        if not isinstance(parent, (ast.Assign, ast.AnnAssign)):
            return True
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        if not all(isinstance(t, ast.Name) for t in targets):
            return True
        names = {t.id for t in targets if isinstance(t, ast.Name)}
        scope = ctx.enclosing_scope(parent)
        if scope not in loads:
            loads[scope] = _last_load_lines(scope)
        after = parent.end_lineno or parent.lineno
        return any(loads[scope].get(name, 0) > after for name in names)


# --------------------------------------------------------------------------- #
# PP009 forward index in a loop
# --------------------------------------------------------------------------- #


def _offset_of(expr: ast.expr, var: str) -> int | None:
    """Return ``k`` if ``expr`` is ``var``, ``var + k``, ``k + var`` or ``var - k``."""
    if isinstance(expr, ast.Name) and expr.id == var:
        return 0
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
        left, right = expr.left, expr.right
        if isinstance(left, ast.Name) and left.id == var:
            value = _int_literal(right)
            if value is not None:
                return value if isinstance(expr.op, ast.Add) else -value
        if isinstance(expr.op, ast.Add) and isinstance(right, ast.Name) and right.id == var:
            return _int_literal(left)
    return None


def _index_offsets(index: ast.expr, var: str) -> list[int]:
    """Return the loop-variable offsets of the elements of a subscript index."""
    elements = index.elts if isinstance(index, ast.Tuple) else [index]
    offsets = [_offset_of(element, var) for element in elements]
    return [o for o in offsets if o is not None]


def _slice_reach(sub: ast.Subscript, var: str) -> int | None:
    """Return the offset of the last row that ``x[a:var + k]`` reads, relative to row ``var``.

    ``x[i:i + 3]`` and ``x.iloc[i:i + 3]`` end before ``i + 3``, so their last row is
    ``i + 2``; ``x.loc[i:i + 3]`` includes ``i + 3``. Only the row axis (the first index of a
    tuple) is looked at, and only a slice with an upper bound of the form ``var + k``.
    """
    index = sub.slice
    rows = index.elts[0] if isinstance(index, ast.Tuple) and index.elts else index
    if not isinstance(rows, ast.Slice) or rows.upper is None:
        return None
    upper = _offset_of(rows.upper, var)
    if upper is None:
        return None
    by_label = isinstance(sub.value, ast.Attribute) and sub.value.attr == "loc"
    return upper if by_label else upper - 1


def _loop_nodes(loop: ast.For, var: str) -> Iterator[ast.AST]:
    """Yield the nodes in a loop body, skipping nested scopes and loops that rebind ``var``."""
    stack: list[ast.AST] = list(loop.body)
    stack.reverse()
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, _SCOPE_TYPES):
            continue
        if isinstance(node, (ast.For, ast.AsyncFor)) and var in _target_names(node.target):
            continue
        children = list(ast.iter_child_nodes(node))
        children.reverse()
        stack.extend(children)


def _is_range_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
    )


def _steps_over_blocks(node: ast.expr) -> bool:
    """Return True for ``range(a, b, step)`` whose step is not 1: ``i`` then starts a block.

    ``for i in range(0, n, 32): batch = x[i:i + 32]`` walks over batches, not over rows.
    """
    if not isinstance(node, ast.Call) or len(node.args) < 3:
        return False
    return _int_literal(node.args[2]) != 1


class PP009(Rule):
    """``for i in range(...)`` loops and comprehensions that read ahead of row ``i``.

    High confidence: ``x[i + k]`` on the right-hand side of an assignment to row ``i``, in
    ``out.append(...)`` and in a list or dict comprehension. Medium confidence: a slice that
    runs on past row ``i`` (``x[i:i + k]``). Label functions are exempt.
    """

    id = "PP009"
    name = "forward-index-in-loop"
    confidence = "high"
    message = "loop reads a later row while computing row i"
    hint = "Index backwards (x[i - k]) or compute the feature with a vectorised shift(k), k > 0."
    _SLICE_HINT = _confirm(
        "Use a trailing window (x[i - k + 1:i + 1] or rolling(k)) instead of a slice that "
        "runs on past row i."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        found: list[Finding] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.target, ast.Name) and _is_range_call(node.iter):
                    blocks = _steps_over_blocks(node.iter)
                    found.extend(self._check_loop(node, node.target.id, ctx, windows=not blocks))
            elif isinstance(node, (ast.ListComp, ast.DictComp)):
                found.extend(self._check_comprehension(node, ctx))
        return [finding for finding in found if not ctx.in_label_function(finding.line)]

    def _check_loop(
        self, loop: ast.For, var: str, ctx: LintContext, *, windows: bool
    ) -> list[Finding]:
        findings: list[Finding] = []
        for node in _loop_nodes(loop, var):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AugAssign):
                targets, value = [node.target], node.value
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and _method_name(node.value) in {"append", "appendleft"}
            ):
                for argument in node.value.args:  # ``out.append(x[i + 1])`` builds row i
                    findings.extend(self._report(argument, var, 0, ctx, windows=windows))
                continue
            else:
                continue
            offsets = [
                offset
                for target in targets
                for sub in ast.walk(target)
                if isinstance(sub, ast.Subscript)
                for offset in _index_offsets(sub.slice, var)
            ]
            if offsets:  # an assignment to row i (or i + k): later rows must not be read
                findings.extend(self._report(value, var, max(offsets), ctx, windows=windows))
            elif windows:  # a name is assigned: only a window that runs on past row i gives it away
                findings.extend(self._report(value, var, 0, ctx, elements=False))
        return findings

    def _check_comprehension(
        self, node: ast.ListComp | ast.DictComp, ctx: LintContext
    ) -> list[Finding]:
        """Check ``[x[i + 1] ... for i in range(n)]``: element ``i`` is built from row ``i + 1``.

        Only list and dict comprehensions are checked. They build one entry per
        row, unlike a generator that is reduced to a single value by ``sum`` or ``any``.
        """
        if isinstance(node, ast.DictComp):
            parts: list[ast.expr] = [node.key, node.value]
        else:
            parts = [node.elt]
        findings: list[Finding] = []
        for generator in node.generators:
            if isinstance(generator.target, ast.Name) and _is_range_call(generator.iter):
                windows = not _steps_over_blocks(generator.iter)
                for part in parts:
                    findings.extend(
                        self._report(part, generator.target.id, 0, ctx, windows=windows)
                    )
        return findings

    def _report(
        self,
        value: ast.expr,
        var: str,
        written: int,
        ctx: LintContext,
        *,
        elements: bool = True,
        windows: bool = True,
    ) -> list[Finding]:
        """Return a finding for the first subscript in ``value`` that reads past row ``written``.

        An element ``x[i + k]`` is high confidence. A slice ``x[i:i + k]`` that runs on past
        the row is medium confidence and only reported if no element read is found. With
        ``elements=False`` only slices are looked at, and with ``windows=False`` only elements.
        """
        window: tuple[ast.Subscript, int] | None = None
        for sub in sorted(_subscripts(value), key=_node_position):
            ahead = [o for o in _index_offsets(sub.slice, var) if o > 0 and o > written]
            if ahead and elements:
                message = (
                    f"{ctx.segment(sub)} reads row {var}+{max(ahead) - written} "
                    f"while computing row {var}"
                )
                return [ctx.finding(self, sub, message=message)]
            reach = _slice_reach(sub, var) if windows else None
            if window is None and reach is not None and reach > 0 and reach > written:
                window = (sub, reach)
        if window is None:
            return []
        sub, reach = window
        message = (
            f"{ctx.segment(sub)} reads rows up to {var}+{reach - written} while computing row {var}"
        )
        return [ctx.finding(self, sub, message=message, hint=self._SLICE_HINT, confidence="medium")]


def _subscripts(node: ast.AST) -> list[ast.Subscript]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Subscript)]


# --------------------------------------------------------------------------- #
# PP010 same-bar execution
# --------------------------------------------------------------------------- #

_RETURNS_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:ret|return)s?(?![A-Za-z0-9])|pct_change\(", re.IGNORECASE
)
_FORWARD_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:fwd|forward|next|ahead|future)(?![A-Za-z0-9])", re.IGNORECASE
)
_SIGNAL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:signal|position|pos|weight|side|exposure)s?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# A forecast or an expected return is not a realised one: ``weights * expected_returns``.
_FORECAST_RE = re.compile(r"expected|exp_|forecast|mu_|alpha", re.IGNORECASE)
# ``np.log(close).diff()`` and ``close.diff()`` are returns, although no name says so.
_PRICE_TOKENS = frozenset({"log", "ln", "price", "prices", "px", "close", "closes", "adjclose"})


class PP010(Rule):
    """A signal or position multiplied by returns without a one-bar lag.

    Returns are names such as ``ret`` or ``returns``, ``pct_change()``, and the difference of
    prices (``close.diff()``, ``np.log(close).diff()``). Shifted or forward-looking operands,
    forecasts (``expected_returns``, ``alpha``) and names assigned from a shift are skipped.
    """

    id = "PP010"
    name = "same-bar-execution"
    confidence = "medium"
    message = "signal is multiplied by returns of the same bar"
    hint = (
        "Lag the position by one bar (position.shift(1) * returns) so a signal only earns "
        "the next bar's return. Confirm with peekproof.check()."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                continue
            for returns_side, signal_side in ((node.left, node.right), (node.right, node.left)):
                # The signal side is the cheaper test: a long product chain is never a signal.
                if self._is_signal(signal_side, ctx) and self._is_returns(returns_side, ctx):
                    if _FORECAST_RE.search(ctx.segment(node)):
                        break
                    findings.append(
                        ctx.finding(
                            self,
                            node,
                            message=(
                                f"'{ctx.segment(signal_side)}' is multiplied by "
                                f"'{ctx.segment(returns_side)}' without a one-bar lag"
                            ),
                        )
                    )
                    break
        return findings

    @staticmethod
    def _looks_ahead(expr: ast.AST) -> bool:
        """Return True if ``expr`` calls ``pct_change`` or ``diff`` with a negative literal."""
        for node in ast.walk(expr):
            if (
                isinstance(node, ast.Call)
                and _method_name(node) in {"pct_change", "diff"}
                and _is_negative_literal(_get_argument(node, 0, "periods", "n"))
            ):
                return True
        return False

    @staticmethod
    def _is_returns(node: ast.expr, ctx: LintContext) -> bool:
        # Text checks come first: they are cheap, and an operand can be a very long expression.
        text = ctx.segment(node)
        if ".shift(" in text or _FORWARD_RE.search(text):
            return False  # lagged or forward-shifted: ``returns.shift(-1)`` is label code (PP001)
        if _RETURNS_RE.search(text) is not None:
            if ("pct_change(" in text or "diff(" in text) and PP010._looks_ahead(node):
                return False
            if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
                # ``returns = prices.pct_change().shift(-1)`` earlier on is a shifted name too
                earlier = ctx.assignments.get(expr_key(node), [])
                return not any(
                    ".shift(" in ctx.segment(value) or PP010._looks_ahead(value)
                    for value in earlier
                )
            return True
        return "diff" in text and PP010._is_price_difference(node, ctx)

    @staticmethod
    def _is_price_difference(node: ast.expr, ctx: LintContext) -> bool:
        """Return True for ``np.log(close).diff()``, ``close.diff()`` or ``np.diff(prices)``."""
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            if sub.args and ctx.qualname(sub.func) == "numpy.diff":
                subject, periods = sub.args[0], _get_argument(sub, 1, "n")
            elif isinstance(sub.func, ast.Attribute) and sub.func.attr == "diff":
                subject, periods = sub.func.value, _get_argument(sub, 0, "periods", "n")
            else:
                continue
            if (
                not _is_negative_literal(periods)
                and _name_tokens(ctx.segment(subject)) & _PRICE_TOKENS
            ):
                return True
        return False

    @staticmethod
    def _is_signal(node: ast.expr, ctx: LintContext) -> bool:
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
            return False
        text = ctx.segment(node)
        if ".shift(" in text:
            return False
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            if not _SIGNAL_RE.search(text):
                return False
            earlier = ctx.assignments.get(expr_key(node), [])
            return not any(".shift(" in ctx.segment(value) for value in earlier)
        return isinstance(node, ast.Call) and ctx.qualname(node.func) == "numpy.sign"


# --------------------------------------------------------------------------- #
# PP011 target as feature
# --------------------------------------------------------------------------- #

# Label columns. ``y`` only counts in lower case (a year column "Y" is no label), and there is
# no generic "_y" suffix: that is what ``merge`` appends to a column name that clashes.
_LABEL_COLUMN_RE = re.compile(
    r"(?i:^(?:target|label|y_true)$|_(?:target|label|y_true)$"
    r"|(?:^|_)(?:fwd_ret|forward_return)(?:_\w+)?$)|^y$"
)
_LABEL_NAME_RE = re.compile(
    r"(?:^|_)(?:y|label|labels|target|targets|y_true|fwd_ret|forward_return)(?:$|_)", re.IGNORECASE
)
_EVALUATION_TOKENS = frozenset(
    {"pnl", "profit", "equity", "strategy", "score", "scores", "loss", "error", "errors"}
    | {"mse", "mae", "rmse", "metric", "metrics", "acc", "accuracy", "corr", "ic", "auc"}
)


def _is_non_feature_name(name: str) -> bool:
    """Return True for names of labels and of evaluation results (they may read the target)."""
    return bool(_LABEL_NAME_RE.search(name)) or bool(_name_tokens(name) & _EVALUATION_TOKENS)


class PP011(Rule):
    """A target/label/forward-return column read while building a feature.

    Label functions are exempt: building labels is their job.
    """

    id = "PP011"
    name = "target-as-feature"
    confidence = "medium"
    message = "target column is read while building a feature"
    hint = _confirm(
        "Keep target, label and forward-return columns out of feature code; use only "
        "lagged values (shift(k) with k > 0)."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        for target, value in self._assignments(tree):
            if any(_is_non_feature_name(name) for name in _target_labels(target)):
                continue
            read = self._first_label_read(value, ctx)
            if read is not None:
                column, node = read
                if ctx.in_label_function(ctx.position(node)[0]):
                    continue
                findings.append(
                    ctx.finding(
                        self,
                        node,
                        message=f"column '{column}' is read while building a feature",
                    )
                )
        return findings

    @staticmethod
    def _assignments(tree: ast.AST) -> Iterator[tuple[ast.expr, ast.expr]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    yield from _assignment_pairs(target, node.value)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                yield node.target, node.value
            elif isinstance(node, ast.Call) and _method_name(node) == "assign":
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        yield ast.Name(id=keyword.arg, ctx=ast.Store()), keyword.value

    @staticmethod
    def _first_label_read(value: ast.expr, ctx: LintContext) -> tuple[str, ast.Subscript] | None:
        reads: list[tuple[str, ast.Subscript]] = []
        for node in ast.walk(value):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                for column in _column_names(node.slice):
                    if _LABEL_COLUMN_RE.search(column) and not PP011._is_lagged(node, ctx):
                        reads.append((column, node))
        if not reads:
            return None
        return min(reads, key=lambda item: _node_position(item[1]))

    @staticmethod
    def _is_lagged(node: ast.AST, ctx: LintContext) -> bool:
        """Return True if ``node`` feeds into ``.shift(k)`` with a positive literal ``k``."""
        current = node
        while True:
            parent = ctx.parent(current)
            if isinstance(parent, ast.Attribute) and parent.value is current:
                call = ctx.parent(parent)
                if isinstance(call, ast.Call) and call.func is parent:
                    if parent.attr == "shift" and _is_positive_int_literal(
                        _get_argument(call, 0, "periods", "n")
                    ):
                        return True
                    current = call
                    continue
                current = parent
                continue
            if isinstance(parent, ast.Subscript) and parent.value is current:
                current = parent
                continue
            return False


# --------------------------------------------------------------------------- #
# PP012 current constituents
# --------------------------------------------------------------------------- #

_CONSTITUENT_URL_RE = re.compile(
    r"wikipedia\.org/wiki/(?:List_of_S(?:%26|&)P_500_companies|Nasdaq-100|FTSE_100|DAX)",
    re.IGNORECASE,
)
_CONSTITUENT_NAME_RE = re.compile(
    r"(?:sp500|nasdaq100|ftse100|dax(?:40)?)_?(?:tickers|symbols|constituents)", re.IGNORECASE
)
_POINT_IN_TIME_NAME_RE = re.compile(
    r"as_?of|histor|point_in_time|(?:^|_)pit(?:$|_)|snapshot|archive|membership", re.IGNORECASE
)


class PP012(Rule):
    """A universe built from today's index constituents."""

    id = "PP012"
    name = "current-constituents"
    confidence = "medium"
    message = "universe is built from today's index constituents"
    hint = _confirm(
        "Use historical, point-in-time index membership for each date; a list of current "
        "members only contains the survivors."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        findings: list[Finding] = []
        seen_names: set[str] = set()
        named: list[tuple[tuple[int, int], str, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_call(node, ctx, findings)
            identifier = self._identifier(node)
            if (
                identifier is not None
                and _CONSTITUENT_NAME_RE.search(identifier)
                and not _POINT_IN_TIME_NAME_RE.search(identifier)
            ):
                named.append((_node_position(node), identifier, node))
        for _position, identifier, node in sorted(named, key=lambda item: item[0]):
            if identifier.lower() in seen_names:
                continue
            seen_names.add(identifier.lower())
            findings.append(
                ctx.finding(
                    self,
                    node,
                    message=f"'{identifier}' looks like a list of current index constituents",
                    method=isinstance(node, ast.Attribute),
                )
            )
        return findings

    @staticmethod
    def _identifier(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.arg):
            return node.arg
        return None

    def _check_call(self, call: ast.Call, ctx: LintContext, findings: list[Finding]) -> None:
        qualified = ctx.qualname(call.func)
        if qualified is None:
            return
        if _last_segment(qualified) == "read_html" and ctx.is_from(call.func, "pandas"):
            source = _get_argument(call, 0, "io")
            url = _string_value(source)
            if url is None and isinstance(source, ast.Name):
                for value in ctx.assignments.get(expr_key(source), []):
                    url = url or _string_value(value)
            if url is not None and _CONSTITUENT_URL_RE.search(url):
                findings.append(
                    ctx.finding(
                        self,
                        call,
                        message="read_html on an index constituents page returns today's members",
                    )
                )
        elif qualified == "yfinance.Tickers":
            findings.append(
                ctx.finding(
                    self,
                    call,
                    message="yfinance.Tickers(...) builds a universe from currently listed tickers",
                )
            )


# --------------------------------------------------------------------------- #
# PP013 latest fundamentals
# --------------------------------------------------------------------------- #

_FUNDAMENTAL_ATTRIBUTES = frozenset(
    {
        "financials",
        "balance_sheet",
        "cashflow",
        "income_stmt",
        "info",
        "quarterly_financials",
        "quarterly_balance_sheet",
        "quarterly_cashflow",
        "quarterly_income_stmt",
    }
)


class PP013(Rule):
    """yfinance's latest fundamentals used next to historical prices."""

    id = "PP013"
    name = "latest-fundamentals"
    confidence = "low"
    message = "latest fundamentals snapshot is used next to historical prices"
    hint = _confirm(
        "Use point-in-time fundamentals indexed by filing date; Ticker(...).info and the "
        "financial statements only describe the present."
    )

    def visit(self, tree: ast.AST, ctx: LintContext) -> list[Finding]:
        accesses: list[ast.Attribute] = []
        has_prices = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _FUNDAMENTAL_ATTRIBUTES:
                if self._is_ticker(node.value, ctx):
                    accesses.append(node)
            elif isinstance(node, ast.Call):
                is_history = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "history"
                    and self._is_ticker(node.func.value, ctx)
                )
                if is_history or ctx.qualname(node.func) == "yfinance.download":
                    has_prices = True
        if not has_prices:
            return []
        findings: list[Finding] = []
        reported_lines: set[int] = set()
        for access in sorted(accesses, key=_node_position):
            finding = ctx.finding(
                self,
                access,
                message=f"Ticker(...).{access.attr} is a latest snapshot, used with old prices",
                method=True,
            )
            if finding.line not in reported_lines:
                reported_lines.add(finding.line)
                findings.append(finding)
        return findings

    @staticmethod
    def _is_ticker(node: ast.expr, ctx: LintContext) -> bool:
        """Return True for ``yf.Ticker(...)`` or a name assigned from it."""
        if isinstance(node, ast.Call):
            return ctx.qualname(node.func) == "yfinance.Ticker"
        if isinstance(node, ast.Name):
            return any(
                isinstance(value, ast.Call) and ctx.qualname(value.func) == "yfinance.Ticker"
                for value in ctx.assignments.get(expr_key(node), [])
            )
        return False


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

RULES: dict[str, Rule] = {
    rule.id: rule
    for rule in (
        PP001(),
        PP002(),
        PP003(),
        PP004(),
        PP005(),
        PP006(),
        PP007(),
        PP008(),
        PP009(),
        PP010(),
        PP011(),
        PP012(),
        PP013(),
    )
}
