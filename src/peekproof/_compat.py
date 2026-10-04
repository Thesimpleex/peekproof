"""Input adapters and comparison helpers for the oracle.

The oracle works on plain pandas objects. This module turns the supported
inputs (pandas ``DataFrame`` / ``Series`` and polars ``DataFrame``) into pandas,
validates the time index, hands pandas slices back to a polars function as
polars frames, and holds the small helpers that compare two aligned columns or
frames and turn index positions into readable labels.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from peekproof.errors import PeekproofError

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "ColumnDiff",
    "PolarsAdapter",
    "column_atol",
    "column_scale",
    "compare_columns",
    "compare_frames",
    "frames_equal",
    "is_numeric",
    "label_at",
    "labels_at",
    "polars_adapter",
    "robust_scale",
    "to_pandas",
    "validate_index",
]

BoolArray = npt.NDArray[np.bool_]

_TINY = float(np.finfo(np.float64).tiny)
_NORMAL_MAD = 1.4826  # factor that turns a median absolute deviation into a standard deviation
_NORMAL_IQR = 1.349  # the interquartile range of a standard normal distribution


# --------------------------------------------------------------------------- #
# Input adapters
# --------------------------------------------------------------------------- #


def to_pandas(data: object, time_col: str | None = None) -> pd.DataFrame | pd.Series:
    """Return ``data`` as a pandas object with a strictly increasing index.

    Accepted inputs:

    - A pandas ``DataFrame`` or ``Series``. If ``time_col`` names a column of a
      ``DataFrame``, that column becomes the index and the rows are sorted by
      it. Otherwise the existing index is used as it is and must already be
      sorted. ``time_col`` is ignored for a ``Series``.
    - A polars ``DataFrame``. ``time_col`` is required. The frame is converted
      to pandas, indexed by that column and sorted by it. polars is imported
      only when a polars object is passed in.

    The returned object may be the caller's own object; it is never modified.

    Raises:
        PeekproofError: for an unsupported type, a missing or unknown
            ``time_col``, or an index that is not monotonic increasing and
            unique.
    """
    result: pd.DataFrame | pd.Series
    if isinstance(data, pd.DataFrame):
        result = data if time_col is None else _index_by_column(data, time_col)
    elif isinstance(data, pd.Series):
        result = data
    else:
        frame = _as_polars_frame(data)
        if frame is None:
            raise PeekproofError(
                f"unsupported data type {type(data).__name__}; pass a pandas DataFrame or "
                "Series, or a polars DataFrame together with time_col"
            )
        result = _from_polars(frame, time_col)
    validate_index(result.index)
    return result


def validate_index(index: pd.Index) -> None:
    """Require a single-level, monotonic increasing index without duplicate labels.

    Rows are matched between evaluations of the function by label, so labels
    must be unique. Missing labels (``NaT``/``NaN``) make an index
    non-monotonic and are rejected by the first check.

    Raises:
        PeekproofError: if the index is not strictly increasing.
    """
    if isinstance(index, pd.MultiIndex):
        raise PeekproofError(
            "long-format panels with a MultiIndex are not supported: cutoffs would split the "
            "rows of one timestamp; pass one wide frame per feature (timestamps as the index, "
            "assets as columns)"
        )
    if not index.is_monotonic_increasing:
        raise PeekproofError(
            "the index must be monotonic increasing and free of missing labels; sort the "
            "data by time first (for example with sort_index())"
        )
    if not index.is_unique:
        raise PeekproofError(
            "the index contains duplicate labels; rows are matched between evaluations by "
            "label, so every label must be unique"
        )


def _as_polars_frame(data: object) -> pl.DataFrame | None:
    """Return ``data`` if it is a polars ``DataFrame``, else ``None``.

    polars is an optional dependency. The module name of the object's type is
    checked first, so polars is only imported for objects that come from it.
    """
    if type(data).__module__.partition(".")[0] != "polars":
        return None
    import polars as pl

    return data if isinstance(data, pl.DataFrame) else None


def _from_polars(frame: pl.DataFrame, time_col: str | None) -> pd.DataFrame:
    if time_col is None:
        raise PeekproofError("polars input needs time_col=<name of the time column>")
    if time_col not in frame.columns:
        raise PeekproofError(f"time_col {time_col!r} is not a column of the polars frame")
    try:
        converted = frame.to_pandas()
    except ImportError as exc:
        raise PeekproofError(
            "converting a polars DataFrame to pandas needs pyarrow; install it or pass a "
            "pandas DataFrame"
        ) from exc
    return _index_by_column(converted, time_col)


def _index_by_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Use ``column`` as a sorted index (stable sort, so ties keep their order)."""
    if column not in frame.columns:
        if frame.index.name == column:
            return frame
        raise PeekproofError(f"time_col {column!r} is neither a column nor the name of the index")
    indexed = frame.set_index(column)
    if indexed.index.hasnans:
        raise PeekproofError(f"time_col {column!r} contains missing values")
    return indexed.sort_index(kind="stable")


def polars_adapter(data: object, time_col: str | None) -> PolarsAdapter | None:
    """Return an adapter for a polars ``DataFrame``, and ``None`` for anything else."""
    frame = _as_polars_frame(data)
    if frame is None or time_col is None:
        return None
    return PolarsAdapter(frame, time_col)


class PolarsAdapter:
    """Gives a function polars frames although the oracle works on pandas slices.

    A polars function should be tested as it will be used, so it receives polars
    ``DataFrame`` objects: the pandas slice is turned back into polars with the
    time column restored from the index, in the column order and with the
    dtypes of the frame that the caller passed in (a ``Date`` column that went
    through pandas as a timestamp is cast back, for example). What the function
    returns is turned into a pandas frame indexed by the time column.
    """

    def __init__(self, frame: pl.DataFrame, time_col: str) -> None:
        self._columns = list(frame.columns)
        self._schema = dict(frame.schema)
        self._time_col = time_col

    def to_native(self, frame: pd.DataFrame) -> pl.DataFrame:
        """Return the pandas frame (indexed by the time column) as a polars frame."""
        import polars as pl

        try:
            native = pl.from_pandas(frame.reset_index()).select(self._columns)
        except Exception as exc:  # pyarrow and polars raise many different types
            raise PeekproofError(
                "could not convert a pandas slice of the data back to polars: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for name, dtype in self._schema.items():
            if native.schema[name] != dtype:
                with contextlib.suppress(pl.exceptions.PolarsError):
                    native = native.with_columns(pl.col(name).cast(dtype))
        return native

    def to_pandas(self, raw: object, where: str) -> object:
        """Return a polars result as a pandas frame indexed by the time column.

        Anything that is not a polars object is returned unchanged. ``where``
        names the evaluation for the error messages.

        Raises:
            PeekproofError: if the result has no time column, so that its rows
                cannot be matched to the input rows.
        """
        import polars as pl

        if isinstance(raw, pl.LazyFrame):
            raw = raw.collect()
        if isinstance(raw, pl.Series):
            raise PeekproofError(
                f"the output of {where} is a polars Series, which has no time column; the "
                f"output must keep the time column {self._time_col!r}, so return a DataFrame"
            )
        if not isinstance(raw, pl.DataFrame):
            return raw
        if self._time_col not in raw.columns:
            raise PeekproofError(
                f"the output of {where} has no column {self._time_col!r}; the output must keep "
                f"the time column {self._time_col!r} so that its rows can be matched to the "
                "input rows"
            )
        try:
            converted = raw.to_pandas()
        except ImportError as exc:
            raise PeekproofError(
                "converting a polars DataFrame to pandas needs pyarrow; install it or pass a "
                "pandas DataFrame"
            ) from exc
        indexed = converted.set_index(self._time_col)
        if indexed.index.hasnans:
            raise PeekproofError(
                f"the time column {self._time_col!r} of the output of {where} has missing values"
            )
        return indexed


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ColumnDiff:
    """Result of comparing two aligned columns.

    ``differs`` has one entry per row and is True where the columns differ.
    ``max_abs_diff`` is the largest finite absolute difference among the
    differing rows of a numeric comparison. It is ``None`` for non-numeric
    columns and when no differing row has a finite difference (for example a
    value compared with a missing value).
    """

    differs: BoolArray
    max_abs_diff: float | None


def is_numeric(values: pd.Series) -> bool:
    """True for real-valued numeric columns.

    Booleans and complex numbers are excluded: they are compared for equality,
    not with a tolerance.
    """
    dtype = values.dtype
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def column_scale(values: pd.Series) -> float:
    """Typical size of a column: the median absolute value of its finite entries.

    Missing values and infinities are ignored. The result is 1.0 if the column
    is not numeric, has no finite entry, or has a median of zero (a column that
    is mostly zeros has no scale of its own).
    """
    if not is_numeric(values):
        return 1.0
    numbers = values.to_numpy(dtype="float64", na_value=np.nan)
    finite = np.abs(numbers[np.isfinite(numbers)])
    if finite.size == 0:
        return 1.0
    scale = float(np.median(finite))
    return scale if scale > 0.0 else 1.0


def column_atol(atol: float, values: pd.Series) -> float:
    """Absolute tolerance for one column: ``atol`` times the scale of the column.

    A fixed absolute tolerance hides every difference in a column whose values
    are smaller than the tolerance (squared returns are around 1e-8). Scaling
    it by :func:`column_scale` keeps the comparison meaningful at any magnitude.
    """
    return atol * max(column_scale(values), _TINY)


def robust_scale(values: pd.Series) -> float:
    """A positive measure of the spread of a column that outliers do not inflate.

    The first of these that is finite and above zero: the median absolute
    deviation, the interquartile range and the standard deviation (each scaled
    to equal the standard deviation of a normal distribution), and 1.0. Missing
    values and infinities are ignored. The result is therefore never zero, also
    for a constant column or a column that is mostly one value.
    """
    numbers = values.to_numpy(dtype="float64", na_value=np.nan)
    finite = numbers[np.isfinite(numbers)]
    if finite.size == 0:
        return 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        centre = float(np.median(finite))
        q25, q75 = np.percentile(finite, [25.0, 75.0])
        candidates = (
            _NORMAL_MAD * float(np.median(np.abs(finite - centre))),
            float(q75 - q25) / _NORMAL_IQR,
            float(finite.std()),
        )
    for candidate in candidates:
        if math.isfinite(candidate) and candidate > 0.0:
            return candidate
    return 1.0


def compare_columns(a: pd.Series, b: pd.Series, *, atol: float, rtol: float) -> ColumnDiff:
    """Compare two equally long columns row by row, by position.

    Numeric columns use ``np.isclose(a, b, rtol, atol, equal_nan=True)``. All
    other columns use equality, and two missing values count as equal.
    """
    if len(a) != len(b):
        raise ValueError("columns must have the same length")
    if is_numeric(a) and is_numeric(b):
        left = a.to_numpy(dtype="float64", na_value=np.nan)
        right = b.to_numpy(dtype="float64", na_value=np.nan)
        with np.errstate(invalid="ignore", over="ignore"):
            differs = ~np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
            gaps = np.abs(left - right)[differs]
        finite = gaps[np.isfinite(gaps)]
        return ColumnDiff(differs, float(finite.max()) if finite.size else None)
    return ColumnDiff(_exact_mismatch(a, b), None)


def compare_frames(
    a: pd.DataFrame, b: pd.DataFrame, *, atol: float, rtol: float, scaled: bool = False
) -> list[ColumnDiff]:
    """Compare two frames of the same shape column by column, by position.

    With ``scaled=True`` the absolute tolerance of each column is
    ``column_atol(atol, column of a)`` instead of ``atol``.
    """
    if a.shape != b.shape:
        raise ValueError("frames must have the same shape")
    diffs: list[ColumnDiff] = []
    for j in range(a.shape[1]):
        left = a.iloc[:, j]
        column_tol = column_atol(atol, left) if scaled else atol
        diffs.append(compare_columns(left, b.iloc[:, j], atol=column_tol, rtol=rtol))
    return diffs


def frames_equal(
    a: pd.DataFrame, b: pd.DataFrame, *, atol: float, rtol: float, scaled: bool = False
) -> bool:
    """True if both frames have the same index, the same columns and equal values.

    ``scaled`` is passed on to :func:`compare_frames`.
    """
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        return False
    diffs = compare_frames(a, b, atol=atol, rtol=rtol, scaled=scaled)
    return not any(d.differs.any() for d in diffs)


def format_label(label: object) -> str:
    """Readable form of an index label.

    Timestamps at midnight are shown as dates; everything else uses ``str()``.
    """
    if isinstance(label, pd.Timestamp) and label == label.normalize():
        return label.strftime("%Y-%m-%d")
    return str(label)


def label_at(index: pd.Index, position: int) -> str:
    """Readable label of the index entry at integer ``position``."""
    return format_label(index[position])


def labels_at(index: pd.Index, positions: Sequence[int] | npt.NDArray[np.integer]) -> list[str]:
    """Readable labels of the index entries at the given integer positions."""
    return [format_label(index[int(i)]) for i in positions]


def _exact_mismatch(a: pd.Series, b: pd.Series) -> BoolArray:
    """Rows where two non-numeric columns differ. Missing equals missing."""
    a_missing = a.isna().to_numpy(dtype=bool)
    b_missing = b.isna().to_numpy(dtype=bool)
    mismatch: BoolArray = a_missing != b_missing
    present = ~(a_missing | b_missing)
    if present.any():
        left = a.to_numpy(dtype=object)[present]
        right = b.to_numpy(dtype=object)[present]
        mismatch[present] = ~_elementwise_equal(left, right)
    return mismatch


def _elementwise_equal(left: npt.NDArray[np.object_], right: npt.NDArray[np.object_]) -> BoolArray:
    try:
        equal = np.asarray(left == right, dtype=bool)
    except (TypeError, ValueError):
        equal = np.zeros(0, dtype=bool)
    if equal.shape == left.shape:
        return equal
    return np.fromiter(
        (_safe_equal(x, y) for x, y in zip(left, right, strict=True)),
        dtype=bool,
        count=len(left),
    )


def _safe_equal(x: object, y: object) -> bool:
    try:
        return bool(x == y)
    except (TypeError, ValueError):
        pass
    try:
        return bool(np.array_equal(np.asarray(x), np.asarray(y)))
    except (TypeError, ValueError):
        return False
