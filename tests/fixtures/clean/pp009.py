"""Clean: a loop only reads rows that are not later than the one it computes."""

import numpy as np
import pandas as pd


def prev_bar_move(close: pd.Series) -> np.ndarray:
    values = close.to_numpy()
    out = np.zeros(len(values))
    for i in range(1, len(values)):
        out[i] = values[i] - values[i - 1]
    return out


def prev_bar_move_listcomp(close: pd.Series) -> pd.Series:
    steps = [close.iloc[i] - close.iloc[i - 1] for i in range(1, len(close))]
    return pd.Series(steps, index=close.index[1:])


def prev_bar_move_append(close: pd.Series) -> list[float]:
    moves: list[float] = []
    for i in range(1, len(close)):
        moves.append(close.iloc[i] - close.iloc[i - 1])
    return moves


def trailing_window_mean(close: pd.Series, window: int = 5) -> np.ndarray:
    """The slice ends on row i, so row i is the last row it reads."""
    out = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        out[i] = close.iloc[i - window + 1 : i + 1].mean()
    return out


def make_label(close: pd.Series) -> np.ndarray:
    """A label function looks ahead on purpose, so PP009 stays quiet inside it."""
    out = np.zeros(len(close))
    for i in range(len(close) - 1):
        out[i] = close.iloc[i + 1] / close.iloc[i] - 1
    return out
