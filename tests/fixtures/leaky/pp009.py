"""Leaky: a loop reads the next row while computing the current one."""

import numpy as np
import pandas as pd


def next_bar_move(close: pd.Series) -> np.ndarray:
    values = close.to_numpy()
    out = np.zeros(len(values))
    for i in range(len(values) - 1):
        out[i] = values[i + 1] - values[i]
    return out


def next_bar_move_listcomp(close: pd.Series) -> pd.Series:
    steps = [close.iloc[i + 1] - close.iloc[i] for i in range(len(close) - 1)]
    return pd.Series(steps, index=close.index[:-1])


def next_bar_move_append(close: pd.Series) -> list[float]:
    """The list gets one element per row, and element i is built from row i + 1."""
    moves: list[float] = []
    for i in range(len(close) - 1):
        moves.append(close.iloc[i + 1] - close.iloc[i])
    return moves


def forward_window_mean(close: pd.Series) -> np.ndarray:
    """The slice starts at row i and runs on into later rows (medium confidence)."""
    out = np.zeros(len(close))
    for i in range(len(close) - 5):
        out[i] = close.iloc[i : i + 5].mean()
    return out
