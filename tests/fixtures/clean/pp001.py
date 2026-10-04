"""Clean: features look back only; forward looks live in label code."""

import numpy as np
import pandas as pd


def build_features(prices: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    out["prev_close"] = prices.shift(1)
    out["move_3"] = prices.pct_change(3)
    out["step"] = prices.diff(1)
    out["rolled"] = np.roll(prices.to_numpy(), 1)
    return out


def across_maturities(curve: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Shifting across columns (maturities, assets) does not move values in time."""
    neighbour = curve.shift(-1, axis=1)
    rolled = np.roll(weights, -1, axis=1)
    return neighbour - pd.DataFrame(rolled, index=curve.index, columns=curve.columns)


def forward_return(prices: pd.Series) -> pd.Series:
    """Label: return over the next five bars (the name matches the label pattern)."""
    return prices.shift(-5) / prices - 1


def next_direction(prices: pd.Series) -> pd.Series:  # peekproof: label
    """Label: sign of the next bar's move (marked with the label comment)."""
    return np.sign(prices.shift(-1) - prices)


def add_labels(px: pd.DataFrame) -> pd.DataFrame:
    """Columns and variables with a label name hold labels, wherever they are built."""
    px["target"] = px["close"].shift(-1) / px["close"] - 1
    px["fwd_ret_5d"] = px["close"].pct_change(-5)
    y = px["close"].shift(-10) > px["close"]
    return px.assign(label=y.astype(int))
