"""Clean: positions are lagged by one bar before they earn returns."""

import numpy as np
import pandas as pd


def strategy_pnl(signal: pd.Series, returns: pd.Series) -> pd.Series:
    position = signal.shift(1)
    return position * returns


def sign_pnl(prediction: pd.Series, returns: pd.Series) -> pd.Series:
    return np.sign(prediction).shift(1) * returns


def weighted_pnl(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    return (weights.shift(1) * returns).sum(axis=1)


def forward_return(returns: pd.Series) -> pd.Series:
    """Label: the return of the next bar."""
    return returns.shift(-1)


def aligned_pnl(position: pd.Series, returns: pd.Series) -> pd.Series:
    """The position decided on this bar earns the next bar's return."""
    next_returns = forward_return(returns)
    return position * next_returns


def expected_pnl(weights: pd.Series, expected_returns: pd.Series) -> float:
    """A forecast is not a realised return, so there is nothing to lag."""
    return float((weights * expected_returns).sum())
