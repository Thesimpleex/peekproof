"""Leaky: the signal earns the return of the bar that produced it."""

import numpy as np
import pandas as pd


def strategy_pnl(signal: pd.Series, returns: pd.Series) -> pd.Series:
    return signal * returns


def sign_pnl(prediction: pd.Series, returns: pd.Series) -> pd.Series:
    return np.sign(prediction) * returns


def log_price_pnl(position: pd.Series, close: pd.Series) -> pd.Series:
    """The log difference of the close is the return of the bar, although no name says so."""
    return position * np.log(close).diff()
