"""Clean: smoothing windows only use the current and earlier rows."""

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def smooth_all(prices: pd.Series) -> pd.DataFrame:
    values = prices.to_numpy()
    kernel = np.ones(5) / 5
    return pd.DataFrame(
        {
            "rolling": prices.rolling(5).mean(),
            "ewm": prices.ewm(span=5).mean(),
            "lfilter": lfilter(kernel, 1.0, values),
        },
        index=prices.index,
    )


def trailing_average(values: np.ndarray) -> np.ndarray:
    kernel = np.ones(5) / 5
    return np.convolve(values, kernel, mode="valid")


def trailing_weighted_sum(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Mode 'full' cut to the input length keeps the trailing weighted sum at every row."""
    return np.convolve(values, kernel)[: len(values)]


def pairwise_mean(prices: pd.Series) -> pd.Series:
    """A centered window of two rows ends on the current row: pandas looks ahead from three."""
    return prices.rolling(2, center=True).mean()
