"""Leaky: smoothing windows that are centered on the current row."""

import numpy as np
import pandas as pd
from pandas.api.indexers import FixedForwardWindowIndexer
from scipy import ndimage
from scipy.signal import filtfilt, savgol_filter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose


def smooth_all(prices: pd.Series) -> pd.DataFrame:
    values = prices.to_numpy()
    kernel = np.ones(5) / 5
    return pd.DataFrame(
        {
            "rolling": prices.rolling(5, center=True).mean(),
            "savgol": savgol_filter(values, 11, 3),
            "convolve": np.convolve(values, kernel, mode="same"),
            "gaussian": ndimage.gaussian_filter1d(values, 2),
        },
        index=prices.index,
    )


def two_sided_filters(prices: pd.Series) -> pd.DataFrame:
    """Each of these estimates a value at t from rows on both sides of t."""
    values = prices.to_numpy()
    cycle, trend = hpfilter(values, lamb=1600)
    forward = prices.rolling(FixedForwardWindowIndexer(window_size=5)).mean()
    return pd.DataFrame(
        {
            "zero_phase": filtfilt([0.2] * 5, [1.0], values),
            "hp_trend": trend,
            "hp_cycle": cycle,
            "seasonal": seasonal_decompose(prices, period=12).trend,
            "slope": np.gradient(values),
            "forward_mean": forward,
        },
        index=prices.index,
    )
