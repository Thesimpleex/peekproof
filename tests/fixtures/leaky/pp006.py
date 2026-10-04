"""Leaky: statistics of the whole sample are used on every row."""

import pandas as pd
from sklearn.preprocessing import scale


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df["z"] = (df["close"] - df["close"].mean()) / df["close"].std()
    df["close_rank"] = df["close"].rank(pct=True)
    return df


def global_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def stored_statistics(returns: pd.Series) -> pd.Series:
    """The statistics are computed first and combined with the series on a later line."""
    mu = returns.mean()
    sigma = returns.std()
    return (returns - mu) / sigma


def winsorize(returns: pd.Series) -> pd.Series:
    low = returns.quantile(0.01)
    high = returns.quantile(0.99)
    return returns.clip(low, high)


def whole_series_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["scaled"] = scale(df["volume"])
    df["decile"] = pd.qcut(df["volume"], 10)
    df["capped"] = df["close"].clip(df["close"].quantile(0.01), df["close"].quantile(0.99))
    return df
