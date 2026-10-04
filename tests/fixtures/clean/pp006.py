"""Clean: statistics only use the rows seen so far, or run across assets on one date."""

import numpy as np
import pandas as pd


def normalise(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    z = (close - close.expanding().mean()) / close.expanding().std()
    trend = close.rolling(20).mean()
    cross_section = returns.rank(axis=1)
    by_date = prices.groupby("date")["close"].rank(pct=True)
    sharpe = close.pct_change().mean() / close.pct_change().std()
    return pd.DataFrame(
        {"z": z, "trend": trend, "rank": cross_section.stack(), "by_date": by_date, "sharpe": sharpe}
    )


def expanding_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.expanding().mean()) / df.expanding().std()


def t_statistic(returns: pd.Series) -> float:
    """The series only appears as a length, so this is one number, not a normalised series."""
    mean, std = returns.mean(), returns.std()
    first = mean / std * np.sqrt(len(returns))
    second = returns.mean() / (returns.std() / np.sqrt(returns.count()))
    return float(first + second)


def fit_on_the_past(returns: pd.Series, split: int) -> pd.Series:
    """The statistics come from the rows before the split, not from the whole series."""
    mu = returns.iloc[:split].mean()
    sigma = returns.iloc[:split].std()
    return (returns - mu) / sigma


def per_date_deciles(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["volume"].transform(lambda s: pd.qcut(s, 10, labels=False))
