"""Clean: gaps are filled from the past only."""

import pandas as pd


def fill_prices(prices: pd.Series) -> pd.Series:
    return prices.ffill()


def fill_volume(volume: pd.Series) -> pd.Series:
    return volume.fillna(0.0)


def fill_curve(curve: pd.Series) -> pd.Series:
    return curve.interpolate(method="pad")


def align_to_calendar(prices: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward fills only carry the last known value forward."""
    return pd.DataFrame(
        {
            "carried": prices.reindex(calendar, method="ffill"),
            "daily": prices.asfreq("D", method="ffill"),
            "change": prices.pct_change(fill_method="ffill"),
        }
    )


def fill_across_maturities(yields: pd.DataFrame) -> pd.DataFrame:
    """Filling across columns (maturities) does not move values between dates."""
    return yields.bfill(axis=1).interpolate(axis="columns")
