"""Leaky: gaps are filled with values that only arrive later."""

import pandas as pd
import polars as pl


def fill_prices(prices: pd.Series) -> pd.Series:
    return prices.bfill()


def fill_volume(volume: pd.Series) -> pd.Series:
    return volume.fillna(method="bfill")


def fill_curve(curve: pd.Series) -> pd.Series:
    return curve.interpolate(limit_direction="both")


def align_to_calendar(prices: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Each of these takes a value from a later row for an earlier date."""
    return pd.DataFrame(
        {
            "backfilled": prices.reindex(calendar, method="bfill"),
            "nearest": prices.reindex(calendar, method="nearest"),
            "daily": prices.asfreq("D", method="bfill"),
            "change": prices.pct_change(fill_method="bfill"),
        }
    )


def smooth_gaps(prices: pd.Series, frame: pl.DataFrame) -> tuple[pd.Series, pl.DataFrame]:
    """Linear interpolation bridges a gap with the next observation (medium confidence)."""
    linear = prices.interpolate()
    timed = prices.interpolate(method="time")
    return linear + timed, frame.select(pl.col("price").interpolate())
