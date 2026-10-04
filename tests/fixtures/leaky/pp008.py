"""Leaky: bars are labeled with their start time and then used as features."""

import pandas as pd


def five_minute_returns(ticks: pd.DataFrame) -> pd.Series:
    bars = ticks["price"].resample("5min").last()
    changes = bars.pct_change()
    return changes
