"""Clean: bars are stamped with their close time."""

import pandas as pd


def five_minute_returns(ticks: pd.DataFrame) -> pd.Series:
    bars = ticks["price"].resample("5min", label="right", closed="right").last()
    changes = bars.pct_change()
    return changes
