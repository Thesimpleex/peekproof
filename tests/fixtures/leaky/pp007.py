"""Leaky: joins that attach information from after the join date."""

import pandas as pd


def attach(trades: pd.DataFrame, quotes: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    with_quotes = pd.merge_asof(trades, quotes, on="ts", direction="forward")
    return pd.merge_asof(
        with_quotes.sort_values("date"),
        fundamentals.sort_values("period_end"),
        left_on="date",
        right_on="period_end",
    )
