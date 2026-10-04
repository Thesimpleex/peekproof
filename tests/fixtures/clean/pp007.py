"""Clean: as-of joins look backwards and use the date the data became available."""

import pandas as pd


def attach(trades: pd.DataFrame, quotes: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    with_quotes = pd.merge_asof(trades, quotes, on="ts", direction="backward")
    return pd.merge_asof(
        with_quotes.sort_values("date"),
        fundamentals.sort_values("filing_date"),
        left_on="date",
        right_on="filing_date",
    )
