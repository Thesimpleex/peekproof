"""Clean: prices from yfinance, fundamentals from a point-in-time table."""

import pandas as pd
import yfinance as yf


def load(symbol: str, fundamentals_path: str) -> pd.DataFrame:
    prices = yf.download(symbol, start="2015-01-01", end="2020-01-01")
    filings = pd.read_csv(fundamentals_path, parse_dates=["filing_date"])
    return pd.merge_asof(
        prices.sort_index(),
        filings.sort_values("filing_date"),
        left_index=True,
        right_on="filing_date",
        direction="backward",
    )
