"""Leaky: today's fundamentals attached to historical prices."""

import pandas as pd
import yfinance as yf


def load(symbol: str) -> pd.DataFrame:
    prices = yf.download(symbol, start="2015-01-01", end="2020-01-01")
    info = yf.Ticker(symbol).info
    prices["trailing_pe"] = info["trailingPE"]
    return prices
