"""Leaky: the universe is today's S&P 500 membership."""

import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_universe() -> list[str]:
    tables = pd.read_html(URL)
    sp500_tickers = tables[0]["Symbol"].tolist()
    return sp500_tickers
