"""Leaky: features that read values from later rows."""

import numpy as np
import pandas as pd

HORIZON = 5


def build_features(prices: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    out["next_close"] = prices.shift(-1)
    out["three_ahead"] = prices.shift(periods=-3)
    out["move_2"] = prices.pct_change(-2)
    out["step"] = prices.diff(-1)
    out["rolled"] = np.roll(prices.to_numpy(), -1)
    return out


def horizon_feature(prices: pd.Series) -> pd.Series:
    """The sign of the shift is only known at run time (medium confidence)."""
    return prices.shift(-HORIZON)


def next_open_fill(bars: pd.DataFrame) -> pd.DataFrame:
    """A next-bar fill price is only safe as a fill price (medium confidence)."""
    bars["fill_price"] = bars["open"].shift(-1)
    entry = bars["open"].shift(-1)
    return bars.assign(entry_signal=entry > bars["close"])


def target_position(signal: pd.Series) -> pd.Series:
    """The name starts with a label word, but this builds a position, not a label."""
    return signal.shift(-1)


def forward_fill_features(prices: pd.Series) -> pd.DataFrame:
    """The name contains 'forward' and still builds features from later rows."""
    return pd.DataFrame({"next_close": prices.shift(-1)})
