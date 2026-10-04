"""Leaky: a feature is built from the target column."""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["momentum"] = df["close"].pct_change(5)
    df["leaky_feature"] = df["fwd_ret"].rolling(3).mean()
    return df


def target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """The name starts with 'target' but is no label function, so the target read is a leak."""
    df["encoded"] = df["next_target"].expanding().mean()
    return df
