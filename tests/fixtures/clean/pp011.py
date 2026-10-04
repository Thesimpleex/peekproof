"""Clean: the target is separated from the features; only its past is used."""

import pandas as pd


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["target"]
    X = df.drop(columns=["target", "fwd_ret"])
    X["prev_outcome"] = df["target"].shift(1)
    return X, y


def merged_columns(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """'_y' is the suffix that merge adds to a clashing column, and 'Y' is a year column."""
    both = left.merge(right, on="date", suffixes=("_x", "_y"))
    both["spread"] = both["close_x"] - both["close_y"]
    both["decade"] = both["Y"] // 10
    return both


def make_label(df: pd.DataFrame) -> pd.DataFrame:
    """A label function may read the target: building the labels is its job."""
    df["sign"] = (df["target"] > 0).astype(int)
    return df
