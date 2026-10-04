"""Triggers PP001 and PP006; the pyproject.toml next to it ignores PP006."""

import pandas as pd


def features(df: pd.DataFrame) -> pd.DataFrame:
    df["next_close"] = df["close"].shift(-1)
    df["z"] = (df["close"] - df["close"].mean()) / df["close"].std()
    return df
