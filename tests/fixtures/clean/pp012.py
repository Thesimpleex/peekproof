"""Clean: the universe comes from a point-in-time membership table."""

import pandas as pd


def members_on(membership: pd.DataFrame, asof: str) -> list[str]:
    """Return the index members on one date from a table with start and end dates."""
    rows = membership[(membership["start"] <= asof) & (membership["end"] > asof)]
    return rows["symbol"].tolist()
