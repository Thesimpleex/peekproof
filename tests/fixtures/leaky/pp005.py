"""Leaky: evaluation splits that ignore time order."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


def evaluate(X: pd.DataFrame, y: pd.Series) -> list[float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    random_splits = ShuffleSplit(n_splits=3, test_size=0.2)
    stratified = StratifiedKFold(n_splits=5)
    scores = cross_val_score(Ridge(), X, y)
    return [float(score) for score in scores]


def tune(X: pd.DataFrame, y: pd.Series) -> tuple[GridSearchCV, RandomizedSearchCV]:
    """A search without a time-aware cv= validates on folds that train on later rows."""
    grid = {"alpha": [0.1, 1.0, 10.0]}
    exhaustive = GridSearchCV(Ridge(), grid, cv=5).fit(X, y)
    sampled = RandomizedSearchCV(Ridge(), grid, n_iter=3).fit(X, y)
    return exhaustive, sampled
