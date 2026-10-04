"""Clean: evaluation splits respect time order."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)


def evaluate(X: pd.DataFrame, y: pd.Series) -> list[float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    folds = KFold(n_splits=5)
    splitter = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(Ridge(), X, y, cv=splitter)
    return [float(score) for score in scores]


def tune(X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    grid = {"alpha": [0.1, 1.0, 10.0]}
    return GridSearchCV(Ridge(), grid, cv=TimeSeriesSplit(n_splits=5)).fit(X, y)
