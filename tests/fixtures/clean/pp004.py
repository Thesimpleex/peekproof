"""Clean: the data is split first and the scaler is fitted on the training part."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare(features: pd.DataFrame, target: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def prepare_by_cut(features: pd.DataFrame, split: int):
    scaler = StandardScaler()
    scaler.fit(features.iloc[:split])
    return scaler.transform(features.iloc[:split]), scaler.transform(features.iloc[split:])
