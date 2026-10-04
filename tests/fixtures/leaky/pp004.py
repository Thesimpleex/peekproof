"""Leaky: the scaler sees the whole sample before the data is split."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare(features: pd.DataFrame, target: pd.Series):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return train_test_split(scaled, target, test_size=0.2, shuffle=False)


def prepare_by_cut(features: pd.DataFrame, split: int):
    scaler = StandardScaler()
    scaler.fit(features)
    return features.iloc[:split], features.iloc[split:]
