"""
data_utils.py

Loading, cleaning, and splitting utilities for the AirQualityUCI dataset.
No scikit-learn is used anywhere in this file.
"""

import numpy as np
import pandas as pd

MISSING_CODE = -200.0


def load_raw(csv_path):
    """Load the raw AirQualityUCI.csv file.

    The UCI file is semicolon-separated, uses a comma as the decimal
    separator, and has two trailing empty columns caused by trailing
    semicolons in every row. Missing sensor/GT readings are encoded as
    -200 rather than NaN.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    # Drop the two trailing unnamed empty columns produced by the
    # trailing ';;' at the end of every data row.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    # Drop fully empty rows (the raw file has a few blank trailing rows).
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def replace_missing_with_nan(df, columns):
    """Replace the dataset's -200 sentinel with np.nan for given columns."""
    df = df.copy()
    for c in columns:
        df[c] = df[c].replace(MISSING_CODE, np.nan)
    return df


def build_timestamp(df):
    """Combine Date + Time into a sortable datetime column (for reference /
    plotting only -- not used as a model feature, to avoid leaking a
    monotonic index into the models)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
        format="%d/%m/%Y %H:%M:%S",
    )
    return df


def train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, seed=42):
    """Shuffle-split arrays into train/val/test. Implemented manually
    (no sklearn.model_selection.train_test_split).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
        train_idx, val_idx, test_idx,
    )


class StandardScaler:
    """Minimal from-scratch standardization (zero mean, unit variance).

    Fit on training data only, then applied to val/test to avoid
    preprocessing leakage.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
