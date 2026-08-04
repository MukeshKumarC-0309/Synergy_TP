"""
data_utils.py

Loading and splitting utilities for the sensor -> Temperature regression task.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Temperature"
FEATURES = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5"]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def inspect_data(df):
    """Returns a small dict of dataset-understanding facts used in the report."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "target_mean": float(df[TARGET].mean()),
        "target_std": float(df[TARGET].std()),
        "feature_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES},
        "correlations_with_target": df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).to_dict(),
    }


def split_data(df, seed=42):
    """70% train / 15% validation / 15% test, split with a fixed seed."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
