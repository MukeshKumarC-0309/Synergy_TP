"""
baselines.py

Trivial baseline predictors that every real model must beat.
"""

import numpy as np


class MeanBaseline:
    """Regression baseline: always predicts the training-set mean target."""

    def __init__(self):
        self.value_ = None

    def fit(self, y_train):
        self.value_ = float(np.mean(y_train))
        return self

    def predict(self, n):
        return np.full(n, self.value_)


class MajorityClassBaseline:
    """Classification baseline: always predicts the most frequent training class."""

    def __init__(self):
        self.majority_class_ = None

    def fit(self, y_train):
        values, counts = np.unique(y_train, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])
        return self

    def predict(self, n):
        return np.full(n, self.majority_class_)
