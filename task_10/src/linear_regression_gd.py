"""
linear_regression_gd.py

Linear regression trained with batch gradient descent, implemented from
scratch with NumPy only.
"""

import numpy as np


class LinearRegressionGD:
    def __init__(self, lr=0.1, n_iters=2000, seed=42):
        self.lr = lr
        self.n_iters = n_iters
        self.seed = seed
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _predict_raw(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []
        self.val_loss_history = []

        for it in range(self.n_iters):
            y_pred = self._predict_raw(X)
            error = y_pred - y

            grad_w = (2 / n_samples) * (X.T @ error)
            grad_b = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            train_loss = float(np.mean(error ** 2))
            self.loss_history.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self._predict_raw(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                self.val_loss_history.append(val_loss)

        return self

    def predict(self, X):
        return self._predict_raw(X)
