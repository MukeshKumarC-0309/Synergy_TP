"""
logistic_regression_gd.py

Binary logistic regression trained with batch gradient descent on the
binary cross-entropy loss, implemented from scratch with NumPy only.
"""

import numpy as np


def _sigmoid(z):
    # Clip to avoid overflow in exp for very negative/positive z.
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iters=2000, threshold=0.5, seed=42):
        self.lr = lr
        self.n_iters = n_iters
        self.threshold = threshold
        self.seed = seed
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss_history = []

    def _prob(self, X):
        z = X @ self.weights + self.bias
        return _sigmoid(z)

    @staticmethod
    def _bce(y_true, p, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []
        self.val_loss_history = []

        for it in range(self.n_iters):
            p = self._prob(X)
            error = p - y

            grad_w = (1 / n_samples) * (X.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            self.loss_history.append(self._bce(y, p))

            if X_val is not None and y_val is not None:
                p_val = self._prob(X_val)
                self.val_loss_history.append(self._bce(y_val, p_val))

        return self

    def predict_proba(self, X):
        return self._prob(X)

    def predict(self, X):
        return (self._prob(X) >= self.threshold).astype(int)
