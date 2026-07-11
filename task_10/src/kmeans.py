"""
kmeans.py

KMeans clustering implemented from scratch with NumPy (k-means++ style
initialization + Lloyd's algorithm). No scikit-learn is used.
"""

import numpy as np


class KMeansScratch:
    def __init__(self, n_clusters=3, n_iters=100, seed=42, tol=1e-6):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.seed = seed
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.n_iter_run_ = 0

    def _init_centroids(self, X, rng):
        """k-means++ initialization: pick centroids spread out from each
        other, weighted by squared distance to the nearest existing
        centroid, rather than pure random picks."""
        n_samples = X.shape[0]
        centroids = [X[rng.integers(0, n_samples)]]

        for _ in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum() if dist_sq.sum() > 0 else None
            if probs is None:
                next_idx = rng.integers(0, n_samples)
            else:
                next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.centroids = self._init_centroids(X, rng)

        for it in range(self.n_iters):
            distances = np.sqrt(
                ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
            )
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.sqrt(((new_centroids - self.centroids) ** 2).sum())
            self.centroids = new_centroids
            self.n_iter_run_ = it + 1
            if shift < self.tol:
                break

        distances = np.sqrt(
            ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        )
        self.labels_ = np.argmin(distances, axis=1)
        return self

    def predict(self, X):
        distances = np.sqrt(
            ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        )
        return np.argmin(distances, axis=1)
