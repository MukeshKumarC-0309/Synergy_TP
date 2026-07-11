"""
metrics.py

All evaluation metrics implemented manually with NumPy.
No sklearn.metrics functions are used anywhere in this file.
"""

import numpy as np


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def regression_report(y_true, y_pred):
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r_squared(y_true, y_pred),
    }


# ---------------------------------------------------------------------
# Classification metrics (binary, labels assumed to be 0/1)
# ---------------------------------------------------------------------

def confusion_matrix(y_true, y_pred):
    """Returns a 2x2 matrix as [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]]), {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    _, d = confusion_matrix(y_true, y_pred)
    denom = d["TP"] + d["FP"]
    return float(d["TP"] / denom) if denom > 0 else 0.0


def recall(y_true, y_pred):
    _, d = confusion_matrix(y_true, y_pred)
    denom = d["TP"] + d["FN"]
    return float(d["TP"] / denom) if denom > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def classification_report(y_true, y_pred):
    cm, counts = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": cm.tolist(),
        "counts": counts,
    }


# ---------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------

def inertia(X, labels, centroids):
    total = 0.0
    for k in range(centroids.shape[0]):
        pts = X[labels == k]
        if len(pts) == 0:
            continue
        total += np.sum((pts - centroids[k]) ** 2)
    return float(total)


def silhouette_score(X, labels, sample_size=1500, seed=42):
    """Manual silhouette score. Full O(n^2) pairwise distance computation
    is too expensive for ~9000 rows, so we evaluate on a random subsample
    (documented in error_analysis.md) which is a standard practical
    compromise for large datasets.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        Xs, ls = X[idx], labels[idx]
    else:
        Xs, ls = X, labels

    unique_labels = np.unique(ls)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise distance matrix on the subsample.
    diff = Xs[:, None, :] - Xs[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    s_vals = np.zeros(len(Xs))
    for i in range(len(Xs)):
        own_cluster = ls[i]
        same_mask = (ls == own_cluster)
        same_mask[i] = False
        if same_mask.sum() > 0:
            a = dist[i, same_mask].mean()
        else:
            a = 0.0

        b = np.inf
        for k in unique_labels:
            if k == own_cluster:
                continue
            other_mask = (ls == k)
            if other_mask.sum() > 0:
                b = min(b, dist[i, other_mask].mean())

        if b == np.inf:
            s_vals[i] = 0.0
        else:
            s_vals[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return float(np.mean(s_vals))


def clustering_report(X, labels, centroids):
    return {
        "inertia": inertia(X, labels, centroids),
        "silhouette_score": silhouette_score(X, labels),
        "cluster_counts": {int(k): int(np.sum(labels == k)) for k in np.unique(labels)},
        "n_clusters": int(centroids.shape[0]),
    }
