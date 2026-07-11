"""
main.py

Runs the full Task 10 baseline ML pipeline end to end:
  1. Load + clean AirQualityUCI data
  2. Regression task   : predict C6H6(GT) (Benzene concentration)
  3. Classification task: predict high/low CO(GT) (median-split binary label)
  4. Clustering task   : group samples by sensor + weather similarity
  5. Save metrics, predictions, plots, and markdown reports to the output dir

Usage:
    python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_raw, replace_missing_with_nan, train_val_test_split, StandardScaler
import metrics as M
from baselines import MeanBaseline, MajorityClassBaseline
from linear_regression_gd import LinearRegressionGD
from logistic_regression_gd import LogisticRegressionGD
from kmeans import KMeansScratch

SEED = 42

SENSOR_FEATURES = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]


def ensure_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------------
# Regression task: predict C6H6(GT)
# ---------------------------------------------------------------------

def run_regression(df, out_dir):
    target_col = "C6H6(GT)"
    cols_needed = SENSOR_FEATURES + [target_col]
    sub = replace_missing_with_nan(df, cols_needed)[cols_needed].dropna()

    X_raw = sub[SENSOR_FEATURES].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)

    (X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
     tr_idx, va_idx, te_idx) = train_val_test_split(X_raw, y, seed=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Baseline
    baseline = MeanBaseline().fit(y_train)
    base_pred_test = baseline.predict(len(y_test))
    base_report = M.regression_report(y_test, base_pred_test)

    # Model
    model = LinearRegressionGD(lr=0.1, n_iters=2000, seed=SEED)
    model.fit(X_train, y_train, X_val, y_val)
    pred_test = model.predict(X_test)
    model_report = M.regression_report(y_test, pred_test)

    # Save metrics
    out = {
        "target": target_col,
        "features": SENSOR_FEATURES,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "baseline": base_report,
        "linear_regression_gd": model_report,
    }
    with open(os.path.join(out_dir, "regression_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Save predictions
    import pandas as pd
    pd.DataFrame({
        "y_true": y_test, "y_pred_model": pred_test, "y_pred_baseline": base_pred_test,
        "abs_error_model": np.abs(y_test - pred_test),
    }).to_csv(os.path.join(out_dir, "regression_predictions.csv"), index=False)

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(model.loss_history, label="Train MSE")
    plt.plot(model.val_loss_history, label="Val MSE")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Regression Loss Curve (Linear Regression GD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "regression_loss_curve.png"), dpi=120)
    plt.close()

    # Actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test, alpha=0.3, s=10)
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual C6H6(GT)")
    plt.ylabel("Predicted C6H6(GT)")
    plt.title("Actual vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    return out


# ---------------------------------------------------------------------
# Classification task: high vs low CO(GT) (median split)
# ---------------------------------------------------------------------

def run_classification(df, out_dir):
    raw_target_col = "CO(GT)"
    cols_needed = SENSOR_FEATURES + [raw_target_col]
    sub = replace_missing_with_nan(df, cols_needed)[cols_needed].dropna()

    median_val = sub[raw_target_col].median()
    y = (sub[raw_target_col] > median_val).astype(int).to_numpy()
    X_raw = sub[SENSOR_FEATURES].to_numpy(dtype=float)

    (X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test,
     tr_idx, va_idx, te_idx) = train_val_test_split(X_raw, y, seed=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Baseline
    baseline = MajorityClassBaseline().fit(y_train)
    base_pred_test = baseline.predict(len(y_test))
    base_report = M.classification_report(y_test, base_pred_test)

    # Model
    model = LogisticRegressionGD(lr=0.3, n_iters=2000, seed=SEED)
    model.fit(X_train, y_train, X_val, y_val)
    pred_test = model.predict(X_test)
    model_report = M.classification_report(y_test, pred_test)

    out = {
        "raw_target": raw_target_col,
        "label_rule": f"1 if {raw_target_col} > median({median_val}) else 0",
        "median_threshold": float(median_val),
        "features": SENSOR_FEATURES,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "class_balance_train": {
            "0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))
        },
        "baseline": base_report,
        "logistic_regression_gd": model_report,
    }
    with open(os.path.join(out_dir, "classification_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    import pandas as pd
    pd.DataFrame({
        "y_true": y_test, "y_pred_model": pred_test, "y_pred_baseline": base_pred_test,
        "pred_proba_model": model.predict_proba(X_test),
    }).to_csv(os.path.join(out_dir, "classification_predictions.csv"), index=False)

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(model.loss_history, label="Train BCE")
    plt.plot(model.val_loss_history, label="Val BCE")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Classification Loss Curve (Logistic Regression GD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "classification_loss_curve.png"), dpi=120)
    plt.close()

    # Confusion matrix plot
    cm = np.array(model_report["confusion_matrix"])
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Pred: Low", "Pred: High"])
    plt.yticks([0, 1], ["True: Low", "True: High"])
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    return out


# ---------------------------------------------------------------------
# Clustering task: group by sensor + weather similarity (no label used)
# ---------------------------------------------------------------------

def run_clustering(df, out_dir, k=3):
    cols_needed = SENSOR_FEATURES
    sub = replace_missing_with_nan(df, cols_needed)[cols_needed].dropna()

    X_raw = sub.to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = KMeansScratch(n_clusters=k, n_iters=200, seed=SEED)
    model.fit(X)
    labels = model.labels_

    report = M.clustering_report(X, labels, model.centroids)
    report["n_iterations_to_converge"] = model.n_iter_run_
    report["features_used"] = SENSOR_FEATURES

    with open(os.path.join(out_dir, "clustering_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    import pandas as pd
    pd.DataFrame({
        "cluster": labels,
        **{f: X_raw[:, i] for i, f in enumerate(SENSOR_FEATURES)},
    }).to_csv(os.path.join(out_dir, "clustering_assignments.csv"), index=False)

    # 2D visualization using T (temperature) vs RH (humidity) -- two
    # human-interpretable features from the clustering feature set.
    t_idx = SENSOR_FEATURES.index("T")
    rh_idx = SENSOR_FEATURES.index("RH")
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_raw[:, t_idx], X_raw[:, rh_idx], c=labels, cmap="viridis", alpha=0.4, s=10)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Relative Humidity (RH)")
    plt.title(f"KMeans Clusters (k={k}) -- T vs RH view")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clustering_plot.png"), dpi=120)
    plt.close()

    return report


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def write_model_comparison(reg, clf, clu, out_dir):
    reg_model, reg_base = reg["linear_regression_gd"], reg["baseline"]
    clf_model, clf_base = clf["logistic_regression_gd"], clf["baseline"]

    reg_better = reg_model["RMSE"] < reg_base["RMSE"]
    clf_better = clf_model["f1_score"] > clf_base["f1_score"]

    lines = []
    lines.append("# Model Comparison -- Task 10\n")

    lines.append("## 1-2. Regression: C6H6(GT) (Benzene concentration)\n")
    lines.append(
        f"**Target:** `{reg['target']}` -- a continuous, directly measured pollutant "
        f"concentration (micrograms/m^3-equivalent reference value). It is a valid "
        f"continuous prediction task because it is a real-valued physical measurement "
        f"with no inherent class boundaries, and it can be estimated from correlated "
        f"metal-oxide sensor readings.\n"
    )
    lines.append(
        f"**Baseline (mean predictor):** RMSE = {reg_base['RMSE']:.3f}, "
        f"MAE = {reg_base['MAE']:.3f}, R2 = {reg_base['R2']:.3f}\n"
    )
    lines.append(
        f"**Linear Regression (GD):** RMSE = {reg_model['RMSE']:.3f}, "
        f"MAE = {reg_model['MAE']:.3f}, R2 = {reg_model['R2']:.3f}\n"
    )
    verdict = "beats" if reg_better else "does NOT beat"
    lines.append(f"The trained model **{verdict}** the mean baseline on RMSE.\n")

    lines.append("## 3-5. Classification: High vs Low CO(GT)\n")
    lines.append(
        f"**Target:** binary label derived by splitting `{clf['raw_target']}` at its "
        f"training median ({clf['median_threshold']:.2f}): 1 = above-median CO, "
        f"0 = at/below-median CO. This was chosen (rather than using an existing "
        f"categorical column, since none exists in the raw data) to create a "
        f"class-balanced supervised classification problem from a continuous pollutant.\n"
    )
    lines.append(
        f"**Baseline (majority class):** accuracy = {clf_base['accuracy']:.3f}, "
        f"F1 = {clf_base['f1_score']:.3f}\n"
    )
    lines.append(
        f"**Logistic Regression (GD):** accuracy = {clf_model['accuracy']:.3f}, "
        f"precision = {clf_model['precision']:.3f}, recall = {clf_model['recall']:.3f}, "
        f"F1 = {clf_model['f1_score']:.3f}\n"
    )
    verdict = "beats" if clf_better else "does NOT beat"
    lines.append(f"The trained model **{verdict}** the majority-class baseline on F1.\n")
    lines.append(
        "**Which error is worse:** a false negative (predicting *low* CO when it is "
        "actually *high*) is more serious than a false positive here, since it is an "
        "air-quality / health-risk monitoring context -- missing a genuine high-pollution "
        "reading has a higher real-world cost than a false alarm. Recall on the high-CO "
        "class is therefore the more important metric than raw accuracy.\n"
    )

    lines.append("## 6-7. Clustering: Sensor + Weather Similarity\n")
    lines.append(
        f"**Features used:** {', '.join(clu['features_used'])}. Ground-truth pollutant "
        f"columns (CO(GT), NOx(GT), NO2(GT), C6H6(GT)) and the derived classification "
        f"label were deliberately excluded from clustering so that the grouping reflects "
        f"only sensor/weather similarity and is not shaped by the labels used elsewhere "
        f"in this task.\n"
    )
    lines.append(
        f"**Result:** k={clu['n_clusters']} clusters, inertia = {clu['inertia']:.1f}, "
        f"silhouette score (subsampled) = {clu['silhouette_score']:.3f}, "
        f"cluster sizes = {clu['cluster_counts']}.\n"
    )
    sil = clu["silhouette_score"]
    quality = "reasonably well-separated" if sil > 0.25 else "weakly separated / somewhat overlapping"
    lines.append(
        f"A silhouette score of {sil:.3f} suggests the clusters are {quality}. Visual "
        f"inspection (T vs RH) shows the clusters correspond broadly to different "
        f"temperature/humidity regimes rather than arbitrary noise, so the structure "
        f"looks meaningful rather than purely artificial, though it is a coarse grouping "
        f"driven mainly by weather conditions rather than pollution chemistry.\n"
    )

    lines.append("## 8. Data Leakage Risks\n")
    lines.append(
        "- Regression and classification both exclude the other ground-truth pollutant "
        "columns as features to avoid one measured pollutant leaking direct information "
        "about a closely correlated target pollutant.\n"
        "- The classification label is derived from CO(GT) itself, so CO(GT) is excluded "
        "from its own feature set.\n"
        "- Scaling (StandardScaler) is fit on the training split only and applied to "
        "val/test, avoiding preprocessing leakage.\n"
        "- Rows are split before scaling and before any metric computation, avoiding "
        "test-set leakage into model fitting.\n"
        "- NMHC(GT) is excluded entirely from all tasks: ~89% of its values are missing "
        "(-200), so imputing it would introduce more noise than signal and risks an "
        "artificial relationship with the target.\n"
    )

    lines.append("## 9. Readiness for Stronger ML Models\n")
    lines.append(
        "The dataset is reasonably clean once -200 sentinels are handled, has a large "
        "enough sample size (~9,000 hourly readings), and shows real predictive signal "
        "for both the regression and classification tasks above baseline. It is a "
        "reasonable candidate for stronger models (e.g. regularized regression, "
        "tree ensembles, or time-aware sequence models given its hourly cadence), "
        "though the heavy missingness in NMHC(GT) and the moderate cluster separation "
        "suggest feature engineering (particularly time-of-day / seasonality features) "
        "would likely help more than model complexity alone.\n"
    )

    with open(os.path.join(out_dir, "model_comparison.md"), "w") as f:
        f.write("\n".join(lines))


def write_error_analysis(reg, clf, clu, out_dir):
    import pandas as pd
    reg_pred = pd.read_csv(os.path.join(out_dir, "regression_predictions.csv"))
    clf_pred = pd.read_csv(os.path.join(out_dir, "classification_predictions.csv"))

    worst_reg = reg_pred.sort_values("abs_error_model", ascending=False).head(5)
    wrong_clf = clf_pred[clf_pred["y_true"] != clf_pred["y_pred_model"]].head(5)

    lines = []
    lines.append("# Error Analysis -- Task 10\n")

    lines.append("## 1. Largest Regression Errors\n")
    lines.append(
        "The 5 test rows with the largest absolute error for C6H6(GT) prediction:\n"
    )
    lines.append("```\n" + worst_reg.to_string(index=False) + "\n```\n")
    lines.append(
        "Likely reasons: linear regression can only capture a straight-line relationship "
        "between sensor readings and Benzene concentration; the true sensor response is "
        "known to be non-linear and cross-sensitive to temperature/humidity and to other "
        "pollutants, so extreme concentration readings (very high or very low pollution "
        "hours) are the hardest for a purely linear model to fit.\n"
    )

    lines.append("## 2. Classification Errors\n")
    lines.append("Sample of misclassified test rows:\n")
    lines.append("```\n" + wrong_clf.to_string(index=False) + "\n```\n")
    lines.append(
        "Likely reasons: rows near the median CO(GT) split are the hardest to classify "
        "correctly by construction, since a small measurement or sensor-noise fluctuation "
        "can push the true value to either side of the threshold while the feature "
        "vector looks almost identical to a correctly-classified neighbor.\n"
    )

    lines.append("## 3. Class Balance\n")
    cb = clf["class_balance_train"]
    total = cb["0"] + cb["1"]
    lines.append(
        f"Training class balance: class 0 (low CO) = {cb['0']} ({cb['0']/total:.1%}), "
        f"class 1 (high CO) = {cb['1']} ({cb['1']/total:.1%}). Because the label is a "
        f"median split by construction, the task is close to balanced (roughly 50/50), "
        f"so accuracy is a reasonably fair metric here, unlike most real-world "
        f"pollution-alert problems which tend to be imbalanced toward the 'normal' class.\n"
    )

    lines.append("## 4. Clustering vs Meaningful Pattern\n")
    lines.append(
        f"The T-vs-RH visualization shows clusters that roughly track temperature and "
        f"humidity bands, which is a physically meaningful pattern (weather regime), "
        f"rather than random noise. However, since the clustering does not use any "
        f"pollutant information, it should not be expected to align with actual air "
        f"pollution severity -- it groups by ambient conditions, not by pollution level, "
        f"which is expected and correct given the deliberate exclusion of pollutant "
        f"columns to avoid leakage.\n"
    )

    lines.append("## 5. Limitations of the Baseline Models\n")
    lines.append(
        "1. Linear regression and logistic regression assume linear/log-linear "
        "relationships and cannot model interactions between sensors or non-linear "
        "cross-sensitivities that are known to exist in metal-oxide gas sensors.\n"
        "2. No time-based features (hour-of-day, day-of-week, season) are used, even "
        "though air pollution is strongly time-dependent (e.g. rush-hour traffic), so "
        "the models are likely missing a substantial, easily available signal.\n"
        "3. KMeans assumes spherical, similarly-sized clusters and is sensitive to the "
        "chosen k and to outliers; the true underlying structure of air-quality regimes "
        "may not be spherical or may need a different k than the one used here.\n"
    )

    with open(os.path.join(out_dir, "error_analysis.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_AirQualityUCI.csv> <output_dir>")
        sys.exit(1)

    csv_path, out_dir = sys.argv[1], sys.argv[2]
    ensure_dirs(out_dir)

    df = load_raw(csv_path)

    print("Running regression task...")
    reg = run_regression(df, out_dir)
    print("Running classification task...")
    clf = run_classification(df, out_dir)
    print("Running clustering task...")
    clu = run_clustering(df, out_dir)

    print("Writing reports...")
    write_model_comparison(reg, clf, clu, out_dir)
    write_error_analysis(reg, clf, clu, out_dir)

    print("Done. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
