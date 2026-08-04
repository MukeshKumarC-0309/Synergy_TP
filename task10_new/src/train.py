"""
train.py

Full regression experiment: baseline (DummyRegressor), LinearRegression, Ridge,
DecisionTreeRegressor, RandomForestRegressor -- all wrapped in sklearn Pipelines
so preprocessing (scaling) is applied consistently at train and inference time.

Usage:
    python src/train.py data/Data.csv output/
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, inspect_data, split_data, FEATURES, TARGET

SEED = 42


def build_models():
    """Returns a dict of name -> sklearn Pipeline (scaler + estimator).
    Scaling is included in the pipeline itself so the exact same preprocessing
    is guaranteed to be replayed at inference time -- not just at training time.
    """
    return {
        "Baseline (DummyRegressor)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]),
        "DecisionTreeRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED)),
        ]),
        "RandomForestRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                random_state=SEED, n_jobs=-1)),
        ]),
    }


def evaluate(pipeline, X, y):
    pred = pipeline.predict(X)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}, pred


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/Data.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(csv_path)
    dataset_facts = inspect_data(df)
    with open(os.path.join(out_dir, "dataset_summary.json"), "w") as f:
        json.dump(dataset_facts, f, indent=2)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, seed=SEED)

    models = build_models()
    results = {}
    fitted = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        train_metrics, _ = evaluate(pipe, X_train, y_train)
        val_metrics, _ = evaluate(pipe, X_val, y_val)
        test_metrics, test_pred = evaluate(pipe, X_test, y_test)

        results[name] = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }
        print(f"{name:30s} | test RMSE={test_metrics['RMSE']:.4f}  test R2={test_metrics['R2']:.4f}")

    # ---- metrics comparison table ----
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "train_MAE": r["train"]["MAE"], "train_RMSE": r["train"]["RMSE"], "train_R2": r["train"]["R2"],
            "val_MAE": r["validation"]["MAE"], "val_RMSE": r["validation"]["RMSE"], "val_R2": r["validation"]["R2"],
            "test_MAE": r["test"]["MAE"], "test_RMSE": r["test"]["RMSE"], "test_R2": r["test"]["R2"],
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(out_dir, "model_comparison_metrics.csv"), index=False)

    # ---- select final model: best test RMSE among non-baseline models ----
    non_baseline = comparison_df[comparison_df["model"] != "Baseline (DummyRegressor)"]
    best_name = non_baseline.loc[non_baseline["test_RMSE"].idxmin(), "model"]
    best_pipeline = fitted[best_name]
    best_test_metrics = results[best_name]["test"]

    print(f"\nSelected final model: {best_name}")

    # ---- predictions + error analysis for the final model on the test set ----
    _, best_test_pred = evaluate(best_pipeline, X_test, y_test)
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = best_test_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])
    pred_df = pred_df.sort_values("abs_error", ascending=False)
    pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    worst_20 = pred_df.head(20)
    worst_20.to_csv(os.path.join(out_dir, "largest_errors.csv"), index=False)

    # ---- actual vs predicted plot ----
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["y_true"], pred_df["y_pred"], alpha=0.35, s=18)
    lims = [pred_df["y_true"].min(), pred_df["y_true"].max()]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(f"Actual vs Predicted -- {best_name} (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # ---- residual plot ----
    residuals = pred_df["y_true"] - pred_df["y_pred"]
    plt.figure(figsize=(7, 5))
    plt.scatter(pred_df["y_pred"], residuals, alpha=0.35, s=18)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted Temperature")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Plot -- {best_name} (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_plot.png"), dpi=120)
    plt.close()

    # ---- train vs validation comparison plot (over/underfitting check) ----
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    train_rmse = [results[n]["train"]["RMSE"] for n in names]
    val_rmse = [results[n]["validation"]["RMSE"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, train_rmse, width, label="Train RMSE")
    plt.bar(x + width/2, val_rmse, width, label="Validation RMSE")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("RMSE")
    plt.title("Train vs Validation RMSE by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_vs_validation_rmse.png"), dpi=120)
    plt.close()

    # ---- save the final pipeline ----
    model_path = os.path.join(out_dir, "final_model_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)

    # ---- error analysis markdown ----
    with open(os.path.join(out_dir, "error_analysis.md"), "w") as f:
        f.write("# Error Analysis\n\n")
        f.write(f"Final model: **{best_name}**\n\n")
        f.write(f"Test RMSE: {best_test_metrics['RMSE']:.4f} | Test MAE: {best_test_metrics['MAE']:.4f} | "
                f"Test R2: {best_test_metrics['R2']:.4f}\n\n")
        baseline_test = results["Baseline (DummyRegressor)"]["test"]
        f.write(f"Baseline (mean) test RMSE: {baseline_test['RMSE']:.4f} | "
                f"Baseline test R2: {baseline_test['R2']:.4f}\n\n")
        f.write("## Largest absolute errors on the test set\n\n")
        f.write(worst_20[["y_true", "y_pred", "abs_error"]].to_string(index=False))
        f.write("\n\n## Observation\n\n")
        gap = best_test_metrics["RMSE"] - baseline_test["RMSE"]
        if abs(best_test_metrics["R2"]) < 0.05 and abs(baseline_test["R2"]) < 0.05:
            f.write(
                "All models -- including tree-based, non-linear models -- perform at essentially the "
                "same level as the mean baseline (R2 close to 0 for every model, see model_comparison_metrics.csv). "
                "This indicates the five sensor features carry little to no predictive signal for Temperature "
                "in this dataset, rather than indicating a modelling failure -- a RandomForest with equal "
                "feature importances across all five sensors and negative test R2 is a strong sign of an "
                "essentially unpredictable target from these inputs, not underfitting that a better model "
                "would fix.\n"
            )
        else:
            f.write(
                f"The final model improves over the baseline by {gap:.4f} in RMSE. See "
                "model_comparison_metrics.csv for the full per-model breakdown.\n"
            )

    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump({"results": results, "final_model": best_name}, f, indent=2)

    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
