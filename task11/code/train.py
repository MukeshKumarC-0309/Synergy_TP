"""
Task 11 - Practical Classification Using Machine-Learning Libraries
Dataset: Crop_recommendation.csv (22-class crop recommendation)

Runs the full workflow: EDA -> split -> baseline -> model comparison ->
evaluation/error analysis -> threshold analysis -> save final pipeline.

Usage: python train.py
Outputs land in ../outputs and ../plots.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data.csv"
PLOTS = ROOT / "plots"
EDA = ROOT / "eda"
METRICS = ROOT / "metrics"
ERRORS = ROOT / "error_analysis"
MODEL = ROOT / "model"
for d in (PLOTS, EDA, METRICS, ERRORS, MODEL):
    d.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Part 1: Dataset understanding
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

summary = {
    "n_rows": len(df),
    "n_features": len(FEATURES),
    "n_classes": df[TARGET].nunique(),
    "missing_values": int(df.isna().sum().sum()),
    "class_counts": df[TARGET].value_counts().to_dict(),
    "feature_describe": df[FEATURES].describe().to_dict(),
}
with open(EDA / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Rows: {summary['n_rows']}, Classes: {summary['n_classes']}, Missing: {summary['missing_values']}")

# Class distribution plot
plt.figure(figsize=(10, 6))
df[TARGET].value_counts().sort_values().plot(kind="barh")
plt.title("Class distribution (Crop_recommendation.csv)")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(PLOTS / "class_distribution.png", dpi=120)
plt.close()

# Feature correlation with class separability hint (pairwise means per class)
class_means = df.groupby(TARGET)[FEATURES].mean()
class_means.to_csv(EDA / "class_feature_means.csv")

# ---------------------------------------------------------------------------
# Part 2: Split
# ---------------------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
)  # 0.25 * 0.8 = 0.2 -> 60/20/20 overall

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ---------------------------------------------------------------------------
# Part 2/3: Models + evaluation
# ---------------------------------------------------------------------------
models = {
    "baseline_majority": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = []
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    train_pred = pipe.predict(X_train)
    val_pred = pipe.predict(X_val)

    row = {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_precision_macro": precision_score(y_val, val_pred, average="macro", zero_division=0),
        "val_recall_macro": recall_score(y_val, val_pred, average="macro", zero_division=0),
        "val_f1_macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
    }
    row["train_val_gap"] = row["train_accuracy"] - row["val_accuracy"]
    results.append(row)
    print(f"{name:20s} train_acc={row['train_accuracy']:.4f}  val_acc={row['val_accuracy']:.4f}  "
          f"val_f1_macro={row['val_f1_macro']:.4f}  gap={row['train_val_gap']:.4f}")

results_df = pd.DataFrame(results).sort_values("val_f1_macro", ascending=False)
results_df.to_csv(METRICS / "model_comparison_val.csv", index=False)

# ---------------------------------------------------------------------------
# Confusion matrices (validation set) for every model
# ---------------------------------------------------------------------------
labels_sorted = sorted(y.unique())
for name, pipe in fitted.items():
    val_pred = pipe.predict(X_val)
    cm = confusion_matrix(y_val, val_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(11, 11))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix (val) - {name}")
    plt.tight_layout()
    plt.savefig(PLOTS / f"confusion_matrix_{name}.png", dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
# Select final model: best val_f1_macro, tie-break by smallest train/val gap
# ---------------------------------------------------------------------------
final_name = results_df.iloc[0]["model"]
if final_name == "baseline_majority":
    final_name = results_df.iloc[1]["model"]  # never ship the baseline
final_pipe = fitted[final_name]
print(f"\nSelected final model: {final_name}")

# Refit selected model on train+val, evaluate once on held-out test set
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_pipe.fit(X_trainval, y_trainval)
test_pred = final_pipe.predict(X_test)

test_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
pd.DataFrame(test_report).transpose().to_csv(METRICS / "final_model_test_classification_report.csv")

test_summary = {
    "final_model": final_name,
    "test_accuracy": accuracy_score(y_test, test_pred),
    "test_precision_macro": precision_score(y_test, test_pred, average="macro", zero_division=0),
    "test_recall_macro": recall_score(y_test, test_pred, average="macro", zero_division=0),
    "test_f1_macro": f1_score(y_test, test_pred, average="macro", zero_division=0),
}
with open(METRICS / "final_model_test_summary.json", "w") as f:
    json.dump(test_summary, f, indent=2)
print(json.dumps(test_summary, indent=2))

cm_test = confusion_matrix(y_test, test_pred, labels=labels_sorted)
fig, ax = plt.subplots(figsize=(11, 11))
ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels_sorted).plot(
    ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues"
)
ax.set_title(f"Confusion matrix (TEST) - final model: {final_name}")
plt.tight_layout()
plt.savefig(PLOTS / "confusion_matrix_final_TEST.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# Error analysis: misclassified samples + most-confused class pairs
# ---------------------------------------------------------------------------
error_mask = test_pred != y_test.values
errors_df = X_test[error_mask].copy()
errors_df["true_label"] = y_test.values[error_mask]
errors_df["predicted_label"] = test_pred[error_mask]
errors_df.to_csv(ERRORS / "misclassified_test_samples.csv", index=False)

confused_pairs = (
    errors_df.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
    .sort_values("count", ascending=False)
)
confused_pairs.to_csv(ERRORS / "most_confused_pairs.csv", index=False)
print(f"\nTest errors: {error_mask.sum()} / {len(y_test)}")
print("Top confused pairs:")
print(confused_pairs.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# Part 4: Probability / threshold analysis
# Dataset is multiclass (22 classes), so the literal "binary threshold" recipe
# doesn't apply directly. We adapt it: take the single most-confused class
# pair from the error analysis above and run a one-vs-rest binary threshold
# sweep between those two classes using the final model's predicted
# probabilities, which is the closest faithful analogue.
# ---------------------------------------------------------------------------
if len(confused_pairs) > 0:
    class_a, class_b = confused_pairs.iloc[0]["true_label"], confused_pairs.iloc[0]["predicted_label"]
    mask_ab = y_test.isin([class_a, class_b])
    X_ab, y_ab = X_test[mask_ab], y_test[mask_ab]

    proba = final_pipe.predict_proba(X_ab)
    class_list = list(final_pipe.named_steps["clf"].classes_) if hasattr(
        final_pipe.named_steps["clf"], "classes_"
    ) else list(final_pipe.classes_)
    idx_a = class_list.index(class_a)

    thresholds = np.arange(0.1, 1.0, 0.1)
    sweep_rows = []
    y_ab_bin = (y_ab == class_a).astype(int).values
    p_a = proba[:, idx_a]
    for t in thresholds:
        pred_bin = (p_a >= t).astype(int)
        tp = int(((pred_bin == 1) & (y_ab_bin == 1)).sum())
        fp = int(((pred_bin == 1) & (y_ab_bin == 0)).sum())
        fn = int(((pred_bin == 0) & (y_ab_bin == 1)).sum())
        tn = int(((pred_bin == 0) & (y_ab_bin == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        sweep_rows.append({"threshold": round(t, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                            "precision": precision, "recall": recall})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(ERRORS / f"threshold_sweep_{class_a}_vs_{class_b}.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], marker="o", label="precision")
    plt.plot(sweep_df["threshold"], sweep_df["recall"], marker="o", label="recall")
    plt.xlabel(f"P({class_a}) threshold")
    plt.ylabel("score")
    plt.title(f"Threshold sweep: {class_a} vs {class_b}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / f"threshold_sweep_{class_a}_vs_{class_b}.png", dpi=120)
    plt.close()

    with open(ERRORS / "threshold_analysis_pair.json", "w") as f:
        json.dump({"class_a": class_a, "class_b": class_b}, f, indent=2)

# ---------------------------------------------------------------------------
# Save the final pipeline
# ---------------------------------------------------------------------------
joblib.dump(final_pipe, MODEL / "final_pipeline.joblib")
with open(MODEL / "final_model_name.txt", "w") as f:
    f.write(final_name)

print("\nDone. Pipeline saved to model/final_pipeline.joblib")
