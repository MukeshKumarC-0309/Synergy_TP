# Task 10 -- Baseline Machine Learning from Scratch (AirQualityUCI)

Software/ML Domain Task 10: a complete regression + classification +
clustering ML workflow built entirely from scratch (NumPy/pandas/matplotlib
only) on the UCI Air Quality dataset.

## Dataset

**Source:** [AirQualityUCI](https://archive.ics.uci.edu/dataset/360/air+quality)
-- UCI Machine Learning Repository. Hourly averaged responses from an array
of 5 metal-oxide chemical sensors embedded in an Air Quality Chemical
Multisensor Device, plus reference concentrations from a co-located
certified analyzer, plus temperature/humidity, over ~9,000 hours
(March 2004 -- April 2005) in an Italian city.

## Implemented Models (all from scratch, no scikit-learn)

- **Baselines:** mean predictor (regression), majority-class predictor (classification)
- **Linear Regression** -- batch gradient descent, MSE loss
- **Logistic Regression** -- batch gradient descent, binary cross-entropy loss
- **KMeans** -- k-means++ initialization + Lloyd's algorithm

All metrics (MAE, MSE, RMSE, R2, accuracy, precision, recall, F1,
confusion matrix, inertia, silhouette score) are implemented manually in
`src/metrics.py`.

## Tasks

| Task | Target | Type |
|---|---|---|
| Regression | `C6H6(GT)` (Benzene concentration) | continuous |
| Classification | `CO(GT) > median` | binary (high vs low CO) |
| Clustering | grouping by sensor + weather similarity | unsupervised, label not used |

See `output/model_comparison.md` for target justification, baseline
comparisons, and leakage analysis, and `output/error_analysis.md` for
failure-case analysis.

## Generated Outputs (`output/`)

| File | Contents |
|---|---|
| `regression_metrics.json` | Baseline vs. linear regression metrics for C6H6(GT) |
| `classification_metrics.json` | Baseline vs. logistic regression metrics for high/low CO |
| `clustering_metrics.json` | Inertia, silhouette score, cluster sizes |
| `regression_predictions.csv` | Per-row test-set predictions (model + baseline) and errors |
| `classification_predictions.csv` | Per-row test-set predictions, probabilities, and true labels |
| `clustering_assignments.csv` | Cluster assignment per row with raw feature values |
| `regression_loss_curve.png` | Train/val MSE vs. gradient descent iteration |
| `classification_loss_curve.png` | Train/val binary cross-entropy vs. iteration |
| `actual_vs_predicted.png` | Scatter of actual vs. predicted C6H6(GT) on test set |
| `confusion_matrix.png` | Test-set confusion matrix for the high/low CO classifier |
| `clustering_plot.png` | 2D (Temperature vs. Humidity) view of the 3 clusters |
| `model_comparison.md` | Target justification, baseline comparisons, leakage discussion |
| `error_analysis.md` | Worst predictions, class balance, cluster interpretation, limitations |

## How to Run

```bash
python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output
```

This single command runs the full pipeline: loads and cleans the data,
trains all three baselines and all three models, computes every metric,
writes every output file listed above, and regenerates all plots.

## Confirmation

**No scikit-learn or any other ready-made ML library (XGBoost, LightGBM,
TensorFlow, PyTorch, Keras, statsmodels, etc.) was used anywhere in this
task.** Only `pandas` (data loading/tabular handling), `numpy` (numerical
operations), and `matplotlib` (plotting) are used. Every model, baseline,
and metric is implemented in-house in `src/`.
