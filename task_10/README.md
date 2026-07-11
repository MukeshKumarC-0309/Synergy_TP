# Task 10 -- Baseline ML from Scratch (AirQualityUCI)

Regression + classification + clustering on the UCI Air Quality dataset,
built from scratch with only NumPy, pandas, and matplotlib -- no scikit-learn.

## Dataset

[AirQualityUCI](https://archive.ics.uci.edu/dataset/360/air+quality) --
~9,000 hourly readings (March 2004 -- April 2005) from 5 gas sensors plus
a certified reference analyzer, temperature, and humidity.

## Tasks

| Task | Target | Type |
|---|---|---|
| Regression | `C6H6(GT)` (Benzene) | continuous |
| Classification | `CO(GT) > median` | binary (high vs low CO) |
| Clustering | sensor + weather similarity | unsupervised |

## Models  

- Baselines: mean predictor, majority-class predictor
- Linear Regression (batch gradient descent, MSE loss)
- Logistic Regression (batch gradient descent, cross-entropy loss)
- KMeans (k-means++ init + Lloyd's algorithm)

All metrics (MAE, RMSE, R2, accuracy, precision, recall, F1, confusion
matrix, inertia, silhouette) are implemented manually in `src/metrics.py`.

## Outputs (`output/`)

Metrics (JSON), predictions (CSV), plots (PNG), and two reports:
- `model_comparison.md` -- target justification, baseline comparisons, leakage
- `error_analysis.md` -- worst predictions, class balance, cluster interpretation

## Run it

```bash
python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output
```

## Confirmation

No scikit-learn or any ready-made ML library used anywhere. Every model,
baseline, and metric is implemented in `src/`.
