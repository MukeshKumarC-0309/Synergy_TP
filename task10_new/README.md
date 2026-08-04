# Task 10 -- Practical Regression Using Machine-Learning Libraries

Library-based regression workflow (scikit-learn) predicting `Temperature`
from five sensor readings, comparing a baseline, linear models, and
tree-based models, with a saved pipeline and a separate inference script.

## Dataset

`data/Data.csv` -- 3,457 rows, 6 columns: `Temperature` (target, continuous,
range ~19-32) and `Sensor1`-`Sensor5` (features, each roughly in [0, 1], no
missing values).

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## Run the training pipeline

```bash
python src/train.py data/Data.csv output/
```

This loads the data, splits it 70/15/15 (train/validation/test), trains a
`DummyRegressor` baseline plus `LinearRegression`, `Ridge`,
`DecisionTreeRegressor`, and `RandomForestRegressor` (each wrapped in a
`Pipeline` with `StandardScaler`), evaluates all five on train/validation/test,
selects the best non-baseline model by test RMSE, and writes every output
listed below.

## Run inference on new data

```bash
# Single record
python src/inference.py --model output/final_model_pipeline.joblib \
    --values 0.51 0.49 0.50 0.48 0.52

# Batch of records from a CSV (must have columns Sensor1..Sensor5)
python src/inference.py --model output/final_model_pipeline.joblib \
    --csv new_records.csv --out output/new_predictions.csv
```

`inference.py` is fully standalone -- it only needs the saved `.joblib` file
and does not import anything from `train.py`.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `dataset_summary.json` | Row/column counts, dtypes, missing values, target range, feature ranges, correlations |
| `model_comparison_metrics.csv` | MAE/RMSE/R2 for every model on train, validation, and test |
| `results_summary.json` | Same metrics plus which model was selected as final |
| `test_predictions.csv` | Every test-set row with true value, prediction, and absolute error |
| `largest_errors.csv` | The 20 worst test-set predictions by absolute error |
| `error_analysis.md` | Final model, its metrics vs. baseline, worst errors, and a written observation |
| `actual_vs_predicted.png` | Scatter of actual vs. predicted Temperature (test set) |
| `residual_plot.png` | Residuals vs. predicted value (test set) |
| `train_vs_validation_rmse.png` | RMSE per model, train vs. validation, for over/underfitting comparison |
| `final_model_pipeline.joblib` | The saved scaler+model pipeline (reusable via inference.py) |

## Key finding

Every model -- including the non-linear RandomForest -- performs at
essentially the same level as the mean-only baseline (R2 near zero or
negative for all five models on the test set; see
`model_comparison_metrics.csv`). RandomForest's feature importances came
out almost perfectly equal across all five sensors (~0.20 each), which is
the signature of a model unable to find any real feature to split on. Full
reasoning is in `error_analysis.md` and the technical report -- this is
reported as a property of the dataset (little to no genuine relationship
between these sensors and Temperature), not a pipeline or modelling error.

## Confirmation

No manual/from-scratch model code here -- this task specifically uses
scikit-learn (`DummyRegressor`, `LinearRegression`, `Ridge`,
`DecisionTreeRegressor`, `RandomForestRegressor`) and `joblib` for saving
and reloading the trained pipeline, as required by the task brief.
