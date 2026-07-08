# Task 9 — Calibration Statistics, Correlation Analysis, and Feature Engineering

## Run

```
python task_9/src/main.py task_9/data/calibration_measurements.csv task_9/output
```

## Pipeline

1. **replicate_statistics.py** — groups replicates and computes mean, median, variance,
   sample SD, standard error, 95% t-confidence interval, coefficient of variation, and a
   stability flag. Groups with < 2 valid readings are marked unreliable.
2. **correlation_analysis.py** — Pearson/Spearman correlation and a simple linear calibration
   fit (slope, intercept, R-squared, MAE, RMSE) for each domain relationship; produces the
   calibration curves and the raw signal-vs-input scatter plot.
3. **feature_engineering.py** — adds rolling_average_signal, normalized_signal, power_w
   (Electronics), error_percent, stress_ratio (Mechanical), stability_flag, and the ml_ready gate.
4. **main.py** — orchestrates the three modules and writes every CSV, markdown, and PNG output.

## Outputs (in `output/`)

CSV: replicate_summary, calibration_summary, correlation_summary, engineered_features, ml_ready_dataset.
Markdown: replicate_analysis, correlation_limitations, feature_dictionary, feature_summary.
PNG: calibration_curve_biochem, calibration_curve_electronics, calibration_curve_mechanical, correlation_signal_input.
