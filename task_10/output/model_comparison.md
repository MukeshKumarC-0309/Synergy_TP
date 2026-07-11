# Model Comparison -- Task 10

## 1-2. Regression: C6H6(GT) (Benzene concentration)

**Target:** `C6H6(GT)` -- a continuous, directly measured pollutant concentration (micrograms/m^3-equivalent reference value). It is a valid continuous prediction task because it is a real-valued physical measurement with no inherent class boundaries, and it can be estimated from correlated metal-oxide sensor readings.

**Baseline (mean predictor):** RMSE = 7.422, MAE = 5.706, R2 = -0.001

**Linear Regression (GD):** RMSE = 1.204, MAE = 0.857, R2 = 0.974

The trained model **beats** the mean baseline on RMSE.

## 3-5. Classification: High vs Low CO(GT)

**Target:** binary label derived by splitting `CO(GT)` at its training median (1.80): 1 = above-median CO, 0 = at/below-median CO. This was chosen (rather than using an existing categorical column, since none exists in the raw data) to create a class-balanced supervised classification problem from a continuous pollutant.

**Baseline (majority class):** accuracy = 0.509, F1 = 0.000

**Logistic Regression (GD):** accuracy = 0.928, precision = 0.944, recall = 0.908, F1 = 0.926

The trained model **beats** the majority-class baseline on F1.

**Which error is worse:** a false negative (predicting *low* CO when it is actually *high*) is more serious than a false positive here, since it is an air-quality / health-risk monitoring context -- missing a genuine high-pollution reading has a higher real-world cost than a false alarm. Recall on the high-CO class is therefore the more important metric than raw accuracy.

## 6-7. Clustering: Sensor + Weather Similarity

**Features used:** PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3), T, RH, AH. Ground-truth pollutant columns (CO(GT), NOx(GT), NO2(GT), C6H6(GT)) and the derived classification label were deliberately excluded from clustering so that the grouping reflects only sensor/weather similarity and is not shaped by the labels used elsewhere in this task.

**Result:** k=3 clusters, inertia = 36956.8, silhouette score (subsampled) = 0.260, cluster sizes = {0: 2396, 1: 3145, 2: 3450}.

A silhouette score of 0.260 suggests the clusters are reasonably well-separated. Visual inspection (T vs RH) shows the clusters correspond broadly to different temperature/humidity regimes rather than arbitrary noise, so the structure looks meaningful rather than purely artificial, though it is a coarse grouping driven mainly by weather conditions rather than pollution chemistry.

## 8. Data Leakage Risks

- Regression and classification both exclude the other ground-truth pollutant columns as features to avoid one measured pollutant leaking direct information about a closely correlated target pollutant.
- The classification label is derived from CO(GT) itself, so CO(GT) is excluded from its own feature set.
- Scaling (StandardScaler) is fit on the training split only and applied to val/test, avoiding preprocessing leakage.
- Rows are split before scaling and before any metric computation, avoiding test-set leakage into model fitting.
- NMHC(GT) is excluded entirely from all tasks: ~89% of its values are missing (-200), so imputing it would introduce more noise than signal and risks an artificial relationship with the target.

## 9. Readiness for Stronger ML Models

The dataset is reasonably clean once -200 sentinels are handled, has a large enough sample size (~9,000 hourly readings), and shows real predictive signal for both the regression and classification tasks above baseline. It is a reasonable candidate for stronger models (e.g. regularized regression, tree ensembles, or time-aware sequence models given its hourly cadence), though the heavy missingness in NMHC(GT) and the moderate cluster separation suggest feature engineering (particularly time-of-day / seasonality features) would likely help more than model complexity alone.
