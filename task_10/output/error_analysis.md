# Error Analysis -- Task 10

## 1. Largest Regression Errors

The 5 test rows with the largest absolute error for C6H6(GT) prediction:

```
 y_true  y_pred_model  y_pred_baseline  abs_error_model
   50.6     40.588135        10.054616        10.011865
   50.7     41.212608        10.054616         9.487392
   48.2     39.336427        10.054616         8.863573
   45.2     37.476568        10.054616         7.723432
   44.5     37.028411        10.054616         7.471589
```

Likely reasons: linear regression can only capture a straight-line relationship between sensor readings and Benzene concentration; the true sensor response is known to be non-linear and cross-sensitive to temperature/humidity and to other pollutants, so extreme concentration readings (very high or very low pollution hours) are the hardest for a purely linear model to fit.

## 2. Classification Errors

Sample of misclassified test rows:

```
 y_true  y_pred_model  y_pred_baseline  pred_proba_model
      1             0                0          0.458226
      0             1                0          0.898930
      1             0                0          0.007848
      0             1                0          0.947049
      1             0                0          0.485328
```

Likely reasons: rows near the median CO(GT) split are the hardest to classify correctly by construction, since a small measurement or sensor-noise fluctuation can push the true value to either side of the threshold while the feature vector looks almost identical to a correctly-classified neighbor.

## 3. Class Balance

Training class balance: class 0 (low CO) = 2634 (51.2%), class 1 (high CO) = 2506 (48.8%). Because the label is a median split by construction, the task is close to balanced (roughly 50/50), so accuracy is a reasonably fair metric here, unlike most real-world pollution-alert problems which tend to be imbalanced toward the 'normal' class.

## 4. Clustering vs Meaningful Pattern

The T-vs-RH visualization shows clusters that roughly track temperature and humidity bands, which is a physically meaningful pattern (weather regime), rather than random noise. However, since the clustering does not use any pollutant information, it should not be expected to align with actual air pollution severity -- it groups by ambient conditions, not by pollution level, which is expected and correct given the deliberate exclusion of pollutant columns to avoid leakage.

## 5. Limitations of the Baseline Models

1. Linear regression and logistic regression assume linear/log-linear relationships and cannot model interactions between sensors or non-linear cross-sensitivities that are known to exist in metal-oxide gas sensors.
2. No time-based features (hour-of-day, day-of-week, season) are used, even though air pollution is strongly time-dependent (e.g. rush-hour traffic), so the models are likely missing a substantial, easily available signal.
3. KMeans assumes spherical, similarly-sized clusters and is sensitive to the chosen k and to outliers; the true underlying structure of air-quality regimes may not be spherical or may need a different k than the one used here.
