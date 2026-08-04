# Error Analysis

Final model: **Ridge**

Test RMSE: 2.9243 | Test MAE: 2.5129 | Test R2: -0.0070

Baseline (mean) test RMSE: 2.9264 | Baseline test R2: -0.0084

## Largest absolute errors on the test set

   y_true    y_pred  abs_error
30.659196 24.889473   5.769723
19.244612 24.836556   5.591944
30.468278 24.917247   5.551031
19.378066 24.900693   5.522627
19.476667 24.984356   5.507689
19.632229 25.114906   5.482677
19.453496 24.908777   5.455280
30.365757 24.944633   5.421124
19.673706 25.071741   5.398036
30.412324 25.057334   5.354990
30.492791 25.249811   5.242980
20.081981 25.307983   5.226002
19.727484 24.952773   5.225288
30.202844 25.025220   5.177624
30.024210 24.848848   5.175362
19.768664 24.921370   5.152705
30.163724 25.016139   5.147584
19.841930 24.983507   5.141577
19.913929 25.014952   5.101023
19.812733 24.892665   5.079932

## Observation

All models -- including tree-based, non-linear models -- perform at essentially the same level as the mean baseline (R2 close to 0 for every model, see model_comparison_metrics.csv). This indicates the five sensor features carry little to no predictive signal for Temperature in this dataset, rather than indicating a modelling failure -- a RandomForest with equal feature importances across all five sensors and negative test R2 is a strong sign of an essentially unpredictable target from these inputs, not underfitting that a better model would fix.
