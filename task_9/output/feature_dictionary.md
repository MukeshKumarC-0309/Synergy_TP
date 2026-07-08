# Feature Dictionary

Each engineered feature is defined below with its formula, applicable domain, required columns, invalidity condition, and machine-learning rationale.

| Feature | Formula / rule | Applies to | Required columns | Invalid when | Why useful for ML |
|---|---|---|---|---|---|
| `rolling_average_signal` | rolling mean of `signal`, window 3, within each (domain, condition) group, ordered by `time_step` | all domains where row order is meaningful | `signal`, `time_step`, `domain`, `condition` | rows are unordered, or the window crosses into an unrelated condition | smooths measurement noise so the model sees the underlying trend, not single-reading jitter |
| `normalized_signal` | `signal / baseline_signal` | all domains with a valid baseline | `signal`, `baseline_signal` | `baseline_signal` is missing or zero | puts readings from different scales on a common baseline-relative scale, making them comparable |
| `power_w` | `voltage_v * current_a` | Electronics only | `voltage_v`, `current_a` | the row is Biochem or Mechanical (no voltage/current) | electrical power is a physically meaningful quantity that predicts heating and load behaviour better than voltage alone |
| `error_percent` | `((signal - expected_signal) / expected_signal) * 100` | all domains with a valid expected signal | `signal`, `expected_signal` | `expected_signal` is missing or zero | expresses calibration accuracy directly, flagging rows that deviate from their expected value |
| `stress_ratio` | `stress_mpa / reference_stress_mpa` | Mechanical only | `stress_mpa`, `reference_stress_mpa` | the row is Biochem or Electronics (no stress values) | a dimensionless load-relative stress showing how close a sample is to its reference limit |
| `stability_flag` | from group CoV — stable (≤ 0.05), moderate (0.05–0.085), unstable (> 0.085) | all replicate groups | replicate summary's `coefficient_of_variation` | CoV cannot be computed (fewer than two valid replicates) | lets the pipeline down-weight or exclude noisy groups before training |
| `ml_ready` | boolean AND of valid signal, valid non-zero expected signal, valid input value, present domain/condition, valid normalized_signal, and stability_flag of stable or moderate | all domains | all of the above | any required value is missing/zero or the group is unstable | a single gate for selecting only trustworthy rows for model training |
