# Feature Summary

## Chosen stability thresholds

Stable ≤ 0.05, moderate 0.05–0.085, unstable > 0.085. The cutoff sits deliberately above Biochem low-concentration's 0.083 — that group's high relative spread comes from tiny absolute absorbance values near the detection floor, which is inherent measurement behaviour, not a quality failure, so it's flagged moderate and kept. It sits below Mechanical high-load's 0.101, which is driven by a single anomalous 2.10 mm reading — a genuine data-quality flag — so that group is marked unstable and excluded from the ML-ready set.

## Answers

**Which features are general across all domains?** `rolling_average_signal`, `normalized_signal`, `error_percent`, `stability_flag`, and `ml_ready` — they only need columns that every domain has.

**Which features are domain-specific?** `power_w` (Electronics only) and `stress_ratio` (Mechanical only).

**Which rows are not ML-ready and why?** Rows M007–M009, the Mechanical high-load group. Their replicate group is flagged unstable — its coefficient of variation (0.101) exceeds the 0.085 threshold because of an outlying 2.10 mm reading — so the ml_ready gate returns False.

**Which engineered feature is most useful for Electronics?** `power_w`, as it converts two raw columns belonging to the domain into a new physical quantity, which provides more predictive information than voltage alone.

**Which engineered feature is most useful for Mechanical?** `stress_ratio`. It compares each sample's stress to a reference limit, providing information about how close the sample is to failure.

**Which engineered feature is most useful for Biochem?** `error_percent`. It directly measures how far each reading drifts from its expected value, which is the core accuracy signal for a calibration assay.

**Why should invalid domain features be left blank instead of forcing a value?** Invalid domain features should be left blank (NaN), not forced to 0, because 0 is itself a valid measurement. A forced zero looks like a real reading of zero, so it lies to any analysis — dragging down means and skewing correlations. NaN is ignored by statistics, so it honestly signals "not applicable" instead.

**How can feature engineering introduce misleading information?** Feature engineering can introduce misleading information in three main ways:

1. **Leakage** — if a feature secretly contains the answer, the model looks great in training but collapses in reality.
2. **Fake values** — forcing in values that were never measured (e.g. writing 0 instead of leaving NaN) distorts the data with readings that don't exist.
3. **Over-smoothing** — a rolling average can erase a real spike that signals a genuine problem you needed to catch.
