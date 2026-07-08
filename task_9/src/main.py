import os
import sys

import numpy as np
import pandas as pd

import replicate_statistics as rs
import correlation_analysis as ca
import feature_engineering as fe


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unreliable"
    return f"{x:.{nd}f}"


def _group_label(row):
    return f"{row['domain']} / {row['condition']} (input {row['input_value']} {row['input_unit']})"


def write_replicate_analysis(summary, path):
    valid = summary.dropna(subset=["coefficient_of_variation"]).copy()
    most_stable = valid.loc[valid["coefficient_of_variation"].idxmin()]
    most_noisy = valid.loc[valid["coefficient_of_variation"].idxmax()]
    summary["ci_width"] = summary["confidence_interval_upper"] - summary["confidence_interval_lower"]
    widest = summary.loc[summary["ci_width"].idxmax()]
    highest_cov = valid.loc[valid["coefficient_of_variation"].idxmax()]

    lines = []
    lines.append("# Replicate Analysis\n")
    lines.append("Each replicate group below is one (domain, condition, input_value) cell measured "
                 "three times. Sample variance and sample standard deviation (ddof = 1) are used, and "
                 "the 95% confidence interval uses the t-distribution with df = n - 1.\n")

    lines.append("## Per-group summary\n")
    show = ["domain", "condition", "input_value", "replicate_count", "mean_signal",
            "standard_deviation_signal", "standard_error_signal",
            "confidence_interval_lower", "confidence_interval_upper",
            "coefficient_of_variation", "stability_flag"]
    hdr = "| " + " | ".join(show) + " |"
    sep = "| " + " | ".join(["---"] * len(show)) + " |"
    lines.append(hdr)
    lines.append(sep)
    for _, r in summary.iterrows():
        cells = []
        for c in show:
            v = r[c]
            cells.append(_fmt(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Answers\n")
    lines.append(f"**Which replicate group is most stable?** "
                 f"{_group_label(most_stable)} — lowest coefficient of variation "
                 f"({_fmt(most_stable['coefficient_of_variation'])}). Its three readings sit almost on "
                 f"top of each other, so the mean is trustworthy.\n")
    lines.append(f"**Which replicate group is most noisy?** "
                 f"{_group_label(most_noisy)} — highest coefficient of variation "
                 f"({_fmt(most_noisy['coefficient_of_variation'])}). One reading pulls away from the "
                 f"other two, which inflates the spread.\n")
    lines.append(f"**Which group has the widest confidence interval?** "
                 f"{_group_label(widest)} — CI width {_fmt(widest['ci_width'])}. A wide interval means "
                 f"we are far less sure where the true mean signal actually lies.\n")
    lines.append(f"**Which group has the highest coefficient of variation?** "
                 f"{_group_label(highest_cov)} ({_fmt(highest_cov['coefficient_of_variation'])}). CoV is "
                 f"scale-free, so it is the fair way to compare noise across groups with different mean "
                 f"magnitudes.\n")
    lines.append("**Why is mean alone not enough for judging reliability?** The mean tells you the "
                 "centre of the readings but nothing about how tightly they cluster. Two groups can share "
                 "the same mean while one is tight and one is scattered; only the spread (SD, SE, CI, CoV) "
                 "reveals that.\n")
    lines.append("**Why does replicate count affect confidence interval width?** The interval scales with "
                 "the standard error (SD / sqrt(n)) and with the t-value for df = n - 1. More replicates "
                 "shrink SE and lower the t-multiplier, so the interval narrows as n grows.\n")
    lines.append("**Which readings should be investigated before using the data for machine learning?** "
                 f"The {most_noisy['domain']} {most_noisy['condition']} group, driven by its outlying "
                 f"replicate, should be checked before training. High-CoV groups either need the outlier "
                 f"explained/removed or the measurement repeated.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_correlation_limitations(corr, path):
    biochem = corr[corr["relationship"] == "Biochem: signal vs concentration"].iloc[0]
    strongest = corr.loc[corr["pearson_correlation"].abs().idxmax()]
    weakest = corr.loc[corr["pearson_correlation"].abs().idxmin()]

    lines = []
    lines.append("# Correlation and Calibration Limitations\n")
    lines.append("## Correlation and fit summary\n")
    show = ["relationship", "n_samples", "pearson_correlation", "spearman_correlation",
            "slope", "intercept", "r_squared", "mean_absolute_error", "root_mean_squared_error"]
    lines.append("| " + " | ".join(show) + " |")
    lines.append("| " + " | ".join(["---"] * len(show)) + " |")
    for _, r in corr.iterrows():
        cells = [str(r["relationship"]), str(r["n_samples"])] + [_fmt(r[c]) for c in show[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Answers\n")
    lines.append("**Does signal increase or decrease with input value?** It depends on the domain. In "
                 f"Biochem, absorbance rises with concentration (positive slope, "
                 f"Pearson {_fmt(biochem['pearson_correlation'])}). In Electronics the measured voltage "
                 "falls as load resistance and temperature rise (negative slope). In Mechanical both "
                 "displacement and stress climb with load (positive slope).\n")
    lines.append(f"**Which domain shows the strongest signal-input relationship?** "
                 f"{strongest['relationship']} with |Pearson| = {_fmt(abs(strongest['pearson_correlation']))} "
                 f"and R-squared = {_fmt(strongest['r_squared'])}.\n")
    lines.append(f"**Which domain shows the weakest or noisiest relationship?** "
                 f"{weakest['relationship']} has the smallest absolute correlation "
                 f"({_fmt(weakest['pearson_correlation'])}); its calibration line explains the least "
                 f"variance.\n")
    lines.append("**Does high correlation prove causation?** No. Correlation only measures how two numbers "
                 "move together. A confounder can drive both, or the association can be coincidental. "
                 "Causation needs a controlled mechanism, not just a high r.\n")
    lines.append("**Can correlation be trusted with small sample size?** Not on its own. With only three "
                 "distinct input levels a single stray point can swing the coefficient a long way, and the "
                 "estimate has very wide uncertainty. Small-n correlations are suggestive, not conclusive.\n")
    lines.append("**Can correlation miss nonlinear relationships?** Yes. Pearson only captures the linear "
                 "component. A curved or saturating trend can have a modest Pearson value while still being "
                 "strongly related; Spearman (rank-based) catches monotonic-but-curved trends better.\n")
    lines.append("**How can outliers affect correlation?** A single outlier shifts the slope and intercept, "
                 "can inflate or deflate r, and raises MAE and RMSE. Least-squares fitting is especially "
                 "sensitive because it squares the residual of the far point.\n")
    lines.append("**How can temperature, load, material type, or experimental condition act as confounders?** "
                 "In Electronics, load and temperature rise together, so a voltage-vs-temperature correlation "
                 "may really be a voltage-vs-load effect in disguise. Material type or test condition can "
                 "similarly move the signal independently of the input you think you are studying.\n")
    lines.append("**Why should mixed-domain correlation be avoided?** The domains use different units, "
                 "different signal meanings, and different physical mechanisms. Pooling absorbance, voltage, "
                 "and displacement into one correlation produces a number with no physical interpretation.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_feature_dictionary(path):
    text = """# Feature Dictionary

Each engineered feature is defined below with its formula, applicable domain,
required columns, invalidity condition, and machine-learning rationale.

## rolling_average_signal
- **Formula:** rolling mean of `signal`, window size 3, within each (domain, condition) group, ordered by `time_step`.
- **Applies to:** all domains where row order / time_step is meaningful.
- **Required columns:** `signal`, `time_step`, `domain`, `condition`.
- **Invalid when:** rows are unordered, or the window would cross into an unrelated condition.
- **Why useful for ML:** smooths measurement noise so the model sees the underlying trend rather than single-reading jitter.

## normalized_signal
- **Formula:** `signal / baseline_signal`.
- **Applies to:** all domains with a valid baseline.
- **Required columns:** `signal`, `baseline_signal`.
- **Invalid when:** `baseline_signal` is missing or zero.
- **Why useful for ML:** puts readings from different instruments/scales on a common baseline-relative scale, making them comparable as features.

## power_w
- **Formula:** `voltage_v * current_a`.
- **Applies to:** Electronics only.
- **Required columns:** `voltage_v`, `current_a`.
- **Invalid when:** the row is Biochem or Mechanical (no voltage/current).
- **Why useful for ML:** electrical power is a physically meaningful derived quantity that often predicts heating and load behaviour better than voltage alone.

## error_percent
- **Formula:** `((signal - expected_signal) / expected_signal) * 100`.
- **Applies to:** all domains with a valid expected signal.
- **Required columns:** `signal`, `expected_signal`.
- **Invalid when:** `expected_signal` is missing or zero.
- **Why useful for ML:** expresses calibration accuracy directly, flagging rows that deviate from their expected value.

## stress_ratio
- **Formula:** `stress_mpa / reference_stress_mpa`.
- **Applies to:** Mechanical only.
- **Required columns:** `stress_mpa`, `reference_stress_mpa`.
- **Invalid when:** the row is Biochem or Electronics (no stress values).
- **Why useful for ML:** a dimensionless load-relative stress that indicates how close a sample is to its reference limit.

## stability_flag
- **Formula / rule:** derived from the group coefficient of variation — stable (CoV <= 0.05), moderate (0.05 < CoV <= 0.15), unstable (CoV > 0.15).
- **Applies to:** all replicate groups.
- **Required columns:** the replicate summary's `coefficient_of_variation`.
- **Invalid when:** CoV cannot be computed (fewer than two valid replicates).
- **Why useful for ML:** lets the pipeline down-weight or exclude noisy groups before training.

## ml_ready
- **Formula / rule:** boolean AND of valid signal, valid non-zero expected signal, valid input value, present domain/condition, valid normalized_signal, and a stability_flag of stable or moderate.
- **Applies to:** all domains.
- **Invalid when:** any required value is missing/zero or the group is unstable.
- **Why useful for ML:** provides a single gate for selecting only trustworthy rows for model training.
"""
    with open(path, "w") as f:
        f.write(text)


def write_feature_summary(features, path):
    not_ready = features[~features["ml_ready"]]
    lines = []
    lines.append("# Feature Summary\n")
    lines.append("## Chosen stability thresholds\n")
    lines.append("The suggested rule is used unchanged: stable when CoV <= 0.05, moderate when "
                 "0.05 < CoV <= 0.15, unstable when CoV > 0.15. These thresholds are conventional for "
                 "replicate measurement work — 5% relative spread is tight, and anything past 15% is "
                 "noisy enough to warrant investigation before training.\n")

    lines.append("## Answers\n")
    lines.append("**Which features are general across all domains?** `rolling_average_signal`, "
                 "`normalized_signal`, `error_percent`, `stability_flag`, and `ml_ready` — they only "
                 "need columns every domain has.\n")
    lines.append("**Which features are domain-specific?** `power_w` (Electronics only) and `stress_ratio` "
                 "(Mechanical only).\n")

    if len(not_ready):
        rows_desc = ", ".join(sorted(not_ready["record_id"].astype(str).tolist()))
        dom = not_ready["domain"].mode().iloc[0]
        cond = not_ready["condition"].mode().iloc[0]
        lines.append(f"**Which rows are not ML-ready and why?** Rows {rows_desc} "
                     f"(the {dom} {cond} group). Their replicate group is flagged unstable — the "
                     f"coefficient of variation exceeds the 0.15 threshold because of an outlying "
                     f"reading — so the ml_ready gate returns False.\n")
    else:
        lines.append("**Which rows are not ML-ready and why?** All rows passed the readiness gate.\n")

    lines.append("**Which engineered feature is most useful for Electronics?** `power_w` — it fuses "
                 "voltage and current into the physically meaningful quantity that tracks load and "
                 "heating.\n")
    lines.append("**Which engineered feature is most useful for Mechanical?** `stress_ratio` — a "
                 "dimensionless measure of how close each sample is to its reference stress limit.\n")
    lines.append("**Which engineered feature is most useful for Biochem?** `error_percent` (backed by "
                 "`normalized_signal`) — it captures how far absorbance drifts from the expected "
                 "calibration value, which is the core quality signal in an assay.\n")
    lines.append("**Why should invalid domain features be left blank instead of forcing a value?** A "
                 "forced value (e.g. zero power for a Biochem row) is a fabricated measurement. It would "
                 "bias statistics and teach the model a relationship that does not physically exist. Blank "
                 "correctly signals 'not applicable'.\n")
    lines.append("**How can feature engineering introduce misleading information?** By encoding target "
                 "information into a feature (leakage), by imputing fake values, by smoothing away real "
                 "events with rolling windows, or by normalizing against a wrong baseline — each makes the "
                 "training data look cleaner or more predictive than reality.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_readme(path):
    text = """# Task 9 — Calibration Statistics, Correlation Analysis, and Feature Engineering

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
"""
    with open(path, "w") as f:
        f.write(text)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "task_9/data/calibration_measurements.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "task_9/output"
    os.makedirs(out_dir, exist_ok=True)

    df = rs.load_data(csv_path)

    # Part 1: replicate statistics
    summary = rs.calculate_replicate_statistics(df)
    rs.save_replicate_summary(summary, os.path.join(out_dir, "replicate_summary.csv"))

    # Part 2: correlation + calibration
    corr = ca.calculate_correlations(df)
    corr.to_csv(os.path.join(out_dir, "correlation_summary.csv"), index=False)

    calib = summary[["domain", "condition", "input_value", "input_unit", "mean_signal",
                     "confidence_interval_lower", "confidence_interval_upper",
                     "standard_deviation_signal", "coefficient_of_variation"]].copy()
    calib.to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)

    for domain, fname in [("Biochem", "calibration_curve_biochem.png"),
                          ("Electronics", "calibration_curve_electronics.png"),
                          ("Mechanical", "calibration_curve_mechanical.png")]:
        ca.plot_calibration_curve(summary, domain, os.path.join(out_dir, fname))
    ca.plot_signal_input_scatter(df, os.path.join(out_dir, "correlation_signal_input.png"))

    # Part 3: feature engineering
    feats = fe.add_rolling_average(df)
    feats = fe.add_normalized_signal(feats)
    feats = fe.add_power_feature(feats)
    feats = fe.add_error_percent(feats)
    feats = fe.add_stress_ratio(feats)
    feats = fe.add_stability_from_summary(feats, summary)
    feats = fe.add_ml_readiness_flag(feats)
    feats = feats.sort_values("record_id").reset_index(drop=True)
    fe.save_engineered_features(feats, os.path.join(out_dir, "engineered_features.csv"))

    ml_ready = feats[feats["ml_ready"]].copy()
    ml_ready.to_csv(os.path.join(out_dir, "ml_ready_dataset.csv"), index=False)

    # Markdown interpretation files
    write_replicate_analysis(summary, os.path.join(out_dir, "replicate_analysis.md"))
    write_correlation_limitations(corr, os.path.join(out_dir, "correlation_limitations.md"))
    write_feature_dictionary(os.path.join(out_dir, "feature_dictionary.md"))
    write_feature_summary(feats, os.path.join(out_dir, "feature_summary.md"))
    write_readme(os.path.join(os.path.dirname(out_dir), "README.md"))

    print("Pipeline complete.")
    print(f"Rows: {len(df)} | replicate groups: {len(summary)} | ml_ready rows: {len(ml_ready)}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
