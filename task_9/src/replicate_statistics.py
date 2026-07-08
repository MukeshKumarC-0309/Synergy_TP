import numpy as np
import pandas as pd
from scipy import stats

GROUP_KEYS = ["domain", "condition", "input_type", "input_value", "input_unit", "signal_unit"]

STABLE_MAX = 0.05
MODERATE_MAX = 0.085


def load_data(file_path):
    df = pd.read_csv(file_path)
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    return df


def calculate_confidence_interval(mean, std, n):
    if n is None or n < 2 or std is None or np.isnan(std):
        return (np.nan, np.nan)
    se = std / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return (mean - t * se, mean + t * se)


def assign_stability_flag(coefficient_of_variation):
    if coefficient_of_variation is None or np.isnan(coefficient_of_variation):
        return "unreliable"
    if coefficient_of_variation <= STABLE_MAX:
        return "stable"
    if coefficient_of_variation <= MODERATE_MAX:
        return "moderate"
    return "unstable"


def calculate_replicate_statistics(df):
    rows = []
    for keys, g in df.groupby(GROUP_KEYS, sort=False):
        s = g["signal"].dropna()
        n = len(s)
        mean = s.mean()
        median = s.median()
        if n >= 2:
            var = s.var(ddof=1)
            std = s.std(ddof=1)
            se = std / np.sqrt(n)
            ci_low, ci_high = calculate_confidence_interval(mean, std, n)
            cov = std / mean if mean != 0 else np.nan
            reliable = True
        else:
            var = std = se = ci_low = ci_high = cov = np.nan
            reliable = False

        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "replicate_count": n,
            "mean_signal": mean,
            "median_signal": median,
            "variance_signal": var,
            "standard_deviation_signal": std,
            "standard_error_signal": se,
            "confidence_interval_lower": ci_low,
            "confidence_interval_upper": ci_high,
            "coefficient_of_variation": cov,
            "minimum_signal": s.min() if n else np.nan,
            "maximum_signal": s.max() if n else np.nan,
            "reliable": reliable,
            "stability_flag": assign_stability_flag(cov),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def save_replicate_summary(summary_df, output_path):
    summary_df.to_csv(output_path, index=False)
