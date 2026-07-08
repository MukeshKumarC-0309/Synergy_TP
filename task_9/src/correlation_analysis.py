import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RELATIONSHIPS = [
    ("Biochem: signal vs concentration", "Biochem", "input_value", "signal"),
    ("Electronics: signal vs load", "Electronics", "input_value", "signal"),
    ("Electronics: signal vs temperature", "Electronics", "temperature_c", "signal"),
    ("Mechanical: signal vs load", "Mechanical", "input_value", "signal"),
    ("Mechanical: stress_mpa vs load", "Mechanical", "input_value", "stress_mpa"),
]


def _fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    mae = np.mean(np.abs(y - pred))
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    return slope, intercept, r2, mae, rmse


def fit_calibration_line(df):
    x = df["input_value"].to_numpy(float)
    y = df["signal"].to_numpy(float)
    slope, intercept, *_ = _fit(x, y)
    return slope, intercept


def calculate_fit_metrics(df):
    x = df["input_value"].to_numpy(float)
    y = df["signal"].to_numpy(float)
    slope, intercept, r2, mae, rmse = _fit(x, y)
    return {"slope": slope, "intercept": intercept, "r_squared": r2, "mae": mae, "rmse": rmse}


def calculate_correlations(df):
    rows = []
    for label, domain, xcol, ycol in RELATIONSHIPS:
        sub = df[df["domain"] == domain][[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
        x = sub[xcol].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        n = len(x)
        pearson = stats.pearsonr(x, y)[0] if n >= 2 else np.nan
        spearman = stats.spearmanr(x, y)[0] if n >= 2 else np.nan
        slope, intercept, r2, mae, rmse = _fit(x, y)
        rows.append({
            "relationship": label,
            "domain": domain,
            "x_variable": xcol,
            "y_variable": ycol,
            "n_samples": n,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r2,
            "mean_absolute_error": mae,
            "root_mean_squared_error": rmse,
        })
    return pd.DataFrame(rows)


def plot_calibration_curve(summary_df, domain, output_path):
    d = summary_df[summary_df["domain"] == domain].copy()
    d = d.sort_values("input_value")
    x = d["input_value"].to_numpy(float)
    y = d["mean_signal"].to_numpy(float)
    lower = d["confidence_interval_lower"].to_numpy(float)
    upper = d["confidence_interval_upper"].to_numpy(float)
    yerr = np.vstack([y - lower, upper - y])
    yerr = np.nan_to_num(yerr, nan=0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=5, color="#2b6cb0",
                ecolor="#e53e3e", markerfacecolor="#2b6cb0", label="mean signal ± 95% CI")
    ax.set_xlabel("input_value")
    ax.set_ylabel("mean_signal")
    ax.set_title(f"Calibration curve — {domain}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_signal_input_scatter(df, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Biochem": "#2b6cb0", "Electronics": "#dd6b20", "Mechanical": "#38a169"}
    for domain, g in df.groupby("domain"):
        ax.scatter(g["input_value"], g["signal"], label=domain,
                   color=colors.get(domain, "gray"), alpha=0.8, s=45)
    ax.set_xlabel("input_value")
    ax.set_ylabel("signal (raw)")
    ax.set_title("Raw signal vs input value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
