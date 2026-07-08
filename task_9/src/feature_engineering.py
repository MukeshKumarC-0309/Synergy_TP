import numpy as np
import pandas as pd

GROUP_KEYS = ["domain", "condition", "input_type", "input_value", "input_unit", "signal_unit"]


def add_rolling_average(df):
    df = df.sort_values(["domain", "condition", "time_step"]).copy()
    df["rolling_average_signal"] = (
        df.groupby(["domain", "condition"])["signal"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    return df


def add_normalized_signal(df):
    b = pd.to_numeric(df["baseline_signal"], errors="coerce")
    df["normalized_signal"] = np.where((b.notna()) & (b != 0), df["signal"] / b, np.nan)
    return df


def add_power_feature(df):
    v = pd.to_numeric(df["voltage_v"], errors="coerce")
    i = pd.to_numeric(df["current_a"], errors="coerce")
    df["power_w"] = np.where(df["domain"] == "Electronics", v * i, np.nan)
    return df


def add_error_percent(df):
    e = pd.to_numeric(df["expected_signal"], errors="coerce")
    df["error_percent"] = np.where((e.notna()) & (e != 0),
                                   (df["signal"] - e) / e * 100, np.nan)
    return df


def add_stress_ratio(df):
    s = pd.to_numeric(df["stress_mpa"], errors="coerce")
    r = pd.to_numeric(df["reference_stress_mpa"], errors="coerce")
    df["stress_ratio"] = np.where((df["domain"] == "Mechanical") & (r.notna()) & (r != 0),
                                  s / r, np.nan)
    return df


def add_stability_from_summary(df, summary_df):
    cols = GROUP_KEYS + ["coefficient_of_variation", "stability_flag"]
    return df.merge(summary_df[cols], on=GROUP_KEYS, how="left")


def add_ml_readiness_flag(df):
    signal_ok = pd.to_numeric(df["signal"], errors="coerce").notna()
    expected_ok = pd.to_numeric(df["expected_signal"], errors="coerce").replace(0, np.nan).notna()
    input_ok = pd.to_numeric(df["input_value"], errors="coerce").notna()
    meta_ok = df["domain"].notna() & df["condition"].notna()
    norm_ok = df["normalized_signal"].notna()
    stable_ok = df["stability_flag"].isin(["stable", "moderate"])
    df["ml_ready"] = signal_ok & expected_ok & input_ok & meta_ok & norm_ok & stable_ok
    return df


def save_engineered_features(df, output_path):
    df.to_csv(output_path, index=False)
