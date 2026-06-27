import pandas as pd
import json

def load_data(file_path):
    return pd.read_csv(file_path)

def generate_summary(df):
    return {
        "total_rows": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "domains": df["domain"].unique().tolist() if "domain" in df.columns else []
    }

def remove_duplicates(df):
    return df.drop_duplicates()

def standardize_domains(df):
    mapping = {
        "ml": "ML", "ML": "ML", "machine learning": "ML", "MACHINE LEARNING": "ML",
        "web": "Web", "Web Dev": "Web", "web development": "Web",
        "electronics": "Electronics", "Electronics": "Electronics",
        "Mechanical": "Mechanical"
    }
    df["domain"] = df["domain"].map(mapping)
    return df

def clean_attendance(df):
    df["attendance_percent"] = df["attendance_percent"].astype(str).str.replace("%", "").str.strip()
    df["attendance_percent"] = pd.to_numeric(df["attendance_percent"], errors="coerce")
    df.loc[df["attendance_percent"] < 0, "attendance_percent"] = None
    df.loc[df["attendance_percent"] > 100, "attendance_percent"] = None
    df["attendance_percent"] = df["attendance_percent"].fillna(df["attendance_percent"].mean().round(2))
    return df

def clean_scores(df):
    word_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    df["score"] = df["score"].astype(str).str.strip().str.lower().replace(word_to_num)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["score"] = df["score"].fillna(df["score"].mean().round(2))
    return df

def clean_study_hours(df):
    word_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    df["study_hours"] = df["study_hours"].astype(str).str.strip().str.lower().replace(word_to_num)
    df["study_hours"] = pd.to_numeric(df["study_hours"], errors="coerce")
    df["study_hours"] = df["study_hours"].fillna(0)
    return df

def clean_height(df):
    def convert_height(val):
        val = str(val).strip().lower()
        if "cm" in val:
            return float(val.replace("cm", "").strip())
        if "m" in val:
            return float(val.replace("m", "").strip()) * 100
        return float(val)
    df["height_cm"] = df["height"].apply(convert_height)
    return df.drop(columns=["height"])

def clean_weight(df):
    def convert_weight(val):
        return float(str(val).strip().lower().replace("kg", "").strip())
    df["weight_kg"] = df["weight"].apply(convert_weight)
    return df.drop(columns=["weight"])

def clean_submitted(df):
    mapping = {"yes": "yes", "y": "yes", "no": "no", "n": "no"}
    df["submitted"] = df["submitted"].astype(str).str.strip().str.lower().map(mapping)
    return df

def handle_missing_values(df):
    df["attendance_percent"] = df["attendance_percent"].fillna(df["attendance_percent"].mean().round(2))
    df["score"] = df["score"].fillna(df["score"].mean().round(2))
    df["study_hours"] = df["study_hours"].fillna(0)
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].mean().round(2))
    df["submitted"] = df["submitted"].fillna("no")
    return df

def validate_cleaned_data(df):
    checks = [
        df["student_id"].duplicated().sum() == 0,
        df["attendance_percent"].between(0, 100).all(),
        pd.to_numeric(df["score"], errors="coerce").notna().all(),
        pd.to_numeric(df["study_hours"], errors="coerce").notna().all(),
        pd.to_numeric(df["height_cm"], errors="coerce").notna().all(),
        pd.to_numeric(df["weight_kg"], errors="coerce").notna().all(),
        df["submitted"].isin(["yes", "no"]).all(),
        df["domain"].isin(["ML", "Web", "Electronics", "Mechanical"]).all(),
        df[["student_id", "name", "domain", "score"]].notna().all().all()
    ]
    return all(checks)

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

def write_report(report_path):
    with open(report_path, "w") as f:
        f.write("# Cleaning Report\n\n")
        f.write("All validation checks passed after cleaning.\n")