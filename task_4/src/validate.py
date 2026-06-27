import pandas as pd

def run_validation(df):
    print("Running validation checks...")
    print("No duplicate student IDs:", df["student_id"].duplicated().sum() == 0)
    print("Attendance between 0-100:", df["attendance_percent"].between(0, 100).all())
    print("Score is numeric:", pd.to_numeric(df["score"], errors="coerce").notna().all())
    print("Study hours is numeric:", pd.to_numeric(df["study_hours"], errors="coerce").notna().all())
    print("Height is numeric:", pd.to_numeric(df["height_cm"], errors="coerce").notna().all())
    print("Weight is numeric:", pd.to_numeric(df["weight_kg"], errors="coerce").notna().all())
    print("Submitted values consistent:", df["submitted"].isin(["yes", "no"]).all())
    print("Domain values valid:", df["domain"].isin(["ML", "Web", "Electronics", "Mechanical"]).all())