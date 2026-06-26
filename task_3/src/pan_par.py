import pandas as pd
import json


def calculate_summary_pandas(df):
    df["submitted"] = df["submitted"].str.strip().str.lower()
    submitted = df[df["submitted"] == "yes"]
    not_submitted = df[df["submitted"] == "no"]

    highest = df.loc[df["score"].idxmax()]
    lowest = submitted.loc[submitted["score"].idxmin()]
    domain_avg = df.groupby("domain")["score"].mean().round(2).to_dict()

    return {
        "total_students": len(df),
        "submitted_count": len(submitted),
        "missing_submissions": len(not_submitted),
        "average_score": round(df["score"].mean(), 2),
        "highest_scorer": {"name": highest["name"], "score": int(highest["score"])},
        "lowest_scorer_among_submitted": {"name": lowest["name"], "score": int(lowest["score"])},
        "domain_average_score": domain_avg,
        "students_not_submitted": not_submitted["name"].tolist(),
        "students_below_5": df[df["score"] < 5]["name"].tolist()
    }

def write_json_pandas(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)