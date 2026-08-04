"""
inference.py

Standalone script that loads the saved regression pipeline and predicts
Temperature for new records. Completely separate from train.py -- this is
what a downstream application would actually run, without needing sklearn
training code, the original training data, or any knowledge of how the
model was built.

Usage (single record):
    python src/inference.py --model output/final_model_pipeline.joblib \\
        --values 0.51 0.49 0.50 0.48 0.52

Usage (CSV of new records, must have columns Sensor1..Sensor5):
    python src/inference.py --model output/final_model_pipeline.joblib \\
        --csv new_records.csv --out output/new_predictions.csv
"""

import argparse

import joblib
import pandas as pd

FEATURES = ["Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5"]


def predict_single(pipeline, values):
    row = pd.DataFrame([values], columns=FEATURES)
    return float(pipeline.predict(row)[0])


def predict_csv(pipeline, csv_path, out_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    preds = pipeline.predict(df[FEATURES])
    df["predicted_Temperature"] = preds
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict Temperature from sensor readings.")
    parser.add_argument("--model", required=True, help="Path to the saved .joblib pipeline")
    parser.add_argument("--values", nargs=5, type=float, metavar=("S1", "S2", "S3", "S4", "S5"),
                         help="Five sensor values for a single prediction")
    parser.add_argument("--csv", help="Path to a CSV of new records (columns Sensor1..Sensor5)")
    parser.add_argument("--out", default="output/new_predictions.csv",
                         help="Where to write predictions when using --csv")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)

    if args.values:
        pred = predict_single(pipeline, args.values)
        print(f"Predicted Temperature: {pred:.4f}")
    elif args.csv:
        df = predict_csv(pipeline, args.csv, args.out)
        print(f"Predicted {len(df)} rows. Written to: {args.out}")
        print(df[FEATURES + ['predicted_Temperature']].head())
    else:
        parser.error("Provide either --values (single record) or --csv (batch of records).")


if __name__ == "__main__":
    main()
