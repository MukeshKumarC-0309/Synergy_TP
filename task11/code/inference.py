"""
Task 11 - Inference script.

Loads the saved pipeline (outputs/final_pipeline.joblib) independently of the
training code and returns predicted class + confidence for new samples.

Usage:
    python inference.py --N 90 --P 42 --K 43 --temperature 20.8 --humidity 82.0 --ph 6.5 --rainfall 202.9
    python inference.py --csv new_samples.csv        # batch mode, same 7 columns, no label needed
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "model" / "final_pipeline.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_pipeline():
    if not PIPELINE_PATH.exists():
        sys.exit(f"No saved pipeline at {PIPELINE_PATH}. Run train.py first.")
    return joblib.load(PIPELINE_PATH)


def predict(pipe, X: pd.DataFrame) -> pd.DataFrame:
    preds = pipe.predict(X)
    out = pd.DataFrame({"predicted_label": preds})
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        classes = pipe.named_steps["clf"].classes_
        confidence = proba.max(axis=1)
        out["confidence"] = np.round(confidence, 4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Crop classification inference")
    parser.add_argument("--csv", type=str, help="CSV file with columns: " + ", ".join(FEATURES))
    for feat in FEATURES:
        parser.add_argument(f"--{feat}", type=float)
    args = parser.parse_args()

    pipe = load_pipeline()

    if args.csv:
        X = pd.read_csv(args.csv)[FEATURES]
        result = predict(pipe, X)
        result_full = pd.concat([X.reset_index(drop=True), result], axis=1)
        print(result_full.to_string(index=False))
        result_full.to_csv("inference_output.csv", index=False)
        print("\nSaved -> inference_output.csv")
    else:
        values = [getattr(args, f) for f in FEATURES]
        if any(v is None for v in values):
            sys.exit("Provide either --csv or all of: " + ", ".join(f"--{f}" for f in FEATURES))
        X = pd.DataFrame([values], columns=FEATURES)
        result = predict(pipe, X)
        row = result.iloc[0]
        conf = f"{row['confidence']:.4f}" if "confidence" in row else "n/a"
        print(f"Predicted crop: {row['predicted_label']}  (confidence: {conf})")


if __name__ == "__main__":
    main()
