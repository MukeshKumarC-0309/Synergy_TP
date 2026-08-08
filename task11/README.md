# Task 11 — Crop Classification Pipeline

Classifies crop recommendation based on soil/climate readings (N, P, K,
temperature, humidity, ph, rainfall) into one of 22 crop labels.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## Folder layout

```
task11/
├── data.csv                  # Crop_recommendation.csv (2200 rows, 22 balanced classes)
├── README.md
├── code/
│   ├── train.py               # full pipeline: EDA -> split -> train -> evaluate -> save
│   ├── inference.py           # loads model/final_pipeline.joblib, predicts on new data
│   └── build_report.js        # regenerates report/Task11_Report.docx from metrics/plots
├── report/
│   └── Task11_Report.docx     # the required technical report (11 sections)
├── model/
│   ├── final_pipeline.joblib  # saved scaler + classifier pipeline
│   └── final_model_name.txt
├── metrics/
│   ├── model_comparison_val.csv               # all 5 models, validation metrics
│   ├── final_model_test_summary.json          # final model's one-shot test metrics
│   └── final_model_test_classification_report.csv  # per-class precision/recall/F1
├── error_analysis/
│   ├── misclassified_test_samples.csv
│   ├── most_confused_pairs.csv
│   ├── threshold_analysis_pair.json
│   └── threshold_sweep_<A>_vs_<B>.csv
├── eda/
│   ├── dataset_summary.json
│   └── class_feature_means.csv
└── plots/
    ├── class_distribution.png
    ├── confusion_matrix_<model>.png            # one per model, on validation set
    ├── confusion_matrix_final_TEST.png
    └── threshold_sweep_<A>_vs_<B>.png
```

## Run

```bash
cd code
python train.py          # trains all 5 models, writes to eda/, metrics/, error_analysis/, model/, plots/
node build_report.js     # rebuilds report/Task11_Report.docx from the latest metrics/plots (needs `docx` npm package)
```

Then run inference independently of training:

```bash
# single sample
python inference.py --N 90 --P 42 --K 43 --temperature 20.88 --humidity 82.0 --ph 6.5 --rainfall 202.9

# batch (CSV with the 7 feature columns, no label column needed)
python inference.py --csv new_samples.csv
```

## Notes

- Split is 60/20/20 (train/val/test), stratified on the label.
- Model selection uses validation macro-F1 (the baseline is excluded from
  selection since it exists only as a floor).
- The final model is refit on train+val before the one-shot test evaluation.
- Threshold/probability analysis (Part 4) is written for binary problems;
  this dataset is 22-class, so the sweep is adapted to a one-vs-rest binary
  comparison between the two most-confused classes found in test-set error
  analysis, using the final model's predicted probabilities.
