const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle,
  PageBreak,
} = require("docx");

const PLOTS = "../plots";
const METRICS = "../metrics";
const REPORT_DIR = "../report";

const testSummary = JSON.parse(fs.readFileSync(`${METRICS}/final_model_test_summary.json`));

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}

function simpleTable(headerRow, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const mkCell = (text, isHeader) => new TableCell({
    width: { size: colWidths[0], type: WidthType.DXA },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: !!isHeader, size: 18 })] })],
  });
  const buildRow = (cells, isHeader) => new TableRow({
    children: cells.map((c, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text: String(c), bold: !!isHeader, size: 18 })] })],
    })),
  });
  return new Table({
    columnWidths: colWidths,
    width: { size: total, type: WidthType.DXA },
    rows: [buildRow(headerRow, true), ...rows.map((r) => buildRow(r, false))],
  });
}

function image(path, width, height) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new ImageRun({ data: fs.readFileSync(path), transformation: { width, height }, type: "png" })],
  });
}

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "Task 11 Report", bold: true, size: 36 })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Practical Classification Using Machine-Learning Libraries — Crop Recommendation Dataset", size: 24, italics: true })],
      }),

      h1("1. Introduction and Problem Statement"),
      p("This report covers Task 11 of the Synergy AI/ML taskphase: building, comparing, evaluating, saving, and reusing a classification pipeline using scikit-learn. The dataset is a crop recommendation table where each row is a soil/climate reading and the target is the crop best suited to those conditions. The objective is multiclass classification across 22 crop labels, not a binary decision, so the emphasis throughout is on per-class behaviour rather than a single accuracy number."),

      h1("2. Dataset Description, Target Classes, and Class Distribution"),
      p("Each row represents one soil/climate sample. The seven input features are nitrogen (N), phosphorus (P), and potassium (K) soil content, temperature (°C), relative humidity (%), soil pH, and rainfall (mm). The target column is the recommended crop, with 22 distinct classes (e.g. rice, maize, chickpea, banana, coffee)."),
      p(`The dataset has ${2200} rows and no missing values in any column. Every class has exactly 100 samples — the classes are perfectly balanced. This matters for two reasons: a majority-class baseline is close to a random 1-in-22 guess rather than a meaningful floor, and macro-averaged metrics are not distorted by class-size weighting, so macro and weighted averages track closely throughout this report.`),
      image(`${PLOTS}/class_distribution.png`, 500, 300),
      p("Because classes are balanced and the features are all continuous soil/climate measurements with clearly different physical ranges (e.g. rainfall in the hundreds of mm vs. pH in single digits), the main preprocessing need is feature scaling rather than imputation or encoding.", { italics: true }),

      h1("3. Data Preparation and Experimental Approach"),
      p("Data was split 60/20/20 into train, validation, and test sets, stratified on the label so every crop is represented proportionally in each split. All five models were trained on the same train split and compared on the same validation split, keeping the comparison fair. The test split was held out entirely until one final model was chosen, and was touched exactly once."),
      bullet("Preprocessing: StandardScaler fit on training data only, applied inside a scikit-learn Pipeline so the same transform is reproduced automatically at inference time."),
      bullet("Random state fixed at 42 across splits and models for reproducibility."),
      bullet("Model selection metric: validation macro-F1, since all classes matter equally and there is no class-imbalance reason to weight by support."),

      h1("4. Classification Models Used"),
      bullet("Majority-class baseline (DummyClassifier) — establishes the floor any real model must clear."),
      bullet("Logistic Regression — linear decision boundaries, fast, interpretable coefficients."),
      bullet("K-Nearest Neighbours (k=5) — distance-based, sensitive to feature scaling, no explicit training phase."),
      bullet("Decision Tree (max_depth=8) — single interpretable tree, depth-capped to limit overfitting."),
      bullet("Random Forest (200 trees) — bagged ensemble of trees, generally more robust than a single tree."),

      h1("5. Evaluation Metrics and Model-Comparison Table"),
      p("All metrics below are computed on the validation set (models never saw this data during training)."),
      simpleTable(
        ["Model", "Train Acc.", "Val Acc.", "Val Precision (macro)", "Val Recall (macro)", "Val F1 (macro)", "Train–Val Gap"],
        [
          ["Random Forest", "1.000", "0.993", "0.994", "0.993", "0.993", "0.007"],
          ["Logistic Regression", "0.976", "0.973", "0.974", "0.973", "0.973", "0.003"],
          ["KNN (k=5)", "0.979", "0.959", "0.963", "0.959", "0.959", "0.020"],
          ["Decision Tree", "0.849", "0.839", "0.796", "0.839", "0.804", "0.011"],
          ["Baseline (majority)", "0.045", "0.045", "0.002", "0.045", "0.004", "0.000"],
        ],
        [2050, 950, 900, 1550, 1350, 1200, 1150]
      ),
      p(""),
      p("Every trained model clears the baseline by a wide margin — expected, given 22 balanced classes make the majority-class floor close to random chance (~4.5%). Random Forest and Logistic Regression are the strongest performers and are close to each other; KNN trails slightly; the depth-capped Decision Tree is the weakest of the non-baseline models, consistent with a single shallow tree struggling to separate 22 classes on continuous features that Random Forest's ensemble handles better."),

      h1("6. Confusion Matrix and Error Analysis"),
      p("Random Forest was selected as the final model (highest validation macro-F1, and a small train–val gap relative to its near-perfect training accuracy, suggesting the ensemble is generalising rather than purely memorising). It was refit on train+validation combined and evaluated once on the held-out test set:"),
      simpleTable(
        ["Metric", "Test value"],
        [
          ["Accuracy", testSummary.test_accuracy.toFixed(4)],
          ["Precision (macro)", testSummary.test_precision_macro.toFixed(4)],
          ["Recall (macro)", testSummary.test_recall_macro.toFixed(4)],
          ["F1 (macro)", testSummary.test_f1_macro.toFixed(4)],
        ],
        [3000, 3000]
      ),
      p(""),
      image(`${PLOTS}/confusion_matrix_final_TEST.png`, 420, 420),
      p("Only 3 of 440 test samples were misclassified. All three errors are between crops with genuinely overlapping soil/climate profiles: blackgram→maize, lentil→mothbeans, and rice→jute. These are agronomically reasonable confusions — blackgram and maize both tolerate a similar nitrogen/rainfall band, and rice/jute both need high rainfall and humidity — rather than errors spread randomly across unrelated crops, which is a better sign than the same error count spread arbitrarily would be."),
      p("Domain cost of mistakes: in this dataset, a wrong recommendation isn't a labelling nuisance — it means a farmer plants a crop poorly matched to the actual soil/climate conditions, risking yield loss. Since all classes are equally weighted here (no crop is inherently 'safer' to get wrong than another in this dataset), macro-averaged metrics that treat every class equally are the right choice over relying on overall accuracy alone."),

      h1("7. Probability / Threshold Analysis"),
      p("Part 4 of the brief describes threshold analysis for binary classification; this dataset is 22-class, so a literal binary threshold doesn't directly apply. As the closest faithful adaptation, the single most-confused pair from the test-set error analysis (blackgram vs. maize) was isolated and treated as a one-vs-rest binary problem using the final model's predicted probability for the blackgram class, swept across thresholds:"),
      image(`${PLOTS}/threshold_sweep_blackgram_vs_maize.png`, 380, 271),
      p("Precision stays at 1.0 across every threshold tested — the model never mistakes a maize sample for blackgram in this subset. Recall degrades as the threshold rises past 0.3, meaning some true blackgram samples fall below the probability cutoff and would be missed at stricter thresholds. In a deployment context, this argues for keeping the decision threshold low-to-default (i.e. just take the argmax class) rather than requiring high confidence, since the cost here is a missed correct recommendation, not a false one."),

      h1("8. Final Model Selection and Justification"),
      p("Random Forest was selected because it had the best validation macro-F1 (0.993) among non-baseline models, the smallest per-class variance in the confusion matrix, and only a 0.007 train–val accuracy gap despite fitting training data to 1.0 — indicating the perfect training score reflects the ensemble's capacity rather than memorisation that fails to generalise. Logistic Regression was the closest competitor and would be a reasonable simpler alternative if model interpretability or inference latency mattered more than the last percentage point of accuracy."),

      h1("9. Model Saving and Inference Workflow"),
      p("The selected pipeline (StandardScaler + RandomForestClassifier, refit on train+validation) is serialized with joblib to outputs/final_pipeline.joblib. A separate script, inference.py, loads this file independently of the training code, accepts either a single sample via command-line flags or a batch CSV, and returns the predicted crop plus the model's confidence (max class probability). Reloading the saved pipeline and re-predicting on the same input reproduces an identical result, confirming the saved artifact is self-contained and deterministic."),

      h1("10. Limitations and Possible Improvements"),
      bullet("The dataset is synthetic-feeling in its cleanliness — no missing values, no outliers, perfectly balanced classes — which is unusual for real agricultural data and likely explains why every model scores so high. Results on messier field data would probably be lower."),
      bullet("Only 7 features are available; real crop recommendation would benefit from soil type, elevation, prior crop history, and regional data not present here."),
      bullet("KNN and the Decision Tree were not hyperparameter-tuned beyond one reasonable default configuration each (k=5, max_depth=8); a grid search might close some of the gap to Random Forest."),
      bullet("The threshold analysis in Section 7 is a one-vs-rest adaptation of a binary technique and only characterises one class pair — it doesn't generalise to a global multiclass confidence policy."),

      h1("11. Conclusion"),
      p("All four trained models substantially outperform the majority-class baseline, with Random Forest and Logistic Regression clearly ahead of KNN and the Decision Tree. The final Random Forest pipeline reaches 99.3% test accuracy and macro-F1, with the few remaining errors concentrated in agronomically similar crop pairs rather than distributed randomly. The saved pipeline is reusable end-to-end through a standalone inference script, satisfying the model-reuse requirement of the task."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(REPORT_DIR, { recursive: true });
  fs.writeFileSync(`${REPORT_DIR}/Task11_Report.docx`, buf);
  console.log(`Report written to ${REPORT_DIR}/Task11_Report.docx`);
});
