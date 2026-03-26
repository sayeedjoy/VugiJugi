# Thesis ML Pipeline — Notebook Plan

> **Dataset:** `/content/drive/MyDrive/Thesis/dataset/archive.zip`  
> **Convention:** All notebooks read from and write to `/content/drive/MyDrive/Thesis/`  
> **Rule:** SMOTE is applied only on training data. Test set is never touched during preprocessing.

---

## Overview

The project is split into **7 focused notebooks**, each handling one phase of the pipeline. This keeps each notebook clean, reproducible, and independently restartable.

| Notebook | Name | Key Output |
|----------|------|------------|
| NB-00 | Setup & Data Ingestion | `raw.parquet` |
| NB-01 | Cleaning & Feature Engineering | `clean.parquet` |
| NB-02 | Preprocessing Pipeline | `X_train`, `X_test`, `y_train`, `y_test`, `scaler.pkl` |
| NB-03 | Baseline Model Training | `models/baseline/*.pkl`, `baseline_results.csv` |
| NB-04 | Hyperparameter Tuning | `models/tuned/*.pkl`, `tuned_results.csv` |
| NB-05 | Model Selection & Ensemble Voting | `voting_hard.pkl`, `voting_soft.pkl` |
| NB-06 | Final Evaluation & Reporting | Plots, classification reports, final figures |

---

## NB-00 · Setup & Data Ingestion

**Goal:** Mount Drive, extract the dataset, load the raw CSV, run a quick sanity check, and persist the raw data for all downstream notebooks.

### Tasks
- Mount Google Drive
- Extract `archive.zip` from `/content/drive/MyDrive/Thesis/dataset/`
- Load raw CSV into a DataFrame
- Sanity checks: shape, dtypes, null counts, class distribution
- Save raw DataFrame as `raw.parquet` to Drive

### Outputs
```
/content/drive/MyDrive/Thesis/data/raw.parquet
```

### Notes
- Do not modify any values at this stage — this notebook is read-only after first run
- Print class distribution clearly; imbalance will be handled in NB-02

---

## NB-01 · Data Cleaning & Feature Engineering

**Goal:** Remove low-quality columns, fill missing values, and create domain-specific features.

### Tasks
- Load `raw.parquet`
- Drop columns where missingness exceeds threshold (e.g. > 40%)
- Fill remaining numeric nulls with column medians
- Engineer new features:
  - **Shock Index** = Heart Rate / Systolic BP
  - **Pulse Pressure** = Systolic BP − Diastolic BP
- Validate new features (check for nulls, inf, unexpected ranges)
- Save clean DataFrame

### Outputs
```
/content/drive/MyDrive/Thesis/data/clean.parquet
```

### Notes
- Document the missingness threshold choice and which columns were dropped
- Log before/after column counts
- Consider saving a `dropped_columns.txt` for reference

---

## NB-02 · Preprocessing Pipeline

**Goal:** Prepare train/test arrays that are ready for model input — no leakage.

### Tasks
- Load `clean.parquet`
- Separate features (`X`) and target (`y`)
- Stratified 80/20 train-test split (`random_state=42`)
- Fit `StandardScaler` **on training set only**, transform both sets
- Apply **SMOTE on training set only** (never on test set)
- Save all arrays and the fitted scaler

### Outputs
```
/content/drive/MyDrive/Thesis/data/X_train.npy
/content/drive/MyDrive/Thesis/data/X_test.npy
/content/drive/MyDrive/Thesis/data/y_train.npy
/content/drive/MyDrive/Thesis/data/y_test.npy
/content/drive/MyDrive/Thesis/models/scaler.pkl
```

### Notes
- Print class distribution before and after SMOTE to confirm balancing
- Scaler must be saved here and reused in NB-05 and NB-06 — never refit it
- SMOTE is from `imblearn.over_sampling`; use `random_state=42`

---

## NB-03 · Baseline Model Training & Evaluation

**Goal:** Train all 13 models with default hyperparameters and record performance for comparison.

### Models
| # | Model |
|---|-------|
| 1 | MLP (Multi-Layer Perceptron) |
| 2 | KNN (K-Nearest Neighbors) |
| 3 | Decision Tree |
| 4 | AdaBoost |
| 5 | HistGradientBoosting |
| 6 | Random Forest |
| 7 | Bagging Classifier |
| 8 | Gradient Boosting |
| 9 | LightGBM |
| 10 | XGBoost |
| 11 | CatBoost |
| 12 | Extra Trees |
| 13 | SVM |

### Tasks
- Load preprocessed arrays from NB-02
- Train all 13 models with default params and `random_state=42` where applicable
- Evaluate each on the test set:
  - Accuracy
  - Precision, Recall, F1-score (macro & weighted)
  - ROC-AUC
  - Training time
- Save all trained models with `joblib`
- Save results to a summary CSV

### Outputs
```
/content/drive/MyDrive/Thesis/models/baseline/*.pkl
/content/drive/MyDrive/Thesis/results/baseline_results.csv
```

### Notes
- Use a loop over a model dict to keep the notebook clean
- Display results as a sorted DataFrame ranked by F1 or ROC-AUC
- Cross-validation (5-fold) is optional here but recommended for robustness

---

## NB-04 · Hyperparameter Tuning

**Goal:** Tune all 13 baseline models and record the improvement.

### Tasks
- Load preprocessed arrays and baseline results
- Define hyperparameter search spaces for each model
- Run tuning (GridSearchCV or Optuna) for each model
- Save best estimators and their parameters
- Produce a before/after comparison table (baseline vs tuned)

### Tuning Strategy
- Use `GridSearchCV` with `cv=5` and `scoring='f1_weighted'` for smaller search spaces
- Use `Optuna` for tree-based models with large search spaces (LightGBM, XGBoost, CatBoost)
- All searches done on training data only

### Outputs
```
/content/drive/MyDrive/Thesis/models/tuned/*.pkl
/content/drive/MyDrive/Thesis/results/tuned_results.csv
/content/drive/MyDrive/Thesis/results/best_params.json
```

### Notes
- Log tuning time per model
- Set `n_jobs=-1` where supported
- Save `best_params` as JSON for full reproducibility

---

## NB-05 · Model Selection & Ensemble Voting

**Goal:** Select the top 3 tuned models and build hard + soft voting ensembles.

### Tasks
- Load `tuned_results.csv` and rank models by F1-weighted (or ROC-AUC)
- Select top 3 models
- Load their saved `.pkl` files
- Build `VotingClassifier` (hard voting) from top 3
- Build `VotingClassifier` (soft voting) from top 3 — requires `predict_proba` support
- Evaluate both voting classifiers on the test set
- Compare: baseline top-3, tuned top-3, hard voting, soft voting

### Outputs
```
/content/drive/MyDrive/Thesis/models/voting_hard.pkl
/content/drive/MyDrive/Thesis/models/voting_soft.pkl
/content/drive/MyDrive/Thesis/results/ensemble_results.csv
```

### Notes
- If any top-3 model does not support `predict_proba`, swap it out for the next best model for soft voting only
- SVM requires `probability=True` at init time to support soft voting — ensure this is set in NB-04 if SVM is in top 3

---

## NB-06 · Final Evaluation & Reporting

**Goal:** Produce all thesis-ready outputs for the best model(s).

### Tasks
- Load test set, best voting classifier(s), and scaler
- Full classification report (precision, recall, F1 per class)
- Confusion matrix heatmap (seaborn)
- ROC curve(s) with AUC annotation
- Precision-Recall curve
- Feature importance plot (for tree-based models in the ensemble)
- Final summary table: all 13 baseline → tuned → ensemble

### Outputs
```
/content/drive/MyDrive/Thesis/figures/confusion_matrix.png
/content/drive/MyDrive/Thesis/figures/roc_curve.png
/content/drive/MyDrive/Thesis/figures/pr_curve.png
/content/drive/MyDrive/Thesis/figures/feature_importance.png
/content/drive/MyDrive/Thesis/results/final_summary.csv
```

### Notes
- Use `dpi=300` for all saved figures (publication quality)
- All plots should have clear axis labels, titles, and legends
- This notebook should be self-contained — reload everything from saved files; do not re-train

---

## Design Principles

**No data leakage** — the scaler and SMOTE are fit only on training data. The test set is untouched from NB-02 onward.

**Restartable at any stage** — each notebook saves its outputs to Drive, so any notebook can be re-run independently without re-running everything from scratch.

**Reproducibility** — `random_state=42` everywhere it is accepted. Best hyperparameters saved as JSON.

**Separation of concerns** — training logic lives in NB-03, tuning in NB-04, ensemble in NB-05, reporting in NB-06. Do not mix these.

---

*Plan written for thesis ML pipeline · 7 notebooks · 13 models · 1 ensemble*