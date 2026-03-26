# Early Prediction of Sepsis from Clinical Data

> A complete, reproducible machine-learning pipeline for predicting sepsis onset using the **PhysioNet/Computing in Cardiology Challenge 2019** dataset. The pipeline trains, tunes, and ensembles **13 classifiers**, producing thesis-ready evaluation figures and summary tables.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Notebooks](#notebooks)
  - [NB-00 · Setup & Data Ingestion](#nb-00--setup--data-ingestion)
  - [NB-01 · Data Cleaning & Feature Engineering](#nb-01--data-cleaning--feature-engineering)
  - [NB-02 · Preprocessing Pipeline](#nb-02--preprocessing-pipeline)
  - [NB-03 · Baseline Model Training](#nb-03--baseline-model-training)
  - [NB-04 · Hyperparameter Tuning](#nb-04--hyperparameter-tuning)
  - [NB-05 · Model Selection & Ensemble Voting](#nb-05--model-selection--ensemble-voting)
  - [NB-06 · Final Evaluation & Reporting](#nb-06--final-evaluation--reporting)
- [Models](#models)
- [Anti-Leakage Design](#anti-leakage-design)
- [Outputs & File Structure](#outputs--file-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Reproducibility](#reproducibility)

---

## Overview

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Early detection is critical — every hour of delayed treatment increases mortality. This project builds a predictive model that classifies whether a patient will develop sepsis based on hourly clinical vital signs and laboratory values recorded in the ICU.

The pipeline is split into **7 focused notebooks**, each handling one phase. Every notebook reads from and writes to Google Drive, making each stage independently restartable without re-running the entire pipeline.

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/) |
| **Format** | PSV (pipe-separated value) files — one per patient |
| **Training sets** | `training_setA` (~20,336 patients) + `training_setB` (~20,000 patients) |
| **Total patients** | ~40,336 |
| **Total rows** | ~1,552,210 (hourly observations across all patients) |
| **Features** | 40 clinical variables (vitals, labs, demographics) |
| **Target** | `SepsisLabel` — binary (0 = no sepsis, 1 = sepsis) |
| **Class imbalance** | Heavily imbalanced (~98% negative, ~2% positive) |

### Feature Categories

| Category | Features |
|----------|----------|
| **Vital Signs** | HR (Heart Rate), O2Sat, Temp, SBP (Systolic BP), MAP (Mean Arterial Pressure), DBP (Diastolic BP), Resp (Respiration Rate), EtCO2 |
| **Laboratory Values** | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets |
| **Demographics** | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS |
| **Engineered** | ShockIndex (HR/SBP), PulsePressure (SBP − DBP) |

---

## Pipeline Architecture

```
archive.zip
    │
    ▼
┌─────────────────────────────────┐
│  NB-00 · Data Ingestion         │ ──► raw.parquet
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-01 · Cleaning & Features   │ ──► clean.parquet
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-02 · Preprocessing         │ ──► X_train.npy, X_test.npy
│  (Patient-level split,          │     y_train.npy, y_test.npy
│   StandardScaler, SMOTE)        │     scaler.pkl
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-03 · Baseline Training      │ ──► baseline/*.pkl
│  (13 models, default params)    │     baseline_results.csv
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-04 · Hyperparameter Tuning  │ ──► tuned/*.pkl
│  (GridSearchCV + Optuna)        │     tuned_results.csv
│  (SMOTE inside CV folds)        │     best_params.json
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-05 · Ensemble Voting        │ ──► voting_hard.pkl
│  (Top-3 → Hard + Soft voting)  │     voting_soft.pkl
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  NB-06 · Final Evaluation       │ ──► confusion_matrix.png
│  (Publication-quality outputs)  │     roc_curve.png
│                                 │     pr_curve.png
│                                 │     feature_importance.png
│                                 │     final_summary.csv
└─────────────────────────────────┘
```

---

## Notebooks

### NB-00 · Setup & Data Ingestion

**File:** `notebooks/NB_00_Setup_Data_Ingestion.py`

Mounts Google Drive, extracts `archive.zip` from the dataset directory, and loads all ~40,000 individual patient PSV files into a single Pandas DataFrame using `glob` and `tqdm`. Performs initial sanity checks:

- Shape and dtypes validation
- Null percentage per column
- Target class distribution and imbalance ratio
- Unique patient count

**Output:** `data/raw.parquet`

---

### NB-01 · Data Cleaning & Feature Engineering

**File:** `notebooks/NB_01_Cleaning_Feature_Engineering.py`

Removes low-quality columns and creates clinically meaningful features:

1. **Column pruning** — drops any column with >40% missing values (threshold is configurable). Records all dropped columns to `dropped_columns.txt` for auditability.
2. **Null imputation** — fills remaining numeric nulls with column-wise medians.
3. **Feature engineering:**
   - **Shock Index** = HR / SBP — elevated values indicate hemodynamic instability; a key early sepsis marker.
   - **Pulse Pressure** = SBP − DBP — narrowing pulse pressure signals cardiovascular compromise.
4. **Validation** — checks for remaining NaN/inf values and corrects them.

**Output:** `data/clean.parquet`

---

### NB-02 · Preprocessing Pipeline

**File:** `notebooks/NB_02_Preprocessing_Pipeline.py`

Prepares model-ready arrays with strict anti-leakage guarantees:

1. **Patient-level split** — groups all time-series rows by `patient_id` and performs a stratified 80/20 split at the patient level. This prevents temporal data from the same patient appearing in both train and test sets.
2. **StandardScaler** — fitted on training data only; applied to both train and test via `transform()`.
3. **SMOTE** — applied on the training set only to balance the severe class imbalance (~98% vs ~2%).
4. **Pre-SMOTE snapshots** — saves `X_train_pre_smote.npy` and `y_train_pre_smote.npy` so that NB-04 can apply SMOTE inside each CV fold during tuning (avoiding information leakage from global SMOTE).

**Outputs:**
| File | Description |
|------|-------------|
| `X_train.npy` | Training features (SMOTE-balanced, scaled) |
| `X_test.npy` | Test features (scaled, untouched) |
| `y_train.npy` | Training labels (SMOTE-balanced) |
| `y_test.npy` | Test labels (original distribution) |
| `X_train_pre_smote.npy` | Training features before SMOTE (for CV in NB-04) |
| `y_train_pre_smote.npy` | Training labels before SMOTE (for CV in NB-04) |
| `scaler.pkl` | Fitted StandardScaler for reuse |
| `feature_names.csv` | Ordered list of feature column names |

---

### NB-03 · Baseline Model Training

**File:** `notebooks/NB_03_Baseline_Training.py`

Trains all **13 classifiers** with default hyperparameters and evaluates each on the held-out test set:

| # | Model | Key Property |
|---|-------|-------------|
| 1 | MLP (Multi-Layer Perceptron) | Neural network, `max_iter=300` |
| 2 | KNN (K-Nearest Neighbors) | Instance-based learning |
| 3 | Decision Tree | Fully interpretable |
| 4 | AdaBoost | Boosting (SAMME algorithm) |
| 5 | HistGradientBoosting | Histogram-based, fast for large data |
| 6 | Random Forest | Bagging + feature randomness |
| 7 | Bagging Classifier | Bootstrap aggregation |
| 8 | Gradient Boosting | Sequential error correction |
| 9 | LightGBM | Leaf-wise growth, GPU support |
| 10 | XGBoost | Regularised gradient boosting |
| 11 | CatBoost | Ordered boosting, handles categoricals |
| 12 | Extra Trees | Extremely randomised splits |
| 13 | SVM | Support vectors, `probability=True` |

**Metrics captured:** Accuracy, Precision (weighted & macro), Recall (weighted & macro), F1-Score (weighted & macro), ROC-AUC, and training time.

**Outputs:** `models/baseline/*.pkl`, `results/baseline_results.csv`

---

### NB-04 · Hyperparameter Tuning

**File:** `notebooks/NB_04_Hyperparameter_Tuning.py`

Tunes all 13 models with proper SMOTE handling inside cross-validation:

**Strategy:**
- **sklearn models (10):** `GridSearchCV` with `cv=5`, `scoring='f1_weighted'`, and an `imblearn.pipeline.Pipeline` wrapping `SMOTE → Model`. Param keys are prefixed with `model__` to route through the pipeline. This applies SMOTE **only to each CV training fold**, never to the validation fold.
- **Tree-based boosters (3):** `Optuna` (50 trials) for LightGBM, XGBoost, and CatBoost. These models have large, interdependent search spaces where Bayesian optimization significantly outperforms grid search. Same `Pipeline([SMOTE, Model])` pattern wraps each `cross_val_score` call.
- **Data source:** Uses `X_train_pre_smote.npy` (scaled, pre-SMOTE) so that SMOTE is never double-applied.

**Key tuning parameters:**

| Model | Tuned Hyperparameters |
|-------|----------------------|
| MLP | `hidden_layer_sizes`, `activation`, `alpha`, `learning_rate` |
| KNN | `n_neighbors`, `weights`, `metric` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` |
| AdaBoost | `n_estimators`, `learning_rate` |
| HistGB | `max_iter`, `max_depth`, `learning_rate`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split/leaf` |
| Bagging | `n_estimators`, `max_samples`, `max_features` |
| Gradient Boosting | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| Extra Trees | `n_estimators`, `max_depth`, `min_samples_split/leaf` |
| SVM | `C`, `kernel`, `gamma` |
| LightGBM | `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `min_child_samples` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma` |
| CatBoost | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `border_count` |

**Outputs:** `models/tuned/*.pkl`, `results/tuned_results.csv`, `results/best_params.json`

---

### NB-05 · Model Selection & Ensemble Voting

**File:** `notebooks/NB_05_Model_Selection_Ensemble.py`

Selects the top 3 tuned models by F1-Weighted score and constructs two ensemble classifiers:

1. **Hard Voting** — majority-vote prediction from the top 3 models.
2. **Soft Voting** — weighted average of predicted probabilities. Requires `predict_proba` support; if any top-3 model lacks it, the next-best compatible model is substituted automatically.

Produces a comparison table spanning: baseline individual → tuned individual → ensemble.

**Outputs:** `models/voting_hard.pkl`, `models/voting_soft.pkl`, `results/ensemble_results.csv`

---

### NB-06 · Final Evaluation & Reporting

**File:** `notebooks/NB_06_Final_Evaluation_Reporting.py`

Generates all thesis-ready outputs. This notebook is fully self-contained — it reloads everything from saved files and does **not** re-train any models.

**Outputs (all at `dpi=300`):**

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | Seaborn heatmap with annotated counts |
| `roc_curve.png` | ROC curve with AUC annotation |
| `pr_curve.png` | Precision-Recall curve with Average Precision |
| `feature_importance.png` | Top 15 features (averaged across tree-based ensemble members) |
| `final_summary.csv` | Complete results table: all 13 models × (Baseline → Tuned → Ensemble) |

---

## Models

The 13 classifiers span the major families of supervised learning:

| Family | Models |
|--------|--------|
| **Neural Networks** | MLP |
| **Instance-Based** | KNN |
| **Tree-Based** | Decision Tree |
| **Bagging Ensembles** | Random Forest, Extra Trees, Bagging |
| **Boosting Ensembles** | AdaBoost, Gradient Boosting, HistGradientBoosting, LightGBM, XGBoost, CatBoost |
| **Kernel Methods** | SVM |

---

## Anti-Leakage Design

This pipeline was specifically engineered to prevent every common form of data leakage in medical ML:

| Leakage Vector | Mitigation |
|----------------|------------|
| **Temporal leakage** | Patient-level train/test split (NB-02). All rows from a given patient stay entirely within either the train or test set — never split across both. |
| **Scaler leakage** | `StandardScaler` is `fit()` on training data only and `transform()`-ed on test data (NB-02). The fitted scaler is saved once and reused; it is never re-fit. |
| **SMOTE leakage** | SMOTE is applied only to the training set. Within hyperparameter tuning (NB-04), SMOTE is applied inside each CV fold using `imblearn.pipeline.Pipeline`, so synthetic samples never bleed into validation folds. |
| **Pre-SMOTE arrays** | NB-02 saves `X_train_pre_smote.npy` / `y_train_pre_smote.npy`. NB-04 loads these instead of the SMOTE-balanced arrays, ensuring SMOTE is only applied per-fold. |
| **Test set isolation** | The test set is created in NB-02 and never modified, re-sampled, or used for any fitting from that point onward. |

---

## Outputs & File Structure

```
/content/drive/MyDrive/Thesis/
│
├── dataset/
│   └── archive.zip                 # Raw PhysioNet 2019 dataset
│
├── data/
│   ├── raw.parquet                 # NB-00: unmodified concatenated data
│   ├── clean.parquet               # NB-01: cleaned + engineered features
│   ├── dropped_columns.txt         # NB-01: record of pruned columns
│   ├── feature_names.csv           # NB-02: ordered feature list
│   ├── X_train.npy                 # NB-02: train features (scaled + SMOTE)
│   ├── X_test.npy                  # NB-02: test features (scaled only)
│   ├── y_train.npy                 # NB-02: train labels (SMOTE-balanced)
│   ├── y_test.npy                  # NB-02: test labels (original)
│   ├── X_train_pre_smote.npy       # NB-02: train features (scaled, pre-SMOTE)
│   └── y_train_pre_smote.npy       # NB-02: train labels (pre-SMOTE)
│
├── models/
│   ├── scaler.pkl                  # NB-02: fitted StandardScaler
│   ├── voting_hard.pkl             # NB-05: hard voting ensemble
│   ├── voting_soft.pkl             # NB-05: soft voting ensemble
│   ├── baseline/                   # NB-03: 13 default-param models
│   │   ├── MLP.pkl
│   │   ├── KNN.pkl
│   │   ├── DecisionTree.pkl
│   │   ├── AdaBoost.pkl
│   │   ├── HistGradientBoosting.pkl
│   │   ├── RandomForest.pkl
│   │   ├── Bagging.pkl
│   │   ├── GradientBoosting.pkl
│   │   ├── LightGBM.pkl
│   │   ├── XGBoost.pkl
│   │   ├── CatBoost.pkl
│   │   ├── ExtraTrees.pkl
│   │   └── SVM.pkl
│   └── tuned/                      # NB-04: tuned pipelines (SMOTE+Model)
│       └── *.pkl
│
├── results/
│   ├── baseline_results.csv        # NB-03: baseline metrics (13 models)
│   ├── tuned_results.csv           # NB-04: tuned metrics (13 models)
│   ├── best_params.json            # NB-04: optimal hyperparameters
│   ├── ensemble_results.csv        # NB-05: ensemble comparison
│   └── final_summary.csv           # NB-06: complete summary table
│
└── figures/
    ├── confusion_matrix.png        # NB-06: (300 dpi)
    ├── roc_curve.png               # NB-06: (300 dpi)
    ├── pr_curve.png                # NB-06: (300 dpi)
    └── feature_importance.png      # NB-06: (300 dpi)
```

---

## How to Run

### Prerequisites

1. **Google Colab** (recommended) — all notebooks assume Colab runtime with Google Drive mounted at `/content/drive`.
2. Place `archive.zip` at `/content/drive/MyDrive/Thesis/dataset/archive.zip`.

### Execution

Each `.py` file is organized into clearly delimited cells (marked with `# ── Cell N ──`). To use in Colab:

1. Create a new Colab notebook.
2. Copy each cell section into a separate Colab code cell.
3. Run notebooks **sequentially** from NB-00 through NB-06.
4. After first run, any notebook can be re-run independently since all intermediate outputs are persisted to Drive.

```
NB-00  →  NB-01  →  NB-02  →  NB-03  →  NB-04  →  NB-05  →  NB-06
                                  ↕           ↕
                          (independently restartable)
```

### Install commands (run once per Colab session)

```bash
pip install -q catboost lightgbm xgboost optuna imbalanced-learn pyarrow tqdm seaborn
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, metrics, preprocessing |
| `imbalanced-learn` | SMOTE, imblearn Pipeline |
| `lightgbm` | LightGBM classifier |
| `xgboost` | XGBoost classifier |
| `catboost` | CatBoost classifier |
| `optuna` | Bayesian hyperparameter optimization |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |
| `pyarrow` | Parquet file I/O |
| `tqdm` | Progress bars |
| `joblib` | Model serialization |

---

## Reproducibility

- **`random_state=42`** is set everywhere it is accepted: data splits, SMOTE, all 13 classifiers, and Optuna samplers.
- **Best hyperparameters** are saved to `best_params.json` for exact reconstruction of tuned models.
- **Patient-level splitting** ensures identical train/test partitions across re-runs given the same random state.
- **Pipeline serialization** — tuned models are saved as full `imblearn.pipeline.Pipeline` objects (SMOTE + Model), ensuring prediction-time consistency.

---

*Thesis ML pipeline · 7 notebooks · 13 models · 1 ensemble · Zero leakage*
