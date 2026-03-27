# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Early sepsis prediction ML pipeline using the PhysioNet/Computing in Cardiology Challenge 2019 dataset (~40K patients, ~1.5M hourly ICU observations). Trains and evaluates 13 classifiers (baseline → tuned → ensemble) for binary sepsis classification against a heavily imbalanced dataset (~2% positive).

**This project runs in Google Colab** — all paths reference `/content/drive/MyDrive/Thesis/` on Google Drive. Scripts are not designed to run locally.

## Pipeline Execution Order

The 7 notebooks in `notebooks/` must run sequentially; each reads the previous stage's output from Google Drive:

```
NB_00 → NB_01 → NB_02 → NB_03 → NB_04 → NB_05 → NB_06
```

| Notebook | Input | Output |
|---|---|---|
| NB_00_Setup_Data_Ingestion | `archive.zip` | `raw.parquet` |
| NB_01_Cleaning_Feature_Engineering | `raw.parquet` | `clean.parquet` |
| NB_02_Preprocessing_Pipeline | `clean.parquet` | `X/y_train/test.npy`, `scaler.pkl`, `X_train_pre_smote.npy` |
| NB_03_Baseline_Training | arrays from NB_02 | `baseline/*.pkl`, `baseline_results.csv` |
| NB_04_Hyperparameter_Tuning | pre-SMOTE arrays | `tuned/*.pkl`, `best_params.json`, `tuned_results.csv` |
| NB_05_Model_Selection_Ensemble | tuned models | `voting_hard.pkl`, `voting_soft.pkl` |
| NB_06_Final_Evaluation_Reporting | best ensemble | figures (300 dpi), `ensemble_results.csv` |

Each notebook is independently restartable — just mount Drive and run.

## Anti-Leakage Design (Critical)

This pipeline is specifically designed to prevent every common leakage vector. When modifying code, preserve these invariants:

- **Patient-level split**: All rows for a patient go to either train or test — never split across sets. `groupby('patient_id')` determines split boundaries.
- **Scaler fit on train only**: `StandardScaler` is fit on `X_train`, then `transform` applied to both. Never `fit_transform` on test.
- **SMOTE train-only**: SMOTE is applied only after the train/test split. Pre-SMOTE arrays (`X_train_pre_smote.npy`) are saved so NB_04 can wrap SMOTE inside CV folds.
- **SMOTE inside CV pipeline (NB_04)**: During hyperparameter tuning, SMOTE is wrapped in an `imblearn.Pipeline` so validation folds never see synthetic samples.
- **Test set immutability**: Created once in NB_02, never re-sampled or modified.

## Key Implementation Details

- **Random state**: `42` everywhere — data splits, SMOTE, all classifiers, Optuna samplers
- **Tuning strategy**: GridSearchCV (cv=5, `scoring='f1_weighted'`) for sklearn models; Optuna (50 trials, `TPESampler(seed=42)`) for LightGBM, XGBoost, CatBoost
- **SVM**: Implemented as `SGDClassifier` + `CalibratedClassifierCV` (not `SVC`) to support `predict_proba` for ROC-AUC computation
- **Primary metric**: ROC-AUC for ranking models; F1-weighted for CV scoring during tuning
- **Engineered features**: `ShockIndex = HR/SBP`, `PulsePressure = SBP - DBP` (added in NB_01)
- **Missingness threshold**: Columns with >40% missing values are dropped in NB_01

## Dependencies

Installed via `pip install` cells at the top of each notebook (Colab environment):
`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `xgboost`, `catboost`, `optuna`, `matplotlib`, `seaborn`, `joblib`
