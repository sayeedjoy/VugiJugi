# =============================================================================
# NB-03 · Baseline Model Training & Evaluation
# =============================================================================
# Goal: Train all 13 models with default hyperparameters and record
#       performance for comparison.
# =============================================================================

# ── Cell 1 ── Install extra dependencies ────────────────────────────────────
# !pip install -q catboost lightgbm xgboost

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, time, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")

os.makedirs(os.path.join(MODELS_DIR, "baseline"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Cell 3 ── Load preprocessed arrays ──────────────────────────────────────
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")

# ── Cell 4 ── Define all 13 models ──────────────────────────────────────────
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Linear SVM via SGD — orders of magnitude faster than SVC(kernel='rbf')
# CalibratedClassifierCV adds predict_proba support needed for ROC-AUC
_svm_base = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3,
                          random_state=42, n_jobs=-1)

models = {
    "MLP":                  MLPClassifier(max_iter=300, random_state=42),
    "KNN":                  KNeighborsClassifier(),
    "DecisionTree":         DecisionTreeClassifier(random_state=42),
    "AdaBoost":             AdaBoostClassifier(random_state=42, algorithm="SAMME"),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "RandomForest":         RandomForestClassifier(random_state=42, n_jobs=-1),
    "Bagging":              BaggingClassifier(random_state=42, n_jobs=-1),
    "GradientBoosting":     GradientBoostingClassifier(random_state=42),
    "LightGBM":             LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    "XGBoost":              XGBClassifier(random_state=42, n_jobs=-1,
                                          use_label_encoder=False, eval_metric="logloss"),
    "CatBoost":             CatBoostClassifier(random_state=42, verbose=0),
    "ExtraTrees":           ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "SVM":                  CalibratedClassifierCV(_svm_base, cv=3),
}

print(f"\nTotal models to train: {len(models)}")

# ── Cell 5 ── Train & evaluate loop ────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

results = []

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"  Training: {name}")
    print(f"{'=' * 50}")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_m  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    roc   = roc_auc_score(y_test, y_prob)

    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Precision (W): {prec:.4f}   (M): {prec_m:.4f}")
    print(f"  Recall    (W): {rec:.4f}   (M): {rec_m:.4f}")
    print(f"  F1 Score  (W): {f1_w:.4f}   (M): {f1_m:.4f}")
    print(f"  ROC AUC      : {roc:.4f}")
    print(f"  Train time   : {train_time:.1f}s")

    results.append({
        "Model":            name,
        "Accuracy":         round(acc, 4),
        "Precision_W":      round(prec, 4),
        "Recall_W":         round(rec, 4),
        "F1_Weighted":      round(f1_w, 4),
        "Precision_M":      round(prec_m, 4),
        "Recall_M":         round(rec_m, 4),
        "F1_Macro":         round(f1_m, 4),
        "ROC_AUC":          round(roc, 4),
        "Train_Time_Sec":   round(train_time, 1),
    })

    # Save model
    model_path = os.path.join(MODELS_DIR, "baseline", f"{name}.pkl")
    joblib.dump(model, model_path)
    print(f"  → Saved: {model_path}")

# ── Cell 6 ── Summary table ─────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1_Weighted", ascending=False).reset_index(drop=True)

print("\n" + "=" * 80)
print("BASELINE RESULTS — sorted by F1 Weighted")
print("=" * 80)
print(results_df.to_string(index=False))

# Save
csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Saved baseline results → {csv_path}")
