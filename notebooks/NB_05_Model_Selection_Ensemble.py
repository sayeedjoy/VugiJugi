# =============================================================================
# NB-05 · Model Selection & Ensemble Voting
# =============================================================================
# Goal: Select the top 3 models and build hard + soft voting ensembles.
#
# NB-04 skip-safe: if tuned models / tuned_results.csv do not exist (because
# NB-04 was skipped), the notebook falls back to baseline models and results.
#
# Model format: NB-04 saves models as imblearn.Pipeline([SMOTE, model]).
# When used inside VotingClassifier each estimator's .fit() re-applies SMOTE
# on its own copy of X_train (pre-SMOTE). This is the correct anti-leakage
# behaviour — no double SMOTE because X_train here is pre-SMOTE.
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q catboost lightgbm xgboost

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, time, warnings, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")

# ── Cell 3 ── Load data ────────────────────────────────────────────────────
# Pre-SMOTE arrays: tuned models saved by NB-04 are imblearn.Pipeline([SMOTE,
# model]), so SMOTE runs inside each VotingClassifier estimator's fit().
# Using post-SMOTE X_train.npy would cause double-SMOTE on an already-balanced
# dataset — always use the pre-SMOTE split here.
# Cast to float32 to match NB-04 training dtype (avoids hidden dtype mismatch).
X_train = np.load(os.path.join(DATA_DIR, "X_train_pre_smote.npy")).astype(np.float32)
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train_pre_smote.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_train (pre-SMOTE): {X_train.shape}  dtype: {X_train.dtype}")
print(f"X_test:              {X_test.shape}   dtype: {X_test.dtype}")

# ── Cell 4 ── Load results and rank models ───────────────────────────────────
# Set USE_TUNED = False to force baseline models even if tuned_results.csv
# exists on Drive (e.g. when NB-04 cannot be re-run after NB-03 was updated).
USE_TUNED = False   # ← change to True only if NB-04 was run in this session

tuned_csv    = os.path.join(RESULTS_DIR, "tuned_results.csv")
baseline_csv = os.path.join(RESULTS_DIR, "baseline_results.csv")

if USE_TUNED and os.path.exists(tuned_csv):
    results_df    = pd.read_csv(tuned_csv)
    models_subdir = "tuned"
    results_stage = "tuned"
    print(f"\nUsing tuned results ({len(results_df)} models).")
else:
    results_df    = pd.read_csv(baseline_csv)
    models_subdir = "baseline"
    results_stage = "baseline"
    if USE_TUNED:
        print("[WARN] USE_TUNED=True but tuned_results.csv not found — using baseline.")
    else:
        print("Using baseline results (USE_TUNED=False).")

# Sort by ROC_AUC — honest metric under class imbalance.
# F1_Weighted can be >0.98 for a model that always predicts "No Sepsis"
# (because No Sepsis makes up ~98% of samples and dominates the weighted avg).
# ROC_AUC is threshold-independent and not distorted by class frequency.
results_df = results_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

print("\nModel rankings (by ROC_AUC):")
print(results_df[["Model", "ROC_AUC", "F1_Weighted"]].to_string(index=False))

# ── Cell 5 ── Select top 3 models ──────────────────────────────────────────
top3_names = results_df["Model"].head(3).tolist()
print(f"\nTop 3 models: {top3_names}")

top3_models = {}
for name in top3_names:
    model_path = os.path.join(MODELS_DIR, models_subdir, f"{name}.pkl")
    top3_models[name] = joblib.load(model_path)
    print(f"  Loaded {name} from {model_path}")

# ── Cell 6 ── Check predict_proba support for soft voting ──────────────────
hard_estimators = []
soft_eligible   = {}

for name, model in top3_models.items():
    has_proba = hasattr(model, "predict_proba")
    print(f"  {name}: predict_proba = {has_proba}")
    hard_estimators.append((name, model))
    if has_proba:
        soft_eligible[name] = model

# If any top-3 model lacks predict_proba, find a replacement for soft voting.
soft_estimators = list(hard_estimators)

if len(soft_eligible) < 3:
    print(f"\nWARN: Only {len(soft_eligible)}/3 models support predict_proba.")
    print("  Searching ranked list for a predict_proba-capable replacement...")

    for _, row in results_df.iterrows():
        if len(soft_estimators) >= 3 and all(
            hasattr(est[1], "predict_proba") for est in soft_estimators
        ):
            break
        name = row["Model"]
        if name not in [e[0] for e in soft_estimators]:
            model_path = os.path.join(MODELS_DIR, models_subdir, f"{name}.pkl")
            candidate  = joblib.load(model_path)
            if hasattr(candidate, "predict_proba"):
                for i, (ename, emodel) in enumerate(soft_estimators):
                    if not hasattr(emodel, "predict_proba"):
                        print(f"  Replacing {ename} → {name} for soft voting")
                        soft_estimators[i] = (name, candidate)
                        break

print(f"\nHard voting estimators: {[e[0] for e in hard_estimators]}")
print(f"Soft voting estimators: {[e[0] for e in soft_estimators]}")

# ── Cell 7 ── Evaluate helper ───────────────────────────────────────────────
def evaluate_model(model, X, y, label=""):
    y_pred = model.predict(X)
    metrics = {
        "Model":       label,
        "Accuracy":    round(accuracy_score(y, y_pred), 4),
        "Precision_W": round(precision_score(y, y_pred, average="weighted", zero_division=0), 4),
        "Recall_W":    round(recall_score(y, y_pred, average="weighted", zero_division=0), 4),
        "F1_Weighted": round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "F1_Macro":    round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        metrics["ROC_AUC"] = round(roc_auc_score(y, y_prob), 4)
    else:
        metrics["ROC_AUC"] = None
    return metrics

# ── Cell 8 ── Build & evaluate Hard Voting Classifier ──────────────────────
print(f"\n{'=' * 55}")
print("  Training: Hard Voting Classifier")
print(f"{'=' * 55}")

voting_hard = VotingClassifier(
    estimators=hard_estimators,
    voting="hard",
    n_jobs=-1,
)
start = time.time()
voting_hard.fit(X_train, y_train)
hard_fit_time = time.time() - start
print(f"  Fit time: {hard_fit_time:.0f}s")
hard_metrics = evaluate_model(voting_hard, X_test, y_test, "Voting_Hard")
hard_metrics["Fit_Time_Sec"] = round(hard_fit_time, 1)
print(f"  F1 Weighted: {hard_metrics['F1_Weighted']}  |  ROC AUC: {hard_metrics['ROC_AUC']}")

# ── Cell 9 ── Build & evaluate Soft Voting Classifier ──────────────────────
print(f"\n{'=' * 55}")
print("  Training: Soft Voting Classifier")
print(f"{'=' * 55}")

voting_soft = VotingClassifier(
    estimators=soft_estimators,
    voting="soft",
    n_jobs=-1,
)
start = time.time()
voting_soft.fit(X_train, y_train)
soft_fit_time = time.time() - start
print(f"  Fit time: {soft_fit_time:.0f}s")
soft_metrics = evaluate_model(voting_soft, X_test, y_test, "Voting_Soft")
soft_metrics["Fit_Time_Sec"] = round(soft_fit_time, 1)
print(f"  F1 Weighted: {soft_metrics['F1_Weighted']}  |  ROC AUC: {soft_metrics['ROC_AUC']}")

# ── Cell 10 ── Save voting classifiers ─────────────────────────────────────
joblib.dump(voting_hard, os.path.join(MODELS_DIR, "voting_hard.pkl"))
joblib.dump(voting_soft, os.path.join(MODELS_DIR, "voting_soft.pkl"))
print(f"\nSaved voting_hard.pkl")
print(f"Saved voting_soft.pkl")

# ── Cell 11 ── Comparison table ─────────────────────────────────────────────
ensemble_results = []

# Add individual top-3 results (tuned or baseline, whichever was used)
for name in top3_names:
    row = results_df[results_df["Model"] == name].iloc[0]
    ensemble_results.append({
        "Model":       f"{name} ({results_stage})",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC":     row["ROC_AUC"],
        "Fit_Time_Sec": row.get("Tune_Time_Sec", row.get("Fit_Time_Sec", None)),
    })

# If baseline was the fallback source, skip re-loading; if tuned was used,
# also include the corresponding baseline rows for context.
if results_stage == "tuned" and os.path.exists(baseline_csv):
    baseline_df = pd.read_csv(baseline_csv)
    for name in top3_names:
        brow = baseline_df[baseline_df["Model"] == name]
        if not brow.empty:
            brow = brow.iloc[0]
            ensemble_results.append({
                "Model":       f"{name} (baseline)",
                "F1_Weighted": brow["F1_Weighted"],
                "ROC_AUC":     brow["ROC_AUC"],
                "Fit_Time_Sec": brow.get("Fit_Time_Sec", None),
            })

# Add voting classifier results
ensemble_results.append({
    "Model":        "Voting Hard",
    "F1_Weighted":  hard_metrics["F1_Weighted"],
    "ROC_AUC":      hard_metrics.get("ROC_AUC"),
    "Fit_Time_Sec": hard_metrics["Fit_Time_Sec"],
})
ensemble_results.append({
    "Model":        "Voting Soft",
    "F1_Weighted":  soft_metrics["F1_Weighted"],
    "ROC_AUC":      soft_metrics.get("ROC_AUC"),
    "Fit_Time_Sec": soft_metrics["Fit_Time_Sec"],
})

ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("ENSEMBLE COMPARISON")
print("=" * 70)
print(ensemble_df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, "ensemble_results.csv")
ensemble_df.to_csv(csv_path, index=False)
print(f"\nSaved ensemble results → {csv_path}")
