# =============================================================================
# NB-05 · Model Selection & Ensemble Voting
# =============================================================================
# Goal: Select the top 3 tuned models and build hard + soft voting ensembles.
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
# Load pre-SMOTE train arrays. The tuned models saved by NB-04 are
# imblearn.Pipeline([SMOTE, model]), so SMOTE runs inside VotingClassifier.fit().
# Using the post-SMOTE X_train.npy here would cause double SMOTE on an already-
# balanced dataset, skewing the class distribution.
X_train = np.load(os.path.join(DATA_DIR, "X_train_pre_smote.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train_pre_smote.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_train (pre-SMOTE): {X_train.shape}  X_test: {X_test.shape}")

# ── Cell 4 ── Load tuned results and rank models ──────────────────────────
tuned_df = pd.read_csv(os.path.join(RESULTS_DIR, "tuned_results.csv"))
tuned_df = tuned_df.sort_values("F1_Weighted", ascending=False).reset_index(drop=True)

print("\nTuned model rankings:")
print(tuned_df[["Model", "F1_Weighted", "ROC_AUC"]].to_string(index=False))

# ── Cell 5 ── Select top 3 models ──────────────────────────────────────────
top3_names = tuned_df["Model"].head(3).tolist()
print(f"\nTop 3 models: {top3_names}")

# Load the saved tuned models
top3_models = {}
for name in top3_names:
    model_path = os.path.join(MODELS_DIR, "tuned", f"{name}.pkl")
    top3_models[name] = joblib.load(model_path)
    print(f"  ✓ Loaded {name} from {model_path}")

# ── Cell 6 ── Check predict_proba support for soft voting ──────────────────
soft_eligible = {}
hard_estimators = []

for name, model in top3_models.items():
    has_proba = hasattr(model, "predict_proba")
    print(f"  {name}: predict_proba = {has_proba}")
    hard_estimators.append((name, model))
    if has_proba:
        soft_eligible[name] = model

# If any top-3 model lacks predict_proba, find a replacement for soft voting
soft_estimators = list(hard_estimators)  # start with same

if len(soft_eligible) < 3:
    print(f"\n⚠ Only {len(soft_eligible)}/3 models support predict_proba.")
    print("  Searching for replacement(s) for soft voting...")

    for _, row in tuned_df.iterrows():
        if len(soft_estimators) >= 3 and all(
            hasattr(est[1], "predict_proba") for est in soft_estimators
        ):
            break
        name = row["Model"]
        if name not in [e[0] for e in soft_estimators]:
            model_path = os.path.join(MODELS_DIR, "tuned", f"{name}.pkl")
            candidate = joblib.load(model_path)
            if hasattr(candidate, "predict_proba"):
                # Replace a non-proba model
                for i, (ename, emodel) in enumerate(soft_estimators):
                    if not hasattr(emodel, "predict_proba"):
                        print(f"  Replacing {ename} with {name} for soft voting")
                        soft_estimators[i] = (name, candidate)
                        break

print(f"\nHard voting estimators: {[e[0] for e in hard_estimators]}")
print(f"Soft voting estimators: {[e[0] for e in soft_estimators]}")

# ── Cell 7 ── Build & evaluate Hard Voting Classifier ──────────────────────
def evaluate_model(model, X_test, y_test, label=""):
    y_pred = model.predict(X_test)
    metrics = {
        "Model":       label,
        "Accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "Precision_W": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall_W":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1_Weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1_Macro":    round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["ROC_AUC"] = round(roc_auc_score(y_test, y_prob), 4)
    else:
        metrics["ROC_AUC"] = None
    return metrics

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
print(f"  Fit time: {time.time() - start:.0f}s")
hard_metrics = evaluate_model(voting_hard, X_test, y_test, "Voting_Hard")
print(f"  F1 Weighted: {hard_metrics['F1_Weighted']}  |  ROC AUC: {hard_metrics['ROC_AUC']}")

# ── Cell 8 ── Build & evaluate Soft Voting Classifier ──────────────────────
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
print(f"  Fit time: {time.time() - start:.0f}s")
soft_metrics = evaluate_model(voting_soft, X_test, y_test, "Voting_Soft")
print(f"  F1 Weighted: {soft_metrics['F1_Weighted']}  |  ROC AUC: {soft_metrics['ROC_AUC']}")

# ── Cell 9 ── Save voting classifiers ──────────────────────────────────────
joblib.dump(voting_hard, os.path.join(MODELS_DIR, "voting_hard.pkl"))
joblib.dump(voting_soft, os.path.join(MODELS_DIR, "voting_soft.pkl"))
print(f"\n✓ Saved voting_hard.pkl")
print(f"✓ Saved voting_soft.pkl")

# ── Cell 10 ── Comparison table ─────────────────────────────────────────────
ensemble_results = []

# Add individual top-3 tuned results
for name in top3_names:
    row = tuned_df[tuned_df["Model"] == name].iloc[0]
    ensemble_results.append({
        "Model":       f"{name} (tuned)",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC":     row["ROC_AUC"],
    })

# Add baseline top-3 results
baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
for name in top3_names:
    brow = baseline_df[baseline_df["Model"] == name]
    if not brow.empty:
        brow = brow.iloc[0]
        ensemble_results.append({
            "Model":       f"{name} (baseline)",
            "F1_Weighted": brow["F1_Weighted"],
            "ROC_AUC":     brow["ROC_AUC"],
        })

# Add voting results
ensemble_results.append({
    "Model":       "Voting Hard",
    "F1_Weighted": hard_metrics["F1_Weighted"],
    "ROC_AUC":     hard_metrics.get("ROC_AUC"),
})
ensemble_results.append({
    "Model":       "Voting Soft",
    "F1_Weighted": soft_metrics["F1_Weighted"],
    "ROC_AUC":     soft_metrics.get("ROC_AUC"),
})

ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values("F1_Weighted", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("ENSEMBLE COMPARISON")
print("=" * 70)
print(ensemble_df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, "ensemble_results.csv")
ensemble_df.to_csv(csv_path, index=False)
print(f"\n✅ Saved ensemble results → {csv_path}")
