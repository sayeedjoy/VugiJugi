# =============================================================================
# NB-06 · Final Evaluation & Reporting
# =============================================================================
# Goal: Produce all thesis-ready outputs for the best model(s).
#       This notebook is self-contained — it reloads everything from
#       saved files and does NOT re-train any models.
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q catboost lightgbm xgboost seaborn

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

# Publication-quality defaults
plt.rcParams.update({
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.figsize":   (8, 6),
})

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")
FIGURES_DIR = os.path.join(THESIS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Cell 3 ── Load test data and models ─────────────────────────────────────
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
voting_hard  = joblib.load(os.path.join(MODELS_DIR, "voting_hard.pkl"))
voting_soft  = joblib.load(os.path.join(MODELS_DIR, "voting_soft.pkl"))

print(f"X_test: {X_test.shape}  y_test: {y_test.shape}")
print(f"Models loaded: voting_hard, voting_soft, scaler")

# Load feature names
feature_names_path = os.path.join(DATA_DIR, "feature_names.csv")
if os.path.exists(feature_names_path):
    feature_names = pd.read_csv(feature_names_path, header=None)[0].tolist()
else:
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

# ── Cell 4 ── Determine best model ─────────────────────────────────────────
# Evaluate both voting classifiers and use the better one as the "best"
def quick_eval(model, X, y, label):
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    return f1

f1_hard = quick_eval(voting_hard, X_test, y_test, "Hard")
f1_soft = quick_eval(voting_soft, X_test, y_test, "Soft")
print(f"\nVoting Hard F1: {f1_hard:.4f}")
print(f"Voting Soft F1: {f1_soft:.4f}")

if f1_soft >= f1_hard:
    best_model = voting_soft
    best_label = "Soft Voting Ensemble"
    print(f"\n→ Using Soft Voting as the best model.")
else:
    best_model = voting_hard
    best_label = "Hard Voting Ensemble"
    print(f"\n→ Using Hard Voting as the best model.")

y_pred = best_model.predict(X_test)
y_prob = (best_model.predict_proba(X_test)[:, 1]
          if hasattr(best_model, "predict_proba") else None)

# ── Cell 5 ── Full classification report ────────────────────────────────────
print("\n" + "=" * 70)
print(f"CLASSIFICATION REPORT — {best_label}")
print("=" * 70)
target_names = ["No Sepsis (0)", "Sepsis (1)"]
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# ── Cell 6 ── Confusion matrix heatmap ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    linewidths=0.5, linecolor="gray",
    ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — {best_label}")
plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✓ Saved → {cm_path}")

# ── Cell 7 ── ROC curve ────────────────────────────────────────────────────
if y_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"{best_label} (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {roc_path}")
else:
    print("⚠ ROC curve skipped — model does not support predict_proba.")

# ── Cell 8 ── Precision-Recall curve ───────────────────────────────────────
if y_prob is not None:
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, color="#ff7f0e", lw=2,
            label=f"{best_label} (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(FIGURES_DIR, "pr_curve.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {pr_path}")
else:
    print("⚠ PR curve skipped — model does not support predict_proba.")

# ── Cell 9 ── Feature importance (aggregate from tree-based estimators) ──────
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (from tree-based ensemble members)")
print("=" * 70)

importances_sum = np.zeros(X_test.shape[1])
n_tree_models = 0

# Access individual estimators from the voting classifier
estimators = (best_model.estimators_ if hasattr(best_model, "estimators_")
              else [])

for est in estimators:
    if hasattr(est, "feature_importances_"):
        importances_sum += est.feature_importances_
        n_tree_models += 1

if n_tree_models > 0:
    avg_importance = importances_sum / n_tree_models
    fi_df = pd.DataFrame({
        "Feature": feature_names[:len(avg_importance)],
        "Importance": avg_importance,
    }).sort_values("Importance", ascending=False)

    print(fi_df.head(15).to_string(index=False))

    # Plot top 15
    top_n = min(15, len(fi_df))
    plot_df = fi_df.head(top_n).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(plot_df["Feature"], plot_df["Importance"],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)))
    ax.set_xlabel("Average Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Ensemble)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved → {fi_path}")
else:
    print("⚠ No tree-based estimators found in the ensemble — skipping.")

# ── Cell 10 ── Final summary table (all 13 baseline → tuned → ensemble) ────
print("\n" + "=" * 80)
print("FINAL SUMMARY TABLE")
print("=" * 80)

summary_rows = []

# Baseline results
baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
for _, row in baseline_df.iterrows():
    summary_rows.append({
        "Model": row["Model"],
        "Stage": "Baseline",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC": row["ROC_AUC"],
    })

# Tuned results
tuned_df = pd.read_csv(os.path.join(RESULTS_DIR, "tuned_results.csv"))
for _, row in tuned_df.iterrows():
    summary_rows.append({
        "Model": row["Model"],
        "Stage": "Tuned",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC": row["ROC_AUC"],
    })

# Ensemble results
ensemble_df = pd.read_csv(os.path.join(RESULTS_DIR, "ensemble_results.csv"))
for _, row in ensemble_df.iterrows():
    summary_rows.append({
        "Model": row["Model"],
        "Stage": "Ensemble",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC": row.get("ROC_AUC"),
    })

final_df = pd.DataFrame(summary_rows)
final_df = final_df.sort_values(["Stage", "F1_Weighted"],
                                 ascending=[True, False]).reset_index(drop=True)

print(final_df.to_string(index=False))

final_csv = os.path.join(RESULTS_DIR, "final_summary.csv")
final_df.to_csv(final_csv, index=False)

print(f"\n✅ Saved final summary → {final_csv}")
print(f"\nAll figures saved to: {FIGURES_DIR}/")
print("Done! 🎉")
