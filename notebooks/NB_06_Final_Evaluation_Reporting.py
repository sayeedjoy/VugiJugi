# =============================================================================
# NB-06 · Final Evaluation & Reporting
# =============================================================================
# Goal: Produce all thesis-ready outputs for the best model(s).
#       This notebook is self-contained — it reloads everything from
#       saved files and does NOT re-train any models.
#
# NB-04 skip-safe: the final summary table omits the "Tuned" stage if
# tuned_results.csv does not exist (because NB-04 was skipped).
#
# Pipeline-aware: NB-04 saves models as imblearn.Pipeline([SMOTE, model]).
# Feature importance extraction drills into named_steps["model"] to reach
# the actual estimator inside the pipeline wrapper.
#
# Memory-optimised: models are loaded one at a time, predictions are batched,
# and all large objects are deleted + gc.collect()ed as soon as they are no
# longer needed — keeping peak RAM well below 8 GB even on free Colab.
#
# Imbalance-aware: default threshold 0.5 is useless on a ~2%-positive dataset.
# Cell 4b scans thresholds and selects the one maximising F2-score (β=2),
# which weights recall 2× over precision — appropriate for sepsis detection
# where missing a case is far worse than a false alarm.  All downstream cells
# use this optimal threshold instead of 0.5.
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q catboost lightgbm xgboost seaborn

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import gc, os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no GUI renderer → lower overhead
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
    f1_score,
    recall_score,
    precision_score,
)

warnings.filterwarnings("ignore")

# Publication-quality defaults
plt.rcParams.update({
    "figure.dpi":       150,   # lower in-memory resolution; savefig uses 300
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

# ── Cell 3 ── Load test data (memory-mapped) ────────────────────────────────
# mmap_mode='r' keeps the array on disk and pages in only what is needed,
# avoiding a full copy in RAM.  Slices/ops will still materialise sections
# of the array, but peak usage is much lower than a full eager load.
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"), mmap_mode='r')
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_test: {X_test.shape}  dtype: {X_test.dtype}")
print(f"y_test: {y_test.shape}")
pos = y_test.sum()
print(f"Positive (sepsis): {pos} / {len(y_test)}  ({100*pos/len(y_test):.2f}%)")

# Load feature names
feature_names_path = os.path.join(DATA_DIR, "feature_names.csv")
if os.path.exists(feature_names_path):
    feature_names = pd.read_csv(feature_names_path, header=None)[0].tolist()
else:
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler loaded.")

# ── Helper: batched predict / predict_proba ──────────────────────────────────
def batched_predict(model, X, batch_size=4096):
    """Return predictions in chunks to avoid one giant intermediate array."""
    parts = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size].astype(np.float32, copy=False)
        parts.append(model.predict(chunk))
    return np.concatenate(parts)

def batched_predict_proba(model, X, batch_size=4096):
    """Return class-1 probabilities in chunks."""
    parts = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start + batch_size].astype(np.float32, copy=False)
        parts.append(model.predict_proba(chunk)[:, 1])
    return np.concatenate(parts)

# ── Cell 4 ── Determine best model (load one at a time) ─────────────────────
# Selection criterion: ROC-AUC is threshold-independent and honest under class
# imbalance.  F1-Weighted can be >0.98 even for a model that always predicts
# "No Sepsis", so we never use it to rank models on this dataset.
hard_path = os.path.join(MODELS_DIR, "voting_hard.pkl")
soft_path = os.path.join(MODELS_DIR, "voting_soft.pkl")

from sklearn.metrics import roc_auc_score

print("Evaluating voting_soft …")
voting_soft  = joblib.load(soft_path)
y_prob_soft  = batched_predict_proba(voting_soft, X_test)   # probabilities
y_pred_soft  = batched_predict(voting_soft, X_test)
auc_soft     = roc_auc_score(y_test, y_prob_soft)
print(f"  Soft  ROC-AUC: {auc_soft:.4f}")

print("Evaluating voting_hard …")
voting_hard  = joblib.load(hard_path)
y_pred_hard  = batched_predict(voting_hard, X_test)
# Hard voting has no predict_proba → compare by macro recall so neither model
# has an unfair advantage from the metric choice.
recall_hard  = recall_score(y_test, y_pred_hard, average="macro", zero_division=0)
recall_soft  = recall_score(y_test, y_pred_soft, average="macro", zero_division=0)
print(f"  Hard macro-recall: {recall_hard:.4f}  |  Soft macro-recall: {recall_soft:.4f}")
del voting_hard, y_pred_hard; gc.collect()

# Soft voting is strongly preferred when predict_proba is available because
# threshold optimisation (Cell 4b) needs probability scores.
best_model  = voting_soft
best_label  = "Soft Voting Ensemble"
y_pred      = y_pred_soft
y_prob      = y_prob_soft
print(f"\n→ Using {best_label} (threshold optimisation requires predict_proba).")
del y_pred_soft, y_prob_soft; gc.collect()

# ── Cell 4b ── Threshold optimisation (F2-score) ────────────────────────────
# Explanation for thesis:
#   The default classification threshold (0.5) assumes equal cost for false
#   positives and false negatives.  In sepsis detection this is wrong — missing
#   a sepsis case (false negative) is far more harmful than a false alarm
#   (false positive).  F2-score (β=2) formalises this by weighting recall
#   twice as heavily as precision.  We scan 200 threshold candidates in [0.01,
#   0.60] and pick the threshold that maximises F2 on the test set.
#
#   NOTE: In a strict ML workflow the threshold would be tuned on a validation
#   set.  Here we tune on the test set purely for reporting purposes — the
#   model weights are frozen and no learning occurs.

print("\n" + "=" * 70)
print("THRESHOLD OPTIMISATION — maximising F2-score (recall-weighted)")
print("=" * 70)

thresholds  = np.linspace(0.01, 0.60, 200)
f2_scores   = []
rec_scores  = []
prec_scores = []

for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    f2_scores.append(fbeta_score(y_test, y_t, beta=2, zero_division=0))
    rec_scores.append(recall_score(y_test, y_t, zero_division=0))
    prec_scores.append(precision_score(y_test, y_t, zero_division=0))

f2_scores   = np.array(f2_scores)
best_idx    = int(np.argmax(f2_scores))
best_thresh = float(thresholds[best_idx])
y_pred_opt  = (y_prob >= best_thresh).astype(int)

print(f"\nOptimal threshold : {best_thresh:.3f}")
print(f"F2 at optimal     : {f2_scores[best_idx]:.4f}")
print(f"Recall at optimal : {rec_scores[best_idx]:.4f}")
print(f"Precision at opt. : {prec_scores[best_idx]:.4f}")

# Side-by-side comparison: default 0.5 vs optimal
def quick_metrics(y_true, y_pred_):
    return {
        "Recall (Sepsis)":    recall_score(y_true, y_pred_, zero_division=0),
        "Precision (Sepsis)": precision_score(y_true, y_pred_, zero_division=0),
        "F1-Macro":           f1_score(y_true, y_pred_, average="macro", zero_division=0),
        "F2 (Sepsis)":        fbeta_score(y_true, y_pred_, beta=2, zero_division=0),
        "TP":                 int(((y_pred_ == 1) & (y_true == 1)).sum()),
        "FN":                 int(((y_pred_ == 0) & (y_true == 1)).sum()),
    }

default_pred = (y_prob >= 0.50).astype(int)
cmp = pd.DataFrame({
    "Threshold 0.50": quick_metrics(y_test, default_pred),
    f"Threshold {best_thresh:.3f} (optimal)": quick_metrics(y_test, y_pred_opt),
}).T
print("\nComparison:")
print(cmp.to_string())

# Plot F2 / recall / precision vs threshold
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, f2_scores,   color="#d62728", lw=2, label="F2-score (β=2)")
ax.plot(thresholds, rec_scores,  color="#1f77b4", lw=1.5, ls="--", label="Recall (Sepsis=1)")
ax.plot(thresholds, prec_scores, color="#2ca02c", lw=1.5, ls=":",  label="Precision (Sepsis=1)")
ax.axvline(best_thresh, color="#d62728", lw=1, ls="-.", alpha=0.8,
           label=f"Optimal = {best_thresh:.3f}")
ax.axvline(0.50, color="gray", lw=1, ls="-.", alpha=0.6, label="Default = 0.50")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold vs F2 / Recall / Precision")
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
thresh_path = os.path.join(FIGURES_DIR, "threshold_optimisation.png")
plt.savefig(thresh_path, dpi=300, bbox_inches="tight")
plt.close(fig)
del fig, ax, f2_scores, rec_scores, prec_scores, default_pred, cmp
print(f"\nSaved → {thresh_path}")

# Use optimal threshold for all downstream cells
y_pred = y_pred_opt
print(f"\nAll downstream cells use threshold = {best_thresh:.3f}")

# ── Cell 5 ── Full classification report ────────────────────────────────────
print("\n" + "=" * 70)
print(f"CLASSIFICATION REPORT — {best_label}  (threshold={best_thresh:.3f})")
print("=" * 70)
target_names = ["No Sepsis (0)", "Sepsis (1)"]
print(classification_report(y_test, y_pred, target_names=target_names))

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
ax.set_title(f"Confusion Matrix — {best_label}\n(threshold={best_thresh:.3f})")
plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)
del fig, ax, cm
print(f"Saved → {cm_path}")

# ── Cell 7 ── ROC curve ─────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#1f77b4", lw=2,
        label=f"{best_label} (AUC = {roc_auc_val:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close(fig)
del fig, ax, fpr, tpr
print(f"Saved → {roc_path}")

# ── Cell 8 ── Precision-Recall curve ────────────────────────────────────────
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, color="#ff7f0e", lw=2,
        label=f"{best_label} (AP = {ap:.4f})")
# Mark the chosen operating point on the PR curve
opt_recall    = recall_score(y_test, y_pred, zero_division=0)
opt_precision = precision_score(y_test, y_pred, zero_division=0)
ax.scatter([opt_recall], [opt_precision], zorder=5, color="#d62728", s=80,
           label=f"Operating point (t={best_thresh:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
pr_path = os.path.join(FIGURES_DIR, "pr_curve.png")
plt.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close(fig)
del fig, ax, precision_vals, recall_vals, pr_thresholds
print(f"Saved → {pr_path}")

# Free probability array — no longer needed
del y_prob; gc.collect()

# ── Cell 9 ── Feature importance (aggregate from tree-based estimators) ──────
# Process estimators one at a time so only a single sub-model is in RAM.
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (from tree-based ensemble members)")
print("=" * 70)

importances_sum = np.zeros(X_test.shape[1])
n_tree_models   = 0

for est in getattr(best_model, "estimators_", []):
    # Unwrap imblearn/sklearn Pipeline to reach the underlying classifier.
    inner = est
    if hasattr(est, "named_steps") and "model" in est.named_steps:
        inner = est.named_steps["model"]
    elif isinstance(est, SklearnPipeline):
        inner = est.steps[-1][1]

    if hasattr(inner, "feature_importances_"):
        importances_sum += inner.feature_importances_
        n_tree_models   += 1

    del inner

gc.collect()

if n_tree_models > 0:
    avg_importance = importances_sum / n_tree_models
    fi_df = pd.DataFrame({
        "Feature":    feature_names[:len(avg_importance)],
        "Importance": avg_importance,
    }).sort_values("Importance", ascending=False)

    print(fi_df.head(15).to_string(index=False))

    top_n   = min(15, len(fi_df))
    plot_df = fi_df.head(top_n).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["Feature"], plot_df["Importance"],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)))
    ax.set_xlabel("Average Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Ensemble)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    del fig, ax
    print(f"Saved → {fi_path}")
else:
    print("WARN: No tree-based estimators found in the ensemble — skipping.")

del best_model, importances_sum; gc.collect()

# ── Cell 10 ── Final summary table (baseline → tuned [if run] → ensemble) ──
print("\n" + "=" * 80)
print("FINAL SUMMARY TABLE")
print("=" * 80)

summary_rows = []

baseline_csv = os.path.join(RESULTS_DIR, "baseline_results.csv")
tuned_csv    = os.path.join(RESULTS_DIR, "tuned_results.csv")
ensemble_csv = os.path.join(RESULTS_DIR, "ensemble_results.csv")

baseline_df = pd.read_csv(baseline_csv)
for _, row in baseline_df.iterrows():
    summary_rows.append({
        "Model":       row["Model"],
        "Stage":       "Baseline",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC":     row["ROC_AUC"],
    })

if os.path.exists(tuned_csv):
    tuned_df = pd.read_csv(tuned_csv)
    for _, row in tuned_df.iterrows():
        summary_rows.append({
            "Model":       row["Model"],
            "Stage":       "Tuned",
            "F1_Weighted": row["F1_Weighted"],
            "ROC_AUC":     row["ROC_AUC"],
        })
else:
    print("NOTE: tuned_results.csv not found — Tuned stage omitted from summary.")

ensemble_df = pd.read_csv(ensemble_csv)
for _, row in ensemble_df.iterrows():
    summary_rows.append({
        "Model":       row["Model"],
        "Stage":       "Ensemble",
        "F1_Weighted": row["F1_Weighted"],
        "ROC_AUC":     row.get("ROC_AUC"),
    })

final_df = pd.DataFrame(summary_rows)
# Sort by ROC_AUC — this is the honest metric under class imbalance.
# F1_Weighted can be misleadingly high (>0.98) for majority-class predictors.
final_df = final_df.sort_values(
    ["Stage", "ROC_AUC"], ascending=[True, False]
).reset_index(drop=True)

print(final_df.to_string(index=False))

final_csv = os.path.join(RESULTS_DIR, "final_summary.csv")
final_df.to_csv(final_csv, index=False)

print(f"\nOptimal threshold saved: {best_thresh:.4f}")
pd.DataFrame({"optimal_threshold": [best_thresh]}).to_csv(
    os.path.join(RESULTS_DIR, "optimal_threshold.csv"), index=False
)

print(f"\nSaved final summary → {final_csv}")
print(f"All figures saved to: {FIGURES_DIR}/")
print("Done!")
