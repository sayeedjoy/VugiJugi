# =============================================================================
# NB-04 · Hyperparameter Tuning
# =============================================================================
# Goal: Tune all 12 baseline models and record the improvement.
#       GridSearchCV/RandomizedSearchCV for sklearn models,
#       Optuna (TPESampler + MedianPruner) for LightGBM, XGBoost, CatBoost.
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q catboost lightgbm xgboost optuna

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, time, json, warnings, joblib, subprocess, datetime
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")

os.makedirs(os.path.join(MODELS_DIR, "tuned"), exist_ok=True)

# ── GPU detection ────────────────────────────────────────────────────────────
def _gpu_available():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

HAS_GPU = _gpu_available()
print(f"GPU available: {HAS_GPU}")

NOTEBOOK_START = time.time()

def _check_runtime(model_name):
    elapsed = time.time() - NOTEBOOK_START
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
    print(f"  [Total elapsed: {elapsed_str}]")
    if elapsed > 3 * 3600:
        print(f"  ⚠️  WARNING: {elapsed_str} elapsed — consider saving a checkpoint.")

# ── Cell 3 ── Load preprocessed data ───────────────────────────────────────
# Use pre-SMOTE train split so CV can apply SMOTE inside each fold only.
pre_x_path = os.path.join(DATA_DIR, "X_train_pre_smote.npy")
pre_y_path = os.path.join(DATA_DIR, "y_train_pre_smote.npy")
if os.path.exists(pre_x_path) and os.path.exists(pre_y_path):
    X_train = np.load(pre_x_path)
    y_train = np.load(pre_y_path)
else:
    print(
        "WARNING: pre-SMOTE train arrays not found. "
        "Using X_train.npy/y_train.npy can leak information in CV. "
        "Rerun NB-02 to regenerate *_pre_smote.npy files."
    )
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

# ── Cell 4 ── Load baseline results for comparison ─────────────────────────
baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "baseline_results.csv"))
print(f"\nBaseline results loaded ({len(baseline_df)} models)")

# ── Cell 5 ── Import all model classes ──────────────────────────────────────
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
# Linear SVM via SGD — orders of magnitude faster than SVC on large datasets.
# CalibratedClassifierCV adds predict_proba support needed for ROC-AUC.
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ── Cell 6 ── Define search spaces (sklearn models) ─────────────────────────
# "cv" key overrides the default cv=5 for slow models (KNN, MLP).
# RandomizedSearchCV is used automatically when combos exceed 20.
gridsearch_configs = {
    "MLP": {
        "estimator": MLPClassifier(
            max_iter=500, random_state=42,
            early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
        ),
        "param_grid": {
            "hidden_layer_sizes": [(100,), (100, 50), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "adaptive"],
        },
        "cv": 3,  # early_stopping handles overfitting; cv=3 saves ~40% time
    },
    "KNN": {
        "estimator": KNeighborsClassifier(algorithm="ball_tree"),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "cv": 3,  # distance queries on ~2M post-SMOTE rows; cv=3 saves ~40%
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
        },
    },
    "HistGradientBoosting": {
        "estimator": HistGradientBoostingClassifier(random_state=42),
        "param_grid": {
            "max_iter": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_samples_leaf": [10, 20, 30],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "Bagging": {
        "estimator": BaggingClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [10, 20, 50],
            "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
        },
    },
    # GradientBoostingClassifier removed — HistGradientBoosting covers this
    # family and is significantly faster (histogram-based binning).
    "ExtraTrees": {
        "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    # SVM: SGDClassifier + CalibratedClassifierCV is O(n) vs SVC's O(n²–n³).
    # Nested param keys use "estimator__" to reach the inner SGDClassifier.
    "SVM": {
        "estimator": CalibratedClassifierCV(
            SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3,
                          random_state=42, n_jobs=-1),
            cv=3,
        ),
        "param_grid": {
            "estimator__alpha": [1e-4, 1e-3, 1e-2],
            "estimator__penalty": ["l2", "l1", "elasticnet"],
        },
    },
}

# Prefix all param keys with "model__" to target the Pipeline's model step.
for config in gridsearch_configs.values():
    config["param_grid"] = {f"model__{k}": v for k, v in config["param_grid"].items()}

# ── Cell 7 ── Run search for sklearn models ─────────────────────────────────
# Auto-selects RandomizedSearchCV (n_iter=25) when combos exceed 20,
# otherwise falls back to exhaustive GridSearchCV.
RANDOMIZED_THRESHOLD = 20
RANDOMIZED_N_ITER    = 25

all_best_params = {}
tuned_results = []

for name, config in gridsearch_configs.items():
    # Count total grid combinations
    n_combos = 1
    for v in config["param_grid"].values():
        n_combos *= len(v)

    cv      = config.get("cv", 5)
    n_iter  = min(n_combos, RANDOMIZED_N_ITER)
    use_rand = n_combos > RANDOMIZED_THRESHOLD
    search_label = (f"RandomizedSearchCV, n_iter={n_iter}, cv={cv}"
                    if use_rand else f"GridSearchCV, {n_combos} combos, cv={cv}")

    print(f"\n{'=' * 60}")
    print(f"  Tuning: {name}  ({search_label})")
    print(f"{'=' * 60}")

    start = time.time()
    tune_pipeline = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", config["estimator"]),
    ])

    if use_rand:
        gs = RandomizedSearchCV(
            estimator=tune_pipeline,
            param_distributions=config["param_grid"],
            n_iter=n_iter,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
            refit=True,
            verbose=0,
        )
    else:
        gs = GridSearchCV(
            estimator=tune_pipeline,
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

    gs.fit(X_train, y_train)
    tune_time = time.time() - start

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = (best_model.predict_proba(X_test)[:, 1]
              if hasattr(best_model, "predict_proba") else y_pred.astype(float))

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc   = roc_auc_score(y_test, y_prob)

    print(f"  Best params: {gs.best_params_}")
    print(f"  F1 Weighted: {f1_w:.4f}  |  ROC AUC: {roc:.4f}  |  Time: {tune_time:.0f}s")

    all_best_params[name] = {
        k.replace("model__", "", 1): (v if not isinstance(v, np.generic) else v.item())
        for k, v in gs.best_params_.items()
    }

    tuned_results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision_W": round(prec, 4), "Recall_W": round(rec, 4),
        "F1_Weighted": round(f1_w, 4), "F1_Macro": round(f1_m, 4),
        "ROC_AUC": round(roc, 4), "Tune_Time_Sec": round(tune_time, 1),
    })

    joblib.dump(best_model, os.path.join(MODELS_DIR, "tuned", f"{name}.pkl"))
    _check_runtime(name)

# ── Cell 8 ── Optuna tuning for LightGBM, XGBoost, CatBoost ────────────────
# Uses TPESampler(seed=42) for reproducibility (required by project spec).
# MedianPruner kills bad trials after 1-2 folds — saves ~40-60% of Optuna time.
# Manual StratifiedKFold loop enables intermediate value reporting for pruning.
# SMOTE is applied inside each fold (anti-leakage invariant preserved).
OPTUNA_N_TRIALS = 30
CV_FOLDS        = 5

def _make_smote_pipeline(model):
    return Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", model),
    ])

def _cv_with_pruning(trial, model):
    """Run stratified k-fold CV, reporting intermediate values for pruning."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = []
    for step, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_tr, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_tr, y_fold_val = y_train[train_idx], y_train[val_idx]

        pipe = _make_smote_pipeline(model)
        pipe.fit(X_fold_tr, y_fold_tr)
        y_pred = pipe.predict(X_fold_val)
        scores.append(f1_score(y_fold_val, y_pred, average="weighted", zero_division=0))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))

def optuna_lgbm(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    if HAS_GPU:
        params["device"] = "gpu"
    return _cv_with_pruning(trial, LGBMClassifier(**params))

def optuna_xgb(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42, "n_jobs": -1, "eval_metric": "logloss",
    }
    if HAS_GPU:
        params["tree_method"] = "gpu_hist"
        params["device"] = "cuda"
    return _cv_with_pruning(trial, XGBClassifier(**params))

def optuna_catboost(trial):
    params = {
        "iterations":   trial.suggest_int("iterations", 100, 500),
        "depth":        trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_state": 42, "verbose": 0,
    }
    if HAS_GPU:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    return _cv_with_pruning(trial, CatBoostClassifier(**params))

optuna_configs = {
    "LightGBM":  (optuna_lgbm,     LGBMClassifier),
    "XGBoost":   (optuna_xgb,      XGBClassifier),
    "CatBoost":  (optuna_catboost,  CatBoostClassifier),
}

for name, (objective_fn, ModelClass) in optuna_configs.items():
    print(f"\n{'=' * 60}")
    print(f"  Tuning: {name}  (Optuna, {OPTUNA_N_TRIALS} trials, cv={CV_FOLDS})")
    print(f"{'=' * 60}")

    start = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective_fn, n_trials=OPTUNA_N_TRIALS, n_jobs=1)
    tune_time = time.time() - start

    best_params = study.best_params
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1 Weighted: {study.best_value:.4f}  |  Pruned trials: {n_pruned}")

    # Fixed params for final refit
    fixed = {"random_state": 42}
    if name == "LightGBM":
        fixed.update({"n_jobs": -1, "verbose": -1})
        if HAS_GPU:
            fixed["device"] = "gpu"
    elif name == "XGBoost":
        fixed.update({"n_jobs": -1, "eval_metric": "logloss"})
        if HAS_GPU:
            fixed.update({"tree_method": "gpu_hist", "device": "cuda"})
    elif name == "CatBoost":
        fixed.update({"verbose": 0})
        if HAS_GPU:
            fixed.update({"task_type": "GPU", "devices": "0"})

    best_model = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", ModelClass(**best_params, **fixed)),
    ])
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc   = roc_auc_score(y_test, y_prob)

    print(f"  F1 Weighted: {f1_w:.4f}  |  ROC AUC: {roc:.4f}  |  Time: {tune_time:.0f}s")

    all_best_params[name] = {k: (v if not isinstance(v, np.generic) else v.item())
                              for k, v in best_params.items()}

    tuned_results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision_W": round(prec, 4), "Recall_W": round(rec, 4),
        "F1_Weighted": round(f1_w, 4), "F1_Macro": round(f1_m, 4),
        "ROC_AUC": round(roc, 4), "Tune_Time_Sec": round(tune_time, 1),
    })

    joblib.dump(best_model, os.path.join(MODELS_DIR, "tuned", f"{name}.pkl"))
    _check_runtime(name)

# ── Cell 9 ── Save results and best_params ──────────────────────────────────
tuned_df = pd.DataFrame(tuned_results)
tuned_df = tuned_df.sort_values("F1_Weighted", ascending=False).reset_index(drop=True)

csv_path = os.path.join(RESULTS_DIR, "tuned_results.csv")
tuned_df.to_csv(csv_path, index=False)

params_path = os.path.join(RESULTS_DIR, "best_params.json")
with open(params_path, "w") as f:
    json.dump(all_best_params, f, indent=2)

print("\n" + "=" * 80)
print("TUNED RESULTS — sorted by F1 Weighted")
print("=" * 80)
print(tuned_df.to_string(index=False))

# ── Cell 10 ── Before / After comparison ────────────────────────────────────
comparison = baseline_df[["Model", "F1_Weighted", "ROC_AUC"]].merge(
    tuned_df[["Model", "F1_Weighted", "ROC_AUC"]],
    on="Model", suffixes=("_Baseline", "_Tuned"),
)
comparison["F1_Improvement"] = comparison["F1_Weighted_Tuned"] - comparison["F1_Weighted_Baseline"]
comparison["AUC_Improvement"] = comparison["ROC_AUC_Tuned"] - comparison["ROC_AUC_Baseline"]
comparison = comparison.sort_values("F1_Improvement", ascending=False)

print("\n" + "=" * 80)
print("BASELINE vs TUNED — Improvement")
print("=" * 80)
print(comparison.to_string(index=False))

total_elapsed = str(datetime.timedelta(seconds=int(time.time() - NOTEBOOK_START)))
print(f"\n✅ Saved tuned results  → {csv_path}")
print(f"✅ Saved best params    → {params_path}")
print(f"✅ Total runtime        → {total_elapsed}")
