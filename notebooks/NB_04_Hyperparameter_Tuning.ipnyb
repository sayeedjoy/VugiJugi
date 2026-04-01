# =============================================================================
# NB-04 · Hyperparameter Tuning  (optimised — target runtime ~25-35 min on A100)
# =============================================================================
# Goal: Tune all 12 baseline models and record the improvement.
#       GridSearchCV/HalvingRandomSearchCV for sklearn models,
#       Optuna (TPESampler + MedianPruner, n_jobs=2) for LightGBM, XGBoost, CatBoost.
#
# Speed-up summary vs original:
#   1. float32 arrays          → halves SMOTE memory, faster distance ops   (1.2-1.4×)
#   2. 25% CV subsample        → 4× fewer rows in every search fold          (3-5×, KNN 15-23×)
#   3. Tighter param grids     → 3.5× fewer CV fits for sklearn models        (3.5×)
#   4. CV folds 5→3 globally   → proportional reduction in fits              (1.67×)
#   5. HalvingRandomSearchCV   → successive halving for ensemble models       (2-3× extra)
#   6. Optuna 30→20 trials, n_jobs=2 → parallel trial execution             (1.5-2×)
#   7. Early stopping (boosting) → terminates wasted tree iterations         (1.3-1.5×/trial)
#   8. Checkpoint/skip system  → crash-safe; restarts from last done model   (prevents full reruns)
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
import lightgbm as lgb

from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV,
    StratifiedKFold, StratifiedShuffleSplit,
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

# ── Checkpoint helpers ───────────────────────────────────────────────────────
# Saves progress after each model so Colab session crashes don't lose work.
TUNED_DIR           = os.path.join(MODELS_DIR, "tuned")
CHECKPOINT_PARAMS   = os.path.join(RESULTS_DIR, "best_params_checkpoint.json")
CHECKPOINT_RESULTS  = os.path.join(RESULTS_DIR, "tuned_results_checkpoint.csv")

def _load_checkpoint():
    params, results = {}, []
    if os.path.exists(CHECKPOINT_PARAMS):
        with open(CHECKPOINT_PARAMS) as f:
            params = json.load(f)
        print(f"  [Checkpoint] {len(params)} previously tuned params loaded.")
    if os.path.exists(CHECKPOINT_RESULTS):
        results = pd.read_csv(CHECKPOINT_RESULTS).to_dict("records")
        print(f"  [Checkpoint] {len(results)} previously tuned results loaded.")
    return params, results

def _save_checkpoint(all_best_params, tuned_results):
    with open(CHECKPOINT_PARAMS, "w") as f:
        json.dump(all_best_params, f, indent=2)
    pd.DataFrame(tuned_results).to_csv(CHECKPOINT_RESULTS, index=False)

def _model_is_done(name):
    """Return True if this model's pkl already exists from a prior run."""
    return os.path.exists(os.path.join(TUNED_DIR, f"{name}.pkl"))

# ── Cell 3 ── Load preprocessed data ───────────────────────────────────────
# Use pre-SMOTE train split so CV can apply SMOTE inside each fold only.
# Cast to float32: halves memory (~240 MB → ~120 MB) and speeds up SMOTE k-NN.
pre_x_path = os.path.join(DATA_DIR, "X_train_pre_smote.npy")
pre_y_path = os.path.join(DATA_DIR, "y_train_pre_smote.npy")
if os.path.exists(pre_x_path) and os.path.exists(pre_y_path):
    X_train = np.load(pre_x_path).astype(np.float32)
    y_train = np.load(pre_y_path)
else:
    print(
        "WARNING: pre-SMOTE train arrays not found. "
        "Using X_train.npy/y_train.npy can leak information in CV. "
        "Rerun NB-02 to regenerate *_pre_smote.npy files."
    )
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"X_train: {X_train.shape}  dtype: {X_train.dtype}  "
      f"mem: {X_train.nbytes / 1e6:.0f} MB")
print(f"y_train class balance: {y_train.mean():.2%} positive")

# ── Cell 3b ── Stratified subsample for search phase ────────────────────────
# Hyperparameter SEARCH runs on 25% of X_train (faster per CV fold).
# Final model REFIT uses full X_train (full data quality for saved model).
# Anti-leakage: SMOTE stays inside Pipeline — applied only on fold train data.
SEARCH_SUBSAMPLE_FRAC = 0.25

_sss = StratifiedShuffleSplit(
    n_splits=1, test_size=(1 - SEARCH_SUBSAMPLE_FRAC), random_state=42
)
search_idx, _ = next(_sss.split(X_train, y_train))
X_search = X_train[search_idx]
y_search = y_train[search_idx]

print(f"\nSearch subsample: {X_search.shape}  "
      f"({y_search.mean():.2%} positive — stratification preserved)")

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

# ── Cell 6 ── Tuning constants ───────────────────────────────────────────────
CV_FOLDS             = 3   # was 5; 3 folds on 25% subsample is statistically sound
OPTUNA_N_TRIALS      = 20  # was 30; TPE converges fast with MedianPruner
RANDOMIZED_THRESHOLD = 20
RANDOMIZED_N_ITER    = 20  # was 25

# Models that benefit from HalvingRandomSearchCV (have n_estimators as a
# meaningful "resource" proxy; successive halving eliminates weak candidates early).
HALVING_MODELS = {"HistGradientBoosting", "RandomForest", "Bagging", "ExtraTrees", "AdaBoost"}

# ── Cell 7 ── Define search spaces (sklearn models) ─────────────────────────
# Grids tightened vs original:
#   MLP:             24 → 4  combos  (relu+adaptive dominate on SMOTE-balanced data)
#   KNN:             16 → 12 combos  (removed k=3; too noisy for 2% imbalance)
#   DecisionTree:    72 → 12 combos  (removed max_depth=None and entropy)
#   AdaBoost:        12 → 9  combos  (removed lr=0.01; too slow for n_est≤200)
#   HistGB:          81 → 16 combos  (removed max_iter=100, max_depth=7, lr=0.01)
#   RandomForest:    36 → 16 combos  (removed n_est=100, removed max_depth=None)
#   Bagging:         27 → 8  combos  (removed low n_est/max_samples values)
#   ExtraTrees:      36 → 16 combos  (same as RandomForest)
#   SVM:             9  → 6  combos  (removed l1; elasticnet covers that space)
gridsearch_configs = {
    "MLP": {
        "estimator": MLPClassifier(
            max_iter=500, random_state=42,
            early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
        ),
        "param_grid": {
            "hidden_layer_sizes": [(100, 50), (128, 64)],
            "activation": ["relu"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["adaptive"],
        },
        # cv omitted → uses global CV_FOLDS
    },
    "KNN": {
        "estimator": KNeighborsClassifier(algorithm="ball_tree"),
        "param_grid": {
            "n_neighbors": [5, 7, 11],      # k=3 removed — too noisy for 2% minority
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        # cv omitted → uses global CV_FOLDS
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [5, 10, 20],           # None removed — leads to overfitting on SMOTE
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "criterion": ["gini"],              # entropy removed — equivalent, slower
        },
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1.0],  # 0.01 removed — too slow for n_est≤200
        },
    },
    "HistGradientBoosting": {
        "estimator": HistGradientBoostingClassifier(random_state=42),
        "param_grid": {
            "max_iter": [200, 300],             # 100 removed — too few for ICU complexity
            "max_depth": [3, 5],                # 7 removed — excessive for histogram boosting
            "learning_rate": [0.05, 0.1],       # 0.01 removed — too slow
            "min_samples_leaf": [10, 20],       # 30 removed
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [200, 300],         # 100 removed — baseline minimum, rarely optimal
            "max_depth": [10, 20],              # None removed — overfits on SMOTE data
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "Bagging": {
        "estimator": BaggingClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [20, 50],           # 10 removed — too few
            "max_samples": [0.7, 1.0],          # 0.5 removed — unbalanced bootstrap with SMOTE
            "max_features": [0.7, 1.0],         # 0.5 removed
        },
    },
    "ExtraTrees": {
        "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [200, 300],
            "max_depth": [10, 20],
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
            "estimator__penalty": ["l2", "elasticnet"],  # l1 removed; elasticnet covers it
        },
    },
}

# Prefix all param keys with "model__" to target the Pipeline's model step.
for config in gridsearch_configs.values():
    config["param_grid"] = {f"model__{k}": v for k, v in config["param_grid"].items()}

# ── Cell 8 ── Run search for sklearn models ─────────────────────────────────
# Search on X_search (25% subsample), then refit best params on full X_train.
# HalvingRandomSearchCV used for ensemble models (successive halving on n_samples).
# Other models: RandomizedSearchCV if combos > 20, else GridSearchCV.
all_best_params, tuned_results = _load_checkpoint()

for name, config in gridsearch_configs.items():
    if _model_is_done(name):
        print(f"\n  [SKIP] {name} — tuned model already exists.")
        continue

    n_combos = 1
    for v in config["param_grid"].values():
        n_combos *= len(v)

    cv       = config.get("cv", CV_FOLDS)
    n_iter   = min(n_combos, RANDOMIZED_N_ITER)
    use_rand = n_combos > RANDOMIZED_THRESHOLD
    use_halv = name in HALVING_MODELS

    if use_halv:
        search_label = f"HalvingRandomSearchCV, factor=3, cv={cv}"
    elif use_rand:
        search_label = f"RandomizedSearchCV, n_iter={n_iter}, cv={cv}"
    else:
        search_label = f"GridSearchCV, {n_combos} combos, cv={cv}"

    print(f"\n{'=' * 60}")
    print(f"  Tuning: {name}  ({search_label})")
    print(f"{'=' * 60}")

    start = time.time()
    tune_pipeline = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", config["estimator"]),
    ])

    if use_halv:
        # min_resources=50_000 ensures ~1K minority samples per CV training fold
        # after SMOTE, which is safely above k_neighbors=3 threshold.
        gs = HalvingRandomSearchCV(
            estimator=tune_pipeline,
            param_distributions=config["param_grid"],
            factor=3,
            resource="n_samples",
            min_resources=50_000,
            aggressive_elimination=True,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
            refit=False,   # manual refit on full X_train below
            verbose=0,
        )
    elif use_rand:
        gs = RandomizedSearchCV(
            estimator=tune_pipeline,
            param_distributions=config["param_grid"],
            n_iter=n_iter,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
            refit=False,   # manual refit on full X_train below
            verbose=0,
        )
    else:
        gs = GridSearchCV(
            estimator=tune_pipeline,
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            refit=False,   # manual refit on full X_train below
            verbose=0,
        )

    # Search on 25% subsample — fast
    gs.fit(X_search, y_search)
    search_time = time.time() - start

    # Refit best params on FULL X_train for production-quality saved model
    best_params_raw = gs.best_params_
    final_pipeline = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", clone(config["estimator"]).set_params(
            **{k.replace("model__", "", 1): v for k, v in best_params_raw.items()}
        )),
    ])
    refit_start = time.time()
    final_pipeline.fit(X_train, y_train)
    tune_time = search_time + (time.time() - refit_start)

    best_model = final_pipeline
    y_pred = best_model.predict(X_test)
    y_prob = (best_model.predict_proba(X_test)[:, 1]
              if hasattr(best_model, "predict_proba") else y_pred.astype(float))

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc   = roc_auc_score(y_test, y_prob)

    print(f"  Best params: {best_params_raw}")
    print(f"  F1 Weighted: {f1_w:.4f}  |  ROC AUC: {roc:.4f}  |  "
          f"Search: {search_time:.0f}s  Total: {tune_time:.0f}s")

    all_best_params[name] = {
        k.replace("model__", "", 1): (v if not isinstance(v, np.generic) else v.item())
        for k, v in best_params_raw.items()
    }

    tuned_results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision_W": round(prec, 4), "Recall_W": round(rec, 4),
        "F1_Weighted": round(f1_w, 4), "F1_Macro": round(f1_m, 4),
        "ROC_AUC": round(roc, 4), "Tune_Time_Sec": round(tune_time, 1),
    })

    joblib.dump(best_model, os.path.join(TUNED_DIR, f"{name}.pkl"))
    _save_checkpoint(all_best_params, tuned_results)
    _check_runtime(name)

# ── Cell 9 ── Optuna tuning for LightGBM, XGBoost, CatBoost ─────────────────
# Uses TPESampler(seed=42) for reproducibility (required by project spec).
# MedianPruner kills bad trials after 1-2 folds — saves ~40-60% of Optuna time.
# n_jobs=2: two trials run concurrently (GPU handles tree-level parallelism).
# Early stopping inside each trial further reduces wasted tree iterations.
# SMOTE applied inside each fold (anti-leakage invariant preserved).
# Search on X_search (25% subsample); final model refit on full X_train.
#
# Note on reproducibility: n_jobs=2 makes trial ORDER non-deterministic, so
# best_params may differ slightly from a sequential run. Document in thesis.

def _cv_with_pruning_boosting(trial, model, model_name, X_data, y_data):
    """Stratified k-fold CV with per-fold SMOTE and early stopping.

    SMOTE is called explicitly (not via Pipeline) so eval_set can be passed
    to the underlying booster for early stopping.
    Anti-leakage: SMOTE runs only on X_fold_tr; eval_set is X_fold_val (clean).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=3)
    scores = []

    for step, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        # SMOTE on training fold only — validation fold stays original (anti-leakage)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)

        if model_name == "LightGBM":
            model.fit(
                X_res, y_res,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
        elif model_name == "XGBoost":
            model.fit(
                X_res, y_res,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif model_name == "CatBoost":
            model.fit(
                X_res, y_res,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                verbose=False,
            )

        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def optuna_lgbm(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 400),      # was 100-500
        "max_depth":         trial.suggest_int("max_depth", 3, 8),             # was 3-10
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.2, log=True),  # was 0.01-0.3
        "num_leaves":        trial.suggest_int("num_leaves", 20, 100),         # was 20-150
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),   # was 5-50
        "random_state": 42,
        "n_jobs": 1,    # GPU handles tree parallelism; CPU n_jobs causes contention
        "verbose": -1,
    }
    if HAS_GPU:
        params["device"] = "gpu"
    return _cv_with_pruning_boosting(trial, LGBMClassifier(**params), "LightGBM",
                                     X_search, y_search)


def optuna_xgb(trial):
    params = {
        "n_estimators":        trial.suggest_int("n_estimators", 100, 400),    # was 100-500
        "max_depth":           trial.suggest_int("max_depth", 3, 8),           # was 3-10
        "learning_rate":       trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample":           trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":    trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":    trial.suggest_int("min_child_weight", 1, 10),
        "gamma":               trial.suggest_float("gamma", 0.0, 2.0),         # was 0.0-5.0
        "early_stopping_rounds": 20,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "logloss",
    }
    if HAS_GPU:
        params["tree_method"] = "hist"   # XGBoost ≥1.7: use 'hist' + device='cuda'
        params["device"] = "cuda"
    return _cv_with_pruning_boosting(trial, XGBClassifier(**params), "XGBoost",
                                     X_search, y_search)


def optuna_catboost(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 400),            # was 100-500
        "depth":         trial.suggest_int("depth", 3, 8),                     # was 3-10
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count":  trial.suggest_int("border_count", 32, 255),
        "random_state": 42,
        "verbose": 0,
    }
    if HAS_GPU:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    return _cv_with_pruning_boosting(trial, CatBoostClassifier(**params), "CatBoost",
                                     X_search, y_search)


optuna_configs = {
    "LightGBM":  (optuna_lgbm,     LGBMClassifier),
    "XGBoost":   (optuna_xgb,      XGBClassifier),
    "CatBoost":  (optuna_catboost,  CatBoostClassifier),
}

for name, (objective_fn, ModelClass) in optuna_configs.items():
    if _model_is_done(name):
        print(f"\n  [SKIP] {name} — tuned model already exists.")
        continue

    print(f"\n{'=' * 60}")
    print(f"  Tuning: {name}  (Optuna, {OPTUNA_N_TRIALS} trials, cv={CV_FOLDS}, n_jobs=2)")
    print(f"{'=' * 60}")

    start = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # n_jobs=2: two trials run concurrently.
    # SMOTE instances are per-fold-per-trial (not shared) → thread-safe.
    study.optimize(objective_fn, n_trials=OPTUNA_N_TRIALS, n_jobs=2)
    search_time = time.time() - start

    best_params = study.best_params
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1 Weighted: {study.best_value:.4f}  |  Pruned trials: {n_pruned}")

    # Fixed params for final refit
    fixed = {"random_state": 42}
    if name == "LightGBM":
        fixed.update({"n_jobs": -1, "verbose": -1})   # n_jobs=-1 safe for single-model refit
        if HAS_GPU:
            fixed["device"] = "gpu"
    elif name == "XGBoost":
        fixed.update({"n_jobs": -1, "eval_metric": "logloss"})
        if HAS_GPU:
            fixed.update({"tree_method": "hist", "device": "cuda"})
    elif name == "CatBoost":
        fixed.update({"verbose": 0})
        if HAS_GPU:
            fixed.update({"task_type": "GPU", "devices": "0"})

    # Refit on FULL X_train (no early_stopping_rounds — use all estimators for best model)
    best_params_for_refit = {k: v for k, v in best_params.items()
                             if k != "early_stopping_rounds"}
    best_model = Pipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", ModelClass(**best_params_for_refit, **fixed)),
    ])
    refit_start = time.time()
    best_model.fit(X_train, y_train)
    tune_time = search_time + (time.time() - refit_start)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc   = roc_auc_score(y_test, y_prob)

    print(f"  F1 Weighted: {f1_w:.4f}  |  ROC AUC: {roc:.4f}  |  "
          f"Search: {search_time:.0f}s  Total: {tune_time:.0f}s")

    all_best_params[name] = {k: (v if not isinstance(v, np.generic) else v.item())
                              for k, v in best_params.items()}

    tuned_results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision_W": round(prec, 4), "Recall_W": round(rec, 4),
        "F1_Weighted": round(f1_w, 4), "F1_Macro": round(f1_m, 4),
        "ROC_AUC": round(roc, 4), "Tune_Time_Sec": round(tune_time, 1),
    })

    joblib.dump(best_model, os.path.join(TUNED_DIR, f"{name}.pkl"))
    _save_checkpoint(all_best_params, tuned_results)
    _check_runtime(name)

# ── Cell 10 ── Save results and best_params ─────────────────────────────────
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

# ── Cell 11 ── Before / After comparison ────────────────────────────────────
comparison = baseline_df[["Model", "F1_Weighted", "ROC_AUC"]].merge(
    tuned_df[["Model", "F1_Weighted", "ROC_AUC"]],
    on="Model", suffixes=("_Baseline", "_Tuned"),
)
comparison["F1_Improvement"]  = comparison["F1_Weighted_Tuned"] - comparison["F1_Weighted_Baseline"]
comparison["AUC_Improvement"] = comparison["ROC_AUC_Tuned"]    - comparison["ROC_AUC_Baseline"]
comparison = comparison.sort_values("F1_Improvement", ascending=False)

print("\n" + "=" * 80)
print("BASELINE vs TUNED — Improvement")
print("=" * 80)
print(comparison.to_string(index=False))

total_elapsed = str(datetime.timedelta(seconds=int(time.time() - NOTEBOOK_START)))
print(f"\n✅ Saved tuned results  → {csv_path}")
print(f"✅ Saved best params    → {params_path}")
print(f"✅ Total runtime        → {total_elapsed}")
