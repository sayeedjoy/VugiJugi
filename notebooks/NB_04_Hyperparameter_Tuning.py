# =============================================================================
# NB-04 · Hyperparameter Tuning
# =============================================================================
# Goal: Tune all 13 baseline models and record the improvement.
#       GridSearchCV for small search spaces, Optuna for large ones
#       (LightGBM, XGBoost, CatBoost).
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q catboost lightgbm xgboost optuna

# ── Cell 2 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, time, json, warnings, joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")

os.makedirs(os.path.join(MODELS_DIR, "tuned"), exist_ok=True)

# ── Cell 3 ── Load preprocessed data ───────────────────────────────────────
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
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
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

# ── Cell 6 ── Define GridSearchCV search spaces (sklearn models) ────────────
gridsearch_configs = {
    "MLP": {
        "estimator": MLPClassifier(max_iter=500, random_state=42),
        "param_grid": {
            "hidden_layer_sizes": [(100,), (100, 50), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "adaptive"],
        },
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
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
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
    "ExtraTrees": {
        "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "SVM": {
        "estimator": SVC(random_state=42, probability=True),
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
}

# ── Cell 7 ── Run GridSearchCV for sklearn models ──────────────────────────
all_best_params = {}
tuned_results = []

for name, config in gridsearch_configs.items():
    print(f"\n{'=' * 55}")
    print(f"  Tuning: {name}  (GridSearchCV, cv=5)")
    print(f"{'=' * 55}")

    start = time.time()
    gs = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
        refit=True,
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

    all_best_params[name] = {k: (v if not isinstance(v, np.generic) else v.item())
                              for k, v in gs.best_params_.items()}

    tuned_results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Precision_W": round(prec, 4), "Recall_W": round(rec, 4),
        "F1_Weighted": round(f1_w, 4), "F1_Macro": round(f1_m, 4),
        "ROC_AUC": round(roc, 4), "Tune_Time_Sec": round(tune_time, 1),
    })

    joblib.dump(best_model, os.path.join(MODELS_DIR, "tuned", f"{name}.pkl"))

# ── Cell 8 ── Optuna tuning for LightGBM, XGBoost, CatBoost ────────────────
OPTUNA_N_TRIALS = 50

def optuna_lgbm(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
        "max_depth":      trial.suggest_int("max_depth", 3, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":     trial.suggest_int("num_leaves", 20, 150),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="f1_weighted", n_jobs=-1)
    return scores.mean()

def optuna_xgb(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
        "max_depth":      trial.suggest_int("max_depth", 3, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":          trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42, "n_jobs": -1,
        "use_label_encoder": False, "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="f1_weighted", n_jobs=-1)
    return scores.mean()

def optuna_catboost(trial):
    params = {
        "iterations":     trial.suggest_int("iterations", 100, 500),
        "depth":          trial.suggest_int("depth", 3, 10),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count":   trial.suggest_int("border_count", 32, 255),
        "random_state": 42, "verbose": 0,
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring="f1_weighted", n_jobs=-1)
    return scores.mean()

optuna_configs = {
    "LightGBM":  (optuna_lgbm,    LGBMClassifier),
    "XGBoost":   (optuna_xgb,     XGBClassifier),
    "CatBoost":  (optuna_catboost, CatBoostClassifier),
}

for name, (objective_fn, ModelClass) in optuna_configs.items():
    print(f"\n{'=' * 55}")
    print(f"  Tuning: {name}  (Optuna, {OPTUNA_N_TRIALS} trials)")
    print(f"{'=' * 55}")

    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=OPTUNA_N_TRIALS, n_jobs=1)
    tune_time = time.time() - start

    best_params = study.best_params
    print(f"  Best params: {best_params}")
    print(f"  Best CV F1 Weighted: {study.best_value:.4f}")

    # Fixed params
    fixed = {"random_state": 42}
    if name == "LightGBM":
        fixed.update({"n_jobs": -1, "verbose": -1})
    elif name == "XGBoost":
        fixed.update({"n_jobs": -1, "use_label_encoder": False, "eval_metric": "logloss"})
    elif name == "CatBoost":
        fixed.update({"verbose": 0})

    best_model = ModelClass(**best_params, **fixed)
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

print(f"\n✅ Saved tuned results  → {csv_path}")
print(f"✅ Saved best params    → {params_path}")
