# =============================================================================
# NB-02 · Preprocessing Pipeline
# =============================================================================
# Goal: Prepare train/test arrays that are ready for model input — no leakage.
#       Scaler fit on train only. SMOTE on train only.
# =============================================================================

# ── Cell 1 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Cell 2 ── Load clean.parquet ────────────────────────────────────────────
df = pd.read_parquet(os.path.join(DATA_DIR, "clean.parquet"))
print(f"Loaded clean data: {df.shape}")

# ── Cell 3 ── Separate features (X) and target (y) ─────────────────────────
# Keep target separately.
y = df["SepsisLabel"]
has_patient_id = "patient_id" in df.columns

if has_patient_id:
    # Split by patient to prevent train/test leakage across repeated time rows.
    patient_df = (
        df.groupby("patient_id", as_index=False)["SepsisLabel"]
        .max()
        .rename(columns={"SepsisLabel": "patient_label"})
    )
    train_patients, test_patients = train_test_split(
        patient_df["patient_id"],
        test_size=0.20,
        random_state=42,
        stratify=patient_df["patient_label"],
    )
    train_mask = df["patient_id"].isin(train_patients)
    test_mask = df["patient_id"].isin(test_patients)

    drop_cols = ["SepsisLabel", "patient_id"]
    X = df.drop(columns=drop_cols)
    X_train = df.loc[train_mask].drop(columns=drop_cols)
    X_test = df.loc[test_mask].drop(columns=drop_cols)
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
else:
    drop_cols = ["SepsisLabel"]
    X = df.drop(columns=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

feature_names = list(X.columns)
print(f"\nFeatures ({X.shape[1]}): {feature_names}")
print(f"Target shape: {y.shape}")
print(f"\nClass distribution before split:")
print(y.value_counts())

# Save feature names for later use
pd.Series(feature_names).to_csv(
    os.path.join(DATA_DIR, "feature_names.csv"), index=False, header=False
)

# ── Cell 4 ── Stratified 80/20 train-test split ────────────────────────────
if has_patient_id:
    print(
        f"\nPatient-level split: {train_patients.nunique()} train patients | "
        f"{test_patients.nunique()} test patients"
    )

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test  target distribution:\n{y_test.value_counts()}")

# ── Cell 5 ── StandardScaler — fit on training set ONLY ─────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✓ StandardScaler fitted on training data.")
print(f"  Train mean ≈ {X_train_scaled.mean():.6f}, std ≈ {X_train_scaled.std():.4f}")

# ── Cell 6 ── SMOTE on training set ONLY ────────────────────────────────────
print(f"\nBefore SMOTE:")
print(f"  Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE:")
print(f"  Train class distribution: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")
print(f"  Train shape: {X_train_res.shape}")

# ── Cell 7 ── Save all arrays and scaler ────────────────────────────────────
np.save(os.path.join(DATA_DIR, "X_train_pre_smote.npy"), X_train_scaled)
np.save(os.path.join(DATA_DIR, "y_train_pre_smote.npy"), y_train.values)

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train_res)
np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test_scaled)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train_res)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test.values)

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)

print(f"\n✅ Saved preprocessed data:")
print(f"   X_train.npy  →  {X_train_res.shape}")
print(f"   X_test.npy   →  {X_test_scaled.shape}")
print(f"   y_train.npy  →  {y_train_res.shape}")
print(f"   y_test.npy   →  {y_test.shape}")
print(f"   scaler.pkl   →  {scaler_path}")
