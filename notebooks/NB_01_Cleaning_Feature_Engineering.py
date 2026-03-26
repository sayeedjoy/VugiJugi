# =============================================================================
# NB-01 · Data Cleaning & Feature Engineering
# =============================================================================
# Goal: Remove low-quality columns, fill missing values, and create
#       domain-specific features (Shock Index, Pulse Pressure).
# =============================================================================

# ── Cell 1 ── Mount Drive & imports ─────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np

THESIS_DIR = "/content/drive/MyDrive/Thesis"
DATA_DIR   = os.path.join(THESIS_DIR, "data")

# ── Cell 2 ── Load raw.parquet ──────────────────────────────────────────────
df = pd.read_parquet(os.path.join(DATA_DIR, "raw.parquet"))
print(f"Loaded raw data: {df.shape}")
print(f"Columns ({df.shape[1]}): {list(df.columns)}")

# ── Cell 3 ── Drop high-missingness columns (> 40%) ────────────────────────
MISS_THRESHOLD = 0.40                  # 40 %
null_pct = df.isnull().mean()
cols_to_drop = null_pct[null_pct > MISS_THRESHOLD].index.tolist()

# Never drop the target or patient_id
cols_to_drop = [c for c in cols_to_drop if c not in ["SepsisLabel", "patient_id"]]

print(f"\nMissingness threshold : {MISS_THRESHOLD * 100:.0f}%")
print(f"Columns to drop ({len(cols_to_drop)}):")
for c in cols_to_drop:
    print(f"  - {c:20s}  ({null_pct[c] * 100:.1f}% missing)")

# Save a record of dropped columns
dropped_path = os.path.join(DATA_DIR, "dropped_columns.txt")
with open(dropped_path, "w") as fh:
    fh.write(f"Missingness threshold: {MISS_THRESHOLD * 100:.0f}%\n\n")
    for c in cols_to_drop:
        fh.write(f"{c}: {null_pct[c] * 100:.1f}% missing\n")
print(f"\nSaved dropped-columns record → {dropped_path}")

before_cols = df.shape[1]
df.drop(columns=cols_to_drop, inplace=True)
after_cols = df.shape[1]
print(f"\nColumn count: {before_cols} → {after_cols}  (dropped {before_cols - after_cols})")

# ── Cell 4 ── Fill remaining numeric nulls with median ──────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
null_before = df[numeric_cols].isnull().sum().sum()

for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

null_after = df[numeric_cols].isnull().sum().sum()
print(f"\nNumeric nulls filled: {null_before:,} → {null_after:,}")

# ── Cell 5 ── Feature engineering ───────────────────────────────────────────
# Shock Index  = HR / SBP  (higher → worse perfusion)
# Pulse Pressure = SBP − DBP (narrowing indicates shock)

if "HR" in df.columns and "SBP" in df.columns:
    df["ShockIndex"] = df["HR"] / df["SBP"].replace(0, np.nan)
    df["ShockIndex"].fillna(df["ShockIndex"].median(), inplace=True)
    print(f"\n✓ ShockIndex created  — range: [{df['ShockIndex'].min():.2f}, {df['ShockIndex'].max():.2f}]")
else:
    print("\n⚠ HR or SBP missing — ShockIndex not created")

if "SBP" in df.columns and "DBP" in df.columns:
    df["PulsePressure"] = df["SBP"] - df["DBP"]
    df["PulsePressure"].fillna(df["PulsePressure"].median(), inplace=True)
    print(f"✓ PulsePressure created — range: [{df['PulsePressure'].min():.2f}, {df['PulsePressure'].max():.2f}]")
else:
    print("⚠ SBP or DBP missing — PulsePressure not created")

# ── Cell 6 ── Validate ──────────────────────────────────────────────────────
# Check for remaining NaNs, infs, unexpected ranges
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

remaining_nulls = df.isnull().sum()
if remaining_nulls.sum() == 0:
    print("✓ No null values remain.")
else:
    print("⚠ Remaining nulls:")
    print(remaining_nulls[remaining_nulls > 0])

inf_check = np.isinf(df.select_dtypes(include=[np.number])).sum()
if inf_check.sum() == 0:
    print("✓ No infinite values detected.")
else:
    print("⚠ Infinite values found:")
    print(inf_check[inf_check > 0])
    # Replace infinities with column median
    for col in inf_check[inf_check > 0].index:
        median_val = df.loc[np.isfinite(df[col]), col].median()
        df[col].replace([np.inf, -np.inf], median_val, inplace=True)
    print("  → Replaced inf values with column medians.")

print(f"\nFinal shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ── Cell 7 ── Save clean.parquet ────────────────────────────────────────────
output_path = os.path.join(DATA_DIR, "clean.parquet")
df.to_parquet(output_path, index=False)
print(f"\n✅ Saved clean data to: {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
