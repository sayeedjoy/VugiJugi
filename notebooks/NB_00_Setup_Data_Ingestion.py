# =============================================================================
# NB-00 · Setup & Data Ingestion
# =============================================================================
# Goal: Mount Drive, extract the dataset, load the raw CSV files,
#       run a quick sanity check, and persist the raw data for all
#       downstream notebooks.
# =============================================================================

# ── Cell 1 ── Install dependencies ──────────────────────────────────────────
# !pip install -q pyarrow tqdm

# ── Cell 2 ── Mount Google Drive ────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 3 ── Extract archive.zip ───────────────────────────────────────────
import zipfile, os

THESIS_DIR  = "/content/drive/MyDrive/Thesis"
DATASET_ZIP = os.path.join(THESIS_DIR, "dataset", "archive.zip")
EXTRACT_DIR = "/content"

DATA_DIR    = os.path.join(THESIS_DIR, "data")
MODELS_DIR  = os.path.join(THESIS_DIR, "models")
RESULTS_DIR = os.path.join(THESIS_DIR, "results")
FIGURES_DIR = os.path.join(THESIS_DIR, "figures")

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Extracting {DATASET_ZIP} ...")
with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
    z.extractall(EXTRACT_DIR)
print("Extraction complete.")

# ── Cell 4 ── Load all PSV patient files into one DataFrame ─────────────────
import glob
import pandas as pd
from tqdm import tqdm

files  = glob.glob("/content/training_setA/training/*.psv")
files += glob.glob("/content/training_setA/training_setA/*.psv")
files += glob.glob("/content/training_setB/training_setB/*.psv")

# De-duplicate in case both structures exist
files = sorted(set(files))
print(f"Total patient files found: {len(files)}")

df_list = []
for f in tqdm(files, desc="Loading patient files"):
    patient_df = pd.read_csv(f, sep="|")
    patient_df["patient_id"] = os.path.basename(f)
    df_list.append(patient_df)

df = pd.concat(df_list, ignore_index=True)
print(f"\nCombined dataset shape: {df.shape}")

# ── Cell 5 ── Sanity checks ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# 5a. Shape & dtypes
print(f"\nRows   : {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")

# 5b. Null counts
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print(f"\nNull percentage per column:\n{null_pct.to_string()}")

# 5c. Class distribution
print("\n" + "-" * 40)
print("Target class distribution (SepsisLabel):")
print("-" * 40)
class_counts = df["SepsisLabel"].value_counts()
class_pct    = df["SepsisLabel"].value_counts(normalize=True) * 100
dist = pd.DataFrame({"Count": class_counts, "Percent": class_pct.round(2)})
print(dist)
print(f"\nImbalance ratio (neg/pos): {class_counts[0] / class_counts[1]:.1f} : 1")

# 5d. Unique patients
print(f"\nUnique patients: {df['patient_id'].nunique()}")

# ── Cell 6 ── Save raw.parquet ──────────────────────────────────────────────
output_path = os.path.join(DATA_DIR, "raw.parquet")
df.to_parquet(output_path, index=False)
print(f"\n✅ Saved raw data to: {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
