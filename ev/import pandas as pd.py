import pandas as pd
import numpy as np
import os
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

CHUNK_SIZE = 250_000
SAMPLE_SIZE = 400_000

TARGET = "Will_Buy_EV"

# ============================================================
# 1. CHECK FILE
# ============================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Could not find {TRAIN_PATH}. "
        f"Put this script in the same folder as train.csv."
    )

print("=" * 80)
print("REVERSE ENGINEERING - EV PURCHASE DATASET")
print("=" * 80)

file_size_mb = os.path.getsize(TRAIN_PATH) / (1024 ** 2)

print(f"\nFile size: {file_size_mb:.2f} MB")
print("Reading dataset in chunks...")

# ============================================================
# 2. READ DATA IN CHUNKS
# ============================================================

samples = []
total_rows = 0
target_counts = Counter()

for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    # Target distribution
    if TARGET in chunk.columns:
        target_counts.update(
            chunk[TARGET].value_counts().to_dict()
        )

    # Take sample from each chunk
    remaining = SAMPLE_SIZE - sum(len(x) for x in samples)

    if remaining > 0:
        take = min(remaining, len(chunk))

        # deterministic sample
        samples.append(
            chunk.sample(
                n=take,
                random_state=42
            )
        )

    print(
        f"Processed: {total_rows:,} rows | "
        f"Sample: {sum(len(x) for x in samples):,}"
    )

    if sum(len(x) for x in samples) >= SAMPLE_SIZE:
        # We already have enough sample
        break

df = pd.concat(samples, ignore_index=True)
# Convert target to numeric
# ============================================================
# TARGET CONVERSION: Yes / No -> 1 / 0
# ============================================================

print("\nTarget values BEFORE conversion:")
print(df[TARGET].value_counts(dropna=False))

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

# Check conversion
print("\nTarget values AFTER conversion:")
print(df[TARGET].value_counts(dropna=False))

# Make sure conversion was successful
if df[TARGET].isna().any():
    bad_values = df.loc[
        df[TARGET].isna(),
        TARGET
    ]

    raise ValueError(
        "Unknown target values found after conversion."
    )

df[TARGET] = df[TARGET].astype("int8")

print("\nTarget dtype:", df[TARGET].dtype)
print("Target mean:", df[TARGET].mean())

print("\n" + "=" * 80)
print("BASIC DATASET INFORMATION")
print("=" * 80)

print(f"Sample rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")

# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

if TARGET in df.columns:

    print(df[TARGET].value_counts())

    print("\nTarget ratio:")
    print(
        df[TARGET]
        .value_counts(normalize=True)
        .round(5)
    )

# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(df.dtypes)

# ============================================================
# 5. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 80)
print("UNIQUE VALUE ANALYSIS")
print("=" * 80)

for col in df.columns:

    if col == TARGET:
        continue

    nunique = df[col].nunique(dropna=False)

    ratio = nunique / len(df)

    print(
        f"{col:40s} "
        f"unique={nunique:10,d} "
        f"unique_ratio={ratio:.5f}"
    )

# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = df.isna().sum()

for col, value in missing.items():

    if value > 0:
        print(
            f"{col:40s} "
            f"{value:,} "
            f"({value / len(df) * 100:.2f}%)"
        )

# ============================================================
# 7. NUMERIC COLUMN ANALYSIS
# ============================================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

print("\n" + "=" * 80)
print("NUMERIC COLUMN ANALYSIS")
print("=" * 80)

for col in numeric_cols:

    if col == TARGET:
        continue

    s = df[col].dropna()

    print("\n" + "-" * 70)
    print(col)

    print(f"min       : {s.min()}")
    print(f"max       : {s.max()}")
    print(f"mean      : {s.mean():.6f}")
    print(f"median    : {s.median():.6f}")
    print(f"std       : {s.std():.6f}")
    print(f"nunique   : {s.nunique():,}")

    # Decimal analysis
    if pd.api.types.is_float_dtype(s):

        decimal_part = np.abs(s - np.floor(s))

        print(
            f"decimal_unique : "
            f"{decimal_part.nunique():,}"
        )

        print(
            f"integer_unique : "
            f"{np.floor(s).nunique():,}"
        )

# ============================================================
# 8. INTEGER / DIGIT ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("DIGIT / MODULO ANALYSIS")
print("=" * 80)

for col in numeric_cols:

    if col == TARGET:
        continue

    s = df[col].dropna()

    # Only analyze columns that are reasonably numeric
    if len(s) == 0:
        continue

    print("\n" + "-" * 70)
    print(col)

    # Convert to integer representation
    values = np.floor(s).astype(np.int64)

    for mod in [2, 3, 5, 7, 9, 10, 11, 100]:

        unique_mod = values.mod(mod).nunique()

        print(
            f"mod {mod:3d}: "
            f"unique={unique_mod:5d}"
        )

    # Last digit distribution
    last_digit = values.mod(10)

    print("\nLast digit distribution:")

    print(
        last_digit
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_dict()
    )

# ============================================================
# 9. FREQUENCY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FREQUENCY / REPEATED VALUE ANALYSIS")
print("=" * 80)

for col in df.columns:

    if col == TARGET:
        continue

    counts = df[col].value_counts(dropna=False)

    repeated = (counts > 1).sum()

    max_frequency = counts.iloc[0]

    print(
        f"{col:40s} "
        f"repeated_values={repeated:,} "
        f"max_frequency={max_frequency:,}"
    )

# ============================================================
# 10. EXACT VALUE -> TARGET RELATIONSHIP
# ============================================================

print("\n" + "=" * 80)
print("EXACT VALUE TARGET SIGNAL")
print("=" * 80)

if TARGET in df.columns:

    global_mean = df[TARGET].mean()

    print(f"\nGlobal target mean: {global_mean:.6f}")

    results = []

    for col in df.columns:

        if col == TARGET:
            continue

        # Only useful if repeated values exist
        stats = (
            df.groupby(col, dropna=False)[TARGET]
            .agg(
                count="count",
                mean="mean"
            )
        )

        # Only groups repeated at least 2 times
        repeated = stats[stats["count"] >= 2]

        if len(repeated) == 0:
            continue

        # weighted variance from global mean
        signal = np.average(
            np.abs(
                repeated["mean"] - global_mean
            ),
            weights=repeated["count"]
        )

        results.append(
            {
                "feature": col,
                "groups": len(stats),
                "repeated_groups": len(repeated),
                "signal": signal,
                "max_count": stats["count"].max()
            }
        )

    if results:

        signal_df = pd.DataFrame(results)

        signal_df = signal_df.sort_values(
            "signal",
            ascending=False
        )

        print(
            signal_df.to_string(
                index=False
            )
        )

# ============================================================
# 11. CATEGORICAL TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CATEGORICAL TARGET SIGNAL")
print("=" * 80)

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

for col in categorical_cols:

    if col == TARGET:
        continue

    stats = (
        df.groupby(col, dropna=False)[TARGET]
        .agg(
            count="count",
            target_mean="mean"
        )
        .sort_values(
            "target_mean",
            ascending=False
        )
    )

    print("\n" + "-" * 70)
    print(col)

    print(
        stats.head(15).to_string()
    )

# ============================================================
# 12. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 80)
print("NUMERIC CORRELATION WITH TARGET")
print("=" * 80)

if TARGET in df.columns:

    correlations = []

    for col in numeric_cols:

        if col == TARGET:
            continue

        corr = df[col].corr(
            df[TARGET],
            method="spearman"
        )

        correlations.append(
            (col, corr)
        )

    correlations.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for col, corr in correlations:

        print(
            f"{col:40s} "
            f"Spearman={corr:.6f}"
        )

# ============================================================
# 13. POSSIBLE GENERATOR FINGERPRINTS
# ============================================================

print("\n" + "=" * 80)
print("POSSIBLE GENERATOR FINGERPRINTS")
print("=" * 80)

for col in numeric_cols:

    if col == TARGET:
        continue

    s = df[col].dropna()

    if len(s) == 0:
        continue

    values = np.floor(s).astype(np.int64)

    features = {
        "last_digit": values % 10,
        "last_2_digits": values % 100,
        "mod_3": values % 3,
        "mod_7": values % 7,
        "mod_9": values % 9,
        "mod_11": values % 11,
    }

    print("\n" + "-" * 70)
    print(col)

    for name, feature in features.items():

        try:
            corr = np.corrcoef(
                feature,
                df.loc[s.index, TARGET]
            )[0, 1]

            print(
                f"{name:20s} "
                f"corr={corr:.6f}"
            )

        except:
            pass

# ============================================================
# 14. TOP VALUE / TARGET TABLE
# ============================================================

print("\n" + "=" * 80)
print("TOP EXACT VALUES BY TARGET RATE")
print("=" * 80)

for col in numeric_cols:

    if col == TARGET:
        continue

    stats = (
        df.groupby(col)[TARGET]
        .agg(
            count="count",
            target_mean="mean"
        )
    )

    # Avoid tiny groups
    stats = stats[
        stats["count"] >= 10
    ]

    if len(stats) == 0:
        continue

    stats["distance"] = (
        stats["target_mean"]
        - df[TARGET].mean()
    ).abs()

    stats = stats.sort_values(
        "distance",
        ascending=False
    )

    print("\n" + "-" * 70)
    print(col)

    print(
        stats.head(10).to_string()
    )

# ============================================================
# 15. SAVE SAMPLE
# ============================================================

sample_path = "reverse_engineering_sample.csv"

df.to_csv(
    sample_path,
    index=False
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nSaved sample to: {sample_path}"
)

print(
    "\nSend me the COMPLETE TERMINAL OUTPUT."
)

print(
    "\nEspecially send me these sections:"
)

print("""
1. UNIQUE VALUE ANALYSIS
2. NUMERIC COLUMN ANALYSIS
3. DIGIT / MODULO ANALYSIS
4. EXACT VALUE TARGET SIGNAL
5. CATEGORICAL TARGET SIGNAL
6. NUMERIC CORRELATION WITH TARGET
7. POSSIBLE GENERATOR FINGERPRINTS
8. TOP EXACT VALUES BY TARGET RATE
""")