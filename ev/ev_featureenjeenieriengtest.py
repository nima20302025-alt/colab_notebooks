# ============================================================
# EV S6E9 - OOF TARGET ENCODING COMBINATION SEARCH
# Leakage-Safe Search for 2-way and 3-way categorical combos
# ============================================================

import os
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42

SMOOTHING = 10

# ویژگی‌های مناسب برای ترکیب
CANDIDATE_COLS = [
    "Environmental_Concern_Level",
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Home_Charging_Possible",
    "City_Type",
    "Current_Car_Type",
    "Gender",
    "Number_of_Cars_Owned",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(TRAIN_PATH)

print("Shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

target_raw = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_map = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
}

y = target_raw.map(target_map)

if y.isna().any():
    print("Unknown target values:")
    print(target_raw[y.isna()].value_counts())
    raise ValueError("Target conversion failed.")

y = y.astype(np.int8)

print("Positive rate:", y.mean())


# ============================================================
# CHECK CANDIDATES
# ============================================================

missing_cols = [c for c in CANDIDATE_COLS if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")


print("\nCandidate columns:")
for col in CANDIDATE_COLS:
    print(
        f"{col:35s} "
        f"unique={df[col].nunique():4d}"
    )


# ============================================================
# FOLDS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

folds = list(skf.split(df, y))


# ============================================================
# OOF TARGET ENCODING
# ============================================================

def make_oof_te(data, target, combo, folds, smoothing=10):

    oof = np.zeros(len(data), dtype=np.float64)

    global_mean = target.mean()

    for fold_id, (train_idx, valid_idx) in enumerate(folds):

        train_part = data.iloc[train_idx]
        valid_part = data.iloc[valid_idx]

        train_y = target.iloc[train_idx]

        # ----------------------------------------------------
        # Build temporary dataframe
        # ----------------------------------------------------

        temp = train_part[list(combo)].copy()
        temp["_target_"] = train_y.values

        stats = (
            temp
            .groupby(list(combo), observed=True)["_target_"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # ----------------------------------------------------
        # Smoothed target encoding
        #
        # TE = (count * group_mean + smoothing * global_mean)
        #      / (count + smoothing)
        # ----------------------------------------------------

        stats["te"] = (
            (
                stats["count"] * stats["mean"]
                + smoothing * global_mean
            )
            /
            (
                stats["count"] + smoothing
            )
        )

        # ----------------------------------------------------
        # Map validation rows
        # ----------------------------------------------------

        valid_keys = valid_part[list(combo)].merge(
            stats[list(combo) + ["te"]],
            on=list(combo),
            how="left"
        )

        encoded = valid_keys["te"].fillna(global_mean).values

        oof[valid_idx] = encoded

    return oof


# ============================================================
# SEARCH
# ============================================================

results = []

all_combos = []

# 2-way
for combo in itertools.combinations(CANDIDATE_COLS, 2):
    all_combos.append(combo)

# 3-way
for combo in itertools.combinations(CANDIDATE_COLS, 3):
    all_combos.append(combo)


print("\n" + "=" * 70)
print("COMBINATION SEARCH")
print("=" * 70)

print("Total combinations:", len(all_combos))
print("2-way:", len(list(itertools.combinations(CANDIDATE_COLS, 2))))
print("3-way:", len(list(itertools.combinations(CANDIDATE_COLS, 3))))


for i, combo in enumerate(all_combos, 1):

    print(
        f"\r[{i:3d}/{len(all_combos)}] "
        f"{' + '.join(combo)}",
        end=""
    )

    oof_te = make_oof_te(
        df,
        y,
        combo,
        folds,
        smoothing=SMOOTHING
    )

    auc = roc_auc_score(y, oof_te)

    # additional diagnostics
    unique_te = np.unique(oof_te).size

    results.append({
        "combo": " + ".join(combo),
        "n_features": len(combo),
        "auc": auc,
        "unique_TE": unique_te,
        "min_TE": oof_te.min(),
        "max_TE": oof_te.max(),
        "mean_TE": oof_te.mean(),
        "std_TE": oof_te.std(),
    })


print("\n")


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "auc",
    ascending=False
).reset_index(drop=True)

results_df["gain_vs_global"] = (
    results_df["auc"] - 0.5
)


# ============================================================
# TOP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP COMBINATIONS")
print("=" * 70)

print(
    results_df[
        [
            "combo",
            "n_features",
            "auc",
            "unique_TE",
            "min_TE",
            "max_TE",
            "std_TE"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# BEST 2-WAY
# ============================================================

best_2way = (
    results_df[
        results_df["n_features"] == 2
    ]
    .head(15)
)

print("\n" + "=" * 70)
print("BEST 2-WAY COMBINATIONS")
print("=" * 70)

print(
    best_2way[
        ["combo", "auc", "unique_TE"]
    ].to_string(index=False)
)


# ============================================================
# BEST 3-WAY
# ============================================================

best_3way = (
    results_df[
        results_df["n_features"] == 3
    ]
    .head(15)
)

print("\n" + "=" * 70)
print("BEST 3-WAY COMBINATIONS")
print("=" * 70)

print(
    best_3way[
        ["combo", "auc", "unique_TE"]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "combo_te_search_results.csv",
    index=False
)

print("\nSaved:")
print("combo_te_search_results.csv")


# ============================================================
# COMPARE WITH CURRENT BEST
# ============================================================

current_combo = (
    "Environmental_Concern_Level + "
    "Subsidy_Available + "
    "Range_Anxiety_Level"
)

current_row = results_df[
    results_df["combo"] == current_combo
]

print("\n" + "=" * 70)
print("CURRENT COMBO CHECK")
print("=" * 70)

if len(current_row):
    print(
        current_row[
            ["combo", "auc", "unique_TE"]
        ].to_string(index=False)
    )

print("\nDone.")