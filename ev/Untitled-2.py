# ============================================================
# EV S6E9
# 4-WAY OOF TARGET ENCODING + MULTI-TE LIGHTGBM
# ============================================================

import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42
SMOOTHING = 10


# ============================================================
# CANDIDATE FEATURES
# ============================================================

CANDIDATE_COLS = [
    "Environmental_Concern_Level",
    "Subsidy_Available",
    "Home_Charging_Possible",
    "Number_of_Cars_Owned",
    "Range_Anxiety_Level",
    "City_Type",
    "Gender",
    "Current_Car_Type",
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
    raise ValueError(
        f"Unknown target values:\n"
        f"{target_raw[y.isna()].value_counts()}"
    )

y = y.astype(np.int8)

print("Positive rate:", y.mean())


# ============================================================
# BASE X
# ============================================================

X = df.drop(
    columns=[TARGET, ID_COL]
).copy()


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype("category")


# ============================================================
# FOLDS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

folds = list(skf.split(X, y))


# ============================================================
# OOF TARGET ENCODING
# ============================================================

def make_oof_te(
    data,
    target,
    combo,
    folds,
    smoothing=10
):

    oof = np.zeros(
        len(data),
        dtype=np.float64
    )

    global_mean = target.mean()

    for train_idx, valid_idx in folds:

        train_part = data.iloc[train_idx]
        valid_part = data.iloc[valid_idx]

        train_y = target.iloc[train_idx]

        temp = train_part[
            list(combo)
        ].copy()

        temp["_target_"] = train_y.values

        stats = (
            temp
            .groupby(
                list(combo),
                observed=True
            )["_target_"]
            .agg(["mean", "count"])
            .reset_index()
        )

        # Smoothed TE
        stats["te"] = (
            (
                stats["count"] * stats["mean"]
                +
                smoothing * global_mean
            )
            /
            (
                stats["count"]
                +
                smoothing
            )
        )

        valid_keys = valid_part[
            list(combo)
        ].merge(
            stats[
                list(combo) + ["te"]
            ],
            on=list(combo),
            how="left"
        )

        oof[valid_idx] = (
            valid_keys["te"]
            .fillna(global_mean)
            .values
        )

    return oof


# ============================================================
# COMBINATION LIST
# ============================================================

# فقط ترکیب‌هایی که Env + Subsidy را دارند
# چون آزمایش قبلی نشان دادند هسته اصلی سیگنال هستند.

EXTRA_COLS = [
    "Home_Charging_Possible",
    "Number_of_Cars_Owned",
    "Range_Anxiety_Level",
    "City_Type",
    "Gender",
    "Current_Car_Type",
]


COMBOS = []

# 2-way
COMBOS.append(
    (
        "Environmental_Concern_Level",
        "Subsidy_Available"
    )
)

# 3-way
for col in EXTRA_COLS:

    COMBOS.append(
        (
            "Environmental_Concern_Level",
            "Subsidy_Available",
            col
        )
    )


# 4-way
for col1, col2 in itertools.combinations(
    EXTRA_COLS,
    2
):

    COMBOS.append(
        (
            "Environmental_Concern_Level",
            "Subsidy_Available",
            col1,
            col2
        )
    )


print("\n" + "=" * 70)
print("COMBINATIONS")
print("=" * 70)

for i, combo in enumerate(COMBOS, 1):

    print(
        f"{i:2d}. "
        + " + ".join(combo)
    )

print("\nTotal combinations:", len(COMBOS))


# ============================================================
# CREATE ALL OOF TE FEATURES
# ============================================================

TE_FEATURES = {}
TE_RESULTS = []


for i, combo in enumerate(COMBOS, 1):

    print(
        f"\n[{i}/{len(COMBOS)}] "
        f"Creating TE:"
    )

    print(
        " + ".join(combo)
    )

    te = make_oof_te(
        X,
        y,
        combo,
        folds,
        smoothing=SMOOTHING
    )

    name = (
        "TE_"
        + "_".join(combo)
    )

    TE_FEATURES[name] = te

    te_auc = roc_auc_score(
        y,
        te
    )

    TE_RESULTS.append({
        "feature": name,
        "combo": " + ".join(combo),
        "n_features": len(combo),
        "te_auc": te_auc,
        "unique": np.unique(te).size
    })

    print(
        f"TE AUC: {te_auc:.6f}"
    )


# ============================================================
# TE SUMMARY
# ============================================================

te_results_df = pd.DataFrame(
    TE_RESULTS
).sort_values(
    "te_auc",
    ascending=False
)

print("\n" + "=" * 70)
print("TE RANKING")
print("=" * 70)

print(
    te_results_df.to_string(
        index=False
    )
)


# ============================================================
# BASELINE LIGHTGBM
# ============================================================

params = {
    "n_estimators": 700,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": -1,

    "subsample": 0.85,
    "colsample_bytree": 0.85,

    "reg_alpha": 0.1,
    "reg_lambda": 1.0,

    "random_state": SEED,
    "n_jobs": -1,

    "verbosity": -1
}


def run_lgbm(
    X_model,
    name
):

    print("\n" + "=" * 70)
    print("MODEL:", name)
    print("=" * 70)

    oof = np.zeros(len(X_model))

    fold_scores = []

    model_cats = [
        c for c in categorical_cols
        if c in X_model.columns
    ]

    for fold_id, (
        train_idx,
        valid_idx
    ) in enumerate(folds, 1):

        model = LGBMClassifier(
            **params
        )

        model.fit(
            X_model.iloc[train_idx],
            y.iloc[train_idx],
            categorical_feature=model_cats
        )

        pred = model.predict_proba(
            X_model.iloc[valid_idx]
        )[:, 1]

        oof[valid_idx] = pred

        auc = roc_auc_score(
            y.iloc[valid_idx],
            pred
        )

        fold_scores.append(auc)

        print(
            f"Fold {fold_id}: "
            f"{auc:.6f}"
        )

    overall = roc_auc_score(
        y,
        oof
    )

    print(
        f"\nOOF AUC: {overall:.6f}"
    )

    print(
        f"Mean fold: "
        f"{np.mean(fold_scores):.6f}"
    )

    print(
        f"Std fold: "
        f"{np.std(fold_scores):.6f}"
    )

    return overall, oof


# ============================================================
# BASELINE
# ============================================================

baseline_auc, baseline_oof = run_lgbm(
    X,
    "BASELINE"
)


# ============================================================
# SINGLE BEST TE
# ============================================================

best_single = te_results_df.iloc[0]

best_te_name = best_single["feature"]

X_single = X.copy()

X_single[best_te_name] = TE_FEATURES[
    best_te_name
]

single_auc, single_oof = run_lgbm(
    X_single,
    "BASE + BEST SINGLE TE"
)


# ============================================================
# MULTI-TE EXPERIMENTS
# ============================================================

# چند تعداد مختلف TE را تست می‌کنیم
TOP_K_VALUES = [2, 3, 4, 5, 6]


multi_results = []


for k in TOP_K_VALUES:

    selected = (
        te_results_df
        .head(k)
        ["feature"]
        .tolist()
    )

    print("\n" + "=" * 70)
    print(
        f"MULTI-TE TOP {k}"
    )
    print("=" * 70)

    print("\n".join(selected))

    X_multi = X.copy()

    for feature_name in selected:

        X_multi[feature_name] = (
            TE_FEATURES[feature_name]
        )

    auc, oof = run_lgbm(
        X_multi,
        f"BASE + TOP {k} TE"
    )

    multi_results.append({
        "model": f"TOP_{k}_TE",
        "auc": auc
    })


# ============================================================
# FINAL RESULTS
# ============================================================

final_results = [
    {
        "model": "BASELINE",
        "auc": baseline_auc
    },
    {
        "model": "BEST_SINGLE_TE",
        "auc": single_auc
    }
]

final_results.extend(
    multi_results
)

final_df = pd.DataFrame(
    final_results
).sort_values(
    "auc",
    ascending=False
)

print("\n\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    final_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

te_results_df.to_csv(
    "four_way_te_ranking.csv",
    index=False
)

final_df.to_csv(
    "multi_te_lgbm_results.csv",
    index=False
)

print("\nSaved:")
print("four_way_te_ranking.csv")
print("multi_te_lgbm_results.csv")


# ============================================================
# BEST RESULT
# ============================================================

best = final_df.iloc[0]

print("\n" + "=" * 70)
print("BEST RESULT")
print("=" * 70)

print(
    f"Model: {best['model']}"
)

print(
    f"OOF AUC: {best['auc']:.6f}"
)

print(
    f"Previous best: 0.941795"
)

print(
    f"Improvement: "
    f"{best['auc'] - 0.941795:+.6f}"
)

print("\nDone.")