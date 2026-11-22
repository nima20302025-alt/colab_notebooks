# ============================================================
# EV S6E9 - TOP COMBO TE + LIGHTGBM
# ============================================================

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
# TOP COMBINATIONS FROM SEARCH
# ============================================================

COMBOS = [

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
    ),

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ),

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Home_Charging_Possible",
    ),

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "City_Type",
    ),

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Number_of_Cars_Owned",
    ),

    (
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Gender",
    ),
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(TRAIN_PATH)

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

y = target_raw.map(target_map).astype(np.int8)


# ============================================================
# BASE FEATURES
# ============================================================

DROP_COLS = [
    TARGET,
    ID_COL
]

X = df.drop(columns=DROP_COLS).copy()


# ============================================================
# CONVERT CATEGORICALS
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

def make_oof_te(data, target, combo, folds, smoothing=10):

    oof = np.zeros(len(data), dtype=np.float64)

    global_mean = target.mean()

    for train_idx, valid_idx in folds:

        train_part = data.iloc[train_idx]
        valid_part = data.iloc[valid_idx]

        train_y = target.iloc[train_idx]

        temp = train_part[list(combo)].copy()
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
# LIGHTGBM PARAMETERS
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


# ============================================================
# BASELINE + COMBO EXPERIMENTS
# ============================================================

results = []


for combo_id, combo in enumerate(COMBOS, 1):

    print("\n" + "=" * 70)
    print(f"COMBO {combo_id}/{len(COMBOS)}")
    print(" + ".join(combo))
    print("=" * 70)

    oof_te = make_oof_te(
        X,
        y,
        combo,
        folds,
        smoothing=SMOOTHING
    )

    X_exp = X.copy()

    te_name = (
        "TE_"
        + "_".join(combo)
    )

    X_exp[te_name] = oof_te

    fold_scores = []

    oof_pred = np.zeros(len(X_exp))

    for fold_id, (train_idx, valid_idx) in enumerate(folds, 1):

        model = LGBMClassifier(**params)

        model.fit(
            X_exp.iloc[train_idx],
            y.iloc[train_idx],

            categorical_feature=[
                c for c in categorical_cols
                if c in X_exp.columns
            ]
        )

        pred = model.predict_proba(
            X_exp.iloc[valid_idx]
        )[:, 1]

        oof_pred[valid_idx] = pred

        fold_auc = roc_auc_score(
            y.iloc[valid_idx],
            pred
        )

        fold_scores.append(fold_auc)

        print(
            f"Fold {fold_id}: "
            f"{fold_auc:.6f}"
        )

    overall_auc = roc_auc_score(
        y,
        oof_pred
    )

    print(
        f"\nOOF AUC: "
        f"{overall_auc:.6f}"
    )

    print(
        f"Mean fold: "
        f"{np.mean(fold_scores):.6f}"
    )

    print(
        f"Std fold: "
        f"{np.std(fold_scores):.6f}"
    )

    results.append({
        "combo": " + ".join(combo),
        "auc": overall_auc,
        "mean_fold": np.mean(fold_scores),
        "std_fold": np.std(fold_scores)
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "auc",
    ascending=False
)

print("\n\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    "top_combo_lgbm_results.csv",
    index=False
)

print("\nSaved:")
print("top_combo_lgbm_results.csv")