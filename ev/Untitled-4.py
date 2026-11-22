# ============================================================
# EV PURCHASE — DIGIT FEATURE ABLATION
# ============================================================
# هدف:
# پیدا کردن اینکه کدام Digit Feature واقعاً به AUC کمک می‌کند.
#
# آزمایش‌ها:
# 1) BASELINE
# 2) هر Digit Feature به صورت جداگانه
# 3) BASE + ALL DIGITS
#
# مدل:
# LightGBM
#
# Validation:
# 5-Fold Stratified OOF
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42

# اگر GPU داری True کن
USE_GPU = False


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Train file not found:\n{TRAIN_PATH}\n\n"
        "TRAIN_PATH را به مسیر صحیح train.csv تغییر بده."
    )

df = pd.read_csv(TRAIN_PATH)

print(f"Shape: {df.shape}")


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

df[TARGET] = target_raw.map(target_map)

if df[TARGET].isna().any():
    bad_values = target_raw[df[TARGET].isna()].unique()
    raise ValueError(
        f"Unknown target values found: {bad_values}"
    )

y = df[TARGET].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

print(f"\nPositive rate: {y.mean():.6f}")


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

missing_base = [
    c for c in BASE_FEATURES
    if c not in df.columns
]

if missing_base:
    raise ValueError(
        f"Missing base columns:\n{missing_base}"
    )

X_raw = df[BASE_FEATURES].copy()


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


# ============================================================
# PREPARE BASE DATA
# ============================================================

def prepare_features(X):
    X = X.copy()

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


# ============================================================
# DIGIT FEATURE ENGINEERING
# ============================================================

def add_digit_features(X):
    """
    Add all 8 digit/modulo features.
    """

    X = X.copy()

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    income = X["Annual_Income_USD"].astype(np.int64)

    X["Income_LastDigit"] = income % 10
    X["Income_Last2Digits"] = income % 100
    X["Income_Mod3"] = income % 3
    X["Income_Mod7"] = income % 7
    X["Income_Mod9"] = income % 9

    # --------------------------------------------------------
    # Commute
    # --------------------------------------------------------

    # Daily_Commute_km may be float
    commute_int = (
        X["Daily_Commute_km"]
        .round()
        .astype(np.int64)
    )

    X["Commute_LastDigit"] = commute_int % 10
    X["Commute_Last2Digits"] = commute_int % 100
    X["Commute_Mod3"] = commute_int % 3

    return X


# ============================================================
# LIST OF DIGIT FEATURES
# ============================================================

DIGIT_FEATURES = [
    "Income_LastDigit",
    "Income_Last2Digits",
    "Income_Mod3",
    "Income_Mod7",
    "Income_Mod9",
    "Commute_LastDigit",
    "Commute_Last2Digits",
    "Commute_Mod3",
]


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 30,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

if USE_GPU:
    LGB_PARAMS["device"] = "gpu"


# ============================================================
# RUN OOF EXPERIMENT
# ============================================================

def run_experiment(name, X):
    """
    Run 5-fold OOF LightGBM and return:
        overall AUC
        mean fold AUC
        std fold AUC
        number of features
    """

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    X = prepare_features(X)

    print(f"Features: {X.shape[1]}")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    oof_pred = np.zeros(len(X))

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(f"\nFold {fold}/{N_SPLITS}")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = LGBMClassifier(
            **LGB_PARAMS
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=[
                c for c in CATEGORICAL_COLS
                if c in X.columns
            ],
            eval_set=[
                (X_valid, y_valid)
            ],
            callbacks=[]
        )

        pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof_pred[valid_idx] = pred

        fold_auc = roc_auc_score(
            y_valid,
            pred
        )

        fold_scores.append(fold_auc)

        print(
            f"Fold {fold} AUC: "
            f"{fold_auc:.6f}"
        )

    overall_auc = roc_auc_score(
        y,
        oof_pred
    )

    mean_fold = np.mean(fold_scores)
    std_fold = np.std(fold_scores)

    print("\n" + "-" * 50)
    print(f"Overall OOF AUC : {overall_auc:.6f}")
    print(f"Mean Fold AUC   : {mean_fold:.6f}")
    print(f"Std Fold AUC    : {std_fold:.6f}")
    print("-" * 50)

    return {
        "experiment": name,
        "auc": overall_auc,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "n_features": X.shape[1],
    }


# ============================================================
# EXPERIMENTS
# ============================================================

results = []


# ------------------------------------------------------------
# 1. BASELINE
# ------------------------------------------------------------

baseline_X = X_raw.copy()

results.append(
    run_experiment(
        "BASELINE",
        baseline_X
    )
)


# ------------------------------------------------------------
# 2. EACH DIGIT FEATURE SEPARATELY
# ------------------------------------------------------------

X_all_digits = add_digit_features(X_raw)

for digit_feature in DIGIT_FEATURES:

    X_single = X_raw.copy()

    X_single[digit_feature] = (
        X_all_digits[digit_feature]
    )

    results.append(
        run_experiment(
            f"BASE + {digit_feature}",
            X_single
        )
    )


# ------------------------------------------------------------
# 3. ALL DIGIT FEATURES
# ------------------------------------------------------------

results.append(
    run_experiment(
        "BASE + ALL DIGITS",
        X_all_digits
    )
)


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

baseline_auc = results_df.loc[
    results_df["experiment"] == "BASELINE",
    "auc"
].iloc[0]

results_df["gain_vs_baseline"] = (
    results_df["auc"] - baseline_auc
)

results_df = results_df.sort_values(
    "auc",
    ascending=False
).reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n")
print("=" * 70)
print("DIGIT FEATURE ABLATION RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "experiment",
            "auc",
            "mean_fold",
            "std_fold",
            "gain_vs_baseline",
            "n_features",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST SINGLE DIGIT FEATURE
# ============================================================

single_results = results_df[
    results_df["experiment"].str.startswith(
        "BASE + "
    )
    &
    ~results_df["experiment"].isin(
        ["BASE + ALL DIGITS"]
    )
]

if len(single_results) > 0:

    best_single = single_results.iloc[0]

    print("\n" + "=" * 70)
    print("BEST SINGLE DIGIT FEATURE")
    print("=" * 70)

    print(
        f"Feature : {best_single['experiment']}"
    )

    print(
        f"AUC     : {best_single['auc']:.6f}"
    )

    print(
        f"Gain    : "
        f"{best_single['gain_vs_baseline']:+.6f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "digit_feature_ablation.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_FILE}"
)

print("\nDone.")