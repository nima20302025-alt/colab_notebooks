# ============================================================
# EV PURCHASE — INCOME DECOMPOSITION ABLATION
# ============================================================
# هدف:
# بررسی دقیق ساختار عددی Annual_Income_USD
#
# تست می‌کنیم:
#   - Last 1 / 2 / 3 digits
#   - Modulo features
#   - Individual decimal digits
#   - Income scale features
#
# Validation:
#   5-Fold Stratified OOF
#
# Model:
#   LightGBM
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

N_SPLITS = 5
SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"File not found: {TRAIN_PATH}"
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

y = target_raw.map(target_map)

if y.isna().any():
    bad = target_raw[y.isna()].unique()

    raise ValueError(
        f"Unknown target values: {bad}"
    )

y = y.astype(int)

print("\nTarget:")
print(y.value_counts())

print(
    f"Positive rate: {y.mean():.6f}"
)


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

X_base = df[BASE_FEATURES].copy()


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
# PREPARE FEATURES
# ============================================================

def prepare_features(X):

    X = X.copy()

    for col in CATEGORICAL_COLS:

        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


# ============================================================
# INCOME FEATURE ENGINEERING
# ============================================================

def create_income_features(X):

    X = X.copy()

    income = (
        X["Annual_Income_USD"]
        .round()
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # Last digits
    # --------------------------------------------------------

    X["Income_LastDigit"] = income % 10

    X["Income_Last2Digits"] = income % 100

    X["Income_Last3Digits"] = income % 1000

    # --------------------------------------------------------
    # Modulo features
    # --------------------------------------------------------

    X["Income_Mod2"] = income % 2
    X["Income_Mod3"] = income % 3
    X["Income_Mod4"] = income % 4
    X["Income_Mod5"] = income % 5
    X["Income_Mod6"] = income % 6
    X["Income_Mod7"] = income % 7
    X["Income_Mod8"] = income % 8
    X["Income_Mod9"] = income % 9
    X["Income_Mod10"] = income % 10
    X["Income_Mod11"] = income % 11
    X["Income_Mod12"] = income % 12

    # --------------------------------------------------------
    # Decimal digits
    # --------------------------------------------------------

    X["Income_TensDigit"] = (income // 10) % 10

    X["Income_HundredsDigit"] = (
        income // 100
    ) % 10

    X["Income_ThousandsDigit"] = (
        income // 1000
    ) % 10

    X["Income_10ThousandsDigit"] = (
        income // 10000
    ) % 10

    # --------------------------------------------------------
    # Scale / decomposition
    # --------------------------------------------------------

    X["Income_Thousands"] = (
        income // 1000
    )

    X["Income_Hundreds"] = (
        income // 100
    )

    X["Income_Tens"] = (
        income // 10
    )

    # --------------------------------------------------------
    # Additional useful decomposition
    # --------------------------------------------------------

    X["Income_Thousands_Remainder"] = (
        income % 1000
    )

    X["Income_Hundreds_Remainder"] = (
        income % 100
    )

    return X


# ============================================================
# CREATE ALL INCOME FEATURES
# ============================================================

X_income_all = create_income_features(
    X_base
)


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    # Original best feature
    "LAST_2_DIGITS": [
        "Income_Last2Digits"
    ],

    "LAST_DIGIT": [
        "Income_LastDigit"
    ],

    "LAST_3_DIGITS": [
        "Income_Last3Digits"
    ],

    # Modulo
    "MOD_2": [
        "Income_Mod2"
    ],

    "MOD_3": [
        "Income_Mod3"
    ],

    "MOD_4": [
        "Income_Mod4"
    ],

    "MOD_5": [
        "Income_Mod5"
    ],

    "MOD_6": [
        "Income_Mod6"
    ],

    "MOD_7": [
        "Income_Mod7"
    ],

    "MOD_8": [
        "Income_Mod8"
    ],

    "MOD_9": [
        "Income_Mod9"
    ],

    "MOD_10": [
        "Income_Mod10"
    ],

    "MOD_11": [
        "Income_Mod11"
    ],

    "MOD_12": [
        "Income_Mod12"
    ],

    # Individual decimal digits
    "TENS_DIGIT": [
        "Income_TensDigit"
    ],

    "HUNDREDS_DIGIT": [
        "Income_HundredsDigit"
    ],

    "THOUSANDS_DIGIT": [
        "Income_ThousandsDigit"
    ],

    "10THOUSANDS_DIGIT": [
        "Income_10ThousandsDigit"
    ],

    # Scale
    "INCOME_THOUSANDS": [
        "Income_Thousands"
    ],

    "INCOME_HUNDREDS": [
        "Income_Hundreds"
    ],

    "INCOME_TENS": [
        "Income_Tens"
    ],

    # Remainders
    "THOUSANDS_REMAINDER": [
        "Income_Thousands_Remainder"
    ],

    "HUNDREDS_REMAINDER": [
        "Income_Hundreds_Remainder"
    ],
}


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


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(name, feature_names):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Start with base
    X = X_base.copy()

    # Add selected income features
    for feature in feature_names:

        X[feature] = (
            X_income_all[feature]
        )

    X = prepare_features(X)

    print(
        f"Features: {X.shape[1]}"
    )

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    oof = np.zeros(len(X))

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        model = LGBMClassifier(
            **LGB_PARAMS
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

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

        oof[valid_idx] = pred

        fold_auc = roc_auc_score(
            y_valid,
            pred
        )

        fold_scores.append(
            fold_auc
        )

        print(
            f"Fold {fold}: "
            f"{fold_auc:.6f}"
        )

    overall_auc = roc_auc_score(
        y,
        oof
    )

    mean_fold = np.mean(
        fold_scores
    )

    std_fold = np.std(
        fold_scores
    )

    print(
        f"\nOverall AUC : {overall_auc:.6f}"
    )

    print(
        f"Mean Fold   : {mean_fold:.6f}"
    )

    print(
        f"Std Fold    : {std_fold:.6f}"
    )

    return {
        "experiment": name,
        "auc": overall_auc,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "n_features": X.shape[1],
    }


# ============================================================
# RUN
# ============================================================

results = []


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

results.append(
    run_experiment(
        "BASELINE",
        []
    )
)


# ------------------------------------------------------------
# EACH FEATURE GROUP
# ------------------------------------------------------------

for group_name, features in FEATURE_GROUPS.items():

    results.append(
        run_experiment(
            f"BASE + {group_name}",
            features
        )
    )


# ============================================================
# IMPORTANT COMBINATIONS
# ============================================================

COMBINATIONS = {

    # Current best
    "BEST_LAST2_PLUS_LASTDIGIT": [
        "Income_Last2Digits",
        "Income_LastDigit",
    ],

    "BEST_LAST2_PLUS_MODS": [
        "Income_Last2Digits",
        "Income_Mod3",
        "Income_Mod7",
        "Income_Mod9",
    ],

    "LAST_DIGIT_FAMILY": [
        "Income_LastDigit",
        "Income_Last2Digits",
        "Income_Last3Digits",
    ],

    "ALL_MODS": [
        "Income_Mod2",
        "Income_Mod3",
        "Income_Mod4",
        "Income_Mod5",
        "Income_Mod6",
        "Income_Mod7",
        "Income_Mod8",
        "Income_Mod9",
        "Income_Mod10",
        "Income_Mod11",
        "Income_Mod12",
    ],

    "ALL_DECIMAL_DIGITS": [
        "Income_LastDigit",
        "Income_TensDigit",
        "Income_HundredsDigit",
        "Income_ThousandsDigit",
        "Income_10ThousandsDigit",
    ],

    "ALL_SCALE_FEATURES": [
        "Income_Thousands",
        "Income_Hundreds",
        "Income_Tens",
    ],

    "ALL_REMAINDERS": [
        "Income_Last2Digits",
        "Income_Thousands_Remainder",
        "Income_Hundreds_Remainder",
    ],

    "ALL_INCOME_ENGINEERING": list(
        X_income_all.columns[
            ~X_income_all.columns.isin(
                X_base.columns
            )
        ]
    ),
}


for combo_name, features in COMBINATIONS.items():

    results.append(
        run_experiment(
            f"BASE + {combo_name}",
            features
        )
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

baseline_auc = results_df.loc[
    results_df["experiment"] == "BASELINE",
    "auc"
].iloc[0]

results_df["gain_vs_baseline"] = (
    results_df["auc"]
    - baseline_auc
)

results_df = results_df.sort_values(
    "auc",
    ascending=False
).reset_index(drop=True)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 70)
print("INCOME DECOMPOSITION RESULTS")
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
# BEST RESULT
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 70)
print("CURRENT BEST")
print("=" * 70)

print(
    f"Experiment : {best['experiment']}"
)

print(
    f"AUC        : {best['auc']:.6f}"
)

print(
    f"Gain       : "
    f"{best['gain_vs_baseline']:+.6f}"
)

print(
    f"Features   : "
    f"{best['n_features']}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = (
    "income_decomposition_ablation.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    f"File: {OUTPUT_FILE}"
)

print("\nDone.")