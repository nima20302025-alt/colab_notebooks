# ============================================================
# FAST CROSS-CATEGORY ABLATION
# LightGBM sklearn API
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"

RANDOM_STATE = 42

N_SPLITS = 5

SCREENING_ESTIMATORS = 600
FULL_ESTIMATORS = 1500

LEARNING_RATE = 0.05

EARLY_STOPPING_ROUNDS = 75

TOP_K = 3


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
    bad_values = target_raw[y.isna()].unique()

    raise ValueError(
        f"Unknown target values found: {bad_values}"
    )

y = y.astype(np.int8)


# ============================================================
# BASE X
# ============================================================

X = df.drop(columns=[TARGET]).copy()


# ============================================================
# INCOME FEATURE ENGINEERING
# EXACT SAME 23 FEATURES
# ============================================================

INCOME_FEATURES = [
    "Income_LastDigit",
    "Income_Last2Digits",
    "Income_Last3Digits",

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

    "Income_TensDigit",
    "Income_HundredsDigit",
    "Income_ThousandsDigit",
    "Income_10ThousandsDigit",

    "Income_Thousands",
    "Income_Hundreds",
    "Income_Tens",

    "Income_Thousands_Remainder",
    "Income_Hundreds_Remainder",
]


def add_income_features(data):

    income = pd.to_numeric(
        data["Annual_Income_USD"],
        errors="coerce"
    ).fillna(0)

    income_int = income.astype(np.int64)

    # Last digits
    data["Income_LastDigit"] = income_int % 10
    data["Income_Last2Digits"] = income_int % 100
    data["Income_Last3Digits"] = income_int % 1000

    # Modulo features
    for m in range(2, 13):
        data[f"Income_Mod{m}"] = income_int % m

    # Individual digits
    data["Income_TensDigit"] = (income_int // 10) % 10
    data["Income_HundredsDigit"] = (income_int // 100) % 10
    data["Income_ThousandsDigit"] = (income_int // 1000) % 10
    data["Income_10ThousandsDigit"] = (income_int // 10000) % 10

    # Higher-order parts
    data["Income_Thousands"] = income_int // 1000
    data["Income_Hundreds"] = income_int // 100
    data["Income_Tens"] = income_int // 10

    # Remainders
    data["Income_Thousands_Remainder"] = income_int % 1000
    data["Income_Hundreds_Remainder"] = income_int % 100

    return data


X = add_income_features(X)


# ============================================================
# ORIGINAL CATEGORICAL FEATURES
# ============================================================

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]


# ============================================================
# CROSS-CATEGORY FEATURES
# ============================================================

CROSS_FEATURES = {

    "Env_x_Subsidy_Cat": [
        "Environmental_Concern_Level",
        "Subsidy_Available"
    ],

    "Env_x_Range_Cat": [
        "Environmental_Concern_Level",
        "Range_Anxiety_Level"
    ],

    "Env_x_Home_Cat": [
        "Environmental_Concern_Level",
        "Home_Charging_Possible"
    ],

    "Income_x_Subsidy_Cat": [
        "Annual_Income_USD",
        "Subsidy_Available"
    ],

    "Income_x_Home_Cat": [
        "Annual_Income_USD",
        "Home_Charging_Possible"
    ],

    "Range_x_Home_Cat": [
        "Range_Anxiety_Level",
        "Home_Charging_Possible"
    ],

    "City_x_Home_Cat": [
        "City_Type",
        "Home_Charging_Possible"
    ],

    "Car_x_Range_Cat": [
        "Current_Car_Type",
        "Range_Anxiety_Level"
    ]
}


# ============================================================
# CREATE CROSS FEATURES
# ============================================================

for feature_name, cols in CROSS_FEATURES.items():

    X[feature_name] = (
        X[cols[0]].astype(str)
        + "__"
        + X[cols[1]].astype(str)
    )


# ============================================================
# EXACT 37-FEATURE BASELINE
# ============================================================

BASE_COLS = [
    c for c in X.columns
    if c not in CROSS_FEATURES
]

print("\n" + "=" * 70)
print("BASELINE CHECK")
print("=" * 70)

print("Baseline features:", len(BASE_COLS))

if len(BASE_COLS) != 37:
    raise ValueError(
        f"Expected exactly 37 baseline features, "
        f"but got {len(BASE_COLS)}"
    )


# ============================================================
# CATEGORICAL TYPES
# ============================================================

ALL_CAT_COLS = CAT_COLS + list(CROSS_FEATURES.keys())

for col in ALL_CAT_COLS:
    X[col] = X[col].astype("category")


# ============================================================
# MODEL PARAMETERS
# ============================================================

MODEL_PARAMS = {
    "objective": "binary",

    "n_estimators": FULL_ESTIMATORS,
    "learning_rate": LEARNING_RATE,

    "num_leaves": 63,
    "max_depth": -1,

    "subsample": 0.85,
    "colsample_bytree": 0.85,

    "reg_alpha": 0.1,
    "reg_lambda": 1.0,

    "min_child_samples": 30,

    "random_state": RANDOM_STATE,
    "n_jobs": -1,

    "verbosity": -1
}


# ============================================================
# SCREENING FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

screen_train_idx, screen_valid_idx = next(
    skf.split(X, y)
)


# ============================================================
# FAST SCREENING FUNCTION
# ============================================================

def fast_evaluate(feature_cols):

    X_train = X.iloc[screen_train_idx][feature_cols]
    X_valid = X.iloc[screen_valid_idx][feature_cols]

    y_train = y.iloc[screen_train_idx]
    y_valid = y.iloc[screen_valid_idx]

    model = LGBMClassifier(
        **{
            **MODEL_PARAMS,
            "n_estimators": SCREENING_ESTIMATORS
        }
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        eval_metric="auc",

        categorical_feature=[
            c for c in feature_cols
            if c in ALL_CAT_COLS
        ],

        callbacks=[
            early_stopping(
                EARLY_STOPPING_ROUNDS,
                verbose=False
            ),
            log_evaluation(0)
        ]
    )

    pred = model.predict_proba(
        X_valid
    )[:, 1]

    return roc_auc_score(
        y_valid,
        pred
    )


# ============================================================
# STAGE 1 — FAST SCREENING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — FAST CROSS-CATEGORY SCREENING")
print("=" * 70)

baseline_screen_auc = fast_evaluate(
    BASE_COLS
)

print(
    f"\nBaseline screening AUC: "
    f"{baseline_screen_auc:.6f}"
)


screening_results = []


for feature_name in CROSS_FEATURES:

    print(
        f"\nTesting: {feature_name}"
    )

    auc = fast_evaluate(
        BASE_COLS + [feature_name]
    )

    gain = auc - baseline_screen_auc

    screening_results.append({
        "feature": feature_name,
        "screen_auc": auc,
        "gain": gain
    })

    print(
        f"AUC  : {auc:.6f}"
    )

    print(
        f"Gain : {gain:+.6f}"
    )


screening_df = pd.DataFrame(
    screening_results
)

screening_df = screening_df.sort_values(
    "screen_auc",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SCREENING RESULTS
# ============================================================

print("\n\n")
print("=" * 70)
print("FAST SCREENING RESULTS")
print("=" * 70)

print(
    screening_df.to_string(
        index=False,
        formatters={
            "screen_auc": "{:.6f}".format,
            "gain": "{:+.6f}".format
        }
    )
)


# ============================================================
# SELECT TOP K
# ============================================================

top_features = (
    screening_df
    .head(TOP_K)
    ["feature"]
    .tolist()
)


print("\n")
print("=" * 70)
print("TOP FEATURES FOR FULL VALIDATION")
print("=" * 70)

for i, feature in enumerate(
    top_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# FULL 5-FOLD FUNCTION
# ============================================================

def full_evaluate(
    feature_cols,
    experiment_name
):

    print("\n" + "-" * 70)
    print(
        f"FULL 5-FOLD: {experiment_name}"
    )
    print(
        f"Features: {len(feature_cols)}"
    )
    print("-" * 70)

    oof_pred = np.zeros(
        len(X),
        dtype=np.float32
    )

    fold_aucs = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx][feature_cols]
        X_valid = X.iloc[valid_idx][feature_cols]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = LGBMClassifier(
            **MODEL_PARAMS
        )

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (X_valid, y_valid)
            ],

            eval_metric="auc",

            categorical_feature=[
                c for c in feature_cols
                if c in ALL_CAT_COLS
            ],

            callbacks=[
                early_stopping(
                    EARLY_STOPPING_ROUNDS,
                    verbose=False
                ),
                log_evaluation(0)
            ]
        )

        pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof_pred[valid_idx] = pred

        fold_auc = roc_auc_score(
            y_valid,
            pred
        )

        fold_aucs.append(
            fold_auc
        )

        print(
            f"Fold {fold}: "
            f"AUC = {fold_auc:.6f} | "
            f"Best Iter = {model.best_iteration_}"
        )


    overall_auc = roc_auc_score(
        y,
        oof_pred
    )

    mean_fold = np.mean(
        fold_aucs
    )

    std_fold = np.std(
        fold_aucs
    )

    print(
        f"\nOOF AUC: {overall_auc:.6f}"
    )

    return {
        "experiment": experiment_name,
        "auc": overall_auc,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "n_features": len(feature_cols)
    }


# ============================================================
# STAGE 2 — FULL VALIDATION
# ============================================================

print("\n\n")
print("=" * 70)
print("STAGE 2 — FULL 5-FOLD VALIDATION")
print("=" * 70)


full_results = []


# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

baseline_result = full_evaluate(
    BASE_COLS,
    "BASELINE — 37 FEATURES"
)

full_results.append(
    baseline_result
)

baseline_auc = baseline_result["auc"]


# ------------------------------------------------------------
# Top individual features
# ------------------------------------------------------------

for feature in top_features:

    result = full_evaluate(
        BASE_COLS + [feature],
        f"BASELINE + {feature}"
    )

    full_results.append(
        result
    )


# ------------------------------------------------------------
# Top features together
# ------------------------------------------------------------

if len(top_features) >= 2:

    result = full_evaluate(
        BASE_COLS + top_features,
        "BASELINE + TOP CROSS FEATURES"
    )

    full_results.append(
        result
    )


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(
    full_results
)

results_df["gain_vs_baseline"] = (
    results_df["auc"] - baseline_auc
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
print("FINAL CROSS-CATEGORY RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "auc": "{:.6f}".format,
            "mean_fold": "{:.6f}".format,
            "std_fold": "{:.6f}".format,
            "gain_vs_baseline": "{:+.6f}".format
        }
    )
)


# ============================================================
# CURRENT BEST
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 70)
print("CURRENT BEST")
print("=" * 70)

print(
    f"AUC     : {best['auc']:.6f}"
)

print(
    f"Gain    : {best['gain_vs_baseline']:+.6f}"
)

print(
    f"Config  : {best['experiment']}"
)

print(
    f"Features: {int(best['n_features'])}"
)


# ============================================================
# SAVE
# ============================================================

screening_df.to_csv(
    "cross_category_fast_screening.csv",
    index=False
)

results_df.to_csv(
    "cross_category_full_results.csv",
    index=False
)

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "cross_category_fast_screening.csv"
)

print(
    "cross_category_full_results.csv"
)