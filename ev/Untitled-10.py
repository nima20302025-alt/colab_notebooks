
# ============================================================
# EV PROJECT - FREQUENCY ENCODING ABLATION
# BASELINE = EXACT ORIGINAL 37 FEATURES
# ============================================================

import gc
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"
TARGET = "Will_Buy_EV"

N_SPLITS = 5
SEED = 42

N_ESTIMATORS = 1500
LEARNING_RATE = 0.05


# ============================================================
# ALL INCOME ENGINEERING
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


# ============================================================
# LOAD
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
    "false": 0,
}

y = target_raw.map(target_map)

if y.isna().any():
    print("\nUnexpected target values:")
    print(target_raw[y.isna()].value_counts())
    raise ValueError("Target conversion failed.")

y = y.astype(np.int8)

print("Target dtype:", y.dtype)
print("Target mean:", y.mean())
print("Target distribution:")
print(y.value_counts())


# ============================================================
# BASE FEATURES
# ============================================================

X = df.drop(columns=[TARGET]).copy()


# ============================================================
# INCOME FEATURE GENERATION
# ============================================================

income = (
    pd.to_numeric(
        X["Annual_Income_USD"],
        errors="coerce"
    )
    .fillna(0)
    .astype(np.int64)
)


# ------------------------------------------------------------
# Last digits
# ------------------------------------------------------------

X["Income_LastDigit"] = income % 10
X["Income_Last2Digits"] = income % 100
X["Income_Last3Digits"] = income % 1000


# ------------------------------------------------------------
# Modulo
# ------------------------------------------------------------

for m in range(2, 13):
    X[f"Income_Mod{m}"] = income % m


# ------------------------------------------------------------
# Decimal digits
# ------------------------------------------------------------

X["Income_TensDigit"] = (income // 10) % 10
X["Income_HundredsDigit"] = (income // 100) % 10
X["Income_ThousandsDigit"] = (income // 1000) % 10
X["Income_10ThousandsDigit"] = (income // 10000) % 10


# ------------------------------------------------------------
# Scale
# ------------------------------------------------------------

X["Income_Thousands"] = income // 1000
X["Income_Hundreds"] = income // 100
X["Income_Tens"] = income // 10


# ------------------------------------------------------------
# Remainders
# ------------------------------------------------------------

X["Income_Thousands_Remainder"] = income % 1000
X["Income_Hundreds_Remainder"] = income % 100


# ============================================================
# INTERACTION COLUMNS
# These are created ONLY so that they can be excluded
# from the exact 37-feature baseline.
# ============================================================

X["Env_x_Subsidy"] = (
    X["Environmental_Concern_Level"].astype(str)
    + "_"
    + X["Subsidy_Available"].astype(str)
)

X["Env_x_Subsidy_x_Range"] = (
    X["Environmental_Concern_Level"].astype(str)
    + "_"
    + X["Subsidy_Available"].astype(str)
    + "_"
    + X["Range_Anxiety_Level"].astype(str)
)

X["Env_x_Subsidy_x_Home"] = (
    X["Environmental_Concern_Level"].astype(str)
    + "_"
    + X["Subsidy_Available"].astype(str)
    + "_"
    + X["Home_Charging_Possible"].astype(str)
)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",

    "Env_x_Subsidy",
    "Env_x_Subsidy_x_Range",
    "Env_x_Subsidy_x_Home",
]


for col in CAT_COLS:
    if col in X.columns:
        X[col] = X[col].astype("category")


# ============================================================
# INTERACTION LIST
# ============================================================

INTERACTION_COLS = [
    "Env_x_Subsidy",
    "Env_x_Subsidy_x_Range",
    "Env_x_Subsidy_x_Home",
]


# ============================================================
# EXACT 37-FEATURE BASELINE
# ============================================================

BASE_COLS = [
    c for c in X.columns
    if c not in INTERACTION_COLS
]

BASE_COLS = list(
    dict.fromkeys(BASE_COLS)
)


# ============================================================
# VERIFY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("EXACT BASELINE CHECK")
print("=" * 70)

print("Number of BASELINE features:", len(BASE_COLS))

if len(BASE_COLS) != 37:
    raise ValueError(
        f"Baseline should have exactly 37 features, "
        f"but got {len(BASE_COLS)}"
    )

print("\n37 BASELINE FEATURES:")

for i, col in enumerate(BASE_COLS, 1):
    print(f"{i:02d}. {col}")


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",

    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,

    "num_leaves": 63,
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
# FIXED FOLDS
# ============================================================

print("\nCreating fixed folds...")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

folds = list(skf.split(X, y))

print("5 folds ready.")


# ============================================================
# FREQUENCY ENCODING
# ============================================================

FREQ_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


print("\n" + "=" * 70)
print("CREATING FREQUENCY FEATURES")
print("=" * 70)


for col in FREQ_COLS:

    freq_map = (
        X[col]
        .value_counts(normalize=True)
    )

    X[f"{col}_Freq"] = (
        X[col]
        .map(freq_map)
        .astype("float32")
    )

    print(
        f"{col:30s} -> "
        f"{col}_Freq"
    )


FREQ_FEATURES = [
    f"{col}_Freq"
    for col in FREQ_COLS
]


print("\nFrequency features:", len(FREQ_FEATURES))


# ============================================================
# EVALUATION
# ============================================================

def evaluate(experiment_name, extra_features):

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    selected_cols = BASE_COLS + extra_features

    selected_cols = list(
        dict.fromkeys(selected_cols)
    )

    print("Number of features:", len(selected_cols))

    X_model = X[selected_cols]

    oof = np.zeros(
        len(X_model),
        dtype=np.float32
    )

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        folds,
        1
    ):

        print(
            f"\nFold {fold}/{N_SPLITS} ..."
        )

        X_train = X_model.iloc[train_idx]
        X_valid = X_model.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = LGBMClassifier(
            **LGB_PARAMS
        )

        categorical_features = [
            c for c in CAT_COLS
            if c in selected_cols
        ]

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (X_valid, y_valid)
            ],

            categorical_feature=categorical_features,

            callbacks=[
                early_stopping(
                    100,
                    verbose=False
                ),
                log_evaluation(0),
            ],
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
            f"  AUC = {fold_auc:.6f}"
        )

        del model
        del X_train
        del X_valid
        del y_train
        del y_valid

        gc.collect()


    # --------------------------------------------------------
    # Overall OOF
    # --------------------------------------------------------

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

    print("-" * 70)

    print(
        f"Overall OOF AUC : {overall_auc:.6f}"
    )

    print(
        f"Mean Fold AUC   : {mean_fold:.6f}"
    )

    print(
        f"Std Fold AUC    : {std_fold:.6f}"
    )

    print("-" * 70)

    return {
        "experiment": experiment_name,
        "auc": overall_auc,
        "mean_fold": mean_fold,
        "std_fold": std_fold,
        "n_features": len(selected_cols),
    }


# ============================================================
# EXPERIMENT 1
# EXACT BASELINE
# ============================================================

results = []

baseline_result = evaluate(
    "BASELINE — EXACT 37 FEATURES",
    []
)

results.append(
    baseline_result
)

baseline_auc = baseline_result["auc"]


# ============================================================
# EXPERIMENT 2
# ALL SINGLE FREQUENCY FEATURES
# ============================================================

all_freq_result = evaluate(
    "BASELINE + ALL FREQUENCY",
    FREQ_FEATURES
)

results.append(
    all_freq_result
)


# ============================================================
# EXPERIMENT 3
# INDIVIDUAL FREQUENCY ABLATION
# ============================================================

for freq_feature in FREQ_FEATURES:

    result = evaluate(
        f"BASELINE + {freq_feature}",
        [freq_feature]
    )

    results.append(
        result
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df["gain_vs_baseline"] = (
    results_df["auc"]
    - baseline_auc
)

results_df = results_df.sort_values(
    "auc",
    ascending=False
).reset_index(drop=True)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FREQUENCY ENCODING RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# BEST FREQUENCY FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("BEST FREQUENCY FEATURES")
print("=" * 70)

for _, row in results_df.iterrows():

    if row["experiment"] == "BASELINE — EXACT 37 FEATURES":
        continue

    print(
        f"{row['experiment']:45s} "
        f"AUC={row['auc']:.6f} "
        f"Gain={row['gain_vs_baseline']:+.6f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "frequency_ablation_results.csv",
    index=False
)

print("\nSaved:")
print("frequency_ablation_results.csv")


# ============================================================
# CURRENT BEST
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 70)
print("CURRENT BEST")
print("=" * 70)

print(
    f"AUC    : {best['auc']:.6f}"
)

print(
    f"Gain   : {best['gain_vs_baseline']:+.6f}"
)

print(
    f"Config : {best['experiment']}"
)

print(
    f"Features: {best['n_features']}"
)
