# ============================================================
# EV PURCHASE — SEQUENTIAL INCOME FEATURE SELECTION
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

# برای اینکه جستجو خیلی طولانی نشود
MAX_STEPS = 15

# اگر AUC حداقل این مقدار بهتر نشود، feature اضافه نمی‌شود
MIN_IMPROVEMENT = 0.00001


# ============================================================
# LOAD
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
    raise ValueError(
        f"Unknown target values: "
        f"{target_raw[y.isna()].unique()}"
    )

y = y.astype(int)

print(f"Positive rate: {y.mean():.6f}")


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
# CATEGORICAL
# ============================================================

CATEGORICAL_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def prepare_features(X):

    X = X.copy()

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X


# ============================================================
# INCOME ENGINEERING
# ============================================================

def create_income_features(X):

    X = X.copy()

    income = (
        X["Annual_Income_USD"]
        .round()
        .astype(np.int64)
    )

    # Last digits
    X["Income_LastDigit"] = income % 10
    X["Income_Last2Digits"] = income % 100
    X["Income_Last3Digits"] = income % 1000

    # Modulo
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

    # Decimal digits
    X["Income_TensDigit"] = (
        income // 10
    ) % 10

    X["Income_HundredsDigit"] = (
        income // 100
    ) % 10

    X["Income_ThousandsDigit"] = (
        income // 1000
    ) % 10

    X["Income_10ThousandsDigit"] = (
        income // 10000
    ) % 10

    # Scale
    X["Income_Thousands"] = (
        income // 1000
    )

    X["Income_Hundreds"] = (
        income // 100
    )

    X["Income_Tens"] = (
        income // 10
    )

    # Remainders
    X["Income_Thousands_Remainder"] = (
        income % 1000
    )

    X["Income_Hundreds_Remainder"] = (
        income % 100
    )

    return X


# ============================================================
# ALL CANDIDATES
# ============================================================

X_engineered = create_income_features(
    X_base
)

CANDIDATE_FEATURES = [
    c for c in X_engineered.columns
    if c not in X_base.columns
]

print("\nCandidate features:")
for i, feature in enumerate(
    CANDIDATE_FEATURES,
    1
):
    print(f"{i:02d}. {feature}")


# ============================================================
# LIGHTGBM
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
# FIXED FOLDS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

FOLDS = list(
    skf.split(X_base, y)
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_features(feature_list):

    X = X_base.copy()

    for feature in feature_list:
        X[feature] = X_engineered[feature]

    X = prepare_features(X)

    oof = np.zeros(len(X))

    fold_scores = []

    for train_idx, valid_idx in FOLDS:

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

        oof[valid_idx] = pred

        fold_scores.append(
            roc_auc_score(
                y_valid,
                pred
            )
        )

    overall_auc = roc_auc_score(
        y,
        oof
    )

    return {
        "auc": overall_auc,
        "mean_fold": np.mean(fold_scores),
        "std_fold": np.std(fold_scores),
    }


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

baseline = evaluate_features([])

print(
    f"AUC: {baseline['auc']:.6f}"
)

best_auc = baseline["auc"]

selected_features = []

history = []


# ============================================================
# SEQUENTIAL FORWARD SELECTION
# ============================================================

remaining_features = (
    CANDIDATE_FEATURES.copy()
)


for step in range(
    1,
    MAX_STEPS + 1
):

    print("\n")
    print("=" * 70)
    print(f"STEP {step}")
    print("=" * 70)

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"Remaining candidates: "
        f"{len(remaining_features)}"
    )

    step_results = []

    # --------------------------------------------------------
    # Test every remaining feature
    # --------------------------------------------------------

    for i, candidate in enumerate(
        remaining_features,
        1
    ):

        trial_features = (
            selected_features
            + [candidate]
        )

        result = evaluate_features(
            trial_features
        )

        step_results.append({
            "feature": candidate,
            "auc": result["auc"],
            "mean_fold": result["mean_fold"],
            "std_fold": result["std_fold"],
        })

        print(
            f"[{i:02d}/{len(remaining_features)}] "
            f"{candidate:<35} "
            f"AUC={result['auc']:.6f}"
        )

    step_df = pd.DataFrame(
        step_results
    )

    step_df = step_df.sort_values(
        "auc",
        ascending=False
    ).reset_index(drop=True)

    winner = step_df.iloc[0]

    improvement = (
        winner["auc"] - best_auc
    )

    print("\n" + "-" * 70)
    print("STEP WINNER")
    print("-" * 70)

    print(
        f"Feature     : "
        f"{winner['feature']}"
    )

    print(
        f"AUC         : "
        f"{winner['auc']:.6f}"
    )

    print(
        f"Improvement : "
        f"{improvement:+.6f}"
    )

    # --------------------------------------------------------
    # STOP if improvement is too small
    # --------------------------------------------------------

    if improvement < MIN_IMPROVEMENT:

        print("\nNo meaningful improvement.")

        print(
            "Sequential selection stopped."
        )

        break

    # --------------------------------------------------------
    # Accept winner
    # --------------------------------------------------------

    selected_features.append(
        winner["feature"]
    )

    remaining_features.remove(
        winner["feature"]
    )

    best_auc = winner["auc"]

    history.append({
        "step": step,
        "selected_feature": winner["feature"],
        "auc": winner["auc"],
        "mean_fold": winner["mean_fold"],
        "std_fold": winner["std_fold"],
        "improvement": improvement,
        "n_selected": len(
            selected_features
        ),
    })

    print("\nSelected so far:")

    for feature in selected_features:
        print(
            f"  ✓ {feature}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

history_df = pd.DataFrame(
    history
)

print("\n\n")
print("=" * 70)
print("SEQUENTIAL FEATURE SELECTION RESULTS")
print("=" * 70)

if len(history_df) > 0:

    print(
        history_df[
            [
                "step",
                "selected_feature",
                "auc",
                "mean_fold",
                "std_fold",
                "improvement",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL FEATURE SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL SELECTED FEATURES")
print("=" * 70)

for i, feature in enumerate(
    selected_features,
    1
):
    print(
        f"{i:02d}. {feature}"
    )

print("\n" + "=" * 70)

print(
    f"Final AUC: {best_auc:.6f}"
)

print(
    f"Total selected: "
    f"{len(selected_features)}"
)


# ============================================================
# SAVE
# ============================================================

history_df.to_csv(
    "income_sequential_selection.csv",
    index=False
)

pd.DataFrame({
    "selected_feature": selected_features
}).to_csv(
    "best_income_features.csv",
    index=False
)

print("\nSaved:")
print("  income_sequential_selection.csv")
print("  best_income_features.csv")

print("\nDone.")