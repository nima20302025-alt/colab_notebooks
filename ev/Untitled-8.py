
# ============================================================
# EV S6E9 — CLEAN FREQUENCY ENCODING ABLATION
#
# IMPORTANT:
# Baseline = same feature set as previous 37-feature experiment
# No categorical *_Code features are added.
#
# Current reference baseline:
# ~0.9440 OOF AUC
# ============================================================

import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"

SEED = 42
N_SPLITS = 5


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

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
    print("\nUnknown target values:")
    print(target_raw[y.isna()].value_counts())
    raise ValueError("Target conversion failed.")

y = y.astype(int)

X_raw = df.drop(
    columns=[TARGET]
).copy()

print(
    "Positive rate:",
    y.mean()
)


# ============================================================
# BASE FEATURE ENGINEERING
#
# SAME BASELINE AS THE PREVIOUS 37-FEATURE EXPERIMENT
# ============================================================

def make_base_features(df):

    X = df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_cols = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level"
    ]

    for c in numeric_cols:

        if c in X.columns:

            X[c] = pd.to_numeric(
                X[c],
                errors="coerce"
            )

    # ========================================================
    # INCOME ENGINEERING
    # ========================================================

    income = X[
        "Annual_Income_USD"
    ].astype(float)

    # Last digits

    X["Income_LastDigit"] = (
        income % 10
    )

    X["Income_Last2Digits"] = (
        income % 100
    )

    X["Income_Last3Digits"] = (
        income % 1000
    )

    # Modulo 2 ... 12

    for m in range(2, 13):

        X[f"Income_Mod{m}"] = (
            income % m
        )

    # Decimal digits

    X["Income_TensDigit"] = (
        (income // 10) % 10
    )

    X["Income_HundredsDigit"] = (
        (income // 100) % 10
    )

    X["Income_ThousandsDigit"] = (
        (income // 1000) % 10
    )

    X["Income_10ThousandsDigit"] = (
        (income // 10000) % 10
    )

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

    # ========================================================
    # DOMAIN FEATURES
    # ========================================================

    # Charging

    X["Charging_Total"] = (
        X["Charging_Stations_Near_Home"]
        + X["Charging_Stations_Near_Work"]
    )

    X["Charging_Diff"] = (
        X["Charging_Stations_Near_Home"]
        - X["Charging_Stations_Near_Work"]
    )

    X["Charging_AbsDiff"] = (
        X["Charging_Diff"].abs()
    )

    X["Charging_Min"] = X[
        [
            "Charging_Stations_Near_Home",
            "Charging_Stations_Near_Work"
        ]
    ].min(axis=1)

    X["Charging_Max"] = X[
        [
            "Charging_Stations_Near_Home",
            "Charging_Stations_Near_Work"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # Income transforms
    # --------------------------------------------------------

    X["Income_Log"] = np.log1p(
        np.maximum(income, 0)
    )

    X["Income_Sqrt"] = np.sqrt(
        np.maximum(income, 0)
    )

    X["Income_Per_Age"] = (
        income / (X["Age"] + 1)
    )

    # --------------------------------------------------------
    # Commute transforms
    # --------------------------------------------------------

    commute = X[
        "Daily_Commute_km"
    ].astype(float)

    X["Commute_Log"] = np.log1p(
        np.maximum(commute, 0)
    )

    X["Commute_Sqrt"] = np.sqrt(
        np.maximum(commute, 0)
    )

    X["Commute_Sq"] = (
        commute ** 2
    )

    # ========================================================
    # RATIOS
    # ========================================================

    X["Income_Per_Commute"] = (
        income / (commute + 1)
    )

    X["Commute_Per_Income"] = (
        commute /
        (income / 100000 + 1)
    )

    X["Income_Per_Car"] = (
        income /
        (X["Number_of_Cars_Owned"] + 1)
    )

    X["Cars_Per_100k_Income"] = (
        X["Number_of_Cars_Owned"] /
        (income / 100000 + 1)
    )

    X["Stations_Per_Car"] = (
        X["Charging_Total"] /
        (X["Number_of_Cars_Owned"] + 1)
    )

    X["Stations_Per_Commute"] = (
        X["Charging_Total"] /
        (commute + 1)
    )

    # ========================================================
    # INTERACTIONS
    # ========================================================

    # IMPORTANT:
    # These use categorical variables directly.
    # No *_Code columns are created.

    # We only keep the interaction features that belonged
    # to the previous baseline experiment.

    subsidy_code = (
        X["Subsidy_Available"]
        .astype("category")
        .cat.codes
    )

    home_code = (
        X["Home_Charging_Possible"]
        .astype("category")
        .cat.codes
    )

    range_code = (
        X["Range_Anxiety_Level"]
        .astype("category")
        .cat.codes
    )

    X["Income_x_Environmental"] = (
        income *
        X["Environmental_Concern_Level"]
    )

    X["Income_x_Subsidy"] = (
        income *
        subsidy_code
    )

    X["Income_x_RangeAnxiety"] = (
        income *
        range_code
    )

    X["Commute_x_RangeAnxiety"] = (
        commute *
        range_code
    )

    X["Environmental_x_Subsidy"] = (
        X["Environmental_Concern_Level"] *
        subsidy_code
    )

    X["Environmental_x_HomeCharging"] = (
        X["Environmental_Concern_Level"] *
        home_code
    )

    # ========================================================
    # EV DOMAIN SCORE
    # ========================================================

    X["EV_Domain_Score"] = (
        X["Environmental_Concern_Level"] * 2
        + subsidy_code * 3
        + home_code * 2
        - range_code
    )

    # ========================================================
    # ID
    # ========================================================

    if "id" in X.columns:

        X = X.drop(
            columns=["id"]
        )

    return X


# ============================================================
# CREATE BASE
# ============================================================

X_base = make_base_features(
    X_raw
)

print(
    "\nBASE FEATURE COUNT:",
    X_base.shape[1]
)

print(
    "Expected baseline should be around 37."
)


# ============================================================
# FREQUENCY FEATURES
# ============================================================

single_freq_cols = [

    "Environmental_Concern_Level",

    "Subsidy_Available",

    "Range_Anxiety_Level",

    "Home_Charging_Possible",

    "City_Type",

    "Current_Car_Type",

    "Gender",

    "Number_of_Cars_Owned"
]


pair_freq_cols = [

    (
        "Environmental_Concern_Level",
        "Subsidy_Available"
    ),

    (
        "Environmental_Concern_Level",
        "Range_Anxiety_Level"
    ),

    (
        "Subsidy_Available",
        "Range_Anxiety_Level"
    )
]


triple_freq_cols = [

    "Environmental_Concern_Level",

    "Subsidy_Available",

    "Range_Anxiety_Level"
]


# ============================================================
# ADD FREQUENCY
# ============================================================

def add_frequency_features(
    X,
    single_cols=None,
    pair_cols=None,
    triple_cols=None
):

    X = X.copy()

    # --------------------------------------------------------
    # Single
    # --------------------------------------------------------

    if single_cols:

        for c in single_cols:

            freq = (
                X[c]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
            )

            X[
                f"{c}_Freq"
            ] = (
                X[c]
                .map(freq)
                .astype(float)
            )

    # --------------------------------------------------------
    # Pairs
    # --------------------------------------------------------

    if pair_cols:

        for cols in pair_cols:

            name = (
                "_".join(cols)
                + "_Freq"
            )

            key = (
                X[list(cols)]
                .astype(str)
                .agg(
                    "__".join,
                    axis=1
                )
            )

            freq = (
                key
                .value_counts(
                    normalize=True
                )
            )

            X[name] = (
                key
                .map(freq)
                .astype(float)
            )

    # --------------------------------------------------------
    # Triple
    # --------------------------------------------------------

    if triple_cols:

        name = (
            "_".join(triple_cols)
            + "_Freq"
        )

        key = (
            X[triple_cols]
            .astype(str)
            .agg(
                "__".join,
                axis=1
            )
        )

        freq = (
            key
            .value_counts(
                normalize=True
            )
        )

        X[name] = (
            key
            .map(freq)
            .astype(float)
        )

    return X


# ============================================================
# PREPARE FOR LIGHTGBM
# ============================================================

def prepare_lgbm_data(X):

    X = X.copy()

    object_cols = X.select_dtypes(
        include=["object"]
    ).columns

    for c in object_cols:

        X[c] = X[c].astype(
            "category"
        )

    return X


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cv(
    X,
    experiment_name
):

    X = prepare_lgbm_data(
        X
    )

    skf = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=SEED
    )

    oof = np.zeros(
        len(X),
        dtype=np.float64
    )

    fold_scores = []

    categorical_features = [

        c for c in X.columns

        if str(
            X[c].dtype
        ) == "category"
    ]

    print("\n")
    print("=" * 75)

    print(
        experiment_name
    )

    print(
        "Feature count:",
        X.shape[1]
    )

    print("=" * 75)

    # --------------------------------------------------------
    # FOLDS
    # --------------------------------------------------------

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(

        skf.split(X, y),

        start=1
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_valid = X.iloc[
            valid_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_valid = y.iloc[
            valid_idx
        ]

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = LGBMClassifier(

            objective="binary",

            n_estimators=1500,

            learning_rate=0.035,

            num_leaves=31,

            max_depth=-1,

            min_child_samples=80,

            subsample=0.85,

            colsample_bytree=0.85,

            reg_alpha=0.05,

            reg_lambda=0.5,

            random_state=SEED,

            n_jobs=-1,

            verbosity=-1
        )

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        model.fit(

            X_train,

            y_train,

            categorical_feature=
                categorical_features,

            eval_set=[
                (
                    X_valid,
                    y_valid
                )
            ],

            callbacks=[

                early_stopping(
                    100,
                    verbose=False
                ),

                log_evaluation(
                    0
                )
            ]
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        pred = model.predict_proba(
            X_valid
        )[:, 1]

        oof[
            valid_idx
        ] = pred

        auc = roc_auc_score(
            y_valid,
            pred
        )

        fold_scores.append(
            auc
        )

        print(

            f"Fold {fold}/{N_SPLITS} | "

            f"AUC = {auc:.6f} | "

            f"Best Iter = "
            f"{model.best_iteration_}"
        )

        del (
            model,
            X_train,
            X_valid,
            y_train,
            y_valid,
            pred
        )

        gc.collect()

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    overall_auc = roc_auc_score(
        y,
        oof
    )

    mean_auc = np.mean(
        fold_scores
    )

    std_auc = np.std(
        fold_scores
    )

    print("\n")

    print(
        f"Overall OOF AUC : "
        f"{overall_auc:.6f}"
    )

    print(
        f"Mean Fold AUC   : "
        f"{mean_auc:.6f}"
    )

    print(
        f"Std Fold AUC    : "
        f"{std_auc:.6f}"
    )

    print(
        "-" * 75
    )

    return overall_auc


# ============================================================
# EXPERIMENTS
# ============================================================

results = {}


# ============================================================
# 1. PURE BASELINE
# ============================================================

results[
    "BASELINE"
] = run_cv(

    X_base,

    "BASELINE — 37 FEATURES"
)


# ============================================================
# 2. ALL SINGLE FREQUENCY
# ============================================================

X_all_single = add_frequency_features(

    X_base,

    single_cols=
        single_freq_cols
)

results[
    "ALL_SINGLE_FREQ"
] = run_cv(

    X_all_single,

    "BASE + ALL SINGLE FREQUENCY"
)

del X_all_single

gc.collect()


# ============================================================
# 3. BEST SINGLE FREQUENCY
# ============================================================

best_single_cols = [

    "Environmental_Concern_Level",

    "Subsidy_Available",

    "Range_Anxiety_Level",

    "Home_Charging_Possible"
]


X_best_single = add_frequency_features(

    X_base,

    single_cols=
        best_single_cols
)

results[
    "BEST_SINGLE_FREQ"
] = run_cv(

    X_best_single,

    "BASE + BEST SINGLE FREQUENCY"
)

del X_best_single

gc.collect()


# ============================================================
# 4. PAIR FREQUENCY
# ============================================================

X_pair = add_frequency_features(

    X_base,

    pair_cols=
        pair_freq_cols
)

results[
    "PAIR_FREQ"
] = run_cv(

    X_pair,

    "BASE + PAIR FREQUENCY"
)

del X_pair

gc.collect()


# ============================================================
# 5. ALL FREQUENCY
# ============================================================

X_all_freq = add_frequency_features(

    X_base,

    single_cols=
        single_freq_cols,

    pair_cols=
        pair_freq_cols,

    triple_cols=
        triple_freq_cols
)

results[
    "ALL_FREQ"
] = run_cv(

    X_all_freq,

    "BASE + ALL FREQUENCY"
)


# ============================================================
# FINAL RESULTS
# ============================================================

baseline = results[
    "BASELINE"
]

print("\n\n")

print("=" * 75)

print(
    "FINAL RESULTS"
)

print("=" * 75)


for name, score in sorted(

    results.items(),

    key=lambda x: x[1],

    reverse=True

):

    gain = (
        score - baseline
    )

    print(

        f"{name:35s} "

        f"AUC = {score:.6f} | "

        f"Gain = {gain:+.6f}"
    )


print("=" * 75)

print(
    f"Baseline = "
    f"{baseline:.6f}"
)

best_name = max(
    results,
    key=results.get
)

best_score = results[
    best_name
]

print(
    f"Best = "
    f"{best_name}"
)

print(
    f"Best AUC = "
    f"{best_score:.6f}"
)

print(
    f"Best Gain = "
    f"{best_score - baseline:+.6f}"
)

print("=" * 75)
