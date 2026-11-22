# ============================================================
# KAGGLE S6E9 - REVERSE ENGINEERING ABLATION
# OOF ROC-AUC Feature Engineering Test
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError(
        "LightGBM is not installed.\n"
        "Install it with:\n"
        "pip install lightgbm"
    )


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42

# To make experimentation manageable
MAX_ROWS = None
# Example:
# MAX_ROWS = 300_000

EARLY_STOPPING = 100
N_ESTIMATORS = 3000


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(TRAIN_PATH)

if MAX_ROWS is not None:
    df = df.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)

print("Shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# TARGET
# ============================================================

# ============================================================
# TARGET
# ============================================================

print("\nTarget values BEFORE conversion:")
print(df[TARGET].value_counts(dropna=False))
print("\nTarget raw unique values:")
print(df[TARGET].unique()[:30])

# Convert target robustly
target_raw = df[TARGET].astype(str).str.strip().str.lower()

target_map = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
}

df[TARGET] = target_raw.map(target_map)

# Check conversion
if df[TARGET].isna().any():

    invalid_values = target_raw[df[TARGET].isna()].unique()

    print("\nERROR: Unknown target values:")
    print(invalid_values[:50])

    raise ValueError(
        "Target conversion failed. "
        "Check the printed target values above."
    )

df[TARGET] = df[TARGET].astype(np.int8)

y = df[TARGET].values

print("\nTarget AFTER conversion:")
print(pd.Series(y).value_counts())

print("Positive rate:", y.mean())

# ============================================================
# LIGHTGBM PREPROCESSING
# ============================================================

def prepare_categories(train_df, valid_df):

    train_df = train_df.copy()
    valid_df = valid_df.copy()

    for col in categorical_cols:

        # Create category mapping using TRAIN only
        categories = train_df[col].astype("category").cat.categories

        train_df[col] = pd.Categorical(
            train_df[col],
            categories=categories
        ).codes

        valid_df[col] = pd.Categorical(
            valid_df[col],
            categories=categories
        ).codes

    return train_df, valid_df


# ============================================================
# BASE MODEL
# ============================================================

def train_lgbm(X_train, y_train, X_valid, y_valid):

    model = lgb.LGBMClassifier(
        objective="binary",

        n_estimators=N_ESTIMATORS,

        learning_rate=0.03,

        num_leaves=63,

        max_depth=-1,

        min_child_samples=40,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.05,

        reg_lambda=0.5,

        random_state=SEED,

        n_jobs=-1,

        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[(X_valid, y_valid)],

        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING,
                verbose=False
            )
        ]
    )

    pred = model.predict_proba(X_valid)[:, 1]

    auc = roc_auc_score(y_valid, pred)

    return auc, model, pred


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_domain_features(X):

    X = X.copy()

    # --------------------------------------------------------
    # Charging infrastructure
    # --------------------------------------------------------

    if {
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work"
    }.issubset(X.columns):

        home = X["Charging_Stations_Near_Home"]
        work = X["Charging_Stations_Near_Work"]

        X["Charging_Total"] = home + work
        X["Charging_Diff"] = home - work
        X["Charging_Min"] = np.minimum(home, work)
        X["Charging_Max"] = np.maximum(home, work)

    # --------------------------------------------------------
    # Income transformations
    # --------------------------------------------------------

    if "Annual_Income_USD" in X.columns:

        income = X["Annual_Income_USD"].astype(float)

        X["Income_Log"] = np.log1p(income)
        X["Income_Sqrt"] = np.sqrt(income)
        X["Income_100k"] = income / 100000.0

    # --------------------------------------------------------
    # Commute
    # --------------------------------------------------------

    if "Daily_Commute_km" in X.columns:

        commute = X["Daily_Commute_km"].astype(float)

        X["Commute_Log"] = np.log1p(commute)
        X["Commute_Sqrt"] = np.sqrt(commute)

    # --------------------------------------------------------
    # Income / commute
    # --------------------------------------------------------

    if {
        "Annual_Income_USD",
        "Daily_Commute_km"
    }.issubset(X.columns):

        income = X["Annual_Income_USD"].astype(float)
        commute = X["Daily_Commute_km"].astype(float)

        X["Income_per_Commute"] = income / (commute + 1)
        X["Commute_per_100k_Income"] = commute / (
            income / 100000.0 + 1e-6
        )

    # --------------------------------------------------------
    # Income / cars
    # --------------------------------------------------------

    if {
        "Annual_Income_USD",
        "Number_of_Cars_Owned"
    }.issubset(X.columns):

        income = X["Annual_Income_USD"].astype(float)
        cars = X["Number_of_Cars_Owned"].astype(float)

        X["Income_per_Car"] = income / (cars + 1)

    # --------------------------------------------------------
    # Stations / cars
    # --------------------------------------------------------

    if {
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Number_of_Cars_Owned"
    }.issubset(X.columns):

        stations = (
            X["Charging_Stations_Near_Home"] +
            X["Charging_Stations_Near_Work"]
        )

        cars = X["Number_of_Cars_Owned"]

        X["Stations_per_Car"] = stations / (cars + 1)

    # --------------------------------------------------------
    # EV domain score
    # --------------------------------------------------------

    score_parts = []

    if "Environmental_Concern_Level" in X.columns:
        score_parts.append(
            X["Environmental_Concern_Level"].astype(float)
        )

    if "Charging_Stations_Near_Home" in X.columns:
        score_parts.append(
            X["Charging_Stations_Near_Home"].astype(float)
        )

    if "Charging_Stations_Near_Work" in X.columns:
        score_parts.append(
            X["Charging_Stations_Near_Work"].astype(float)
        )

    if score_parts:
        X["EV_Domain_Score"] = sum(score_parts)

    return X


# ============================================================
# INTERACTION FEATURES
# ============================================================

def add_interaction_features(X):

    X = X.copy()

    # --------------------------------------------------------
    # Numeric interactions
    # --------------------------------------------------------

    if {
        "Environmental_Concern_Level",
        "Annual_Income_USD"
    }.issubset(X.columns):

        X["Env_x_Income"] = (
            X["Environmental_Concern_Level"]
            * X["Annual_Income_USD"] / 100000
        )

    if {
        "Environmental_Concern_Level",
        "Daily_Commute_km"
    }.issubset(X.columns):

        X["Env_x_Commute"] = (
            X["Environmental_Concern_Level"]
            * X["Daily_Commute_km"]
        )

    if {
        "Environmental_Concern_Level",
        "Charging_Stations_Near_Home"
    }.issubset(X.columns):

        X["Env_x_HomeCharging"] = (
            X["Environmental_Concern_Level"]
            * X["Charging_Stations_Near_Home"]
        )

    if {
        "Environmental_Concern_Level",
        "Charging_Stations_Near_Work"
    }.issubset(X.columns):

        X["Env_x_WorkCharging"] = (
            X["Environmental_Concern_Level"]
            * X["Charging_Stations_Near_Work"]
        )

    # --------------------------------------------------------
    # Categorical interactions
    # --------------------------------------------------------

    def make_interaction(a, b, name):

        if a in X.columns and b in X.columns:

            X[name] = (
                X[a].astype(str)
                + "__"
                + X[b].astype(str)
            )

    make_interaction(
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Env__Subsidy"
    )

    make_interaction(
        "Environmental_Concern_Level",
        "Range_Anxiety_Level",
        "Env__RangeAnxiety"
    )

    make_interaction(
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Subsidy__RangeAnxiety"
    )

    make_interaction(
        "Home_Charging_Possible",
        "Subsidy_Available",
        "HomeCharge__Subsidy"
    )

    make_interaction(
        "Current_Car_Type",
        "City_Type",
        "Car__City"
    )

    return X


# ============================================================
# FREQUENCY ENCODING
# ============================================================

def add_frequency_features(train_X, valid_X):

    train_X = train_X.copy()
    valid_X = valid_X.copy()

    for col in [
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Age",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work"
    ]:

        if col not in train_X.columns:
            continue

        freq = train_X[col].value_counts(
            normalize=True,
            dropna=False
        )

        train_X[f"{col}_Freq"] = (
            train_X[col].map(freq).fillna(0)
        )

        valid_X[f"{col}_Freq"] = (
            valid_X[col].map(freq).fillna(0)
        )

    return train_X, valid_X


# ============================================================
# LEAKAGE-SAFE TARGET ENCODING
# ============================================================

def add_target_encoding(
    train_X,
    train_y,
    valid_X,
    columns,
    smoothing=30
):

    train_X = train_X.copy()
    valid_X = valid_X.copy()

    global_mean = np.mean(train_y)

    temp = train_X[columns].copy()
    temp["_target_"] = train_y

    stats = (
        temp
        .groupby(columns, dropna=False)["_target_"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Smoothed target encoding
    stats["encoding"] = (
        (
            stats["mean"] * stats["count"]
            + global_mean * smoothing
        )
        /
        (
            stats["count"] + smoothing
        )
    )

    name = "TE_" + "_".join(columns)

    mapping = stats.set_index(columns)["encoding"]

    train_keys = pd.MultiIndex.from_frame(train_X[columns])
    valid_keys = pd.MultiIndex.from_frame(valid_X[columns])

    train_X[name] = (
        pd.Series(
            mapping.reindex(train_keys).values,
            index=train_X.index
        )
        .fillna(global_mean)
        .values
    )

    valid_X[name] = (
        pd.Series(
            mapping.reindex(valid_keys).values,
            index=valid_X.index
        )
        .fillna(global_mean)
        .values
    )

    return train_X, valid_X


# ============================================================
# DIGIT FEATURES
# ============================================================

def add_digit_features(X):

    X = X.copy()

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    if "Annual_Income_USD" in X.columns:

        v = X["Annual_Income_USD"].astype(np.int64)

        X["Income_LastDigit"] = v % 10
        X["Income_Last2"] = v % 100
        X["Income_Mod3"] = v % 3
        X["Income_Mod7"] = v % 7
        X["Income_Mod9"] = v % 9
        X["Income_Mod11"] = v % 11

        X["Income_FirstDigit"] = (
            v // 10000
        )

    # --------------------------------------------------------
    # Commute
    # --------------------------------------------------------

    if "Daily_Commute_km" in X.columns:

        # Convert decimal to integer representation
        v = (
            X["Daily_Commute_km"]
            .astype(float)
            .mul(10)
            .round()
            .astype(np.int64)
        )

        X["Commute_LastDigit"] = v % 10
        X["Commute_Last2"] = v % 100
        X["Commute_Mod3"] = v % 3
        X["Commute_Mod7"] = v % 7

    return X


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    experiment_name,
    feature_mode
):

    print("\n")
    print("=" * 70)
    print("EXPERIMENT:", experiment_name)
    print("=" * 70)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(X_raw, y), 1
    ):

        print(f"\nFold {fold}/{N_SPLITS}")

        train_X = X_raw.iloc[tr_idx].copy()
        valid_X = X_raw.iloc[va_idx].copy()

        train_y = y[tr_idx]
        valid_y = y[va_idx]

        # ----------------------------------------------------
        # Feature groups
        # ----------------------------------------------------

        if "domain" in feature_mode:
            train_X = add_domain_features(train_X)
            valid_X = add_domain_features(valid_X)

        if "interaction" in feature_mode:
            train_X = add_interaction_features(train_X)
            valid_X = add_interaction_features(valid_X)

        if "frequency" in feature_mode:

            train_X, valid_X = add_frequency_features(
                train_X,
                valid_X
            )

        if "target_encoding_income" in feature_mode:

            train_X, valid_X = add_target_encoding(
                train_X,
                train_y,
                valid_X,
                ["Annual_Income_USD"],
                smoothing=50
            )

        if "target_encoding_commute" in feature_mode:

            train_X, valid_X = add_target_encoding(
                train_X,
                train_y,
                valid_X,
                ["Daily_Commute_km"],
                smoothing=50
            )

        if "target_encoding_combo" in feature_mode:

            # Important categorical combination
            cols = []

            for c in [
                "Environmental_Concern_Level",
                "Subsidy_Available",
                "Range_Anxiety_Level"
            ]:
                if c in train_X.columns:
                    cols.append(c)

            if len(cols) >= 2:

                train_X, valid_X = add_target_encoding(
                    train_X,
                    train_y,
                    valid_X,
                    cols,
                    smoothing=100
                )

        if "digits" in feature_mode:

            train_X = add_digit_features(train_X)
            valid_X = add_digit_features(valid_X)

        # ----------------------------------------------------
        # Make sure feature columns match
        # ----------------------------------------------------

        train_X, valid_X = prepare_categories(
            train_X,
            valid_X
        )

        # Convert everything numeric
        for c in train_X.columns:

            train_X[c] = pd.to_numeric(
                train_X[c],
                errors="coerce"
            )

            valid_X[c] = pd.to_numeric(
                valid_X[c],
                errors="coerce"
            )

        # Fill missing
        train_X = train_X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid_X = valid_X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        train_X = train_X.fillna(-999)
        valid_X = valid_X.fillna(-999)

        print("Features:", train_X.shape[1])

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        auc, model, pred = train_lgbm(
            train_X,
            train_y,
            valid_X,
            valid_y
        )

        fold_scores.append(auc)

        print(
            f"Fold {fold} AUC: {auc:.6f}"
        )

        print(
            "Best iteration:",
            model.best_iteration_
        )

        del train_X
        del valid_X

    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)

    print("\nRESULT:")
    print(
        f"{experiment_name}: "
        f"{mean_auc:.6f} ± {std_auc:.6f}"
    )

    return mean_auc


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = [

    (
        "1_BASELINE",
        []
    ),

    (
        "2_DOMAIN",
        [
            "domain"
        ]
    ),

    (
        "3_DOMAIN_INTERACTION",
        [
            "domain",
            "interaction"
        ]
    ),

    (
        "4_DOMAIN_INTERACTION_FREQUENCY",
        [
            "domain",
            "interaction",
            "frequency"
        ]
    ),

    (
        "5_PLUS_INCOME_TE",
        [
            "domain",
            "interaction",
            "frequency",
            "target_encoding_income"
        ]
    ),

    (
        "6_PLUS_COMMUTE_TE",
        [
            "domain",
            "interaction",
            "frequency",
            "target_encoding_income",
            "target_encoding_commute"
        ]
    ),

    (
        "7_PLUS_COMBO_TE",
        [
            "domain",
            "interaction",
            "frequency",
            "target_encoding_income",
            "target_encoding_commute",
            "target_encoding_combo"
        ]
    ),

    (
        "8_PLUS_DIGITS",
        [
            "domain",
            "interaction",
            "frequency",
            "target_encoding_income",
            "target_encoding_commute",
            "target_encoding_combo",
            "digits"
        ]
    ),
]


# ============================================================
# RUN
# ============================================================

results = []

for name, mode in experiments:

    auc = run_experiment(
        name,
        mode
    )

    results.append({
        "Experiment": name,
        "AUC": auc
    })


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

baseline_auc = results_df.loc[
    results_df["Experiment"] == "1_BASELINE",
    "AUC"
].iloc[0]

results_df["Delta_vs_Baseline"] = (
    results_df["AUC"] - baseline_auc
)

results_df = results_df.sort_values(
    "AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.7f}"
    )
)

results_df.to_csv(
    "ablation_results.csv",
    index=False
)

print("\nSaved:")
print("ablation_results.csv")

print("\nDONE.")