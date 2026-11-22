# ============================================================
# EV S6E9 - FEATURE ENGINEERING ABLATION
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


# ============================================================
# LOAD
# ============================================================

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
        target_raw[y.isna()].value_counts()
    )

y = y.astype(np.int8)


# ============================================================
# BASE DATA
# ============================================================

X_raw = df.drop(
    columns=[TARGET, ID_COL]
).copy()


# ============================================================
# FOLDS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

folds = list(
    skf.split(X_raw, y)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    use_income=False,
    use_commute=False,
    use_charging=False,
    use_ratios=False,
    use_interactions=False,
    use_digits=False
):

    X = df.copy()

    eps = 1e-6

    # ========================================================
    # INCOME FEATURES
    # ========================================================

    if use_income:

        income = X["Annual_Income_USD"].astype(float)

        X["Income_Log"] = np.log1p(
            income
        )

        X["Income_Sqrt"] = np.sqrt(
            income
        )

        X["Income_Per_Age"] = (
            income /
            (X["Age"] + 1)
        )


    # ========================================================
    # COMMUTE FEATURES
    # ========================================================

    if use_commute:

        commute = X[
            "Daily_Commute_km"
        ].astype(float)

        X["Commute_Log"] = np.log1p(
            commute
        )

        X["Commute_Sqrt"] = np.sqrt(
            commute
        )

        X["Commute_Sq"] = (
            commute ** 2
        )


    # ========================================================
    # CHARGING FEATURES
    # ========================================================

    if use_charging:

        home = X[
            "Charging_Stations_Near_Home"
        ].astype(float)

        work = X[
            "Charging_Stations_Near_Work"
        ].astype(float)

        X["Charging_Total"] = (
            home + work
        )

        X["Charging_Diff"] = (
            home - work
        )

        X["Charging_AbsDiff"] = (
            np.abs(home - work)
        )

        X["Charging_Min"] = (
            np.minimum(home, work)
        )

        X["Charging_Max"] = (
            np.maximum(home, work)
        )


    # ========================================================
    # RATIO FEATURES
    # ========================================================

    if use_ratios:

        income = (
            X["Annual_Income_USD"]
            .astype(float)
        )

        commute = (
            X["Daily_Commute_km"]
            .astype(float)
        )

        cars = (
            X["Number_of_Cars_Owned"]
            .astype(float)
        )

        home = (
            X["Charging_Stations_Near_Home"]
            .astype(float)
        )

        work = (
            X["Charging_Stations_Near_Work"]
            .astype(float)
        )

        total_stations = (
            home + work
        )

        # Income / usage
        X["Income_Per_Commute"] = (
            income /
            (commute + eps)
        )

        X["Commute_Per_Income"] = (
            commute /
            (income / 100000 + eps)
        )

        # Income / cars
        X["Income_Per_Car"] = (
            income /
            (cars + 1)
        )

        X["Cars_Per_100k_Income"] = (
            cars /
            (income / 100000 + eps)
        )

        # Charging / cars
        X["Stations_Per_Car"] = (
            total_stations /
            (cars + 1)
        )

        # Charging relative to commute
        X["Stations_Per_Commute"] = (
            total_stations /
            (commute + eps)
        )


    # ========================================================
    # INTERACTION FEATURES
    # ========================================================

    if use_interactions:

        # Income × environmental concern
        X["Income_x_Environmental"] = (
            X["Annual_Income_USD"]
            *
            X["Environmental_Concern_Level"]
        )

        # Income × subsidy
        subsidy_num = (
            X["Subsidy_Available"]
            .astype(str)
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0
            })
            .fillna(0)
        )

        X["Income_x_Subsidy"] = (
            X["Annual_Income_USD"]
            * subsidy_num
        )

        # Income × range anxiety
        range_num = (
            X["Range_Anxiety_Level"]
            .astype(str)
            .map({
                "Low": 0,
                "Medium": 1,
                "High": 2
            })
            .fillna(-1)
        )

        X["Income_x_RangeAnxiety"] = (
            X["Annual_Income_USD"]
            * range_num
        )

        # Commute × range anxiety
        X["Commute_x_RangeAnxiety"] = (
            X["Daily_Commute_km"]
            * range_num
        )

        # Environmental × subsidy
        X["Environmental_x_Subsidy"] = (
            X["Environmental_Concern_Level"]
            * subsidy_num
        )

        # Environmental × home charging
        home_charge_num = (
            X["Home_Charging_Possible"]
            .astype(str)
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0
            })
            .fillna(0)
        )

        X["Environmental_x_HomeCharging"] = (
            X["Environmental_Concern_Level"]
            * home_charge_num
        )


    # ========================================================
    # DIGIT FEATURES
    # ========================================================

    if use_digits:

        income = (
            X["Annual_Income_USD"]
            .astype(int)
        )

        commute = (
            X["Daily_Commute_km"]
            .astype(int)
        )

        X["Income_LastDigit"] = (
            income % 10
        )

        X["Income_Last2Digits"] = (
            income % 100
        )

        X["Income_Mod3"] = (
            income % 3
        )

        X["Income_Mod7"] = (
            income % 7
        )

        X["Income_Mod9"] = (
            income % 9
        )

        X["Commute_LastDigit"] = (
            commute % 10
        )

        X["Commute_Last2Digits"] = (
            commute % 100
        )

        X["Commute_Mod3"] = (
            commute % 3
        )

    return X


# ============================================================
# LIGHTGBM
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
# MODEL FUNCTION
# ============================================================

def evaluate_model(X, name):

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].astype("category")

    oof = np.zeros(len(X))

    fold_scores = []

    for fold_id, (
        train_idx,
        valid_idx
    ) in enumerate(folds, 1):

        model = LGBMClassifier(
            **params
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],

            categorical_feature=[
                c for c in categorical_cols
                if c in X.columns
            ]
        )

        pred = model.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

        oof[valid_idx] = pred

        auc = roc_auc_score(
            y.iloc[valid_idx],
            pred
        )

        fold_scores.append(auc)

        print(
            f"  Fold {fold_id}: "
            f"{auc:.6f}"
        )

    overall = roc_auc_score(
        y,
        oof
    )

    print(
        f"  OOF AUC: "
        f"{overall:.6f}"
    )

    return {
        "experiment": name,
        "auc": overall,
        "mean_fold": np.mean(fold_scores),
        "std_fold": np.std(fold_scores),
        "n_features": X.shape[1]
    }


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = [

    (
        "BASELINE",
        {}
    ),

    (
        "BASE + INCOME",
        {
            "use_income": True
        }
    ),

    (
        "BASE + COMMUTE",
        {
            "use_commute": True
        }
    ),

    (
        "BASE + CHARGING",
        {
            "use_charging": True
        }
    ),

    (
        "BASE + RATIOS",
        {
            "use_ratios": True
        }
    ),

    (
        "BASE + INTERACTIONS",
        {
            "use_interactions": True
        }
    ),

    (
        "BASE + DIGITS",
        {
            "use_digits": True
        }
    ),

    (
        "ALL ENGINEERING",
        {
            "use_income": True,
            "use_commute": True,
            "use_charging": True,
            "use_ratios": True,
            "use_interactions": True,
            "use_digits": True
        }
    )
]


# ============================================================
# RUN
# ============================================================

results = []


for name, kwargs in experiments:

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    X_exp = build_features(
        X_raw,
        **kwargs
    )

    print(
        "Features:",
        X_exp.shape[1]
    )

    result = evaluate_model(
        X_exp,
        name
    )

    results.append(result)


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "auc",
    ascending=False
)

print("\n\n" + "=" * 70)
print("FEATURE ENGINEERING ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# GAIN
# ============================================================

baseline_auc = results_df.loc[
    results_df["experiment"] == "BASELINE",
    "auc"
].iloc[0]

results_df["gain_vs_baseline"] = (
    results_df["auc"]
    - baseline_auc
)

print("\n" + "=" * 70)
print("GAIN VS BASELINE")
print("=" * 70)

print(
    results_df[
        [
            "experiment",
            "auc",
            "gain_vs_baseline",
            "n_features"
        ]
    ]
    .sort_values(
        "gain_vs_baseline",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "feature_engineering_ablation.csv",
    index=False
)

print("\nSaved:")
print("feature_engineering_ablation.csv")

print("\nDone.")