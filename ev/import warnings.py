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
        "LightGBM is not installed. Run:\n"
        "python -m pip install lightgbm"
    )


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42

N_ESTIMATORS = 3000
EARLY_STOPPING = 100


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(TRAIN_PATH)

print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())


# ============================================================
# TARGET
# ============================================================

print("\n" + "=" * 70)
print("TARGET")
print("=" * 70)

print("\nRaw target values:")
print(df[TARGET].value_counts(dropna=False))

print("\nUnique target values:")
print(df[TARGET].unique())


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

df[TARGET] = target_raw.map(target_map)


if df[TARGET].isna().any():

    invalid_values = target_raw[df[TARGET].isna()].unique()

    print("\nUnknown target values:")
    print(invalid_values)

    raise ValueError(
        "Target conversion failed."
    )


df[TARGET] = df[TARGET].astype(np.int8)

y = df[TARGET].values

print("\nConverted target:")
print(pd.Series(y).value_counts())

print("Positive rate:", y.mean())


# ============================================================
# RAW FEATURES
# ============================================================

X_raw = df.drop(columns=[TARGET]).copy()

if ID_COL in X_raw.columns:
    X_raw = X_raw.drop(columns=[ID_COL])

print("\n" + "=" * 70)
print("RAW FEATURES")
print("=" * 70)

print("Shape:", X_raw.shape)


categorical_cols = [
    c for c in X_raw.columns
    if X_raw[c].dtype == "object"
]

print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# DOMAIN FEATURES
# ============================================================

def add_domain_features(X):

    X = X.copy()

    # --------------------------------------------------------
    # Charging
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
    # Income
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
    # Income / Commute
    # --------------------------------------------------------

    if {
        "Annual_Income_USD",
        "Daily_Commute_km"
    }.issubset(X.columns):

        income = X["Annual_Income_USD"].astype(float)
        commute = X["Daily_Commute_km"].astype(float)

        X["Income_per_Commute"] = (
            income / (commute + 1)
        )

        X["Commute_per_100k_Income"] = (
            commute / (income / 100000.0 + 1e-6)
        )

    # --------------------------------------------------------
    # Income / Cars
    # --------------------------------------------------------

    if {
        "Annual_Income_USD",
        "Number_of_Cars_Owned"
    }.issubset(X.columns):

        income = X["Annual_Income_USD"].astype(float)
        cars = X["Number_of_Cars_Owned"].astype(float)

        X["Income_per_Car"] = (
            income / (cars + 1)
        )

    # --------------------------------------------------------
    # Stations / Cars
    # --------------------------------------------------------

    if {
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Number_of_Cars_Owned"
    }.issubset(X.columns):

        stations = (
            X["Charging_Stations_Near_Home"]
            + X["Charging_Stations_Near_Work"]
        )

        cars = X["Number_of_Cars_Owned"].astype(float)

        X["Stations_per_Car"] = (
            stations / (cars + 1)
        )

    # --------------------------------------------------------
    # EV domain score
    # --------------------------------------------------------

    score = 0

    if "Environmental_Concern_Level" in X.columns:
        score += X["Environmental_Concern_Level"].astype(float)

    if "Charging_Stations_Near_Home" in X.columns:
        score += X["Charging_Stations_Near_Home"].astype(float)

    if "Charging_Stations_Near_Work" in X.columns:
        score += X["Charging_Stations_Near_Work"].astype(float)

    X["EV_Domain_Score"] = score

    return X


# ============================================================
# INTERACTIONS
# ============================================================

def add_interaction_features(X):

    X = X.copy()

    if {
        "Environmental_Concern_Level",
        "Annual_Income_USD"
    }.issubset(X.columns):

        X["Env_x_Income"] = (
            X["Environmental_Concern_Level"]
            * X["Annual_Income_USD"]
            / 100000
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

    # categorical interactions

    def interaction(a, b, name):

        if a in X.columns and b in X.columns:

            X[name] = (
                X[a].astype(str)
                + "__"
                + X[b].astype(str)
            )

    interaction(
        "Environmental_Concern_Level",
        "Subsidy_Available",
        "Env__Subsidy"
    )

    interaction(
        "Environmental_Concern_Level",
        "Range_Anxiety_Level",
        "Env__RangeAnxiety"
    )

    interaction(
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Subsidy__RangeAnxiety"
    )

    interaction(
        "Home_Charging_Possible",
        "Subsidy_Available",
        "HomeCharge__Subsidy"
    )

    interaction(
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

    columns = [
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Age",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work"
    ]

    for col in columns:

        if col not in train_X.columns:
            continue

        freq = train_X[col].value_counts(
            normalize=True,
            dropna=False
        )

        train_X[f"{col}_Freq"] = (
            train_X[col]
            .map(freq)
            .fillna(0)
        )

        valid_X[f"{col}_Freq"] = (
            valid_X[col]
            .map(freq)
            .fillna(0)
        )

    return train_X, valid_X


# ============================================================
# TARGET ENCODING
# ============================================================

def add_target_encoding(
    train_X,
    train_y,
    valid_X,
    columns,
    smoothing=50
):

    train_X = train_X.copy()
    valid_X = valid_X.copy()

    global_mean = np.mean(train_y)

    temp = train_X[columns].copy()
    temp["_target"] = train_y

    stats = (
        temp
        .groupby(columns, dropna=False)["_target"]
        .agg(["mean", "count"])
        .reset_index()
    )

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

    # Single column
    if len(columns) == 1:

        col = columns[0]

        mapping = stats.set_index(col)["encoding"]

        train_X[name] = (
            train_X[col]
            .map(mapping)
            .fillna(global_mean)
        )

        valid_X[name] = (
            valid_X[col]
            .map(mapping)
            .fillna(global_mean)
        )

    # Multiple columns
    else:

        mapping = (
            stats
            .set_index(columns)["encoding"]
        )

        train_keys = pd.MultiIndex.from_frame(
            train_X[columns]
        )

        valid_keys = pd.MultiIndex.from_frame(
            valid_X[columns]
        )

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

    if "Annual_Income_USD" in X.columns:

        v = (
            X["Annual_Income_USD"]
            .astype(np.int64)
        )

        X["Income_LastDigit"] = v % 10
        X["Income_Last2"] = v % 100
        X["Income_Mod3"] = v % 3
        X["Income_Mod7"] = v % 7
        X["Income_Mod9"] = v % 9
        X["Income_Mod11"] = v % 11
        X["Income_FirstDigit"] = v // 10000

    if "Daily_Commute_km" in X.columns:

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
# CATEGORY ENCODING
# ============================================================

def encode_categories(train_X, valid_X):

    train_X = train_X.copy()
    valid_X = valid_X.copy()

    categorical = [
        c for c in train_X.columns
        if train_X[c].dtype == "object"
    ]

    for col in categorical:

        categories = (
            train_X[col]
            .astype("category")
            .cat.categories
        )

        train_X[col] = pd.Categorical(
            train_X[col],
            categories=categories
        ).codes

        valid_X[col] = pd.Categorical(
            valid_X[col],
            categories=categories
        ).codes

    return train_X, valid_X


# ============================================================
# MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid
):

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

        verbosity=-1
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (X_valid, y_valid)
        ],

        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING,
                verbose=False
            )
        ]
    )

    pred = model.predict_proba(
        X_valid
    )[:, 1]

    auc = roc_auc_score(
        y_valid,
        pred
    )

    return auc, model


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    experiment_name,
    features
):

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X_raw, y),
        1
    ):

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        train_X = X_raw.iloc[
            train_idx
        ].copy()

        valid_X = X_raw.iloc[
            valid_idx
        ].copy()

        train_y = y[train_idx]
        valid_y = y[valid_idx]

        # ----------------------------------------------------
        # Domain
        # ----------------------------------------------------

        if "domain" in features:

            train_X = add_domain_features(
                train_X
            )

            valid_X = add_domain_features(
                valid_X
            )

        # ----------------------------------------------------
        # Interactions
        # ----------------------------------------------------

        if "interaction" in features:

            train_X = add_interaction_features(
                train_X
            )

            valid_X = add_interaction_features(
                valid_X
            )

        # ----------------------------------------------------
        # Frequency
        # ----------------------------------------------------

        if "frequency" in features:

            train_X, valid_X = (
                add_frequency_features(
                    train_X,
                    valid_X
                )
            )

        # ----------------------------------------------------
        # Income TE
        # ----------------------------------------------------

        if "income_te" in features:

            train_X, valid_X = (
                add_target_encoding(
                    train_X,
                    train_y,
                    valid_X,
                    ["Annual_Income_USD"],
                    smoothing=50
                )
            )

        # ----------------------------------------------------
        # Commute TE
        # ----------------------------------------------------

        if "commute_te" in features:

            train_X, valid_X = (
                add_target_encoding(
                    train_X,
                    train_y,
                    valid_X,
                    ["Daily_Commute_km"],
                    smoothing=50
                )
            )

        # ----------------------------------------------------
        # Combo TE
        # ----------------------------------------------------

        if "combo_te" in features:

            combo_cols = [
                "Environmental_Concern_Level",
                "Subsidy_Available",
                "Range_Anxiety_Level"
            ]

            train_X, valid_X = (
                add_target_encoding(
                    train_X,
                    train_y,
                    valid_X,
                    combo_cols,
                    smoothing=100
                )
            )

        # ----------------------------------------------------
        # Digits
        # ----------------------------------------------------

        if "digits" in features:

            train_X = add_digit_features(
                train_X
            )

            valid_X = add_digit_features(
                valid_X
            )

        # ----------------------------------------------------
        # Encode categoricals
        # ----------------------------------------------------

        train_X, valid_X = encode_categories(
            train_X,
            valid_X
        )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for col in train_X.columns:

            train_X[col] = pd.to_numeric(
                train_X[col],
                errors="coerce"
            )

            valid_X[col] = pd.to_numeric(
                valid_X[col],
                errors="coerce"
            )

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

        print(
            "Number of features:",
            train_X.shape[1]
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        auc, model = train_model(
            train_X,
            train_y,
            valid_X,
            valid_y
        )

        fold_scores.append(auc)

        print(
            f"Fold {fold} AUC: {auc:.7f}"
        )

        print(
            "Best iteration:",
            model.best_iteration_
        )

        del train_X
        del valid_X
        del model

    mean_auc = np.mean(
        fold_scores
    )

    std_auc = np.std(
        fold_scores
    )

    print("\nExperiment result:")
    print(
        f"{experiment_name}: "
        f"{mean_auc:.7f} "
        f"+/- {std_auc:.7f}"
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
            "income_te"
        ]
    ),

    (
        "6_PLUS_COMMUTE_TE",
        [
            "domain",
            "interaction",
            "frequency",
            "income_te",
            "commute_te"
        ]
    ),

    (
        "7_PLUS_COMBO_TE",
        [
            "domain",
            "interaction",
            "frequency",
            "income_te",
            "commute_te",
            "combo_te"
        ]
    ),

    (
        "8_PLUS_DIGITS",
        [
            "domain",
            "interaction",
            "frequency",
            "income_te",
            "commute_te",
            "combo_te",
            "digits"
        ]
    )
]


# ============================================================
# RUN ALL
# ============================================================

results = []

for name, feature_set in experiments:

    auc = run_experiment(
        name,
        feature_set
    )

    results.append({
        "Experiment": name,
        "AUC": auc
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

baseline_auc = results_df.loc[
    results_df["Experiment"]
    == "1_BASELINE",
    "AUC"
].iloc[0]

results_df[
    "Delta_vs_Baseline"
] = (
    results_df["AUC"]
    - baseline_auc
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

print("\nFinished successfully.")