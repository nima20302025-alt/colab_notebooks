import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

N_SPLITS = 5
SEED = 42

N_ESTIMATORS = 2500
EARLY_STOPPING = 100

SMOOTHING_VALUES = [10, 30, 50, 100, 200]


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

print("\nRaw target:")
print(df[TARGET].value_counts(dropna=False))

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

y_series = target_raw.map(target_map)

if y_series.isna().any():

    print("\nUnknown target values:")
    print(
        target_raw[y_series.isna()].unique()
    )

    raise ValueError(
        "Target conversion failed."
    )

y = y_series.astype(np.int8).values

print("\nConverted target:")
print(pd.Series(y).value_counts())

print("Positive rate:", y.mean())


# ============================================================
# FEATURES
# ============================================================

X = df.drop(
    columns=[TARGET, ID_COL],
    errors="ignore"
).copy()

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# COMBO
# ============================================================

COMBO_COLS = [
    "Environmental_Concern_Level",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]

for col in COMBO_COLS:

    if col not in X.columns:
        raise ValueError(
            f"Missing column: {col}"
        )


# ============================================================
# OOF TARGET ENCODING
# ============================================================

def create_oof_target_encoding(
    X,
    y,
    smoothing,
    n_splits=5,
    seed=42
):

    oof_te = np.zeros(len(X), dtype=np.float64)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        1
    ):

        print(
            f"    TE Fold {fold}/{n_splits}"
        )

        train_part = X.iloc[
            train_idx
        ][COMBO_COLS].copy()

        valid_part = X.iloc[
            valid_idx
        ][COMBO_COLS].copy()

        train_target = y[train_idx]

        global_mean = train_target.mean()

        temp = train_part.copy()

        temp["_target"] = train_target

        stats = (
            temp
            .groupby(
                COMBO_COLS,
                dropna=False
            )["_target"]
            .agg(["mean", "count"])
            .reset_index()
        )

        stats["encoding"] = (
            (
                stats["mean"]
                * stats["count"]
                +
                global_mean
                * smoothing
            )
            /
            (
                stats["count"]
                + smoothing
            )
        )

        mapping = (
            stats
            .set_index(COMBO_COLS)["encoding"]
        )

        valid_keys = pd.MultiIndex.from_frame(
            valid_part[COMBO_COLS]
        )

        encoded = (
            mapping
            .reindex(valid_keys)
            .values
        )

        encoded = np.where(
            np.isnan(encoded),
            global_mean,
            encoded
        )

        oof_te[valid_idx] = encoded

    return oof_te


# ============================================================
# ADD OOF TE TO MODEL DATA
# ============================================================

def prepare_model_data(
    X,
    oof_te
):

    data = X.copy()

    data[
        "OOF_Combo_TE"
    ] = oof_te

    categorical_cols = [
        c for c in data.columns
        if data[c].dtype == "object"
    ]

    for col in categorical_cols:

        categories = (
            data[col]
            .astype("category")
            .cat.categories
        )

        data[col] = pd.Categorical(
            data[col],
            categories=categories
        ).codes

    for col in data.columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.fillna(-999)

    return data


# ============================================================
# LIGHTGBM OOF EVALUATION
# ============================================================

def evaluate_oof_model(
    X_model,
    y,
    seed=42
):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    oof_pred = np.zeros(
        len(y),
        dtype=np.float64
    )

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X_model, y),
        1
    ):

        print(
            f"    Model Fold {fold}/{N_SPLITS}"
        )

        X_train = X_model.iloc[
            train_idx
        ]

        X_valid = X_model.iloc[
            valid_idx
        ]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

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

            random_state=seed,

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

        oof_pred[valid_idx] = pred

        fold_auc = roc_auc_score(
            y_valid,
            pred
        )

        fold_scores.append(
            fold_auc
        )

        print(
            f"        AUC: {fold_auc:.7f}"
        )

        print(
            f"        Best iteration: "
            f"{model.best_iteration_}"
        )

        del model

    overall_auc = roc_auc_score(
        y,
        oof_pred
    )

    return (
        overall_auc,
        np.mean(fold_scores),
        np.std(fold_scores)
    )


# ============================================================
# RUN
# ============================================================

results = []

print("\n")
print("=" * 70)
print("OOF COMBO TARGET ENCODING TEST")
print("=" * 70)

for smoothing in SMOOTHING_VALUES:

    print("\n")
    print("-" * 70)
    print(
        f"SMOOTHING = {smoothing}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # TRUE OOF TARGET ENCODING
    # --------------------------------------------------------

    oof_te = create_oof_target_encoding(
        X,
        y,
        smoothing=smoothing,
        n_splits=N_SPLITS,
        seed=SEED
    )

    print("\nOOF TE statistics:")

    print(
        "Min:",
        oof_te.min()
    )

    print(
        "Max:",
        oof_te.max()
    )

    print(
        "Mean:",
        oof_te.mean()
    )

    print(
        "Std:",
        oof_te.std()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    X_model = prepare_model_data(
        X,
        oof_te
    )

    print(
        "\nNumber of features:",
        X_model.shape[1]
    )

    overall_auc, mean_fold_auc, std_fold_auc = (
        evaluate_oof_model(
            X_model,
            y,
            seed=SEED
        )
    )

    print("\nRESULT")

    print(
        f"Overall OOF AUC: "
        f"{overall_auc:.7f}"
    )

    print(
        f"Mean Fold AUC: "
        f"{mean_fold_auc:.7f}"
    )

    print(
        f"Std Fold AUC: "
        f"{std_fold_auc:.7f}"
    )

    results.append({

        "Smoothing": smoothing,

        "Overall_OOF_AUC":
            overall_auc,

        "Mean_Fold_AUC":
            mean_fold_auc,

        "Std_Fold_AUC":
            std_fold_auc
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Overall_OOF_AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("FINAL OOF COMBO TE RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.7f}"
    )
)

results_df.to_csv(
    "combo_te_oof_results.csv",
    index=False
)

print("\nSaved:")
print("combo_te_oof_results.csv")

print("\nFinished.")