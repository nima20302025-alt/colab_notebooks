# ============================================================
# LIGHTGBM HYPERPARAMETER TUNING WITH OPTUNA
# ============================================================

import gc
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "train(1).csv"
TARGET = "Will_Buy_EV"

N_SPLITS = 5
SEED = 42

N_TRIALS = 50          # تعداد trial ها؛ اگر وقت داری بکن 100
TIMEOUT = None         # مثلاً 3600 برای یک ساعت


# ============================================================
# INCOME FEATURES
# ============================================================

INCOME_FEATURES = [
    "Income_LastDigit", "Income_Last2Digits", "Income_Last3Digits",
    "Income_Mod2", "Income_Mod3", "Income_Mod4", "Income_Mod5",
    "Income_Mod6", "Income_Mod7", "Income_Mod8", "Income_Mod9",
    "Income_Mod10", "Income_Mod11", "Income_Mod12",
    "Income_TensDigit", "Income_HundredsDigit",
    "Income_ThousandsDigit", "Income_10ThousandsDigit",
    "Income_Thousands", "Income_Hundreds", "Income_Tens",
    "Income_Thousands_Remainder", "Income_Hundreds_Remainder",
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

target_raw = df[TARGET].astype(str).str.strip().str.lower()

target_map = {
    "yes": 1, "no": 0,
    "1": 1, "0": 0,
    "true": 1, "false": 0,
}

y = target_raw.map(target_map)

if y.isna().any():
    print(target_raw[y.isna()].value_counts())
    raise ValueError("Target conversion failed.")

y = y.astype(np.int8)
print("Target mean:", y.mean())


# ============================================================
# BASE FEATURES
# ============================================================

X = df.drop(columns=[TARGET]).copy()


# ============================================================
# INCOME FEATURE GENERATION
# ============================================================

income = (
    pd.to_numeric(X["Annual_Income_USD"], errors="coerce")
    .fillna(0)
    .astype(np.int64)
)

X["Income_LastDigit"] = income % 10
X["Income_Last2Digits"] = income % 100
X["Income_Last3Digits"] = income % 1000

for m in range(2, 13):
    X[f"Income_Mod{m}"] = income % m

X["Income_TensDigit"] = (income // 10) % 10
X["Income_HundredsDigit"] = (income // 100) % 10
X["Income_ThousandsDigit"] = (income // 1000) % 10
X["Income_10ThousandsDigit"] = (income // 10000) % 10

X["Income_Thousands"] = income // 1000
X["Income_Hundreds"] = income // 100
X["Income_Tens"] = income // 10

X["Income_Thousands_Remainder"] = income % 1000
X["Income_Hundreds_Remainder"] = income % 100


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
]

for col in CAT_COLS:
    if col in X.columns:
        X[col] = X[col].astype("category")


# ============================================================
# FIXED FOLDS (same as baseline for fair comparison)
# ============================================================

print("\nCreating fixed folds...")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

folds = list(skf.split(X, y))
print(f"{N_SPLITS} folds ready.")


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

# Use 3 folds during tuning for speed (instead of 5)
TUNING_FOLDS = folds[:3]

def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",

        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),

        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),

        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),

        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    fold_scores = []

    for fold_idx, (train_idx, valid_idx) in enumerate(TUNING_FOLDS, 1):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = LGBMClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            categorical_feature=CAT_COLS,
            callbacks=[
                early_stopping(100, verbose=False),
                log_evaluation(0),
            ],
        )

        pred = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, pred)
        fold_scores.append(auc)

        del model, X_train, X_valid, y_train, y_valid
        gc.collect()

    mean_auc = float(np.mean(fold_scores))

    # Report intermediate score for pruning
    trial.report(mean_auc, step=0)

    return mean_auc


# ============================================================
# RUN OPTUNA
# ============================================================

print("\n")
print("=" * 70)
print(f"STARTING OPTUNA  ({N_TRIALS} trials)")
print("=" * 70)

# Pruner for early stopping of bad trials
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=0,
)

sampler = optuna.samplers.TPESampler(seed=SEED)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    timeout=TIMEOUT,
    show_progress_bar=True,
    gc_after_trial=True,
)


# ============================================================
# BEST TRIAL
# ============================================================

print("\n")
print("=" * 70)
print("OPTUNA RESULTS")
print("=" * 70)

print(f"Best 3-fold AUC : {study.best_value:.6f}")
print(f"Best trial #    : {study.best_trial.number}")
print("\nBest params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")


# ============================================================
# EVALUATE BEST PARAMS ON ALL 5 FOLDS
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING BEST PARAMS ON ALL 5 FOLDS")
print("=" * 70)

best_params = {
    "objective": "binary",
    "metric": "auc",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "subsample_freq": 1,
    **study.best_params,
}

oof = np.zeros(len(X), dtype=np.float32)
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(folds, 1):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = LGBMClassifier(**best_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=CAT_COLS,
        callbacks=[
            early_stopping(100, verbose=False),
            log_evaluation(0),
        ],
    )

    pred = model.predict_proba(X_valid)[:, 1]
    oof[valid_idx] = pred

    auc = roc_auc_score(y_valid, pred)
    fold_scores.append(auc)
    print(f"Fold {fold}/{N_SPLITS}  AUC = {auc:.6f}")

    del model, X_train, X_valid, y_train, y_valid
    gc.collect()


# ============================================================
# FINAL RESULTS
# ============================================================

overall_auc = roc_auc_score(y, oof)
mean_fold = np.mean(fold_scores)
std_fold = np.std(fold_scores)

print("\n")
print("=" * 70)
print("FINAL RESULTS — TUNED LIGHTGBM")
print("=" * 70)

print(f"Overall OOF AUC : {overall_auc:.6f}")
print(f"Mean Fold AUC   : {mean_fold:.6f}")
print(f"Std Fold AUC    : {std_fold:.6f}")

BASELINE_AUC = 0.944016
gain = overall_auc - BASELINE_AUC

print(f"\nBaseline AUC    : {BASELINE_AUC:.6f}")
print(f"Gain vs baseline: {gain:+.6f}")


# ============================================================
# SAVE
# ============================================================

# Save best params
pd.DataFrame([study.best_params]).to_csv(
    "best_lgb_params.csv", index=False
)

# Save OOF
np.save("oof_tuned_lgb.npy", oof)

# Save study trials
trials_df = study.trials_dataframe()
trials_df.to_csv("optuna_trials.csv", index=False)

print("\nSaved:")
print("  best_lgb_params.csv")
print("  oof_tuned_lgb.npy")
print("  optuna_trials.csv")