# ============================================================
# S6E9 - Meta Stacking Preparation
# LGBM + XGBoost + CatBoost
# Generate OOF + Test predictions
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

TRAIN_PATH = "train(1).csv"
TEST_PATH = "test(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

SEED = 42
N_FOLDS = 5

# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train:", train.shape)
print("Test :", test.shape)

# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = (
    train[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if y.isna().any():
    raise ValueError("Target contains unexpected values.")

y = y.astype(int)

print("\nTarget:")
print(y.value_counts())
print("Positive rate:", y.mean())

# ------------------------------------------------------------
# EXACT 23 INCOME FEATURES
# ------------------------------------------------------------

def add_income_features(df):
    df = df.copy()

    income = pd.to_numeric(
        df["Annual_Income_USD"],
        errors="coerce"
    ).fillna(0).astype(np.int64)

    # Digits
    df["Income_LastDigit"] = income % 10
    df["Income_Last2Digits"] = income % 100
    df["Income_Last3Digits"] = income % 1000

    # Modulo features
    for m in range(2, 13):
        df[f"Income_Mod{m}"] = income % m

    # Decimal digits
    df["Income_TensDigit"] = (income // 10) % 10
    df["Income_HundredsDigit"] = (income // 100) % 10
    df["Income_ThousandsDigit"] = (income // 1000) % 10
    df["Income_10ThousandsDigit"] = (income // 10000) % 10

    # Decomposition
    df["Income_Thousands"] = income // 1000
    df["Income_Hundreds"] = income // 100
    df["Income_Tens"] = income // 10

    # Remainders
    df["Income_Thousands_Remainder"] = income % 1000
    df["Income_Hundreds_Remainder"] = income % 100
    df["Income_Tens_Remainder"] = income % 10

    return df


print("\nCreating income features...")

train_fe = add_income_features(train.drop(columns=[TARGET]))
test_fe = add_income_features(test.copy())

print("Train after FE:", train_fe.shape)
print("Test after FE :", test_fe.shape)

# ------------------------------------------------------------
# VERIFY 37-FEATURE BASE
# ------------------------------------------------------------

feature_cols = [c for c in train_fe.columns if c != TARGET]

# We expect 37 columns including id
print("\nNumber of features:", len(feature_cols))

if len(feature_cols) != 37:
    print("WARNING: Feature count is not 37.")
    print(feature_cols)

# ------------------------------------------------------------
# CATEGORICAL COLUMNS
# ------------------------------------------------------------

categorical_cols = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Environmental_Concern_Level",
    "Range_Anxiety_Level",
]

# Make categorical values consistent
for col in categorical_cols:
    train_fe[col] = train_fe[col].astype("category")

    # Same categories in train/test
    categories = train_fe[col].cat.categories

    test_fe[col] = pd.Categorical(
        test_fe[col],
        categories=categories
    )

# ------------------------------------------------------------
# FOLDS
# ------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

folds = list(skf.split(train_fe, y))

print("\nFolds created.")

# ============================================================
# MODEL 1 — LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 — LIGHTGBM")
print("=" * 70)

try:
    from lightgbm import (
        LGBMClassifier,
        early_stopping,
        log_evaluation
    )

    print("LightGBM import: OK")

except Exception as e:
    print("\nLIGHTGBM IMPORT FAILED")
    print("-" * 70)
    print(type(e).__name__, ":", e)
    print("-" * 70)
    raise RuntimeError(
        "LightGBM هنوز روی محیط Python تو درست نصب/لود نشده است."
    )

X_lgb = train_fe[feature_cols].copy()
Xtest_lgb = test_fe[feature_cols].copy()

oof_lgb = np.zeros(len(train_fe))
pred_lgb = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):

    print(f"\nLGBM Fold {fold}/{N_FOLDS}")

    X_tr = X_lgb.iloc[tr_idx]
    X_va = X_lgb.iloc[va_idx]

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = LGBMClassifier(
        objective="binary",
        metric="auc",

        n_estimators=2000,
        learning_rate=0.05,

        num_leaves=63,
        max_depth=-1,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.1,
        reg_lambda=1.0,

        min_child_samples=30,

        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_tr,
        y_tr,

        eval_set=[(X_va, y_va)],

        callbacks=[
            early_stopping(
                100,
                verbose=False
            ),
            log_evaluation(0)
        ]
    )

    oof_lgb[va_idx] = model.predict_proba(
        X_va
    )[:, 1]

    pred_lgb += model.predict_proba(
        Xtest_lgb
    )[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(
        y_va,
        oof_lgb[va_idx]
    )

    print(
        f"Fold {fold} AUC: {fold_auc:.6f} | "
        f"Best iteration: {model.best_iteration_}"
    )

lgb_auc = roc_auc_score(y, oof_lgb)

print("\nLGBM OOF AUC:", f"{lgb_auc:.6f}")

# ------------------------------------------------------------
# SAVE LGBM
# ------------------------------------------------------------

pd.DataFrame({
    ID_COL: train[ID_COL],
    "prediction": oof_lgb
}).to_csv(
    "lgbm_oof.csv",
    index=False
)

pd.DataFrame({
    ID_COL: test[ID_COL],
    "prediction": pred_lgb
}).to_csv(
    "lgbm_test.csv",
    index=False
)

# ============================================================
# MODEL 2 — XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 — XGBOOST")
print("=" * 70)

try:
    from xgboost import XGBClassifier

    print("XGBoost import: OK")

except Exception as e:
    print("\nXGBoost import failed:")
    print(e)
    raise

# ------------------------------------------------------------
# XGBoost preprocessing
# ------------------------------------------------------------

X_xgb = train_fe[feature_cols].copy()
Xtest_xgb = test_fe[feature_cols].copy()

# XGBoost gets numeric category codes
for col in categorical_cols:

    categories = train_fe[col].cat.categories

    X_xgb[col] = (
        pd.Categorical(
            X_xgb[col],
            categories=categories
        ).codes
    )

    Xtest_xgb[col] = (
        pd.Categorical(
            Xtest_xgb[col],
            categories=categories
        ).codes
    )

X_xgb = X_xgb.astype(np.float32)
Xtest_xgb = Xtest_xgb.astype(np.float32)

oof_xgb = np.zeros(len(train_fe))
pred_xgb = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):

    print(f"\nXGB Fold {fold}/{N_FOLDS}")

    X_tr = X_xgb.iloc[tr_idx]
    X_va = X_xgb.iloc[va_idx]

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",

        n_estimators=1500,
        learning_rate=0.05,

        max_depth=7,
        min_child_weight=5,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.1,
        reg_lambda=1.0,

        tree_method="hist",

        random_state=SEED + fold,
        n_jobs=-1,

        verbosity=0
    )

    model.fit(
        X_tr,
        y_tr,

        eval_set=[(X_va, y_va)],

        verbose=False
    )

    oof_xgb[va_idx] = model.predict_proba(
        X_va
    )[:, 1]

    pred_xgb += model.predict_proba(
        Xtest_xgb
    )[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(
        y_va,
        oof_xgb[va_idx]
    )

    print(
        f"Fold {fold} AUC: {fold_auc:.6f}"
    )

xgb_auc = roc_auc_score(y, oof_xgb)

print("\nXGBoost OOF AUC:", f"{xgb_auc:.6f}")

# ------------------------------------------------------------
# SAVE XGB
# ------------------------------------------------------------

pd.DataFrame({
    ID_COL: train[ID_COL],
    "prediction": oof_xgb
}).to_csv(
    "xgb_oof.csv",
    index=False
)

pd.DataFrame({
    ID_COL: test[ID_COL],
    "prediction": pred_xgb
}).to_csv(
    "xgb_test.csv",
    index=False
)

# ============================================================
# MODEL 3 — CATBOOST
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3 — CATBOOST")
print("=" * 70)

try:
    from catboost import CatBoostClassifier

    print("CatBoost import: OK")

except Exception as e:
    print("\nCatBoost import failed:")
    print(e)
    raise

X_cat = train_fe[feature_cols].copy()
Xtest_cat = test_fe[feature_cols].copy()

# CatBoost needs categorical columns as strings
for col in categorical_cols:

    X_cat[col] = X_cat[col].astype(str)
    Xtest_cat[col] = Xtest_cat[col].astype(str)

cat_indices = [
    X_cat.columns.get_loc(col)
    for col in categorical_cols
]

oof_cat = np.zeros(len(train_fe))
pred_cat = np.zeros(len(test_fe))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):

    print(f"\nCatBoost Fold {fold}/{N_FOLDS}")

    X_tr = X_cat.iloc[tr_idx]
    X_va = X_cat.iloc[va_idx]

    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",

        iterations=1500,
        learning_rate=0.05,

        depth=8,

        l2_leaf_reg=5,

        random_seed=SEED + fold,

        verbose=False,

        thread_count=-1,

        allow_writing_files=False
    )

    model.fit(
        X_tr,
        y_tr,

        cat_features=cat_indices,

        eval_set=(X_va, y_va),

        early_stopping_rounds=100,

        verbose=False
    )

    oof_cat[va_idx] = model.predict_proba(
        X_va
    )[:, 1]

    pred_cat += model.predict_proba(
        Xtest_cat
    )[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(
        y_va,
        oof_cat[va_idx]
    )

    print(
        f"Fold {fold} AUC: {fold_auc:.6f} | "
        f"Best iteration: {model.get_best_iteration()}"
    )

cat_auc = roc_auc_score(y, oof_cat)

print("\nCatBoost OOF AUC:", f"{cat_auc:.6f}")

# ------------------------------------------------------------
# SAVE CATBOOST
# ------------------------------------------------------------

pd.DataFrame({
    ID_COL: train[ID_COL],
    "prediction": oof_cat
}).to_csv(
    "catboost_oof.csv",
    index=False
)

pd.DataFrame({
    ID_COL: test[ID_COL],
    "prediction": pred_cat
}).to_csv(
    "catboost_test.csv",
    index=False
)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame({
    "Model": [
        "LGBM",
        "XGBoost",
        "CatBoost"
    ],
    "OOF_AUC": [
        lgb_auc,
        xgb_auc,
        cat_auc
    ]
})

results = results.sort_values(
    "OOF_AUC",
    ascending=False
)

print("\nAUC:")
print(results.to_string(index=False))

# ------------------------------------------------------------
# Rank correlation
# ------------------------------------------------------------

rank_preds = pd.DataFrame({
    "LGBM": pd.Series(oof_lgb).rank(pct=True),
    "XGBoost": pd.Series(oof_xgb).rank(pct=True),
    "CatBoost": pd.Series(oof_cat).rank(pct=True),
})

corr = rank_preds.corr(
    method="spearman"
)

print("\n" + "=" * 70)
print("SPEARMAN RANK CORRELATION")
print("=" * 70)

print(corr.round(6))

# ============================================================
# SIMPLE RANK MEAN
# ============================================================

rank_mean_oof = rank_preds.mean(axis=1)

rank_mean_auc = roc_auc_score(
    y,
    rank_mean_oof
)

print("\nSimple Rank Mean AUC:", f"{rank_mean_auc:.6f}")

# ============================================================
# SAVE ALL OOF
# ============================================================

all_oof = pd.DataFrame({
    ID_COL: train[ID_COL],
    "y": y,
    "LGBM": oof_lgb,
    "XGBoost": oof_xgb,
    "CatBoost": oof_cat,
})

all_test = pd.DataFrame({
    ID_COL: test[ID_COL],
    "LGBM": pred_lgb,
    "XGBoost": pred_xgb,
    "CatBoost": pred_cat,
})

all_oof.to_csv(
    "meta_oof_predictions.csv",
    index=False
)

all_test.to_csv(
    "meta_test_predictions.csv",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"LGBM      AUC: {lgb_auc:.6f}")
print(f"XGBoost   AUC: {xgb_auc:.6f}")
print(f"CatBoost  AUC: {cat_auc:.6f}")
print(f"Rank Mean AUC: {rank_mean_auc:.6f}")

print("\nFiles created:")
print("  lgbm_oof.csv")
print("  lgbm_test.csv")
print("  xgb_oof.csv")
print("  xgb_test.csv")
print("  catboost_oof.csv")
print("  catboost_test.csv")
print("  meta_oof_predictions.csv")
print("  meta_test_predictions.csv")

print("\nNext step:")
print("Send me the AUCs + Spearman correlation matrix.")
print("Then we will build the Logistic Meta + Hill Climbing stage.")