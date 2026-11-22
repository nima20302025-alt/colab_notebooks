import warnings
warnings.filterwarnings("ignore")

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
TEST_PATH = "test(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

SEED = 42
N_FOLDS = 5



# ============================================================
# FEATURE ENGINEERING
# 13 BASE + ALL INCOME ENGINEERING = 37 FEATURES
# ============================================================

def create_features(df):

    df = df.copy()


    income = (
        pd.to_numeric(
            df["Annual_Income_USD"],
            errors="coerce"
        )
        .fillna(0)
        .astype(np.int64)
    )


    # Last digits

    df["Income_LastDigit"] = income % 10

    df["Income_Last2Digits"] = income % 100

    df["Income_Last3Digits"] = income % 1000



    # Mod family

    for m in range(2,13):

        df[f"Income_Mod{m}"] = income % m



    # Digit positions

    df["Income_TensDigit"] = (
        income // 10
    ) % 10


    df["Income_HundredsDigit"] = (
        income // 100
    ) % 10


    df["Income_ThousandsDigit"] = (
        income // 1000
    ) % 10


    df["Income_10ThousandsDigit"] = (
        income // 10000
    ) % 10



    # Scale

    df["Income_Thousands"] = income // 1000

    df["Income_Hundreds"] = income // 100

    df["Income_Tens"] = income // 10



    # Remainders

    df["Income_Thousands_Remainder"] = (
        income % 1000
    )


    df["Income_Hundreds_Remainder"] = (
        income % 100
    )


    return df



# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("LOADING DATA")
print("="*70)


train = pd.read_csv(TRAIN_PATH)

test = pd.read_csv(TEST_PATH)


print("Train shape:", train.shape)

print("Test shape :", test.shape)



# ============================================================
# TARGET
# ============================================================


y = (
    train[TARGET]
    .map(
        {
            "No":0,
            "Yes":1
        }
    )
)


if y.isna().sum() > 0:

    raise ValueError(
        "Target conversion failed"
    )


y = y.astype(int)



# ============================================================
# FEATURES
# ============================================================


X = train.drop(
    columns=[TARGET]
)


X_test = test.copy()



X = create_features(X)

X_test = create_features(X_test)



print("\nFeature count:", X.shape[1])



# ============================================================
# CATEGORICAL
# ============================================================


cat_cols = X.select_dtypes(
    include=[
        "object",
        "category",
        "string"
    ]
).columns



for c in cat_cols:

    X[c] = X[c].astype("category")

    X_test[c] = X_test[c].astype("category")



# ============================================================
# MODEL
# ============================================================


params = {


    "n_estimators":4000,

    "learning_rate":0.03,

    "num_leaves":63,

    "subsample":0.85,

    "colsample_bytree":0.85,

    "reg_alpha":0.1,

    "reg_lambda":1.0,

    "min_child_samples":30,

    "random_state":SEED,

    "n_jobs":-1,

    "verbosity":-1

}



# ============================================================
# 5 FOLD TRAIN + TEST PREDICTION
# ============================================================


skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)



oof = np.zeros(len(X))

test_pred = np.zeros(len(X_test))


fold_auc = []



for fold,(tr_idx,val_idx) in enumerate(
    skf.split(X,y),1
):


    print("\n"+"="*50)

    print("FOLD",fold)

    print("="*50)



    model = LGBMClassifier(
        **params
    )



    model.fit(

        X.iloc[tr_idx],

        y.iloc[tr_idx],


        eval_set=[

            (
                X.iloc[val_idx],
                y.iloc[val_idx]
            )

        ],


        categorical_feature=list(cat_cols),


        callbacks=[

            early_stopping(
                100,
                verbose=False
            ),

            log_evaluation(0)

        ]

    )



    val_pred = model.predict_proba(
        X.iloc[val_idx]
    )[:,1]


    oof[val_idx] = val_pred



    auc = roc_auc_score(
        y.iloc[val_idx],
        val_pred
    )


    fold_auc.append(auc)


    print(
        "Fold AUC:",
        round(auc,6)
    )



    test_pred += (

        model.predict_proba(
            X_test
        )[:,1]

        /N_FOLDS

    )



    del model

    gc.collect()



# ============================================================
# RESULTS
# ============================================================


print("\n")
print("="*70)

print(
    "OOF AUC:",
    roc_auc_score(y,oof)
)


print(
    "Mean Fold:",
    np.mean(fold_auc)
)


print("="*70)



# ============================================================
# SUBMISSION
# ============================================================


submission = pd.DataFrame({

    "id":
        test[ID_COL],

    TARGET:
        test_pred

})



submission.to_csv(
    "submission.csv",
    index=False
)



print("\nSubmission created!")

print(
    submission.head()
)


print(
    "Shape:",
    submission.shape
)

print(
    "Saved: submission.csv"
)