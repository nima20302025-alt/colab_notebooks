import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score


# =====================================================
# CONFIG
# =====================================================

TRAIN_PATH = "train(1).csv"
TEST_PATH = "test(1).csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

OUTPUT = "submission.csv"


# =====================================================
# FEATURE ENGINEERING
# 37 FEATURES
# =====================================================

def create_income_features(df):

    df = df.copy()

    income = df["Annual_Income_USD"].astype(int)


    # digits
    df["Income_LastDigit"] = income % 10
    df["Income_Last2Digits"] = income % 100
    df["Income_Last3Digits"] = income % 1000


    # mods
    for m in range(2,13):
        df[f"Income_Mod{m}"] = income % m


    # digits positions
    df["Income_TensDigit"] = (income // 10) % 10
    df["Income_HundredsDigit"] = (income // 100) % 10
    df["Income_ThousandsDigit"] = (income // 1000) % 10
    df["Income_10ThousandsDigit"] = (income // 10000) % 10


    # scale parts
    df["Income_Thousands"] = income // 1000
    df["Income_Hundreds"] = income // 100
    df["Income_Tens"] = income // 10


    # remainders
    df["Income_Thousands_Remainder"] = income % 1000
    df["Income_Hundreds_Remainder"] = income % 100


    return df



# =====================================================
# LOAD DATA
# =====================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)


print("Train:", train.shape)
print("Test :", test.shape)



# =====================================================
# TARGET
# =====================================================

y = train[TARGET].map({
    "No":0,
    "Yes":1
})


# remove target

X = train.drop(columns=[TARGET])


# keep test copy
X_test = test.copy()



# =====================================================
# FEATURE ENGINEERING
# =====================================================

X = create_income_features(X)
X_test = create_income_features(X_test)



print("Features:", X.shape[1])



# =====================================================
# CATEGORICAL HANDLING
# =====================================================

cat_cols = X.select_dtypes(
    include=["object","category","string"]
).columns


for c in cat_cols:

    X[c] = X[c].astype("category")

    X_test[c] = X_test[c].astype(
        "category"
    )



# =====================================================
# LIGHTGBM
# =====================================================


params = {

    "n_estimators":4000,
    "learning_rate":0.03,

    "num_leaves":63,

    "subsample":0.8,
    "colsample_bytree":0.8,

    "random_state":42,

    "n_jobs":-1,

    "verbosity":-1
}



# =====================================================
# 5 FOLD TRAIN + PREDICT
# =====================================================


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


test_pred = np.zeros(
    len(X_test)
)


oof = np.zeros(
    len(X)
)



for fold,(tr,va) in enumerate(
        skf.split(X,y),1):


    print("="*50)
    print("Fold",fold)


    model = LGBMClassifier(
        **params
    )


    model.fit(
        X.iloc[tr],
        y.iloc[tr],

        eval_set=[
            (
                X.iloc[va],
                y.iloc[va]
            )
        ],

        callbacks=[
            # early stopping
        ]
    )


    val_pred = model.predict_proba(
        X.iloc[va]
    )[:,1]


    oof[va]=val_pred


    score = roc_auc_score(
        y.iloc[va],
        val_pred
    )


    print(
        "Fold AUC:",
        score
    )


    test_pred += (
        model.predict_proba(
            X_test
        )[:,1]
        /5
    )



print("="*50)
print(
    "OOF AUC:",
    roc_auc_score(y,oof)
)



# =====================================================
# SUBMISSION
# =====================================================


submission = pd.DataFrame({

    ID_COL:
        test[ID_COL],

    TARGET:
        test_pred

})


submission.to_csv(
    OUTPUT,
    index=False
)


print("\nDONE")
print(submission.head())

print(
    "Saved:",
    OUTPUT
)