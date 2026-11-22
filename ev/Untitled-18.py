import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


# ============================
# CONFIG
# ============================

DATA_PATH = "train(1).csv"

TARGET = "Will_Buy_EV"

N_FOLDS_FAST = 3
N_FOLDS_FINAL = 5


# ============================
# LOAD DATA
# ============================

df = pd.read_csv(DATA_PATH)

print("="*70)
print("DATA")
print("="*70)

print(df.shape)


# ============================
# TARGET
# ============================

y = df[TARGET].map({
    "No":0,
    "Yes":1
})

if y.isna().sum() > 0:
    raise ValueError(
        f"Invalid target values: {df[TARGET].unique()}"
    )


X = df.drop(columns=[TARGET])


# ============================
# BASIC ENCODING
# ============================

cat_cols = X.select_dtypes(
    include=["object","category","string"]
).columns


for c in cat_cols:
    X[c] = X[c].astype("category")


print("Features:", X.shape[1])


# ============================
# MODEL
# ============================

def run_cv(X_data, y_data, folds=3):

    skf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    oof = np.zeros(len(y_data))

    scores=[]


    for fold,(tr,va) in enumerate(skf.split(X_data,y_data),1):

        model = LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,

            subsample=0.8,
            colsample_bytree=0.8,

            random_state=42,
            n_jobs=-1,

            verbosity=-1
        )


        model.fit(
            X_data.iloc[tr],
            y_data.iloc[tr],

            eval_set=[
                (
                    X_data.iloc[va],
                    y_data.iloc[va]
                )
            ],

            callbacks=[
                early_stopping(100),
                log_evaluation(0)
            ]
        )


        pred = model.predict_proba(
            X_data.iloc[va]
        )[:,1]


        auc = roc_auc_score(
            y_data.iloc[va],
            pred
        )

        scores.append(auc)


        oof[va]=pred


    return (
        roc_auc_score(y_data,oof),
        np.mean(scores),
        np.std(scores)
    )



# ============================
# BASELINE
# ============================

print("\nBASELINE")

base_auc,base_mean,base_std = run_cv(
    X,y,N_FOLDS_FAST
)


print(
    f"AUC={base_auc:.6f}"
)


# ============================
# LEAVE ONE OUT
# ============================

results=[]


features=list(X.columns)


print("\nSTART FEATURE REMOVAL")


for i,f in enumerate(features,1):

    print(
        f"[{i}/{len(features)}] Removing {f}"
    )


    X_tmp=X.drop(columns=[f])


    auc,mean,std=run_cv(
        X_tmp,
        y,
        N_FOLDS_FAST
    )


    results.append({

        "removed_feature":f,

        "auc":auc,

        "mean_fold":mean,

        "std_fold":std,

        "change_vs_base":
            auc-base_auc
    })


res=pd.DataFrame(results)


res=res.sort_values(
    "auc",
    ascending=False
)


print("\n")
print("="*70)
print("FEATURE REMOVAL RESULTS")
print("="*70)


print(res.to_string(index=False))


res.to_csv(
    "feature_removal_results.csv",
    index=False
)


# ============================
# BEST CONFIG
# ============================


best_removed = res.iloc[0]


print("\nBEST REMOVAL")
print("="*70)

print(
    best_removed
)


if best_removed["auc"] > base_auc:

    print(
        "\nRemoving this feature improves model:"
    )

    print(
        best_removed["removed_feature"]
    )

else:

    print(
        "\nNo single feature removal improved baseline."
    )

