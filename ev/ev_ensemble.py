import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

TRAIN = 'train(1).csv'
TEST = 'test(1).csv'
SUB = 'submission_ensemble.csv'

print('Loading data')
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

X = train.drop(columns=['Will_Buy_EV'])
y = train['Will_Buy_EV'].map({'Yes':1,'No':0})

if 'id' in X.columns:
    X = X.drop(columns=['id'])

if 'id' in test.columns:
    test_ids = test['id']
    test_features = test.drop(columns=['id'])
else:
    test_ids = None
    test_features = test.copy()

cat_cols = X.select_dtypes(include=['object','string']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object','string']).columns.tolist()

print('num cols:', num_cols)
print('cat cols:', cat_cols)

num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_enc, cat_cols)])

# base pipelines (preprocessing + estimator)
pipe_log = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'))])
pipe_rf = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=5))])

# HGB uses categorical codes
X_hgb = X.copy()
for c in cat_cols:
    X_hgb[c] = X_hgb[c].astype('category').cat.codes

# Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xh_train, Xh_valid = None, None
if len(cat_cols) > 0:
    Xh = X_hgb.copy()
    Xh_train, Xh_valid, yh_train, yh_valid = train_test_split(Xh, y, test_size=0.2, random_state=42, stratify=y)

# Fit base models on train
print('Fitting logistic...')
pipe_log.fit(X_train, y_train)
print('Fitting random forest...')
pipe_rf.fit(X_train, y_train)
print('Fitting hgb...')
hgb = HistGradientBoostingClassifier(random_state=42, max_iter=400)
if len(cat_cols) > 0:
    hgb.fit(Xh_train, yh_train)

# Validation probs
print('Predicting validation probabilities...')
val_log = pipe_log.predict_proba(X_valid)[:,1]
val_rf = pipe_rf.predict_proba(X_valid)[:,1]
if len(cat_cols) > 0:
    val_hgb = hgb.predict_proba(Xh_valid)[:,1]
else:
    val_hgb = np.zeros_like(val_log)

# AUCs
auc_log = roc_auc_score(y_valid, val_log)
auc_rf = roc_auc_score(y_valid, val_rf)
auc_hgb = roc_auc_score(y_valid, val_hgb)
print('Validation AUCs:', {'logreg':auc_log, 'rf':auc_rf, 'hgb':auc_hgb})

# Calibrate using validation probabilities via Platt scaling (logistic on predicted probs)
print('Calibrating models on validation (Platt scaling)...')
from sklearn.linear_model import LogisticRegression as _PlattLR

def platt_fit(probs, y):
    lr = _PlattLR(C=1.0, max_iter=1000)
    lr.fit(probs.reshape(-1, 1), y)
    return lr

platt_log = platt_fit(val_log, y_valid)
platt_rf = platt_fit(val_rf, y_valid)
if len(cat_cols) > 0:
    platt_hgb = platt_fit(val_hgb, y_valid)
else:
    platt_hgb = None

cal_val_log = platt_log.predict_proba(val_log.reshape(-1, 1))[:, 1]
cal_val_rf = platt_rf.predict_proba(val_rf.reshape(-1, 1))[:, 1]
cal_val_hgb = platt_hgb.predict_proba(val_hgb.reshape(-1, 1))[:, 1] if platt_hgb is not None else np.zeros_like(cal_val_log)

print('Calibrated AUCs:', roc_auc_score(y_valid, cal_val_log), roc_auc_score(y_valid, cal_val_rf), roc_auc_score(y_valid, cal_val_hgb))

# Blends
val_stack = np.vstack([cal_val_log, cal_val_rf, cal_val_hgb]).T
mean_blend = val_stack.mean(axis=1)
weights = np.array([roc_auc_score(y_valid, cal_val_log), roc_auc_score(y_valid, cal_val_rf), roc_auc_score(y_valid, cal_val_hgb)])
weights = weights / weights.sum()
weighted_blend = (val_stack * weights).sum(axis=1)

# Train stacking meta
meta = LogisticRegression(max_iter=1000)
meta.fit(val_stack, y_valid)
stack_preds = meta.predict_proba(val_stack)[:,1]

print('Blend AUCs:', {'mean':roc_auc_score(y_valid, mean_blend), 'weighted':roc_auc_score(y_valid, weighted_blend), 'stack':roc_auc_score(y_valid, stack_preds)})

# Choose best
scores = {'mean':roc_auc_score(y_valid, mean_blend), 'weighted':roc_auc_score(y_valid, weighted_blend), 'stack':roc_auc_score(y_valid, stack_preds)}
best_blend = max(scores.items(), key=lambda x: x[1])[0]
print('Best blend on validation:', best_blend, scores[best_blend])
# Instead of calibrating on full data (which can be memory-heavy),
# use an out-of-fold stacking approach to train a blender, then retrain
# base models on full data and apply the blender to test predictions.
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder

print('Building out-of-fold predictions for stacking...')
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_log = np.zeros(X.shape[0])
oof_rf = np.zeros(X.shape[0])
oof_hgb = np.zeros(X.shape[0])

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print('Fold', fold+1)
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # logistic
    clf_log = clone(pipe_log)
    clf_log.fit(X_tr, y_tr)
    oof_log[val_idx] = clf_log.predict_proba(X_val)[:,1]

    # rf
    clf_rf = clone(pipe_rf)
    clf_rf.fit(X_tr, y_tr)
    oof_rf[val_idx] = clf_rf.predict_proba(X_val)[:,1]

    # hgb with ordinal encoding
    if len(cat_cols) > 0:
        ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ord_enc.fit(X_tr[cat_cols])
        Xh_tr = X_tr.copy()
        Xh_val = X_val.copy()
        Xh_tr[cat_cols] = ord_enc.transform(X_tr[cat_cols])
        Xh_val[cat_cols] = ord_enc.transform(X_val[cat_cols])
        clf_hgb = HistGradientBoostingClassifier(random_state=42, max_iter=400)
        clf_hgb.fit(Xh_tr, y_tr)
        oof_hgb[val_idx] = clf_hgb.predict_proba(Xh_val)[:,1]
    else:
        oof_hgb[val_idx] = 0.0

# Evaluate OOF AUCs
auc_oof_log = roc_auc_score(y, oof_log)
auc_oof_rf = roc_auc_score(y, oof_rf)
auc_oof_hgb = roc_auc_score(y, oof_hgb)
print('OOF AUCs:', {'logreg':auc_oof_log, 'rf':auc_oof_rf, 'hgb':auc_oof_hgb})

# Train meta on OOF
stack_oof = np.vstack([oof_log, oof_rf, oof_hgb]).T
meta = LogisticRegression(max_iter=1000)
meta.fit(stack_oof, y)

print('Retraining base models on full data...')
pipe_log_full = clone(pipe_log)
pipe_rf_full = clone(pipe_rf)
pipe_log_full.fit(X, y)
pipe_rf_full.fit(X, y)

if len(cat_cols) > 0:
    ord_full = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ord_full.fit(X[cat_cols])
    X_full_hgb = X.copy()
    X_full_hgb[cat_cols] = ord_full.transform(X[cat_cols])
    hgb_full = HistGradientBoostingClassifier(random_state=42, max_iter=400)
    hgb_full.fit(X_full_hgb, y)
else:
    hgb_full = None

print('Predicting test set with base models...')
test_log = pipe_log_full.predict_proba(test_features)[:,1]
test_rf = pipe_rf_full.predict_proba(test_features)[:,1]

if hgb_full is not None:
    test_hgb = test_features.copy()
    test_hgb[cat_cols] = ord_full.transform(test_features[cat_cols])
    test_hgb_preds = hgb_full.predict_proba(test_hgb)[:,1]
else:
    test_hgb_preds = np.zeros_like(test_log)

test_stack = np.vstack([test_log, test_rf, test_hgb_preds]).T

if best_blend == 'mean':
    final_probs = test_stack.mean(axis=1)
elif best_blend == 'weighted':
    # recompute weights from OOF AUCs
    w = np.array([auc_oof_log, auc_oof_rf, auc_oof_hgb])
    w = w / w.sum()
    final_probs = (test_stack * w).sum(axis=1)
else:
    final_probs = meta.predict_proba(test_stack)[:,1]

out = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_probs})
out.to_csv(SUB, index=False)
print('Saved ensemble submission to', SUB)
print(out.head().to_string(index=False))
