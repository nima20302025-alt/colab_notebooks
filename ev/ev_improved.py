import time
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TRAIN = 'train(1).csv'
TEST = 'test(1).csv'
SUB = 'submission_improved.csv'

print('Loading data...')
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

X = train.drop(columns=['Will_Buy_EV'])
y = train['Will_Buy_EV'].map({'Yes':1,'No':0})

# remove id from features
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

print('Numeric columns:', num_cols)
print('Categorical columns:', cat_cols)

# Preprocessor for one-hot models
num_pipeline = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler())
])
cat_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
pre = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

models = {}
models['logreg'] = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear'))])
models['rf'] = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5))])

# Prepare HGB data with category codes
X_hgb = X.copy()
X_hgb = X_hgb.reset_index(drop=True)
for c in cat_cols:
    X_hgb[c] = X_hgb[c].astype('category').cat.codes

# split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Xh_train, Xh_valid = None, None
if len(cat_cols) > 0:
    Xh = X_hgb.copy()
    Xh_train, Xh_valid, yh_train, yh_valid = train_test_split(Xh, y, test_size=0.2, random_state=42, stratify=y)

results = []

for name, pipe in models.items():
    print(f'Training {name}...')
    t0 = time.time()
    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_valid)[:,1]
    auc = roc_auc_score(y_valid, preds)
    t1 = time.time()
    print(f'{name} AUC: {auc:.6f} time: {t1-t0:.1f}s')
    results.append((name, auc, pipe))

# Train HGB
print('Training hgb...')
hgb = HistGradientBoostingClassifier(random_state=42, max_iter=400)
if len(cat_cols) > 0:
    t0 = time.time()
    hgb.fit(Xh_train, yh_train)
    preds = hgb.predict_proba(Xh_valid)[:,1]
    auc = roc_auc_score(y_valid, preds)
    t1 = time.time()
    print(f'hgb AUC: {auc:.6f} time: {t1-t0:.1f}s')
    results.append(('hgb', auc, hgb))
else:
    print('No categorical columns for HGB test; skipping')

# choose best
best = max(results, key=lambda x: x[1])
print('Best model:', best[0], 'AUC:', best[1])

# retrain on full data and predict
if best[0] in ['logreg','rf']:
    best_pipe = best[2]
    print('Retraining best pipeline on full data...')
    best_pipe.fit(X, y)
    # prepare test features for pipe: drop id if present
    test_X = test_features.copy()
    if 'id' in test_X.columns:
        test_X = test_X.drop(columns=['id'])
    preds_test = best_pipe.predict_proba(test_X)[:,1]
else:
    # hgb: prepare full dataset with cat codes
    X_full_hgb = X.copy()
    for c in cat_cols:
        X_full_hgb[c] = X_full_hgb[c].astype('category').cat.codes
    print('Retraining HGB on full data...')
    best[2].fit(X_full_hgb, y)
    test_hgb = test_features.copy()
    for c in cat_cols:
        # map test categories to codes using training categories; unseen -> -1
        test_hgb[c] = test_hgb[c].astype('category').cat.codes
    preds_test = best[2].predict_proba(test_hgb)[:,1]

# save submission
out = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': preds_test})
out.to_csv(SUB, index=False)
print(f'Saved improved submission to {SUB} with shape {out.shape}')
print(out.head().to_string(index=False))
