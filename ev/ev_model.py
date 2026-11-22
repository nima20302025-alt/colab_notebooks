import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRAIN_PATH = 'train(1).csv'
TEST_PATH = 'test(1).csv'
SUBMISSION_PATH = 'submission.csv'

train = pd.read_csv(TRAIN_PATH)
X = train.drop(columns=['Will_Buy_EV'])
y = train['Will_Buy_EV'].map({'Yes': 1, 'No': 0})

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object', 'string']).columns.tolist()

preprocessor = ColumnTransformer([
    (
        'num',
        Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]),
        num_cols,
    ),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model),
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

pipe.fit(X_train, y_train)
valid_prob = pipe.predict_proba(X_valid)[:, 1]
print(f'Validation ROC-AUC: {roc_auc_score(y_valid, valid_prob):.6f}')

pipe.fit(X, y)

test = pd.read_csv(TEST_PATH)
probs = pipe.predict_proba(test)[:, 1]
sub = pd.DataFrame({'id': test['id'], 'Will_Buy_EV': probs})
sub.to_csv(SUBMISSION_PATH, index=False)

print(f'Saved {SUBMISSION_PATH} with {sub.shape[0]} rows')
print(sub.head().to_string(index=False))
