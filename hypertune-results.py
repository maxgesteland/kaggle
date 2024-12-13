import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


train_data = pd.read_csv('train_final.csv')
test_data = pd.read_csv('test_final.csv')

categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                  'capital.loss', 'hours.per.week']

for col in categorical_cols:
    train_data[col].replace('?', np.nan, inplace=True)
    test_data[col].replace('?', np.nan, inplace=True)

for col in categorical_cols:
    mode_val = train_data[col].mode()[0]
    train_data[col].fillna(mode_val, inplace=True)
    test_data[col].fillna(mode_val, inplace=True)

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

binary_cols = ['sex']
le = LabelEncoder()
for col in binary_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

multiclass_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                   'relationship', 'race', 'native.country']

combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

combined_data = pd.get_dummies(combined_data, columns=multiclass_cols)

combined_data.fillna(0, inplace=True)

train_data = combined_data.iloc[:len(train_data), :].copy()
test_data = combined_data.iloc[len(train_data):, :].copy()

scaler = StandardScaler()
train_data.loc[:, numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
test_data.loc[:, numerical_cols] = scaler.transform(test_data[numerical_cols])

X = train_data.drop(['income>50K'], axis=1, errors='ignore')
y = train_data['income>50K'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

estimator = DecisionTreeClassifier(random_state=42, max_depth=1)

ada = AdaBoostClassifier(
    estimator=estimator,
    n_estimators=175,
    random_state=42
)

param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3],
    'estimator__min_samples_leaf': [1, 2, 5],
    'algorithm': ['SAMME', 'SAMME.R']
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=skf,
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

grid_results = pd.DataFrame(grid_search.cv_results_)

top_10 = grid_results.sort_values(by='mean_test_score', ascending=False).head(10)

top_10.to_csv('top_10_hyperparameter_combinations.csv', index=False)
print("\nTop 10 parameter combinations saved to 'top_10_hyperparameter_combinations.csv'")



