import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import numpy as np

train_data = pd.read_csv('train_final.csv')

test_data = pd.read_csv('test_final.csv')

categorical_cols = ['workclass', 'education', 'marital.status', 'occupation',
                    'relationship', 'race', 'sex', 'native.country']

numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

for col in categorical_cols:
    train_data[col].replace('?', np.nan, inplace=True)
    test_data[col].replace('?', np.nan, inplace=True)
    
    mode = train_data[col].mode()[0]
    train_data[col].fillna(mode, inplace=True)
    test_data[col].fillna(mode, inplace=True)

for col in numerical_cols:
    median = train_data[col].median()
    train_data[col].fillna(median, inplace=True)
    test_data[col].fillna(median, inplace=True)

binary_cols = ['sex']

le = LabelEncoder()
for col in binary_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

multiclass_cols = ['workclass', 'education', 'marital.status', 'occupation',
                   'relationship', 'race', 'native.country']

combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

combined_data = pd.get_dummies(combined_data, columns=multiclass_cols)

train_data_encoded = combined_data.iloc[:len(train_data), :].copy()
test_data_encoded = combined_data.iloc[len(train_data):, :].copy()

scaler = StandardScaler()
train_data_encoded[numerical_cols] = scaler.fit_transform(train_data_encoded[numerical_cols])
test_data_encoded[numerical_cols] = scaler.transform(test_data_encoded[numerical_cols])

X = train_data_encoded.drop(['income>50K'], axis=1)
y = train_data_encoded['income>50K'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]


print('AUC Score ', roc_auc_score(y_val, y_pred_proba))


if 'income>50K' in test_data_encoded.columns:
    test_features = test_data_encoded.drop(['income>50K'], axis=1)
else:
    test_features = test_data_encoded.copy()

test_preds_proba = model.predict_proba(test_features)[:, 1]

submission = pd.DataFrame({
    'ID': test_data['ID'].astype(int),
    'Prediction': test_preds_proba
})

submission['Prediction'] = submission['Prediction'].clip(0, 1)

submission.to_csv('ran-forest.csv', index=False)
