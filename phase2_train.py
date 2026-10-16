import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("1. Loading Master Data...")
df = pd.read_csv('master_training_data.csv')

# Create Binary Target: 1 if Cancelled/Returned, else 0
df['is_risk'] = df['status'].isin(['Cancelled', 'Returned']).astype(int)
print(f"Risk Class Balance:\n{df['is_risk'].value_counts(normalize=True) * 100}")

# Time-aware split: sort by timestamp to prevent data leakage
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp')

# Drop IDs, dates, and the original target to prevent leakage
cols_to_drop = ['status', 'is_risk']
for c in ['transaction_id', 'customer_id', 'product_id', 'timestamp', 'signup_date', 'date']:
    if c in df.columns: cols_to_drop.append(c)

X = df.drop(columns=cols_to_drop)
y = df['is_risk']

# 80/20 Time-aware split
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n2. Preprocessing {len(X.columns)} features...")
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

print("\n3. Training Baseline (Logistic Regression)...")
base_pipe = Pipeline([
    ('prep', preprocessor), 
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
base_pipe.fit(X_train, y_train)
base_roc = roc_auc_score(y_test, base_pipe.predict_proba(X_test)[:, 1])
print(f"Baseline ROC-AUC: {base_roc:.4f}")

print("\n4. Training Champion (XGBoost)...")
# Calculate weight for the minority risk class
weight = (len(y_train) - sum(y_train)) / sum(y_train)
champ_pipe = Pipeline([
    ('prep', preprocessor), 
    ('clf', XGBClassifier(scale_pos_weight=weight, random_state=42, eval_metric='logloss'))
])
champ_pipe.fit(X_train, y_train)
champ_preds = champ_pipe.predict(X_test)
champ_roc = roc_auc_score(y_test, champ_pipe.predict_proba(X_test)[:, 1])
print(f"Champion ROC-AUC: {champ_roc:.4f}")

print("\nChampion Classification Report:\n")
print(classification_report(y_test, champ_preds, target_names=['Healthy', 'Risk']))

joblib.dump(champ_pipe, 'model.joblib')
print("\n✅ Saved 'model.joblib'.")

with open('AGENTS.md', 'a') as f:
    f.write(f"- **[Phase 2/3 Modeling]**: Engineered binary 'is_risk' target. Time-aware 80/20 split. XGBoost champion achieved ROC-AUC {champ_roc:.3f}, saved to model.joblib.\n")
