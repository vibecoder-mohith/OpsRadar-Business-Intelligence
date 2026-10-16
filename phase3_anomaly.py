import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

print("1. Loading Master Data...")
df = pd.read_csv('master_training_data.csv')

# Business Rule: Calculate absolute gross revenue and theoretical margin
df['gross_revenue'] = df['quantity'] * df['unit_price']
if 'unit_cost' in df.columns:
    df['margin'] = df['gross_revenue'] - (df['quantity'] * df['unit_cost'])
else:
    df['margin'] = 0

print("2. Training ML Anomaly Detector (Isolation Forest)...")
# Select numerical features to find abnormal transaction patterns
num_cols = ['quantity', 'unit_price', 'discount_pct', 'gross_revenue']
X_anomaly = df[num_cols].fillna(0)

# Isolation Forest isolates data points that are mathematically strange
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_anomaly)

# -1 means anomaly (Risk), 1 means normal. Let's map to 1 (Risk) and 0 (Normal)
df['ml_anomaly_flag'] = (iso_forest.predict(X_anomaly) == -1).astype(int)

# Hard Business Rule Flag (e.g., negative margin or extremely high discount)
df['rule_risk_flag'] = ((df['margin'] < 0) | (df['discount_pct'] > 0.4)).astype(int)

# Combined System Flag
df['is_flagged'] = ((df['ml_anomaly_flag'] == 1) | (df['rule_risk_flag'] == 1)).astype(int)

print(f"\nFlagged Transactions: {df['is_flagged'].sum()} out of {len(df)}")
print(f"Total Revenue at Risk: ${df[df['is_flagged'] == 1]['gross_revenue'].sum():,.2f}")

# Save the finalized dataset and model for the Streamlit UI
df.to_csv('dashboard_data.csv', index=False)
joblib.dump(iso_forest, 'anomaly_model.joblib')
print("\n✅ Saved 'dashboard_data.csv' and 'anomaly_model.joblib'. Ready for UI Dashboard.")
