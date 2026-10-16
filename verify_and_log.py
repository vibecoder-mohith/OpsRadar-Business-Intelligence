import pandas as pd

df = pd.read_csv('dashboard_data.csv')

ml_only = len(df[(df['ml_anomaly_flag'] == 1) & (df['rule_risk_flag'] == 0)])
rule_only = len(df[(df['ml_anomaly_flag'] == 0) & (df['rule_risk_flag'] == 1)])
both = len(df[(df['ml_anomaly_flag'] == 1) & (df['rule_risk_flag'] == 1)])

# Calculate actual money lost (only summing negative margins on flagged rows)
actual_loss = df[df['is_flagged'] == 1]['margin'].clip(upper=0).sum()

with open('AGENTS.md', 'a') as f:
    f.write(f"- **[Phase 3 Analytics]**: Overlap check complete. ML-only: {ml_only}, Rule-only: {rule_only}, Overlap: {both}. Confirmed margin loss on flagged rows: ${actual_loss:,.2f}.\n")
    
print(f"Overlap Check: ML-only({ml_only}), Rule-only({rule_only}), Both({both})")
print(f"Confirmed Margin Loss: ${actual_loss:,.2f}")
