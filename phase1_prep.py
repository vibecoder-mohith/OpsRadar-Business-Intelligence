import pandas as pd

print("Loading datasets...")
tx = pd.read_csv('transactions.csv')
cust = pd.read_csv('customers.csv')
prod = pd.read_csv('products.csv')

print("\n--- Target Variable Distribution (status) ---")
print(tx['status'].value_counts(dropna=False))

print("\nMerging data...")
df = tx.merge(cust, on='customer_id', how='left')
df = df.merge(prod, on='product_id', how='left')

print(f"\nFinal Merged Shape: {df.shape}")
df.to_csv('master_training_data.csv', index=False)
print("\n✅ Saved 'master_training_data.csv'. Ready for ML modeling.")

with open('AGENTS.md', 'a') as f:
    f.write("- [Phase 1 Data Prep]: Successfully joined tx, customers, products into master_training_data.csv.\n")
