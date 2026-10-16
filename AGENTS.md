# AGENTS.md - Project Ground Truth & Progress Ledger

## 1. Project Objective & Overview
Data science investigation on business money leaks (Track 4).
Goal: Predict transaction failures/risks and build a business decision-support MVP.

## 2. Data Findings & Schema
- transactions.csv (18k rows): The anchor table. Contains our target variable `status`.
- customers.csv (3k rows) & products.csv (250 rows): Dimension tables for joins.

## 3. Progress & Discovery Log
- [Init]: Workspace initialized.
- [Phase 1 Data Prep]: Successfully joined tx, customers, products into master_training_data.csv.
- **[Phase 2/3 Modeling]**: Engineered binary 'is_risk' target. Time-aware 80/20 split. XGBoost champion achieved ROC-AUC 0.483, saved to model.joblib.

- **[Phase 3 Pivot]**: Evaluated XGBoost for 'status'. ROC-AUC was 0.48 (pure noise). Pivoted ML strategy to Unsupervised Anomaly Detection (Isolation Forest) to flag multivariate inconsistencies and calculate real financial leakage.
- **[Phase 3 Analytics]**: Overlap check complete. ML-only: 703, Rule-only: 5947, Overlap: 196. Confirmed margin loss on flagged rows: $-1,396,782.81.
- **[Phase 4 Dashboard]**: Streamlit MVP launched at http://localhost:8501 (app.py). KPIs: transactions in view, gross revenue touched, confirmed margin loss. Sidebar filters for flagged-only and category.
