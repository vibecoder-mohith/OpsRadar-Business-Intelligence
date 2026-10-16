# OpsRadar: Revenue Protection & Decision Console
**Track 4: Business Decision-Support / Retail Intelligence**
**Team:** Cool (Mohith & Abhay)

## 1. Executive Summary
OpsRadar is an operational decision console that surfaces active retail margin leaks, diagnoses drivers, and enables real-time manager intervention.

## 2. ML Methodology & Dual-Engine Triage
1. **Deterministic Margin Safeguard (Rule Engine):** Flags transactions where unit price is below cost basis ($P_{unit} < C_{unit}$).
2. **Unsupervised Anomaly Detector (Isolation Forest - Trained from Scratch):** Trained on multi-dimensional transaction features (`quantity`, `unit_price`, `discount_pct`, `gross_revenue`) to isolate behavioral outliers missed by single-variable rules.
3. **Data Science Insight:** Classification models targeting `status` yielded validation ROC-AUC of ~0.49–0.50, revealing `status` was statistical noise in this dataset. We executed an honest pivot to forensic margin quantification and anomaly isolation.

## 3. Key Findings
- **Total Transactions Analyzed:** 18,000 unified records across catalog, customer, inventory, and campaign datasets.
- **Confirmed Cost Drag:** $1,396,782.81 in gross margin loss across 5,947 below-cost sales.
- **ML Anomaly Discovery:** 703 multivariate anomalies isolated.
- **Critical Overlap:** 196 transactions flagged by both engines (Tier-1 urgent triage).
- **Zero-Conversion Ad Waste:** $7,459.97 identified from unconverting marketing campaigns.

## 4. Run Instructions
pip install -r requirements.txt && streamlit run app.py
