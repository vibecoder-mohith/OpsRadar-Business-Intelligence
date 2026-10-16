import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="OpsRadar | Revenue & Margin Protection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-theme compatible styling (fixes white text on white background)
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
    }
    .metric-label { font-size: 13px; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #f8fafc; margin-top: 4px; }
    .metric-sub { font-size: 12px; margin-top: 4px; font-weight: 500; }
    .danger-text { color: #f87171; }
    .warning-text { color: #fbbf24; }
    .audit-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        color: #f1f5f9;
        font-size: 14px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_datasets():
    df = pd.read_csv('dashboard_data.csv')
    try:
        camp = pd.read_csv('campaigns.csv')
    except Exception:
        camp = pd.DataFrame()
    try:
        inv = pd.read_csv('inventory.csv')
    except Exception:
        inv = pd.DataFrame()
    return df, camp, inv

df, camp, inv = load_all_datasets()

if "actions_taken" not in st.session_state:
    st.session_state.actions_taken = []

# --- TOP HEADER BAR ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("OpsRadar: Revenue Protection & Decision Console")
    st.caption("Cross-functional investigation tool linking transactions, product margins, inventory stockouts, and campaign performance.")
with col_head2:
    st.write("")
    st.write(f"**Audit Ledger Logged:** {len(st.session_state.actions_taken)} interventions")

st.write("---")

# --- SIDEBAR: STRICT OPERATIONAL FILTERS ---
st.sidebar.subheader("Filter & Triage Scope")

category_list = ["All Categories"] + sorted([c for c in df['category'].dropna().unique()])
selected_cat = st.sidebar.selectbox("Merchandise Category", category_list)

detection_scope = st.sidebar.radio(
    "Detection Mechanism",
    ["All Flagged Issues", "Negative Margin Leaks (Rule)", "Multivariate Anomalies (ML)", "Clean Transactions"]
)

min_loss_threshold = st.sidebar.slider("Minimum Negative Margin ($ USD)", 0, 500, 0, step=25)

# Filtering logic
filtered_df = df.copy()
if selected_cat != "All Categories":
    filtered_df = filtered_df[filtered_df['category'] == selected_cat]

if detection_scope == "Negative Margin Leaks (Rule)":
    filtered_df = filtered_df[filtered_df['rule_risk_flag'] == 1]
elif detection_scope == "Multivariate Anomalies (ML)":
    filtered_df = filtered_df[filtered_df['ml_anomaly_flag'] == 1]
elif detection_scope == "All Flagged Issues":
    filtered_df = filtered_df[filtered_df['is_flagged'] == 1]
elif detection_scope == "Clean Transactions":
    filtered_df = filtered_df[filtered_df['is_flagged'] == 0]

if min_loss_threshold > 0:
    filtered_df = filtered_df[filtered_df['margin'] <= -min_loss_threshold]

# --- SECTION 1: EXECUTIVE LOSS SUMMARY ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_scanned = len(filtered_df)
confirmed_loss = filtered_df[filtered_df['margin'] < 0]['margin'].sum()
unusual_orders = filtered_df['ml_anomaly_flag'].sum()
wasted_ad_spend = camp[camp['conversions'] == 0]['spend'].sum() if not camp.empty else 0.0

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Transactions in Scope</div>
        <div class="metric-value">{total_scanned:,}</div>
        <div class="metric-sub" style="color: #94a3b8;">Out of {len(df):,} total</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Confirmed Cost Drag</div>
        <div class="metric-value danger-text">${abs(confirmed_loss):,.2f}</div>
        <div class="metric-sub danger-text">Direct negative gross margin</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ML Identified Anomalies</div>
        <div class="metric-value warning-text">{unusual_orders:,}</div>
        <div class="metric-sub warning-text">Isolation Forest flags</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Zero-Conversion Ad Waste</div>
        <div class="metric-value">${wasted_ad_spend:,.2f}</div>
        <div class="metric-sub" style="color: #94a3b8;">Marketing expenditure leakage</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- SECTION 2: MACRO VISUAL DIAGNOSTICS ---
tab_diag1, tab_diag2, tab_diag3 = st.tabs(["Category Loss Distribution", "Inventory Stockout Impact", "Ad Spend Inefficiency"])

with tab_diag1:
    chart_scope = st.radio("Chart Scope:", ["Filtered Selection", "Full Catalog Baseline"], horizontal=True)
    chart_source = filtered_df if chart_scope == "Filtered Selection" else df

    cat_perf = chart_source.groupby('category').agg(
        negative_margin=('margin', lambda x: abs(x[x < 0].sum())),
        total_volume=('transaction_id', 'count'),
        avg_discount=('discount_pct', 'mean')
    ).reset_index().sort_values(by='negative_margin', ascending=False)

    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.caption(f"Net Dollar Loss Attributable to Sub-Cost Selling ({chart_scope})")
        st.bar_chart(cat_perf, x='category', y='negative_margin', color='#dc2626')
    with col_c2:
        st.caption("Category Risk Reference")
        cat_perf['avg_discount'] = (cat_perf['avg_discount'] * 100).round(1).astype(str) + "%"
        cat_perf['negative_margin'] = cat_perf['negative_margin'].map('${:,.2f}'.format)
        st.dataframe(cat_perf, hide_index=True, use_container_width=True)

with tab_diag2:
    if not inv.empty:
        inv_display = inv.copy()
        if 'product_id' in inv.columns and 'product_id' in df.columns and selected_cat != "All Categories":
            cat_pids = set(df[df['category'] == selected_cat]['product_id'].dropna().unique())
            inv_display = inv_display[inv_display['product_id'].isin(cat_pids)]

        stockouts = inv_display[inv_display['stockout_flag'] == 1]

        i_col1, i_col2 = st.columns(2)
        i_col1.metric("Active Stockout Incidents", f"{len(stockouts):,}")
        avg_delay = stockouts['supplier_delay_days'].mean() if 'supplier_delay_days' in stockouts.columns else 0.0
        i_col2.metric("Avg Supplier Restock Delay", f"{avg_delay:.1f} days")

        st.write("Products with zero on-hand availability causing unfulfilled demand:")
        disp_cols = [c for c in ['inventory_record_id', 'product_id', 'date', 'stock_on_hand', 'reorder_threshold', 'supplier_delay_days'] if c in inv_display.columns]
        st.dataframe(stockouts[disp_cols].head(25), use_container_width=True)
    else:
        st.info("Inventory dataset not found.")

with tab_diag3:
    if not camp.empty:
        camp_display = camp.copy()
        if selected_cat != "All Categories" and 'category' in camp_display.columns:
            camp_display = camp_display[camp_display['category'] == selected_cat]

        zero_conv = camp_display[camp_display['conversions'] == 0]
        waste_amt = zero_conv['spend'].sum()

        ad_col1, ad_col2 = st.columns(2)
        ad_col1.metric("Zero-Conversion Campaigns", f"{len(zero_conv):,}")
        ad_col2.metric("Total Wasted Budget", f"${waste_amt:,.2f}", delta="100% Loss", delta_color="inverse")

        st.write("Campaigns ranked by highest spend with zero or minimal customer conversions:")
        sort_cols = [c for c in ['spend', 'clicks', 'conversions'] if c in camp_display.columns]
        st.dataframe(camp_display.sort_values(by=sort_cols, ascending=[False, True, True]).head(25), use_container_width=True)
    else:
        st.info("Campaigns dataset not found.")

st.write("---")

# --- SECTION 3: DECISION SUPPORT & AUDIT CONSOLE ---
st.subheader("Operational Triage: Investigate & Act")
st.caption("Select a flagged record to inspect underlying signals, evaluate unit economics, and record an operational decision.")

priority_queue = filtered_df.sort_values(by='margin', ascending=True)

if priority_queue.empty:
    st.info("No records match the current filter selection.")
else:
    id_list = priority_queue['transaction_id'].astype(str).tolist()[:60]
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        target_id = st.selectbox("Transaction ID Queue (Sorted by Severity):", id_list)

    selected_row = priority_queue[priority_queue['transaction_id'].astype(str) == target_id].iloc[0]

    col_inspect1, col_inspect2 = st.columns([1, 1])

    with col_inspect1:
        st.markdown(f"""
        <div class="audit-box">
            <h4 style="margin:0 0 10px 0; color:#38bdf8;">Record Profile: {target_id}</h4>
            • <b>Customer ID:</b> <span style="background:#334155; padding:2px 6px; border-radius:4px;">{selected_row.get('customer_id', 'N/A')}</span><br>
            • <b>Product ID:</b> <span style="background:#334155; padding:2px 6px; border-radius:4px;">{selected_row.get('product_id', 'N/A')}</span><br>
            • <b>Channel:</b> {selected_row.get('channel', 'N/A')}<br>
            • <b>Order Quantity:</b> <b>{selected_row.get('quantity', 0)} units</b><br>
            • <b>Unit Selling Price:</b> <b>${selected_row.get('unit_price', 0):.2f}</b><br>
            • <b>Unit Cost Basis:</b> <b>${selected_row.get('unit_cost', 0):.2f}</b><br>
            • <b>Applied Discount:</b> <b>{selected_row.get('discount_pct', 0)*100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

    with col_inspect2:
        st.markdown("**Evidence Signals & Diagnosis**")
        if selected_row.get('margin', 0) < 0:
            st.error(f"Cost Inversion: Unit cost (${selected_row.get('unit_cost',0):.2f}) exceeds selling price (${selected_row.get('unit_price',0):.2f}). Direct transaction loss: ${abs(selected_row.get('margin',0)):,.2f}")
        if selected_row.get('ml_anomaly_flag', 0) == 1:
            st.warning("Isolation Forest Flag: Statistical outlier across price, discount, and volume distribution.")
        if selected_row.get('discount_pct', 0) > 0.35:
            st.warning(f"Excessive Discount: {selected_row.get('discount_pct',0)*100:.1f}% exceeds standard operational margin threshold.")
        if selected_row.get('margin', 0) >= 0 and selected_row.get('ml_anomaly_flag', 0) == 0:
            st.success("No active margin violation detected on this record.")

        st.write("**Direct Manager Action:**")
        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            if st.button("Halt / Cancel SKU Order", key="btn_halt"):
                st.session_state.actions_taken.append({
                    "tx_id": target_id,
                    "action": "Order Cancelled / SKU Frozen",
                    "reason": "Sub-cost sale",
                    "margin_saved": abs(selected_row.get('margin', 0))
                })
                st.rerun()
        with act_col2:
            if st.button("Cap Discount at 15%", key="btn_cap"):
                st.session_state.actions_taken.append({
                    "tx_id": target_id,
                    "action": "Discount Capped at 15%",
                    "reason": "Margin protection override",
                    "margin_saved": abs(selected_row.get('margin', 0)) * 0.5
                })
                st.rerun()
        with act_col3:
            if st.button("Mark as Cleared / Authorized", key="btn_clear"):
                st.session_state.actions_taken.append({
                    "tx_id": target_id,
                    "action": "Cleared by Manager",
                    "reason": "Promotional clearance",
                    "margin_saved": 0.0
                })
                st.rerun()

# --- SECTION 4: AUDIT LOG ---
if st.session_state.actions_taken:
    st.write("---")
    st.subheader("Session Decision & Audit History")
    audit_df = pd.DataFrame(st.session_state.actions_taken)
    st.dataframe(audit_df, use_container_width=True)
