import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SupplySignal", page_icon="🔍", layout="wide")

st.title("🔍 SupplySignal")
st.markdown("**AI-Powered Supplier Risk Detection & Backup Recommendation System**")
st.divider()

@st.cache_data
def load_data():
    suppliers = pd.read_csv("data/suppliers.csv")
    orders = pd.read_csv("data/orders.csv")
    orders["Due_Date"] = pd.to_datetime(orders["Due_Date"])
    orders["Creation_Date"] = pd.to_datetime(orders["Creation_Date"])
    return suppliers, orders

suppliers, orders = load_data()

@st.cache_resource
def train_model(suppliers):
    def label_risk(row):
        if row["On_Time_Rate"] < 0.65 or row["Fulfillment_Rate"] < 0.90 or row["On_Time_Rate"] == 0.0:
            return "High"
        elif row["On_Time_Rate"] < 0.80 or row["Fulfillment_Rate"] < 0.97:
            return "Medium"
        else:
            return "Low"
    suppliers = suppliers.copy()
    suppliers["Risk"] = suppliers.apply(label_risk, axis=1)
    features = suppliers[["On_Time_Rate", "Fulfillment_Rate", "Total_Orders", "Open_Value"]]
    le = LabelEncoder()
    labels = le.fit_transform(suppliers["Risk"])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    return clf, le, suppliers

clf, le, suppliers = train_model(suppliers)
features = suppliers[["On_Time_Rate", "Fulfillment_Rate", "Total_Orders", "Open_Value"]]
suppliers["Risk_Label"] = le.inverse_transform(clf.predict(features))

@st.cache_resource
def load_embeddings(profiles):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, model.encode(profiles)

emb_model, embeddings = load_embeddings(suppliers["Profile"].tolist())

def get_backups(disrupted_idx, suppliers, embeddings, top_n=3):
    sim_scores = cosine_similarity([embeddings[disrupted_idx]], embeddings)[0]
    sim_scores[disrupted_idx] = -1
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    backups = suppliers.iloc[top_indices][["Supplier","City","Category","Total_Orders","On_Time_Rate","Risk_Label"]].copy()
    backups["Match_Score"] = [f"{sim_scores[i]:.0%}" for i in top_indices]
    return backups

# ── Summary Metrics ───────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏭 Total Suppliers", len(suppliers))
c2.metric("🔴 High Risk", len(suppliers[suppliers["Risk_Label"] == "High"]))
c3.metric("🟡 Medium Risk", len(suppliers[suppliers["Risk_Label"] == "Medium"]))
c4.metric("🟢 Low Risk", len(suppliers[suppliers["Risk_Label"] == "Low"]))
c5.metric("📋 2026 Orders", f"{len(orders):,}")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Supplier Risk Dashboard", "📦 2026 Future Orders", "🔄 Disruption & Backup Analysis"])

# ── Tab 1 ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Supplier Risk Dashboard")
    risk_filter = st.multiselect("Filter by Risk Level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    filtered = suppliers[suppliers["Risk_Label"].isin(risk_filter)]

    def color_risk(val):
        return {"High": "background-color: #ffcccc", "Medium": "background-color: #fff3cc",
                "Low": "background-color: #ccffcc"}.get(val, "")

    display = filtered[["Supplier","City","Category","On_Time_Rate","Fulfillment_Rate","Total_Orders","Open_Value","Risk_Label"]].copy()
    display.columns = ["Supplier","City","Category","On-Time Rate","Fulfillment Rate","Total Orders","Open Value ($)","Risk"]
    st.dataframe(display.style.applymap(color_risk, subset=["Risk"]), use_container_width=True, height=500)

# ── Tab 2 ─────────────────────────────────────────────────────
with tab2:
    st.subheader("2026 Proposed Orders & Risk Exposure")

    vendor_risk = suppliers[["Supplier","Risk_Label"]].drop_duplicates()
    orders_merged = orders.merge(vendor_risk, on="Supplier", how="left")
    orders_merged["Risk_Label"] = orders_merged["Risk_Label"].fillna("Unknown")
    orders_merged["Month"] = orders_merged["Due_Date"].dt.strftime("%B %Y")

    # Top metrics
    high_risk_orders = orders_merged[orders_merged["Risk_Label"] == "High"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total 2026 Orders", f"{len(orders_merged):,}")
    m2.metric("⚠️ High-Risk Supplier Orders", f"{len(high_risk_orders):,}")
    m3.metric("💰 Total Order Value", f"${orders_merged['Order_Value'].sum():,.0f}")
    m4.metric("🔴 Value at Risk", f"${high_risk_orders['Order_Value'].sum():,.0f}")

    st.markdown("---")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        months = ["All"] + sorted(orders_merged["Month"].unique().tolist())
        selected_month = st.selectbox("Filter by Delivery Month:", months)
    with col_f2:
        order_risk_filter = st.multiselect("Filter by Risk Level:", ["High","Medium","Low","Unknown"],
                                            default=["High","Medium","Low","Unknown"])

    view = orders_merged.copy()
    if selected_month != "All":
        view = view[view["Month"] == selected_month]
    view = view[view["Risk_Label"].isin(order_risk_filter)]

    def color_order_risk(val):
        return {"High": "background-color: #ffcccc", "Medium": "background-color: #fff3cc",
                "Low": "background-color: #ccffcc"}.get(val, "")

    display_orders = view[["PO_Number","Supplier","Material","PO_Quantity","Order_Value","Due_Date","Prior_Year_Orders","Risk_Label"]].copy()
    display_orders["Due_Date"] = display_orders["Due_Date"].dt.strftime("%Y-%m-%d")
    display_orders.columns = ["PO Number","Supplier","Material","Qty","Value ($)","Due Date","Prior Year Orders","Risk"]
    st.dataframe(display_orders.style.applymap(color_order_risk, subset=["Risk"]), use_container_width=True, height=400)

    st.markdown("**📅 Monthly Order Value by Risk Level**")
    monthly = orders_merged.groupby(["Month","Risk_Label"])["Order_Value"].sum().reset_index()
    monthly_pivot = monthly.pivot(index="Month", columns="Risk_Label", values="Order_Value").fillna(0)
    # Sort months chronologically
    month_order = pd.to_datetime(monthly_pivot.index, format="%B %Y").argsort()
    monthly_pivot = monthly_pivot.iloc[month_order]
    st.bar_chart(monthly_pivot)

# ── Tab 3 ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Disruption Analysis & Backup Supplier Recommendations")
    high_risk_list = suppliers[suppliers["Risk_Label"] == "High"]["Supplier"].tolist()

    if high_risk_list:
        selected = st.selectbox("Select a high-risk supplier to analyze:", high_risk_list)
        idx = suppliers[suppliers["Supplier"] == selected].index[0]
        row = suppliers.loc[idx]

        supplier_orders = orders_merged[orders_merged["Supplier"] == selected]
        future_count = len(supplier_orders)
        future_qty = supplier_orders["PO_Quantity"].sum()
        future_value = supplier_orders["Order_Value"].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("On-Time Rate", f"{row['On_Time_Rate']:.0%}")
        col2.metric("Fulfillment Rate", f"{row['Fulfillment_Rate']:.0%}")
        col3.metric("2026 Orders Scheduled", f"{future_count:,}")
        col4.metric("2026 Order Value", f"${future_value:,.0f}")

        st.markdown(f"""
        **Disruption Summary — {selected}**

        > Supplier **{selected}** ({row['City']}) has been classified as **High Risk** based on the following signals:
        > - On-Time Delivery Rate: **{row['On_Time_Rate']:.0%}** (below acceptable threshold)
        > - Order Fulfillment Rate: **{row['Fulfillment_Rate']:.0%}**
        > - Total Historical Purchase Orders: **{int(row['Total_Orders'])}**
        > - 2026 Scheduled Orders: **{future_count:,} POs** totaling **{future_qty:,.0f} units** worth **${future_value:,.0f}**
        > - Category: **{row['Category']}**
        >
        > ⚠️ Immediate procurement review is recommended. Proactive engagement with backup
        > suppliers is critical to protect **${future_value:,.0f}** in 2026 order commitments.
        """)

        if future_count > 0:
            st.markdown("**📋 Upcoming 2026 Orders from this Supplier:**")
            future_display = supplier_orders[["PO_Number","Material","PO_Quantity","Order_Value","Due_Date"]].copy()
            future_display["Due_Date"] = future_display["Due_Date"].dt.strftime("%Y-%m-%d")
            future_display.columns = ["PO Number","Material","Qty","Value ($)","Due Date"]
            st.dataframe(future_display, use_container_width=True, height=300)

        st.markdown("**🔁 Top Backup Supplier Recommendations (AI-Matched):**")
        backups = get_backups(idx, suppliers, embeddings)
        st.dataframe(backups, use_container_width=True)
    else:
        st.success("✅ No high-risk suppliers detected at this time.")

st.divider()
st.caption("SupplySignal | AI-Powered Supply Chain Risk Intelligence | MSCM Applied AI Project")
