import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SupplySignal",
    page_icon="🔍",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("🔍 SupplySignal")
st.markdown("**AI-Powered Supplier Risk Detection & Backup Recommendation System**")
st.divider()

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    suppliers = pd.read_csv("data/suppliers.csv")
    orders = pd.read_csv("data/orders.csv")
    orders["Due_Date"] = pd.to_datetime(orders["Due_Date"])
    return suppliers, orders

suppliers, orders = load_data()

# ── Train Risk Classifier ─────────────────────────────────────
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

# ── Predict Risk ──────────────────────────────────────────────
features = suppliers[["On_Time_Rate", "Fulfillment_Rate", "Total_Orders", "Open_Value"]]
suppliers["Risk_Label"] = le.inverse_transform(clf.predict(features))

# ── Embeddings ────────────────────────────────────────────────
@st.cache_resource
def load_embeddings(profiles):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(profiles)
    return model, embeddings

emb_model, embeddings = load_embeddings(suppliers["Profile"].tolist())

def get_backups(disrupted_idx, suppliers, embeddings, top_n=3):
    sim_scores = cosine_similarity([embeddings[disrupted_idx]], embeddings)[0]
    sim_scores[disrupted_idx] = -1
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    backups = suppliers.iloc[top_indices][["Supplier", "City", "Category", "Total_Orders", "On_Time_Rate", "Risk_Label"]].copy()
    backups["Match_Score"] = [f"{sim_scores[i]:.0%}" for i in top_indices]
    return backups

# ── Risk Summary Metrics ──────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏭 Total Suppliers", len(suppliers))
c2.metric("🔴 High Risk", len(suppliers[suppliers["Risk_Label"] == "High"]))
c3.metric("🟡 Medium Risk", len(suppliers[suppliers["Risk_Label"] == "Medium"]))
c4.metric("🟢 Low Risk", len(suppliers[suppliers["Risk_Label"] == "Low"]))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Supplier Risk Dashboard", "📦 2026 Future Orders", "🔄 Disruption & Backup Analysis"])

# ── Tab 1: Supplier Risk Dashboard ───────────────────────────
with tab1:
    st.subheader("Supplier Risk Dashboard")

    risk_filter = st.multiselect("Filter by Risk Level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    filtered = suppliers[suppliers["Risk_Label"].isin(risk_filter)]

    def color_risk(val):
        colors = {
            "High": "background-color: #ffcccc",
            "Medium": "background-color: #fff3cc",
            "Low": "background-color: #ccffcc"
        }
        return colors.get(val, "")

    display = filtered[["Supplier", "City", "Category", "On_Time_Rate", "Fulfillment_Rate", "Total_Orders", "Open_Value", "Risk_Label"]].copy()
    display.columns = ["Supplier", "City", "Category", "On-Time Rate", "Fulfillment Rate", "Total Orders", "Open Value ($)", "Risk"]
    st.dataframe(display.style.applymap(color_risk, subset=["Risk"]), use_container_width=True, height=500)

# ── Tab 2: 2026 Future Orders ─────────────────────────────────
with tab2:
    st.subheader("2026 Future Orders & Risk Exposure")

    vendor_risk = suppliers[["Supplier", "Risk_Label"]].drop_duplicates()
    orders_merged = orders.merge(vendor_risk, on="Supplier", how="left")
    orders_merged["Risk_Label"] = orders_merged["Risk_Label"].fillna("Unknown")

    total_orders = len(orders_merged)
    high_risk_orders = orders_merged[orders_merged["Risk_Label"] == "High"]
    total_qty_at_risk = high_risk_orders["PO_Quantity"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("📋 Total 2026 Orders", total_orders)
    m2.metric("⚠️ Orders from High-Risk Suppliers", len(high_risk_orders))
    m3.metric("📦 Units at Risk", f"{total_qty_at_risk:,.0f}")

    st.markdown("---")

    orders_merged["Month"] = orders_merged["Due_Date"].dt.strftime("%B %Y")
    months = ["All"] + sorted(orders_merged["Month"].unique().tolist())
    selected_month = st.selectbox("Filter by Delivery Month:", months)

    if selected_month != "All":
        view = orders_merged[orders_merged["Month"] == selected_month]
    else:
        view = orders_merged.copy()

    order_risk_filter = st.multiselect("Filter by Supplier Risk:", ["High", "Medium", "Low", "Unknown"],
                                        default=["High", "Medium", "Low", "Unknown"])
    view = view[view["Risk_Label"].isin(order_risk_filter)]

    def color_order_risk(val):
        colors = {
            "High": "background-color: #ffcccc",
            "Medium": "background-color: #fff3cc",
            "Low": "background-color: #ccffcc"
        }
        return colors.get(val, "")

    display_orders = view[["PO_Number", "Supplier", "Material", "PO_Quantity", "Open_Quantity",
                             "Open_Value", "Due_Date", "Risk_Label"]].copy()
    display_orders["Due_Date"] = display_orders["Due_Date"].dt.strftime("%Y-%m-%d")
    display_orders.columns = ["PO Number", "Supplier", "Material", "PO Qty", "Open Qty",
                                "Open Value ($)", "Due Date", "Risk"]
    st.dataframe(display_orders.style.applymap(color_order_risk, subset=["Risk"]),
                 use_container_width=True, height=450)

    st.markdown("**📅 Monthly Order Volume by Risk Level**")
    monthly = orders_merged.groupby(["Month", "Risk_Label"])["PO_Quantity"].sum().reset_index()
    monthly_pivot = monthly.pivot(index="Month", columns="Risk_Label", values="PO_Quantity").fillna(0)
    st.bar_chart(monthly_pivot)

# ── Tab 3: Disruption & Backup Analysis ──────────────────────
with tab3:
    st.subheader("Disruption Analysis & Backup Supplier Recommendations")
    high_risk_suppliers = suppliers[suppliers["Risk_Label"] == "High"]["Supplier"].tolist()

    if high_risk_suppliers:
        selected = st.selectbox("Select a high-risk supplier to analyze:", high_risk_suppliers)
        idx = suppliers[suppliers["Supplier"] == selected].index[0]
        row = suppliers.loc[idx]

        supplier_future_orders = orders[orders["Supplier"] == selected]
        future_order_count = len(supplier_future_orders)
        future_qty = supplier_future_orders["PO_Quantity"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("On-Time Rate", f"{row['On_Time_Rate']:.0%}")
        col2.metric("Fulfillment Rate", f"{row['Fulfillment_Rate']:.0%}")
        col3.metric("2026 Orders Scheduled", future_order_count)

        st.markdown(f"""
        **Disruption Summary — {selected}**

        > Supplier **{selected}** ({row['City']}) has been classified as **High Risk** based on the following signals:
        > - On-Time Delivery Rate: **{row['On_Time_Rate']:.0%}** (below acceptable threshold)
        > - Order Fulfillment Rate: **{row['Fulfillment_Rate']:.0%}**
        > - Total Historical Purchase Orders: **{int(row['Total_Orders'])}**
        > - Open Order Value: **${row['Open_Value']:,.2f}**
        > - 2026 Scheduled Orders: **{future_order_count} POs** totaling **{future_qty:,.0f} units**
        > - Category: **{row['Category']}**
        >
        > ⚠️ Immediate procurement review is recommended. With **{future_order_count} future orders**
        > scheduled, proactive engagement with backup suppliers is critical before disruption
        > impacts 2026 operations.
        """)

        if future_order_count > 0:
            st.markdown("**📋 Upcoming 2026 Orders from this Supplier:**")
            future_display = supplier_future_orders[["PO_Number", "Material", "PO_Quantity", "Due_Date"]].copy()
            future_display["Due_Date"] = pd.to_datetime(future_display["Due_Date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(future_display, use_container_width=True)

        st.markdown("**🔁 Top Backup Supplier Recommendations (AI-Matched):**")
        backups = get_backups(idx, suppliers, embeddings)
        st.dataframe(backups, use_container_width=True)
    else:
        st.success("✅ No high-risk suppliers detected at this time.")

st.divider()
st.caption("SupplySignal | AI-Powered Supply Chain Risk Intelligence | MSCM Applied AI Project")
