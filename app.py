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

# ── Layout ────────────────────────────────────────────────────
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("📊 Supplier Risk Dashboard")

    def color_risk(val):
        colors = {
            "High": "background-color: #ffcccc",
            "Medium": "background-color: #fff3cc",
            "Low": "background-color: #ccffcc"
        }
        return colors.get(val, "")

    display = suppliers[["Supplier", "City", "Category", "On_Time_Rate", "Fulfillment_Rate", "Total_Orders", "Risk_Label"]].copy()
    display.columns = ["Supplier", "City", "Category", "On-Time Rate", "Fulfillment Rate", "Total Orders", "Risk"]
    st.dataframe(display.style.applymap(color_risk, subset=["Risk"]), use_container_width=True, height=400)

with col2:
    st.subheader("📦 Open Orders at Risk")
    high_risk = suppliers[suppliers["Risk_Label"] == "High"]["Vendor"].tolist()
    at_risk_orders = orders[orders["Vendor"].isin(high_risk)]

    if not at_risk_orders.empty:
        st.dataframe(at_risk_orders[["PO_Number", "Supplier", "Open_Quantity", "Open_Value", "Due_Date", "Material"]], 
                     use_container_width=True)
        total_value = at_risk_orders["Open_Value"].sum()
        st.error(f"⚠️ Total Order Value at Risk: **${total_value:,.2f}**")
    else:
        st.success("✅ No open orders are currently linked to high-risk suppliers.")

    # Risk summary metrics
    st.subheader("📈 Risk Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk", len(suppliers[suppliers["Risk_Label"] == "High"]))
    c2.metric("🟡 Medium Risk", len(suppliers[suppliers["Risk_Label"] == "Medium"]))
    c3.metric("🟢 Low Risk", len(suppliers[suppliers["Risk_Label"] == "Low"]))

st.divider()

# ── Disruption Analysis ───────────────────────────────────────
st.subheader("🔄 Disruption Analysis & Backup Supplier Recommendations")
high_risk_suppliers = suppliers[suppliers["Risk_Label"] == "High"]["Supplier"].tolist()

if high_risk_suppliers:
    selected = st.selectbox("Select a high-risk supplier to analyze:", high_risk_suppliers)
    idx = suppliers[suppliers["Supplier"] == selected].index[0]
    row = suppliers.loc[idx]

    st.markdown(f"""
    **Disruption Summary — {selected}**

    > Supplier **{selected}** ({row['City']}) has been classified as **High Risk** based on the following signals:
    > - On-Time Delivery Rate: **{row['On_Time_Rate']:.0%}** (below acceptable threshold)
    > - Order Fulfillment Rate: **{row['Fulfillment_Rate']:.0%}**
    > - Total Purchase Orders on Record: **{int(row['Total_Orders'])}**
    > - Open Order Value: **${row['Open_Value']:,.2f}**
    > - Category: **{row['Category']}**
    >
    > Immediate procurement review is recommended. Proactive engagement with backup 
    > suppliers is advised before disruption escalates to open orders.
    """)

    st.markdown("**🔁 Top Backup Supplier Recommendations (AI-Matched):**")
    backups = get_backups(idx, suppliers, embeddings)
    st.dataframe(backups, use_container_width=True)
else:
    st.success("✅ No high-risk suppliers detected at this time.")

st.divider()
st.caption("SupplySignal | AI-Powered Supply Chain Risk Intelligence | MSCM Applied AI Project")
