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

# ── Simulated Supplier Data ───────────────────────────────────
@st.cache_data
def load_data():
    suppliers = pd.DataFrame({
        "Supplier": ["AlphaCore", "BetaWorks", "GammaTech", "DeltaSupply", "EpsilonMfg",
                     "ZetaParts", "EtaGoods", "ThetaLogix", "IotaMaterials", "KappaCo"],
        "On_Time_Rate": [0.95, 0.60, 0.88, 0.45, 0.92, 0.78, 0.55, 0.91, 0.70, 0.83],
        "Fulfillment_Rate": [0.97, 0.65, 0.90, 0.50, 0.95, 0.80, 0.60, 0.93, 0.72, 0.85],
        "Incident_Days_Ago": [120, 15, 90, 5, 200, 60, 20, 150, 45, 75],
        "Region": ["North America", "Asia", "Europe", "Asia", "North America",
                   "Europe", "South America", "North America", "Asia", "Europe"],
        "Category": ["Electronics", "Electronics", "Mechanical", "Electronics", "Mechanical",
                     "Mechanical", "Electronics", "Mechanical", "Electronics", "Mechanical"],
        "Capacity_Units": [5000, 3000, 4500, 2000, 6000, 3500, 2500, 5500, 3200, 4000],
        "Profile": [
            "High-volume electronics supplier in North America with strong delivery record",
            "Mid-size electronics manufacturer in Asia with recent delays",
            "Reliable European mechanical parts supplier with good capacity",
            "Small Asian electronics supplier with frequent disruptions",
            "Large North American mechanical manufacturer with excellent performance",
            "European mechanical supplier with moderate performance",
            "South American electronics supplier with inconsistent fulfillment",
            "Dependable North American mechanical supplier with high capacity",
            "Asian electronics supplier with average performance and recent incidents",
            "Stable European mechanical supplier with good track record"
        ]
    })

    orders = pd.DataFrame({
        "Order_ID": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
        "Supplier": ["BetaWorks", "DeltaSupply", "ZetaParts", "EtaGoods", "IotaMaterials"],
        "Units_Ordered": [1200, 800, 600, 400, 1000],
        "Due_Date": ["2026-03-01", "2026-03-10", "2026-03-15", "2026-03-20", "2026-03-25"],
        "Value_USD": [48000, 32000, 24000, 16000, 40000]
    })

    return suppliers, orders

suppliers, orders = load_data()

# ── Train Risk Classifier ─────────────────────────────────────
@st.cache_resource
def train_model(suppliers):
    def label_risk(row):
        if row["On_Time_Rate"] < 0.65 or row["Fulfillment_Rate"] < 0.65 or row["Incident_Days_Ago"] < 20:
            return "High"
        elif row["On_Time_Rate"] < 0.85 or row["Fulfillment_Rate"] < 0.85 or row["Incident_Days_Ago"] < 60:
            return "Medium"
        else:
            return "Low"

    suppliers["Risk"] = suppliers.apply(label_risk, axis=1)
    features = suppliers[["On_Time_Rate", "Fulfillment_Rate", "Incident_Days_Ago", "Capacity_Units"]]
    le = LabelEncoder()
    labels = le.fit_transform(suppliers["Risk"])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    return clf, le, suppliers

clf, le, suppliers = train_model(suppliers)

# ── Predict Risk ──────────────────────────────────────────────
features = suppliers[["On_Time_Rate", "Fulfillment_Rate", "Incident_Days_Ago", "Capacity_Units"]]
suppliers["Risk_Label"] = le.inverse_transform(clf.predict(features))

# ── Embedding-Based Backup Matching ──────────────────────────
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
    backups = suppliers.iloc[top_indices][["Supplier", "Region", "Category", "Capacity_Units", "Risk_Label"]].copy()
    backups["Match_Score"] = [f"{sim_scores[i]:.0%}" for i in top_indices]
    return backups

# ── Layout: Three Panels ──────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Supplier Risk Dashboard")
    def color_risk(val):
        colors = {"High": "background-color: #ffcccc", 
                  "Medium": "background-color: #fff3cc",
                  "Low": "background-color: #ccffcc"}
        return colors.get(val, "")

    display = suppliers[["Supplier", "On_Time_Rate", "Fulfillment_Rate", 
                          "Incident_Days_Ago", "Region", "Category", "Risk_Label"]].copy()
    display.columns = ["Supplier", "On-Time Rate", "Fulfillment Rate", 
                       "Days Since Incident", "Region", "Category", "Risk"]
    st.dataframe(display.style.applymap(color_risk, subset=["Risk"]), use_container_width=True)

with col2:
    st.subheader("📦 Open Orders at Risk")
    high_risk = suppliers[suppliers["Risk_Label"] == "High"]["Supplier"].tolist()
    at_risk_orders = orders[orders["Supplier"].isin(high_risk)]
    if not at_risk_orders.empty:
        st.dataframe(at_risk_orders, use_container_width=True)
        total_value = at_risk_orders["Value_USD"].sum()
        st.error(f"⚠️ Total Order Value at Risk: **${total_value:,.0f}**")
    else:
        st.success("No open orders are currently at risk.")

st.divider()

# ── Disruption Analysis & Backup Recommendations ──────────────
st.subheader("🔄 Disruption Analysis & Backup Supplier Recommendations")
high_risk_suppliers = suppliers[suppliers["Risk_Label"] == "High"]["Supplier"].tolist()

if high_risk_suppliers:
    selected = st.selectbox("Select a high-risk supplier to analyze:", high_risk_suppliers)
    idx = suppliers[suppliers["Supplier"] == selected].index[0]
    row = suppliers.loc[idx]

    st.markdown(f"""
    **Disruption Summary — {selected}**
    
    > Supplier **{selected}** has been classified as **High Risk** based on the following signals:
    > - On-Time Delivery Rate: **{row['On_Time_Rate']:.0%}** (below acceptable threshold)
    > - Order Fulfillment Rate: **{row['Fulfillment_Rate']:.0%}**
    > - Last Incident: **{row['Incident_Days_Ago']} days ago**
    > - Region: **{row['Region']}** | Category: **{row['Category']}**
    >
    > Immediate procurement review is recommended. Open orders assigned to this supplier 
    > are at elevated risk of delay or non-fulfillment.
    """)

    st.markdown("**Top Backup Supplier Recommendations:**")
    backups = get_backups(idx, suppliers, embeddings)
    st.dataframe(backups, use_container_width=True)
else:
    st.success("✅ No high-risk suppliers detected at this time.")

st.divider()
st.caption("SupplySignal | AI-Powered Supply Chain Risk Intelligence | MSCM Applied AI Project")
