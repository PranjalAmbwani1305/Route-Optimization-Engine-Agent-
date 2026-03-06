import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import re
from collections import Counter
import math
from math import radians, cos, sin, asin, sqrt

# ─── Configuration ───────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI · AI Route Optimization", layout="wide", page_icon="🚚")

# ─── Helpers & Original Logic ───────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except:
        st.error("📡 Telemetry Offline: Backend CSV files not detected.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"lat": 19.0760, "lon": 72.8777}
COLORS = px.colors.qualitative.Bold

# ─── Permutation Importance Logic (Restored) ────────────────────────────────
@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {"Travel Time": "travel_time_hr", "Fuel Cost": "fuel_cost", "Toll Cost": "toll_cost",
                "Driver Cost": "driver_cost", "Carbon": "carbon_kg", "SLA Breach": "sla_breach_hr"}
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    baseline_mae = np.mean(np.abs(y - y.mean()))
    importances = {}
    for label, col in features.items():
        shuffled = X.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        proxy = shuffled.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        importances[label] = abs(np.mean(np.abs(y - proxy.values)) - baseline_mae)
    total = sum(importances.values()) + 1e-9
    return {k: round(v / total * 100, 1) for k, v in sorted(importances.items(), key=lambda x: -x[1])}

# ─── Copilot Logic (Restored) ────────────────────────────────────────────────
def get_rag_response(query: str) -> str:
    kb = [
        "LoRRI AI utilizes a Multi-Objective CVRP model: Cost (35%), Time (30%), Carbon (20%), and SLA (15%).",
        "The engine balances travel time, fuel, toll, driver cost, and SLA penalties.",
        "The re-optimization engine re-plans routes when traffic spikes > 30% or priorities change.",
        "Optimization yields 8-20% reduction in travel distance and 5-15% cost savings."
    ]
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    if not query_tokens: return "Console Ready."
    best_score, best_response = 0.0, "Context not found. Ask about CVRP weights or cost factors."
    for chunk in kb:
        chunk_tokens = re.findall(r'\b\w+\b', chunk.lower())
        score = sum(Counter(chunk_tokens)[t] for t in query_tokens if t in chunk_tokens)
        norm_score = score / (math.log(len(chunk_tokens) + 1.1))
        if norm_score > best_score:
            best_score, best_response = norm_score, chunk
    return best_response

# ─── High-Density Visual Styling (Removing All Gaps) ────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f1a; font-family: 'Inter', sans-serif; color: #f1f5f9; }
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    
    /* Sidebar Tighter */
    section[data-testid="stSidebar"] { background-color: #131926 !important; border-right: 1px solid #2d3748; }
    
    /* Metrics Gaps */
    [data-testid="stMetric"] { background: #1a202c; border: 1px solid #2d3748; padding: 10px 12px !important; border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #38bdf8 !important; }
    
    /* Compact Panels */
    .saas-card { background: #131926; padding: 12px; border-radius: 8px; border: 1px solid #2d3748; margin-bottom: 8px; }
    
    /* Remove default Streamlit vertical space */
    .element-container { margin-bottom: 0.2rem !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem !important; }
    
    .section-head {
        font-size: 0.8rem; font-weight: 800; color: #38bdf8; margin-bottom: 6px;
        text-transform: uppercase; letter-spacing: 1px; border-left: 2px solid #38bdf8; padding-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=40)
    st.title("LoRRI INTELLIGENCE")
    module = st.radio("WORKSPACES", ["📊 EXECUTIVE PERFORMANCE", "🗺️ FLEET OPERATIONS", "🧠 AI ENGINE LOGIC", "💬 COPILOT TERMINAL"])
    st.divider()
    st.markdown("🟢 **Model:** OPTIMAL")
    st.progress(92, text="Fleet Load: 92%")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: EXECUTIVE & FINANCIALS
# ═══════════════════════════════════════════════════════════════════
if module == "📊 EXECUTIVE PERFORMANCE":
    st.markdown("<div class='section-head'>Network ROI Performance</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("📏 Distance", f"{metrics['opt_distance_km']:,.0f} km", f"{metrics['opt_distance_km']-metrics['baseline_distance_km']:,.0f} km", delta_color="inverse")
    c2.metric("⏱️ Time", f"{metrics['opt_time_hr']:,.1f} hr", f"{metrics['opt_time_hr']-metrics['baseline_time_hr']:,.1f} hr", delta_color="inverse")
    c3.metric("💰 Cost Opt", f"₹{metrics['opt_total_cost']:,.0f}", f"₹{metrics['opt_total_cost']-metrics['baseline_total_cost']:,.0f}", delta_color="inverse")
    c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']} pts")

    col_l, col_r = st.columns([1.6, 1], gap="small")
    with col_l:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","relative","total"], x=["Fuel Sav", "Toll Sav", "Wage Sav", "NET ROI"],
            y=[metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'], metrics['baseline_toll_cost']-metrics['opt_toll_cost'], metrics['baseline_driver_cost']-metrics['opt_driver_cost'], metrics['baseline_total_cost']-metrics['opt_total_cost']],
            connector={"line":{"color":"#334155"}}, decreasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#38bdf8"}}))
        fig_wf.update_layout(title="Financial Savings Waterfall", template="plotly_dark", height=380, margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.dataframe(veh_summary[["vehicle","stops","total_cost","utilization_pct"]].style.background_gradient(subset=["utilization_pct"], cmap="RdYlGn"), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: FLEET OPERATIONS
# ═══════════════════════════════════════════════════════════════════
elif module == "🗺️ FLEET OPERATIONS":
    st.markdown("<div class='section-head'>Real-Time Geospatial Telemetry</div>", unsafe_allow_html=True)
    col_map, col_ctrl = st.columns([4, 1], gap="small")
    with col_ctrl:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        show_unassigned = st.toggle("Unassigned Load", value=True)
        v_filter = st.multiselect("Fleet Filter", sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        st.divider()
        st.markdown("**SLA Compliance**")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=metrics["opt_sla_adherence_pct"], gauge={'bar': {'color': "#38bdf8"}, 'axis': {'range': [0, 100]}}))
        fig_g.update_layout(height=180, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map:
        fig_map = go.Figure()
        for v in v_filter:
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats, lons = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]], [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
            fig_map.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode="lines+markers", line=dict(width=2, color=COLORS[(v-1)%len(COLORS)]), marker=dict(size=8), name=f"V_{v}"))
        if show_unassigned:
            asgn = set(routes["shipment_id"].unique())
            un_df = ships[~ships["id"].isin(asgn)]
            if not un_df.empty:
                fig_map.add_trace(go.Scattermapbox(lat=un_df["latitude"], lon=un_df["longitude"], mode="markers", marker=dict(size=12, color="rgba(148, 163, 184, 0.4)"), name="Unassigned"))
        fig_map.add_trace(go.Scattermapbox(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", text=["🏭 Depot"], marker=dict(size=20, color="#38bdf8", symbol="star"), name="Origin"))
        fig_map.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(center=dict(lat=20.5, lon=78.9), zoom=4.1), height=680, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: AI ENGINE LOGIC
# ═══════════════════════════════════════════════════════════════════
elif module == "🧠 AI ENGINE LOGIC":
    st.markdown("<div class='section-head'>Diagnostic Intelligence</div>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Explainability", "Simulator"])
    with t1:
        colA, colB = st.columns([1, 2], gap="small")
        with colA:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.plotly_chart(px.pie(names=["Fuel", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6).update_layout(template="plotly_dark", height=350, margin=dict(t=0, b=0)), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            fi = compute_feature_importance(routes)
            st.plotly_chart(px.bar(x=list(fi.values()), y=list(fi.keys()), orientation='h', title="Permutation Importance").update_layout(template="plotly_dark", height=350, margin=dict(t=40, b=0)), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        city = st.selectbox("Impact Node:", sorted(ships["city"].unique()))
        st.button("Trigger Gridlock Event", type="primary")
        risk_df = pd.DataFrame({"City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Raipur"], "Risk": [0.98, 0.95, 0.92, 0.88, 0.82]})
        st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="Top Cities by Re-Optimization Risk").update_layout(template="plotly_dark", height=300), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: COPILOT TERMINAL
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 COPILOT TERMINAL":
    st.markdown("<div class='section-head'>LoRRI Knowledge retrieval</div>", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Ask about CVRP weights..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        ans = get_rag_response(p)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='terminal-box'>{ans}</div>", unsafe_allow_html=True)

st.divider()
st.caption("LoRRI AI Intelligence Console · Enterprise Strategy · Built with Streamlit")
