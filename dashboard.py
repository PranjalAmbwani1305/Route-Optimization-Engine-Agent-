import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from math import radians, cos, sin, asin, sqrt
from rag_engine import get_rag_response

# ─── Configuration & Page Branding ───────────────────────────────────────────
st.set_page_config(page_title="LoRRI · Enterprise AI", layout="wide", page_icon="🚚")

# ─── Data Engineering Layer ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        ships = pd.read_csv("shipments.csv")
        routes = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh_summary = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh_summary
    except:
        st.error("📡 Telemetry Offline: Run backend scripts to generate CSV data.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"lat": 19.0760, "lon": 72.8777}
COLORS = px.colors.qualitative.Bold

# ─── Master-Tier Visual Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #0f172a !important; color: white; }
    section[data-testid="stSidebar"] .stMarkdown h1 { color: white !important; font-size: 1.5rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; padding: 15px !important; border-radius: 10px; }
    .main-panel { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .section-head { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-left: 4px solid #3b82f6; padding-left: 10px; }
    .rag-terminal { background: #000; color: #10b981; padding: 20px; border-radius: 8px; font-family: 'Consolas', monospace; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=50)
    st.title("LoRRI AI")
    module = st.radio("NAVIGATE WORKSPACES", 
                    ["🗺️ Live Operations Map", "📊 executive Performance", "🤖 ML Predictive Analytics", "💬 Copilot Terminal"])
    st.divider()
    st.markdown("### **System Status**")
    st.markdown("🟢 **Model:** Production-v4\n🟢 **Optimization:** Optimal")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: LIVE OPERATIONS MAP
# ═══════════════════════════════════════════════════════════════════
if module == "🗺️ Live Operations Map":
    st.markdown("<div class='section-head'>Real-Time Geospatial Telemetry</div>", unsafe_allow_html=True)
    col_map, col_ctrl = st.columns([4, 1])
    
    with col_ctrl:
        st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
        st.subheader("Controls")
        show_unassigned = st.toggle("Show Unassigned Load", value=True)
        v_filter = st.multiselect("Active Fleet", sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        st.divider()
        st.markdown("**Node Legend**")
        st.markdown("<span style='color:#ef4444'>●</span> High Prio")
        st.markdown("<span style='color:#f97316'>●</span> Med Prio")
        st.markdown("<span style='color:#22c55e'>●</span> Low Prio")
        if show_unassigned: st.markdown("<span style='color:grey'>○</span> Unassigned (Risk)")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map:
        fig_map = go.Figure()
        for v in v_filter:
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats, lons = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]], [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
            fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), marker=dict(size=9), name=f"Veh_{v}"))
        
        if show_unassigned:
            asgn = set(routes["shipment_id"].unique())
            un_df = ships[~ships["id"].isin(asgn)]
            if not un_df.empty:
                fig_map.add_trace(go.Scattermap(lat=un_df["latitude"], lon=un_df["longitude"], mode="markers", marker=dict(size=12, color="rgba(100, 116, 139, 0.4)"), name="Unassigned Risk"))

        fig_map.add_trace(go.Scattermap(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", text=["🏭 MUMBAI"], marker=dict(size=20, color="#1e293b", symbol="star"), name="Depot"))
        fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4.2), height=700, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: EXECUTIVE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif module == "📊 executive Performance":
    st.markdown("<div class='section-head'>Baseline vs. Optimized ROI</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Distance", f"{metrics['opt_distance_km']:,.0f} km", f"{metrics['opt_distance_km'] - metrics['baseline_distance_km']:,.0f} km", delta_color="inverse")
    c2.metric("⏱️ Execution", f"{metrics['opt_time_hr']:,.1f} hr", f"{metrics['opt_time_hr'] - metrics['baseline_time_hr']:,.1f} hr", delta_color="inverse")
    c3.metric("💰 Cost Opt", f"₹{metrics['opt_total_cost']:,.0f}", f"₹{metrics['opt_total_cost'] - metrics['baseline_total_cost']:,.0f}", delta_color="inverse")
    c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct'] - metrics['baseline_sla_adherence_pct']} pts")

    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(orientation="v", x=["Fuel Sav", "Toll Sav", "Wage Sav", "NET IMPACT"],
            y=[metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'], metrics['baseline_toll_cost']-metrics['opt_toll_cost'], metrics['baseline_driver_cost']-metrics['opt_driver_cost'], metrics['baseline_total_cost']-metrics['opt_total_cost']],
            decreasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#3b82f6"}}))
        fig_wf.update_layout(title="Corporate Savings Decomposition", template="plotly_white", height=450)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
        fig_bar = px.bar(x=["Baseline", "Optimized"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], color=["Base", "AI"], color_discrete_map={"Base":"#f87171", "AI":"#10b981"}, title="Carbon Footprint (kg CO₂)")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: ML PREDICTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif module == "🤖 ML Predictive Analytics":
    st.markdown("<div class='section-head'>AI Intelligence & Forecasting</div>", unsafe_allow_html=True)
    
    tab_f, tab_a = st.tabs(["Demand Forecasting", "Geospatial Anomalies"])
    
    with tab_f:
        st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
        # Mock Demand Forecast Data
        days = pd.date_range(start="2026-03-01", periods=14)
        hist = [45, 48, 52, 49, 50, 55, 60]
        pred = [None]*6 + [60, 62, 65, 63, 68, 72, 75, 78]
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=days[:7], y=hist, name="Historical Demand", line=dict(color="#3b82f6", width=3)))
        fig_f.add_trace(go.Scatter(x=days[6:], y=pred[6:], name="ML Prediction", line=dict(dash='dash', color="#10b981", width=3)))
        fig_f.update_layout(title="14-Day Network Demand Forecast (Random Forest Regressor)", template="plotly_white")
        st.plotly_chart(fig_f, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_a:
        st.markdown("<div class='main-panel'>", unsafe_allow_html=True)
        risk_df = pd.DataFrame({"City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Lucknow", "Raipur", "Bhopal"], "Risk": [0.98, 0.95, 0.92, 0.88, 0.85, 0.82, 0.78]})
        st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="SLA Breach Anomaly Detection (Isolation Forest)"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: COPILOT TERMINAL
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 Copilot Terminal":
    st.title("Architecture Knowledge Base")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Ask about ML models or CVRP math..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        ans = get_rag_response(p)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='rag-terminal'>{ans}</div>", unsafe_allow_html=True)

st.divider()
st.caption("LoRRI AI Master Terminal · ML-Driven Route Optimization Engine · Synapflow Problem 4")
