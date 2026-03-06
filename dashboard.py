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

# ─── Configuration & Page Branding ───────────────────────────────────────────
st.set_page_config(page_title="LoRRI · SaaS Intelligence Console", layout="wide", page_icon="🚚")

# ─── Intelligence Core (Self-Contained Logic) ───────────────────────────────
def get_rag_response(query: str) -> str:
    kb = [
        "LoRRI AI utilizes a Multi-Objective CVRP model: Cost (35%), Time (30%), Carbon (20%), and SLA (15%).",
        "The 'Lane Ranking System' evaluates carriers via Quality, Performance, Price, Reliability, and Sustainability.",
        "Predictive Analytics uses Random Forest for Demand and Isolation Forest for Anomaly Detection.",
        "Re-optimization is triggered by traffic spikes > 30% or urgent priority escalations.",
        "Sustainability tracking measures net CO2 reduction against unoptimized sequential baselines."
    ]
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    if not query_tokens: return "Console Ready. Query architecture pillars..."
    best_score, best_response = 0.0, "Context not found. Try asking about CVRP weights or Carrier Pillars."
    for chunk in kb:
        chunk_tokens = re.findall(r'\b\w+\b', chunk.lower())
        score = sum(Counter(chunk_tokens)[t] for t in query_tokens if t in chunk_tokens)
        norm_score = score / (math.log(len(chunk_tokens) + 1.1))
        if norm_score > best_score:
            best_score, best_response = norm_score, chunk
    return best_response

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
        st.error("📡 Telemetry Offline: Backend CSV files (shipments, routes, metrics) not detected.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"lat": 19.0760, "lon": 72.8777}
COLORS = px.colors.qualitative.Bold

# ─── Master-Tier Visual Styling (High-Density Dark SaaS) ────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f1a; font-family: 'Inter', sans-serif; color: #f1f5f9; }
    
    /* Tighter Sidebar */
    section[data-testid="stSidebar"] { background-color: #131926 !important; border-right: 1px solid #2d3748; }
    section[data-testid="stSidebar"] .stMarkdown h1 { color: #38bdf8 !important; font-size: 1.1rem; font-weight: 800; margin-bottom: 0px; }
    
    /* Compact Metric Cards */
    [data-testid="stMetric"] {
        background: #1a202c;
        border: 1px solid #2d3748;
        padding: 10px 15px !important;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.4rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
    
    /* Low-Padding Panel Cards */
    .saas-card {
        background: #131926;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2d3748;
        margin-bottom: 10px;
    }
    
    /* Typography refinements */
    .section-head {
        font-size: 0.75rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border-left: 2px solid #38bdf8;
        padding-left: 8px;
    }

    .terminal-box {
        background: #000;
        color: #10b981;
        padding: 12px;
        border-radius: 6px;
        font-family: 'Consolas', monospace;
        font-size: 0.8rem;
    }

    /* Remove Streamlit default top padding */
    .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=40)
    st.title("LoRRI INTELLIGENCE")
    module = st.radio("SELECT WORKSPACE", 
                    ["🗺️ Geospatial Control", "📊 Executive Performance", "🧠 AI Strategy Hub", "💬 Copilot Terminal"])
    st.divider()
    st.markdown("### **Deployment State**")
    st.markdown("🟢 **Model:** Production-v4.2\n🟢 **Optimization:** OPTIMAL")
    st.progress(92, text="Fleet Load: 92%")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: GEOSPATIAL CONTROL (The Masterpiece Map)
# ═══════════════════════════════════════════════════════════════════
if module == "🗺️ Geospatial Control":
    st.markdown("<div class='section-head'>Network Geospatial Telemetry</div>", unsafe_allow_html=True)
    col_map, col_ctrl = st.columns([4, 1], gap="small")
    
    with col_ctrl:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.caption("Map Controller")
        show_unassigned = st.toggle("Unassigned Risk", value=True)
        v_filter = st.multiselect("Fleet Filter", sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        
        st.divider()
        st.markdown("**Legend**")
        st.markdown("<span style='color:#ef4444'>●</span> Urgent")
        st.markdown("<span style='color:#f97316'>●</span> Standard")
        if show_unassigned: st.markdown("<span style='color:#64748b'>○</span> Unassigned")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.caption("Fleet Utilization")
        st.metric("Total Nodes", len(routes))
        st.metric("SLA Status", "Nominal", delta="+4%")
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
                fig_map.add_trace(go.Scattermapbox(lat=un_df["latitude"], lon=un_df["longitude"], mode="markers", marker=dict(size=12, color="rgba(148, 163, 184, 0.4)", symbol="circle"), name="Unassigned Load", hovertext=un_df["city"]))

        fig_map.add_trace(go.Scattermapbox(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", text=["🏭 HUB: MUMBAI"], marker=dict(size=20, color="#38bdf8", symbol="star"), name="Origin"))
        fig_map.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(center=dict(lat=20.5, lon=78.9), zoom=4.1), height=720, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: EXECUTIVE PERFORMANCE (Baseline vs Optimized)
# ═══════════════════════════════════════════════════════════════════
elif module == "📊 Executive Performance":
    st.markdown("<div class='section-head'>Baseline vs. AI Optimized ROI</div>", unsafe_allow_html=True)
    
    # Restored high-density baseline comparisons from original
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("📏 Distance Savings", f"{(metrics['baseline_distance_km'] - metrics['opt_distance_km']):,.0f} km", f"-{((metrics['baseline_distance_km']-metrics['opt_distance_km'])/metrics['baseline_distance_km']*100):.1f}%")
    c2.metric("⏱️ Efficiency Gain", f"{(metrics['baseline_time_hr'] - metrics['opt_time_hr']):,.1f} hr", "Operational ↑")
    c3.metric("💰 Cost Optimization", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost']):,.0f}", "Net Margin Gain")
    c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct'] - metrics['baseline_sla_adherence_pct']} pts")

    col_l, col_r = st.columns([1.6, 1], gap="small")
    with col_l:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","relative","total"], x=["Fuel Sav", "Toll Sav", "Wage Sav", "NET ROI"],
            y=[metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'], metrics['baseline_toll_cost']-metrics['opt_toll_cost'], metrics['baseline_driver_cost']-metrics['opt_driver_cost'], metrics['baseline_total_cost']-metrics['opt_total_cost']],
            connector={"line":{"color":"#334155"}}, decreasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#38bdf8"}}))
        fig_wf.update_layout(title="Financial Savings Breakdown (₹)", template="plotly_dark", height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        fig_bar = px.bar(x=["Baseline", "Optimized"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], color=["Base", "Opt"], color_discrete_map={"Base":"#ef4444", "Opt":"#10b981"}, title="Sustainability Impact (kg CO₂)")
        fig_bar.update_layout(template="plotly_dark", height=420, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: AI STRATEGY HUB
# ═══════════════════════════════════════════════════════════════════
elif module == "🧠 AI Strategy Hub":
    st.markdown("<div class='section-head'>Carrier Logic & Risk Architecture</div>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Carrier Scoring Matrix", "Disruption Simulator"])
    
    with t1:
        colA, colB = st.columns([1, 1.5], gap="small")
        with colA:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.caption("Flowchart Pillars")
            pillars = pd.DataFrame({"Pillar": ["Quality", "Performance", "Price", "Reliability", "Sustainability"], "Score": [92, 88, 75, 95, 82]})
            fig_radar = px.line_polar(pillars, r='Score', theta='Pillar', line_close=True, template="plotly_dark", color_discrete_sequence=['#38bdf8'])
            fig_radar.update_traces(fill='toself')
            fig_radar.update_layout(height=350, margin=dict(t=30, b=30, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.caption("Decision Weights")
            fig_pie = px.pie(names=["Fuel", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6)
            fig_pie.update_layout(template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.caption("Re-optimization Sandbox")
        c_sim1, c_sim2 = st.columns(2, gap="small")
        with c_sim1:
            city = st.selectbox("Impact Node:", sorted(ships["city"].unique()))
            st.button("Trigger Gridlock Event", type="primary")
        with c_sim2:
            st.info(f"Analyzing re-route variants for {city}...")
        st.markdown("</div>", unsafe_allow_html=True)
        
        risk_df = pd.DataFrame({"City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Raipur"], "Risk": [0.98, 0.95, 0.92, 0.88, 0.82]})
        st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="Top Cities by Re-Optimization Risk", template="plotly_dark"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: INTELLIGENCE TERMINAL
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 Copilot Terminal":
    st.markdown("<div class='section-head'>LoRRI Technical Core retrieval</div>", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Ask about CVRP weights or carrier performance pillars..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        ans = get_rag_response(p)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='terminal-box'>{ans}</div>", unsafe_allow_html=True)

st.divider()
st.caption("LoRRI SaaS Intelligence Console · Built with Enterprise-Grade SaaS Constraints")
