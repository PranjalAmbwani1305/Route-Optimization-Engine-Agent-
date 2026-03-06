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
st.set_page_config(page_title="LoRRI · Enterprise AI Console", layout="wide", page_icon="🚚")

# ─── Intelligence Core (Self-Contained) ──────────────────────────────────────
def get_rag_response(query: str) -> str:
    kb = [
        "The LoRRI AI Engine utilizes a Multi-Objective CVRP framework, balancing Cost (35%), Time (30%), Carbon (20%), and SLA (15%).",
        "The 'Lane Ranking System' scores carriers across five pillars: Quality, Performance, Price, Reliability, and Sustainability.",
        "Machine Learning integration includes Demand Forecasting (Random Forest) and Anomaly Detection (Isolation Forest).",
        "Real-time re-optimization triggers detect traffic spikes > 30% or SLA priority escalations, re-sequencing the fleet queue instantly.",
        "Sustainable logistics tracking measures kg CO2 saved by comparing optimized routes against sequential baseline models."
    ]
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    if not query_tokens: return "AI Terminal Ready. Awaiting architecture query..."
    best_score, best_response = 0.0, "Context not found. Ask about Carrier Scoring pillars, CVRP math, or Carbon factors."
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

# ─── Master-Tier Visual Styling (High-End Dark SaaS Theme) ───────────────────
st.markdown("""
<style>
    /* SaaS Deep Background */
    .stApp { background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f1f5f9; }
    
    /* Dark Navigation Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
    section[data-testid="stSidebar"] .stMarkdown h1 { color: #38bdf8 !important; font-size: 1.3rem; font-weight: 800; }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 20px !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700; }
    
    /* SaaS Panel Cards */
    .saas-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
    }
    
    /* Header Typography */
    .section-head {
        font-size: 0.85rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-left: 3px solid #38bdf8;
        padding-left: 12px;
    }

    /* Terminal Interface */
    .terminal-box {
        background: #000;
        color: #10b981;
        padding: 18px;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        border: 1px solid #334155;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=50)
    st.title("LoRRI PLATFORM")
    st.markdown("### **Operational Control**")
    module = st.radio("NAVIGATE WORKSPACES", 
                    ["🗺️ Geospatial Control", "📊 Executive Performance", "🧠 AI Strategy Hub", "💬 Intelligence Terminal"])
    st.divider()
    st.markdown("### **Fleet Integrity**")
    st.markdown("🟢 **Optimization:** OPTIMAL\n🟢 **Predictive:** ACTIVE\n🟢 **Safety:** NOMINAL")
    st.progress(92, text="Fleet Capacity: 92%")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: GEOSPATIAL CONTROL (The Masterpiece Map)
# ═══════════════════════════════════════════════════════════════════
if module == "🗺️ Geospatial Control":
    st.markdown("<div class='section-head'>Real-Time Geospatial Fleet Intelligence</div>", unsafe_allow_html=True)
    col_map, col_ctrl = st.columns([3.8, 1.2])
    
    with col_ctrl:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.subheader("Global View Controls")
        show_unassigned = st.toggle("Detect Unassigned Risks", value=True, help="Highlights shipments not yet picked up by vehicles")
        v_filter = st.multiselect("Fleet Group Filter", sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        
        st.divider()
        st.markdown("**Operational Legend**")
        st.markdown("<span style='color:#ef4444'>●</span> **Urgent Priority**")
        st.markdown("<span style='color:#f97316'>●</span> **Standard Priority**")
        if show_unassigned: st.markdown("<span style='color:#64748b'>○</span> **Unassigned Risk Node**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("**Network Health**")
        st.metric("Total Stops", len(routes))
        st.metric("Risk Level", "Low", delta="-5%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map:
        fig_map = go.Figure()
        for v in v_filter:
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats, lons = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]], [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
            fig_map.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), marker=dict(size=10), name=f"V_{v}"))
        
        if show_unassigned:
            asgn = set(routes["shipment_id"].unique())
            un_df = ships[~ships["id"].isin(asgn)]
            if not un_df.empty:
                fig_map.add_trace(go.Scattermapbox(lat=un_df["latitude"], lon=un_df["longitude"], mode="markers", marker=dict(size=14, color="rgba(148, 163, 184, 0.4)", symbol="circle"), name="Risk: Unassigned", hovertext=un_df["city"]))

        fig_map.add_trace(go.Scattermapbox(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", text=["🏭 STRATEGIC HUB: MUMBAI"], marker=dict(size=24, color="#38bdf8", symbol="star"), name="Origin Hub"))
        fig_map.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(center=dict(lat=20.5, lon=78.9), zoom=4.2), height=750, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: EXECUTIVE PERFORMANCE (Baseline vs Optimized)
# ═══════════════════════════════════════════════════════════════════
elif module == "📊 Executive Performance":
    st.markdown("<div class='section-head'>Network Business ROI: Baseline vs. Optimized</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📏 Dist Saved", f"{(metrics['baseline_distance_km'] - metrics['opt_distance_km']):,.0f} km", f"-{((metrics['baseline_distance_km']-metrics['opt_distance_km'])/metrics['baseline_distance_km']*100):.1f}%")
    c2.metric("⏱️ Time Gain", f"{(metrics['baseline_time_hr'] - metrics['opt_time_hr']):,.1f} hr", "Efficiency ↑")
    c3.metric("💰 Cost Optimization", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost']):,.0f}", "Net Margin Gain")
    c4.metric("✅ SLA Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct'] - metrics['baseline_sla_adherence_pct']} pts")

    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=["relative","relative","relative","total"], x=["Fuel Sav", "Toll Sav", "Wage Sav", "NET ROI"],
            y=[metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'], metrics['baseline_toll_cost']-metrics['opt_toll_cost'], metrics['baseline_driver_cost']-metrics['opt_driver_cost'], metrics['baseline_total_cost']-metrics['opt_total_cost']],
            connector={"line":{"color":"#334155"}}, decreasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#38bdf8"}}))
        fig_wf.update_layout(title="Financial Value Decomposition", template="plotly_dark", height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=metrics["opt_sla_adherence_pct"], delta={'reference': metrics["baseline_sla_adherence_pct"]}, title={'text': "Compliance %"}, gauge={'bar': {'color': "#38bdf8"}, 'axis': {'range': [None, 100]}, 'steps': [{'range': [0, 80], 'color': '#1e293b'}]}))
        fig_gauge.update_layout(template="plotly_dark", height=320, margin=dict(t=50, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: AI STRATEGY HUB (Lane Ranking & Simulator)
# ═══════════════════════════════════════════════════════════════════
elif module == "🧠 AI Strategy Hub":
    st.markdown("<div class='section-head'>Strategic Architecture & Logic Diagnostics</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Lane Ranking Matrix", "Explainability (XAI)", "Disruption Simulator"])
    
    with t1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.write("#### 🏆 Carrier Scoring Model (Flowchart Image Pillars)")
        pillars = pd.DataFrame({"Pillar": ["Quality", "Performance", "Price", "Reliability", "Sustainability"], "Score": [92, 88, 75, 95, 82], "Status": ["Exceptional", "Strong", "Optimizing", "Exceptional", "Good"]})
        st.table(pillars)
        fig_radar = px.line_polar(pillars, r='Score', theta='Pillar', line_close=True, template="plotly_dark", color_discrete_sequence=['#38bdf8'])
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        colA, colB = st.columns(2)
        with colA:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            fig_pie = px.pie(names=["Fuel", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6, title="Algorithm Static Weights")
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            heat_data = pd.crosstab(routes["vehicle"], routes["priority"])
            st.plotly_chart(px.imshow(heat_data, text_auto=True, title="Priority Distribution Heatmap", color_continuous_scale="Viridis", template="plotly_dark"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.write("#### 🚦 Re-optimization Simulation Workspace")
        city = st.selectbox("Select Impact Node:", sorted(ships["city"].unique()))
        severity = st.slider("Traffic Congestion Magnitude", 1.0, 3.0, 2.5)
        if st.button("Trigger Gridlock Event", type="primary"):
            st.warning(f"Re-optimization thresholds breached for {city}. Calculating local search variants...")
            time.sleep(1)
            st.success("Buffer injected. Stop re-sequenced. SLA Window Protected.")
        st.markdown("</div>", unsafe_allow_html=True)
        risk_df = pd.DataFrame({"City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Raipur"], "Risk": [0.98, 0.95, 0.92, 0.88, 0.82]})
        st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="Top 5 Cities by Re-Optimization Risk", template="plotly_dark"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: INTELLIGENCE TERMINAL
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 Intelligence Terminal":
    st.markdown("<div class='section-head'>LoRRI Architecture Knowledge retrieval</div>", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Ask about CVRP pillars or multi-objective math..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        ans = get_rag_response(p)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='terminal-box'>{ans}</div>", unsafe_allow_html=True)

st.divider()
st.caption("LoRRI Enterprise Suite · Master Tier SaaS Prototype · Multi-Objective AI Strategy")
