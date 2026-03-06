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

# ─── Master-Tier Visual Styling (Modern SaaS Theme) ─────────────────────────
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp { background-color: #f1f5f9; font-family: 'Inter', sans-serif; }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] { background-color: #0f172a !important; color: white; }
    section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2 { color: white !important; }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 20px !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Panel Containers */
    .main-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Header Typography */
    .section-head {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
        border-left: 4px solid #3b82f6;
        padding-left: 12px;
    }
    
    /* Chat Terminal UI */
    .rag-terminal {
        background: #000000;
        color: #10b981;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=60)
    st.title("LoRRI AI")
    st.markdown("### **Control Panel**")
    module = st.radio("OPERATIONAL MODULES", 
                    ["🗺️ Live Operations Map", "📊 executive Analytics", "🧠 Logic Diagnostics", "💬 Copilot Terminal"])
    st.divider()
    st.markdown("### **System Telemetry**")
    st.progress(92, text="Fleet Capacity: 92%")
    st.markdown("🟢 **Heuristic Solver:** ACTIVE")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: LIVE OPERATIONS MAP (The Masterpiece)
# ═══════════════════════════════════════════════════════════════════
if module == "🗺️ Live Operations Map":
    st.markdown("<div class='section-head'>Real-Time Geospatial Route Telemetry</div>", unsafe_allow_html=True)
    
    col_map, col_ctrl = st.columns([3.5, 1])
    
    with col_ctrl:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("🎛️ Map Controls")
        show_unassigned = st.toggle("Show Unassigned Load", value=True, help="Highlights shipments not yet picked up by AI")
        v_filter = st.multiselect("Filter Vehicles", sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        
        st.divider()
        st.markdown("### 📍 Operational Legend")
        st.markdown("<span style='color:#ef4444'>●</span> **High Priority**")
        st.markdown("<span style='color:#f97316'>●</span> **Medium Priority**")
        st.markdown("<span style='color:#22c55e'>●</span> **Low Priority**")
        if show_unassigned:
            st.markdown("<span style='color:grey'>○</span> **Unassigned Risk**")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map:
        fig_map = go.Figure()
        
        # 1. Routes (The Perfect Lines)
        for v in v_filter:
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]]
            lons = [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
            fig_map.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines+markers", 
                line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), 
                marker=dict(size=9), name=f"Fleet_{v}"
            ))
        
        # 2. Unassigned Shipments (User Requested Addition)
        if show_unassigned:
            assigned_ids = set(routes["shipment_id"].unique())
            unassigned_df = ships[~ships["id"].isin(assigned_ids)]
            if not unassigned_df.empty:
                fig_map.add_trace(go.Scattermap(
                    lat=unassigned_df["latitude"], lon=unassigned_df["longitude"],
                    mode="markers", marker=dict(size=12, color="rgba(148, 163, 184, 0.5)", symbol="circle"),
                    name="Risk: Unassigned",
                    hovertext=unassigned_df["city"] + " (" + unassigned_df["weight"].astype(str) + "kg)"
                ))

        # 3. Distribution Depot
        fig_map.add_trace(go.Scattermap(
            lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", 
            text=["🏭 MUMBAI HUB"], textposition="top right",
            marker=dict(size=22, color="#0f172a", symbol="star"), 
            name="Depot Origin"
        ))

        fig_map.update_layout(
            map_style="open-street-map", 
            map=dict(center=dict(lat=20.5, lon=78.9), zoom=4.2), 
            height=700, margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: EXECUTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif module == "📊 executive Analytics":
    st.markdown("<div class='section-head'>Network ROI Performance</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Units Processed", f"{int(metrics['num_shipments'])}")
    c2.metric("📏 Distance Optimized", f"{(metrics['baseline_distance_km'] - metrics['opt_distance_km']):,.0f} km", "Net reduction")
    c3.metric("💰 Cost Optimization", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost']):,.0f}")
    c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']} pts")

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        # Corporate Waterfall
        fig_wf = go.Figure(go.Waterfall(
            orientation = "v", x = ["Fuel Opt", "Toll Opt", "Wage Opt", "TOTAL"],
            y = [metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'],
                 metrics['baseline_toll_cost']-metrics['opt_toll_cost'],
                 metrics['baseline_driver_cost']-metrics['opt_driver_cost'],
                 metrics['baseline_total_cost']-metrics['opt_total_cost']],
            connector = {"line":{"color":"#cbd5e1"}},
            decreasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#3b82f6"}}
        ))
        fig_wf.update_layout(title="Savings Decomposition (Waterfall)", template="plotly_white", height=450)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        # Corporate Bar
        fig_bar = px.bar(x=["Baseline", "AI Optimized"], 
                         y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]],
                         color=["#ef4444", "#10b981"], title="Carbon Emissions (kg CO₂)")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: LOGIC DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════
elif module == "🧠 Logic Diagnostics":
    st.markdown("<div class='section-head'>AI Multi-Objective Diagnostic Console</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.plotly_chart(px.pie(names=["Cost", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6, title="Static Weights"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        risk_df = pd.DataFrame({"City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Lucknow", "Raipur", "Bhopal"], "Risk": [0.98, 0.95, 0.92, 0.88, 0.85, 0.82, 0.78]})
        st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="Node-Level Re-Optimization Risk Monitor"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: COPILOT TERMINAL
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 Copilot Terminal":
    st.title("LoRRI AI Terminal")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if p := st.chat_input("Query CVRP math or SLA parameters..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        ans = get_rag_response(p)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='rag-terminal'>{ans}</div>", unsafe_allow_html=True)

st.divider()
st.caption("LoRRI · Enterprise AI Platform · Developed for Synapflow Hackathon")
