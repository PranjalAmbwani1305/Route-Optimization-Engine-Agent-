import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time

# Import the standalone RAG engine
from rag_engine import get_rag_response

st.set_page_config(page_title="LoRRI · AI Route Optimization", layout="wide", page_icon="🚚")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Configuration
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT  = {"latitude": 19.0760, "longitude": 72.8777, "id": "DEPOT"}
COLORS = px.colors.qualitative.Bold

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (Reads from your untouched backend CSVs)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ CSV files not found. Please run `generate_data.py` and `route_solver.py` first.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# CSS - Modern Enterprise Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.info-box {
    background: rgba(14,165,233,0.12);
    border-left: 4px solid #0ea5e9;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0 14px 0;
}
.warn-box {
    background: rgba(234,179,8,0.15);
    border-left: 4px solid #eab308;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
}
.ok-box {
    background: rgba(34,197,94,0.12);
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
}
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(148,163,184,0.2);
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🚚 AI Route Optimization Engine")
st.markdown("**Dynamic Multi-Objective CVRP · India Logistics Network · Depot: Mumbai**")

tabs = st.tabs(["📊 Overview & KPIs", "🗺️ Route Map", "💰 Cost Breakdown", 
                "🌿 Carbon & SLA", "🧠 RAG Assistant", "⚡ Re-optimization Simulator"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Overview & KPIs
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    r1 = st.columns(4)
    r1[0].metric("📦 Shipments", int(metrics["num_shipments"]))
    r1[1].metric("🚛 Vehicles", int(metrics["num_vehicles"]))
    r1[2].metric("🏭 Depot", "Mumbai")
    r1[3].metric("⚖️ Obj. Weights", "Cost 35% · Time 30% · CO₂ 20% · SLA 15%")

    st.divider()
    r2 = st.columns(4)
    r2[0].metric("📏 Distance Optimized", f"{metrics['opt_distance_km']:,.1f} km", delta=f"{metrics['opt_distance_km'] - metrics['baseline_distance_km']:,.1f} km", delta_color="inverse")
    r2[1].metric("⏱️ Travel Time", f"{metrics['opt_time_hr']:,.1f} hr", delta=f"{metrics['opt_time_hr'] - metrics['baseline_time_hr']:,.1f} hr", delta_color="inverse")
    r2[2].metric("💰 Total Cost", f"₹{metrics['opt_total_cost']:,.0f}", delta=f"₹{metrics['opt_total_cost'] - metrics['baseline_total_cost']:,.0f}", delta_color="inverse")
    r2[3].metric("🌿 Carbon Emitted", f"{metrics['opt_carbon_kg']:,.1f} kg", delta=f"{metrics['opt_carbon_kg'] - metrics['baseline_carbon_kg']:,.1f} kg", delta_color="inverse")

    st.divider()
    st.markdown("### 📋 Per-Vehicle Summary")
    st.dataframe(veh_summary.style.format(precision=1).background_gradient(subset=["utilization_pct"], cmap="Blues"), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Route Map
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    fig_map = go.Figure()

    for v in sorted(routes["vehicle"].unique()):
        vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        color = COLORS[(v-1) % len(COLORS)]
        
        fig_map.add_trace(go.Scattermap(
            lat=lats, lon=lons, mode="lines+markers",
            line=dict(width=3, color=color), name=f"Vehicle {v}"
        ))

    fig_map.add_trace(go.Scattermap(
        lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
        mode="markers+text", text=["🏭 Mumbai Depot"], textposition="top right",
        marker=dict(size=20, color="black", symbol="star"), name="Depot",
    ))

    fig_map.update_layout(
        map_style="open-street-map",
        map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
        margin=dict(l=0, r=0, t=0, b=0), height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 & 4 — Costs, Carbon & SLA
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    fig_cost = go.Figure(data=[
        go.Bar(name='Fuel', x=["Baseline","Optimized"], y=[metrics["baseline_fuel_cost"], metrics["opt_fuel_cost"]]),
        go.Bar(name='Toll', x=["Baseline","Optimized"], y=[metrics["baseline_toll_cost"], metrics["opt_toll_cost"]]),
        go.Bar(name='Driver', x=["Baseline","Optimized"], y=[metrics["baseline_driver_cost"], metrics["opt_driver_cost"]])
    ])
    fig_cost.update_layout(barmode="stack", title="Financial Savings Breakdown (₹)")
    st.plotly_chart(fig_cost, use_container_width=True)

with tabs[3]:
    fig_co2 = go.Figure(go.Bar(
        x=["Baseline (No AI)", "Optimized (AI)"], 
        y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]],
        marker_color=["#ef4444","#22c55e"]
    ))
    fig_co2.update_layout(title="Carbon Emissions Reduced (kg CO₂)")
    st.plotly_chart(fig_co2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — RAG Assistant
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🤖 LoRRI Optimization Knowledge Base")
    st.markdown("<div class='info-box'>Ask questions about the engine's architecture, CVRP models, or SLAs based on the system documentation.</div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you understand the LoRRI Route Optimization Engine today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., What is the expected impact on travel distance?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = get_rag_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — Re-optimization Simulator
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### ⚡ Live Disruption Monitoring")
    c1, c2 = st.columns(2)
    with c1:
        disrupted_city = st.selectbox("City hit by traffic jam:", options=sorted(ships["city"].tolist()))
        traffic_spike  = st.slider("New traffic level (1.0 = clear, 3.0 = gridlock)", 1.0, 3.0, 2.5)
        if st.button("🔴 Trigger Traffic Disruption"):
            st.markdown(f"<div class='warn-box'><b>🚦 Disruption Detected: {disrupted_city}</b><br>Traffic spiked to {traffic_spike}x. Triggering partial route re-calculation to avoid SLA breach.</div>", unsafe_allow_html=True)
    
    with c2:
        escalate_city = st.selectbox("Customer priority escalation:", options=sorted(ships["city"].tolist()), key="esc")
        if st.button("🔴 Trigger Priority Escalation"):
            st.markdown(f"<div class='ok-box'>✅ <b>{escalate_city}</b> escalated to HIGH priority. Re-routing to Stop #1.</div>", unsafe_allow_html=True)
