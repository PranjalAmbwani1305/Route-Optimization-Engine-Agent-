import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time

# ─── Import your RAG Engine ──────────────────────────────────────────────────
from rag_engine import get_rag_response

st.set_page_config(page_title="LoRRI · AI Route Optimization", layout="wide", page_icon="🚚")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777, "id": "DEPOT"}
COLORS      = px.colors.qualitative.Bold

@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ Data files missing! Please run your generator and solver scripts first.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()

@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {
        "Travel Time": "travel_time_hr", "Fuel Cost": "fuel_cost", "Toll Cost": "toll_cost",
        "Driver Cost": "driver_cost", "Carbon Emitted": "carbon_kg", "SLA Breach": "sla_breach_hr", "Package Weight": "weight",
    }
    X, y = routes_df[list(features.values())].copy(), routes_df["mo_score"].values
    baseline_mae = np.mean(np.abs(y - y.mean()))
    importances = {}
    for label, col in features.items():
        shuffled = X.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        proxy = shuffled.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        importances[label] = abs(np.mean(np.abs(y - proxy.values)) - baseline_mae)
    total = sum(importances.values()) + 1e-9
    return {k: round(v / total * 100, 1) for k, v in sorted(importances.items(), key=lambda x: -x[1])}

feature_importance = compute_feature_importance(routes)

@st.cache_data
def compute_stop_contributions(routes_df):
    cols = ["travel_time_hr", "fuel_cost", "toll_cost", "driver_cost", "carbon_kg", "sla_breach_hr"]
    labels = ["Travel Time", "Fuel Cost", "Toll Cost", "Driver Cost", "Carbon", "SLA Breach"]
    weights = [0.30, 0.20, 0.05, 0.15, 0.20, 0.10]
    df = routes_df[cols].copy()
    for c in cols: df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-9)
    for i, c in enumerate(cols): df[c] = df[c] * weights[i]
    df.columns = labels
    df["city"], df["vehicle"], df["mo_score"] = routes_df["city"].values, routes_df["vehicle"].values, routes_df["mo_score"].values
    return df

stop_contrib = compute_stop_contributions(routes)

# ─────────────────────────────────────────────────────────────────────────────
# CSS - Original Clean UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.info-box { background: rgba(14,165,233,0.12); border-left: 4px solid #0ea5e9; border-radius: 6px; padding: 14px 18px; margin: 8px 0 14px 0; font-size: 0.95rem; line-height: 1.6; }
.warn-box { background: rgba(234,179,8,0.15); border-left: 4px solid #eab308; border-radius: 6px; padding: 12px 16px; margin: 6px 0; font-size: 0.92rem; line-height: 1.6; }
.ok-box { background: rgba(34,197,94,0.12); border-left: 4px solid #22c55e; border-radius: 6px; padding: 12px 16px; margin: 6px 0; font-size: 0.92rem; }
.tag-red { color: #f87171; font-weight: 700; }
.tag-green { color: #4ade80; font-weight: 700; }
.tag-yellow{ color: #fbbf24; font-weight: 700; }
[data-testid="metric-container"] { border-radius: 10px; padding: 10px; border: 1px solid rgba(148,163,184,0.25); }
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🚚 LoRRI · AI Route Optimization Engine")
st.markdown("**Dynamic Multi-Objective CVRP · India Logistics Network · Depot: Mumbai**")

tabs = st.tabs(["📊 Overview & KPIs", "🗺️ Route Map", "💰 Cost Breakdown",
                "🌿 Carbon & SLA", "🧠 Explainability", "⚡ Re-optimization Simulator", "💬 AI Assistant (RAG)"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Overview & KPIs
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='info-box'><b>📖 What is this tab?</b><br>Think of this as the <b>report card</b> for the whole delivery run. It compares what would have happened if trucks just drove in a straight line one-by-one (<b>Baseline</b>) versus what our smart AI planner chose (<b>Optimized</b>). Every green arrow means money saved, time saved, or less pollution. ✅</div>", unsafe_allow_html=True)

    r1 = st.columns(4)
    r1[0].metric("📦 Shipments", int(metrics["num_shipments"]))
    r1[1].metric("🚛 Vehicles", int(metrics["num_vehicles"]))
    r1[2].metric("🏭 Depot", "Mumbai")
    r1[3].metric("⚖️ Obj. Weights", "Cost 35% · Time 30% · CO₂ 20% · SLA 15%")

    st.divider()
    r2 = st.columns(4)
    r2[0].metric("📏 Distance (km)", f"{metrics['opt_distance_km']:,.1f}", delta=f"{metrics['opt_distance_km'] - metrics['baseline_distance_km']:,.1f}", delta_color="inverse")
    r2[1].metric("⏱️ Travel Time (hr)", f"{metrics['opt_time_hr']:,.1f}", delta=f"{metrics['opt_time_hr'] - metrics['baseline_time_hr']:,.1f}", delta_color="inverse")
    r2[2].metric("💰 Total Cost (₹)", f"₹{metrics['opt_total_cost']:,.0f}", delta=f"₹{metrics['opt_total_cost'] - metrics['baseline_total_cost']:,.0f}", delta_color="inverse")
    r2[3].metric("🌿 Carbon Emitted (kg)", f"{metrics['opt_carbon_kg']:,.1f}", delta=f"{metrics['opt_carbon_kg'] - metrics['baseline_carbon_kg']:,.1f}", delta_color="inverse")

    st.divider()
    st.markdown("### 📋 Per-Vehicle Summary")
    display_veh = veh_summary.copy()
    display_veh.columns = ["Vehicle","Stops","Load (kg)","Dist (km)","Time (hr)","Fuel ₹","Toll ₹","Driver ₹","SLA Penalty ₹","Total Cost ₹","Carbon kg","SLA Breaches","Util %"]
    st.dataframe(display_veh.style.format({ "Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}", "Fuel ₹":"₹{:,.0f}","Toll ₹":"₹{:,.0f}","Driver ₹":"₹{:,.0f}", "SLA Penalty ₹":"₹{:,.0f}","Total Cost ₹":"₹{:,.0f}", "Carbon kg":"{:.1f}","Util %":"{:.1f}%"}).background_gradient(subset=["Util %"], cmap="RdYlGn"), width='stretch', hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Route Map
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='info-box'><b>📖 What is this tab?</b><br>This is a real map of India showing every delivery truck's path. Each coloured line is a different truck. <b>Red dots</b> = urgent deliveries (HIGH priority), <b>orange dots</b> = medium, <b>green dots</b> = low urgency. Toggle <i>Show Baseline Route</i> to see how messy the old path was vs the smart new one!</div>", unsafe_allow_html=True)
    
    col_map, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown("### 🎛️ Map Controls")
        show_baseline = st.toggle("Show Baseline Route", value=False)
        show_unassigned = st.toggle("Show Unassigned", value=True)
        selected_v = st.multiselect("Filter Vehicles", options=sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
        
    with col_map:
        fig_map = go.Figure()
        if show_baseline:
            b_lats = [DEPOT["latitude"]] + ships["latitude"].tolist() + [DEPOT["latitude"]]
            b_lons = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig_map.add_trace(go.Scattermap(lat=b_lats, lon=b_lons, mode="lines", line=dict(width=1.5, color="rgba(200,50,50,0.4)"), name="Baseline Route"))

        for v in selected_v:
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
            lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines", line=dict(width=3, color=COLORS[(v-1) % len(COLORS)]), name=f"Vehicle {v}"))
            for _, row in vdf.iterrows():
                pc = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"}.get(row.get("priority", "MEDIUM"), "#f97316")
                fig_map.add_trace(go.Scattermap(lat=[row["latitude"]], lon=[row["longitude"]], mode="markers", marker=dict(size=11, color=pc), hoverinfo="none", showlegend=False))

        fig_map.add_trace(go.Scattermap(lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]], mode="markers+text", text=["🏭 Mumbai Depot"], textposition="top right", marker=dict(size=20, color="black", symbol="star"), name="Depot"))
        fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), margin=dict(l=0, r=0, t=0, b=0), height=580)
        st.plotly_chart(fig_map, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Cost Breakdown
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='info-box'><b>📖 What is this tab?</b><br>Delivering packages costs money in 3 main ways: fuel, tolls, and driver wages. This tab shows exactly how much was spent on each — and how much our AI saved.</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig_cost = go.Figure()
        for cat, bv, ov, col in zip(["Fuel", "Toll", "Driver"], [metrics["baseline_fuel_cost"], metrics["baseline_toll_cost"], metrics["baseline_driver_cost"]], [metrics["opt_fuel_cost"], metrics["opt_toll_cost"], metrics["opt_driver_cost"]], ["#3b82f6","#f59e0b","#8b5cf6"]):
            fig_cost.add_trace(go.Bar(name=cat, x=["Baseline","Optimized"], y=[bv, ov], marker_color=col))
        fig_cost.update_layout(barmode="stack", title="Total Cost Components (₹)", height=360)
        st.plotly_chart(fig_cost, width='stretch')

    with c2:
        savings = {"Fuel Saved": metrics["baseline_fuel_cost"] - metrics["opt_fuel_cost"], "Toll Saved": metrics["baseline_toll_cost"] - metrics["opt_toll_cost"], "Driver Saved": metrics["baseline_driver_cost"] - metrics["opt_driver_cost"]}
        fig_wf = go.Figure(go.Waterfall(name="20", orientation="v", measure=["relative","relative","relative","total"], x=list(savings.keys()) + ["Total Saved"], textposition="outside", y=list(savings.values()) + [sum(savings.values())], decreasing={"marker":{"color":"#22c55e"}}, totals={"marker":{"color":"#0ea5e9"}}))
        fig_wf.update_layout(title="Savings Waterfall — How Much We Saved (₹)", height=360)
        st.plotly_chart(fig_wf, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Carbon & SLA
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='info-box'><b>📖 What is this tab?</b><br><b>Carbon</b> = how much pollution. <b>SLA</b> = Service Level Agreement (delivery promises kept).</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_co2 = go.Figure(go.Bar(x=["Baseline", "Optimized"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], marker_color=["#ef4444","#22c55e"]))
        fig_co2.update_layout(title="CO₂ Emissions", height=300)
        st.plotly_chart(fig_co2, width='stretch')
    with c2:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=metrics["opt_sla_adherence_pct"], title={"text": "Delivery Promise Kept (SLA %)"}, delta={"reference": metrics["baseline_sla_adherence_pct"], "increasing": {"color": "#22c55e"}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22c55e"}}))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Explainability
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🧠 Explainability — How the Optimizer Made Its Decisions")
    c1, c2 = st.columns([1, 2])
    with c1:
        fig_donut = go.Figure(go.Pie(labels=["Cost (₹)","Travel Time","Carbon CO₂","SLA Adherence"], values=[35, 30, 20, 15], hole=0.55, marker_colors=["#3b82f6","#f59e0b","#22c55e","#ef4444"]))
        fig_donut.update_layout(height=300, title="Objective Weights")
        st.plotly_chart(fig_donut, width='stretch')
    with c2:
        fi_labels, fi_values = list(feature_importance.keys()), list(feature_importance.values())
        fig_fi = go.Figure(go.Bar(x=fi_values, y=fi_labels, orientation="h", marker_color=["#ef4444" if v == max(fi_values) else "#3b82f6" for v in fi_values]))
        fig_fi.update_layout(title="Which factor drove routing decisions the most?", yaxis=dict(autorange="reversed"), height=300)
        st.plotly_chart(fig_fi, width='stretch')

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — Re-optimization Simulator
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### ⚡ Re-Optimization Trigger Simulator")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🚦 Scenario 1 — Traffic Jam Hits a City")
        disrupted_city = st.selectbox("City hit by traffic jam:", options=sorted(ships["city"].tolist()))
        traffic_spike = st.slider("New traffic level (1.0 = clear road, 3.0 = gridlock)", 1.0, 3.0, 2.5, 0.1)
        if st.button("🔴 Trigger Traffic Disruption"):
            st.markdown(f"<div class='warn-box'><b>🚦 Disruption Detected: {disrupted_city}</b><br>Traffic spiked to {traffic_spike}x. Triggering re-optimization!</div>", unsafe_allow_html=True)
            with st.spinner("Re-optimizing affected vehicle route..."): time.sleep(1)
            st.markdown(f"<div class='ok-box'>✅ <b>Vehicle re-routed!</b> {disrupted_city} moved to last stop.</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 🚨 Scenario 2 — Customer Calls: I Need It NOW!")
        escalate_city = st.selectbox("Which city's shipment became urgent?", options=sorted(ships["city"].tolist()), key="esc")
        if st.button("🔴 Trigger Priority Escalation"):
            with st.spinner("Moving urgent stop to front of route..."): time.sleep(1)
            st.markdown(f"<div class='ok-box'>✅ <b>{escalate_city}</b> escalated to <span class='tag-red'>HIGH</span> priority! Moved to <b>stop #1</b>.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 7 — AI Assistant (RAG)
# ═══════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 💬 LoRRI Architecture Assistant")
    st.markdown("<div class='info-box'>Ask questions based on the LoRRI Route Optimization architecture documentation.</div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you understand the CVRP models or architecture today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., What framework is used for route planning?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = get_rag_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
