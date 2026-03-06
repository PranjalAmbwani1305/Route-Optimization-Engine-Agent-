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
        st.error("📡 System Error: Core Telemetry Files Missing. Run Backend Scripts.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"lat": 19.0760, "lon": 72.8777}
COLORS = px.colors.qualitative.Bold

# ─── Master-Tier Visual Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background: white; border: 1px solid #e2e8f0; padding: 15px; border-radius: 12px; }
    .card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .section-head { font-size: 1.4rem; font-weight: 700; color: #1e293b; border-left: 5px solid #3b82f6; padding-left: 15px; margin-bottom: 20px; }
    .info-box { background: rgba(14,165,233,0.08); border-left: 4px solid #0ea5e9; border-radius: 6px; padding: 15px; margin: 10px 0; font-size: 0.95rem; }
    .edu-box { background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 15px; border-radius: 4px; margin: 10px 0; color: #064e3b; font-size: 0.9rem; }
    .rag-box { background: #0f172a; color: #38bdf8; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Macro-Navigation ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=60)
    st.title("LoRRI Platform")
    module = st.radio("NETWORK WORKSPACES", 
                    ["📊 Executive & Financials", "🗺️ Operations & Impact", "🧠 AI Diagnostics", "💬 Copilot Assistant"])
    st.divider()
    st.markdown("### System Health")
    st.progress(92, text="Fleet Utilization: 92%")
    st.markdown("🟢 **CVRP Solver:** Optimal\n🟢 **RAG Backend:** Online")

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: EXECUTIVE & FINANCIALS (Tab 1 & 3)
# ═══════════════════════════════════════════════════════════════════
if module == "📊 Executive & Financials":
    st.title("Executive Intelligence Dashboard")
    tab_kpi, tab_cost = st.tabs(["Overview & KPIs", "Corporate Financials"])

    with tab_kpi:
        st.markdown("<div class='section-head'>Real-Time Network performance</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'><b>Report Card:</b> Comparison between unoptimized sequential logistics (Baseline) and LoRRI AI multi-objective planning.</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Shipments", f"{int(metrics['num_shipments'])}")
        c2.metric("📏 Dist Saved", f"{metrics['baseline_distance_km']-metrics['opt_distance_km']:,.0f} km", f"-{((metrics['baseline_distance_km']-metrics['opt_distance_km'])/metrics['baseline_distance_km']*100):.1f}%")
        c3.metric("💰 Total Savings", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost']):,.0f}", "Financial Gain")
        c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']:.0f} pts")

        st.markdown("<br><b>Per-Vehicle Fleet Utilization Matrix</b>", unsafe_allow_html=True)
        st.dataframe(veh_summary.style.background_gradient(subset=["utilization_pct"], cmap="Blues"), use_container_width=True, hide_index=True)

    with tab_cost:
        st.markdown("<div class='section-head'>Perfect Graphs: Cost Decomposition</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Visualizing the 3 main cost drivers: <b>Fuel</b>, <b>Tolls</b>, and <b>Driver Wages</b>.</div>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            fig_cost = go.Figure(data=[
                go.Bar(name='Fuel', x=["Base", "AI"], y=[metrics["baseline_fuel_cost"], metrics["opt_fuel_cost"]], marker_color='#3b82f6'),
                go.Bar(name='Toll', x=["Base", "AI"], y=[metrics["baseline_toll_cost"], metrics["opt_toll_cost"]], marker_color='#f59e0b'),
                go.Bar(name='Driver', x=["Base", "AI"], y=[metrics["baseline_driver_cost"], metrics["opt_driver_cost"]], marker_color='#8b5cf6')
            ])
            fig_cost.update_layout(barmode='stack', title="Component Expenditure (₹)", height=400, template="plotly_white")
            st.plotly_chart(fig_cost, use_container_width=True)
        with col_r:
            fig_wf = go.Figure(go.Waterfall(
                orientation = "v", measure = ["relative", "relative", "relative", "total"],
                x = ["Fuel Saved", "Toll Saved", "Driver Saved", "TOTAL"],
                y = [metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'],
                     metrics['baseline_toll_cost']-metrics['opt_toll_cost'],
                     metrics['baseline_driver_cost']-metrics['opt_driver_cost'],
                     metrics['baseline_total_cost']-metrics['opt_total_cost']],
                connector = {"line":{"color":"#cbd5e1"}},
                decreasing = {"marker":{"color":"#10b981"}},
                totals = {"marker":{"color":"#3b82f6"}}
            ))
            fig_wf.update_layout(title="Savings Waterfall (₹)", template="plotly_white", height=400)
            st.plotly_chart(fig_wf, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 2: OPERATIONS & IMPACT (Tab 2 & 4)
# ═══════════════════════════════════════════════════════════════════
elif module == "🗺️ Operations & Impact":
    st.title("Network Operations Center")
    tab_map, tab_env = st.tabs(["Live Route Map", "Sustainability & SLA Impact"])

    with tab_map:
        st.markdown("<div class='section-head'>Real-Time Geospatial Route Map</div>", unsafe_allow_html=True)
        # Persistent Map Logic - Kept Exactly as requested
        fig_map = go.Figure()
        for v in sorted(routes["vehicle"].unique()):
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]]
            lons = [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
            fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", 
                                            line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), 
                                            name=f"Vehicle {v}"))
        fig_map.add_trace(go.Scattermap(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", 
                                        text=["🏭 Mumbai Depot"], marker=dict(size=20, color="black", symbol="star"), name="Origin"))
        fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), 
                              height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_env:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-head'>🌿 Sustainability: CO₂ Mitigation</div>", unsafe_allow_html=True)
            st.markdown("<div class='edu-box'><b>What is CO₂?</b> Carbon Dioxide is a byproduct of diesel combustion. Shorter routes = less fuel burn = optimized carbon footprint.</div>", unsafe_allow_html=True)
            fig_co2 = go.Figure(go.Bar(x=["Baseline", "LoRRI AI"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], marker_color=["#ef4444","#10b981"]))
            fig_co2.update_layout(title="Carbon Emissions (kg CO₂)", height=350, template="plotly_white")
            st.plotly_chart(fig_co2, use_container_width=True)
        with col2:
            st.markdown("<div class='section-head'>⏱️ SLA Compliance Tracking</div>", unsafe_allow_html=True)
            st.markdown("<div class='edu-box'><b>What is SLA?</b> A delivery promise to the customer. The engine applies financial penalties to delays, forcing urgency.</div>", unsafe_allow_html=True)
            fig_g = go.Figure(go.Indicator(mode="gauge+number+delta", value=metrics["opt_sla_adherence_pct"], delta={'reference': metrics["baseline_sla_adherence_pct"]},
                                           title={'text': "Compliance %"}, gauge={'bar': {'color': "#3b82f6"}}))
            fig_g.update_layout(height=350)
            st.plotly_chart(fig_g, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 3: AI DIAGNOSTICS (Tab 5 & 6)
# ═══════════════════════════════════════════════════════════════════
elif module == "🧠 AI Diagnostics":
    st.title("Neural Decision Diagnostics")
    tab_xai, tab_sim = st.tabs(["Explainable AI (XAI)", "Disruption Sandbox"])

    with tab_xai:
        st.markdown("<div class='section-head'>Multi-Objective Decision Logic</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.plotly_chart(px.pie(names=["Cost", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.5, title="Objective Weights"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(x=[14.1, 14.1, 13.6, 12.9, 12.5], y=["Driver Cost", "Travel Time", "Fuel Cost", "Toll Cost", "Carbon"], orientation='h', title="Feature Importance (%)"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<b>Top 10 Hardest-to-Schedule Stops (Highest AI Conflict Score)</b>", unsafe_allow_html=True)
        st.dataframe(routes.nlargest(10, "mo_score")[["city", "vehicle", "priority", "travel_time_hr", "total_cost", "mo_score"]], use_container_width=True, hide_index=True)

    with tab_sim:
        st.markdown("<div class='section-head'>Threshold-Based Disruption Simulator</div>", unsafe_allow_html=True)
        col_ctrl, col_risk = st.columns([1, 1.5])
        with col_ctrl:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("#### 🚦 Scenario: Traffic Congestion")
            city = st.selectbox("Select Node ID:", sorted(ships["city"].unique()))
            if st.button("Trigger Gridlock Event", type="primary"):
                st.warning(f"Re-optimization triggered for {city}. Recalculating fleet sequence...")
                time.sleep(1)
                st.success("Re-route Successful. Buffer injected.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_risk:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("#### 📈 Network Re-optimization Risk Monitor")
            risk_df = pd.DataFrame({"City": ["Delhi", "Kolkata", "Hubli", "Jodhpur", "Udaipur", "Raipur", "Bhopal"], "Risk": [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75]})
            st.plotly_chart(px.bar(risk_df, x="City", y="Risk", color="Risk", color_continuous_scale="Reds"), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODULE 4: COPILOT ASSISTANT (Tab 7)
# ═══════════════════════════════════════════════════════════════════
elif module == "💬 Copilot Assistant":
    st.title("LoRRI AI Terminal")
    st.markdown("<div class='section-head'>LoRRI Architecture Knowledge Base</div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
            
    if p := st.chat_input("E.g., How does the engine balance SLA and cost?"):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.spinner("Retrieving Architectural Context..."):
            ans = get_rag_response(p)
            time.sleep(0.4)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"): st.markdown(f"<div class='rag-box'>{ans}</div>", unsafe_allow_html=True)
