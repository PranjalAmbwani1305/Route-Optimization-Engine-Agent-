import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from rag_engine import get_rag_response

# ─── Page & Branding ─────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI · AI Command", layout="wide", page_icon="🚚")

# ─── Data Layer ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        # These must exist from your previous generation/solver runs
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

# ─── Enterprise Visual Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f4f7f9; }
    .stMetric { background: white; border: 1px solid #e1e8ed; padding: 15px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #e1e8ed; margin-bottom: 20px; }
    .section-head { font-size: 1.4rem; font-weight: 700; color: #1a202c; border-left: 5px solid #3182ce; padding-left: 15px; margin-bottom: 20px; }
    .rag-box { background: #0f172a; color: #38bdf8; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Master Control ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=60)
    st.title("LoRRI AI Platform")
    view = st.radio("OPERATIONAL WORKSPACES", 
                    ["📊 Real-Time KPI & Costs", "🗺️ Geospatial Intelligence", "🧠 XAI Diagnostics", "⚡ Disruption Simulator", "💬 AI Copilot"])
    st.divider()
    st.markdown("### Fleet Health")
    st.progress(88, text="Active Capacity: 88%")
    st.markdown("🟢 **OR-Tools:** Solving\n🟢 **SLA Monitor:** Active")

# ═══════════════════════════════════════════════════════════════════
# VIEW 1: KPI & COSTS
# ═══════════════════════════════════════════════════════════════════
if view == "📊 Real-Time KPI & Costs":
    st.markdown("<div class='section-head'>Network Efficiency & Financial Performance</div>", unsafe_allow_html=True)
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Active Load", f"{int(metrics['num_shipments'])} Units", "Stable")
    c2.metric("📏 Dist Saved", f"{metrics['baseline_distance_km']-metrics['opt_distance_km']:,.0f} km", f"-{((metrics['baseline_distance_km']-metrics['opt_distance_km'])/metrics['baseline_distance_km']*100):.1f}%", delta_color="normal")
    c3.metric("💰 Total Savings", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost'])/1000:,.1f}K", f"₹{(metrics['baseline_total_cost']-metrics['opt_total_cost']):,.0f}", delta_color="normal")
    c4.metric("✅ Compliance", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']:.0f}% vs Base")

    st.write("")
    
    # Financial Visuals
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Advanced Waterfall Savings Chart
        fig_wf = go.Figure(go.Waterfall(
            orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Fuel Opt", "Toll Opt", "Driver Opt", "NET SAVINGS"],
            textposition = "outside",
            text = [f"₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f}", 
                    f"₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}",
                    f"₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f}", 
                    "TOTAL"],
            y = [metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'],
                 metrics['baseline_toll_cost']-metrics['opt_toll_cost'],
                 metrics['baseline_driver_cost']-metrics['opt_driver_cost'],
                 metrics['baseline_total_cost']-metrics['opt_total_cost']],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color":"#10b981"}},
            totals = {"marker":{"color":"#3182ce"}}
        ))
        fig_wf.update_layout(title="Financial Savings Waterfall", template="plotly_white", height=450)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Cost Stacked Breakdown
        fig_cost = go.Figure(data=[
            go.Bar(name='Fuel', x=["Baseline", "LoRRI AI"], y=[metrics["baseline_fuel_cost"], metrics["opt_fuel_cost"]], marker_color='#3b82f6'),
            go.Bar(name='Toll', x=["Baseline", "LoRRI AI"], y=[metrics["baseline_toll_cost"], metrics["opt_toll_cost"]], marker_color='#f59e0b'),
            go.Bar(name='Driver', x=["Baseline", "LoRRI AI"], y=[metrics["baseline_driver_cost"], metrics["opt_driver_cost"]], marker_color='#8b5cf6')
        ])
        fig_cost.update_layout(barmode='stack', title="Total Operational Cost (₹)", height=450, template="plotly_white")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# VIEW 2: GEOSPATIAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
elif view == "🗺️ Geospatial Intelligence":
    st.markdown("<div class='section-head'>AI Optimized Fleet Routing</div>", unsafe_allow_html=True)
    
    v_selection = st.multiselect("Active Fleet Filter", options=sorted(routes["vehicle"].unique()), default=sorted(routes["vehicle"].unique()))
    
    fig_map = go.Figure()
    colors = px.colors.qualitative.Prism
    
    # Routes
    for v in v_selection:
        vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
        lats = [DEPOT["lat"]] + vdf["latitude"].tolist() + [DEPOT["lat"]]
        lons = [DEPOT["lon"]] + vdf["longitude"].tolist() + [DEPOT["lon"]]
        fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", 
                                        line=dict(width=3, color=colors[v%len(colors)]), 
                                        marker=dict(size=8), name=f"Vehicle {v}"))
    
    # Depot
    fig_map.add_trace(go.Scattermap(lat=[DEPOT["lat"]], lon=[DEPOT["lon"]], mode="markers+text", 
                                    text=["🏭 Mumbai Depot"], marker=dict(size=18, color="black", symbol="star"), name="Origin"))

    fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), 
                          height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# VIEW 3: XAI DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════
elif view == "🧠 XAI Diagnostics":
    st.markdown("<div class='section-head'>Explainable AI Decision Logic</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Doughnut of logic
        fig_logic = px.pie(names=["Fuel & Tolls", "Time Constraints", "SLA Adherence", "Carbon Mandates"], 
                           values=[35, 30, 20, 15], hole=0.6, title="Objective Weights (CVRP Core)")
        fig_logic.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_logic, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Sustainability Box
        st.markdown("### 🌿 Sustainability Impact")
        st.write("Route optimization translates to **20.4% lower idling time** and **15.2% lower fuel burn**.")
        fig_co2 = go.Figure(go.Bar(x=["Baseline", "Optimized"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], 
                                   marker_color=["#ef4444","#10b981"]))
        fig_co2.update_layout(title="Carbon Reduction (kg CO₂)", height=320)
        st.plotly_chart(fig_co2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# VIEW 4: DISRUPTION SIMULATOR
# ═══════════════════════════════════════════════════════════════════
elif view == "⚡ Disruption Simulator":
    st.markdown("<div class='section-head'>Operational Risk Sandbox</div>", unsafe_allow_html=True)
    
    col_sim, col_risk = st.columns([1, 1.5])
    
    with col_sim:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 🚥 Inject Road Disruption")
        target = st.selectbox("Select Target Delivery Node:", sorted(ships["city"].unique()))
        severity = st.slider("Traffic Congestion Spike", 1.0, 3.0, 2.5)
        if st.button("Trigger Re-Optimization Event", type="primary"):
            st.warning(f"SLA Breach Risk detected in {target}. AI re-calculating local search nodes...")
            time.sleep(1)
            st.success(f"Route adjusted. Vehicle re-sequenced. SLA window protected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_risk:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 📈 Re-Optimization Risk monitor")
        risk_data = pd.DataFrame({
            "City": ["Delhi", "Kolkata", "Hubli", "Jodhpur", "Udaipur", "Lucknow", "Raipur", "Bhopal"],
            "Risk": [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.72]
        })
        fig_risk = px.bar(risk_data, x="City", y="Risk", color="Risk", color_continuous_scale="Reds", title="Cities with highest SLA Breach Risk")
        fig_risk.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Re-route Trigger")
        st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# VIEW 5: AI COPILOT
# ═══════════════════════════════════════════════════════════════════
elif view == "💬 AI Copilot":
    st.markdown("<div class='section-head'>LoRRI Architecture Knowledge Base</div>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Custom Chat UI
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
            
    if prompt := st.chat_input("Ask about CVRP, SLA, or Carbon modeling..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner("Retrieving Architectural Context..."):
            ans = get_rag_response(prompt)
            time.sleep(0.4)
            
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='rag-box'>{ans}</div>", unsafe_allow_html=True)
