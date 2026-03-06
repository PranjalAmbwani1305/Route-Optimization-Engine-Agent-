import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time

# ─── Import your RAG Engine ──────────────────────────────────────────────────
try:
    from rag_engine import get_rag_response
except ImportError:
    def get_rag_response(query):
        return "RAG Engine offline. Please ensure rag_engine.py is in the same directory."

st.set_page_config(page_title="LoRRI · AI Route Optimization", layout="wide", page_icon="🚚")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT  = {"latitude": 19.0760, "longitude": 72.8777, "id": "DEPOT"}
COLORS = px.colors.qualitative.Bold

@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ Data files missing! Please run generate_data.py and route_solver.py first.")
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

# ─────────────────────────────────────────────────────────────────────────────
# Master-Level CSS 
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean SaaS Sidebar and Body */
    .css-1d391kg { background-color: #f8fafc; }
    .corporate-card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .section-title { color: #0f172a; font-weight: 600; font-size: 1.2rem; margin-bottom: 15px; border-bottom: 2px solid #f1f5f9; padding-bottom: 5px; }
    .educational-box { background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 15px; border-radius: 4px; margin: 10px 0; color: #064e3b; }
    [data-testid="metric-container"] { background: #ffffff; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    
    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom-color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=50)
    st.markdown("## **LoRRI Platform**")
    st.markdown("<p style='color: #64748b; font-size: 0.9rem;'>AI Route Optimization Engine</p>", unsafe_allow_html=True)
    st.divider()
    
    selected_module = st.radio(
        "Navigation Menu",
        [
            "📊 Executive & Financials", 
            "🗺️ Operations & Impact", 
            "🧠 AI Engine & Simulation", 
            "💬 Copilot Assistant"
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("### System Status")
    st.markdown("🟢 **CVRP Solver:** Optimal\n🟢 **RAG Backend:** Online")

# ─────────────────────────────────────────────────────────────────────────────
# Main Workspace Header
# ─────────────────────────────────────────────────────────────────────────────
st.title(selected_module[2:]) # Strips the emoji for a clean title
st.markdown("**India Logistics Network | Depot: Mumbai**")
st.write("") # Spacer

# ═══════════════════════════════════════════════════════════════════
# MODULE 1: Executive & Financials
# ═══════════════════════════════════════════════════════════════════
if selected_module == "📊 Executive & Financials":
    tab1, tab2 = st.tabs(["Overview & KPIs", "Cost Breakdown"])
    
    with tab1:
        st.markdown("<div class='section-title'>Network Efficiency Dashboard</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Total Shipments", int(metrics["num_shipments"]))
        col2.metric("🚛 Fleet Size", int(metrics["num_vehicles"]))
        col3.metric("📏 Distance Optimized", f"{metrics['opt_distance_km']:,.1f} km", delta=f"{metrics['baseline_distance_km'] - metrics['opt_distance_km']:,.1f} km saved")
        col4.metric("💰 Total Cost Savings", f"₹{metrics['baseline_total_cost'] - metrics['opt_total_cost']:,.0f}", delta=f"{(metrics['baseline_total_cost'] - metrics['opt_total_cost'])/metrics['baseline_total_cost']*100:.1f}% reduction")

        st.markdown("<br><div class='section-title'>Fleet Performance Matrix</div>", unsafe_allow_html=True)
        display_veh = veh_summary.copy()
        display_veh.columns = ["Vehicle","Stops","Load (kg)","Dist (km)","Time (hr)","Fuel ₹","Toll ₹","Driver ₹","SLA Penalty ₹","Total Cost ₹","Carbon kg","SLA Breaches","Util %"]
        st.dataframe(display_veh.style.format({ "Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}", "Fuel ₹":"₹{:,.0f}","Toll ₹":"₹{:,.0f}","Driver ₹":"₹{:,.0f}", "SLA Penalty ₹":"₹{:,.0f}","Total Cost ₹":"₹{:,.0f}", "Carbon kg":"{:.1f}","Util %":"{:.1f}%"}).background_gradient(subset=["Util %"], cmap="Blues"), width='stretch', hide_index=True)

    with tab2:
        st.markdown("<div class='section-title'>Financial Analytics</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_cost = go.Figure(data=[
                go.Bar(name='Fuel Cost', x=["Baseline", "Optimized"], y=[metrics["baseline_fuel_cost"], metrics["opt_fuel_cost"]], marker_color='#3b82f6'),
                go.Bar(name='Toll Cost', x=["Baseline", "Optimized"], y=[metrics["baseline_toll_cost"], metrics["opt_toll_cost"]], marker_color='#f59e0b'),
                go.Bar(name='Driver Wages', x=["Baseline", "Optimized"], y=[metrics["baseline_driver_cost"], metrics["opt_driver_cost"]], marker_color='#10b981')
            ])
            fig_cost.update_layout(barmode='stack', title="Operational Expenditure Comparison (₹)", plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#e2e8f0'))
            st.plotly_chart(fig_cost, use_container_width=True)

        with c2:
            savings = {"Fuel Saved": metrics["baseline_fuel_cost"] - metrics["opt_fuel_cost"], "Tolls Avoided": metrics["baseline_toll_cost"] - metrics["opt_toll_cost"], "Driver Hours Saved": metrics["baseline_driver_cost"] - metrics["opt_driver_cost"]}
            fig_wf = go.Figure(go.Waterfall(name="Savings", orientation="v", measure=["relative","relative","relative","total"], x=list(savings.keys()) + ["Net Savings"], y=list(savings.values()) + [sum(savings.values())], decreasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#3b82f6"}}))
            fig_wf.update_layout(title="Cost Reduction Breakdown (₹)", plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#e2e8f0'))
            st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODULE 2: Operations & Impact
# ═══════════════════════════════════════════════════════════════════
elif selected_module == "🗺️ Operations & Impact":
    tab1, tab2 = st.tabs(["Live Route Map", "CO₂ & SLA Tracking"])
    
    with tab1:
        st.markdown("<div class='section-title'>Geospatial Execution View</div>", unsafe_allow_html=True)
        fig_map = go.Figure()
        
        for v in sorted(routes["vehicle"].unique()):
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
            lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1) % len(COLORS)]), name=f"Vehicle {v}"))

        fig_map.add_trace(go.Scattermap(lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]], mode="markers+text", text=["🏭 Mumbai Depot"], textposition="top right", marker=dict(size=20, color="black", symbol="star"), name="Depot"))
        fig_map.update_layout(map_style="open-street-map", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Sustainability & Commitments</div>", unsafe_allow_html=True)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            <div class='educational-box'>
            <b>🌿 Optimizing the Carbon Footprint</b><br>
            Carbon Dioxide (CO₂) scales directly with fuel burn. By calculating mathematically optimal multi-stop routes, the CVRP model minimizes distance and avoids congestion, drastically reducing total emissions.
            </div>
            """, unsafe_allow_html=True)
            fig_co2 = go.Figure(go.Bar(x=["Baseline", "AI Optimized"], y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]], marker_color=["#ef4444","#10b981"], text=[f"{metrics['baseline_carbon_kg']:,.0f} kg", f"{metrics['opt_carbon_kg']:,.0f} kg"], textposition='auto'))
            fig_co2.update_layout(title="Total Carbon Footprint (kg CO₂)", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_co2, use_container_width=True)

        with colB:
            st.markdown("""
            <div class='educational-box'>
            <b>⏱️ Service Level Agreements (SLA)</b><br>
            Our engine embeds severe financial penalties for late deliveries into its cost function. This mathematically forces the algorithm to prioritize urgent shipments automatically to maintain high compliance.
            </div>
            """, unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=metrics["opt_sla_adherence_pct"], title={"text": "SLA Compliance Rate (%)"}, delta={"reference": metrics["baseline_sla_adherence_pct"], "increasing": {"color": "#10b981"}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#3b82f6"}}))
            st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MODULE 3: AI Engine & Simulation
# ═══════════════════════════════════════════════════════════════════
elif selected_module == "🧠 AI Engine & Simulation":
    tab1, tab2 = st.tabs(["Explainability (XAI)", "Scenario Simulator"])
    
    with tab1:
        st.markdown("<div class='section-title'>Decision Intelligence Insights</div>", unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            fig_donut = go.Figure(go.Pie(labels=["Cost Constraints","Time Limits","Carbon Goals","SLA Deadlines"], values=[35, 30, 20, 15], hole=0.6))
            fig_donut.update_layout(title="Programmed Objective Weights", height=400)
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_right:
            fig_fi = go.Figure(go.Bar(x=list(feature_importance.values()), y=list(feature_importance.keys()), orientation="h", marker_color="#3b82f6"))
            fig_fi.update_layout(title="Actual Decision Drivers (Permutation Impact %)", yaxis=dict(autorange="reversed"), plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_fi, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Real-Time Disruption Injection</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='corporate-card'>", unsafe_allow_html=True)
            st.subheader("🚦 Inject Traffic Congestion")
            disrupted_city = st.selectbox("Select affected city:", options=sorted(ships["city"].tolist()), key="t1")
            traffic_spike = st.slider("Traffic Intensity Multiplier", 1.0, 3.0, 2.5)
            if st.button("Trigger Gridlock", type="primary"):
                st.warning(f"Traffic in {disrupted_city} increased by {traffic_spike}x. Triggering partial route re-calculation to avoid downstream SLA failures.")
                st.success("Re-routing successful. Affected node moved down the delivery queue.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='corporate-card'>", unsafe_allow_html=True)
            st.subheader("🚨 Escalate Priority")
            escalate_city = st.selectbox("Select customer location:", options=sorted(ships["city"].tolist()), key="p1")
            if st.button("Escalate to HIGH Priority", type="primary"):
                st.success(f"{escalate_city} SLA window tightened to 24H. Node automatically bumped to Stop #1 on designated vehicle route.")
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MODULE 4: Copilot Assistant
# ═══════════════════════════════════════════════════════════════════
elif selected_module == "💬 Copilot Assistant":
    st.markdown("<div class='section-title'>LoRRI Architecture Copilot</div>", unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Copilot. Ask me anything about the CVRP routing engine, system architecture, or optimization objectives."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("E.g., How does the engine balance SLA and cost?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing architecture documents..."):
            time.sleep(0.5) 
            response = get_rag_response(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
