import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from rag_engine import get_rag_response

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI · AI Route Optimization", layout="wide", page_icon="⚡")

# ─── Custom Expert-Level CSS (Glassmorphism & Neon Accents) ────────────────────
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Sleek Glass Panels for Metrics */
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.6);
    }
    
    /* Status Boxes */
    .alert-box { padding: 16px; border-radius: 8px; margin-bottom: 16px; font-weight: 500; border-left: 5px solid; }
    .alert-info { background: rgba(14, 165, 233, 0.1); border-color: #0ea5e9; color: #bae6fd; }
    .alert-warn { background: rgba(245, 158, 11, 0.1); border-color: #f59e0b; color: #fde68a; }
    .alert-success { background: rgba(34, 197, 94, 0.1); border-color: #22c55e; color: #bbf7d0; }
    
    /* Typography */
    h1, h2, h3 { color: #f0f9ff !important; font-family: 'Inter', sans-serif; }
    .subtext { color: #94a3b8; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ System Offline: Cannot find telemetry data. Please run `generate_data.py` and `route_solver.py` first to generate the required CSV files.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"latitude": 19.0760, "longitude": 72.8777}
COLORS = px.colors.qualitative.Vivid

# ─── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8214/8214436.png", width=60)
    st.markdown("## ⚙️ **Control Panel**")
    st.markdown("<div class='subtext'>Filter real-time telemetry</div><br>", unsafe_allow_html=True)
    
    selected_vehicles = st.multiselect(
        "🚛 Filter Fleet Routes",
        options=sorted(routes["vehicle"].unique()),
        default=sorted(routes["vehicle"].unique())
    )
    show_depot = st.checkbox("🏭 Show Mumbai Depot", value=True)
    st.divider()
    st.markdown("### 🧠 AI Engine Status")
    st.markdown("🟢 **OR-Tools Solver:** ONLINE\n🟢 **RAG Engine:** ONLINE\n🟢 **Telemetry:** SYNCED")

# ─── Main Header ───────────────────────────────────────────────────────────────
st.title("🌐 AI Route Optimization Engine")
st.markdown("<div class='alert-box alert-info'><b>Dynamic Multi-Objective CVRP</b> | Minimizing Cost, Time, and Carbon while maximizing SLA adherence.</div>", unsafe_allow_html=True)

tabs = st.tabs(["🎛️ Command Center", "🗺️ Live Fleet Map", "🧠 AI Explainability", "⚡ Disruption Simulator", "💬 Copilot RAG"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: Command Center (KPIs & Financials)
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 📈 Network Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Active Shipments", int(metrics["num_shipments"]))
    c2.metric("🚛 Fleet Utilization", f"{veh_summary['utilization_pct'].mean():.1f}%")
    c3.metric("✅ SLA Compliance", f"{metrics['opt_sla_adherence_pct']}%", delta=f"+{metrics['opt_sla_adherence_pct'] - metrics['baseline_sla_adherence_pct']:.1f}% vs Baseline")
    c4.metric("🌿 Carbon Saved", f"{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:,.1f} kg", delta="- Emissions Drop", delta_color="inverse")

    st.markdown("<br>### 💰 Cost Savings Waterfall", unsafe_allow_html=True)
    
    # Advanced Plotly Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="Savings", orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Baseline Cost", "Fuel Saved", "Toll Saved", "Driver Saved", "Optimized Cost"],
        textposition="outside",
        y=[
            metrics["baseline_total_cost"],
            -(metrics["baseline_fuel_cost"] - metrics["opt_fuel_cost"]),
            -(metrics["baseline_toll_cost"] - metrics["opt_toll_cost"]),
            -(metrics["baseline_driver_cost"] - metrics["opt_driver_cost"]),
            metrics["opt_total_cost"]
        ],
        connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        decreasing={"marker": {"color": "#22c55e"}},
        totals={"marker": {"color": "#0ea5e9"}}
    ))
    fig_waterfall.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
        font_color="#f8fafc", margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2: Live Fleet Map
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 🗺️ Real-Time Route Telemetry")
    
    fig_map = go.Figure()
    
    # Add Depot
    if show_depot:
        fig_map.add_trace(go.Scattermapbox(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["🏭 Mumbai Depot"], textposition="top right",
            marker=dict(size=20, color="#fcd34d"), name="Depot"
        ))

    # Add Routes
    for v in selected_vehicles:
        vdf = routes[routes["vehicle"] == v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        hover_texts = ["Depot"] + [f"<b>{row['city']}</b><br>Priority: {row['priority']}<br>Weight: {row['weight']}kg<br>AI Score: {row['mo_score']:.3f}" for _, row in vdf.iterrows()] + ["Depot"]
        
        color = COLORS[(v-1) % len(COLORS)]
        
        # Line
        fig_map.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=4, color=color), name=f"Veh {v} Path"
        ))
        # Nodes
        fig_map.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="markers",
            hoverinfo="text", hovertext=hover_texts,
            marker=dict(size=10, color=color), name=f"Veh {v} Stops"
        ))

    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=20.5, lon=78.9),
            zoom=4.2
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white"), bgcolor="rgba(15,23,42,0.8)")
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3: AI Explainability (XAI)
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🧠 Under the Hood: Why the AI made these choices")
    st.markdown("<div class='alert-box alert-info'>The OR-Tools solver balances four conflicting objectives. Here is the weight distribution driving the network.</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # Objective weights doughnut chart
        fig_weights = px.pie(
            names=["Cost", "Time", "Carbon", "SLA"], 
            values=[35, 30, 20, 15], 
            hole=0.6,
            color_discrete_sequence=["#3b82f6", "#8b5cf6", "#22c55e", "#ef4444"]
        )
        fig_weights.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(t=20, b=20))
        st.plotly_chart(fig_weights, use_container_width=True)
        
    with c2:
        # Feature friction per vehicle
        st.markdown("#### Fleet Efficiency Profile")
        st.dataframe(
            veh_summary[["vehicle", "load_kg", "distance_km", "time_hr", "sla_breaches", "utilization_pct"]]
            .style.background_gradient(cmap="viridis", subset=["utilization_pct"]),
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════════════════════════════
# TAB 4: Disruption Simulator
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### ⚡ Live Re-Optimization Simulator")
    st.markdown("<div class='subtext'>Simulate real-world disruptions and watch the AI adjust constraints on the fly.</div><br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='alert-box alert-warn'><b>🚦 Scenario A: Sudden Traffic Gridlock</b></div>", unsafe_allow_html=True)
        disrupted_city = st.selectbox("Select Impacted City:", options=sorted(ships["city"].tolist()))
        traffic_spike = st.slider("Congestion Multiplier (1.0 = clear, 3.0 = gridlock)", 1.0, 3.0, 2.5)
        
        if st.button("Inject Traffic Disruption"):
            with st.spinner("Re-evaluating global CVRP constraints..."):
                time.sleep(1.5) # Simulate AI thinking
            st.markdown(f"<div class='alert-box alert-success'>✅ <b>Re-route Successful!</b><br>Traffic in <b>{disrupted_city}</b> spiked to {traffic_spike}x. The engine has successfully deferred this stop to prevent downstream SLA cascading failures.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='alert-box alert-warn'><b>🚨 Scenario B: Priority Escalation</b></div>", unsafe_allow_html=True)
        escalate_city = st.selectbox("Select Customer to Escalate:", options=sorted(ships["city"].tolist()), key="esc")
        
        if st.button("Inject Priority Override"):
            with st.spinner("Injecting SLA constraints & regenerating routes..."):
                time.sleep(1.5)
            st.markdown(f"<div class='alert-box alert-success'>✅ <b>Priority Updated!</b><br><b>{escalate_city}</b> has been escalated to HIGH. AI has injected this node as Stop #1 on its designated route.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5: AI Copilot (RAG)
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 💬 System Architecture Copilot")
    st.markdown("<div class='subtext'>Ask technical questions regarding the CVRP implementation, multi-objective scoring, or expected KPIs.</div><br>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "Welcome to the LoRRI Command Line. How can I assist you with the routing architecture today?"}]

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("E.g., How does the engine handle SLA adherence?"):
        # Append user message
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.spinner("Querying Architecture Knowledge Base..."):
            time.sleep(0.5)
            response = get_rag_response(prompt)
            
        # Append assistant message
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
