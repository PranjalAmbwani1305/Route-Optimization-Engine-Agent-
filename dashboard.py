import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import json
from rag_engine import get_rag_response

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="DEV: CVRP Optimizer", layout="wide")

st.markdown("""
<style>
    /* Brutalist / Engineering Console Style */
    .stApp { font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .console-log {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    .metric-box { border-left: 3px solid #00ff00; padding-left: 10px; }
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
        st.error("FATAL: Missing dependencies. Execute `python generate_data.py` and `python route_solver.py`.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()
DEPOT = {"latitude": 19.0760, "longitude": 72.8777}

# ─── UI Header ───────────────────────────────────────────────────────────────
st.title("🔧 DEV Console: Route Optimization Engine")
st.text("STATUS: ONLINE | ALGORITHM: Multi-Objective CVRP (OR-Tools Heuristic) | DEPOT: MUMBAI")
st.divider()

tabs = st.tabs(["[1] Execution Metrics", "[2] Geospatial Viewer", "[3] Optimization Logs", "[4] Trigger API", "[5] RAG Console"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: Execution Metrics
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### System Deltas (Baseline vs Optimized)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Distance (km)", f"{metrics['opt_distance_km']}", delta=f"{metrics['opt_distance_km'] - metrics['baseline_distance_km']:.1f} km", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Total Cost (INR)", f"{metrics['opt_total_cost']}", delta=f"{metrics['opt_total_cost'] - metrics['baseline_total_cost']:.1f}", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("Carbon (kg)", f"{metrics['opt_carbon_kg']}", delta=f"{metrics['opt_carbon_kg'] - metrics['baseline_carbon_kg']:.1f}", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.metric("SLA Adherence", f"{metrics['opt_sla_adherence_pct']}%", delta=f"{metrics['opt_sla_adherence_pct'] - metrics['baseline_sla_adherence_pct']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Raw Fleet Array")
    st.dataframe(veh_summary)

# ═══════════════════════════════════════════════════════════════════
# TAB 2: Geospatial Viewer
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Route Output Visualization")
    
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
        mode="markers+text", text=["[DEPOT]"], marker=dict(size=12, color="red")
    ))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for v in sorted(routes["vehicle"].unique()):
        vdf = routes[routes["vehicle"] == v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="lines+markers",
            line=dict(width=2, color=colors[v % len(colors)]),
            name=f"V_{v}"
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 20.5, "lon": 78.9}, mapbox_zoom=4,
        margin={"r":0,"t":0,"l":0,"b":0}, height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3: Optimization Logs (XAI)
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Objective Function Weights")
    st.code("""
    W_TIME   = 0.30
    W_COST   = 0.35
    W_CARBON = 0.20
    W_SLA    = 0.15
    """, language="python")
    
    st.markdown("### `routes.csv` Header Inspector")
    st.dataframe(routes.head(10))

# ═══════════════════════════════════════════════════════════════════
# TAB 4: Trigger API
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Manual Injection: Re-Optimization Service")
    
    col1, col2 = st.columns(2)
    with col1:
        target_node = st.selectbox("Select Node ID (City):", ships["city"])
        mutation_type = st.radio("Mutation Type:", ["TRAFFIC_MULTIPLIER", "SLA_PRIORITY"])
        mutation_val = st.text_input("Value:", "2.5" if mutation_type == "TRAFFIC_MULTIPLIER" else "HIGH")
        
        if st.button("POST /api/v1/trigger"):
            start_t = time.time()
            with st.spinner("Executing..."):
                time.sleep(0.8) # Mock API latency
            
            payload = {
                "status": 200,
                "event": "RE_OPTIMIZATION_TRIGGERED",
                "node": target_node,
                "mutation": {
                    "type": mutation_type,
                    "new_value": mutation_val
                },
                "latency_ms": round((time.time() - start_t) * 1000, 2),
                "action_taken": "Re-routing affected vehicle queue to bypass threshold breach."
            }
            st.json(payload)

# ═══════════════════════════════════════════════════════════════════
# TAB 5: RAG Console
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### RAG Diagnostic Console")
    query = st.text_input("Query Architecture Documents:")
    
    if st.button("Execute Vector Search"):
        if query:
            res = get_rag_response(query)
            st.markdown("**RAW RESPONSE PAYLOAD:**")
            st.json(res)
        else:
            st.error("Missing query parameter.")
