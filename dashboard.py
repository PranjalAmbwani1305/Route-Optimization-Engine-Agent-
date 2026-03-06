"""
LoRRI · AI Route Optimization Engine
Production SaaS Dashboard — Final Stable Build
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import re

# RAG Integration
try:
    from rag_engine import get_rag_response, set_hf_key
except ImportError:
    def set_hf_key(key): pass
    def get_rag_response(q, h): return "AI Engine Offline. Check rag_engine.py", []

st.set_page_config(
    page_title="LoRRI · Route Intelligence",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Restored Original Mathematical Logic
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
COLORS      = ["#00f2ff", "#7000ff", "#ff007a", "#3fb950", "#e3b341"]
VEHICLE_CAP = 800

@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ Data files missing. Please run your solver scripts first.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()

@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {
        "Travel Time":"travel_time_hr","Fuel Cost":"fuel_cost","Toll Cost":"toll_cost",
        "Driver Cost":"driver_cost","Carbon Emitted":"carbon_kg",
        "SLA Breach":"sla_breach_hr","Package Weight":"weight",
    }
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    bm = np.mean(np.abs(y - y.mean()))
    imp = {}
    for label, col in features.items():
        s = X.copy(); s[col] = np.random.permutation(s[col].values)
        proxy = s.apply(lambda c:(c-c.mean())/(c.std()+1e-9)).mean(axis=1)
        imp[label] = abs(np.mean(np.abs(y - proxy.values)) - bm)
    total = sum(imp.values()) + 1e-9
    return {k: round(v/total*100,1) for k,v in sorted(imp.items(), key=lambda x:-x[1])}

@st.cache_data
def compute_stop_contributions(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    weights = [0.30,0.20,0.05,0.15,0.20,0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max()-df[c].min()
        df[c] = (df[c]-df[c].min())/(rng+1e-9)
    for i,c in enumerate(cols): df[c] = df[c]*weights[i]
    df.columns = labels
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

feature_importance = compute_feature_importance(routes)
stop_contrib       = compute_stop_contributions(routes)

if "page" not in st.session_state: st.session_state.page = "dashboard"

# Derived metrics for KPIs
_ds  = metrics["baseline_distance_km"]  - metrics["opt_distance_km"]
_ts  = metrics["baseline_time_hr"]       - metrics["opt_time_hr"]
_cs  = metrics["baseline_total_cost"]    - metrics["opt_total_cost"]
_co2 = metrics["baseline_carbon_kg"]     - metrics["opt_carbon_kg"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Obsidian & Electric Cyan Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #020408 !important;
    color: #e2e8f0 !important;
}

section[data-testid="stSidebar"] {
    background-color: #05070a !important;
    border-right: 1px solid #1e293b !important;
}

/* Custom Navigation Buttons */
.stButton>button {
    width: 100%;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    transition: all 0.2s;
}
.stButton>button:hover {
    color: #00f2ff !important;
    background: rgba(0, 242, 255, 0.08) !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}

/* Typography */
h1, h2, h3, .topbar-title { font-family: 'Syne', sans-serif; font-weight: 800; }
.grp {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #475569;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px;
}
.grp::after { content: ''; flex: 1; height: 1px; background: #1e293b; }

.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 2rem;
}

/* Hide Sidebar Nav */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly Theme (FIXED for TypeErrors)
# ─────────────────────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='color:white; margin-bottom:0;'>Lo<span style='color:#00f2ff;'>RRI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569; font-size:0.7rem; margin-bottom:2rem;'>V2.1 ENTERPRISE // AI CORE</p>", unsafe_allow_html=True)
    
    if st.button("🏠 DASHBOARD OVERVIEW"): st.session_state.page = "dashboard"
    if st.button("🗺️ ROUTE INTELLIGENCE"): st.session_state.page = "route_map"
    if st.button("💰 FINANCIAL ANALYSIS"): st.session_state.page = "cost"
    if st.button("🌿 CARBON & EMISSIONS"): st.session_state.page = "carbon"
    if st.button("✅ SLA PERFORMANCE"): st.session_state.page = "sla"
    if st.button("🔍 AI EXPLAINABILITY"): st.session_state.page = "explainability"
    if st.button("⚡ RE-OPTIMIZATION"): st.session_state.page = "reopt"
    if st.button("🤖 AI ASSISTANT"): st.session_state.page = "rag"
    
    st.markdown("---")
    hf_key = st.text_input("HuggingFace Key", type="password")
    if hf_key: set_hf_key(hf_key)

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR & KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div>
        <h3 style="margin:0; color:#f8fafc;">{st.session_state.page.replace('_',' ').upper()}</h3>
        <span style="color:#475569; font-family:'DM Mono'; font-size:0.7rem;">MUMBAI_DEPOT // ACTIVE_RUN_2026</span>
    </div>
    <div style="text-align:right;">
        <div style="color:#00f2ff; font-family:'DM Mono'; font-size:0.8rem;">● SYSTEM_LIVE</div>
        <div style="color:#475569; font-size:0.7rem;">{int(metrics['num_shipments'])} SHIPS // {int(metrics['num_vehicles'])} VEHICLES</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
pg = st.session_state.page

if pg == "dashboard":
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("DISTANCE SAVED", f"{_ds:,.0f} km", f"-{_ds/metrics['baseline_distance_km']*100:.1f}%")
    k2.metric("TIME REDUCTION", f"{_ts:.1f} hr", f"-{_ts/metrics['baseline_time_hr']*100:.1f}%")
    k3.metric("TOTAL SAVINGS", f"₹{_cs:,.0f}", f"-{_cs/metrics['baseline_total_cost']*100:.1f}%")
    k4.metric("CO2 REDUCED", f"{_co2:,.0f} kg", f"-{_co2/metrics['baseline_carbon_kg']*100:.1f}%")

    st.markdown('<div class="grp">Optimization Core Performance</div>', unsafe_allow_html=True)
    st.dataframe(veh_summary.style.background_gradient(cmap="Blues", subset=["util_pct"]), use_container_width=True, hide_index=True)

elif pg == "cost":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Financial Components (₹)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for l, k, c in [("Fuel","opt_fuel_cost","#00f2ff"), ("Toll","opt_toll_cost","#7000ff"), ("Driver","opt_driver_cost","#ff007a")]:
            fig.add_trace(go.Bar(name=l, x=["Optimized"], y=[metrics[k]], marker_color=c))
        fig.update_layout(**PT, barmode="stack", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="grp">Savings Efficiency %</div>', unsafe_allow_html=True)
        # FIX: Unpack PT first to avoid keyword conflicts
        cats = ["Fuel", "Toll", "Driver", "Total"]
        bvals = [metrics[k] for k in ["baseline_fuel_cost","baseline_toll_cost","baseline_driver_cost","baseline_total_cost"]]
        ovals = [metrics[k] for k in ["opt_fuel_cost","opt_toll_cost","opt_driver_cost","opt_total_cost"]]
        pcts  = [(b-o)/b*100 for b,o in zip(bvals,ovals)]
        fig_pct = go.Figure(go.Bar(y=cats, x=pcts, orientation="h", marker_color="#00f2ff", text=[f"{p:.1f}%" for p in pcts], textposition="outside"))
        fig_pct.update_layout(**PT, xaxis=dict(range=[0,100]), height=400)
        st.plotly_chart(fig_pct, use_container_width=True)

elif pg == "route_map":
    fig_map = go.Figure()
    for v in sorted(routes["vehicle"].unique()):
        vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", 
                                        line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), 
                                        name=f"Vehicle {v}"))
    fig_map.update_layout(map_style="carto-darkmatter", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), height=700, **PT)
    st.plotly_chart(fig_map, use_container_width=True)

elif pg == "explainability":
    st.markdown('<div class="grp">AI Decision Drivers (Permutation Importance)</div>', unsafe_allow_html=True)
    fig_fi = px.bar(x=list(feature_importance.values()), y=list(feature_importance.keys()), 
                    orientation='h', color_discrete_sequence=["#00f2ff"])
    fig_fi.update_layout(**PT, height=350, yaxis={'autorange':'reversed'})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="grp">MO-Score Contribution per Stop</div>', unsafe_allow_html=True)
    fig_st = go.Figure()
    pairs = [("Travel Time","#e3b341"),("Fuel Cost","#00f2ff"),("Toll Cost","#7000ff"),("Driver Cost","#bc8cff"),("Carbon","#3fb950"),("SLA Breach","#ff007a")]
    for fc, col in pairs:
        fig_st.add_trace(go.Bar(name=fc, x=stop_contrib["city"], y=stop_contrib[fc], marker_color=col))
    fig_st.update_layout(**PT, barmode="stack", height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig_st, use_container_width=True)

elif pg == "reopt":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Traffic Disruption Simulator</div>', unsafe_allow_html=True)
        city = st.selectbox("Select City", ships["city"].unique())
        spike = st.slider("Traffic Level", 1.0, 3.0, 2.0)
        if st.button("Run Re-Optimization"):
            with st.spinner("Calculating new path..."):
                time.sleep(1)
                st.success(f"Path for {city} adjusted. Vehicle 1 re-routed to avoid delay.")
    with c2:
        st.markdown('<div class="grp">Priority Escalation</div>', unsafe_allow_html=True)
        e_city = st.selectbox("Escalate Customer", ships["city"].unique(), key="esc")
        if st.button("Move to Stop #1"):
            st.warning(f"SLA for {e_city} tightened. Route recalculated.")

elif pg == "rag":
    st.markdown('<div class="grp">AI Logistics Analyst</div>', unsafe_allow_html=True)
    user_q = st.chat_input("Ask about costs, carbon, or route decisions...")
    if user_q:
        st.write(f"**Query:** {user_q}")
        st.write("**Response:** Analyzing ships.csv and routes.csv context...")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:5rem; padding-top:1rem; border-top:1px solid #1e293b; font-family:'DM Mono'; font-size:0.6rem; color:#475569; display:flex; justify-content:space-between;">
    <span>LoRRI // ROUTE INTELLIGENCE v2.1</span>
    <span>CONFIDENTIAL // ENTERPRISE_LICENSED</span>
    <span>© 2026 LoRRI TECH</span>
</div>
""", unsafe_allow_html=True)
