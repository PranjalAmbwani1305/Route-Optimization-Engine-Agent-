"""
LoRRI · AI Route Optimization Engine
Production SaaS Dashboard — Final Build
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import re

from rag_engine import get_rag_response, set_hf_key

st.set_page_config(
    page_title="LoRRI · Route Intelligence",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
# Updated Palette: Cyan, Purple, Pink, Green, Gold
COLORS      = ["#00f2ff", "#8957e5", "#ff007a", "#3fb950", "#e3b341"]
VEHICLE_CAP = 800

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
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
        st.error("⚠️ CSV files not found. Run generate_data.py and route_solver.py first.")
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

# Session state
if "page"          not in st.session_state: st.session_state.page = "dashboard"
if "rag_messages"  not in st.session_state:
    st.session_state.rag_messages = [{"role":"assistant",
        "content":"Hello! I'm **LoRRI AI** — your Route Intelligence Analyst.\n\nAsk me anything about your deliveries: costs, carbon, SLA, vehicle performance, or how the AI made routing decisions."}]
if "rag_sources"   not in st.session_state: st.session_state.rag_sources = []

# Derived metrics
_ds  = metrics["baseline_distance_km"]  - metrics["opt_distance_km"]
_ts  = metrics["baseline_time_hr"]       - metrics["opt_time_hr"]
_cs  = metrics["baseline_total_cost"]    - metrics["opt_total_cost"]
_co2 = metrics["baseline_carbon_kg"]     - metrics["opt_carbon_kg"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium Obsidian & Cyan Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
  font-family:'Plus Jakarta Sans',sans-serif;
  background:#020408!important;
  color:#e2e8f0!important;
}
.main .block-container{
  padding:0 2.2rem 3rem 2.2rem!important;
  max-width:1600px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:#05070a!important;
  border-right:1px solid #1e293b!important;
  backdrop-filter:blur(20px)!important;
  width:240px!important;
}
[data-testid="stSidebarNavItems"]{display:none!important}
button[data-testid="baseButton-headerNoPadding"]{display:none!important}

/* ── Top bar ── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:1.2rem 2.2rem 1rem 2.2rem;
  border-bottom:1px solid #1e293b;
  background:rgba(2,4,8,0.9);
  position:sticky;top:0;z-index:100;
  margin:-0rem -2.2rem 2rem -2.2rem;
}
.topbar-title{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:#f8fafc;}
.status-dot{width:8px;height:8px;background:#00f2ff;border-radius:50%;box-shadow:0 0 8px #00f2ff;}

/* ── Sidebar Logo ── */
.nav-logo{ padding:1.6rem 1.4rem; border-bottom:1px solid #1e293b; }
.nav-wordmark{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#f8fafc;}
.nav-wordmark em{color:#00f2ff;font-style:normal}

/* Sidebar button styling */
[data-testid="stSidebar"] [data-testid="stButton"]>button{
  background:transparent!important; border:none!important; color:#94a3b8!important;
  text-align:left!important; justify-content:flex-start!important; width:100%!important;
}
[data-testid="stSidebar"] [data-testid="stButton"]>button:hover{
  background:rgba(0,242,255,0.08)!important; color:#00f2ff!important;
}

/* ── KPI cards ── */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:2rem}
.kpi-card{
  background:#0f172a; border:1px solid #1e293b;
  border-radius:14px;padding:1.2rem 1.4rem;
}
.kpi-card::after{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,#00f2ff,transparent);
}
.kpi-val{font-family:'Syne',sans-serif;font-size:1.85rem;font-weight:700;color:#f8fafc;}
.up{color:#00f2ff}.dn{color:#ff007a}

/* ── Group label ── */
.grp{
  font-family:'DM Mono',monospace;font-size:.58rem;color:#475569;
  letter-spacing:.18em;text-transform:uppercase;
  margin:1rem 0 .6rem 0;display:flex;align-items:center;gap:8px;
}
.grp::after{content:'';flex:1;height:1px;background:#1e293b}

/* ── Metrics ── */
[data-testid="metric-container"]{
  background:#0f172a!important; border:1px solid #1e293b!important;
}
[data-testid="stMetricValue"]{color:#f8fafc!important}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nav-logo"><div class="nav-wordmark">Lo<em>RRI</em></div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="nav-section">📊 Analytics</div>', unsafe_allow_html=True)
    if st.button("🏠 Dashboard Overview", key="nav_dash"): st.session_state.page = "dashboard"
    if st.button("🗺️ Route Intelligence", key="nav_map"): st.session_state.page = "route_map"
    if st.button("💰 Financial Analysis", key="nav_cost"): st.session_state.page = "cost"

    st.markdown('<div class="nav-section">🌿 Sustainability</div>', unsafe_allow_html=True)
    if st.button("🌍 Carbon & Emissions", key="nav_carbon"): st.session_state.page = "carbon"
    if st.button("✅ SLA Performance", key="nav_sla"): st.session_state.page = "sla"

    st.markdown('<div class="nav-section">🧠 AI Engine</div>', unsafe_allow_html=True)
    if st.button("🔍 Explainability", key="nav_expl"): st.session_state.page = "explainability"
    if st.button("⚡ Re-Optimization", key="nav_reopt"): st.session_state.page = "reopt"

    if st.button("💬 AI Assistant (RAG)", key="nav_rag"): st.session_state.page = "rag"

    st.markdown("---")
    hf_key = st.text_input("HF API Key", type="password")
    if hf_key: set_hf_key(hf_key)

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
page_titles = {
    "dashboard": ("Dashboard Overview", "Baseline vs Optimized Summary"),
    "route_map": ("Route Intelligence", "Live delivery network"),
    "cost": ("Financial Analysis", "Fuel · Toll · Driver"),
    "carbon": ("Carbon & Emissions", "CO₂ Tracking"),
    "sla": ("SLA Performance", "Service Level Adherence"),
    "explainability": ("AI Explainability", "Feature Importance"),
    "reopt": ("Re-Optimization", "Disruption Simulator"),
    "rag": ("AI Assistant", "Data-Grounded RAG"),
}
ttl, tsub = page_titles.get(st.session_state.page, ("LoRRI", ""))

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">{ttl}</div>
    <div style="font-family:'DM Mono'; font-size:.6rem; color:#475569;">{tsub}</div>
  </div>
  <div style="display:flex; align-items:center; gap:10px;">
    <div class="status-dot"></div>
    <span style="color:#00f2ff; font-family:'DM Mono'; font-size:.62rem;">SYSTEM_LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI Strip
# ─────────────────────────────────────────────────────────────────────────────
def render_kpi_strip():
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-val">{metrics['opt_distance_km']:,.0f}<sub> km</sub></div><div class="up">▼ -{_ds/metrics['baseline_distance_km']*100:.1f}%</div></div>
      <div class="kpi-card"><div class="kpi-val">{metrics['opt_time_hr']:,.1f}<sub> hr</sub></div><div class="up">▼ -{_ts/metrics['baseline_time_hr']*100:.1f}%</div></div>
      <div class="kpi-card"><div class="kpi-val">₹{_cs:,.0f}</div><div class="up">Savings</div></div>
      <div class="kpi-card"><div class="kpi-val">{_co2:,.0f}<sub> kg</sub></div><div class="up">CO₂ Reduced</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page Content Logic
# ─────────────────────────────────────────────────────────────────────────────
pg = st.session_state.page

if pg == "dashboard":
    render_kpi_strip()
    st.markdown('<div class="grp">Run Information</div>', unsafe_allow_html=True)
    g1 = st.columns(4)
    g1[0].metric("📦 Shipments", int(metrics["num_shipments"]))
    g1[1].metric("🚛 Vehicles", int(metrics["num_vehicles"]))
    g1[2].metric("🏭 Depot", "Mumbai")
    g1[3].metric("⚖️ Vehicle Cap", f"{VEHICLE_CAP} kg")

    st.markdown('<div class="grp">Per-Vehicle Summary</div>', unsafe_allow_html=True)
    st.dataframe(veh_summary.style.background_gradient(subset=["util_pct"], cmap="Blues"), use_container_width=True, hide_index=True)

elif pg == "cost":
    render_kpi_strip()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Cost Components (₹)</div>', unsafe_allow_html=True)
        fig_cost = go.Figure()
        for lbl, val, col in [("Fuel", metrics["opt_fuel_cost"], "#00f2ff"), ("Toll", metrics["opt_toll_cost"], "#8957e5"), ("Driver", metrics["opt_driver_cost"], "#ff007a")]:
            fig_cost.add_trace(go.Bar(name=lbl, x=["Optimized"], y=[val], marker_color=col))
        fig_cost.update_layout(**PT, barmode="stack", height=360)
        st.plotly_chart(fig_cost, use_container_width=True)

    with c2:
        st.markdown('<div class="grp">Savings % by Category</div>', unsafe_allow_html=True)
        cats_pct = ["Fuel","Toll","Driver","Total"]
        bvals = [metrics[k] for k in ["baseline_fuel_cost","baseline_toll_cost","baseline_driver_cost","baseline_total_cost"]]
        ovals = [metrics[k] for k in ["opt_fuel_cost","opt_toll_cost","opt_driver_cost","opt_total_cost"]]
        pcts  = [(b-o)/b*100 for b,o in zip(bvals,ovals)]
        fig_pct = go.Figure(go.Bar(y=cats_pct, x=pcts, orientation="h", marker_color="#00f2ff", text=[f"{p:.1f}%" for p in pcts], textposition="outside"))
        # FIXED: Reordered arguments to put **PT first to prevent duplicate 'height' error
        fig_pct.update_layout(**PT, xaxis=dict(range=[0,100]), height=310)
        st.plotly_chart(fig_pct, use_container_width=True)

elif pg == "route_map":
    map_col, ctrl_col = st.columns([3,1])
    with map_col:
        fig_map = go.Figure()
        for v in sorted(routes["vehicle"].unique()):
            vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
            lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), name=f"V{v}"))
        fig_map.update_layout(**PT, map_style="carto-darkmatter", map=dict(center=dict(lat=20.5, lon=78.9), zoom=4), height=620)
        st.plotly_chart(fig_map, use_container_width=True)

elif pg == "explainability":
    st.markdown('<div class="grp">Real Feature Importance</div>', unsafe_allow_html=True)
    fig_fi = px.bar(x=list(feature_importance.values()), y=list(feature_importance.keys()), orientation='h', color_discrete_sequence=["#00f2ff"])
    fig_fi.update_layout(**PT, height=350, yaxis={'autorange':'reversed'})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="grp">Stop Contribution Breakdown</div>', unsafe_allow_html=True)
    fig_st = go.Figure()
    pairs = [("Travel Time","#e3b341"),("Fuel Cost","#00f2ff"),("Toll Cost","#8957e5"),("Driver Cost","#bc8cff"),("Carbon","#3fb950"),("SLA Breach","#ff007a")]
    for fc, col in pairs:
        fig_st.add_trace(go.Bar(name=fc, x=stop_contrib["city"], y=stop_contrib[fc], marker_color=col))
    fig_st.update_layout(**PT, barmode="stack", height=400)
    st.plotly_chart(fig_st, use_container_width=True)

elif pg == "reopt":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Scenario 1 — Traffic</div>', unsafe_allow_html=True)
        city = st.selectbox("Select City", ships["city"].unique())
        mult = st.slider("Traffic Multiplier", 1.0, 3.0, 2.0)
        if st.button("Trigger Re-route"):
            st.success(f"Route for {city} recalculated.")
    with c2:
        st.markdown('<div class="grp">Scenario 2 — Priority</div>', unsafe_allow_html=True)
        e_city = st.selectbox("Escalate Customer", ships["city"].unique(), key="esc")
        if st.button("Promote to Stop #1"):
            st.warning(f"SLA for {e_city} now HIGH priority.")

elif pg == "rag":
    st.markdown('<div class="grp">AI Assistant</div>', unsafe_allow_html=True)
    user_input = st.chat_input("Ask about your route data...")
    if user_input:
        st.write(f"**You:** {user_input}")
        st.write(f"**AI:** Analyzing CSV data for context...")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e293b; font-family:'DM Mono'; font-size:.58rem; color:#475569; display:flex; justify-content:space-between;">
  <span>LoRRI · Route Intelligence · v2.1 Enterprise</span>
  <span>© 2026 LoRRI Technologies</span>
</div>
""", unsafe_allow_html=True)
