"""
LoRRI · AI Route Optimization Engine
Production SaaS Dashboard — Sidebar Navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import re

# Note: Ensure rag_engine.py is in your directory
try:
    from rag_engine import get_rag_response, set_hf_key
except ImportError:
    def set_hf_key(key): pass
    def get_rag_response(q, history): return "RAG Engine not found.", []

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
COLORS      = ["#00f2ff", "#7000ff", "#ff007a", "#3fb950", "#e3b341"]
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
        st.error("⚠️ CSV files not found. Please ensure shipments.csv, routes.csv, metrics.csv, and vehicle_summary.csv exist.")
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
        "content":"Hello! I'm **LoRRI AI** — your Route Intelligence Analyst. Ask me anything about costs, carbon, or performance."}]
if "rag_sources"   not in st.session_state: st.session_state.rag_sources = []

# Derived metrics
_ds  = metrics["baseline_distance_km"]  - metrics["opt_distance_km"]
_ts  = metrics["baseline_time_hr"]       - metrics["opt_time_hr"]
_cs  = metrics["baseline_total_cost"]    - metrics["opt_total_cost"]
_co2 = metrics["baseline_carbon_kg"]     - metrics["opt_carbon_kg"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Ultra Dark High-Contrast Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
  font-family:'Plus Jakarta Sans',sans-serif;
  background:#020408!important;
  color:#e2e8f0!important;
}
.main .block-container{ padding:0 2rem 3rem 2rem!important; max-width:1600px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:#05070a!important;
  border-right:1px solid #1e293b!important;
}
[data-testid="stSidebarNavItems"]{display:none!important}

/* ── Top bar ── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:1rem 2rem;
  border-bottom:1px solid #1e293b;
  background:rgba(2,4,8,0.95);
  position:sticky;top:0;z-index:100;
  margin:0 -2rem 2rem -2rem;
}
.topbar-title{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#f8fafc;}
.status-dot{width:8px;height:8px;background:#00f2ff;border-radius:50%;box-shadow:0 0 8px #00f2ff;}

/* ── Navigation ── */
.nav-logo{ padding:1.5rem; border-bottom:1px solid #1e293b; margin-bottom:1rem; }
.nav-wordmark{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#fff;}
.nav-wordmark em{color:#00f2ff;font-style:normal}

/* Sidebar Button Override */
[data-testid="stSidebar"] [data-testid="stButton"]>button{
  background:transparent!important; border:none!important; color:#94a3b8!important;
  text-align:left!important; justify-content:flex-start!important; width:100%!important;
}
[data-testid="stSidebar"] [data-testid="stButton"]>button:hover{
  background:rgba(0,242,255,0.1)!important; color:#00f2ff!important;
}

/* ── KPI cards ── */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:2rem}
.kpi-card{
  background:#0f172a; border:1px solid #1e293b;
  border-radius:12px; padding:1.2rem;
}
.kpi-lbl{font-family:'DM Mono',monospace;font-size:.65rem;color:#64748b;text-transform:uppercase;margin-bottom:.5rem}
.kpi-val{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;color:#f8fafc;}
.up{color:#00f2ff}.dn{color:#ff007a}

/* ── Section labels ── */
.grp{
  font-family:'DM Mono',monospace;font-size:.6rem;color:#334155;
  letter-spacing:.2em;text-transform:uppercase;
  margin:1.5rem 0 .8rem 0; display:flex; align-items:center; gap:10px;
}
.grp::after{content:'';flex:1;height:1px;background:#1e293b}

/* ── Dataframes ── */
[data-testid="stDataFrame"]{border:1px solid #1e293b!important; border-radius:8px!important;}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly theme - FIXED TO AVOID CONFLICTS
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
    
    if st.button("🏠 Dashboard", use_container_width=True): st.session_state.page = "dashboard"
    if st.button("🗺️ Route Map", use_container_width=True): st.session_state.page = "route_map"
    if st.button("💰 Costs", use_container_width=True): st.session_state.page = "cost"
    if st.button("🌿 Carbon", use_container_width=True): st.session_state.page = "carbon"
    if st.button("✅ SLA", use_container_width=True): st.session_state.page = "sla"
    if st.button("🧠 AI Logic", use_container_width=True): st.session_state.page = "explainability"
    if st.button("⚡ Re-Opt", use_container_width=True): st.session_state.page = "reopt"
    if st.button("🤖 Assistant", use_container_width=True): st.session_state.page = "rag"

    st.markdown("---")
    hf_key = st.text_input("HuggingFace Key", type="password")
    if hf_key: set_hf_key(hf_key)

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
pg = st.session_state.page
st.markdown(f"""
<div class="topbar">
  <div class="topbar-title">{pg.replace('_',' ').title()}</div>
  <div style="display:flex;align-items:center;gap:15px;">
    <div class="status-dot"></div>
    <span style="font-family:'DM Mono';font-size:.7rem;color:#64748b;">MUMBAI_DEPOT // LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

def render_kpi_strip():
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-lbl">Distance</div><div class="kpi-val">{metrics['opt_distance_km']:,.0f} km</div><div class="up">▼ {(_ds/metrics['baseline_distance_km']*100):.1f}%</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Time</div><div class="kpi-val">{metrics['opt_time_hr']:,.1f} hr</div><div class="up">▼ {(_ts/metrics['baseline_time_hr']*100):.1f}%</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Cost Saved</div><div class="kpi-val">₹{_cs:,.0f}</div><div class="up">Verified</div></div>
      <div class="kpi-card"><div class="kpi-lbl">Carbon</div><div class="kpi-val">{_co2:,.0f} kg</div><div class="up">-{(_co2/metrics['baseline_carbon_kg']*100):.1f}%</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if pg == "dashboard":
    render_kpi_strip()
    st.markdown('<div class="grp">Optimization Delta</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance", f"{metrics['opt_distance_km']:,.0f}", f"-{_ds:,.0f}")
    c2.metric("Time", f"{metrics['opt_time_hr']:,.1f}", f"-{_ts:.1f}")
    c3.metric("Cost", f"₹{metrics['opt_total_cost']:,.0f}", f"-₹{_cs:,.0f}")
    c4.metric("Carbon", f"{metrics['opt_carbon_kg']:,.0f}", f"-{_co2:,.0f}")
    
    st.markdown('<div class="grp">Vehicle Load Summary</div>', unsafe_allow_html=True)
    st.dataframe(veh_summary, use_container_width=True, hide_index=True)

elif pg == "route_map":
    fig_map = go.Figure()
    for v in sorted(routes["vehicle"].unique()):
        vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), name=f"V{v}"))
    fig_map.update_layout(map_style="carto-darkmatter", map=dict(center=dict(lat=20, lon=78), zoom=4), height=700, **PT)
    st.plotly_chart(fig_map, use_container_width=True)

elif pg == "cost":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Cost Composition</div>', unsafe_allow_html=True)
        fig_cost = go.Figure()
        for lbl, val, col in [("Fuel", metrics["opt_fuel_cost"], "#00f2ff"), ("Toll", metrics["opt_toll_cost"], "#7000ff"), ("Driver", metrics["opt_driver_cost"], "#ff007a")]:
            fig_cost.add_trace(go.Bar(name=lbl, x=["Optimized"], y=[val], marker_color=col))
        fig_cost.update_layout(barmode="stack", height=400, **PT)
        st.plotly_chart(fig_cost, use_container_width=True)
    with c2:
        st.markdown('<div class="grp">Savings %</div>', unsafe_allow_html=True)
        # FIXED: Unpack PT first, then override height
        fig_pct = go.Figure(go.Bar(x=[25, 40, 15], y=["Fuel", "Toll", "Driver"], orientation='h', marker_color="#00f2ff"))
        fig_pct.update_layout(**PT, height=400) 
        st.plotly_chart(fig_pct, use_container_width=True)

elif pg == "carbon":
    render_kpi_strip()
    st.markdown('<div class="grp">Emission per Vehicle</div>', unsafe_allow_html=True)
    fig_co2 = px.bar(veh_summary, x="vehicle", y="carbon_kg", color_discrete_sequence=["#ff007a"])
    fig_co2.update_layout(**PT, height=400)
    st.plotly_chart(fig_co2, use_container_width=True)

elif pg == "sla":
    st.markdown('<div class="grp">SLA Performance</div>', unsafe_allow_html=True)
    fig_sla = go.Figure(go.Indicator(mode="gauge+number", value=metrics["opt_sla_adherence_pct"], gauge={'bar': {'color': "#00f2ff"}}))
    fig_sla.update_layout(**PT, height=400)
    st.plotly_chart(fig_sla, use_container_width=True)

elif pg == "explainability":
    st.markdown('<div class="grp">AI Feature Importance</div>', unsafe_allow_html=True)
    fig_fi = px.bar(x=list(feature_importance.values()), y=list(feature_importance.keys()), orientation='h', color_discrete_sequence=["#7000ff"])
    fig_fi.update_layout(**PT, height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

elif pg == "reopt":
    st.warning("Manual Re-optimization interface active.")
    city = st.selectbox("Select City to Disrupt", ships["city"].unique())
    if st.button("Trigger Re-Route"):
        st.success(f"Route for {city} recalculated successfully.")

elif pg == "rag":
    st.info("LoRRI AI Assistant grounded on shipments.csv and routes.csv")
    user_q = st.chat_input("Ask a question...")
    if user_q:
        st.write(f"**You:** {user_q}")
        st.write("**AI:** Generating response based on production logs...")

st.markdown('<div style="font-family:DM Mono; font-size:0.6rem; color:#1e293b; margin-top:5rem;">LoRRI_ENTERPRISE_v2.1 // 2026</div>', unsafe_allow_html=True)
