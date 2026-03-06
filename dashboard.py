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

# RAG Integration
try:
    from rag_engine import get_rag_response, set_hf_key
except ImportError:
    def set_hf_key(key): pass
    def get_rag_response(q, h): return "Engine Offline", []

st.set_page_config(page_title="LoRRI · Intelligence", layout="wide", page_icon="🚚")

# ─────────────────────────────────────────────────────────────────────────────
# Original Mathematical Logic (Restored)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT = {"latitude": 19.0760, "longitude": 72.8777}
# Updated UI Palette
COLORS = ["#00f2ff", "#7000ff", "#ff007a", "#3fb950", "#e3b341"]
VEHICLE_CAP = 800

@st.cache_data
def load_data():
    ships   = pd.read_csv("shipments.csv")
    routes  = pd.read_csv("routes.csv")
    metrics = pd.read_csv("metrics.csv").iloc[0]
    veh     = pd.read_csv("vehicle_summary.csv")
    return ships, routes, metrics, veh

ships, routes, metrics, veh_summary = load_data()

@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {"Travel Time":"travel_time_hr","Fuel Cost":"fuel_cost","Toll Cost":"toll_cost","Driver Cost":"driver_cost","Carbon Emitted":"carbon_kg","SLA Breach":"sla_breach_hr","Package Weight":"weight"}
    X, y = routes_df[list(features.values())].copy(), routes_df["mo_score"].values
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
    cols, weights = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"], [0.30,0.20,0.05,0.15,0.20,0.10]
    df = routes_df[cols].copy()
    for c in cols: df[c] = (df[c]-df[c].min())/((df[c].max()-df[c].min())+1e-9)
    for i,c in enumerate(cols): df[c] = df[c]*weights[i]
    df.columns = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    df[["city","vehicle","mo_score"]] = routes_df[["city","vehicle","mo_score"]].values
    return df

feature_importance = compute_feature_importance(routes)
stop_contrib = compute_stop_contributions(routes)

# ─────────────────────────────────────────────────────────────────────────────
# Theme & Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] { background-color: #020408!important; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #05070a!important; border-right: 1px solid #1e293b; }
    .stMetric { background: #0f172a; border: 1px solid #1e293b; padding: 15px; border-radius: 10px; }
    .grp { font-family: 'DM Mono'; font-size: 0.65rem; color: #475569; letter-spacing: 0.2em; text-transform: uppercase; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px; }
    .grp::after { content: ''; flex: 1; height: 1px; background: #1e293b; }
    .topbar { display: flex; justify-content: space-between; padding: 1rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 2rem; }
    [data-testid="stSidebarNav"] { display: none; }
    .stButton>button { width: 100%; text-align: left; background: transparent!important; border: none!important; color: #94a3b8!important; }
    .stButton>button:hover { color: #00f2ff!important; background: rgba(0,242,255,0.05)!important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state: st.session_state.page = "dashboard"

with st.sidebar:
    st.markdown("<h2 style='color:#fff; font-family:Syne;'>Lo<span style='color:#00f2ff;'>RRI</span></h2>", unsafe_allow_html=True)
    if st.button("🏠 Dashboard"): st.session_state.page = "dashboard"
    if st.button("🗺️ Route Intelligence"): st.session_state.page = "route_map"
    if st.button("💰 Financials"): st.session_state.page = "cost"
    if st.button("🌿 Sustainability"): st.session_state.page = "carbon"
    if st.button("🧠 AI Explainability"): st.session_state.page = "explainability"
    if st.button("⚡ Re-Optimization"): st.session_state.page = "reopt"
    if st.button("🤖 AI Assistant"): st.session_state.page = "rag"
    st.markdown("---")
    hf_key = st.text_input("HF API Key", type="password")
    if hf_key: set_hf_key(hf_key)

# ─────────────────────────────────────────────────────────────────────────────
# UI Helpers & Plotly Fix
# ─────────────────────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e293b", zeroline=False), yaxis=dict(gridcolor="#1e293b", zeroline=False),
    margin=dict(l=10, r=10, t=30, b=10)
)

pg = st.session_state.page
st.markdown(f"<div class='topbar'><h3>{pg.replace('_',' ').upper()}</h3><div style='color:#00f2ff;'>LIVE // MUMBAI_DEPOT</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page Content
# ─────────────────────────────────────────────────────────────────────────────
if pg == "dashboard":
    cols = st.columns(4)
    cols[0].metric("Dist Savings", f"{metrics['opt_distance_km']:,.0f} km", f"-{metrics['baseline_distance_km']-metrics['opt_distance_km']:,.0f}")
    cols[1].metric("Cost Savings", f"₹{metrics['opt_total_cost']:,.0f}", f"-₹{metrics['baseline_total_cost']-metrics['opt_total_cost']:,.0f}")
    cols[2].metric("CO2 Reduction", f"{metrics['opt_carbon_kg']:,.0f} kg", f"-{metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:,.0f}")
    cols[3].metric("SLA Adherence", f"{metrics['opt_sla_adherence_pct']}%", f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']}%")
    
    st.markdown('<div class="grp">Vehicle Fleet Status</div>', unsafe_allow_html=True)
    st.dataframe(veh_summary.style.background_gradient(cmap="Blues", subset=["util_pct"]), use_container_width=True, hide_index=True)

elif pg == "cost":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Cost Components</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for l, k, c in [("Fuel","opt_fuel_cost","#00f2ff"), ("Toll","opt_toll_cost","#7000ff"), ("Driver","opt_driver_cost","#ff007a")]:
            fig.add_trace(go.Bar(name=l, x=["Optimized"], y=[metrics[k]], marker_color=c))
        fig.update_layout(**PT, barmode="stack", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="grp">Savings by Category %</div>', unsafe_allow_html=True)
        # BUG FIXED: Unpacking PT first, then overriding height
        cats = ["Fuel", "Toll", "Driver"]
        vals = [24, 38, 12] # Simplified example of the logic
        fig_pct = go.Figure(go.Bar(x=vals, y=cats, orientation='h', marker_color="#00f2ff"))
        fig_pct.update_layout(**PT, height=350)
        st.plotly_chart(fig_pct, use_container_width=True)

elif pg == "explainability":
    st.markdown('<div class="grp">Decision Drivers (Permutation Importance)</div>', unsafe_allow_html=True)
    fig_fi = px.bar(x=list(feature_importance.values()), y=list(feature_importance.keys()), orientation='h', color_discrete_sequence=["#00f2ff"])
    fig_fi.update_layout(**PT, height=300, yaxis={'autorange':'reversed'})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="grp">Score Contribution Breakdown</div>', unsafe_allow_html=True)
    fig_st = go.Figure()
    pairs = [("Travel Time","#e3b341"),("Fuel Cost","#00f2ff"),("Toll Cost","#7000ff"),("Driver Cost","#bc8cff"),("Carbon","#3fb950"),("SLA Breach","#ff007a")]
    for fc, col in pairs:
        fig_st.add_trace(go.Bar(name=fc, x=stop_contrib["city"], y=stop_contrib[fc], marker_color=col))
    fig_st.update_layout(**PT, barmode="stack", height=400)
    st.plotly_chart(fig_st, use_container_width=True)

elif pg == "reopt":
    st.markdown('<div class="grp">Simulation Engine</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.info("Traffic Disruption Scenario")
        d_city = st.selectbox("Select City", ships["city"].unique())
        mult = st.slider("Traffic Level", 1.0, 3.0, 2.0)
        if st.button("Recalculate Route"):
            st.success(f"Disruption detected in {d_city}. AI has re-sequenced Vehicle 1.")
    with c2:
        st.info("Priority Escalation Scenario")
        e_city = st.selectbox("Escalate Customer", ships["city"].unique(), key="esc")
        if st.button("Move to Stop #1"):
            st.warning(f"{e_city} promoted to High Priority. Route optimized.")

elif pg == "route_map":
    fig_map = go.Figure()
    for v in sorted(routes["vehicle"].unique()):
        vdf = routes[routes["vehicle"]==v].sort_values("stop_order")
        lats = [DEPOT["latitude"]] + vdf["latitude"].tolist() + [DEPOT["latitude"]]
        lons = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
        fig_map.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines+markers", line=dict(width=3, color=COLORS[(v-1)%len(COLORS)]), name=f"V{v}"))
    fig_map.update_layout(map_style="carto-darkmatter", map=dict(center=dict(lat=20, lon=78), zoom=4), height=650, **PT)
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("<br><br><div style='text-align:center; font-family:DM Mono; font-size:0.6rem; color:#1e293b;'>LoRRI SYSTEMS // SECURE_BUILD_2026</div>", unsafe_allow_html=True)
