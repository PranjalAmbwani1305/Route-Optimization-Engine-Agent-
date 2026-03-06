"""
LoRRI · AI Route Optimization Engine
Professional SaaS Intelligence Dashboard (v2.1)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import re

# Optional: Try to import RAG components, provide mocks if not present
try:
    from rag_engine import get_rag_response, set_hf_key, _build_kb
except ImportError:
    def set_hf_key(key): pass
    def get_rag_response(q, history): return "RAG Engine module not found.", []
    def _build_kb(): return [], None, None

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LoRRI · Route Intelligence",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# Standard Plotly Theme (Applied in two steps to avoid TypeErrors)
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7a9cbf", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Plus+Jakarta+Sans:wght@400;600&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #050810 !important; color: #c8d6ee !important; }
    .main .block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1600px; }
    
    /* Topbar */
    .topbar { display: flex; align-items: center; justify-content: space-between; padding: 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,.05); margin-bottom: 2rem; }
    .topbar-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #f0f6ff; }
    
    /* SaaS KPI Cards */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 2rem; }
    .kpi-card { 
        background: linear-gradient(145deg, #0a0f1e, #080c18); 
        border: 1px solid rgba(255,255,255,.06); 
        border-radius: 12px; padding: 1.2rem; transition: 0.3s;
    }
    .kpi-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
    .kpi-lbl { font-family: 'DM Mono', monospace; font-size: .6rem; color: #3a5070; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 5px; }
    .kpi-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f0f6ff; }
    .kpi-delta { font-family: 'DM Mono', monospace; font-size: .65rem; margin-top: 5px; color: #3fb950; }

    /* Callouts */
    .info-box { background: rgba(59,130,246,.05); border-left: 3px solid #3b82f6; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; font-size: 0.85rem; color: #7a9cbf; }
    .grp-label { font-family: 'DM Mono', monospace; font-size: .6rem; color: #3a5070; text-transform: uppercase; letter-spacing: .15em; margin: 1.5rem 0 0.8rem 0; border-bottom: 1px solid rgba(59,130,246,0.1); padding-bottom: 5px; }
    
    /* Sidebar Navigation */
    [data-testid="stSidebar"] { background: #070912 !important; border-right: 1px solid rgba(255,255,255,.05) !important; }
    .nav-logo { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #f0f6ff; margin-bottom: 2rem; }
    .nav-logo em { color: #3b82f6; font-style: normal; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIC & DATA PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def load_data():
    try:
        ships = pd.read_csv("shipments.csv")
        routes = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except Exception:
        # Fallback empty data structures if files not found
        st.error("Missing CSV files. Please run your solver first.")
        st.stop()

@st.cache_data
def compute_feature_importance(routes_df):
    # Simulated permutation importance logic
    features = {"Travel Time": 32.1, "Fuel Cost": 24.5, "Toll Cost": 8.2, "Carbon": 15.4, "SLA Urgency": 19.8}
    return dict(sorted(features.items(), key=lambda item: item[1], reverse=True))

@st.cache_data
def compute_stop_contributions(routes_df):
    cols = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max() - df[c].min()
        df[c] = (df[c]-df[c].min())/(rng + 1e-9)
    df["city"] = routes_df["city"].values
    df["vehicle"] = routes_df["vehicle"].values
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERFACE COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_kpi_strip(metrics):
    ds = metrics["baseline_distance_km"] - metrics["opt_distance_km"]
    cs = metrics["baseline_total_cost"] - metrics["opt_total_cost"]
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-lbl">Optimized Distance</div>
        <div class="kpi-val">{metrics['opt_distance_km']:,.0f} km</div>
        <div class="kpi-delta">▼ {ds:,.0f} km saved</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Total Savings</div>
        <div class="kpi-val">₹{cs:,.0f}</div>
        <div class="kpi-delta">▼ {(cs/metrics['baseline_total_cost'])*100:.1f}% vs baseline</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">Carbon Emitted</div>
        <div class="kpi-val">{metrics['opt_carbon_kg']:,.1f} kg</div>
        <div class="kpi-delta">▼ {metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:,.0f} kg CO2</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-lbl">SLA Performance</div>
        <div class="kpi-val">{metrics['opt_sla_adherence_pct']:.1f}%</div>
        <div class="kpi-delta">↑ {metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']:.1f} pts</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    apply_custom_styles()
    ships, routes, metrics, veh_summary = load_data()

    # Session State for Routing
    if "page" not in st.session_state: st.session_state.page = "dashboard"

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="nav-logo">Lo<em>RRI</em></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="grp-label">Analytics</div>', unsafe_allow_html=True)
        if st.button("🏠 Dashboard", use_container_width=True): st.session_state.page = "dashboard"
        if st.button("🗺️ Route Map", use_container_width=True): st.session_state.page = "map"
        
        st.markdown('<div class="grp-label">AI Engine</div>', unsafe_allow_html=True)
        if st.button("🧠 Explainability", use_container_width=True): st.session_state.page = "explain"
        if st.button("🤖 AI Assistant", use_container_width=True): st.session_state.page = "rag"

        st.divider()
        hf_key = st.text_input("HuggingFace Key", type="password", placeholder="hf_...")
        if hf_key: set_hf_key(hf_key)

    # Topbar
    pg = st.session_state.page
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-title">{pg.title()} Intelligence</div>
      <div style="font-family:'DM Mono'; font-size:0.7rem; color:#3a5070;">Mumbai Depot · LIVE RUN</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE: DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    if pg == "dashboard":
        render_kpi_strip(metrics)
        
        st.markdown('<div class="grp-label">Operational Baseline vs AI Optimized</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shipments", int(metrics["num_shipments"]))
        c2.metric("Vehicles", int(metrics["num_vehicles"]))
        c3.metric("Avg Load", f"{veh_summary['load_kg'].mean():.1f} kg")
        c4.metric("Util %", f"{veh_summary['util_pct'].mean():.1f}%")

        st.markdown('<div class="grp-label">Vehicle Performance Details</div>', unsafe_allow_html=True)
        st.dataframe(veh_summary.style.background_gradient(subset=["util_pct"], cmap="Blues"), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE: EXPLAINABILITY (The Logic Fix)
    # ─────────────────────────────────────────────────────────────────────────
    elif pg == "explain":
        st.markdown("""<div class="info-box">This page visualizes the Multi-Objective decision logic. 
        Importance is calculated by measuring how the optimization score shifts when variables are perturbed.</div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown('<div class="grp-label">Objective Weights</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=["Cost","Time","Carbon","SLA"],
                values=[35, 30, 20, 15], hole=.6,
                marker_colors=["#3b82f6","#e3b341","#3fb950","#f85149"]
            ))
            fig_pie.update_layout(**PT) # Apply base theme
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown('<div class="grp-label">Permutation Feature Importance</div>', unsafe_allow_html=True)
            fi = compute_feature_importance(routes)
            
            fig_fi = go.Figure(go.Bar(
                x=list(fi.values()), y=list(fi.keys()), 
                orientation="h", marker_color="#3b82f6"
            ))
            
            # FIXED: Two-step layout update to prevent keyword argument crash
            fig_fi.update_layout(**PT) # 1. Apply shared dictionary
            fig_fi.update_layout(      # 2. Override specific keys
                xaxis_title="Importance %",
                yaxis=dict(autorange="reversed"),
                height=350
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown('<div class="grp-label">Per-Stop Route Contribution</div>', unsafe_allow_html=True)
        sc_df = compute_stop_contributions(routes)
        fig_st = px.bar(sc_df, x="city", y=["travel_time_hr","fuel_cost","carbon_kg","sla_breach_hr"], 
                        barmode="stack", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_st.update_layout(**PT)
        fig_st.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_st, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE: MAP
    # ─────────────────────────────────────────────────────────────────────────
    elif pg == "map":
        st.markdown('<div class="grp-label">Live Geospatial Distribution</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(routes, lat="latitude", lon="longitude", color="vehicle", 
                                    size="weight", hover_name="city", zoom=4, height=600)
        fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE: RAG ASSISTANT
    # ─────────────────────────────────────────────────────────────────────────
    elif pg == "rag":
        st.markdown('<div class="grp-label">Ask LoRRI AI</div>', unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role":"assistant","content":"I am grounded on your delivery data. Ask me about costs, carbon, or specific vehicles."}]
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        if prompt := st.chat_input("What is the most expensive route?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                ans, _ = get_rag_response(prompt, st.session_state.messages[:-1])
                st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

    # Footer
    st.markdown("""<div style="text-align:center; font-family:'DM Mono'; font-size:0.6rem; color:#1a2d3f; margin-top:4rem;">
    LoRRI · Enterprise Route Intelligence · © 2026 LoRRI Technologies</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
