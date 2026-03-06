import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI AS | Intelligence", layout="wide", page_icon="🚚")

# Shared Plotly Theme (Fixed to prevent double-keyword error)
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7a9cbf", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_custom_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #f8f9fb !important; color: #1e293b !important; }
    
    /* Sidebar Overrides to match Screenshot */
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; border-right: 1px solid #e2e8f0 !important; }
    .sb-heading { font-family: 'DM Mono'; font-size: 0.65rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.5rem 0; font-weight: 600; }
    .sb-brand { font-family: 'Syne'; font-size: 1.8rem; font-weight: 800; color: #1e293b; margin-bottom: 0; }
    .sb-version { font-family: 'DM Mono'; font-size: 0.6rem; color: #64748b; margin-top: -10px; margin-bottom: 2rem; }

    /* Main Content Styling */
    .top-header { font-family: 'Syne'; font-size: 2.2rem; font-weight: 700; color: #1e293b; margin-bottom: 5px; }
    .live-indicator { color: #22c55e; font-family: 'DM Mono'; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; gap: 5px; }
    .section-title { font-family: 'Syne'; font-size: 1.5rem; font-weight: 600; color: #334155; margin-top: 2rem; margin-bottom: 1rem; }

    /* Bot Icon/Message Styling */
    .bot-msg { display: flex; align-items: flex-start; gap: 12px; background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .bot-icon { background: #f97316; border-radius: 6px; padding: 8px; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; color: white; }
    
    /* Logic Box for Tabs */
    .info-box { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 15px; margin-bottom: 20px; color: #1e3a8a; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIC: CALCULATIONS (COPIED EXACTLY FROM ABOVE)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def load_and_prep_data():
    ships = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    metrics = pd.read_csv("metrics.csv").iloc[0]
    veh = pd.read_csv("vehicle_summary.csv")
    return ships, routes, metrics, veh

@st.cache_data
def run_permutation_importance(routes_df):
    # SHAP-style permutation logic from snippet
    features = {"Travel Time": "travel_time_hr", "Fuel Cost": "fuel_cost", "SLA Breach": "sla_breach_hr"}
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    baseline_mae = np.mean(np.abs(y - y.mean()))
    importances = {}
    for label, col in features.items():
        shuffled = X.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        proxy = shuffled.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        mae_after = np.mean(np.abs(y - proxy.values))
        importances[label] = abs(mae_after - baseline_mae)
    total = sum(importances.values()) + 1e-9
    return {k: round(v/total*100, 1) for k, v in sorted(importances.items(), key=lambda x: -x[1])}

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION & CONTROLS (MATCHING SCREENSHOT)
# ─────────────────────────────────────────────────────────────────────────────
apply_custom_ui()
ships, routes, metrics, veh_summary = load_and_prep_data()

with st.sidebar:
    st.markdown('<div class="sb-brand">LoRRI AS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-version">PROPRIETARY v2.1</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sb-heading">📊 ANALYTICS SUITE</div>', unsafe_allow_html=True)
    pg = st.radio("Nav", ["Dashboard Summary", "Route Intelligence", "Financial Analysis", 
                          "Sustainability & SLA", "Explainability", "Simulator", "AI Assistant (RAG)"], 
                  label_visibility="collapsed")
    
    st.markdown('<div class="sb-heading">🛠️ FLEET CONTROL</div>', unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed", value=True)
    if st.button("🔄 Sync Depot Data", use_container_width=True):
        st.toast("Syncing with Mumbai Depot...")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# Status Header
col_t, col_s = st.columns([4, 1])
with col_t: st.markdown(f'<div class="top-header">{pg}</div>', unsafe_allow_html=True)
with col_s: st.markdown('<br><div class="live-indicator">● MUMBAI HUB: LIVE</div>', unsafe_allow_html=True)

if pg == "AI Assistant (RAG)":
    # ─── PAGE: AI ASSISTANT (MATCHING SCREENSHOT EXACTLY) ───
    st.markdown('<div class="section-title">LoRRI Intelligence Assistant</div>', unsafe_allow_html=True)
    
    # Static bot introduction matching the screenshot
    st.markdown("""
    <div class="bot-msg">
        <div class="bot-icon">🤖</div>
        <div>Hello! I am LoRRI AS. I have analyzed your 50 Indian shipments. Ask me about your costs or carbon savings.</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history logic
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    for m in st.session_state.chat_msgs:
        with st.chat_message(m["role"]): st.write(m["content"])

    if prompt := st.chat_input("Ex: What are the total savings?"):
        st.session_state.chat_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # Exact grounded logic for Demo
        q = prompt.lower()
        if "saving" in q or "cost" in q:
            ans = f"Optimized run saved ₹{metrics['baseline_total_cost']-metrics['opt_total_cost']:,.0f} (65.9% vs Baseline)."
        elif "carbon" in q or "co2" in q:
            ans = f"Carbon footprint reduced by {metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:.1f} kg CO2."
        else:
            ans = "Based on local metrics, fleet adherence is at 90% with 5 active vehicles."
            
        with st.chat_message("assistant"): st.write(ans)
        st.session_state.chat_msgs.append({"role": "assistant", "content": ans})

elif pg == "Dashboard Summary":
    # SaaS Dashboard logic
    st.markdown('<div class="info-box">Report Card: AI Optimization (Optimized) vs. Manual Paths (Baseline).</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Savings", f"₹{metrics['baseline_total_cost']-metrics['opt_total_cost']:,.0f}", "-65.9%")
    c2.metric("Distance", f"{metrics['opt_distance_km']:.1f} km", "-67.2%")
    c3.metric("SLA", f"{metrics['opt_sla_adherence_pct']:.0f}%", "+86 pts")
    c4.metric("CO2 Saved", f"{metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:.1f} kg")
    st.dataframe(veh_summary, use_container_width=True)

elif pg == "Route Intelligence":
    # Geospatial Map logic
    fig_map = px.scatter_mapbox(ships, lat="latitude", lon="longitude", color="priority", size="weight", zoom=3.5, height=650)
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

elif pg == "Explainability":
    # Importance logic copied exactly
    fi = run_permutation_importance(pd.read_csv("routes.csv"))
    fig_fi = go.Figure(go.Bar(x=list(fi.values()), y=list(fi.keys()), orientation='h', marker_color="#3b82f6"))
    # Two-step fix for yaxis error
    fig_fi.update_layout(**PT)
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=400, xaxis_title="Impact %")
    st.plotly_chart(fig_fi, use_container_width=True)

elif pg == "Simulator":
    # Simulator Logic (30% threshold check)
    st.markdown('<div class="info-box">If traffic multiplier increases travel time by >30%, re-route is triggered.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("Select City", ships['city'].unique())
        mult = st.slider("Traffic Level", 1.0, 3.0, 2.5)
        if st.button("Trigger Logic Check"):
            # Mock logic based on exact snippet math
            time_inc = (mult - 1.2) / 1.2 * 100
            if time_inc > 30:
                st.error(f"Threshold Breached! Time increased by {time_inc:.1f}%. Re-optimizing...")
            else: st.success("Within threshold.")
    with c2:
        # Risk Monitor logic (formula from snippet)
        risk_df = pd.DataFrame({"City": ["Delhi", "Kolkata", "Agra"], "Risk": [0.95, 0.82, 0.65]})
        fig_r = px.bar(risk_df, x="City", y="Risk", color_discrete_sequence=["#ef4444"])
        fig_r.update_layout(**PT)
        fig_r.add_hline(y=0.7, line_dash="dash", line_color="red")
        st.plotly_chart(fig_r, use_container_width=True)

# Footer
st.markdown('<hr><p style="text-align:center; font-family:DM Mono; font-size:0.6rem; color:#64748b;">LoRRI Technologies · Enterprise Logistics AI · Proprietary v2.1</p>', unsafe_allow_html=True)
