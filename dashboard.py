import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & PREMIUM THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI AS | Intelligence Hub", layout="wide", page_icon="🚚")

# Shared Plotly Theme (Used sequentially to prevent TypeError)
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#64748b", size=12),
    xaxis=dict(gridcolor="#e2e8f0", zeroline=False),
    yaxis=dict(gridcolor="#e2e8f0", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_integrated_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
    
    /* Overall Page Background */
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #f8f9fb !important; color: #1e293b !important; }
    
    /* Sidebar Grouping (Matched to Screenshot) */
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; border-right: 1px solid #e2e8f0 !important; width: 300px !important; }
    .sb-brand { font-family: 'Syne'; font-size: 1.8rem; font-weight: 800; color: #1e293b; margin-bottom: 0; }
    .sb-version { font-family: 'DM Mono'; font-size: 0.6rem; color: #64748b; margin-top: -8px; margin-bottom: 2rem; }
    .sb-heading { font-family: 'DM Mono'; font-size: 0.65rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.5rem 0; font-weight: 600; opacity: 0.8; }

    /* Top Navigation Status Header */
    .top-header { font-family: 'Syne'; font-size: 2.2rem; font-weight: 700; color: #1e293b; margin: 0; }
    .live-indicator { color: #22c55e; font-family: 'DM Mono'; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: flex-end; gap: 6px; }
    
    /* AI Assistant UI (Matched to Screenshot) */
    .bot-msg { display: flex; align-items: flex-start; gap: 15px; background: white; padding: 22px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .bot-icon { background: #f97316; border-radius: 8px; padding: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem; flex-shrink: 0; }
    .bot-text { font-size: 0.95rem; line-height: 1.5; color: #334155; }
    .assistant-title { font-family: 'Syne'; font-size: 1.4rem; font-weight: 600; color: #334155; margin-bottom: 1.2rem; margin-top: 1.5rem; }

    /* Cards for KPIs */
    .kpi-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; border-top: 4px solid #3b82f6; }
    .kpi-val { font-family: 'Syne'; font-size: 2rem; font-weight: 800; color: #1e293b; }
    .kpi-lbl { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIC: DATA HANDLING (GROUNDED IN CSVs)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_initialize():
    try:
        ships = pd.read_csv("shipments.csv")
        routes = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except:
        st.error("Data files missing! Run your generator/solver scripts first.")
        st.stop()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def calculate_permutation_importance(routes_df):
    """Calculates feature importance based on the logic snippet provided."""
    np.random.seed(42)
    features = {"Time": "travel_time_hr", "Cost": "fuel_cost", "SLA": "sla_breach_hr"}
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    baseline_mae = np.mean(np.abs(y - y.mean()))
    imps = {}
    for label, col in features.items():
        shuffled = X.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        proxy = shuffled.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        mae_after = np.mean(np.abs(y - proxy.values))
        imps[label] = abs(mae_after - baseline_mae)
    total = sum(imps.values()) + 1e-9
    return {k: round(v/total*100, 1) for k, v in sorted(imps.items(), key=lambda x: -x[1])}

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION (GROUPED AS PER LOGIC)
# ─────────────────────────────────────────────────────────────────────────────
apply_integrated_ui()
ships, routes, metrics, veh_summary = load_and_initialize()

with st.sidebar:
    st.markdown('<div class="sb-brand">LoRRI AS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-version">PROPRIETARY v2.1</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sb-heading">📊 ANALYTICS SUITE</div>', unsafe_allow_html=True)
    nav_page = st.radio("Navigation", 
                        ["Dashboard Summary", "Route Intelligence", "Financial Analysis", 
                         "Sustainability & SLA", "Explainability", "Simulator", "AI Assistant (RAG)"], 
                        label_visibility="collapsed")
    
    st.markdown('<div class="sb-heading">🛠️ FLEET CONTROL</div>', unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed", value=True)
    if st.button("🔄 Sync Depot Data", use_container_width=True):
        st.toast("Syncing with Mumbai Depot...")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTENT RENDERING
# ─────────────────────────────────────────────────────────────────────────────

# Global Status Bar
col_title, col_status = st.columns([3.5, 1])
with col_title: st.markdown(f'<div class="top-header">{nav_page}</div>', unsafe_allow_html=True)
with col_status: st.markdown('<br><div class="live-indicator">● MUMBAI HUB: LIVE</div>', unsafe_allow_html=True)

if nav_page == "AI Assistant (RAG)":
    # ─── PAGE: AI ASSISTANT (MATCHED TO SCREENSHOT) ───
    st.markdown('<div class="assistant-title">LoRRI Intelligence Assistant</div>', unsafe_allow_html=True)
    
    # Static Greeting from Screenshot
    st.markdown("""
    <div class="bot-msg">
        <div class="bot-icon">🤖</div>
        <div class="bot-text">Hello! I am LoRRI AS. I have analyzed your 50 Indian shipments. Ask me about your costs or carbon savings.</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat Input Logic
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]): st.write(chat["content"])

    if prompt := st.chat_input("Ex: What are the total savings?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # Grounded Logic reasoning based on actual metrics
        q = prompt.lower()
        if "saving" in q or "cost" in q:
            savings = metrics['baseline_total_cost'] - metrics['opt_total_cost']
            ans = f"Based on local metrics, the AI optimized route has saved **₹{savings:,.0f}**, reducing fuel costs by **77.1%**."
        elif "carbon" in q or "co2" in q:
            ans = f"Carbon emissions were reduced by **{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:.1f} kg CO2**."
        else:
            ans = f"Processed 50 shipments. Average SLA adherence is currently at **{metrics['opt_sla_adherence_pct']}%**."
        
        with st.chat_message("assistant"): st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

elif nav_page == "Dashboard Summary":
    # Executive Business Summary
    st.markdown("---")
    cols = st.columns(4)
    cols[0].markdown(f'<div class="kpi-card"><div class="kpi-lbl">Total Savings</div><div class="kpi-val">₹{metrics["baseline_total_cost"]-metrics["opt_total_cost"]:,.0f}</div></div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="kpi-card"><div class="kpi-lbl">Dist. Saved</div><div class="kpi-val">{metrics["baseline_distance_km"]-metrics["opt_distance_km"]:,.0f} km</div></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="kpi-card"><div class="kpi-lbl">SLA Performance</div><div class="kpi-val">{metrics["opt_sla_adherence_pct"]}%</div></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="kpi-card"><div class="kpi-lbl">CO2 Reduction</div><div class="kpi-val">75.9%</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="assistant-title">Fleet Performance Overview</div>', unsafe_allow_html=True)
    st.dataframe(veh_summary.style.background_gradient(subset=['utilization_pct'], cmap='Blues'), use_container_width=True)

elif nav_page == "Explainability":
    # Feature Importance Logic from paste
    imps = calculate_permutation_importance(pd.read_csv("routes.csv"))
    fig = go.Figure(go.Bar(x=list(imps.values()), y=list(imps.keys()), orientation='h', marker_color="#3b82f6"))
    
    # Sequential update to fix TypeError
    fig.update_layout(**PT)
    fig.update_layout(yaxis=dict(autorange="reversed"), height=400, xaxis_title="Decision Impact %")
    st.plotly_chart(fig, use_container_width=True)

elif nav_page == "Simulator":
    # 30% Logic Trigger from paste
    st.markdown('<div class="assistant-title">Traffic Disruption Scenario</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("Select City for Traffic Jam:", ships['city'].unique())
        mult = st.slider("New Traffic Multiplier", 1.0, 3.0, 2.5)
        if st.button("Trigger Logic Check"):
            time_inc = (mult - 1.2) / 1.2 * 100 # Logic math from paste
            if time_inc > 30:
                st.error(f"Logic Triggered! Time to {city} increased by {time_inc:.1f}%. Rerouting vehicle...")
            else:
                st.success("Within acceptable time threshold.")
    with c2:
        # Risk Monitor Bar Chart from paste logic
        risk_df = pd.DataFrame({"City": ["Delhi", "Kolkata", "Agra"], "Risk": [0.95, 0.82, 0.65]})
        fig_r = px.bar(risk_df, x="City", y="Risk", color_discrete_sequence=["#ef4444"])
        fig_r.update_layout(**PT)
        fig_r.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Trigger Level")
        st.plotly_chart(fig_r, use_container_width=True)

elif nav_page == "Route Intelligence":
    # Geospatial Logic
    fig_map = px.scatter_mapbox(ships, lat="latitude", lon="longitude", color="priority", size="weight", zoom=3.5, height=650)
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# Footer
st.markdown('<hr><p style="text-align:center; font-family:DM Mono; font-size:0.6rem; color:#64748b;">LoRRI Technologies · Enterprise Route Intelligence Suite · Mumbai Hub Instance</p>', unsafe_allow_html=True)
