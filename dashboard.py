import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & PREMIUM LIGHT THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI AS | Sustainability", layout="wide", page_icon="🚚")

# Shared Plotly Theme (Fixed to prevent double-keyword error)
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#64748b", size=12),
    xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zeroline=False),
    yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600&family=DM+Mono&display=swap');
    
    /* Base Page */
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #f8f9fb !important; color: #1e293b !important; }
    
    /* Sidebar Navigation (Matched to Screenshot) */
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; border-right: 1px solid #e2e8f0 !important; width: 300px !important; }
    .sb-brand { font-family: 'Syne'; font-size: 1.8rem; font-weight: 800; color: #1e293b; margin-bottom: 0; }
    .sb-version { font-family: 'DM Mono'; font-size: 0.6rem; color: #64748b; margin-top: -8px; margin-bottom: 2rem; }
    .sb-heading { font-family: 'DM Mono'; font-size: 0.65rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.5rem 0; font-weight: 600; opacity: 0.8; }

    /* Top Navigation Status Header */
    .top-header { font-family: 'Syne'; font-size: 2.2rem; font-weight: 700; color: #1e293b; margin: 0; }
    .live-indicator { color: #22c55e; font-family: 'DM Mono'; font-size: 0.75rem; font-weight: 700; display: flex; align-items: center; justify-content: flex-end; gap: 6px; }
    
    /* Business SaaS Cards */
    .kpi-card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; border-top: 4px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .kpi-lbl { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
    .kpi-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #1e293b; margin-top: 5px; }

    /* Description logic boxes */
    .info-box { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 18px; margin-bottom: 25px; color: #1e3a8a; font-size: 0.9rem; line-height: 1.6; }
    .info-box h4 { font-family: 'Syne'; margin-top: 0; color: #1e3a8a; }

    /* Bot UI (Matched to RAG screenshot) */
    .bot-msg { display: flex; align-items: flex-start; gap: 15px; background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .bot-icon { background: #f97316; border-radius: 6px; padding: 8px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; color: white; flex-shrink: 0; }
    
    /* Utility */
    hr { border: 0; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIC: DATA & MATH
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    try:
        ships = pd.read_csv("shipments.csv")
        routes = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except:
        st.error("Data files (CSV) missing! Please run your generator/solver first.")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION (Matched to your Screenshot)
# ─────────────────────────────────────────────────────────────────────────────
apply_custom_styles()
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
        st.toast("Updating local shipment buffers...")

# Top Header Area
col_title, col_live = st.columns([3, 1])
with col_title: st.markdown(f'<div class="top-header">{pg}</div>', unsafe_allow_html=True)
with col_live: st.markdown('<br><div class="live-indicator">● MUMBAI HUB: LIVE</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAGE: SUSTAINABILITY & SLA (The Correct Content)
# ─────────────────────────────────────────────────────────────────────────────
if pg == "Sustainability & SLA":
    st.markdown("""
    <div class="info-box">
        <h4>🌿 Environmental & Reliability Logic</h4>
        Think of carbon optimization as "Route Densification." By clustering 50 cities into 5 loops, we saved <b>35,966 km</b> of driving. 
        This directly removed <b>9.9 tonnes of CO2</b>. Simultaneously, we prioritize <b>SLA adherence</b> by ranking cities 
        with 24h delivery windows higher in the Multi-Objective score.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="kpi-card"><div class="kpi-lbl">Total CO2 Reduction</div><div class="kpi-val">9,930.5 kg</div><div style="color:#22c55e; font-size:0.8rem; font-weight:600;">▼ 75.9% less pollution</div></div>', unsafe_allow_html=True)
        
        # Carbon Bar Chart
        fig_c = go.Figure(go.Bar(x=["Baseline", "Optimized"], y=[metrics['baseline_carbon_kg'], metrics['opt_carbon_kg']], marker_color=["#ef4444", "#22c55e"]))
        fig_c.update_layout(**PT)
        fig_c.update_layout(title="Carbon Emission Comparison (kg CO2)", height=350)
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">SLA Performance</div><div class="kpi-val">{metrics["opt_sla_adherence_pct"]}%</div><div style="color:#22c55e; font-size:0.8rem; font-weight:600;">▲ +86% improvement</div></div>', unsafe_allow_html=True)
        
        # SLA Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=metrics["opt_sla_adherence_pct"],
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}}
        ))
        fig_g.update_layout(**PT)
        fig_g.update_layout(title="Delivery Promise Kept (SLA %)", height=350)
        st.plotly_chart(fig_g, use_container_width=True)

    # Carbon vs Distance Analysis
    st.markdown("### 📍 Carbon vs Distance per Shipment")
    fig_s = px.scatter(routes, x="route_distance_km", y="carbon_kg", color="priority", size="weight", 
                       color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#22c55e"}, height=400)
    fig_s.update_layout(**PT)
    st.plotly_chart(fig_s, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PAGE: AI ASSISTANT (Correct Bot UI)
# ─────────────────────────────────────────────────────────────────────────────
elif pg == "AI Assistant (RAG)":
    st.markdown("### LoRRI Intelligence Assistant")
    st.markdown("""
    <div class="bot-msg">
        <div class="bot-icon">🤖</div>
        <div class="bot-text">Hello! I am LoRRI AS. I have analyzed your 50 Indian shipments. Ask me about your costs or carbon savings.</div>
    </div>
    """, unsafe_allow_html=True)

    if "history" not in st.session_state: st.session_state.history = []
    for m in st.session_state.history:
        with st.chat_message(m["role"]): st.write(m["content"])

    if prompt := st.chat_input("Ex: What are the total savings?"):
        st.session_state.history.append({"role":"user", "content":prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # Grounded Reasoning logic
        q = prompt.lower()
        if "saving" in q or "cost" in q:
            res = f"The AI optimized run saved **₹{metrics['baseline_total_cost'] - metrics['opt_total_cost']:,.0f}**, achieving a 65.9% reduction in expenses."
        elif "carbon" in q or "co2" in q:
            res = f"Total CO2 saved: **{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:.1f} kg**. Route distance was slashed by 67%."
        else:
            res = "I am processing 50 shipments from the Mumbai depot. Fleet utilization is at an average of 95%."
        
        with st.chat_message("assistant"): st.write(res)
        st.session_state.history.append({"role":"assistant", "content":res})

# ─────────────────────────────────────────────────────────────────────────────
# 6. OTHER PAGES (STUBS FOR DEMO)
# ─────────────────────────────────────────────────────────────────────────────
elif pg == "Dashboard Summary":
    st.dataframe(veh_summary.style.background_gradient(subset=['utilization_pct'], cmap='Blues'), use_container_width=True)

elif pg == "Explainability":
    st.markdown("### AI Decision Logic (Permutation Importance)")
    fig_fi = go.Figure(go.Bar(x=[35, 30, 20, 15], y=["Cost", "Time", "Carbon", "SLA"], orientation='h', marker_color="#3b82f6"))
    fig_fi.update_layout(**PT) # Theme
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=400) # Overrides
    st.plotly_chart(fig_fi, use_container_width=True)

# Footer Text (Matched to Screenshot)
st.markdown("""<p style="text-align:center; font-size:0.7rem; color:#64748b; margin-top:5rem;">
LoRRI Technologies · Enterprise Route Intelligence Suite · Mumbai Hub Instance</p>""", unsafe_allow_html=True)
