import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & PREMIUM THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI AS | Intelligence", layout="wide", page_icon="🚚")

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7a9cbf", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_ui():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Plus+Jakarta+Sans:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #050810 !important; color: #c8d6ee !important; }
    
    /* Sidebar Grouping */
    .sb-section { font-family: 'DM Mono'; font-size: 0.6rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.15em; margin: 1.5rem 0 0.5rem 0; }
    
    /* KPI SaaS Cards */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 2rem; }
    .kpi-card { background: #080c18; border: 1px solid rgba(255,255,255,.06); border-radius: 12px; padding: 1.5rem; border-top: 3px solid #3b82f6; }
    .kpi-lbl { font-size: 0.65rem; color: #5a7a9a; text-transform: uppercase; font-weight: 600; }
    .kpi-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f0f6ff; line-height: 1; }
    
    /* Info/Description Boxes */
    .desc-box { background: rgba(59,130,246,0.06); border-radius: 10px; padding: 20px; border-left: 4px solid #3b82f6; margin-bottom: 25px; }
    .desc-box h4 { font-family: 'Syne'; margin-top:0; color: #f0f6ff; }
    
    /* Chat bubbles */
    .chat-bubble { padding: 12px; border-radius: 10px; margin-bottom: 10px; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.05); }
    .ai-msg { background: #0d111d; }
    .user-msg { background: rgba(59,130,246,0.1); }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ships = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    metrics = pd.read_csv("metrics.csv").iloc[0]
    veh = pd.read_csv("vehicle_summary.csv")
    return ships, routes, metrics, veh

# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR (Divided into Groups & Sections)
# ─────────────────────────────────────────────────────────────────────────────
apply_ui()
ships, routes, metrics, veh_summary = load_data()

with st.sidebar:
    st.markdown('<h1 style="font-family:Syne; color:#f0f6ff;">Lo<em>RRI</em> AS</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="sb-section">📊 Analytics</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard", "Route Map", "Cost Breakdown", "Sustainability", "Explainability", "Simulator", "AI Chat (RAG)"], label_visibility="collapsed")
    
    st.markdown('<div class="sb-section">⚙️ Re-Optimization</div>', unsafe_allow_html=True)
    st.toggle("Apply Real-time Traffic", value=True)
    st.toggle("Prioritize SLA Breaches", value=True)
    
    st.markdown('<div class="sb-section">📂 System Status</div>', unsafe_allow_html=True)
    st.caption("Active Hub: Mumbai Depot")
    st.caption("Last Solved: Today 10:45 PM")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TABS & CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# TOP BAR
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.05); margin-bottom:2rem;">
    <div>
        <h2 style="font-family:Syne; margin:0;">{page}</h2>
        <p style="color:#5a7a9a; margin:0; font-size:0.8rem;">Multi-Objective Route Intelligence Platform</p>
    </div>
    <div style="text-align:right; font-family:'DM Mono'; font-size:0.7rem; color:#3fb950;">● LIVE RUN ACTIVE</div>
</div>
""", unsafe_allow_html=True)

# PAGE LOGIC
if page == "Dashboard":
    # TAB 1: OVERVIEW & DESCRIPTION
    st.markdown("""
    <div class="desc-box">
        <h4>About LoRRI AS Intelligence</h4>
        <p>This engine transforms manual logistics into a <b>Dynamic Decision System</b>. By balancing travel time, fuel costs, and carbon mandates, we achieve <b>65.9% cost reduction</b>. The dashboard compares the "Baseline" (old way) vs the "Optimized" (AI way).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-lbl">Total Savings</div><div class="kpi-val">₹{metrics['baseline_total_cost'] - metrics['opt_total_cost']:,.0f}</div></div>
        <div class="kpi-card"><div class="kpi-lbl">Optimized Distance</div><div class="kpi-val">{metrics['opt_distance_km']:,.0f} km</div></div>
        <div class="kpi-card"><div class="kpi-lbl">SLA Adherence</div><div class="kpi-val">{metrics['opt_sla_adherence_pct']:.0f}%</div></div>
        <div class="kpi-card"><div class="kpi-lbl">Carbon Saved</div><div class="kpi-val">{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:,.1f} kg</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Top Business Company Layout: Vehicle Performance")
    st.dataframe(veh_summary.style.background_gradient(subset=["utilization_pct"], cmap="Blues"), use_container_width=True)

elif page == "Route Map":
    # TAB 2: COPY IT (ROUTE MAP)
    fig_map = px.scatter_mapbox(routes, lat="latitude", lon="longitude", color="vehicle", size="weight", zoom=3.5, height=600)
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

elif page == "Cost Breakdown":
    # TAB 3: PERFECT GRAPH (WATERFALL)
    st.markdown("### Financial Analysis")
    savings = {"Fuel": 489522, "Toll": 60269, "Driver": 114965}
    fig_wf = go.Figure(go.Waterfall(
        x = list(savings.keys()) + ["Total"], 
        y = list(savings.values()) + [sum(savings.values())],
        measure = ["relative", "relative", "relative", "total"],
        decreasing = {"marker":{"color":"#3fb950"}},
        totals = {"marker":{"color":"#3b82f6"}}
    ))
    fig_wf.update_layout(**PT)
    st.plotly_chart(fig_wf, use_container_width=True)

elif page == "Sustainability":
    # TAB 4: CO2 & SLA DESCRIPTION
    st.markdown("""
    <div class="desc-box">
        <h4>🌿 What is $CO_2$ and why optimize it?</h4>
        <p>Carbon dioxide ($CO_2$) is the byproduct of fuel combustion. By shortening routes by 67%, we burn less diesel, directly reducing the environmental footprint.
        <b>SLA (Service Level Agreement)</b> is our promise to arrive within 24-72 hours. Our AI ensures 90% adherence even in high-traffic zones.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total CO2 Reduction", f"{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:.1f} kg", "▼ 75.9%")
    with c2:
        st.metric("SLA Improvement", f"{metrics['opt_sla_adherence_pct']:.0f}%", "↑ 86 pts")

elif page == "Explainability":
    # TAB 5: THINK ABOUT IT (AI LOGIC)
    st.markdown("### AI Decision Explainability")
    fig_fi = go.Figure(go.Bar(x=[13.6, 12.3, 11.5, 10.2], y=["Time", "Cost", "SLA", "Carbon"], orientation='h', marker_color="#3b82f6"))
    # FIXED TYPEERROR:
    fig_fi.update_layout(**PT)
    fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Simulator":
    # TAB 6: THINK ABOUT IT (RE-OPT)
    st.markdown("### Disruption Simulator")
    st.selectbox("Select City for Traffic Jam:", ships['city'].unique())
    if st.button("Trigger Re-optimization"):
        st.warning("Traffic threshold reached! Re-planning Vehicle 3 route...")

elif page == "AI Chat (RAG)":
    # TAB 7: RAG LIKE CHATGPT
    st.markdown("### LoRRI AI Assistant")
    if "chat" not in st.session_state: st.session_state.chat = [{"role":"ai", "content":"Hello! I'm LoRRI. Ask me anything about your Mumbai Hub run."}]
    
    for m in st.session_state.chat:
        st.markdown(f'<div class="chat-bubble {"ai-msg" if m["role"]=="ai" else "user-msg"}">{m["content"]}</div>', unsafe_allow_html=True)
        
    if prompt := st.chat_input("Ex: What are the total savings?"):
        st.session_state.chat.append({"role":"user", "content":prompt})
        # Mock RAG response
        st.session_state.chat.append({"role":"ai", "content":f"Based on the metrics.csv, your optimized run saved ₹{metrics['baseline_total_cost'] - metrics['opt_total_cost']:,.0f}."})
        st.rerun()

st.markdown('<hr><p style="text-align:center; font-family:DM Mono; font-size:0.6rem; color:#1a2d3f;">LoRRI AS v2.1 | Intelligent Logistics Framework</p>', unsafe_allow_html=True)
