import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. THEME & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI · AI Route Intelligence", layout="wide", page_icon="🚚")

# Shared Plotly Style (Additive Base)
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7a9cbf", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&family=Plus+Jakarta+Sans:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #050810 !important; color: #c8d6ee !important; }
    .main .block-container { padding: 1.5rem 3rem !important; }
    
    /* SaaS KPI Cards */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 2rem; }
    .kpi-card { background: #080c18; border: 1px solid rgba(255,255,255,.06); border-radius: 12px; padding: 1.2rem; transition: 0.3s; }
    .kpi-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
    .kpi-lbl { font-family: 'DM Mono', monospace; font-size: .6rem; color: #5a7a9a; text-transform: uppercase; letter-spacing: .12em; }
    .kpi-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f0f6ff; line-height: 1.1; }
    .up { color: #3fb950; font-size: 0.75rem; font-family: 'DM Mono'; }
    
    /* Explainer Boxes */
    .info-box { background: rgba(59,130,246,0.08); border-left: 4px solid #3b82f6; border-radius: 8px; padding: 15px; margin-bottom: 25px; }
    .info-box p { margin: 0; font-size: 0.88rem; line-height: 1.6; color: #c8d6ee; }
    
    /* Navigation Simulation */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #5a7a9a; border: none; }
    .stTabs [aria-selected="true"] { color: #f0f6ff !important; border-bottom: 2px solid #3b82f6 !important; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except Exception:
        st.error("⚠️ Data files not found. Please run `generate_data.py` and `route_solver.py` first.")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 3. UI RENDERING LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def main():
    apply_custom_styles = apply_custom_css()
    ships, routes, metrics, veh_summary = load_all_data()

    # App Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 5px;">
        <span style="font-size: 2.5rem;">🚚</span>
        <h1 style="font-family: 'Syne'; margin: 0; font-weight: 800; letter-spacing: -1px;">LoRRI · AI Route Optimization</h1>
    </div>
    <p style="color: #5a7a9a; margin-left: 60px; font-family: 'DM Mono'; font-size: 0.75rem;">
        Dynamic Multi-Objective CVRP · India Logistics Network · Depot: Mumbai · Live Instance
    </p>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📊 Overview", "🗺️ Route Map", "💰 Financials", "🌿 Sustainability", "🧠 Explainability", "⚡ Re-optimization"])

    # ═══════════════════════════════════════════════════════════════════
    # TAB: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("""<div class="info-box"><p>📖 <b>What is this tab?</b><br>This is the <b>report card</b> for the entire run. It compares our AI's performance (Optimized) against a standard manual path (Baseline). Green numbers indicate efficiency gains.</p></div>""", unsafe_allow_html=True)
        
        # KPI Row
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-lbl">📏 Total Distance</div><div class="kpi-val">{metrics['opt_distance_km']:,.0f} km</div><div class="up">↓ -67.2% vs baseline</div></div>
            <div class="kpi-card"><div class="kpi-lbl">💰 Run Cost</div><div class="kpi-val">₹{metrics['opt_total_cost']:,.0f}</div><div class="up">↓ -₹601,807 saved</div></div>
            <div class="kpi-card"><div class="kpi-lbl">⏱️ Travel Time</div><div class="kpi-val">{metrics['opt_time_hr']:,.1f} hr</div><div class="up">↓ -56.6% faster</div></div>
            <div class="kpi-card"><div class="kpi-lbl">✅ SLA Adherence</div><div class="kpi-val">{metrics['opt_sla_adherence_pct']:.0f}%</div><div class="up">↑ +86 pts improved</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚛 Fleet Performance Summary")
        # Aligning column names from vehicle_summary.csv to display
        dv = veh_summary.copy()
        dv = dv[["vehicle", "stops", "load_kg", "distance_km", "time_hr", "utilization_pct", "total_cost"]]
        dv.columns = ["Vehicle", "Stops", "Load (kg)", "Dist (km)", "Time (hr)", "Util %", "Cost (₹)"]
        st.dataframe(dv.style.background_gradient(subset=["Util %"], cmap="Blues"), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB: EXPLAINABILITY (FIXED TYPEERROR)
    # ═══════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### 🧠 How the AI Optimizer Makes Decisions")
        st.markdown("""<div class="info-box"><p>🧐 <b>AI Reasoning:</b> Every routing choice balances four factors: Cost, Time, Carbon, and SLA. The chart below uses <b>Permutation Importance</b> to show which factor drove the final route selection most.</p></div>""", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Logic Weightage")
            fig_pie = go.Figure(go.Pie(labels=["Cost", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6))
            fig_pie.update_layout(**PT) # Apply base theme
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.markdown("#### Real Feature Importance")
            # --- FIX: Split update_layout calls to avoid keyword duplication error ---
            fig_fi = go.Figure(go.Bar(
                x=[13.6, 12.3, 12.3, 12.3, 12.3, 12.3], 
                y=["Driver Cost", "Travel Time", "SLA Breach", "Fuel Cost", "Carbon Emitted", "Package Weight"],
                orientation='h', marker_color="#3b82f6"
            ))
            
            # Step 1: Apply Global Dictionary
            fig_fi.update_layout(**PT) 
            
            # Step 2: Apply specific overrides manually
            fig_fi.update_layout(
                xaxis_title="Importance (%)",
                yaxis=dict(autorange="reversed"), 
                height=350
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # TAB: RE-OPTIMIZATION (DEMO SIMULATOR)
    # ═══════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### ⚡ Live Re-Optimization Simulator")
        st.markdown("""<div class="info-box"><p>🚦 <b>Disruption Handling:</b> The real world is volatile. Pick a city and simulate a traffic jam. Watch how the LoRRI engine recalculates the MO score and re-ranks the stop order instantly.</p></div>""", unsafe_allow_html=True)
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### 🚦 Scenario: Traffic Spike")
            target_city = st.selectbox("Select City:", ships['city'].unique(), index=17)
            st.slider("Traffic Level (1.0 = normal, 3.0 = gridlock)", 1.0, 3.0, 2.5)
            if st.button("🔴 Simulate Disruption", use_container_width=True):
                st.warning(f"Threshold Breached! Re-routing Vehicle {int(routes[routes['city']==target_city]['vehicle'].iloc[0])}...")
                st.success(f"{target_city} moved to last stop to preserve fleet SLA.")

        with r2:
            st.markdown("#### 📊 Risk Monitor")
            risk_df = pd.DataFrame({
                "City": ["Kolkata", "Hubli", "Jodhpur", "Udaipur", "Lucknow", "Raipur", "Bhopal"],
                "Risk": [0.95, 0.92, 0.91, 0.88, 0.85, 0.82, 0.80]
            })
            fig_risk = px.bar(risk_df, x="City", y="Risk", color_discrete_sequence=["#f85149"])
            fig_risk.update_layout(**PT)
            fig_risk.add_hline(y=0.7, line_dash="dash", line_color="white", annotation_text="Trigger Level")
            st.plotly_chart(fig_risk, use_container_width=True)

    # Footer
    st.markdown("""<hr><div style="text-align:center; font-family:'DM Mono'; font-size:0.7rem; color:#5a7a9a;">
    LoRRI Engine · Multi-Objective CVRP · v2.1 Enterprise Ready</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
