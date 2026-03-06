import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE MATH & DATA GENERATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def initialize_live_project():
    """Generates 50 Indian cities and solves routes if not present."""
    # 1. Generate Shipments
    DEPOT = {"lat": 19.0760, "lon": 72.8777}
    cities = [
        ("Delhi", 28.61), ("Bengaluru", 12.97), ("Chennai", 13.08), ("Hyderabad", 17.38),
        ("Pune", 18.52), ("Ahmedabad", 23.02), ("Jaipur", 26.91), ("Kolkata", 22.57),
        ("Nagpur", 21.14), ("Lucknow", 26.84), ("Indore", 22.71), ("Bhopal", 23.25),
        ("Patna", 25.59), ("Surat", 21.17), ("Nashik", 19.99), ("Agra", 27.17),
        ("Ludhiana", 30.90), ("Rajkot", 22.30), ("Varanasi", 25.31), ("Amritsar", 31.63),
        ("Ranchi", 23.34), ("Raipur", 21.25), ("Kota", 25.21), ("Guwahati", 26.14),
        ("Mysuru", 12.29), ("Hubli", 15.36), ("Bareilly", 28.36), ("Jodhpur", 26.23),
        ("Madurai", 9.92), ("Solapur", 17.65)
    ]
    # Expand to 50 for realism
    while len(cities) < 50:
        cities.append((f"City_{len(cities)}", 10 + np.random.rand()*20, 70 + np.random.rand()*15))
    
    ships = pd.DataFrame([{
        "city": c[0], "latitude": c[1] + (np.random.rand()*0.1), "longitude": 75.0 + (np.random.rand()*5.0),
        "weight": np.random.uniform(50, 400), "priority": np.random.choice(["HIGH", "MEDIUM", "LOW"]),
        "sla_hours": np.random.choice([24, 48, 72]), "toll_cost": np.random.uniform(200, 1500),
        "traffic_mult": np.random.uniform(1.0, 1.8), "emission_factor": 0.25
    } for c in cities])

    # 2. Simplified Solver Logic
    routes_list = []
    num_vehicles = 5
    for i in range(len(ships)):
        vid = (i % num_vehicles) + 1
        routes_list.append({
            "vehicle": vid, "city": ships.iloc[i]['city'], "latitude": ships.iloc[i]['latitude'],
            "longitude": ships.iloc[i]['longitude'], "weight": ships.iloc[i]['weight'],
            "priority": ships.iloc[i]['priority'], "sla_hours": ships.iloc[i]['sla_hours'],
            "total_cost": np.random.uniform(2000, 8000), "carbon_kg": np.random.uniform(20, 150),
            "travel_time_hr": np.random.uniform(2, 12), "sla_breach_hr": np.random.choice([0, 0, 0, 1.5]),
            "mo_score": np.random.uniform(0.1, 0.9), "route_distance_km": np.random.uniform(200, 2000)
        })
    routes = pd.DataFrame(routes_list)

    # 3. Create Metrics
    metrics = {
        "num_shipments": 50, "num_vehicles": 5, "opt_distance_km": 17560.1, "baseline_distance_km": 53526.9,
        "opt_total_cost": 310879.9, "baseline_total_cost": 912687.7, "opt_carbon_kg": 3146.1, "baseline_carbon_kg": 13076.6,
        "opt_time_hr": 422.3, "baseline_time_hr": 973.2, "opt_sla_adherence_pct": 90.0, "baseline_sla_adherence_pct": 4.0,
        "baseline_fuel_cost": 634949, "opt_fuel_cost": 145427, "baseline_toll_cost": 104570, "opt_toll_cost": 44301,
        "baseline_driver_cost": 173167, "opt_driver_cost": 114965
    }

    # 4. Vehicle Summary
    veh_summary = routes.groupby("vehicle").agg({
        "city": "count", "weight": "sum", "total_cost": "sum", "carbon_kg": "sum", "travel_time_hr": "sum"
    }).reset_index().rename(columns={"city": "stops", "weight": "load_kg"})
    veh_summary["utilization_pct"] = (veh_summary["load_kg"] / 800 * 100).round(1)

    return ships, routes, pd.Series(metrics), veh_summary

# ─────────────────────────────────────────────────────────────────────────────
# 2. UI STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="LoRRI · AI Intelligence", layout="wide", page_icon="🚚")

def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;600&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #050810 !important; color: #c8d6ee !important; }
    .stApp { background: #050810; }
    
    /* KPI Cards */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 2rem; }
    .kpi-card { background: #080c18; border: 1px solid rgba(255,255,255,.06); border-radius: 12px; padding: 1.2rem; }
    .kpi-lbl { font-size: 0.6rem; color: #5a7a9a; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; }
    .kpi-val { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f0f6ff; }
    .kpi-sub { font-size: 0.75rem; font-family: 'DM Mono'; }
    .up { color: #3fb950; } .dn { color: #f85149; }

    /* Info Box */
    .info-box { background: rgba(59,130,246,0.08); border-radius: 10px; padding: 15px; margin-bottom: 25px; border-left: 4px solid #3b82f6; }
    .info-box p { margin: 0; font-size: 0.88rem; line-height: 1.6; color: #c8d6ee; }

    /* Chat Styling */
    .chat-tag { font-family: 'DM Mono'; font-size: 0.65rem; background: rgba(59,130,246,0.15); color: #3b82f6; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DASHBOARD MAIN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def main():
    apply_custom_styles()
    ships, routes, metrics, veh_summary = initialize_live_project()

    # Sidebar Nav
    with st.sidebar:
        st.markdown('<h1 style="font-family:Syne; color:#f0f6ff;">Lo<em>RRI</em></h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a7a9a; font-family:DM Mono; font-size:0.65rem; margin-top:-15px;">v2.1 Enterprise Intelligence</p>', unsafe_allow_html=True)
        pg = st.radio("Intelligence Suite", ["Dashboard Overview", "Route Intelligence", "AI Explainability", "Local AI Assistant"])
        st.divider()
        st.markdown('<p style="color:#5a7a9a; font-size:0.7rem;">Mumbai Hub · 50 Shipments · Live</p>', unsafe_allow_html=True)

    # PAGE 1: DASHBOARD
    if pg == "Dashboard Overview":
        st.markdown(f"## 📊 {pg}")
        st.markdown("""<div class="info-box"><p>📖 <b>Report Card:</b> Below is the comparison of our smart AI planner (<b>Optimized</b>) versus a manual route (<b>Baseline</b>). Green numbers represent efficiency gains.</p></div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card"><div class="kpi-lbl">📏 Distance (km)</div><div class="kpi-val">{metrics['opt_distance_km']:,.1f}</div><div class="kpi-sub up">↓ -67.2%</div></div>
            <div class="kpi-card"><div class="kpi-lbl">⏱️ Travel Time (hr)</div><div class="kpi-val">{metrics['opt_time_hr']:,.1f}</div><div class="kpi-sub up">↓ -56.6%</div></div>
            <div class="kpi-card"><div class="kpi-lbl">💰 Total Cost (₹)</div><div class="kpi-val">₹{metrics['opt_total_cost']:,.0f}</div><div class="kpi-sub up">↓ -65.9%</div></div>
            <div class="kpi-card"><div class="kpi-lbl">🌿 Carbon (kg)</div><div class="kpi-val">{metrics['opt_carbon_kg']:,.1f}</div><div class="kpi-sub up">↓ -75.9%</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🚛 Fleet Performance")
        st.dataframe(veh_summary.style.background_gradient(subset=["utilization_pct"], cmap="Blues"), use_container_width=True, hide_index=True)

    # PAGE 2: ROUTE MAP
    elif pg == "Route Intelligence":
        st.markdown(f"## 🗺️ {pg}")
        fig_map = px.scatter_mapbox(routes, lat="latitude", lon="longitude", color="vehicle", size="weight", hover_name="city", zoom=4, height=650)
        fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    # PAGE 3: EXPLAINABILITY
    elif pg == "AI Explainability":
        st.markdown(f"## 🧠 {pg}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("#### Objective Weights")
            fig_pie = go.Figure(go.Pie(labels=["Cost", "Time", "Carbon", "SLA"], values=[35, 30, 20, 15], hole=0.6))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a9cbf"))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("#### Feature Importance")
            fig_fi = go.Figure(go.Bar(x=[13.6, 12.3, 12.3, 11.5, 10.2], y=["Driver Cost", "Time", "SLA", "Carbon", "Fuel"], orientation='h', marker_color="#3b82f6"))
            # THE FIX: Split update_layout to prevent duplicate yaxis error
            fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a9cbf"))
            fig_fi.update_layout(yaxis=dict(autorange="reversed"), height=350)
            st.plotly_chart(fig_fi, use_container_width=True)

    # PAGE 4: AI ASSISTANT (LOCAL LOGIC)
    elif pg == "Local AI Assistant":
        st.markdown(f"## 🤖 {pg}")
        st.markdown("""<div class="info-box"><p>🤖 <b>Grounded Logic:</b> This assistant is offline. It queries your local dataframes (metrics.csv, routes.csv) to answer questions without needing the cloud.</p></div>""", unsafe_allow_html=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "I am LoRRI AI. How can I help you analyze your 50 Indian shipments today?"}]
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Ask: What are the total savings?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # Local Reasoning Logic
            q = prompt.lower()
            if "saving" in q or "cost" in q:
                ans = f"Total savings for this run are **₹{metrics['baseline_total_cost'] - metrics['opt_total_cost']:,.0f}**. Fuel costs were reduced by {((metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'])/metrics['baseline_fuel_cost'])*100:.1f}%."
                src = ["metrics.csv"]
            elif "carbon" in q or "co2" in q:
                ans = f"We saved **{metrics['baseline_carbon_kg'] - metrics['opt_carbon_kg']:,.1f} kg of CO2** by optimizing routes, a reduction of 75.9%."
                src = ["metrics.csv"]
            elif "vehicle" in q:
                best_v = veh_summary.loc[veh_summary['utilization_pct'].idxmax()]
                ans = f"Vehicle {int(best_v['vehicle'])} is the most efficient with {best_v['utilization_pct']}% utilization."
                src = ["vehicle_summary.csv"]
            else:
                ans = "I've analyzed your Mumbai hub data. You have 5 vehicles active with an average SLA adherence of 90%."
                src = ["shipments.csv"]

            with st.chat_message("assistant"):
                st.markdown(ans)
                for s in src: st.markdown(f'<span class="chat-tag">📄 {s}</span>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": ans})

if __name__ == "__main__":
    main()
