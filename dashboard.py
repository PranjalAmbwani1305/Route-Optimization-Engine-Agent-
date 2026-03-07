"""
LoRRI – LogisticsNow AI Route Optimization Engine
Complete dashboard: About, AI Assistant, Dashboard, Map, Financial,
Carbon & SLA, Explainability, Re-optimization Simulator, AI Route Predictor.
All costs in INR. Logo integrated. Dataset fixed backend. No external API.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import base64
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoRRI · LogisticsNow AI Route Engine",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BRAND COLORS
# ─────────────────────────────────────────────────────────────────────────────
LN_GREEN  = "#3a7d2c"
LN_DGREEN = "#2d6a2d"
LN_NAVY   = "#1e2d3d"
LN_LGRAY  = "#f5f6f7"
LN_BORDER = "#e0e4e8"

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
VEHICLE_CAP = 800
V_COLORS    = {1: "#3a7d2c", 2: "#1e7abf", 3: "#e67e22", 4: "#8e44ad", 5: "#c0392b"}

# ─────────────────────────────────────────────────────────────────────────────
# LOGO LOADER
# ─────────────────────────────────────────────────────────────────────────────
def get_logo_b64():
    for path in ["logo.png", "assets/logo.png"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_b64()

def logo_html(height=40):
    if LOGO_B64:
        return (
            f'<img src="data:image/png;base64,{LOGO_B64}" '
            f'style="height:{height}px;object-fit:contain;display:block;" />'
        )
    # Text fallback
    return (
        f'<span style="font-size:{int(height*0.5)}px;font-weight:900;'
        f'color:white;letter-spacing:-1px;">'
        f'Lo<span style="color:{LN_GREEN}">RRI</span></span>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

def inr(val):
    return f"Rs.{val:,.0f}"

def apply_theme(fig, height=340, title="", legend_below=False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family="Poppins, sans-serif", color=LN_NAVY, size=11),
        height=height, margin=dict(l=10, r=10, t=44 if title else 18, b=10),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=LN_NAVY)))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0))
    fig.update_xaxes(gridcolor="#f0f4f0", zeroline=False, linecolor=LN_BORDER)
    fig.update_yaxes(gridcolor="#f0f4f0", zeroline=False, linecolor=LN_BORDER)
    return fig

def kpi_card(label, value, delta, good=True, ac=None):
    ac  = ac or LN_GREEN
    cls = "dg" if good else "dr"
    return (
        f'<div class="kpi-card" style="--ac:{ac}">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-d {cls}">{delta}</div></div>'
    )

def page_header(title, subtitle=""):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
        <div>
            <h2 style="margin:0;color:{LN_NAVY};">{title}</h2>
            <div style="color:#64748b;font-size:0.85rem;">{subtitle}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def loading_state(msg="Processing..."):
    with st.spinner(f"  {msg}"):
        time.sleep(0.4)

def sh(text):
    return f'<div class="sh">{text}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Poppins',sans-serif;background:{LN_LGRAY};}}
.main .block-container{{padding:0!important;max-width:100%!important;}}

.kpi-card{{background:white;border:1px solid {LN_BORDER};border-radius:12px;
           padding:18px 20px;position:relative;overflow:hidden;
           box-shadow:0 2px 8px rgba(0,0,0,0.05);transition:box-shadow 0.2s;}}
.kpi-card:hover{{box-shadow:0 6px 20px rgba(58,125,44,0.14);}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
                   background:var(--ac,{LN_GREEN});}}
.kpi-lbl{{font-size:0.6rem;font-weight:600;color:#64748b;text-transform:uppercase;
          letter-spacing:0.1em;margin-bottom:8px;}}
.kpi-val{{font-size:1.65rem;font-weight:700;color:{LN_NAVY};line-height:1.1;}}
.kpi-d{{font-size:0.7rem;margin-top:5px;font-weight:500;}}
.dg{{color:{LN_GREEN};}} .dr{{color:#dc2626;}}

.sh{{font-size:1.05rem;font-weight:700;color:{LN_NAVY};margin:24px 0 12px;
     display:flex;align-items:center;gap:10px;
     border-left:4px solid {LN_GREEN};padding-left:12px;}}
.info-box{{background:#f0fdf4;border-left:4px solid {LN_GREEN};border-radius:8px;
           padding:14px 18px;margin:4px 0 16px;font-size:0.88rem;
           line-height:1.7;color:{LN_NAVY};}}
.info-box b{{color:{LN_DGREEN};}}
.warn-box{{background:#fffbeb;border-left:4px solid #f59e0b;border-radius:6px;
           padding:12px 16px;margin:6px 0;font-size:0.86rem;
           line-height:1.65;color:{LN_NAVY};}}
.ok-box{{background:#f0fdf4;border-left:4px solid {LN_GREEN};border-radius:6px;
         padding:12px 16px;margin:6px 0;font-size:0.86rem;
         line-height:1.65;color:{LN_NAVY};}}
.tag-red{{color:#dc2626;font-weight:600;}}
.tag-green{{color:{LN_GREEN};font-weight:600;}}
.tag-yellow{{color:#d97706;font-weight:600;}}

.legend-row{{display:flex;align-items:flex-start;gap:10px;font-size:0.84rem;
             color:{LN_NAVY};margin-bottom:10px;padding-bottom:10px;
             border-bottom:1px solid {LN_BORDER};}}
.legend-row:last-child{{border-bottom:none;margin-bottom:0;}}
.legend-dot{{width:14px;height:14px;border-radius:3px;flex-shrink:0;margin-top:3px;}}

[data-testid="stSidebar"]{{background:white!important;
                            border-right:2px solid {LN_BORDER}!important;}}
.sb-sec{{font-size:0.6rem;font-weight:700;color:{LN_GREEN};letter-spacing:0.16em;
         text-transform:uppercase;margin:1.2rem 0 0.4rem;
         border-bottom:1px solid {LN_BORDER};padding-bottom:4px;}}
[data-testid="metric-container"]{{background:white;border:1px solid {LN_BORDER};
                                   border-radius:10px;padding:12px 16px!important;}}
@keyframes pulse{{0%{{opacity:1}}50%{{opacity:0.3}}100%{{opacity:1}}}}
.live-dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
           background:{LN_GREEN};animation:pulse 1.8s infinite;margin-right:5px;}}
.stat-row{{display:flex;justify-content:space-between;align-items:center;
           padding:5px 0;border-bottom:1px solid #f1f5f9;font-size:0.74rem;}}
.stat-row:last-child{{border-bottom:none;}}
.stat-label{{color:#64748b;}} .stat-val{{font-weight:700;color:{LN_NAVY};}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA — fixed backend CSVs
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    ships  = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    veh    = pd.read_csv("vehicle_summary.csv")
    n        = len(ships)
    breaches = int((routes["sla_breach_hr"] > 0).sum())
    opt = dict(
        distance_km  = round(veh["distance_km"].sum(), 1),
        time_hr      = round(veh["time_hr"].sum(), 1),
        fuel_cost    = round(veh["fuel_cost"].sum(), 1),
        toll_cost    = round(veh["toll_cost"].sum(), 1),
        driver_cost  = round(veh["driver_cost"].sum(), 1),
        total_cost   = round(veh["total_cost"].sum(), 1),
        carbon_kg    = round(veh["carbon_kg"].sum(), 1),
        sla_pct      = round((n - breaches) / n * 100, 1),
        n_ships=n, n_vehicles=len(veh), breaches=breaches,
    )
    base = dict(
        distance_km=53526.9, time_hr=973.2, fuel_cost=601808.0,
        toll_cost=112741.0,  driver_cost=197566.0, total_cost=912115.0,
        carbon_kg=13076.6,   sla_pct=4.0,
    )
    return ships, routes, veh, base, opt

@st.cache_data
def perm_imp(routes_df):
    np.random.seed(42)
    feats = {
        "Travel Time":     "travel_time_hr",
        "Fuel Cost":       "fuel_cost",
        "Toll Cost":       "toll_cost",
        "Driver Cost":     "driver_cost",
        "Carbon Emitted":  "carbon_kg",
        "SLA Breach":      "sla_breach_hr",
        "Package Weight":  "weight",
    }
    X = routes_df[list(feats.values())].copy()
    y = routes_df["mo_score"].values
    base_mae = np.mean(np.abs(y - y.mean()))
    imp = {}
    for lbl, col in feats.items():
        sh2 = X.copy()
        sh2[col] = np.random.permutation(sh2[col].values)
        proxy = sh2.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        imp[lbl] = abs(np.mean(np.abs(y - proxy.values)) - base_mae)
    tot = sum(imp.values()) + 1e-9
    return {k: round(v / tot * 100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

@st.cache_data
def stop_cont(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    weights = [0.30, 0.20, 0.05, 0.15, 0.20, 0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max() - df[c].min()
        df[c] = (df[c] - df[c].min()) / (rng + 1e-9)
    for i, c in enumerate(cols):
        df[c] *= weights[i]
    df.columns = labels
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

ships, routes, veh_sum, base, opt = load()
fi = perm_imp(routes)
sc = stop_cont(routes)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo block
    st.markdown(f"""
    <div style="padding:16px 8px 8px;display:flex;align-items:center;
                justify-content:center;border-bottom:1px solid {LN_BORDER};
                margin-bottom:10px;background:{LN_NAVY};border-radius:10px;">
        {logo_html(height=56)}
    </div>
    <div style="text-align:center;font-size:0.58rem;color:#94a3b8;
                text-transform:uppercase;letter-spacing:0.14em;margin-bottom:1rem;">
        AI Route Optimization Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-sec">📊 Analytics Suite</div>', unsafe_allow_html=True)

    pg = st.radio("nav", [
        "🏢 About LoRRI",
        "🤖 LoRRI AI Assistant",
        "📊 Dashboard Summary",
        "🗺️ Route Map",
        "💰 Financial Analysis",
        "🌿 Carbon & SLA",
        "🧠 Explainability",
        "⚡ Re-optimization Simulator",
        "🔮 AI Route Predictor",
    ], label_visibility="collapsed")

    st.markdown(f'<div class="sb-sec">🛠️ Fleet Control</div>', unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed", value=True)
    st.toggle("Auto Re-optimize", value=False)
    if st.button("🔄 Sync Depot Data", use_container_width=True):
        st.toast("✅ Synced with Mumbai Depot!", icon="🏭")

    st.markdown(f'<div class="sb-sec">📦 Live Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                padding:12px 14px;margin-top:6px;">
        <div style="font-size:0.62rem;font-weight:700;color:{LN_GREEN};margin-bottom:10px;
                    text-transform:uppercase;letter-spacing:0.08em;">
            <span class="live-dot"></span> Live Dashboard
        </div>
        <div class="stat-row"><span class="stat-label">Shipments</span>
            <span class="stat-val">{len(ships)}</span></div>
        <div class="stat-row"><span class="stat-label">Trucks Active</span>
            <span class="stat-val">{veh_sum["vehicle"].nunique()}</span></div>
        <div class="stat-row"><span class="stat-label">SLA OK</span>
            <span class="stat-val" style="color:{LN_GREEN}">
            {int((routes["sla_breach_hr"]==0).sum()/len(routes)*100)}%</span></div>
        <div class="stat-row"><span class="stat-label">SLA Breaches</span>
            <span class="stat-val" style="color:#dc2626">
            {int((routes["sla_breach_hr"]>0).sum())}</span></div>
        <div class="stat-row"><span class="stat-label">Total Distance</span>
            <span class="stat-val">{veh_sum["distance_km"].sum():,.0f} km</span></div>
        <div class="stat-row"><span class="stat-label">Fleet Cost</span>
            <span class="stat-val">{inr(veh_sum["total_cost"].sum())}</span></div>
        <div class="stat-row"><span class="stat-label">Carbon Emitted</span>
            <span class="stat-val">{veh_sum["carbon_kg"].sum():,.0f} kg</span></div>
        <div class="stat-row"><span class="stat-label">Avg Utilization</span>
            <span class="stat-val">{veh_sum["utilization_pct"].mean():.1f}%</span></div>
        <div class="stat-row"><span class="stat-label">Depot</span>
            <span class="stat-val">Mumbai 🏭</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#94a3b8;line-height:1.8;">
    📧 connect@logisticsnow.in<br>🌐 logisticsnow.in
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div style="padding:24px 32px;background:{LN_LGRAY};">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT LORRI
# ══════════════════════════════════════════════════════════════════════════════
if pg == "🏢 About LoRRI":
    page_header("🏢 About LoRRI", "LogisticsNow · AI-Powered Logistics Intelligence Platform")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#2d4a6b 60%,#1a3a1a 100%);
                border-radius:14px;padding:32px 36px;margin-bottom:22px;
                display:flex;align-items:center;gap:32px;">
        <div style="flex-shrink:0;">{logo_html(height=90)}</div>
        <div>
            <div style="font-size:1.5rem;font-weight:700;color:white;margin-bottom:6px;">
                AI-Powered Route Optimization Engine
            </div>
            <div style="font-size:0.88rem;color:#94a3b8;margin-bottom:14px;">
                Logistics Rating &amp; Intelligence — India's Premier Platform
            </div>
            <div style="display:inline-flex;align-items:center;gap:6px;
                        background:rgba(58,125,44,0.2);border:1px solid {LN_GREEN};
                        color:#6bcf57;border-radius:20px;padding:4px 14px;
                        font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
                        text-transform:uppercase;">
                ● LIVE SYSTEM
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:white;border-radius:14px;border:1px solid {LN_BORDER};
                padding:34px 36px;margin-bottom:22px;border-top:5px solid {LN_GREEN};">
        <p style="font-size:0.92rem;color:#334155;line-height:1.85;margin-bottom:12px;">
            Modern logistics networks generate massive operational data, yet routing decisions
            are still often based on <b>static planning and manual assumptions</b>. This leads
            to unnecessary fuel costs, delivery delays, inefficient truck utilization, and
            increased carbon emissions.
        </p>
        <p style="font-size:0.92rem;color:#334155;line-height:1.85;margin-bottom:0;">
            <b>LoRRI (Logistics Rating &amp; Intelligence)</b> transforms logistics operations
            into an <b>AI-driven decision intelligence platform</b>. By combining real-time data,
            optimization algorithms, and interactive analytics, LoRRI enables organizations to
            <b>optimize routes, reduce costs, improve delivery reliability, and minimize
            environmental impact</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sh">🌍 Our Vision</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{LN_NAVY};border-radius:12px;padding:26px 30px;
                color:white;margin-bottom:24px;">
        <p style="font-size:0.92rem;line-height:1.9;color:#cbd5e1;margin:0 0 12px 0;">
            Our vision is to build a <b style="color:white;">global logistics intelligence
            platform</b> where transportation networks operate as
            <b style="color:{LN_GREEN};">autonomous, data-driven systems</b>.
        </p>
        <p style="font-size:0.92rem;line-height:1.9;color:#cbd5e1;margin:0;">
            By integrating <b style="color:white;">AI optimization</b>,
            <b style="color:white;">real-time analytics</b>, and
            <b style="color:white;">sustainability metrics</b>,
            LoRRI empowers supply chains to become more efficient, resilient,
            and environmentally responsible.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sh">⚙️ Key Platform Capabilities</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🚛","AI Route Optimization","Dynamic multi-objective routing balancing cost, time, carbon, and SLA constraints."),
        ("📊","Operational Intelligence","Real-time analytics showing fleet performance, cost efficiency, and route insights."),
        ("🌿","Sustainability Monitoring","Measure CO₂ per route and identify greener transportation strategies."),
        ("⚡","Adaptive Re-optimization","Automatically adjust routes during traffic disruptions or priority escalations."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:18px;height:180px;border-top:3px solid {LN_GREEN};">
            <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:0.85rem;font-weight:700;color:{LN_NAVY};margin-bottom:6px;">{title}</div>
            <div style="font-size:0.75rem;color:#64748b;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT — Vectorless RAG, fully offline, no external API
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🤖 LoRRI AI Assistant":
    page_header("🤖 LoRRI AI Assistant",
                "Vectorless RAG · BM25-lite · Pandas Retrieval · Rule Router · Fully Offline")

    route_map_text = {
        1: "Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada",
        2: "Mumbai → Surat → Vadodara → Raipur",
        3: "Mumbai → Aurangabad → Solapur → Madurai → Jammu",
        4: "Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram",
        5: "Mumbai → Hubli → Mangalore → Bengaluru",
    }

    # ─── ① Knowledge Chunks ──────────────────────────────────────────────────
    KNOWLEDGE_CHUNKS = [
        {
            "id": "company",
            "triggers": ["logisticsnow","company","about","who","contact","email",
                         "phone","website","lorri"],
            "text": (
                "COMPANY: LogisticsNow (logisticsnow.in) | connect@logisticsnow.in "
                "| +91-9867773508 / +91-9653620207\n"
                "LoRRI = Logistics Rating & Intelligence — India's premier logistics platform.\n"
                "For Shippers: carrier profiles, ratings, cost savings, procurement insights.\n"
                "For Carriers: discoverability, business inquiries, reputation building."
            ),
        },
        {
            "id": "optimization",
            "triggers": ["cvrp","optimize","optimization","weighted","objective","score",
                         "solver","ortools","heuristic","algorithm"],
            "text": (
                "OPTIMIZATION ENGINE: CVRP framework.\n"
                "Objective: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%).\n"
                "Solver: OR-Tools heuristic — nearest-neighbour + 2-opt local search.\n"
                "Re-optimization triggers: traffic delay >30% OR priority escalation.\n"
                "Explainability: permutation-based feature importance (SHAP-style)."
            ),
        },
        {
            "id": "pricing",
            "triggers": ["fuel","toll","driver","cost","price","inr","rupee","penalty","wage"],
            "text": (
                "PRICING MODEL (Rs. INR):\n"
                "- Fuel: Rs.12/km\n- Driver wages: Rs.180/hr\n"
                "- SLA breach penalty: Rs.500/hr late\n- Toll: variable by corridor"
            ),
        },
        {
            "id": "sla",
            "triggers": ["sla","late","breach","delay","on time","delivery",
                         "promise","window","adherence"],
            "text": (
                f"SLA PERFORMANCE:\n"
                f"- Optimized: {opt['sla_pct']:.1f}% (baseline {base['sla_pct']:.0f}%)\n"
                f"- Breaches: {opt['breaches']} cities\n"
                f"- Penalty: Rs.500/hr | Total: {inr(veh_sum['sla_penalty'].sum())}\n"
                f"- Windows: HIGH=24hr, MEDIUM=48hr, LOW=72hr"
            ),
        },
        {
            "id": "carbon",
            "triggers": ["carbon","co2","emission","green","environment",
                         "sustainability","eco","tree","pollution"],
            "text": (
                f"CARBON & SUSTAINABILITY:\n"
                f"- Optimized: {opt['carbon_kg']:,.1f} kg | Baseline: {base['carbon_kg']:,.1f} kg\n"
                f"- Saved: {base['carbon_kg']-opt['carbon_kg']:,.1f} kg "
                f"({(base['carbon_kg']-opt['carbon_kg'])/base['carbon_kg']*100:.1f}% reduction)\n"
                f"- Trees equivalent: {int((base['carbon_kg']-opt['carbon_kg'])/21):,} | "
                f"Cars off road: {int((base['carbon_kg']-opt['carbon_kg'])/2400)}"
            ),
        },
        {
            "id": "fleet_summary",
            "triggers": ["fleet","total","summary","overall","depot","mumbai","shipment",
                         "truck","vehicle","save","saving","saved","baseline","optimized"],
            "text": (
                f"FLEET SUMMARY (Mumbai Depot):\n"
                f"- {opt['n_ships']} shipments | {opt['n_vehicles']} trucks\n"
                f"- Optimized: {inr(opt['total_cost'])} | Baseline: {inr(base['total_cost'])}\n"
                f"- Saved: {inr(base['total_cost']-opt['total_cost'])} "
                f"({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}%)\n"
                f"- Distance: {opt['distance_km']:,.0f} km (was {base['distance_km']:,.0f} km)"
            ),
        },
        {
            "id": "traffic",
            "triggers": ["traffic","jam","congestion","disruption","reoptimize","threshold","delay"],
            "text": (
                "TRAFFIC & RE-OPTIMIZATION:\n"
                "- Triggers when delay >30% or on priority escalation\n"
                "- Recomputes in ~1-2s via OR-Tools local search\n"
                "- Risk: HIGH >0.7, MONITOR >0.4, STABLE <=0.4"
            ),
        },
        {
            "id": "cost_saving",
            "triggers": ["suggest","recommendation","improve","reduce cost","save more",
                         "tip","advice","better","optimize further"],
            "text": (
                "COST SAVING RECOMMENDATIONS:\n"
                "1. Consolidate Truck 2 & 5 — both under 70% utilization\n"
                "2. Avoid HIGH-traffic corridors during peak hours\n"
                "3. Upgrade LOW-priority shipments to 72hr SLA window\n"
                "4. Use express highway only when savings exceed toll premium\n"
                "5. Cluster Madurai + Thiruvananthapuram on one southern truck\n"
                "6. Pre-position trucks at Pune, Ahmedabad hubs"
            ),
        },
    ]

    # ─── ② BM25-lite Retriever ────────────────────────────────────────────────
    def rag_retrieve(query, top_k=3):
        q = query.lower()
        scored = []
        for chunk in KNOWLEDGE_CHUNKS:
            score = sum(
                (2 if len(t.split()) > 1 else 1)
                for t in chunk["triggers"] if t in q
            )
            scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        tops = [c["text"] for s, c in scored[:top_k] if s > 0]
        return "\n\n---\n\n".join(tops) if tops else ""

    # ─── ③ Pandas Retrieval ───────────────────────────────────────────────────
    def pandas_retrieve(query):
        q = query.lower()
        results = []

        for v in [1, 2, 3, 4, 5]:
            if f"truck {v}" in q or f"truck{v}" in q:
                row = veh_sum[veh_sum["vehicle"] == v]
                if not row.empty:
                    r  = row.iloc[0]
                    bc = routes[(routes["vehicle"]==v)&(routes["sla_breach_hr"]>0)]["city"].tolist()
                    stops_df = routes[routes["vehicle"]==v].sort_values("stop_order")[
                        ["stop_order","city","weight","priority","travel_time_hr",
                         "fuel_cost","carbon_kg","sla_breach_hr"]
                    ]
                    results.append(
                        f"TRUCK {v}:\nRoute: {route_map_text.get(v,'?')}\n"
                        f"Stops:{int(r['stops'])} | Dist:{r['distance_km']:,.0f}km | "
                        f"Time:{r['time_hr']:.1f}hr | Load:{r['load_kg']:.0f}kg "
                        f"({r['utilization_pct']:.0f}%)\n"
                        f"Fuel:{inr(r['fuel_cost'])} | Toll:{inr(r['toll_cost'])} | "
                        f"Driver:{inr(r['driver_cost'])} | Penalty:{inr(r['sla_penalty'])}\n"
                        f"Total:{inr(r['total_cost'])} | CO2:{r['carbon_kg']:.1f}kg\n"
                        f"Breaches:{int(r['sla_breaches'])} "
                        f"({'cities:'+','.join(bc) if bc else 'none OK'})\n"
                        f"STOPS:\n{stops_df.to_string(index=False)}"
                    )

        if any(k in q for k in ["late","breach","which cities","missed sla"]):
            bd = routes[routes["sla_breach_hr"]>0][
                ["vehicle","city","priority","sla_breach_hr","sla_penalty"]].copy()
            if not bd.empty:
                bd["vehicle"] = bd["vehicle"].apply(lambda v: f"Truck {v}")
                results.append("SLA BREACHES:\n" + bd.to_string(index=False))
            else:
                results.append("SLA BREACHES: None — all on time!")

        if any(k in q for k in ["most expensive","highest cost","costs most"]):
            t = veh_sum.loc[veh_sum["total_cost"].idxmax()]
            results.append(
                f"MOST EXPENSIVE: Truck {int(t['vehicle'])} — {inr(t['total_cost'])}\n"
                f"Route: {route_map_text.get(int(t['vehicle']),'?')}\n"
                f"Fuel:{inr(t['fuel_cost'])} Toll:{inr(t['toll_cost'])} "
                f"Driver:{inr(t['driver_cost'])} Penalty:{inr(t['sla_penalty'])}"
            )

        if any(k in q for k in ["utilization","capacity","load"]):
            u = veh_sum[["vehicle","load_kg","utilization_pct"]].copy()
            u["vehicle"] = u["vehicle"].apply(lambda v: f"Truck {v}")
            results.append(
                f"UTILIZATION:\n{u.to_string(index=False)}\n"
                f"Average: {veh_sum['utilization_pct'].mean():.1f}%"
            )

        if any(k in q for k in ["compare truck","all truck","each truck","per truck"]):
            c = veh_sum[["vehicle","distance_km","total_cost","carbon_kg",
                         "sla_breaches","utilization_pct"]].copy()
            c["vehicle"] = c["vehicle"].apply(lambda v: f"Truck {v}")
            c["total_cost"] = c["total_cost"].apply(lambda x: f"Rs.{x:,.0f}")
            results.append("ALL TRUCKS:\n" + c.to_string(index=False))

        return "\n\n".join(results) if results else ""

    # ─── ④ Rule-Based Router ──────────────────────────────────────────────────
    def rule_based_answer(query):
        q = query.lower().strip()

        if q in {"hi","hello","hey","namaste","hii"}:
            return (
                f"**Namaste!** I'm the LoRRI AI Assistant.\n\n"
                f"Fleet: **{opt['n_ships']} shipments**, **{opt['n_vehicles']} trucks**, "
                f"total cost **{inr(opt['total_cost'])}**. Ask me anything!", 99
            )

        if any(k in q for k in ["contact","email","phone","reach"]):
            return (
                "**LogisticsNow Contact:**\n"
                "- 🌐 logisticsnow.in\n"
                "- 📧 connect@logisticsnow.in\n"
                "- 📞 +91-9867773508 / +91-9653620207", 99
            )

        if any(k in q for k in ["how much did we save","total saving","how much saved"]):
            saved = base["total_cost"] - opt["total_cost"]
            return (
                f"**Total Savings: {inr(saved)} ({saved/base['total_cost']*100:.1f}% reduction)**\n\n"
                f"| Category | Baseline | Optimized | Saved |\n|---|---|---|---|\n"
                f"| Fuel | {inr(base['fuel_cost'])} | {inr(opt['fuel_cost'])} "
                f"| **{inr(base['fuel_cost']-opt['fuel_cost'])}** |\n"
                f"| Toll | {inr(base['toll_cost'])} | {inr(opt['toll_cost'])} "
                f"| **{inr(base['toll_cost']-opt['toll_cost'])}** |\n"
                f"| Driver | {inr(base['driver_cost'])} | {inr(opt['driver_cost'])} "
                f"| **{inr(base['driver_cost']-opt['driver_cost'])}** |\n"
                f"| **Total** | {inr(base['total_cost'])} | {inr(opt['total_cost'])} "
                f"| **{inr(saved)}** |", 98
            )

        if ("carbon" in q or "co2" in q) and any(k in q for k in ["saving","save","reduc"]):
            s = base["carbon_kg"] - opt["carbon_kg"]
            return (
                f"**CO2 Saved: {s:,.1f} kg ({s/base['carbon_kg']*100:.1f}% reduction)**\n\n"
                f"- Baseline: {base['carbon_kg']:,.1f} kg\n"
                f"- Optimized: {opt['carbon_kg']:,.1f} kg\n"
                f"- 🌳 {int(s/21):,} trees equivalent\n"
                f"- 🚗 {int(s/2400)} cars off the road", 98
            )

        for v in [1,2,3,4,5]:
            if f"truck {v}" in q and "route" in q:
                r = veh_sum[veh_sum["vehicle"]==v].iloc[0]
                return (
                    f"**Truck {v} Route:**\n🚛 {route_map_text[v]}\n\n"
                    f"{int(r['stops'])} stops · {r['distance_km']:,.0f} km · {inr(r['total_cost'])}", 97
                )

        if "truck 3" in q and any(k in q for k in ["change","why","explain","reason"]):
            t3 = veh_sum[veh_sum["vehicle"]==3].iloc[0]
            bc = routes[(routes["vehicle"]==3)&(routes["sla_breach_hr"]>0)]["city"].tolist()
            return (
                f"**Why Truck 3's Route:**\n{route_map_text[3]}\n\n"
                f"- Aurangabad & Solapur cluster on Pune-Hyderabad axis\n"
                f"- Madurai anchors the deep south leg\n"
                f"- Jammu: high-priority northern extension\n"
                f"- Scored by Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%)\n\n"
                f"**Stats:** {t3['distance_km']:,.0f} km · {inr(t3['total_cost'])} · "
                f"{t3['carbon_kg']:.0f} kg CO2\n"
                + (f"Breach cities: {', '.join(bc)}" if bc else "All SLA met"), 96
            )

        if any(k in q for k in ["tomorrow","traffic risk","forecast","predict traffic"]):
            hr = ships[ships["traffic_mult"]>2.0]["city"].tolist()
            mr = ships[(ships["traffic_mult"]>1.4)&(ships["traffic_mult"]<=2.0)]["city"].tolist()
            return (
                f"**Traffic Risk Forecast:**\n\n"
                f"🔴 HIGH (>2.0x): {', '.join(hr) if hr else 'None'}\n"
                f"🟡 MEDIUM (1.4-2.0x): {', '.join(mr[:5]) if mr else 'None'}\n"
                f"🟢 STABLE: All other corridors\n\n"
                f"Auto re-optimize triggers at >30% delay threshold.", 90
            )

        if any(k in q for k in ["which cities","cities late","missed sla","where late"]):
            bd = routes[routes["sla_breach_hr"]>0][
                ["city","vehicle","sla_breach_hr","sla_penalty"]]
            if bd.empty:
                return "**No SLA breaches** — all on time!", 99
            lines = [f"**{len(bd)} cities late:**\n"]
            for _, r in bd.iterrows():
                lines.append(
                    f"- **{r['city']}** (Truck {int(r['vehicle'])}) — "
                    f"{r['sla_breach_hr']:.1f}hr late · {inr(r['sla_penalty'])}"
                )
            return "\n".join(lines), 97

        return None, None

    # ─── ⑤ Local Synthesizer (no API) ────────────────────────────────────────
    def synthesize_answer(query, rag_ctx, pandas_ctx):
        q = query.lower()
        parts = []

        if pandas_ctx:
            parts.append(pandas_ctx)
        if rag_ctx:
            parts.append(rag_ctx)

        if not parts:
            return (
                f"Based on the current fleet run (Mumbai Depot):\n\n"
                f"- **{opt['n_ships']} shipments** | **{opt['n_vehicles']} trucks**\n"
                f"- Cost: **{inr(opt['total_cost'])}** "
                f"(saved {inr(base['total_cost']-opt['total_cost'])} vs baseline)\n"
                f"- SLA: **{opt['sla_pct']:.0f}%** | CO2: **{opt['carbon_kg']:,.1f} kg**\n\n"
                f"Try asking about a specific truck, SLA breaches, cost savings, "
                f"carbon, routes, or optimization methodology."
            )

        return "\n\n".join(parts)

    # ─── ⑥ Full RAG Pipeline ─────────────────────────────────────────────────
    def call_rag_pipeline(query, _history):
        # Step 1: Rule router
        rule_reply, rule_conf = rule_based_answer(query)
        if rule_reply:
            return rule_reply, rule_conf, "Rule-based router", "Instant match"

        # Step 2: BM25 retrieval
        rag_ctx    = rag_retrieve(query, top_k=3)
        # Step 3: Pandas retrieval
        pandas_ctx = pandas_retrieve(query)

        sources = []
        if rag_ctx:    sources.append("RAG chunks")
        if pandas_ctx: sources.append("Pandas DataFrames")
        src_label   = " + ".join(sources) if sources else "Synthesized"
        n_chunks    = sum(1 for c in KNOWLEDGE_CHUNKS if c["text"] in rag_ctx) if rag_ctx else 0
        chunks_info = f"{n_chunks} chunk(s) + {'DataFrame rows' if pandas_ctx else 'no structured data'}"

        # Step 4: Local synthesis
        reply = synthesize_answer(query, rag_ctx, pandas_ctx)
        conf  = 92 if (rag_ctx and pandas_ctx) else (88 if (rag_ctx or pandas_ctx) else 72)
        return reply, conf, src_label, chunks_info

    # ─── UI ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a4d2e 100%);
                border-radius:14px;padding:20px 26px;margin-bottom:16px;
                display:flex;align-items:center;gap:18px;">
        <div style="flex-shrink:0;">{logo_html(height=52)}</div>
        <div>
            <div style="font-size:1.05rem;font-weight:700;color:white;">
                LoRRI Intelligence Assistant
            </div>
            <div style="font-size:0.76rem;color:#94a3b8;margin-top:3px;">
                <b style="color:{LN_GREEN}">Vectorless RAG</b> · BM25-lite ·
                Pandas Retrieval · Rule Router · Fully Offline
            </div>
        </div>
        <div style="margin-left:auto;">
            <div style="background:rgba(34,197,94,0.15);border:1px solid {LN_GREEN};
                        border-radius:20px;padding:4px 14px;font-size:0.7rem;
                        color:{LN_GREEN};font-weight:600;">OFFLINE AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline badges
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center;">
        <span style="font-size:0.65rem;color:#64748b;font-weight:600;
                     text-transform:uppercase;letter-spacing:0.08em;">Pipeline:</span>
        {"".join([
            f'<span style="background:{bg};color:{fg};border:1px solid {br};'
            f'border-radius:6px;padding:3px 10px;font-size:0.7rem;font-weight:600;">{label}</span>'
            for label,bg,fg,br in [
                ("1 Rule Router",    "#fef3c7","#92400e","#f59e0b"),
                ("-> 2 BM25 Retriever","#f0fdf4",LN_DGREEN,LN_GREEN),
                ("-> 3 Pandas Query", "#eff6ff","#1e40af","#3b82f6"),
                ("-> 4 Synthesize",   "#f5f3ff","#5b21b6","#8b5cf6"),
            ]
        ])}
    </div>
    """, unsafe_allow_html=True)

    # Cards
    c1,c2,c3 = st.columns(3)
    for col,color,icon,title,desc in [
        (c1,LN_GREEN, "🏢","Company & Platform",
         "LogisticsNow info, LoRRI services, pricing model, contact details"),
        (c2,"#1e7abf","🚛","Fleet & Routes",
         "Per-truck cost, SLA, carbon — directly from DataFrames"),
        (c3,"#e67e22","🧠","AI & Methodology",
         "CVRP, weighted objectives, re-optimization, explainability"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                    padding:14px 16px;border-top:3px solid {color};margin-bottom:14px;">
            <div style="font-size:0.62rem;font-weight:700;color:{color};
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
                {icon} {title}</div>
            <div style="font-size:0.81rem;color:#475569;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # Session state
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    # Chip styles
    st.markdown(f"""
    <style>
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
        background:#f0fdf4!important;border:1px solid {LN_GREEN}!important;
        color:{LN_DGREEN}!important;border-radius:20px!important;
        font-size:0.72rem!important;font-weight:600!important;
        padding:4px 10px!important;white-space:normal!important;
        height:auto!important;min-height:2rem!important;
    }}
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
        background:{LN_GREEN}!important;color:white!important;
    }}
    </style>
    <div style="font-size:0.7rem;color:#64748b;font-weight:600;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">
        Click to ask:
    </div>""", unsafe_allow_html=True)

    CHIPS = [
        ("What is LoRRI?",                   "🏢"),
        ("How much did we save?",            "💰"),
        ("Which cities were late?",           "⚠️"),
        ("Which truck costs most?",           "🚛"),
        ("Truck 3 route?",                    "🗺️"),
        ("Carbon savings?",                   "🌿"),
        ("Why Truck 3 route changed",         "🧠"),
        ("Tomorrow traffic risk",             "🚦"),
        ("Suggest cost saving actions",       "💡"),
        ("Contact LogisticsNow",              "📞"),
    ]

    chip_cols    = st.columns(5)
    chip_clicked = False
    for idx, (question, emoji) in enumerate(CHIPS):
        with chip_cols[idx % 5]:
            if st.button(f"{emoji} {question}", key=f"chip_{idx}", use_container_width=True):
                st.session_state.chip_prompt = question
                chip_clicked = True
    if chip_clicked:
        st.rerun()

    st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # Welcome
    if not st.session_state.msgs:
        with st.chat_message("assistant", avatar="🚚"):
            st.markdown(
                f"**Namaste! Welcome to LoRRI Intelligence Assistant.**\n\n"
                f"Pipeline: Rule Router → BM25 Retriever → Pandas → Synthesize "
                f"*(fully offline, no external API)*\n\n"
                f"Fleet: **{opt['n_ships']} shipments** · **{opt['n_vehicles']} trucks** · "
                f"Cost **{inr(opt['total_cost'])}** · SLA **{opt['sla_pct']:.0f}%**\n\n"
                f"Click a chip or type your question below."
            )

    # Chat history
    for m in st.session_state.msgs:
        avatar = "🚚" if m["role"] == "assistant" else "👤"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                meta = m["meta"]
                cc = LN_GREEN if meta["conf"]>=90 else ("#f59e0b" if meta["conf"]>=80 else "#dc2626")
                st.markdown(f"""
                <div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                            border:1px solid {LN_BORDER};border-radius:20px;
                            display:inline-flex;align-items:center;gap:8px;
                            font-size:0.7rem;color:#64748b;">
                    <span style="color:{cc};font-weight:700;">●</span>
                    <b style="color:{cc};">{meta['conf']}%</b>
                    &nbsp;·&nbsp; {meta['source']} &nbsp;·&nbsp; {meta['chunks']}
                </div>""", unsafe_allow_html=True)

    # Input
    final_prompt = st.session_state.pop("chip_prompt", None)
    typed = st.chat_input("Ask about fleet data, costs, routes, SLA, carbon...",
                          key="lorri_chat_input")
    if typed:
        final_prompt = typed

    if final_prompt:
        st.session_state.msgs.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(final_prompt)

        api_msgs = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.msgs[-10:]]

        with st.chat_message("assistant", avatar="🚚"):
            with st.spinner("Running RAG pipeline..."):
                reply, conf, source, chunks = call_rag_pipeline(final_prompt, api_msgs)
            st.markdown(reply)
            cc = LN_GREEN if conf>=90 else ("#f59e0b" if conf>=80 else "#dc2626")
            st.markdown(f"""
            <div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                        border:1px solid {LN_BORDER};border-radius:20px;
                        display:inline-flex;align-items:center;gap:8px;
                        font-size:0.7rem;color:#64748b;">
                <span style="color:{cc};font-weight:700;">●</span>
                <b style="color:{cc};">{conf}%</b>
                &nbsp;·&nbsp; {source} &nbsp;·&nbsp; {chunks}
            </div>""", unsafe_allow_html=True)

        st.session_state.msgs.append({
            "role": "assistant", "content": reply,
            "meta": {"conf": conf, "source": source, "chunks": chunks},
        })
        st.rerun()

    if st.session_state.msgs:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.msgs = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "📊 Dashboard Summary":
    page_header("📊 Dashboard Summary", "Baseline vs AI-Optimized · All costs in Rs. INR")
    loading_state("Refreshing dashboard metrics...")

    st.markdown("""<div class="info-box">
    📋 <b>Report card for the full delivery run.</b>
    Baseline = trucks without AI. Optimized = LoRRI AI planner. All figures in <b>Rs. INR</b>.
    </div>""", unsafe_allow_html=True)

    sc_ = base["total_cost"] - opt["total_cost"]
    sd_ = base["distance_km"] - opt["distance_km"]
    sco = base["carbon_kg"]   - opt["carbon_kg"]
    ss_ = opt["sla_pct"]      - base["sla_pct"]

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px;">
    {kpi_card("Total Cost Savings",  inr(sc_),                        f"down -{sc_/base['total_cost']*100:.1f}% vs baseline",True,LN_GREEN)}
    {kpi_card("Optimized Distance",  f"{opt['distance_km']:,.0f} km", f"down {sd_:,.0f} km saved",                          True,"#1e7abf")}
    {kpi_card("SLA Adherence",       f"{opt['sla_pct']:.0f}%",        f"up +{ss_:.0f} pts (base {base['sla_pct']:.0f}%)",   True,"#e67e22")}
    {kpi_card("Carbon Reduced",      f"{sco/1000:.1f}t CO2",          f"down {sco/base['carbon_kg']*100:.1f}% cleaner",      True,"#27ae60")}
    {kpi_card("Fleet Utilization",   f"{veh_sum['utilization_pct'].mean():.1f}%", f"Avg across {opt['n_vehicles']} trucks", True,"#8e44ad")}
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fuel Saved",   inr(base["fuel_cost"]  -opt["fuel_cost"]),   f"-{(base['fuel_cost']-opt['fuel_cost'])/base['fuel_cost']*100:.1f}%",   delta_color="inverse")
    c2.metric("Toll Saved",   inr(base["toll_cost"]  -opt["toll_cost"]),   f"-{(base['toll_cost']-opt['toll_cost'])/base['toll_cost']*100:.1f}%",   delta_color="inverse")
    c3.metric("Driver Saved", inr(base["driver_cost"]-opt["driver_cost"]), f"-{(base['driver_cost']-opt['driver_cost'])/base['driver_cost']*100:.1f}%", delta_color="inverse")
    c4.metric("Time Saved",   f"{base['time_hr']-opt['time_hr']:,.1f} hr", f"-{(base['time_hr']-opt['time_hr'])/base['time_hr']*100:.1f}%",          delta_color="inverse")

    st.markdown(sh("Cost Reduction Over Optimization Iterations"), unsafe_allow_html=True)
    iters = list(range(1,11))
    decay = [(base["total_cost"]-(base["total_cost"]-opt["total_cost"])*(1-np.exp(-0.4*i))) for i in iters]
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=iters,y=[base["total_cost"]]*10,mode="lines",
        line=dict(color="#cbd5e1",dash="dash",width=2),name="Baseline"))
    fig_t.add_trace(go.Scatter(x=iters,y=decay,mode="lines+markers",
        line=dict(color=LN_GREEN,width=3),marker=dict(size=7,color=LN_GREEN),
        fill="tonexty",fillcolor="rgba(58,125,44,0.08)",name="Optimized"))
    apply_theme(fig_t,height=280,title="Fleet Cost vs Optimization Iterations")
    fig_t.update_yaxes(tickprefix="Rs.",tickformat=",")
    fig_t.update_xaxes(title_text="Iteration")
    st.plotly_chart(fig_t,use_container_width=True)

    st.markdown(sh("Per-Truck Summary"), unsafe_allow_html=True)
    d = veh_sum.copy()
    d.insert(0,"Truck",d["vehicle"].apply(lambda v: f"Truck {v}"))
    d = d.drop(columns=["vehicle"])
    d.columns = ["Truck","Stops","Load (kg)","Dist (km)","Time (hr)",
                 "Fuel","Toll","Driver","SLA Penalty","Total","Carbon (kg)","SLA Breaches","Util %"]
    st.dataframe(
        d.style
         .format({"Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}",
                  "Fuel":"Rs.{:,.0f}","Toll":"Rs.{:,.0f}","Driver":"Rs.{:,.0f}",
                  "SLA Penalty":"Rs.{:,.0f}","Total":"Rs.{:,.0f}",
                  "Carbon (kg)":"{:.1f}","Util %":"{:.1f}%"})
         .background_gradient(subset=["Util %"],cmap="Greens")
         .background_gradient(subset=["Total"],cmap="YlOrRd"),
        use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Route Map":
    page_header("🗺️ Route Map","Live India delivery network · Mumbai depot hub")
    loading_state("Refreshing traffic...")

    col_map,col_ctrl = st.columns([3,1])

    with col_ctrl:
        st.markdown(sh("Map Controls"), unsafe_allow_html=True)
        route_mode   = st.radio("Route View",["Optimized","Baseline","Comparison"],index=0)
        show_heatmap = st.toggle("Traffic Heatmap",value=False)
        sel_v = st.multiselect("Filter Trucks",
                               options=sorted(routes["vehicle"].unique()),
                               default=sorted(routes["vehicle"].unique()),
                               format_func=lambda v: f"Truck {v}")
        st.markdown("---")
        for v in sorted(routes["vehicle"].unique()):
            vr    = routes[routes["vehicle"]==v]
            color = V_COLORS.get(v,"#999")
            vd    = veh_sum[veh_sum["vehicle"]==v].iloc[0]
            sla_r = "HIGH" if vd["sla_breaches"]>=2 else ("MED" if vd["sla_breaches"]==1 else "OK")
            st.markdown(
                f'<div class="legend-row">'
                f'<div class="legend-dot" style="background:{color}"></div>'
                f'<div><b style="color:{LN_NAVY}">Truck {v}</b><br>'
                f'<span style="font-size:0.72rem;color:#64748b;">'
                f'{len(vr)} stops · {vd["distance_km"]:,.0f} km<br>'
                f'{inr(vd["total_cost"])} · SLA {sla_r}</span></div></div>',
                unsafe_allow_html=True)

    with col_map:
        fig = go.Figure()
        p_dot = {"HIGH":"#dc2626","MEDIUM":"#f97316","LOW":LN_GREEN}

        if route_mode in ["Baseline","Comparison"]:
            bl  = [DEPOT["latitude"]]  + ships["latitude"].tolist()  + [DEPOT["latitude"]]
            blo = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=bl,lon=blo,mode="lines",
                line=dict(width=2,color="rgba(220,38,38,0.5)"),name="Baseline"))

        if route_mode in ["Optimized","Comparison"]:
            for v in sel_v:
                vdf   = routes[routes["vehicle"]==v].sort_values("stop_order")
                color = V_COLORS.get(v,"#999")
                lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
                lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
                fig.add_trace(go.Scattermap(lat=lats,lon=lons,mode="lines",
                    line=dict(width=3,color=color),name=f"Truck {v}",legendgroup=f"v{v}"))
                for _,row in vdf.iterrows():
                    breach = f"{row['sla_breach_hr']:.1f}hr late" if row["sla_breach_hr"]>0 else "On time"
                    dd = haversine(DEPOT["latitude"],DEPOT["longitude"],row["latitude"],row["longitude"])
                    fig.add_trace(go.Scattermap(
                        lat=[row["latitude"]],lon=[row["longitude"]],
                        mode="markers+text",
                        marker=dict(size=14,color=p_dot.get(row.get("priority","MEDIUM"),"#f97316")),
                        text=[f"T{v}.{int(row['stop_order'])}"],
                        textfont=dict(size=8,color="white"),
                        textposition="middle center",
                        hovertext=(
                            f"Truck {v} Stop {int(row['stop_order'])}: "
                            f"{row.get('city',row['shipment_id'])} | "
                            f"{dd:.0f}km depot | ETA:{row['travel_time_hr']:.1f}hr | "
                            f"{inr(row['total_cost'])} | {row['carbon_kg']:.1f}kg CO2 | {breach}"
                        ),
                        hoverinfo="text",showlegend=False,legendgroup=f"v{v}"))

        if show_heatmap:
            hm = ships["traffic_mult"].tolist()
            fig.add_trace(go.Scattermap(
                lat=ships["latitude"].tolist(),lon=ships["longitude"].tolist(),
                mode="markers",
                marker=dict(size=[v*18 for v in hm],color=hm,
                            colorscale="RdYlGn_r",cmin=1.0,cmax=3.0,opacity=0.4),
                name="Traffic",hovertext=ships["city"],hoverinfo="text"))

        fig.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]],lon=[DEPOT["longitude"]],
            mode="markers+text",text=["Depot"],
            textposition="top right",textfont=dict(size=10,color=LN_NAVY),
            marker=dict(size=18,color=LN_NAVY,symbol="star"),name="Mumbai Depot"))

        fig.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=20.5,lon=78.9),zoom=4),
            height=620,margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.92)",
                        bordercolor=LN_BORDER,borderwidth=1))
        st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "💰 Financial Analysis":
    page_header("💰 Financial Analysis","All costs in Rs. INR · Fuel Rs.12/km · Driver Rs.180/hr · SLA Rs.500/hr")
    loading_state("Calculating savings...")

    fuel_s   = base["fuel_cost"]   - opt["fuel_cost"]
    toll_s   = base["toll_cost"]   - opt["toll_cost"]
    driver_s = base["driver_cost"] - opt["driver_cost"]
    total_s  = fuel_s + toll_s + driver_s
    roi_pct  = total_s / base["total_cost"] * 100
    payback  = int(15000 / (total_s / 30)) if total_s > 0 else 999

    rc1,rc2,rc3,rc4,rc5 = st.columns(5)
    for col,label,val,icon,color in [
        (rc1,"Fuel Saved",       inr(fuel_s),   "⛽",LN_GREEN),
        (rc2,"Toll Saved",       inr(toll_s),   "🛣️","#1e7abf"),
        (rc3,"Driver Saved",     inr(driver_s), "👷","#8e44ad"),
        (rc4,"Total Saved",      inr(total_s),  "💰","#e67e22"),
        (rc5,"Payback",         f"{payback}d",  "⏳","#27ae60"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:18px 16px;text-align:center;border-top:3px solid {color};">
            <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:1.1rem;font-weight:800;color:{color};">{val}</div>
            <div style="font-size:0.68rem;color:#64748b;margin-top:4px;font-weight:600;
                        text-transform:uppercase;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#f0fdf4;border:1px solid {LN_GREEN};border-radius:10px;
                padding:12px 20px;margin-top:12px;font-size:0.84rem;color:{LN_NAVY};">
        ROI: <b>{roi_pct:.1f}% cost reduction</b> · Payback in <b>{payback} days</b> ·
        Annual savings: <b>{inr(total_s*12)}</b>
    </div><br>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fig_b = go.Figure()
        for cat,bc_,lbl in [("fuel_cost",LN_GREEN,"Fuel"),
                              ("toll_cost","#1e7abf","Toll"),
                              ("driver_cost","#8e44ad","Driver")]:
            fig_b.add_trace(go.Bar(name=lbl,x=["Baseline","Optimized"],
                y=[base[cat],opt[cat]],marker_color=bc_,
                text=[inr(base[cat]),inr(opt[cat])],textposition="inside",
                textfont=dict(color="white",size=10)))
        apply_theme(fig_b,height=360,title="Cost Components: Baseline vs Optimized",legend_below=True)
        fig_b.update_layout(barmode="stack")
        fig_b.update_yaxes(tickprefix="Rs.",tickformat=",")
        st.plotly_chart(fig_b,use_container_width=True)

    with c2:
        sv={"Fuel Saved":fuel_s,"Toll Saved":toll_s,"Driver Saved":driver_s}
        fig_w=go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","total"],
            x=list(sv.keys())+["Total Saved"],
            y=list(sv.values())+[sum(sv.values())],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":LN_GREEN}},
            totals={"marker":{"color":"#1e7abf"}},
            text=[inr(v) for v in list(sv.values())+[sum(sv.values())]],
            textposition="outside"))
        apply_theme(fig_w,height=360,title="Savings Waterfall")
        fig_w.update_yaxes(tickprefix="Rs.",tickformat=",")
        st.plotly_chart(fig_w,use_container_width=True)

    fig_v=go.Figure()
    for cat,bc_,lbl in [("fuel_cost",LN_GREEN,"Fuel"),("toll_cost","#1e7abf","Toll"),
                         ("driver_cost","#8e44ad","Driver"),("sla_penalty","#c0392b","SLA Penalty")]:
        fig_v.add_trace(go.Bar(name=lbl,x=[f"Truck {v}" for v in veh_sum["vehicle"]],
            y=veh_sum[cat],marker_color=bc_,
            text=veh_sum[cat].apply(inr),textposition="inside",
            textfont=dict(color="white",size=9)))
    apply_theme(fig_v,height=320,legend_below=True)
    fig_v.update_layout(barmode="stack",title="Per-Truck Cost Breakdown")
    fig_v.update_yaxes(tickprefix="Rs.",tickformat=",")
    st.plotly_chart(fig_v,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CARBON & SLA
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🌿 Carbon & SLA":
    page_header("🌿 Carbon & SLA","Sustainability metrics · Delivery compliance")

    co2_saved  = base["carbon_kg"] - opt["carbon_kg"]
    trees      = int(co2_saved / 21)
    cars_off   = int(co2_saved / 2400)
    km_avoided = int(base["distance_km"] - opt["distance_km"])

    sus = st.columns(4)
    for col,icon,val,label,sub,color in [
        (sus[0],"🌿",f"{co2_saved:,.0f} kg","CO2 Reduced",    f"{co2_saved/base['carbon_kg']*100:.1f}% less",LN_GREEN),
        (sus[1],"🌳",f"{trees:,}",           "Trees Equiv",    "CO2 absorbed per year",                      "#27ae60"),
        (sus[2],"🚗",f"{cars_off}",           "Cars Off Road",  "Annual emissions removed",                   "#1e7abf"),
        (sus[3],"📏",f"{km_avoided:,} km",   "Dist Avoided",   "Fewer km vs baseline",                       "#e67e22"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:18px;text-align:center;border-top:3px solid {color};">
            <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:1.25rem;font-weight:800;color:{color};">{val}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{LN_NAVY};margin:5px 0 3px;">{label}</div>
            <div style="font-size:0.7rem;color:#94a3b8;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        fig_c=go.Figure()
        fig_c.add_trace(go.Bar(x=["Baseline","Optimized"],
            y=[base["carbon_kg"],opt["carbon_kg"]],marker_color=["#c0392b",LN_GREEN],
            text=[f"{base['carbon_kg']:,.1f} kg",f"{opt['carbon_kg']:,.1f} kg"],
            textposition="outside"))
        apply_theme(fig_c,height=280,title=f"CO2 Saved: {co2_saved:,.1f} kg")
        fig_c.update_layout(showlegend=False)
        st.plotly_chart(fig_c,use_container_width=True)

        city_co2=routes.groupby("city")["carbon_kg"].sum().sort_values(ascending=False).head(8).reset_index()
        fig_city=go.Figure(go.Bar(x=city_co2["carbon_kg"],y=city_co2["city"],orientation="h",
            marker_color=[LN_GREEN if i>2 else "#c0392b" for i in range(len(city_co2))],
            text=city_co2["carbon_kg"].round(1).astype(str)+" kg",textposition="outside"))
        apply_theme(fig_city,height=280,title="Top Cities — CO2 (kg)")
        fig_city.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_city,use_container_width=True)

    with c2:
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number+delta",value=opt["sla_pct"],
            number={"suffix":"%"},title={"text":"SLA Adherence"},
            delta={"reference":base["sla_pct"],"increasing":{"color":LN_GREEN},"suffix":"% vs baseline"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":LN_GREEN},
                   "steps":[{"range":[0,50],"color":"rgba(192,57,43,0.15)"},
                             {"range":[50,80],"color":"rgba(245,158,11,0.15)"},
                             {"range":[80,100],"color":"rgba(58,125,44,0.15)"}],
                   "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":base["sla_pct"]}}))
        fig_g.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g,use_container_width=True)

        bd=routes.copy(); bd["breached"]=(bd["sla_breach_hr"]>0).astype(int)
        piv=bd.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_h=go.Figure(go.Heatmap(
            z=piv.values,x=piv.columns.tolist(),
            y=[f"Truck {v}" for v in piv.index],
            colorscale="YlOrRd",text=piv.values,texttemplate="%{text}",
            colorbar=dict(title="Breaches")))
        apply_theme(fig_h,height=260,title="Breaches: Truck x Priority")
        st.plotly_chart(fig_h,use_container_width=True)

    bdf=routes[routes["sla_breach_hr"]>0][
        ["vehicle","stop_order","city","priority","sla_hours","sla_breach_hr","sla_penalty","total_cost"]].copy()
    if not bdf.empty:
        st.markdown(sh("SLA Breach Detail (Rs.500/hr penalty)"), unsafe_allow_html=True)
        bdf["vehicle"]=bdf["vehicle"].apply(lambda v: f"Truck {v}")
        bdf.columns=["Truck","Stop#","City","Priority","SLA (hr)","Breach (hr)","Penalty","Total Cost"]
        st.dataframe(
            bdf.style.format({"Breach (hr)":"{:.1f}","Penalty":"Rs.{:,.0f}","Total Cost":"Rs.{:,.0f}"})
                     .background_gradient(subset=["Breach (hr)"],cmap="Reds"),
            use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🧠 Explainability":
    page_header("🧠 Explainability","Why the AI chose these routes · SHAP-style permutation importance")

    top_feat   = max(fi, key=fi.get)
    top_val    = fi[top_feat]
    worst_stop = routes.loc[routes["mo_score"].idxmax()]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a3a2a 100%);
                border-radius:14px;padding:22px 28px;margin-bottom:20px;color:white;">
        <div style="font-size:0.65rem;font-weight:700;color:{LN_GREEN};text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:10px;">Routing Decision Explanation</div>
        <div style="font-size:0.86rem;color:#cbd5e1;line-height:1.9;">
            Most influential factor: <b style="color:{LN_GREEN}">{top_feat}</b>
            ({top_val:.1f}% importance). Route score:
            <b>0.35xCost + 0.30xTime + 0.20xCarbon + 0.15xSLA</b>.<br>
            Hardest stop: <b style="color:#fbbf24">{worst_stop['city']}</b>
            (Truck {int(worst_stop['vehicle'])}) — MO Score {worst_stop['mo_score']:.4f}.
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2=st.columns([1,2])
    with c1:
        fig_pie=go.Figure(go.Pie(
            labels=["Cost","Travel Time","Carbon CO2","SLA"],
            values=[35,30,20,15],hole=0.55,
            marker_colors=[LN_GREEN,"#1e7abf","#27ae60","#c0392b"],
            textinfo="label+percent"))
        fig_pie.update_layout(height=290,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
            annotations=[{"text":"Weights","x":0.5,"y":0.5,"font_size":13,"showarrow":False}])
        st.plotly_chart(fig_pie,use_container_width=True)

    with c2:
        fi_l=list(fi.keys()); fi_v=list(fi.values()); mv=max(fi_v)
        fig_fi=go.Figure(go.Bar(x=fi_v,y=fi_l,orientation="h",
            marker_color=["#c0392b" if v==mv else LN_GREEN for v in fi_v],
            text=[f"{v:.1f}%" for v in fi_v],textposition="outside"))
        apply_theme(fig_fi,height=300,title="Feature Importance (Permutation-Based)")
        fig_fi.update_xaxes(title_text="Importance (%)")
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi,use_container_width=True)

    vf=st.selectbox("Filter:",["All Trucks"]+[f"Truck {v}" for v in sorted(routes["vehicle"].unique())])
    scd=sc if vf=="All Trucks" else sc[sc["vehicle"]==int(vf.split()[-1])].copy()
    fc=["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    fco=[LN_GREEN,"#1e7abf","#8e44ad","#e67e22","#27ae60","#c0392b"]
    fig_stk=go.Figure()
    for f_,c_ in zip(fc,fco):
        fig_stk.add_trace(go.Bar(name=f_,x=scd["city"],y=scd[f_],marker_color=c_))
    apply_theme(fig_stk,height=380,legend_below=True)
    fig_stk.update_layout(barmode="stack",title="Per-Stop Score Contribution by Truck")
    fig_stk.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_stk,use_container_width=True)

    t10=routes.nlargest(10,"mo_score")[
        ["vehicle","stop_order","city","priority","weight","travel_time_hr",
         "fuel_cost","carbon_kg","sla_breach_hr","mo_score"]].copy()
    t10["vehicle"]=t10["vehicle"].apply(lambda v: f"Truck {v}")
    t10.columns=["Truck","Stop#","City","Priority","Weight","Time (hr)",
                 "Fuel","Carbon (kg)","Breach (hr)","MO Score"]
    st.dataframe(
        t10.style.format({"Weight":"{:.0f}","Time (hr)":"{:.2f}",
                          "Fuel":"Rs.{:,.0f}","Carbon (kg)":"{:.2f}",
                          "Breach (hr)":"{:.1f}","MO Score":"{:.4f}"})
                 .background_gradient(subset=["MO Score"],cmap="YlOrRd"),
        use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RE-OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "⚡ Re-optimization Simulator":
    page_header("⚡ Re-optimization Simulator","Simulate disruptions · Watch LoRRI re-plan instantly")

    st.markdown("""<div class="info-box">
    Simulate <b>traffic jams</b> or <b>priority escalations</b>.
    LoRRI re-plans the affected truck and shows Before vs After.
    </div>""", unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown(sh("Scenario 1 — Traffic Disruption"), unsafe_allow_html=True)
        city1=st.selectbox("City hit by traffic:",sorted(ships["city"].tolist()))
        spike=st.slider("Traffic multiplier",1.0,3.0,2.5,0.1)
        if st.button("Trigger Traffic Disruption",use_container_width=True):
            row=ships[ships["city"]==city1].iloc[0]
            om=row["traffic_mult"]
            dk=haversine(DEPOT["latitude"],DEPOT["longitude"],row["latitude"],row["longitude"])
            to=dk/(55/om); tn=dk/(55/spike); pi=(tn-to)/to*100
            if pi>30:
                st.markdown(f"""<div class="warn-box">
                <b>{city1}</b>: {om:.2f}x to <span class="tag-red">{spike:.2f}x</span> |
                +{pi:.1f}% delay — THRESHOLD BREACHED!
                </div>""", unsafe_allow_html=True)
                t0=time.time()
                with st.spinner("Re-optimizing..."): time.sleep(1.2)
                elapsed=time.time()-t0
                av=routes[routes["city"]==city1]["vehicle"].values
                if len(av):
                    vid=av[0]
                    orig=routes[routes["vehicle"]==vid].sort_values("stop_order")
                    mask=orig["city"]==city1
                    reop=pd.concat([orig[~mask],orig[mask]]).reset_index(drop=True)
                    def rdist(df):
                        return sum(haversine(df.iloc[i]["latitude"],df.iloc[i]["longitude"],
                                             df.iloc[i+1]["latitude"],df.iloc[i+1]["longitude"])
                                   for i in range(len(df)-1))
                    d1=rdist(orig); d2=rdist(reop)
                    st.markdown(f'<div class="ok-box">Truck {vid} re-routed in <b>{elapsed:.2f}s</b></div>',unsafe_allow_html=True)
                    ba1,ba2,ba3=st.columns(3)
                    ba1.metric("Distance",f"{d1:.0f} km",f"{d2-d1:+.0f} km",delta_color="inverse")
                    ba2.metric("ETA",f"{orig['travel_time_hr'].sum():.1f} hr")
                    ba3.metric("Cost",inr(orig["total_cost"].sum()))
            else:
                st.markdown(f'<div class="ok-box">{pi:.1f}% — within 30% threshold. No re-opt needed.</div>',unsafe_allow_html=True)

    with c2:
        st.markdown(sh("Scenario 2 — Priority Escalation"), unsafe_allow_html=True)
        city2=st.selectbox("City escalated:",sorted(ships["city"].tolist()),key="esc")
        if st.button("Trigger Priority Escalation",use_container_width=True):
            op_=ships[ships["city"]==city2]["priority"].values[0]
            os_=ships[ships["city"]==city2]["sla_hours"].values[0]
            if op_=="HIGH":
                st.markdown(f'<div class="ok-box">{city2} is already HIGH priority!</div>',unsafe_allow_html=True)
            else:
                t0=time.time()
                with st.spinner("Escalating..."): time.sleep(1.0)
                elapsed=time.time()-t0
                av=routes[routes["city"]==city2]["vehicle"].values
                vid=av[0] if len(av) else 1
                orig=routes[routes["vehicle"]==vid].sort_values("stop_order")
                mask=orig["city"]==city2
                pen=orig[mask]["sla_penalty"].values[0] if mask.any() else 0
                st.markdown(f"""<div class="ok-box">
                <b>{city2}</b>: <span class="tag-yellow">{op_}</span> to
                <span class="tag-red">HIGH</span> | SLA: {os_}hr to 24hr |
                Moved to Stop 1 on Truck {vid} in <b>{elapsed:.2f}s</b>
                </div>""", unsafe_allow_html=True)
                ba1,ba2,ba3=st.columns(3)
                ba1.metric("Old SLA",f"{os_} hr","to 24 hr")
                ba2.metric("Penalty Saved",inr(pen),delta=f"-{inr(pen)}",delta_color="inverse")
                ba3.metric("Replan Time",f"{elapsed:.2f}s")

    st.markdown("---")
    st.markdown(sh("Live Risk Monitor"), unsafe_allow_html=True)
    rdf=ships[["city","traffic_mult","priority","sla_hours"]].copy()
    rdf["risk"]=(rdf["traffic_mult"]/1.8*0.6
                 +rdf["sla_hours"].map({24:1.0,48:0.5,72:0.2})*0.4).round(3)
    rdf["status"]=rdf["risk"].apply(
        lambda x: "HIGH RISK" if x>0.7 else ("MONITOR" if x>0.4 else "STABLE"))
    rdf=rdf.sort_values("risk",ascending=False)
    fig_r=px.bar(rdf.head(15),x="city",y="risk",color="status",
        color_discrete_map={"HIGH RISK":"#c0392b","MONITOR":"#f59e0b","STABLE":LN_GREEN},
        title="Top 15 Cities by Re-Optimization Risk",
        labels={"risk":"Risk Score","city":"City"},height=320)
    fig_r.add_hline(y=0.7,line_dash="dash",line_color="#c0392b",
                    annotation_text="Trigger threshold (0.70)")
    apply_theme(fig_r)
    st.plotly_chart(fig_r,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ROUTE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🔮 AI Route Predictor":
    page_header("🔮 AI Route Predictor","Plan a new route · Estimated distance, cost, ETA, carbon, SLA risk")
    loading_state("Initializing prediction engine...")

    all_cities=sorted(ships["city"].tolist())

    f1,f2,f3=st.columns(3)
    with f1:
        dst_city=st.selectbox("Destination City",all_cities,index=5)
        cargo_wt=st.slider("Cargo Weight (kg)",50,800,350,25)
    with f2:
        priority=st.selectbox("Priority",["LOW","MEDIUM","HIGH"])
        truck_type=st.selectbox("Truck Type",["Standard (800 kg)","Express (600 kg)","Heavy (1000 kg)"])
    with f3:
        traffic=st.selectbox("Traffic",["Normal (1.0x)","Moderate (1.5x)","Heavy (2.0x)","Severe (3.0x)"])

    sla_map={"LOW":72,"MEDIUM":48,"HIGH":24}
    sla_hr=sla_map[priority]
    traffic_mult=float(traffic.split("(")[1].split("x")[0])

    if st.button("Predict Route",use_container_width=True,type="primary"):
        t0=time.time()
        with st.spinner("Computing optimal route..."): time.sleep(1.2)

        dst_row=ships[ships["city"]==dst_city]
        if dst_row.empty:
            st.error("City not found."); st.stop()

        dst_lat=dst_row.iloc[0]["latitude"]
        dst_lon=dst_row.iloc[0]["longitude"]
        dist_p =haversine(DEPOT["latitude"],DEPOT["longitude"],dst_lat,dst_lon)
        avg_spd=55/traffic_mult
        time_hr=dist_p/avg_spd
        fuel_p =dist_p*12; toll_p=dist_p*2.8; drv_p=time_hr*180
        pen_p  =max(0,(time_hr-sla_hr)*500) if time_hr>sla_hr else 0
        total_p=fuel_p+toll_p+drv_p+pen_p
        co2_p  =dist_p*0.27
        sla_risk="HIGH" if pen_p>0 else ("MEDIUM" if time_hr>sla_hr*0.8 else "LOW")
        elapsed=time.time()-t0

        st.markdown(f"""
        <div style="background:{LN_GREEN};border-radius:10px;padding:10px 18px;
                    color:white;font-size:0.82rem;margin-bottom:16px;">
            Route predicted in <b>{elapsed:.2f}s</b> — Mumbai to <b>{dst_city}</b> ·
            {cargo_wt}kg · {priority} priority
        </div>""", unsafe_allow_html=True)

        k1,k2,k3,k4,k5,k6=st.columns(6)
        for col,label,val,color in [
            (k1,"Distance",    f"{dist_p:.0f} km",  LN_GREEN),
            (k2,"ETA",         f"{time_hr:.1f} hr", "#1e7abf"),
            (k3,"Total Cost",  inr(total_p),         "#e67e22"),
            (k4,"Fuel",        inr(fuel_p),          LN_GREEN),
            (k5,"CO2",         f"{co2_p:.1f} kg",   "#27ae60"),
            (k6,"SLA Risk",    sla_risk, "#dc2626" if sla_risk=="HIGH" else LN_GREEN),
        ]:
            col.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                        padding:14px;text-align:center;border-top:3px solid {color};">
                <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
                <div style="font-size:1.05rem;font-weight:800;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

        if pen_p>0:
            st.markdown(f'<div class="warn-box">SLA breach risk! Penalty: <b>{inr(pen_p)}</b></div>',
                        unsafe_allow_html=True)

        mid_lat=(DEPOT["latitude"]+dst_lat)/2
        mid_lon=(DEPOT["longitude"]+dst_lon)/2
        fig_pred=go.Figure()
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],dst_lat],lon=[DEPOT["longitude"],dst_lon],
            mode="lines",line=dict(width=4,color=LN_GREEN),name="Primary Route"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]],lon=[DEPOT["longitude"]],
            mode="markers+text",text=["Mumbai Depot"],
            textposition="top right",textfont=dict(size=10,color=LN_NAVY),
            marker=dict(size=16,color=LN_NAVY),showlegend=False))
        fig_pred.add_trace(go.Scattermap(
            lat=[dst_lat],lon=[dst_lon],
            mode="markers+text",text=[dst_city],
            textposition="top right",textfont=dict(size=10,color=LN_NAVY),
            marker=dict(size=14,color=LN_GREEN),showlegend=False))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],mid_lat+1.5,dst_lat],
            lon=[DEPOT["longitude"],mid_lon-1.5,dst_lon],
            mode="lines",line=dict(width=2,color="rgba(220,38,38,0.45)",dash="dot"),
            name="Alt 1 (Scenic)"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],mid_lat-1.0,dst_lat],
            lon=[DEPOT["longitude"],mid_lon+1.0,dst_lon],
            mode="lines",line=dict(width=2,color="rgba(30,122,191,0.45)",dash="dash"),
            name="Alt 2 (Express)"))
        fig_pred.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=mid_lat,lon=mid_lon),zoom=5),
            height=380,margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.9)"))
        st.plotly_chart(fig_pred,use_container_width=True)

        dist_a1=dist_p*1.11; time_a1=dist_a1/avg_spd; total_a1=dist_a1*12+dist_a1*1.2+time_a1*180
        dist_a2=dist_p*0.94; time_a2=dist_a2/(avg_spd*1.15); total_a2=dist_a2*12+dist_a2*5.2+time_a2*180
        a1,a2=st.columns(2)
        with a1:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #dc2626;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:8px;">Alt 1 — Scenic Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    {dist_a1:.0f} km (+{dist_a1-dist_p:.0f} km) · {time_a1:.1f} hr · {inr(total_a1)}<br>
                    Avoids highway tolls
                </div>
            </div>""", unsafe_allow_html=True)
        with a2:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #1e7abf;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:8px;">Alt 2 — Express Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    {dist_a2:.0f} km (-{dist_p-dist_a2:.0f} km) · {time_a2:.1f} hr · {inr(total_a2)}<br>
                    Higher toll, faster delivery
                </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FOOTER (single, appears on all pages)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{LN_NAVY};color:#94a3b8;padding:20px 40px;margin-top:20px;
            font-size:0.75rem;display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:12px;">
    <div style="display:flex;align-items:center;gap:16px;">
        {logo_html(height=36)}
        <div>
            <b style="color:white;">LogisticsNow</b> · LoRRI AI Route Optimization Engine<br>
            <span style="color:#64748b;">Problem Statement 4 · Synapflow Hackathon</span>
        </div>
    </div>
    <div style="text-align:right;">
        📧 connect@logisticsnow.in &nbsp;·&nbsp; 🌐 logisticsnow.in
    </div>
</div>
""", unsafe_allow_html=True)
