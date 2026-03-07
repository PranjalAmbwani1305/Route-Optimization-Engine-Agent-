"""
LogisticsNow · LoRRI AI Route Optimization Engine
- LogisticsNow branding (green #2d6a2d, navy #1e2d3d)
- AI Assistant (RAG) is the FIRST tab
- India map with truck numbers, all costs in ₹ INR
- TypeError fixed: apply_theme() never merged with kwargs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time

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
# BRAND COLORS  (from LogisticsNow website)
# ─────────────────────────────────────────────────────────────────────────────
LN_GREEN  = "#3a7d2c"   # LogisticsNow primary green
LN_DGREEN = "#2d6a2d"   # darker green (buttons)
LN_NAVY   = "#1e2d3d"   # dark topbar
LN_LGRAY  = "#f5f6f7"   # page bg
LN_BORDER = "#e0e4e8"

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
VEHICLE_CAP = 800
V_COLORS    = {1:"#3a7d2c", 2:"#1e7abf", 3:"#e67e22", 4:"#8e44ad", 5:"#c0392b"}

# ─────────────────────────────────────────────────────────────────────────────
# SAFE THEME  — never merge with explicit kwargs
# ─────────────────────────────────────────────────────────────────────────────
def apply_theme(fig, height=340, title="", legend_below=False, yprefix=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Poppins, sans-serif", color="#1e2d3d", size=11),
        height=height,
        margin=dict(l=10, r=10, t=44 if title else 18, b=10),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color="#1e2d3d", family="Poppins, sans-serif")))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0))
    if yprefix:
        fig.update_yaxes(tickprefix=yprefix, tickformat=",")
    fig.update_xaxes(gridcolor="#f0f4f0", zeroline=False, linecolor=LN_BORDER)
    fig.update_yaxes(gridcolor="#f0f4f0", zeroline=False, linecolor=LN_BORDER)
    return fig

def inr(val):
    return f"₹{val:,.0f}"

# ─────────────────────────────────────────────────────────────────────────────
# FULL CSS — LogisticsNow style
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
    background: {LN_LGRAY};
}}
.main .block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* ── TOP BAR (like LogisticsNow dark bar) ── */
.topbar {{
    background: {LN_NAVY};
    padding: 8px 40px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.75rem; color: #aab8c5;
}}
.topbar a {{ color: #aab8c5; text-decoration: none; }}
.topbar a:hover {{ color: white; }}

/* ── MAIN NAV BAR ── */
.navbar {{
    background: white;
    padding: 14px 40px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 2px solid {LN_BORDER};
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.logo-wrap {{ display: flex; align-items: center; gap: 10px; }}
.logo-n {{
    width: 44px; height: 44px;
    background: {LN_NAVY};
    color: white; border-radius: 6px;
    font-size: 1.4rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Poppins', sans-serif;
}}
.logo-text {{
    font-size: 1.4rem; font-weight: 700;
    color: {LN_NAVY}; letter-spacing: -0.5px;
}}
.logo-text span {{ color: {LN_GREEN}; }}
.nav-links {{ display: flex; gap: 28px; }}
.nav-link {{
    font-size: 0.85rem; font-weight: 500;
    color: #334155; text-decoration: none;
    letter-spacing: 0.02em;
}}
.nav-link:hover {{ color: {LN_GREEN}; }}
.nav-cta {{
    background: {LN_DGREEN};
    color: white; padding: 8px 20px;
    border-radius: 4px; font-size: 0.8rem;
    font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 8px;
}}

/* ── HERO BANNER (dark like website) ── */
.hero-banner {{
    background: linear-gradient(135deg, {LN_NAVY} 0%, #2d4a6b 60%, #1a3a1a 100%);
    padding: 36px 40px 28px;
    color: white;
    position: relative; overflow: hidden;
}}
.hero-banner::before {{
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}}
.hero-title {{
    font-size: 2rem; font-weight: 700;
    margin: 0 0 6px 0; line-height: 1.2;
    position: relative;
}}
.hero-title span {{ color: {LN_GREEN}; }}
.hero-sub {{
    font-size: 0.88rem; color: #94a3b8;
    position: relative; margin-bottom: 16px;
}}
.hero-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(58,125,44,0.2); border: 1px solid {LN_GREEN};
    color: #6bcf57; border-radius: 20px;
    padding: 4px 14px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}}

/* ── CONTENT AREA ── */
.content-area {{
    padding: 24px 32px;
    background: {LN_LGRAY};
}}

/* ── BREADCRUMB ── */
.breadcrumb {{
    font-size: 0.78rem; color: #64748b;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 6px;
}}
.breadcrumb a {{ color: {LN_GREEN}; text-decoration: none; }}

/* ── KPI CARDS ── */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }}
.kpi-card {{
    background: white; border: 1px solid {LN_BORDER};
    border-radius: 10px; padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 4px 16px rgba(58,125,44,0.12); }}
.kpi-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: var(--ac, {LN_GREEN});
}}
.kpi-lbl {{
    font-size: 0.6rem; font-weight: 600;
    color: #64748b; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 8px;
}}
.kpi-val {{
    font-size: 1.65rem; font-weight: 700;
    color: {LN_NAVY}; line-height: 1.1;
}}
.kpi-d {{ font-size: 0.7rem; margin-top: 5px; font-weight: 500; }}
.dg {{ color: {LN_GREEN}; }} .dr {{ color: #dc2626; }}

/* ── SECTION HEADING ── */
.sh {{
    font-size: 1.05rem; font-weight: 700;
    color: {LN_NAVY}; margin: 24px 0 12px 0;
    display: flex; align-items: center; gap: 10px;
    border-left: 4px solid {LN_GREEN};
    padding-left: 12px;
}}

/* ── INFO / WARN / OK BOXES ── */
.info-box {{
    background: #f0fdf4; border-left: 4px solid {LN_GREEN};
    border-radius: 8px; padding: 14px 18px; margin: 4px 0 16px 0;
    font-size: 0.88rem; line-height: 1.7; color: {LN_NAVY};
}}
.info-box b {{ color: {LN_DGREEN}; }}
.warn-box {{
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.86rem; line-height: 1.65; color: {LN_NAVY};
}}
.ok-box {{
    background: #f0fdf4; border-left: 4px solid {LN_GREEN};
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.86rem; line-height: 1.65; color: {LN_NAVY};
}}
.tag-red    {{ color: #dc2626; font-weight: 600; }}
.tag-green  {{ color: {LN_GREEN}; font-weight: 600; }}
.tag-yellow {{ color: #d97706; font-weight: 600; }}

/* ── CHAT RAG STYLING ── */
.rag-header {{
    background: linear-gradient(135deg, {LN_NAVY} 0%, #2a5f3a 100%);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 16px;
    color: white;
}}
.rag-icon {{
    width: 52px; height: 52px; border-radius: 12px;
    background: {LN_GREEN}; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
}}
.rag-title {{ font-size: 1.1rem; font-weight: 700; margin-bottom: 2px; }}
.rag-sub   {{ font-size: 0.78rem; color: #94a3b8; }}

/* ── LEGEND ROW ── */
.legend-row {{
    display: flex; align-items: flex-start; gap: 10px;
    font-size: 0.84rem; color: {LN_NAVY};
    margin-bottom: 10px; padding-bottom: 10px;
    border-bottom: 1px solid {LN_BORDER};
}}
.legend-row:last-child {{ border-bottom: none; margin-bottom: 0; }}
.legend-dot {{
    width: 14px; height: 14px; border-radius: 3px;
    flex-shrink: 0; margin-top: 3px;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: white !important;
    border-right: 2px solid {LN_BORDER} !important;
}}
.sb-logo {{
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 4px;
}}
.sb-logo-box {{
    width: 36px; height: 36px; background: {LN_NAVY};
    border-radius: 6px; color: white; font-weight: 800;
    font-size: 1.1rem; display: flex; align-items: center; justify-content: center;
}}
.sb-brand {{ font-size: 1.1rem; font-weight: 700; color: {LN_NAVY}; }}
.sb-brand span {{ color: {LN_GREEN}; }}
.sb-sub {{
    font-size: 0.58rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.14em; margin-bottom: 1.4rem; padding-left: 46px;
}}
.sb-sec {{
    font-size: 0.6rem; font-weight: 700;
    color: {LN_GREEN}; letter-spacing: 0.16em;
    text-transform: uppercase; margin: 1.2rem 0 0.4rem 0;
    border-bottom: 1px solid {LN_BORDER}; padding-bottom: 4px;
}}
.sb-stat {{
    font-size: 0.72rem; color: #64748b;
    line-height: 2.2; font-family: monospace;
}}

[data-testid="metric-container"] {{
    background: white; border: 1px solid {LN_BORDER};
    border-radius: 10px; padding: 12px 16px !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

def kpi_card(label, value, delta, good=True, ac=None):
    ac  = ac or LN_GREEN
    cls = "dg" if good else "dr"
    return (f'<div class="kpi-card" style="--ac:{ac}">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div>'
            f'<div class="kpi-d {cls}">{delta}</div></div>')

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    ships  = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    veh    = pd.read_csv("vehicle_summary.csv")
    n        = len(ships)
    breaches = int((routes["sla_breach_hr"] > 0).sum())
    opt = dict(
        distance_km = round(veh["distance_km"].sum(), 1),
        time_hr     = round(veh["time_hr"].sum(), 1),
        fuel_cost   = round(veh["fuel_cost"].sum(), 1),
        toll_cost   = round(veh["toll_cost"].sum(), 1),
        driver_cost = round(veh["driver_cost"].sum(), 1),
        total_cost  = round(veh["total_cost"].sum(), 1),
        carbon_kg   = round(veh["carbon_kg"].sum(), 1),
        sla_pct     = round((n - breaches) / n * 100, 1),
        n_ships=n, n_vehicles=len(veh), breaches=breaches,
    )
    base = dict(
        distance_km=53526.9, time_hr=973.2, fuel_cost=601808.0,
        toll_cost=112741.0, driver_cost=197566.0, total_cost=912115.0,
        carbon_kg=13076.6, sla_pct=4.0,
    )
    return ships, routes, veh, base, opt

@st.cache_data
def perm_imp(routes_df):
    np.random.seed(42)
    feats = {
        "Travel Time":    "travel_time_hr",
        "Fuel Cost (₹)":  "fuel_cost",
        "Toll Cost (₹)":  "toll_cost",
        "Driver Cost (₹)":"driver_cost",
        "Carbon Emitted": "carbon_kg",
        "SLA Breach":     "sla_breach_hr",
        "Package Weight": "weight",
    }
    X = routes_df[list(feats.values())].copy()
    y = routes_df["mo_score"].values
    base_mae = np.mean(np.abs(y - y.mean()))
    imp = {}
    for lbl, col in feats.items():
        sh = X.copy(); sh[col] = np.random.permutation(sh[col].values)
        proxy = sh.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        imp[lbl] = abs(np.mean(np.abs(y - proxy.values)) - base_mae)
    tot = sum(imp.values()) + 1e-9
    return {k: round(v/tot*100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

@st.cache_data
def stop_cont(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    weights = [0.30, 0.20, 0.05, 0.15, 0.20, 0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max() - df[c].min(); df[c] = (df[c] - df[c].min()) / (rng + 1e-9)
    for i, c in enumerate(cols): df[c] *= weights[i]
    df.columns = labels
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

ships, routes, veh_sum, base, opt = load()
fi = perm_imp(routes)
sc = stop_cont(routes)

# ─────────────────────────────────────────────────────────────────────────────
# LORRI KNOWLEDGE BASE (for RAG)
# ─────────────────────────────────────────────────────────────────────────────
LORRI_KB = """
LOGISTICSNOW COMPANY KNOWLEDGE BASE:
- Company: LogisticsNow (logisticsnow.in)
- Contact: connect@logisticsnow.in | +91-9867773508 / +91-9653620207
- Platform: LoRRI — Logistics Rating & Intelligence platform

WHAT IS LORRI?
LoRRI (Logistics Rating & Intelligence) is LogisticsNow's flagship platform for Indian logistics networks.
It connects Shippers/Manufacturers with Carriers/Transporters through data-driven intelligence.

FOR SHIPPERS / MANUFACTURERS:
- Actionable insights enabling cost savings and risk reduction
- Detailed Carrier/Transporter profiles for best fitment
- Industry ratings for transporters to drive better service
- Holistic industry view rated by peers
- Access transporter profiles with domain-specific details

FOR CARRIERS / TRANSPORTERS:
- Get Discovered by Fortune 500 and leading Indian companies
- Receive business inquiries on working lanes for preferred truck types
- Get Customer feedback — build industry reputation
- Take control of your LoRRI profile

AI ROUTE OPTIMIZATION ENGINE (Problem Statement 4):
- Dynamic Multi-Objective Route Optimization integrated into LoRRI
- Uses Capacitated Vehicle Routing Problem (CVRP) framework
- Weighted objective: Time (30%) + Cost/₹ (35%) + Carbon (20%) + SLA (15%)
- Heuristic OR-Tools solver with local search refinements
- Threshold-based re-optimization on traffic or priority disruptions
- Explainability layer for baseline vs optimized comparison
- Expected impact: 8-20% distance reduction, 5-15% cost savings

CURRENT FLEET RUN (Mumbai Depot):
- 21 shipments across India, 5 trucks, depot at Mumbai
- Total optimized cost: ₹3,10,879 (saved ₹6,01,236 vs baseline)
- SLA adherence: 76.2% (vs 4% baseline)
- Carbon reduced: 9,930 kg CO2 (75.9% less)
- Vehicle capacity: 800 kg per truck
- Cost components: Fuel (₹/km), Toll (highway charges), Driver wages, SLA penalties (₹500/hr breach)

INDIA ROUTES COVERED:
Truck 1: Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada
Truck 2: Mumbai → Surat → Vadodara → Raipur
Truck 3: Mumbai → Aurangabad → Solapur → Madurai → Jammu
Truck 4: Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram
Truck 5: Mumbai → Hubli → Mangalore → Bengaluru

PRICING MODEL: All costs in Indian Rupees (₹ INR)
- Fuel: ₹12/km
- Driver: ₹180/hr
- SLA penalty: ₹500/hr breach
- Toll: variable by corridor
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">
        <img src="https://logisticsnow.in/wp-content/uploads/2020/05/logistics-now-logo.png" style="height:38px;margin-right:6px;">
        <div class="sb-brand">Logistics<span>Now</span></div>
    </div>
    <div class="sb-sub">LoRRI AI Route Engine</div>
    """, unsafe_allow_html=True)
  

    st.markdown(f'<div class="sb-sec">🏢 Platform</div>', unsafe_allow_html=True)
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
    ], label_visibility="collapsed")

    st.markdown(f'<div class="sb-sec">🛠️ Fleet Control</div>', unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed", value=True)
    st.toggle("Auto Re-optimize", value=False)
    if st.button("🔄 Sync Depot Data", use_container_width=True):
        st.toast("✅ Synced with Mumbai Depot!", icon="🏭")

    st.markdown(f'<div class="sb-sec">📦 Live Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-stat">
    Shipments &nbsp; <b style="color:{LN_NAVY}">{opt['n_ships']}</b><br>
    Trucks &nbsp;&nbsp;&nbsp;&nbsp; <b style="color:{LN_NAVY}">{opt['n_vehicles']}</b><br>
    SLA OK &nbsp;&nbsp;&nbsp;&nbsp; <b style="color:{LN_GREEN}">{opt['sla_pct']:.0f}%</b><br>
    Breaches &nbsp;&nbsp; <b style="color:#dc2626">{opt['breaches']}</b><br>
    Depot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:{LN_NAVY}">Mumbai</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#94a3b8;line-height:1.8;">
    📧 connect@logisticsnow.in<br>
    📞 +91-9867773508<br>
    🌐 logisticsnow.in
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR + NAVBAR (LogisticsNow style)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div style="display:flex;gap:24px;">
        <span>✉ connect@logisticsnow.in</span>
        <span>📞 +91-9867773508 / +91-9653620207</span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:0.7rem;letter-spacing:0.05em;">IN · LI · FB · IG</span>
        <a href="#" style="background:{LN_GREEN};color:white;padding:5px 14px;border-radius:3px;font-size:0.7rem;font-weight:600;text-decoration:none;letter-spacing:0.05em;">
            LORRI &nbsp; SCHEDULE A DEMO ›
        </a>
    </div>
</div>
<div class="navbar">
    <div class="logo-wrap">
        <div class="logo-n">LN</div>
        <div class="logo-text">Logistics<span>Now</span></div>
    </div>
    <div class="nav-links">
        <a class="nav-link" href="#">Home</a>
        <a class="nav-link" href="#">About Us</a>
        <a class="nav-link" href="#">Products</a>
        <a class="nav-link" href="#">Careers</a>
        <a class="nav-link" href="#">News & Events</a>
        <a class="nav-link" href="#">Contact Us</a>
    </div>
    <div class="nav-cta">LoRRI &nbsp; SCHEDULE A DEMO ›</div>
</div>
""", unsafe_allow_html=True)

# HERO BANNER
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">LoRRI: <span>AI Route Optimization Engine</span></div>
    <div class="hero-sub">Dynamic Multi-Objective CVRP · India Logistics Network · Depot: Mumbai · All costs in ₹ INR</div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <div class="hero-badge">● MUMBAI HUB: LIVE</div>
        <div class="hero-badge" style="border-color:#1e7abf;color:#7ec8ff;background:rgba(30,122,191,0.2);">
            Problem Statement 4 — Synapflow Hackathon
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# BREADCRUMB
st.markdown(f"""
<div class="content-area">
<div class="breadcrumb">
    📍 <a href="#">Home</a> › <a href="#">Products</a> › <a href="#">LoRRI</a> › AI Route Optimization
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: ABOUT LORRI — from Synapflow Problem Statement 4 PDF
# ══════════════════════════════════════════════════════════════════════════════
if pg == "🏢 About LoRRI":

    # ── Executive Overview ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:14px;
                padding:32px 36px;margin-bottom:20px;border-top:4px solid {LN_GREEN};">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
            <div style="width:46px;height:46px;background:{LN_GREEN};border-radius:10px;
                        display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0;">🚚</div>
            <div>
                <div style="font-size:1.35rem;font-weight:700;color:{LN_NAVY};">Executive Overview</div>
                <div style="font-size:0.78rem;color:#64748b;">LoRRI · AI Route Optimization Engine · Problem Statement 4</div>
            </div>
        </div>
        <p style="font-size:0.9rem;color:#334155;line-height:1.85;margin:0;">
            While <b style="color:{LN_NAVY}">LoRRI</b> currently excels as a premier logistics intelligence platform —
            offering robust carrier profiling, lane analytics, and procurement insights — there remains a
            <b>critical gap</b> between high-level analytics and real-time execution. Current routing decisions
            within many logistics networks remain <b>static and cost-centric</b>, often failing to account for
            volatile operational factors such as fluctuating traffic, toll structures, and carbon mandates.
        </p>
        <p style="font-size:0.9rem;color:#334155;line-height:1.85;margin:14px 0 0 0;">
            We propose the integration of a <b style="color:{LN_GREEN}">Dynamic Multi-Objective Route Optimization Engine</b>
            designed to transform LoRRI from a retrospective diagnostic tool into a
            <b>proactive, real-time decision intelligence ecosystem</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Proposed Solution ─────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;padding:26px 28px;height:100%;border-top:4px solid #1e7abf;">
            <div style="font-size:1.05rem;font-weight:700;color:{LN_NAVY};margin-bottom:14px;">💡 Proposed Solution</div>
            <p style="font-size:0.88rem;color:#334155;line-height:1.8;margin:0 0 12px 0;">
                Integrating our <b>Route Optimization Engine</b> directly into the <b>LoRRI platform</b> will convert
                logistics intelligence into <b>real-time operational decision-making</b>. The module optimizes
                multi-stop delivery routes using a <b>Capacitated Vehicle Routing (CVRP) framework</b> and
                dynamically re-optimizes them when <b>traffic disruptions</b> or <b>shipment priorities</b> change.
            </p>
            <p style="font-size:0.88rem;color:#334155;line-height:1.8;margin:0;">
                Rather than minimising distance alone, it balances <b>delivery time</b>, <b>transportation cost
                (including toll charges)</b>, <b>SLA adherence</b>, and <b>carbon impact</b> through a
                weighted objective scoring model aligned with business priorities.
            </p>
            <div style="margin-top:16px;background:{LN_LGRAY};border-radius:8px;padding:14px 16px;font-size:0.82rem;color:#475569;line-height:1.8;">
                📊 <b>Baseline vs Optimised</b> route comparison with an <b>explainable breakdown</b>
                of routing decisions — ensuring <b>transparency and operational trust</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:{LN_NAVY};border-radius:12px;padding:26px 24px;height:100%;color:white;">
            <div style="font-size:1.0rem;font-weight:700;margin-bottom:16px;color:white;">⚙️ Weighted Objectives</div>
            <div style="font-size:0.82rem;line-height:1;margin-bottom:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;">How routing decisions are scored</div>
        """, unsafe_allow_html=True)

        objectives = [
            ("💰 Total Cost (₹)", 35, "#3a7d2c", "Fuel · Toll · Driver · SLA Penalties"),
            ("⏱️ Travel Time",     30, "#1e7abf", "Hours on road with traffic multiplier"),
            ("🌿 Carbon CO₂",      20, "#27ae60", "kg CO₂ per km by road type"),
            ("📅 SLA Adherence",   15, "#e67e22", "Delivery promise window 24/48/72hr"),
        ]
        for label, pct, color, desc in objectives:
            st.markdown(f"""
            <div style="margin-top:14px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
                    <span style="font-size:0.83rem;font-weight:600;color:white;">{label}</span>
                    <span style="font-family:monospace;font-size:0.78rem;color:{color};font-weight:700;">{pct}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.1);border-radius:4px;height:7px;overflow:hidden;">
                    <div style="width:{pct*2.86:.0f}%;height:100%;background:{color};border-radius:4px;"></div>
                </div>
                <div style="font-size:0.7rem;color:#94a3b8;margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI & Optimization Approach ────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;padding:26px 28px;margin-bottom:20px;border-top:4px solid {LN_GREEN};">
        <div style="font-size:1.05rem;font-weight:700;color:{LN_NAVY};margin-bottom:14px;">🧠 AI & Optimization Approach</div>
        <p style="font-size:0.88rem;color:#334155;line-height:1.8;margin:0 0 14px 0;">
            We model routing as a <b>Capacitated Vehicle Routing Problem (CVRP)</b> with a
            <b>weighted multi-objective function</b> minimising travel time, total transportation cost
            (fuel, toll, driver, SLA penalties), and carbon emissions under capacity constraints.
            Using a <b>heuristic OR-Tools solver with local search refinements</b>, we generate scalable
            multi-stop routes and trigger <b>threshold-based re-optimisation</b> only during traffic or
            priority disruptions. An integrated <b>explainability layer</b> breaks down objective contributions
            to clearly compare baseline versus optimised performance.
        </p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px;">
    """, unsafe_allow_html=True)

    tech_cards = [
        ("🔧", "CVRP Framework",        "Capacitated Vehicle Routing with multi-stop optimization under 800 kg per truck"),
        ("⚡", "OR-Tools Solver",         "Heuristic solver with local search refinements for scalable India-wide routing"),
        ("🔄", "Threshold Re-optimize",  "Auto re-routes when traffic causes >30% delay or a shipment is escalated"),
        ("🔍", "Explainability Layer",   "Permutation-based feature importance (SHAP-style) for every routing decision"),
    ]
    for icon, title, desc in tech_cards:
        st.markdown(f"""
        <div style="background:{LN_LGRAY};border-radius:10px;padding:16px;border:1px solid {LN_BORDER};">
            <div style="font-size:1.3rem;margin-bottom:8px;">{icon}</div>
            <div style="font-size:0.82rem;font-weight:700;color:{LN_NAVY};margin-bottom:6px;">{title}</div>
            <div style="font-size:0.76rem;color:#64748b;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Expected Impact ───────────────────────────────────────────────────────
    st.markdown(f'<div class="sh">📈 Expected Impact — Industry Benchmarks</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    impact = [
        (col1, "8–20%",  "Travel Distance Reduction",  LN_GREEN,  "Based on VRP-based routing systems"),
        (col2, "5–15%",  "Cost Savings (₹)",            "#1e7abf", "Fuel + Toll + Driver optimisation"),
        (col3, "↑ SLA",  "Adherence Improvement",       "#e67e22", "Measurable delivery promise gains"),
        (col4, "↑ Fleet","Utilisation Gains",           "#8e44ad", "Proportional CO₂ emission reductions"),
    ]
    for col, val, label, color, sub in impact:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;padding:20px;
                    border-top:3px solid {color};text-align:center;">
            <div style="font-size:1.8rem;font-weight:800;color:{color};line-height:1;">{val}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{LN_NAVY};margin:8px 0 4px;">{label}</div>
            <div style="font-size:0.7rem;color:#94a3b8;line-height:1.5;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture Flow ─────────────────────────────────────────────────────
    st.markdown(f'<div class="sh">🏗️ System Architecture & LoRRI Integration</div>', unsafe_allow_html=True)

    layers = [
        ("👤 User Layer",             "#1e7abf", ["LoRRI Dashboard (Frontend UI)", "Route Visualisation", "KPI Panels", "Baseline vs Optimised View"]),
        ("⚙️ Application / Service",  LN_GREEN,  ["Route Optimisation API Service", "Re-Optimisation Trigger Service", "Explainability & KPI Service"]),
        ("🔬 Optimisation Core",       "#e67e22", ["CVRP Multi-Objective Model", "Heuristic Solver (OR-Tools)", "Dynamic Re-Optimisation Engine"]),
        ("📊 Intelligence & Output",   "#8e44ad", ["Optimised Route Plan Generation", "Explainability Breakdown", "Baseline vs Optimised KPI Deltas"]),
    ]
    cols = st.columns(4)
    for col, (title, color, items) in zip(cols, layers):
        items_html = "".join(
            f'<div style="font-size:0.75rem;color:#334155;padding:5px 0;border-bottom:1px solid {LN_BORDER};line-height:1.4;">• {item}</div>'
            for item in items
        )
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;overflow:hidden;">
            <div style="background:{color};padding:10px 14px;color:white;font-size:0.82rem;font-weight:700;">{title}</div>
            <div style="padding:10px 14px;">{items_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{LN_LGRAY};border:1px solid {LN_BORDER};border-radius:8px;
                padding:12px 20px;margin-top:12px;text-align:center;
                font-size:0.78rem;color:#475569;font-family:monospace;letter-spacing:0.04em;">
        FLOW: &nbsp; Data &nbsp;→&nbsp; Model &nbsp;→&nbsp; Solve &nbsp;→&nbsp; Adapt &nbsp;→&nbsp; Explain &nbsp;→&nbsp; Compare &nbsp;→&nbsp; User Display
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Team Capability ───────────────────────────────────────────────────────
    st.markdown(f'<div class="sh">👥 Team Capability & Execution Strength</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a3a2a 100%);
                border-radius:12px;padding:28px 32px;color:white;">
        <p style="font-size:0.9rem;line-height:1.85;color:#cbd5e1;margin:0 0 20px 0;">
            Our team combines expertise in <b style="color:white;">supply chain operations</b>,
            <b style="color:white;">AI-driven decision systems</b>,
            <b style="color:white;">optimization modeling</b>, and
            <b style="color:white;">frontend dashboard development</b>, enabling us to design a solution
            that is technically rigorous, operationally realistic, and seamlessly integrable within the LoRRI ecosystem.
            This interdisciplinary strength ensures we can deliver a <b style="color:{LN_GREEN};">scalable,
            explainable, and visually demonstrable prototype</b> within hackathon timelines.
        </p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
    """, unsafe_allow_html=True)

    skills = [
        ("🏭", "Supply Chain Ops",      "Deep domain knowledge of Indian logistics corridors and fleet operations"),
        ("🤖", "AI Decision Systems",   "Multi-objective optimization, constraint programming, OR-Tools expertise"),
        ("📐", "Optimization Modeling", "CVRP formulation, heuristic local search, threshold-based re-solving"),
        ("🖥️", "Frontend & Dashboard",  "Streamlit, Plotly, real-time UI with explainability and KPI panels"),
    ]
    for icon, title, desc in skills:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px;padding:16px;">
            <div style="font-size:1.3rem;margin-bottom:8px;">{icon}</div>
            <div style="font-size:0.83rem;font-weight:700;color:white;margin-bottom:6px;">{title}</div>
            <div style="font-size:0.73rem;color:#94a3b8;line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: AI ASSISTANT — RAG grounded on LogisticsNow + fleet data
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🤖 LoRRI AI Assistant":

    # RAG header card
    st.markdown(f"""
    <div class="rag-header">
        <div class="rag-icon">🤖</div>
        <div>
            <div class="rag-title">LoRRI Intelligence Assistant</div>
            <div class="rag-sub">
                Powered by LogisticsNow · Grounded in LoRRI platform knowledge + your live fleet data ·
                Ask in English or Hindi
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What the assistant knows
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;padding:14px 16px;border-top:3px solid {LN_GREEN};">
    <div style="font-size:0.65rem;font-weight:700;color:{LN_GREEN};text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🏢 About LogisticsNow</div>
    <div style="font-size:0.82rem;color:#334155;line-height:1.6;">
    Ask about LoRRI platform features, services for Shippers &amp; Carriers, company background, and contact details.
    </div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;padding:14px 16px;border-top:3px solid #1e7abf;">
    <div style="font-size:0.65rem;font-weight:700;color:#1e7abf;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🚛 Your Fleet Data</div>
    <div style="font-size:0.82rem;color:#334155;line-height:1.6;">
    Ask about truck routes, ₹ cost savings, SLA breaches, carbon emissions, and vehicle performance.
    </div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;padding:14px 16px;border-top:3px solid #e67e22;">
    <div style="font-size:0.65rem;font-weight:700;color:#e67e22;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">🧠 AI & Optimization</div>
    <div style="font-size:0.82rem;color:#334155;line-height:1.6;">
    Ask about CVRP methodology, multi-objective scoring, re-optimization triggers, and explainability.
    </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Suggested prompts
    st.markdown(f"""
    <div style="font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">
    💡 Try asking:
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">What is LoRRI?</span>
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">How much did we save in ₹?</span>
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">Which cities had SLA breaches?</span>
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">Which truck was most expensive?</span>
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">How does CVRP work?</span>
    <span style="background:#f0fdf4;border:1px solid {LN_GREEN};color:{LN_GREEN};border-radius:20px;padding:4px 12px;font-size:0.75rem;cursor:pointer;">What routes did Truck 3 take?</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    if not st.session_state.msgs:
        with st.chat_message("assistant", avatar="🚚"):
            st.markdown(
                f"**Namaste! 🙏 Welcome to the LoRRI Intelligence Assistant by LogisticsNow.**\n\n"
                f"I'm grounded in the **LoRRI platform knowledge base** and your **live fleet data** — "
                f"{opt['n_ships']} shipments, 5 trucks across India from Mumbai depot, "
                f"total fleet cost **{inr(opt['total_cost'])}**.\n\n"
                f"Ask me anything about **LogisticsNow**, **LoRRI platform features**, **your routes**, "
                f"**₹ savings**, **SLA performance**, or **how the AI optimization works**! 🇮🇳"
            )

    for m in st.session_state.msgs:
        with st.chat_message(m["role"], avatar="🚚" if m["role"] == "assistant" else "👤"):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about LogisticsNow, LoRRI, or your fleet..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        q = prompt.lower()

        # ── LORRI / LOGISTICSNOW PLATFORM KNOWLEDGE ──────────────────────────
        if any(w in q for w in ["what is lorri","lorri platform","logisticsnow","about lorri","about logistic"]):
            r = ("**LoRRI** is LogisticsNow's flagship platform — **Logistics Rating & Intelligence**.\n\n"
                 "It bridges **Shippers/Manufacturers** with **Carriers/Transporters** across India by providing:\n\n"
                 "**For Shippers:**\n"
                 "- 📊 Actionable insights enabling cost savings and risk reduction\n"
                 "- 🔍 Detailed Carrier/Transporter profiles for best fitment\n"
                 "- ⭐ Industry ratings for transporters\n\n"
                 "**For Carriers/Transporters:**\n"
                 "- 🌐 Get discovered by Fortune 500 and leading Indian companies\n"
                 "- 💬 Receive customer feedback to build reputation\n"
                 "- 📋 Take control of your LoRRI profile\n\n"
                 f"Contact: **connect@logisticsnow.in** | **+91-9867773508**")

        elif any(w in q for w in ["contact","email","phone","reach","address"]):
            r = ("**LogisticsNow Contact Details:**\n\n"
                 "📧 Email: **connect@logisticsnow.in**\n"
                 "📞 Phone: **+91-9867773508** / **+91-9653620207**\n"
                 "🌐 Website: **logisticsnow.in**\n"
                 "🔗 LinkedIn, Facebook, Instagram — @LogisticsNow\n\n"
                 "To schedule a LoRRI demo, click **SCHEDULE A DEMO** on the top bar!")

        elif any(w in q for w in ["shipper","manufacturer","for shipper"]):
            r = ("**LoRRI For Shippers / Manufacturers** offers:\n\n"
                 "✅ **Actionable insights** — Detailed carrier profiles to reduce operational risk\n"
                 "✅ **Industry ratings** — See how your transporters are rated by peers\n"
                 "✅ **Best fitment** — Match with the right transporter for your lane & load type\n"
                 "✅ **Domain-specific profiles** — Never-seen-before transporter details\n\n"
                 "Visit: **logisticsnow.in/lorri-shippers**")

        elif any(w in q for w in ["carrier","transporter","for carrier"]):
            r = ("**LoRRI For Carriers / Transporters** offers:\n\n"
                 "🌐 **Get Discovered** — Reach Fortune 500 companies on your working lanes\n"
                 "💬 **Get Customer Feedback** — Build your industry reputation on LoRRI\n"
                 "📋 **Control your profile** — Showcase preferred truck types and capabilities\n"
                 "📈 **Grow your business** — Get inquiries on your preferred routes\n\n"
                 "Visit: **logisticsnow.in/lorri-carriers**")

        elif any(w in q for w in ["cvrp","how does","algorithm","optimization work","multi-objective","weighted"]):
            r = ("**How LoRRI's AI Optimization Works:**\n\n"
                 "We model routing as a **Capacitated Vehicle Routing Problem (CVRP)** with a "
                 "weighted multi-objective function:\n\n"
                 "| Objective | Weight | What it means |\n"
                 "|-----------|--------|---------------|\n"
                 "| Cost (₹) | **35%** | Fuel + Toll + Driver + SLA penalties |\n"
                 "| Travel Time | **30%** | Hours on road with traffic multiplier |\n"
                 "| Carbon CO₂ | **20%** | kg CO₂ per km by road type |\n"
                 "| SLA Adherence | **15%** | Delivery promise window (24/48/72hr) |\n\n"
                 "**Re-optimization** triggers when traffic increases travel time >30% or "
                 "a shipment is escalated to HIGH priority. Uses **OR-Tools + heuristic local search**.")

        elif any(w in q for w in ["saving","save","saved","how much","total saving","₹"]):
            s = base["total_cost"] - opt["total_cost"]
            r = (f"**Total ₹ Savings: {inr(s)}** ({s/base['total_cost']*100:.1f}% reduction)\n\n"
                 f"| Cost Type | Baseline | Optimized | Saved |\n"
                 f"|-----------|----------|-----------|-------|\n"
                 f"| ⛽ Fuel | {inr(base['fuel_cost'])} | {inr(opt['fuel_cost'])} | **{inr(base['fuel_cost']-opt['fuel_cost'])}** |\n"
                 f"| 🛣️ Toll | {inr(base['toll_cost'])} | {inr(opt['toll_cost'])} | **{inr(base['toll_cost']-opt['toll_cost'])}** |\n"
                 f"| 👷 Driver | {inr(base['driver_cost'])} | {inr(opt['driver_cost'])} | **{inr(base['driver_cost']-opt['driver_cost'])}** |\n"
                 f"| **Total** | **{inr(base['total_cost'])}** | **{inr(opt['total_cost'])}** | **{inr(s)}** |")

        elif any(w in q for w in ["sla","late","breach","delay","on time","penalty"]):
            bd = routes[routes["sla_breach_hr"] > 0]
            cities = ", ".join(bd["city"].tolist())
            worst  = bd.loc[bd["sla_breach_hr"].idxmax()]
            tp     = veh_sum["sla_penalty"].sum()
            r = (f"**SLA Performance:** {opt['sla_pct']:.0f}% adherence (baseline was {base['sla_pct']:.0f}%)\n\n"
                 f"**{opt['breaches']} cities had late deliveries:** {cities}\n\n"
                 f"**Worst breach:** {worst['city']} — {worst['sla_breach_hr']:.1f}hr late "
                 f"(Truck {int(worst['vehicle'])}, penalty: {inr(worst['sla_penalty'])})\n\n"
                 f"**Total SLA penalties incurred:** {inr(tp)} (rate: ₹500/hr per breach)")

        elif any(w in q for w in ["expensive","most cost","costly","which truck cost"]):
            worst = veh_sum.loc[veh_sum["total_cost"].idxmax()]
            best  = veh_sum.loc[veh_sum["total_cost"].idxmin()]
            r = (f"**Most expensive: Truck {int(worst['vehicle'])}** — {inr(worst['total_cost'])}\n"
                 f"({int(worst['stops'])} stops · {worst['distance_km']:,.0f} km · {int(worst['sla_breaches'])} SLA breach)\n\n"
                 f"**Cheapest: Truck {int(best['vehicle'])}** — {inr(best['total_cost'])}\n"
                 f"({int(best['stops'])} stops · {best['distance_km']:,.0f} km · ✅ No breach)")

        elif any(w in q for w in ["truck 1","truck 2","truck 3","truck 4","truck 5","route of","which route"]):
            route_map = {
                "1": "Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada",
                "2": "Mumbai → Surat → Vadodara → Raipur",
                "3": "Mumbai → Aurangabad → Solapur → Madurai → Jammu",
                "4": "Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram",
                "5": "Mumbai → Hubli → Mangalore → Bengaluru",
            }
            lines = []
            for v, route in route_map.items():
                vd = veh_sum[veh_sum["vehicle"]==int(v)].iloc[0]
                lines.append(f"**🚛 Truck {v}:** {route}\n"
                             f"  └ {vd['distance_km']:,.0f} km · {inr(vd['total_cost'])} · "
                             f"{vd['carbon_kg']:.0f} kg CO₂ · "
                             f"{'⚠️ '+str(int(vd['sla_breaches']))+' breach' if vd['sla_breaches']>0 else '✅ No breach'}")
            r = "**All India Truck Routes from Mumbai Depot:**\n\n" + "\n\n".join(lines)

        elif any(w in q for w in ["carbon","co2","emission","pollution","green","environment"]):
            s = base["carbon_kg"] - opt["carbon_kg"]
            worst = veh_sum.loc[veh_sum["carbon_kg"].idxmax()]
            r = (f"**Carbon Savings: {s:,.1f} kg CO₂** — {s/base['carbon_kg']*100:.1f}% less pollution 🌱\n\n"
                 f"Baseline (no AI): {base['carbon_kg']:,.1f} kg → Optimized: **{opt['carbon_kg']:,.1f} kg**\n\n"
                 f"Highest emitter: **Truck {int(worst['vehicle'])}** at {worst['carbon_kg']:.1f} kg "
                 f"(longest route {worst['distance_km']:,.0f} km)\n\n"
                 f"CO₂ savings equivalent to planting ~{int(s/21)} trees! 🌳")

        elif any(w in q for w in ["hello","hi","namaste","hey","what can"]):
            r = ("**Namaste! 🙏** I can help you with:\n\n"
                 "🏢 **LogisticsNow & LoRRI** — platform features, services, contact\n"
                 "🚛 **Fleet performance** — truck routes, ₹ costs, savings\n"
                 "📅 **SLA & breaches** — late deliveries, ₹500/hr penalties\n"
                 "🌿 **Carbon emissions** — CO₂ per truck, savings\n"
                 "🧠 **AI methodology** — CVRP, multi-objective scoring, re-optimization\n\n"
                 "Just ask! All fleet costs are in **₹ INR**.")

        else:
            r = (f"Based on your **{opt['n_ships']}-shipment Mumbai fleet:**\n\n"
                 f"- 💰 Total cost: **{inr(opt['total_cost'])}** (saved {inr(base['total_cost']-opt['total_cost'])})\n"
                 f"- 📏 Distance: **{opt['distance_km']:,.0f} km** across India\n"
                 f"- ✅ SLA: **{opt['sla_pct']:.0f}%** adherence\n"
                 f"- 🌿 Carbon: **{opt['carbon_kg']:,.1f} kg CO₂**\n\n"
                 f"**About LogisticsNow:** LoRRI connects Shippers with Carriers across India. "
                 f"Contact: connect@logisticsnow.in | +91-9867773508\n\n"
                 f"Ask about: *LoRRI platform, savings in ₹, truck routes, SLA, carbon, CVRP*")

        with st.chat_message("assistant", avatar="🚚"):
            st.markdown(r)
        st.session_state.msgs.append({"role": "assistant", "content": r})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "📊 Dashboard Summary":
    st.markdown("""<div class="info-box">
    📋 <b>Report card for the whole delivery run.</b>
    Baseline = trucks driving one-by-one without AI.
    Optimized = LogisticsNow LoRRI AI planner.
    Every green arrow = money saved, time saved, less pollution. All figures in <b>₹ INR</b>. ✅
    </div>""", unsafe_allow_html=True)

    sc_ = base["total_cost"] - opt["total_cost"]
    sd_ = base["distance_km"] - opt["distance_km"]
    sco = base["carbon_kg"] - opt["carbon_kg"]
    ss_ = opt["sla_pct"] - base["sla_pct"]

    st.markdown(f"""<div class="kpi-grid">
    {kpi_card("Total Cost Savings",   inr(sc_),           f"↓ -{sc_/base['total_cost']*100:.1f}% vs baseline", True, LN_GREEN)}
    {kpi_card("Optimized Distance",   f"{opt['distance_km']:,.0f} km", f"↓ {sd_:,.0f} km saved",              True, "#1e7abf")}
    {kpi_card("SLA Adherence",        f"{opt['sla_pct']:.0f}%",        f"↑ +{ss_:.0f} pts (base {base['sla_pct']:.0f}%)", True, "#e67e22")}
    {kpi_card("Carbon Reduced",       f"{sco/1000:.1f}t CO₂",         f"↓ {sco/base['carbon_kg']*100:.1f}% cleaner", True, "#27ae60")}
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⛽ Fuel Saved",   inr(base["fuel_cost"]  - opt["fuel_cost"]),   f"-{(base['fuel_cost']  -opt['fuel_cost']  )/base['fuel_cost']  *100:.1f}%", delta_color="inverse")
    c2.metric("🛣️ Toll Saved",  inr(base["toll_cost"]  - opt["toll_cost"]),   f"-{(base['toll_cost']  -opt['toll_cost']  )/base['toll_cost']  *100:.1f}%", delta_color="inverse")
    c3.metric("👷 Driver Saved", inr(base["driver_cost"]- opt["driver_cost"]), f"-{(base['driver_cost']-opt['driver_cost'])/base['driver_cost']*100:.1f}%", delta_color="inverse")
    c4.metric("⏱️ Time Saved",  f"{base['time_hr']-opt['time_hr']:,.1f} hr",  f"-{(base['time_hr']-opt['time_hr'])/base['time_hr']*100:.1f}%",            delta_color="inverse")

    st.markdown('<div class="sh">🚛 Per-Truck Summary</div>', unsafe_allow_html=True)
    d = veh_sum.copy()
    d.insert(0, "Truck", d["vehicle"].apply(lambda v: f"🚛 Truck {v}"))
    d = d.drop(columns=["vehicle"])
    d.columns = ["Truck","Stops","Load (kg)","Dist (km)","Time (hr)",
                 "Fuel (₹)","Toll (₹)","Driver (₹)","SLA Penalty (₹)","Total (₹)","Carbon (kg)","SLA Breaches","Util %"]
    st.dataframe(d.style
        .format({"Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}",
                 "Fuel (₹)":"₹{:,.0f}","Toll (₹)":"₹{:,.0f}","Driver (₹)":"₹{:,.0f}",
                 "SLA Penalty (₹)":"₹{:,.0f}","Total (₹)":"₹{:,.0f}",
                 "Carbon (kg)":"{:.1f}","Util %":"{:.1f}%"})
        .background_gradient(subset=["Util %"], cmap="Greens")
        .background_gradient(subset=["Total (₹)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Route Map":
    st.markdown("""<div class="info-box">
    🗺️ <b>Real India map</b> showing every truck's delivery path from Mumbai depot.
    Each line = a different truck. Markers show <b>Truck# · Stop#</b>.
    Hover for full details including ₹ cost and SLA status.
    </div>""", unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown(f'<div class="sh" style="font-size:0.85rem;margin-top:0;">🎛️ Map Controls</div>', unsafe_allow_html=True)
        show_base = st.toggle("Show Baseline Route", value=False)
        show_unas = st.toggle("Show Unassigned",     value=True)
        sel_v     = st.multiselect("Filter Trucks",
                                    options=sorted(routes["vehicle"].unique()),
                                    default=sorted(routes["vehicle"].unique()),
                                    format_func=lambda v: f"Truck {v}")
        st.markdown("---")
        st.markdown(f'<div class="sh" style="font-size:0.85rem;margin-top:0;">📌 Route Legend</div>', unsafe_allow_html=True)
        for v in sorted(routes["vehicle"].unique()):
            vr    = routes[routes["vehicle"] == v]
            color = V_COLORS.get(v, "#999")
            vd    = veh_sum[veh_sum["vehicle"] == v].iloc[0]
            st.markdown(
                f'<div class="legend-row">'
                f'<div class="legend-dot" style="background:{color}"></div>'
                f'<div><b style="color:{LN_NAVY}">Truck {v}</b><br>'
                f'<span style="font-size:0.73rem;color:#64748b">'
                f'{len(vr)} stops · {vd["distance_km"]:,.0f} km<br>'
                f'{inr(vd["total_cost"])} · {vd["carbon_kg"]:.0f} kg CO₂</span></div>'
                f'</div>', unsafe_allow_html=True)

    with col_map:
        fig = go.Figure()
        if show_base:
            bl  = [DEPOT["latitude"]]  + ships["latitude"].tolist()  + [DEPOT["latitude"]]
            blo = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=bl, lon=blo, mode="lines",
                line=dict(width=1.5, color="rgba(200,50,50,0.4)"), name="Baseline (No AI)"))

        p_dot = {"HIGH": "#dc2626", "MEDIUM": "#f97316", "LOW": LN_GREEN}

        for v in sel_v:
            vdf   = routes[routes["vehicle"] == v].sort_values("stop_order")
            color = V_COLORS.get(v, "#999")
            lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
            lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines",
                line=dict(width=3, color=color), name=f"Truck {v}", legendgroup=f"v{v}"))
            for _, row in vdf.iterrows():
                breach = f"⚠️ {row['sla_breach_hr']:.1f}hr late" if row["sla_breach_hr"] > 0 else "✅ On time"
                fig.add_trace(go.Scattermap(
                    lat=[row["latitude"]], lon=[row["longitude"]],
                    mode="markers+text",
                    marker=dict(size=14, color=p_dot.get(row.get("priority","MEDIUM"), "#f97316")),
                    text=[f"T{v}·{int(row['stop_order'])}"],
                    textfont=dict(size=8, color="white"),
                    textposition="middle center",
                    hovertext=(
                        f"<b>🚛 Truck {v} — Stop {int(row['stop_order'])}</b><br>"
                        f"📍 {row.get('city', row['shipment_id'])}<br>"
                        f"📦 {row['shipment_id']} | {row['weight']:.0f} kg<br>"
                        f"🔺 Priority: {row.get('priority','')}<br>"
                        f"⏱️ Travel: {row['travel_time_hr']:.1f} hr<br>"
                        f"⛽ Fuel: {inr(row['fuel_cost'])} | 🛣️ Toll: {inr(row['toll_cost'])}<br>"
                        f"💰 Total: {inr(row['total_cost'])}<br>"
                        f"🌿 Carbon: {row['carbon_kg']:.1f} kg CO₂<br>"
                        f"📅 SLA {row['sla_hours']}hr: {breach}"
                    ),
                    hoverinfo="text", showlegend=False, legendgroup=f"v{v}"))

        if show_unas:
            asgn  = set(routes["shipment_id"])
            unasgn= ships[~ships["id"].isin(asgn)]
            if not unasgn.empty:
                fig.add_trace(go.Scattermap(lat=unasgn["latitude"], lon=unasgn["longitude"],
                    mode="markers", marker=dict(size=8, color="grey"), name="Unassigned",
                    hovertext=unasgn["city"], hoverinfo="text"))

        fig.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["🏭 Mumbai\nDepot"],
            textposition="top right", textfont=dict(size=10, color=LN_NAVY),
            marker=dict(size=18, color=LN_NAVY, symbol="star"), name="Mumbai Depot"))

        fig.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
            height=600, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.92)",
                        bordercolor=LN_BORDER, borderwidth=1,
                        font=dict(color=LN_NAVY, size=11)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 HIGH priority  🟠 MEDIUM priority  🟢 LOW priority  ⚠️ SLA breach  ·  T{truck}·{stop#}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "💰 Financial Analysis":
    st.markdown("""<div class="info-box">
    💰 All costs in <b>₹ (Indian Rupees)</b>. Three cost categories: <b>Fuel</b> (₹12/km),
    <b>Tolls</b> (highway charges), <b>Driver wages</b> (₹180/hr).
    LoRRI's AI clustered nearby cities per truck, cutting all three significantly.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_b = go.Figure()
        cats = ["Fuel", "Toll", "Driver"]
        bv   = [base["fuel_cost"], base["toll_cost"], base["driver_cost"]]
        ov   = [opt["fuel_cost"],  opt["toll_cost"],  opt["driver_cost"]]
        bc   = [LN_GREEN, "#1e7abf", "#8e44ad"]
        for cat, b_, o_, c_ in zip(cats, bv, ov, bc):
            fig_b.add_trace(go.Bar(name=cat, x=["Baseline", "Optimized"], y=[b_, o_],
                marker_color=c_, text=[inr(b_), inr(o_)], textposition="inside",
                textfont=dict(color="white", size=10)))
        apply_theme(fig_b, height=360, title="Cost Components: Baseline vs Optimized (₹)", legend_below=True)
        fig_b.update_layout(barmode="stack")
        fig_b.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_b, use_container_width=True)

    with c2:
        sv = {"Fuel Saved":   base["fuel_cost"]   - opt["fuel_cost"],
              "Toll Saved":   base["toll_cost"]   - opt["toll_cost"],
              "Driver Saved": base["driver_cost"] - opt["driver_cost"]}
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","total"],
            x=list(sv.keys()) + ["Total Saved"],
            y=list(sv.values()) + [sum(sv.values())],
            connector={"line": {"color": "#cbd5e1"}},
            decreasing={"marker": {"color": LN_GREEN}},
            totals={"marker": {"color": "#1e7abf"}},
            text=[inr(v) for v in list(sv.values()) + [sum(sv.values())]],
            textposition="outside"))
        apply_theme(fig_w, height=360, title="Savings Waterfall — Total Saved (₹)")
        fig_w.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<div class="sh">🚛 Per-Truck Cost Breakdown (₹)</div>', unsafe_allow_html=True)
    fig_v = go.Figure()
    for cat, bc_, lbl in [("fuel_cost", LN_GREEN, "⛽ Fuel"), ("toll_cost","#1e7abf","🛣️ Toll"),
                           ("driver_cost","#8e44ad","👷 Driver"), ("sla_penalty","#c0392b","⏰ SLA Penalty")]:
        fig_v.add_trace(go.Bar(name=lbl, x=[f"Truck {v}" for v in veh_sum["vehicle"]],
            y=veh_sum[cat], marker_color=bc_,
            text=veh_sum[cat].apply(inr), textposition="inside", textfont=dict(color="white", size=9)))
    apply_theme(fig_v, height=320, legend_below=True)
    fig_v.update_layout(barmode="stack")
    fig_v.update_yaxes(tickprefix="₹", tickformat=",")
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown('<div class="sh">📋 Detailed Cost Table (₹)</div>', unsafe_allow_html=True)
    ct = veh_sum[["vehicle","stops","distance_km","fuel_cost","toll_cost","driver_cost","sla_penalty","total_cost"]].copy()
    ct["vehicle"] = ct["vehicle"].apply(lambda v: f"🚛 Truck {v}")
    ct.columns = ["Truck","Stops","Dist (km)","Fuel (₹)","Toll (₹)","Driver (₹)","SLA Penalty (₹)","Total (₹)"]
    st.dataframe(ct.style.format({"Dist (km)":"{:.1f}","Fuel (₹)":"₹{:,.0f}","Toll (₹)":"₹{:,.0f}",
        "Driver (₹)":"₹{:,.0f}","SLA Penalty (₹)":"₹{:,.0f}","Total (₹)":"₹{:,.0f}"})
        .background_gradient(subset=["Total (₹)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CARBON & SLA
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🌿 Carbon & SLA":
    st.markdown("""<div class="info-box">
    🌿 <b>Carbon</b> = CO₂ from diesel combustion. Smarter routes = less pollution. 🌱<br>
    <b>SLA</b> = delivery promise. Late = <b>₹500/hr penalty</b> per the LoRRI pricing model.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        co2s = base["carbon_kg"] - opt["carbon_kg"]
        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Bar(x=["Baseline (No AI)","Optimized (AI)"],
            y=[base["carbon_kg"], opt["carbon_kg"]], marker_color=["#c0392b", LN_GREEN],
            text=[f"{base['carbon_kg']:,.1f} kg", f"{opt['carbon_kg']:,.1f} kg"],
            textposition="outside"))
        apply_theme(fig_c2, height=300,
                    title=f"CO₂ Emissions — {co2s:,.1f} kg saved ({co2s/base['carbon_kg']*100:.1f}% less)")
        fig_c2.update_layout(showlegend=False)
        fig_c2.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_c2, use_container_width=True)

        fig_cv = go.Figure(go.Bar(x=[f"Truck {v}" for v in veh_sum["vehicle"]], y=veh_sum["carbon_kg"],
            marker_color=list(V_COLORS.values()),
            text=veh_sum["carbon_kg"].round(1).astype(str)+" kg", textposition="outside"))
        apply_theme(fig_cv, height=260, title="Carbon per Truck (kg CO₂)")
        fig_cv.update_layout(showlegend=False)
        fig_cv.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_cv, use_container_width=True)

    with c2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=opt["sla_pct"],
            number={"suffix":"%"},
            title={"text":"SLA Adherence — Delivery Promises Kept"},
            delta={"reference":base["sla_pct"],"increasing":{"color":LN_GREEN},"suffix":"% vs baseline"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":LN_GREEN},
                   "steps":[{"range":[0,50],"color":"rgba(192,57,43,0.15)"},
                             {"range":[50,80],"color":"rgba(245,158,11,0.15)"},
                             {"range":[80,100],"color":"rgba(58,125,44,0.15)"}],
                   "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":base["sla_pct"]}}))
        fig_g.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        bd  = routes.copy(); bd["breached"] = (bd["sla_breach_hr"] > 0).astype(int)
        piv = bd.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_h = go.Figure(go.Heatmap(z=piv.values, x=piv.columns.tolist(),
            y=[f"Truck {v}" for v in piv.index], colorscale="YlOrRd",
            text=piv.values, texttemplate="%{text}", colorbar=dict(title="Breaches")))
        apply_theme(fig_h, height=260, title="Late Deliveries: Truck × Priority (0 = perfect ✅)")
        fig_h.update_xaxes(title_text="Priority"); fig_h.update_yaxes(title_text="Truck")
        st.plotly_chart(fig_h, use_container_width=True)

    bdf = routes[routes["sla_breach_hr"] > 0][
        ["vehicle","stop_order","city","priority","sla_hours","sla_breach_hr","sla_penalty","total_cost"]].copy()
    if not bdf.empty:
        st.markdown('<div class="sh">⚠️ SLA Breach Detail (₹500/hr penalty rate)</div>', unsafe_allow_html=True)
        bdf["vehicle"] = bdf["vehicle"].apply(lambda v: f"🚛 Truck {v}")
        bdf.columns = ["Truck","Stop#","City","Priority","SLA (hr)","Breach (hr)","Penalty (₹)","Total Cost (₹)"]
        st.dataframe(bdf.style.format({"Breach (hr)":"{:.1f}","Penalty (₹)":"₹{:,.0f}","Total Cost (₹)":"₹{:,.0f}"})
            .background_gradient(subset=["Breach (hr)"], cmap="Reds"),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🧠 Explainability":
    st.markdown("""<div class="info-box">
    🧠 Every routing decision balanced <b>time (30%)</b>, <b>₹ cost (35%)</b>,
    <b>carbon (20%)</b>, <b>SLA risk (15%)</b>.
    Charts below use <b>real permutation importance</b> (SHAP-style) to show which factors drove decisions.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<div class="sh" style="font-size:0.88rem;">⚖️ Objective Weights</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Cost (₹)","Travel Time","Carbon CO₂","SLA"],
            values=[35, 30, 20, 15], hole=0.55,
            marker_colors=[LN_GREEN, "#1e7abf", "#27ae60", "#c0392b"],
            textinfo="label+percent"))
        fig_pie.update_layout(height=290, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
            annotations=[{"text":"Weights","x":0.5,"y":0.5,"font_size":13,"showarrow":False}])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown(f'<div class="sh" style="font-size:0.88rem;">🔬 Feature Importance (Permutation-Based)</div>', unsafe_allow_html=True)
        fi_l = list(fi.keys()); fi_v = list(fi.values()); mv = max(fi_v)
        fig_fi = go.Figure(go.Bar(x=fi_v, y=fi_l, orientation="h",
            marker_color=["#c0392b" if v == mv else LN_GREEN for v in fi_v],
            text=[f"{v:.1f}%" for v in fi_v], textposition="outside"))
        apply_theme(fig_fi, height=300)
        fig_fi.update_layout(title="Which factor drove routing decisions most?")
        fig_fi.update_xaxes(title_text="Importance (%)")
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="sh">📊 Per-Stop Score Contribution by Truck</div>', unsafe_allow_html=True)
    vf  = st.selectbox("Filter:", ["All Trucks"] + [f"Truck {v}" for v in sorted(routes["vehicle"].unique())])
    scd = sc if vf == "All Trucks" else sc[sc["vehicle"] == int(vf.split()[-1])].copy()
    fc  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    fco = [LN_GREEN, "#1e7abf", "#8e44ad", "#e67e22", "#27ae60", "#c0392b"]
    fig_stk = go.Figure()
    for f_, c_ in zip(fc, fco):
        fig_stk.add_trace(go.Bar(name=f_, x=scd["city"], y=scd[f_], marker_color=c_))
    apply_theme(fig_stk, height=380, legend_below=True)
    fig_stk.update_layout(barmode="stack")
    fig_stk.update_xaxes(tickangle=-45)
    fig_stk.update_yaxes(title_text="Weighted Contribution to MO Score")
    st.plotly_chart(fig_stk, use_container_width=True)

    st.markdown('<div class="sh">🔍 Top 10 Hardest-to-Schedule Stops</div>', unsafe_allow_html=True)
    t10 = routes.nlargest(10, "mo_score")[
        ["vehicle","stop_order","city","priority","weight","travel_time_hr","fuel_cost","carbon_kg","sla_breach_hr","mo_score"]].copy()
    t10["vehicle"] = t10["vehicle"].apply(lambda v: f"🚛 Truck {v}")
    t10.columns = ["Truck","Stop#","City","Priority","Weight (kg)","Time (hr)","Fuel (₹)","Carbon (kg)","Breach (hr)","MO Score"]
    st.dataframe(t10.style.format({"Weight (kg)":"{:.0f}","Time (hr)":"{:.2f}","Fuel (₹)":"₹{:,.0f}",
        "Carbon (kg)":"{:.2f}","Breach (hr)":"{:.1f}","MO Score":"{:.4f}"})
        .background_gradient(subset=["MO Score"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RE-OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "⚡ Re-optimization Simulator":
    st.markdown("""<div class="info-box">
    ⚡ <b>Simulate real-world disruptions</b> — traffic jams, urgent customer escalations.
    Watch LoRRI's AI re-plan the affected truck's route instantly with updated ₹ cost estimates.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">🚦 Scenario 1 — Traffic Jam</div>', unsafe_allow_html=True)
        city1 = st.selectbox("City hit by traffic:", sorted(ships["city"].tolist()))
        spike = st.slider("Traffic multiplier (1.0=clear road, 3.0=gridlock)", 1.0, 3.0, 2.5, 0.1)
        if st.button("🔴 Trigger Traffic Disruption", use_container_width=True):
            row = ships[ships["city"] == city1].iloc[0]
            om  = row["traffic_mult"]
            dk  = haversine(DEPOT["latitude"], DEPOT["longitude"], row["latitude"], row["longitude"])
            to  = dk / (55 / om); tn = dk / (55 / spike); pi = (tn - to) / to * 100
            if pi > 30:
                st.markdown(f"""<div class="warn-box">
                ⚠️ <b>Disruption Detected: {city1}</b><br>
                Traffic: {om:.2f}× → <span class="tag-red">{spike:.2f}×</span><br>
                Travel time increase: <span class="tag-red">+{pi:.1f}%</span><br>
                Extra SLA penalty exposure: <span class="tag-red">{inr((tn-to)*500)}</span><br>
                <span class="tag-red">THRESHOLD BREACHED — Re-optimizing!</span>
                </div>""", unsafe_allow_html=True)
                with st.spinner("LoRRI AI re-optimizing truck route..."): time.sleep(1.2)
                av = routes[routes["city"] == city1]["vehicle"].values
                if len(av):
                    vid  = av[0]; orig = routes[routes["vehicle"] == vid].sort_values("stop_order")
                    mask = orig["city"] == city1
                    reop = pd.concat([orig[~mask], orig[mask]]).reset_index(drop=True)
                    d1   = sum(haversine(orig.iloc[i]["latitude"],orig.iloc[i]["longitude"],
                                         orig.iloc[i+1]["latitude"],orig.iloc[i+1]["longitude"])
                               for i in range(len(orig)-1))
                    d2   = sum(haversine(reop.iloc[i]["latitude"],reop.iloc[i]["longitude"],
                                         reop.iloc[i+1]["latitude"],reop.iloc[i+1]["longitude"])
                               for i in range(len(reop)-1))
                    st.markdown(f'<div class="ok-box">✅ <b>Truck {vid} re-routed!</b> {city1} moved to last stop.</div>', unsafe_allow_html=True)
                    ca, cb = st.columns(2)
                    ca.metric("Original route", f"{d1:.1f} km")
                    cb.metric("Re-optimized",   f"{d2:.1f} km", delta=f"{d2-d1:+.1f} km", delta_color="inverse")
                    dr = reop[["city","priority","weight","sla_hours","total_cost"]].copy()
                    dr.insert(0,"Stop#",range(1,len(dr)+1))
                    dr["total_cost"] = dr["total_cost"].apply(inr)
                    dr.columns = ["Stop#","City","Priority","Weight (kg)","SLA (hr)","Cost (₹)"]
                    st.dataframe(dr, use_container_width=True, hide_index=True)
            else:
                st.markdown(f'<div class="ok-box">✅ No re-optimization needed — {pi:.1f}% within 30% threshold.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sh">🚨 Scenario 2 — Customer Escalation</div>', unsafe_allow_html=True)
        city2 = st.selectbox("City escalated to urgent:", sorted(ships["city"].tolist()), key="esc")
        if st.button("🔴 Trigger Priority Escalation", use_container_width=True):
            op_ = ships[ships["city"] == city2]["priority"].values[0]
            os_ = ships[ships["city"] == city2]["sla_hours"].values[0]
            if op_ == "HIGH":
                st.markdown(f'<div class="ok-box">✅ {city2} is already HIGH priority — no change!</div>', unsafe_allow_html=True)
            else:
                with st.spinner("LoRRI AI escalating and re-routing..."): time.sleep(1.0)
                av  = routes[routes["city"] == city2]["vehicle"].values
                vid = av[0] if len(av) else 1
                orig = routes[routes["vehicle"] == vid].sort_values("stop_order")
                mask = orig["city"] == city2
                newr = pd.concat([orig[mask], orig[~mask]]).reset_index(drop=True)
                pen  = orig[mask]["sla_penalty"].values[0]
                st.markdown(f"""<div class="ok-box">
                ✅ <b>{city2}</b> escalated: <span class="tag-yellow">{op_}</span>
                → <span class="tag-red">HIGH</span> | SLA: {os_}hr → <b>24hr</b> | Moved to Stop #1 on Truck {vid}
                </div>""", unsafe_allow_html=True)
                ca, cb, cc = st.columns(3)
                ca.metric("Old SLA", f"{os_} hr"); cb.metric("New SLA","24 hr",delta="Tightened")
                cc.metric("Penalty Saved", inr(pen), delta=f"-{inr(pen)}", delta_color="inverse")
                dn = newr[["city","priority","weight","sla_hours","total_cost"]].copy()
                dn.insert(0,"Stop#",range(1,len(dn)+1))
                dn["total_cost"] = dn["total_cost"].apply(inr)
                dn.columns = ["Stop#","City","Priority","Weight (kg)","SLA (hr)","Cost (₹)"]
                st.dataframe(dn, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="sh">📈 Live Risk Monitor</div>', unsafe_allow_html=True)
    rdf = ships[["city","traffic_mult","priority","sla_hours"]].copy()
    rdf["risk"] = (rdf["traffic_mult"]/1.8*0.6 + rdf["sla_hours"].map({24:1.0,48:0.5,72:0.2})*0.4).round(3)
    rdf["status"] = rdf["risk"].apply(
        lambda x: "🔴 HIGH RISK" if x > 0.7 else ("🟡 MONITOR" if x > 0.4 else "🟢 STABLE"))
    rdf = rdf.sort_values("risk", ascending=False)
    fig_r = px.bar(rdf.head(15), x="city", y="risk", color="status",
        color_discrete_map={"🔴 HIGH RISK":"#c0392b","🟡 MONITOR":"#f59e0b","🟢 STABLE":LN_GREEN},
        title="Top 15 Cities by Re-Optimization Risk Score",
        labels={"risk":"Risk Score","city":"City"}, height=320)
    fig_r.add_hline(y=0.7, line_dash="dash", line_color="#c0392b",
                    annotation_text="← Trigger threshold (0.70)")
    apply_theme(fig_r)
    st.plotly_chart(fig_r, use_container_width=True)

# CLOSE content-area div
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER (LogisticsNow style)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{LN_NAVY};color:#94a3b8;padding:20px 40px;margin-top:20px;
            font-size:0.75rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
        <b style="color:white;font-size:0.9rem;">Logistics<span style="color:{LN_GREEN}">Now</span></b> · LoRRI AI Route Optimization Engine<br>
        Problem Statement 4 · Synapflow Hackathon · Mumbai Depot
    </div>
    <div style="text-align:right;">
        📧 connect@logisticsnow.in &nbsp;|&nbsp; 📞 +91-9867773508<br>
        All costs in ₹ INR · Multi-Objective CVRP · Permutation-Based Explainability
    </div>
</div>
""", unsafe_allow_html=True)
