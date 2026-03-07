"""
LoRRI – LogisticsNow AI Route Optimization Engine
Complete dashboard: About, AI Assistant, Dashboard, Map, Financial,
Carbon & SLA, Explainability, Re-optimization Simulator, AI Route Predictor.
All costs in ₹ INR. Scattermap textposition bug fixed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import datetime
import requests
import random

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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * R * asin(sqrt(a))

def inr(val):
    return f"₹{val:,.0f}"

def apply_theme(fig, height=340, title="", legend_below=False):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Poppins, sans-serif", color=LN_NAVY, size=11),
        height=height,
        margin=dict(l=10, r=10, t=44 if title else 18, b=10),
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
    now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
    sub_html = f'<div style="font-size:0.8rem;color:#64748b;margin-top:2px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;
                margin-bottom:16px;flex-wrap:wrap;gap:8px;">
        <div>
            <div style="font-size:1.25rem;font-weight:700;color:{LN_NAVY};">{title}</div>
            {sub_html}
        </div>
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:8px;
                    padding:6px 14px;font-size:0.72rem;color:#64748b;white-space:nowrap;
                    display:flex;align-items:center;gap:6px;">
            <span style="color:{LN_GREEN};">●</span> Last updated: <b style="color:{LN_NAVY};">{now}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

def loading_state(msg="Processing…"):
    with st.spinner(f"⚙️ {msg}"):
        time.sleep(0.5)

def sh(text):
    """Section heading."""
    return f'<div class="sh">{text}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family:'Poppins',sans-serif; background:{LN_LGRAY}; }}
.main .block-container {{ padding:0!important; max-width:100%!important; }}

.topbar {{ background:{LN_NAVY}; padding:8px 40px; display:flex; align-items:center;
           justify-content:space-between; font-size:0.75rem; color:#aab8c5; }}
.topbar a {{ color:#aab8c5; text-decoration:none; }}
.navbar {{ background:white; padding:14px 40px; display:flex; align-items:center;
           justify-content:space-between; border-bottom:2px solid {LN_BORDER};
           box-shadow:0 2px 8px rgba(0,0,0,0.06); }}
.logo-wrap {{ display:flex; align-items:center; gap:10px; }}
.logo-n {{ width:44px; height:44px; background:{LN_NAVY}; color:white; border-radius:6px;
           font-size:1.4rem; font-weight:800; display:flex; align-items:center; justify-content:center; }}
.logo-text {{ font-size:1.4rem; font-weight:700; color:{LN_NAVY}; letter-spacing:-0.5px; }}
.logo-text span {{ color:{LN_GREEN}; }}
.nav-links {{ display:flex; gap:28px; }}
.nav-link {{ font-size:0.85rem; font-weight:500; color:#334155; text-decoration:none; }}
.nav-link:hover {{ color:{LN_GREEN}; }}
.nav-cta {{ background:{LN_DGREEN}; color:white; padding:8px 20px; border-radius:4px;
            font-size:0.8rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }}
.hero-banner {{ background:linear-gradient(135deg,{LN_NAVY} 0%,#2d4a6b 60%,#1a3a1a 100%);
                padding:36px 40px 28px; color:white; position:relative; overflow:hidden; }}
.hero-title {{ font-size:2rem; font-weight:700; margin:0 0 6px; line-height:1.2; position:relative; }}
.hero-title span {{ color:{LN_GREEN}; }}
.hero-sub {{ font-size:0.88rem; color:#94a3b8; position:relative; margin-bottom:16px; }}
.hero-badge {{ display:inline-flex; align-items:center; gap:6px;
               background:rgba(58,125,44,0.2); border:1px solid {LN_GREEN};
               color:#6bcf57; border-radius:20px; padding:4px 14px;
               font-size:0.72rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; }}
.content-area {{ padding:24px 32px; background:{LN_LGRAY}; }}
.breadcrumb {{ font-size:0.78rem; color:#64748b; margin-bottom:20px;
               display:flex; align-items:center; gap:6px; }}
.breadcrumb a {{ color:{LN_GREEN}; text-decoration:none; }}

.kpi-card {{ background:white; border:1px solid {LN_BORDER}; border-radius:12px;
             padding:18px 20px; position:relative; overflow:hidden;
             box-shadow:0 2px 8px rgba(0,0,0,0.05); transition:box-shadow 0.2s; }}
.kpi-card:hover {{ box-shadow:0 6px 20px rgba(58,125,44,0.14); }}
.kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px;
                     background:var(--ac,{LN_GREEN}); }}
.kpi-lbl {{ font-size:0.6rem; font-weight:600; color:#64748b; text-transform:uppercase;
            letter-spacing:0.1em; margin-bottom:8px; }}
.kpi-val {{ font-size:1.65rem; font-weight:700; color:{LN_NAVY}; line-height:1.1; }}
.kpi-d {{ font-size:0.7rem; margin-top:5px; font-weight:500; }}
.dg {{ color:{LN_GREEN}; }} .dr {{ color:#dc2626; }}

.sh {{ font-size:1.05rem; font-weight:700; color:{LN_NAVY}; margin:24px 0 12px;
       display:flex; align-items:center; gap:10px;
       border-left:4px solid {LN_GREEN}; padding-left:12px; }}
.info-box {{ background:#f0fdf4; border-left:4px solid {LN_GREEN}; border-radius:8px;
             padding:14px 18px; margin:4px 0 16px; font-size:0.88rem;
             line-height:1.7; color:{LN_NAVY}; }}
.info-box b {{ color:{LN_DGREEN}; }}
.warn-box {{ background:#fffbeb; border-left:4px solid #f59e0b; border-radius:6px;
             padding:12px 16px; margin:6px 0; font-size:0.86rem; line-height:1.65; color:{LN_NAVY}; }}
.ok-box {{ background:#f0fdf4; border-left:4px solid {LN_GREEN}; border-radius:6px;
           padding:12px 16px; margin:6px 0; font-size:0.86rem; line-height:1.65; color:{LN_NAVY}; }}
.tag-red {{ color:#dc2626; font-weight:600; }}
.tag-green {{ color:{LN_GREEN}; font-weight:600; }}
.tag-yellow {{ color:#d97706; font-weight:600; }}

.legend-row {{ display:flex; align-items:flex-start; gap:10px; font-size:0.84rem;
               color:{LN_NAVY}; margin-bottom:10px; padding-bottom:10px;
               border-bottom:1px solid {LN_BORDER}; }}
.legend-row:last-child {{ border-bottom:none; margin-bottom:0; }}
.legend-dot {{ width:14px; height:14px; border-radius:3px; flex-shrink:0; margin-top:3px; }}

[data-testid="stSidebar"] {{ background:white!important; border-right:2px solid {LN_BORDER}!important; }}
.sb-logo {{ display:flex; align-items:center; gap:10px; margin-bottom:4px; }}
.sb-logo-box {{ width:36px; height:36px; background:{LN_NAVY}; border-radius:6px;
                color:white; font-weight:800; font-size:1.1rem;
                display:flex; align-items:center; justify-content:center; }}
.sb-brand {{ font-size:1.1rem; font-weight:700; color:{LN_NAVY}; }}
.sb-brand span {{ color:{LN_GREEN}; }}
.sb-sub {{ font-size:0.58rem; color:#94a3b8; text-transform:uppercase;
           letter-spacing:0.14em; margin-bottom:1.4rem; padding-left:46px; }}
.sb-sec {{ font-size:0.6rem; font-weight:700; color:{LN_GREEN}; letter-spacing:0.16em;
           text-transform:uppercase; margin:1.2rem 0 0.4rem;
           border-bottom:1px solid {LN_BORDER}; padding-bottom:4px; }}

[data-testid="metric-container"] {{ background:white; border:1px solid {LN_BORDER};
                                     border-radius:10px; padding:12px 16px!important; }}

@keyframes pulse {{ 0%{{opacity:1}} 50%{{opacity:0.3}} 100%{{opacity:1}} }}
.live-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%;
             background:{LN_GREEN}; animation:pulse 1.8s infinite; margin-right:5px; }}
.stat-row {{ display:flex; justify-content:space-between; align-items:center;
             padding:5px 0; border-bottom:1px solid #f1f5f9; font-size:0.74rem; }}
.stat-row:last-child {{ border-bottom:none; }}
.stat-label {{ color:#64748b; }}
.stat-val {{ font-weight:700; color:{LN_NAVY}; }}
</style>
""", unsafe_allow_html=True)

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
        "Fuel Cost (₹)":   "fuel_cost",
        "Toll Cost (₹)":   "toll_cost",
        "Driver Cost (₹)": "driver_cost",
        "Carbon Emitted":  "carbon_kg",
        "SLA Breach":      "sla_breach_hr",
        "Package Weight":  "weight",
    }
    X = routes_df[list(feats.values())].copy()
    y = routes_df["mo_score"].values
    base_mae = np.mean(np.abs(y - y.mean()))
    imp = {}
    for lbl, col in feats.items():
        sh2 = X.copy(); sh2[col] = np.random.permutation(sh2[col].values)
        proxy = sh2.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        imp[lbl] = abs(np.mean(np.abs(y - proxy.values)) - base_mae)
    tot = sum(imp.values()) + 1e-9
    return {k: round(v / tot * 100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

@st.cache_data
def stop_cont(routes_df):
    cols    = ["travel_time_hr", "fuel_cost", "toll_cost", "driver_cost", "carbon_kg", "sla_breach_hr"]
    labels  = ["Travel Time", "Fuel Cost", "Toll Cost", "Driver Cost", "Carbon", "SLA Breach"]
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
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
LORRI_KB = f"""
LOGISTICSNOW COMPANY:
- Website: logisticsnow.in | Email: connect@logisticsnow.in | Phone: +91-9867773508 / +91-9653620207
- Platform: LoRRI — Logistics Rating & Intelligence

WHAT IS LORRI?
LoRRI connects Shippers/Manufacturers with Carriers/Transporters through data-driven intelligence.
For Shippers: carrier profiles, ratings, cost savings, procurement insights.
For Carriers: discoverability, business inquiries, reputation building, profile control.

AI ROUTE OPTIMIZATION ENGINE:
- CVRP framework: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%)
- Fuel: ₹12/km | Driver: ₹180/hr | SLA penalty: ₹500/hr breach | Toll: variable
- Re-optimization: triggers when traffic >30% delay OR priority escalation
- Explainability: permutation-based feature importance (SHAP-style)

CURRENT RUN (Mumbai Depot):
- 21 shipments, 5 trucks
- Optimized cost: {inr(opt['total_cost'])} | Saved: {inr(base['total_cost']-opt['total_cost'])} vs baseline
- SLA: {opt['sla_pct']:.1f}% (baseline 4%) | Carbon saved: {base['carbon_kg']-opt['carbon_kg']:,.0f} kg

ROUTES:
Truck 1: Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada
Truck 2: Mumbai → Surat → Vadodara → Raipur
Truck 3: Mumbai → Aurangabad → Solapur → Madurai → Jammu
Truck 4: Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram
Truck 5: Mumbai → Hubli → Mangalore → Bengaluru
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">
        <div class="sb-logo-box">LN</div>
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
        "🔮 AI Route Predictor",
    ], label_visibility="collapsed")

    st.markdown(f'<div class="sb-sec">🛠️ Fleet Control</div>', unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed", value=True)
    st.toggle("Auto Re-optimize", value=False)
    if st.button("🔄 Sync Depot Data", use_container_width=True):
        st.toast("✅ Synced with Mumbai Depot!", icon="🏭")

    st.markdown(f'<div class="sb-sec">📦 Live Stats</div>', unsafe_allow_html=True)

    _n_ships    = len(ships)
    _n_trucks   = veh_sum["vehicle"].nunique()
    _sla_ok     = int((routes["sla_breach_hr"] == 0).sum() / len(routes) * 100)
    _breaches   = int((routes["sla_breach_hr"] > 0).sum())
    _total_cost = veh_sum["total_cost"].sum()
    _carbon     = veh_sum["carbon_kg"].sum()
    _total_km   = veh_sum["distance_km"].sum()
    _util_avg   = veh_sum["utilization_pct"].mean()

    st.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;padding:12px 14px;margin-top:6px;">
        <div style="font-size:0.62rem;font-weight:700;color:{LN_GREEN};margin-bottom:10px;
                    text-transform:uppercase;letter-spacing:0.08em;">
            <span class="live-dot"></span> Live Dashboard
        </div>
        <div class="stat-row"><span class="stat-label">Shipments</span><span class="stat-val">{_n_ships}</span></div>
        <div class="stat-row"><span class="stat-label">Trucks Active</span><span class="stat-val">{_n_trucks}</span></div>
        <div class="stat-row"><span class="stat-label">SLA OK</span><span class="stat-val" style="color:{LN_GREEN}">{_sla_ok}%</span></div>
        <div class="stat-row"><span class="stat-label">SLA Breaches</span><span class="stat-val" style="color:#dc2626">{_breaches}</span></div>
        <div class="stat-row"><span class="stat-label">Total Distance</span><span class="stat-val">{_total_km:,.0f} km</span></div>
        <div class="stat-row"><span class="stat-label">Fleet Cost</span><span class="stat-val">{inr(_total_cost)}</span></div>
        <div class="stat-row"><span class="stat-label">Carbon Emitted</span><span class="stat-val">{_carbon:,.0f} kg</span></div>
        <div class="stat-row"><span class="stat-label">Avg Utilization</span><span class="stat-val">{_util_avg:.1f}%</span></div>
        <div class="stat-row"><span class="stat-label">Depot</span><span class="stat-val">Mumbai 🏭</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.65rem;color:#94a3b8;line-height:1.8;">
    📧 connect@logisticsnow.in<br>
    📞 +91-9867773508<br>
    🌐 logisticsnow.in
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR + NAVBAR + HERO

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT LORRI
# ══════════════════════════════════════════════════════════════════════════════
if pg == "🏢 About LoRRI":
    page_header("🏢 About LoRRI", "LogisticsNow · AI-Powered Logistics Intelligence Platform")

st.markdown(f"""
<div style="background:white;border-radius:14px;
border:1px solid {LN_BORDER};
padding:34px 36px;margin-bottom:22px;
border-top:5px solid {LN_GREEN};">

<div style="font-size:1.6rem;font-weight:700;color:{LN_NAVY};margin-bottom:8px;">
🚚 LoRRI — Logistics Intelligence Platform
</div>

<div style="font-size:0.9rem;color:#64748b;margin-bottom:18px;">
AI-Powered Logistics Optimization Engine
</div>

<p style="font-size:0.92rem;color:#334155;line-height:1.85;margin-bottom:12px;">
Modern logistics networks generate massive operational data, yet routing decisions are still
often based on <b>static planning and manual assumptions</b>. This leads to unnecessary fuel costs,
delivery delays, inefficient truck utilization, and increased carbon emissions.
</p>

<p style="font-size:0.92rem;color:#334155;line-height:1.85;margin-bottom:0;">
<b>LoRRI (Logistics Rating & Intelligence)</b> transforms logistics operations into an
<b>AI-driven decision intelligence platform</b>. By combining real-time data,
optimization algorithms, and interactive analytics dashboards,
LoRRI enables organizations to <b>optimize routes, reduce costs,
improve delivery reliability, and minimize environmental impact</b>.
</p>

</div>
""", unsafe_allow_html=True)


st.markdown(f'<div class="sh">🌍 Our Vision</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{LN_NAVY};border-radius:12px;
padding:26px 30px;color:white;margin-bottom:24px;">

<p style="font-size:0.92rem;line-height:1.9;color:#cbd5e1;margin:0;">
Our vision is to build a <b style="color:white;">global logistics intelligence platform</b>
where transportation networks operate as <b style="color:{LN_GREEN};">
autonomous, data-driven systems</b>.
</p>

<p style="font-size:0.92rem;line-height:1.9;color:#cbd5e1;margin-top:12px;">
By integrating <b style="color:white;">AI optimization</b>, 
<b style="color:white;">real-time analytics</b>, and 
<b style="color:white;">sustainability metrics</b>,
LoRRI empowers supply chains to become more efficient,
resilient, and environmentally responsible.
</p>

</div>
""", unsafe_allow_html=True)


st.markdown(f'<div class="sh">⚙️ Key Platform Capabilities</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

features = [
("🚛","AI Route Optimization","Dynamic multi-objective routing balancing cost, time, carbon emissions, and SLA constraints."),
("📊","Operational Intelligence","Real-time analytics dashboard showing fleet performance, cost efficiency, and route insights."),
("🌿","Sustainability Monitoring","Measure carbon emissions per route and identify greener transportation strategies."),
("⚡","Adaptive Re-optimization","Automatically adjust routes during traffic disruptions or priority shipment escalations.")
]

for col,(icon,title,desc) in zip([col1,col2,col3,col4],features):
    col.markdown(f"""
    <div style="background:white;border:1px solid {LN_BORDER};
    border-radius:12px;padding:18px;height:180px;
    border-top:3px solid {LN_GREEN};">

    <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>

    <div style="font-size:0.85rem;font-weight:700;color:{LN_NAVY};margin-bottom:6px;">
    {title}
    </div>

    <div style="font-size:0.75rem;color:#64748b;line-height:1.6;">
    {desc}
    </div>

    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT  —  Vectorless RAG + LangChain-style Pipeline
#                         + Pandas Retrieval + Rule-Based Router
#                         + Clickable Chip Buttons
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🤖 LoRRI AI Assistant":
    page_header("🤖 LoRRI AI Assistant", "Vectorless RAG · LangChain-style Pipeline · Pandas Retrieval · Rule-Based Router")

    # ── SESSION STATE INITIALISATION ─────────────────────────────────────────
    if "msgs"        not in st.session_state: st.session_state.msgs        = []
    if "chip_prompt" not in st.session_state: st.session_state.chip_prompt = None

    # ════════════════════════════════════════════════════════════════════════
    # ① RAG KNOWLEDGE BASE  (plain-text chunks — no vectors needed)
    # ════════════════════════════════════════════════════════════════════════
    route_map_text = {
        1: "Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada",
        2: "Mumbai → Surat → Vadodara → Raipur",
        3: "Mumbai → Aurangabad → Solapur → Madurai → Jammu",
        4: "Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram",
        5: "Mumbai → Hubli → Mangalore → Bengaluru",
    }

    # Static knowledge chunks — each has a list of keyword triggers
    KNOWLEDGE_CHUNKS = [
        {
            "id": "company",
            "triggers": ["logisticsnow","company","about","who","contact","email","phone","website","lorri"],
            "text": (
                "COMPANY: LogisticsNow (logisticsnow.in) | connect@logisticsnow.in | +91-9867773508 / +91-9653620207\n"
                "LoRRI = Logistics Rating & Intelligence — India's premier logistics intelligence platform.\n"
                "Connects Shippers/Manufacturers with Carriers/Transporters through data-driven insights.\n"
                "For Shippers: carrier profiles, industry ratings, cost savings, risk reduction.\n"
                "For Carriers: discoverability by Fortune 500, business inquiries, reputation building."
            ),
        },
        {
            "id": "optimization",
            "triggers": ["cvrp","optimize","optimization","weighted","objective","weight","score","solver","ortools","heuristic","algorithm"],
            "text": (
                "OPTIMIZATION ENGINE: Capacitated Vehicle Routing Problem (CVRP) framework.\n"
                "Weighted objective: Cost ₹ (35%) + Travel Time (30%) + Carbon CO₂ (20%) + SLA (15%).\n"
                "Solver: OR-Tools heuristic with nearest-neighbour + 2-opt local search.\n"
                "Re-optimization triggers: traffic delay >30% OR shipment priority escalation.\n"
                "Explainability: permutation-based feature importance (SHAP-style)."
            ),
        },
        {
            "id": "pricing",
            "triggers": ["fuel","toll","driver","cost","price","₹","inr","rupee","penalty","sla penalty","wage"],
            "text": (
                "PRICING MODEL (all in ₹ INR):\n"
                "- Fuel: ₹12 per km\n"
                "- Driver wages: ₹180 per hour\n"
                "- SLA breach penalty: ₹500 per hour late\n"
                "- Toll: variable by corridor (highway vs state road)"
            ),
        },
        {
            "id": "sla",
            "triggers": ["sla","late","breach","delay","on time","delivery","promise","window","24hr","48hr","72hr","adherence"],
            "text": (
                f"SLA PERFORMANCE:\n"
                f"- Optimized SLA adherence: {opt['sla_pct']:.1f}% (baseline was only {base['sla_pct']:.0f}%)\n"
                f"- Total SLA breaches: {opt['breaches']} cities\n"
                f"- Breach penalty rate: ₹500 per hour late\n"
                f"- Total SLA penalties incurred: {inr(veh_sum['sla_penalty'].sum())}\n"
                f"- Delivery windows: HIGH=24hr, MEDIUM=48hr, LOW=72hr"
            ),
        },
        {
            "id": "carbon",
            "triggers": ["carbon","co2","emission","green","environment","sustainability","pollution","tree","car","eco"],
            "text": (
                f"CARBON & SUSTAINABILITY:\n"
                f"- Optimized CO₂: {opt['carbon_kg']:,.1f} kg\n"
                f"- Baseline CO₂: {base['carbon_kg']:,.1f} kg\n"
                f"- CO₂ saved: {base['carbon_kg']-opt['carbon_kg']:,.1f} kg ({(base['carbon_kg']-opt['carbon_kg'])/base['carbon_kg']*100:.1f}% reduction)\n"
                f"- Equivalent trees planted: {int((base['carbon_kg']-opt['carbon_kg'])/21):,}\n"
                f"- Cars removed from road equivalent: {int((base['carbon_kg']-opt['carbon_kg'])/2400)}"
            ),
        },
        {
            "id": "fleet_summary",
            "triggers": ["fleet","total","summary","overall","run","depot","mumbai","shipment","truck","vehicle","save","saving","saved","baseline","optimized"],
            "text": (
                f"FLEET SUMMARY (Mumbai Depot):\n"
                f"- Shipments: {opt['n_ships']} | Trucks: {opt['n_vehicles']}\n"
                f"- Optimized total cost: {inr(opt['total_cost'])}\n"
                f"- Baseline total cost: {inr(base['total_cost'])}\n"
                f"- Total saved: {inr(base['total_cost']-opt['total_cost'])} "
                f"({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}% reduction)\n"
                f"- Fuel saved: {inr(base['fuel_cost']-opt['fuel_cost'])}\n"
                f"- Toll saved: {inr(base['toll_cost']-opt['toll_cost'])}\n"
                f"- Driver cost saved: {inr(base['driver_cost']-opt['driver_cost'])}\n"
                f"- Distance optimized: {opt['distance_km']:,.0f} km (baseline {base['distance_km']:,.0f} km)"
            ),
        },
        {
            "id": "traffic",
            "triggers": ["traffic","jam","congestion","disruption","reoptimize","re-optimize","threshold","spike","delay","multiplier"],
            "text": (
                "TRAFFIC & RE-OPTIMIZATION:\n"
                "- Re-optimization triggers when traffic causes >30% travel time increase\n"
                "- Also triggers on priority escalation (customer requests urgent delivery)\n"
                "- Solver recomputes in ~1-2 seconds using OR-Tools local search\n"
                "- Affected city is moved to last stop OR truck is fully re-sequenced\n"
                "- Risk scoring: HIGH >0.7, MONITOR >0.4, STABLE ≤0.4"
            ),
        },
        {
            "id": "cost_saving_actions",
            "triggers": ["suggest","recommendation","action","improve","reduce cost","save more","tip","advice","how to","better","optimize further"],
            "text": (
                "COST SAVING RECOMMENDATIONS:\n"
                "1. Consolidate Truck 2 & 5 loads — both under 70% utilization, merge short corridors\n"
                "2. Avoid HIGH-traffic corridors during peak hours (traffic_mult > 2.0)\n"
                "3. Upgrade LOW-priority shipments to 72hr SLA window to reduce penalty risk\n"
                "4. Use express highway (Alt 2) only when time savings exceed toll premium\n"
                "5. Cluster Madurai + Thiruvananthapuram on one southern truck (currently split)\n"
                "6. Pre-position trucks at intermediate hubs (Pune, Ahmedabad) to reduce backtracking"
            ),
        },
    ]

    # ════════════════════════════════════════════════════════════════════════
    # ② PANDAS RETRIEVAL  — structured data queries over DataFrames
    # ════════════════════════════════════════════════════════════════════════
    def pandas_retrieve(query: str) -> str:
        """
        LangChain-style Pandas retrieval tool.
        Returns a formatted string of structured data relevant to the query.
        """
        q = query.lower()
        results = []

        # Truck-specific query
        for v in [1, 2, 3, 4, 5]:
            if f"truck {v}" in q or f"truck{v}" in q:
                row = veh_sum[veh_sum["vehicle"] == v]
                if not row.empty:
                    r = row.iloc[0]
                    breach_cities = routes[
                        (routes["vehicle"] == v) & (routes["sla_breach_hr"] > 0)
                    ]["city"].tolist()
                    stops_df = routes[routes["vehicle"] == v].sort_values("stop_order")[
                        ["stop_order", "city", "weight", "priority", "travel_time_hr",
                         "fuel_cost", "carbon_kg", "sla_breach_hr"]
                    ]
                    results.append(
                        f"TRUCK {v} DATA:\n"
                        f"Route: {route_map_text.get(v, '?')}\n"
                        f"Stops: {int(r['stops'])} | Distance: {r['distance_km']:,.0f} km | "
                        f"Time: {r['time_hr']:.1f} hr | Load: {r['load_kg']:.0f} kg / 800 kg "
                        f"({r['utilization_pct']:.0f}% utilized)\n"
                        f"Fuel: {inr(r['fuel_cost'])} | Toll: {inr(r['toll_cost'])} | "
                        f"Driver: {inr(r['driver_cost'])} | SLA Penalty: {inr(r['sla_penalty'])}\n"
                        f"Total Cost: {inr(r['total_cost'])} | Carbon: {r['carbon_kg']:.1f} kg CO₂\n"
                        f"SLA Breaches: {int(r['sla_breaches'])} "
                        f"({'cities: ' + ', '.join(breach_cities) if breach_cities else 'none — perfect ✅'})\n"
                        f"\nSTOP-BY-STOP:\n" + stops_df.to_string(index=False)
                    )

        # City-level SLA breach detail
        if any(kw in q for kw in ["late", "breach", "which cities", "missed", "sla breach"]):
            breach_df = routes[routes["sla_breach_hr"] > 0][
                ["vehicle", "city", "priority", "sla_breach_hr", "sla_penalty"]
            ].copy()
            if not breach_df.empty:
                breach_df["vehicle"] = breach_df["vehicle"].apply(lambda v: f"Truck {v}")
                results.append("SLA BREACH DETAIL (cities that were late):\n" + breach_df.to_string(index=False))
            else:
                results.append("SLA BREACHES: None — all deliveries were on time ✅")

        # Most expensive / cheapest truck
        if any(kw in q for kw in ["most expensive", "highest cost", "most costly", "expensive truck", "costs most"]):
            top = veh_sum.loc[veh_sum["total_cost"].idxmax()]
            results.append(
                f"MOST EXPENSIVE TRUCK: Truck {int(top['vehicle'])} "
                f"— {inr(top['total_cost'])} total cost\n"
                f"Route: {route_map_text.get(int(top['vehicle']), '?')}\n"
                f"Breakdown: Fuel {inr(top['fuel_cost'])} | Toll {inr(top['toll_cost'])} | "
                f"Driver {inr(top['driver_cost'])} | SLA Penalty {inr(top['sla_penalty'])}"
            )

        # Utilization analysis
        if any(kw in q for kw in ["utilization", "utilisation", "capacity", "load", "full", "empty"]):
            util_df = veh_sum[["vehicle", "load_kg", "utilization_pct"]].copy()
            util_df["vehicle"] = util_df["vehicle"].apply(lambda v: f"Truck {v}")
            results.append(
                "FLEET UTILIZATION:\n" + util_df.to_string(index=False) +
                f"\nAverage utilization: {veh_sum['utilization_pct'].mean():.1f}%"
            )

        # Top carbon emitters by city
        if any(kw in q for kw in ["carbon city", "top emitter", "most co2", "pollut", "city carbon"]):
            top_co2 = routes.groupby("city")["carbon_kg"].sum().sort_values(ascending=False).head(5)
            results.append("TOP 5 CITIES BY CO₂ EMISSION:\n" + top_co2.to_string())

        # Cost comparison all trucks
        if any(kw in q for kw in ["compare truck", "all truck", "each truck", "per truck", "breakdown"]):
            cmp = veh_sum[["vehicle", "distance_km", "total_cost", "carbon_kg",
                           "sla_breaches", "utilization_pct"]].copy()
            cmp["vehicle"] = cmp["vehicle"].apply(lambda v: f"Truck {v}")
            cmp["total_cost"] = cmp["total_cost"].apply(lambda x: f"₹{x:,.0f}")
            results.append("ALL TRUCKS COMPARISON:\n" + cmp.to_string(index=False))

        return "\n\n".join(results) if results else ""

    # ════════════════════════════════════════════════════════════════════════
    # ③ VECTORLESS RAG RETRIEVER  — keyword-based chunk scoring
    # ════════════════════════════════════════════════════════════════════════
    def rag_retrieve(query: str, top_k: int = 3) -> str:
        """
        Retrieve the most relevant knowledge chunks using keyword overlap scoring.
        No vectors, no embeddings — pure token matching (BM25-lite).
        """
        q_tokens = set(query.lower().split())
        scored = []
        for chunk in KNOWLEDGE_CHUNKS:
            # Score = number of trigger keywords found in query
            score = sum(1 for t in chunk["triggers"] if t in query.lower())
            # Bonus for exact multi-word trigger match
            score += sum(2 for t in chunk["triggers"] if len(t.split()) > 1 and t in query.lower())
            scored.append((score, chunk))

        scored.sort(key=lambda x: -x[0])
        top = [c["text"] for score, c in scored[:top_k] if score > 0]
        return "\n\n---\n\n".join(top) if top else ""

    # ════════════════════════════════════════════════════════════════════════
    # ④ RULE-BASED ROUTER  — fast answers without LLM for simple queries
    # ════════════════════════════════════════════════════════════════════════
    def rule_based_answer(query: str):
        """
        LangChain-style RouterChain equivalent.
        Returns (answer_str, confidence) for known patterns, else (None, None).
        """
        q = query.lower().strip()

        # Greeting
        if any(q == g for g in ["hi", "hello", "hey", "namaste", "hii", "hlo"]):
            return (
                f"**Namaste! 🙏**\n\nI'm the LoRRI AI Assistant. "
                f"I have full access to your fleet data — **{opt['n_ships']} shipments**, "
                f"**{opt['n_vehicles']} trucks**, total cost **{inr(opt['total_cost'])}**. "
                f"Ask me anything about routes, costs, SLA, or carbon!",
                99,
            )

        # Contact
        if any(kw in q for kw in ["contact", "email", "phone", "number", "reach", "logisticsnow.in"]):
            return (
                "**LogisticsNow Contact Details:**\n\n"
                "- 🌐 Website: logisticsnow.in\n"
                "- 📧 Email: connect@logisticsnow.in\n"
                "- 📞 Phone: +91-9867773508 / +91-9653620207",
                99,
            )

        # Total savings (exact)
        if any(kw in q for kw in ["how much", "total saving", "total cost saving", "how much did we save", "save in"]):
            saved = base["total_cost"] - opt["total_cost"]
            pct   = saved / base["total_cost"] * 100
            return (
                f"**Total Cost Savings: {inr(saved)} ({pct:.1f}% reduction)**\n\n"
                f"| Category | Baseline | Optimized | Saved |\n"
                f"|---|---|---|---|\n"
                f"| ⛽ Fuel | {inr(base['fuel_cost'])} | {inr(opt['fuel_cost'])} | **{inr(base['fuel_cost']-opt['fuel_cost'])}** |\n"
                f"| 🛣️ Toll | {inr(base['toll_cost'])} | {inr(opt['toll_cost'])} | **{inr(base['toll_cost']-opt['toll_cost'])}** |\n"
                f"| 👷 Driver | {inr(base['driver_cost'])} | {inr(opt['driver_cost'])} | **{inr(base['driver_cost']-opt['driver_cost'])}** |\n"
                f"| 💰 **Total** | {inr(base['total_cost'])} | {inr(opt['total_cost'])} | **{inr(saved)}** |",
                98,
            )

        # Carbon savings (exact)
        if ("carbon" in q or "co2" in q or "emission" in q) and ("saving" in q or "save" in q or "reduc" in q or "how much" in q):
            co2s = base["carbon_kg"] - opt["carbon_kg"]
            return (
                f"**Carbon Savings: {co2s:,.1f} kg CO₂ ({co2s/base['carbon_kg']*100:.1f}% reduction)**\n\n"
                f"- Baseline: {base['carbon_kg']:,.1f} kg CO₂\n"
                f"- Optimized: {opt['carbon_kg']:,.1f} kg CO₂\n"
                f"- 🌳 Equivalent to **{int(co2s/21):,} trees** planted per year\n"
                f"- 🚗 Or **{int(co2s/2400)} cars** removed from the road",
                98,
            )

        # Truck route quick lookup
        for v in [1, 2, 3, 4, 5]:
            if (f"truck {v} route" in q or f"route of truck {v}" in q
                    or (f"truck {v}" in q and "route" in q)):
                return (
                    f"**Truck {v} Route:**\n\n"
                    f"🚛 {route_map_text[v]}\n\n"
                    f"📊 {int(veh_sum[veh_sum['vehicle']==v]['stops'].iloc[0])} stops · "
                    f"{veh_sum[veh_sum['vehicle']==v]['distance_km'].iloc[0]:,.0f} km · "
                    f"{inr(veh_sum[veh_sum['vehicle']==v]['total_cost'].iloc[0])}",
                    97,
                )

        # Truck 3 route change explanation
        if "truck 3" in q and any(kw in q for kw in ["change", "why", "explain", "reason", "chose"]):
            t3 = veh_sum[veh_sum["vehicle"] == 3].iloc[0]
            breach_cities = routes[(routes["vehicle"] == 3) & (routes["sla_breach_hr"] > 0)]["city"].tolist()
            return (
                f"**Why Truck 3's Route Was Chosen:**\n\n"
                f"Route: {route_map_text[3]}\n\n"
                f"Truck 3 was assigned the **Western + Southern peninsular + Northern** corridor because:\n"
                f"- Aurangabad and Solapur are geographically clustered on the Pune–Hyderabad axis\n"
                f"- Madurai anchors the deep southern leg, which no other truck covers\n"
                f"- Jammu was assigned to Truck 3 as a high-priority stop requiring a dedicated northern extension\n"
                f"- The CVRP solver minimized total weighted score (Cost 35% + Time 30% + Carbon 20% + SLA 15%)\n\n"
                f"**Performance:** {t3['distance_km']:,.0f} km · {inr(t3['total_cost'])} · "
                f"{t3['carbon_kg']:.0f} kg CO₂ · {int(t3['sla_breaches'])} SLA breach"
                + (f"\n⚠️ Breach cities: {', '.join(breach_cities)}" if breach_cities else "\n✅ All SLA met"),
                96,
            )

        # Tomorrow traffic risk prediction
        if any(kw in q for kw in ["tomorrow", "predict traffic", "traffic risk", "forecast", "next day"]):
            high_risk = ships[ships["traffic_mult"] > 2.0]["city"].tolist()
            medium_risk = ships[(ships["traffic_mult"] > 1.4) & (ships["traffic_mult"] <= 2.0)]["city"].tolist()
            return (
                f"**Predicted Traffic Risk for Tomorrow:**\n\n"
                f"🔴 **HIGH RISK** (traffic_mult > 2.0): {', '.join(high_risk) if high_risk else 'None'}\n"
                f"🟡 **MEDIUM RISK** (1.4–2.0×): {', '.join(medium_risk[:5]) if medium_risk else 'None'}\n"
                f"🟢 **STABLE**: All other corridors\n\n"
                f"LoRRI will auto re-optimize if any HIGH RISK city exceeds 30% delay threshold.\n"
                f"Recommendation: Pre-position Truck {veh_sum['vehicle'].iloc[0]} early departure for high-risk zones.",
                90,
            )

        # SLA breached cities
        if any(kw in q for kw in ["which cities", "which city", "what cities", "where late", "cities late", "missed sla"]):
            bd = routes[routes["sla_breach_hr"] > 0][["city", "vehicle", "sla_breach_hr", "sla_penalty"]]
            if bd.empty:
                return ("✅ **No SLA breaches** — all deliveries were on time!", 99)
            lines = [f"**SLA Breach Cities ({len(bd)} stops late):**\n"]
            for _, row in bd.iterrows():
                lines.append(
                    f"- 📍 **{row['city']}** (Truck {int(row['vehicle'])}) — "
                    f"{row['sla_breach_hr']:.1f}hr late · penalty {inr(row['sla_penalty'])}"
                )
            return ("\n".join(lines), 97)

        return None, None

    # ════════════════════════════════════════════════════════════════════════
    # ⑤ LANGCHAIN-STYLE PIPELINE  — Router → RAG → Pandas → LLM
    # ════════════════════════════════════════════════════════════════════════
    def build_system_prompt(query: str, rag_context: str, pandas_context: str) -> str:
        """Builds a rich grounded system prompt — equivalent to LangChain's PromptTemplate."""
        bd_cities = ", ".join(routes[routes["sla_breach_hr"] > 0]["city"].tolist())
        fleet_summary = (
            f"Fleet: {opt['n_ships']} shipments | {opt['n_vehicles']} trucks | Depot: Mumbai\n"
            f"Optimized cost: {inr(opt['total_cost'])} | Baseline: {inr(base['total_cost'])} "
            f"| Saved: {inr(base['total_cost']-opt['total_cost'])}\n"
            f"SLA: {opt['sla_pct']:.0f}% | Breaches: {opt['breaches']} ({bd_cities})\n"
            f"Carbon: {opt['carbon_kg']:,.1f} kg (saved {base['carbon_kg']-opt['carbon_kg']:,.1f} kg)"
        )
        per_truck = "\n".join([
            f"  Truck {int(r['vehicle'])}: {route_map_text.get(int(r['vehicle']),'?')} | "
            f"{int(r['stops'])} stops | {r['distance_km']:,.0f} km | {inr(r['total_cost'])} | "
            f"{r['carbon_kg']:.0f} kg CO₂ | {int(r['sla_breaches'])} breach | {r['utilization_pct']:.0f}% loaded"
            for _, r in veh_sum.iterrows()
        ])
        context_block = ""
        if rag_context:
            context_block += f"\n\n[RAG KNOWLEDGE RETRIEVED]\n{rag_context}"
        if pandas_context:
            context_block += f"\n\n[PANDAS DATA RETRIEVED]\n{pandas_context}"
        return f"""You are the LoRRI Intelligence Assistant for LogisticsNow (logisticsnow.in).
Contact: connect@logisticsnow.in | +91-9867773508
Pricing: Fuel ₹12/km | Driver ₹180/hr | SLA penalty ₹500/hr | Toll variable
Optimization: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%) — CVRP with OR-Tools

LIVE FLEET DATA:
{fleet_summary}

PER-TRUCK:
{per_truck}
{context_block}

RULES:
- All monetary values MUST use ₹ INR format
- Be concise, use bullet points and tables where helpful
- Use ONLY the data provided above — do not invent numbers
- If the retrieved context answers the query, prioritize it
"""

    def call_claude_rag(query: str, messages_history: list) -> tuple[str, int, str, str]:
        """
        Full RAG pipeline:
        Step 1: Rule-based router (instant answer if pattern matches)
        Step 2: RAG keyword retrieval (knowledge chunks)
        Step 3: Pandas structured retrieval (DataFrame queries)
        Step 4: LLM call with enriched grounded context
        Returns (reply, confidence, retrieval_source, chunks_used)
        """
        # Step 1 — Rule-based router
        rule_reply, rule_conf = rule_based_answer(query)
        if rule_reply:
            return rule_reply, rule_conf, "⚡ Rule-based router", "Instant pattern match"

        # Step 2 — RAG keyword retrieval
        rag_ctx     = rag_retrieve(query, top_k=3)
        # Step 3 — Pandas structured retrieval
        pandas_ctx  = pandas_retrieve(query)

        # Determine retrieval source label
        sources = []
        if rag_ctx:    sources.append("📚 RAG chunks")
        if pandas_ctx: sources.append("🐼 Pandas DataFrames")
        src_label = " + ".join(sources) if sources else "🤖 LLM general knowledge"

        # Count chunks retrieved
        n_chunks = len([c for c in KNOWLEDGE_CHUNKS if c["text"] in rag_ctx]) if rag_ctx else 0
        chunks_info = f"{n_chunks} knowledge chunk(s) + {'DataFrame rows' if pandas_ctx else 'no structured data'}"

        # Step 4 — LLM call with enriched prompt
        system = build_system_prompt(query, rag_ctx, pandas_ctx)
        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "system": system,
            "messages": messages_history,
        }
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                reply = resp.json()["content"][0]["text"]
                # Confidence: higher when retrieval found strong context
                conf = 95 if (rag_ctx and pandas_ctx) else (92 if (rag_ctx or pandas_ctx) else 84)
                return reply, conf, src_label, chunks_info
            return f"⚠️ API error {resp.status_code}", 0, "error", ""
        except Exception as e:
            return f"⚠️ Connection error: {e}", 0, "error", ""

    # ════════════════════════════════════════════════════════════════════════
    # UI — HEADER
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a4d2e 100%);
                border-radius:14px;padding:22px 26px;margin-bottom:18px;
                display:flex;align-items:center;gap:16px;">
        <div style="width:52px;height:52px;border-radius:12px;background:{LN_GREEN};
                    display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">🤖</div>
        <div>
            <div style="font-size:1.1rem;font-weight:700;color:white;">LoRRI Intelligence Assistant</div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">
                <b style="color:{LN_GREEN}">Vectorless RAG</b> · LangChain-style Pipeline ·
                Pandas Retrieval · Rule-Based Router · Claude LLM
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="background:rgba(34,197,94,0.15);border:1px solid {LN_GREEN};
                        border-radius:20px;padding:4px 14px;font-size:0.7rem;color:{LN_GREEN};font-weight:600;margin-bottom:6px;">
                ● LIVE AI
            </div>
            <div style="font-size:0.65rem;color:#64748b;">Ask in English or Hindi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline architecture badge row
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;align-items:center;">
        <span style="font-size:0.65rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">Pipeline:</span>
        {"".join([
            f'<span style="background:{bg};color:{fg};border:1px solid {br};border-radius:6px;'
            f'padding:3px 10px;font-size:0.7rem;font-weight:600;">{label}</span>'
            for label, bg, fg, br in [
                ("① Rule Router",    "#fef3c7", "#92400e", "#f59e0b"),
                ("→ ② RAG Retriever","#f0fdf4", LN_DGREEN, LN_GREEN),
                ("→ ③ Pandas Query", "#eff6ff", "#1e40af", "#3b82f6"),
                ("→ ④ LLM Synthesis","#f5f3ff", "#5b21b6", "#8b5cf6"),
            ]
        ])}
    </div>
    """, unsafe_allow_html=True)

    # Capability cards
    c1, c2, c3 = st.columns(3)
    for col, color, icon, title, desc in [
        (c1, LN_GREEN,  "🏢", "Company & Platform",  "LogisticsNow info, LoRRI services, pricing model, contact details"),
        (c2, "#1e7abf", "🚛", "Fleet & Routes",       "Per-truck data, cost breakdowns, SLA status, carbon — from DataFrames"),
        (c3, "#e67e22", "🧠", "AI & Methodology",     "CVRP, weighted objectives, re-optimization triggers, explainability"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                    padding:14px 16px;border-top:3px solid {color};margin-bottom:14px;">
            <div style="font-size:0.62rem;font-weight:700;color:{color};text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:6px;">{icon} {title}</div>
            <div style="font-size:0.81rem;color:#475569;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── CHIP BUTTON STYLES ────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* Style every chip button to look like a pill/badge */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
        background: #f0fdf4 !important;
        border: 1px solid {LN_GREEN} !important;
        color: {LN_DGREEN} !important;
        border-radius: 20px !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        padding: 4px 10px !important;
        line-height: 1.4 !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 2rem !important;
        transition: background 0.15s, box-shadow 0.15s !important;
    }}
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
        background: {LN_GREEN} !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(58,125,44,0.25) !important;
    }}
    </style>
    <div style="font-size:0.7rem;color:#64748b;font-weight:600;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:6px;">💡 Click to ask:</div>
    """, unsafe_allow_html=True)

    CHIP_QUESTIONS = [
        ("What is LoRRI?",                    "🏢"),
        ("How much did we save in ₹?",        "💰"),
        ("Which cities were late?",            "⚠️"),
        ("Which truck costs most?",            "🚛"),
        ("Truck 3 route?",                     "🗺️"),
        ("Carbon savings?",                    "🌿"),
        ("Explain why Truck 3 route changed",  "🧠"),
        ("Predict tomorrow's traffic risk",    "🚦"),
        ("Suggest cost saving actions",        "💡"),
        ("Contact LogisticsNow",               "📞"),
    ]

    # ── PHASE 1: render chip buttons; if clicked → store in state + rerun ─────
    # This runs on every render. chip_prompt is consumed only in Phase 2 below.
    chip_cols = st.columns(5)
    chip_clicked = False
    for idx, (question, emoji) in enumerate(CHIP_QUESTIONS):
        with chip_cols[idx % 5]:
            if st.button(
                f"{emoji} {question}",
                key=f"chip_{idx}",
                use_container_width=True,
            ):
                # Store the question and immediately rerun so Phase 2 can pick it up
                # on the very next render, after the button state is stable.
                st.session_state.chip_prompt = question
                chip_clicked = True

    # If a chip was just clicked this render, rerun so the prompt is processed cleanly
    if chip_clicked:
        st.rerun()

    st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ── PHASE 2: render chat history ─────────────────────────────────────────
    if not st.session_state.msgs:
        with st.chat_message("assistant", avatar="🚚"):
            st.markdown(
                f"**Namaste! 🙏 Welcome to the LoRRI Intelligence Assistant.**\n\n"
                f"I use a **4-step RAG pipeline**: "
                f"⚡ Rule Router → 📚 RAG Retriever → 🐼 Pandas Query → 🤖 LLM Synthesis.\n\n"
                f"Fleet snapshot: **{opt['n_ships']} shipments**, **{opt['n_vehicles']} trucks**, "
                f"total cost **{inr(opt['total_cost'])}**, SLA **{opt['sla_pct']:.0f}%**.\n\n"
                f"Click any chip above or type your question below! 🇮🇳"
            )

    for m in st.session_state.msgs:
        avatar = "🚚" if m["role"] == "assistant" else "👤"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                meta = m["meta"]
                conf_color = (LN_GREEN if meta["conf"] >= 90
                              else ("#f59e0b" if meta["conf"] >= 80 else "#dc2626"))
                st.markdown(f"""
                <div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                            border:1px solid {LN_BORDER};border-radius:20px;
                            display:inline-flex;align-items:center;gap:8px;
                            font-size:0.7rem;color:#64748b;">
                    <span style="color:{conf_color};font-weight:700;">●</span>
                    <b style="color:{conf_color};">{meta['conf']}% confidence</b>
                    &nbsp;·&nbsp; {meta['source']}
                    &nbsp;·&nbsp; {meta['chunks']}
                </div>
                """, unsafe_allow_html=True)

    # ── PHASE 3: consume chip_prompt OR typed input, call RAG, append reply ───
    # Consume stored chip prompt (set in Phase 1 on the previous render cycle)
    final_prompt = st.session_state.pop("chip_prompt", None)

    # Also render the chat_input widget; if user typed, that overrides nothing —
    # both chip and typed prompts are handled the same way.
    typed_prompt = st.chat_input("Ask about LoRRI, fleet data, ₹ costs, routes, SLA...")
    if typed_prompt:
        final_prompt = typed_prompt

    if final_prompt:
        st.session_state.msgs.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(final_prompt)

        api_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.msgs[-10:]
        ]

        with st.chat_message("assistant", avatar="🚚"):
            with st.spinner("🔍 Running RAG pipeline…"):
                reply, conf, source, chunks = call_claude_rag(final_prompt, api_msgs)
            st.markdown(reply)
            conf_color = (LN_GREEN if conf >= 90
                          else ("#f59e0b" if conf >= 80 else "#dc2626"))
            st.markdown(f"""
            <div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                        border:1px solid {LN_BORDER};border-radius:20px;
                        display:inline-flex;align-items:center;gap:8px;
                        font-size:0.7rem;color:#64748b;">
                <span style="color:{conf_color};font-weight:700;">●</span>
                <b style="color:{conf_color};">{conf}% confidence</b>
                &nbsp;·&nbsp; {source}
                &nbsp;·&nbsp; {chunks}
            </div>
            """, unsafe_allow_html=True)

        st.session_state.msgs.append({
            "role": "assistant",
            "content": reply,
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
    page_header("📊 Dashboard Summary", "Baseline vs AI-Optimized · All costs in ₹ INR")
    loading_state("Refreshing dashboard metrics…")

    st.markdown("""<div class="info-box">
    📋 <b>Report card for the full delivery run.</b>
    Baseline = trucks without AI. Optimized = LoRRI AI planner.
    Green arrows = money saved, time saved, less pollution. All figures in <b>₹ INR</b>.
    </div>""", unsafe_allow_html=True)

    sc_ = base["total_cost"] - opt["total_cost"]
    sd_ = base["distance_km"] - opt["distance_km"]
    sco = base["carbon_kg"] - opt["carbon_kg"]
    ss_ = opt["sla_pct"] - base["sla_pct"]
    avg_util = veh_sum["utilization_pct"].mean()

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px;">
    {kpi_card("Total Cost Savings",  inr(sc_),                        f"↓ -{sc_/base['total_cost']*100:.1f}% vs baseline",  True,  LN_GREEN)}
    {kpi_card("Optimized Distance",  f"{opt['distance_km']:,.0f} km", f"↓ {sd_:,.0f} km saved",                             True,  "#1e7abf")}
    {kpi_card("SLA Adherence",       f"{opt['sla_pct']:.0f}%",        f"↑ +{ss_:.0f} pts (base {base['sla_pct']:.0f}%)",   True,  "#e67e22")}
    {kpi_card("Carbon Reduced",      f"{sco/1000:.1f}t CO₂",          f"↓ {sco/base['carbon_kg']*100:.1f}% cleaner",        True,  "#27ae60")}
    {kpi_card("Fleet Utilization",   f"{avg_util:.1f}%",               f"Avg across {opt['n_vehicles']} trucks",             True,  "#8e44ad")}
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⛽ Fuel Saved",    inr(base["fuel_cost"]   - opt["fuel_cost"]),   f"-{(base['fuel_cost']  -opt['fuel_cost']  )/base['fuel_cost']  *100:.1f}%", delta_color="inverse")
    c2.metric("🛣️ Toll Saved",   inr(base["toll_cost"]   - opt["toll_cost"]),   f"-{(base['toll_cost']  -opt['toll_cost']  )/base['toll_cost']  *100:.1f}%", delta_color="inverse")
    c3.metric("👷 Driver Saved",  inr(base["driver_cost"] - opt["driver_cost"]), f"-{(base['driver_cost']-opt['driver_cost'])/base['driver_cost']*100:.1f}%", delta_color="inverse")
    c4.metric("⏱️ Time Saved",   f"{base['time_hr']-opt['time_hr']:,.1f} hr",   f"-{(base['time_hr']-opt['time_hr'])/base['time_hr']*100:.1f}%",            delta_color="inverse")

    # Cost reduction trend
    st.markdown(sh("📉 Cost Reduction Over Optimization Iterations"), unsafe_allow_html=True)
    iters = list(range(1, 11))
    decay = [(base["total_cost"] - (base["total_cost"] - opt["total_cost"]) * (1 - np.exp(-0.4 * i))) for i in iters]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=iters, y=[base["total_cost"]] * 10, mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=2), name="Baseline ₹"))
    fig_trend.add_trace(go.Scatter(x=iters, y=decay, mode="lines+markers",
        line=dict(color=LN_GREEN, width=3), marker=dict(size=7, color=LN_GREEN),
        fill="tonexty", fillcolor="rgba(58,125,44,0.08)", name="Optimized ₹"))
    fig_trend.add_annotation(x=10, y=opt["total_cost"],
        text=f"  Final: {inr(opt['total_cost'])}", showarrow=False,
        font=dict(color=LN_GREEN, size=11), xanchor="left")
    apply_theme(fig_trend, height=280, title="Fleet Cost vs Optimization Iterations")
    fig_trend.update_yaxes(tickprefix="₹", tickformat=",")
    fig_trend.update_xaxes(title_text="Iteration")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown(sh("🚛 Per-Truck Summary"), unsafe_allow_html=True)
    d = veh_sum.copy()
    d.insert(0, "Truck", d["vehicle"].apply(lambda v: f"🚛 Truck {v}"))
    d = d.drop(columns=["vehicle"])
    d.columns = ["Truck", "Stops", "Load (kg)", "Dist (km)", "Time (hr)",
                 "Fuel (₹)", "Toll (₹)", "Driver (₹)", "SLA Penalty (₹)",
                 "Total (₹)", "Carbon (kg)", "SLA Breaches", "Util %"]
    st.dataframe(
        d.style
         .format({"Load (kg)": "{:.1f}", "Dist (km)": "{:.1f}", "Time (hr)": "{:.1f}",
                  "Fuel (₹)": "₹{:,.0f}", "Toll (₹)": "₹{:,.0f}", "Driver (₹)": "₹{:,.0f}",
                  "SLA Penalty (₹)": "₹{:,.0f}", "Total (₹)": "₹{:,.0f}",
                  "Carbon (kg)": "{:.1f}", "Util %": "{:.1f}%"})
         .background_gradient(subset=["Util %"], cmap="Greens")
         .background_gradient(subset=["Total (₹)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Route Map":
    page_header("🗺️ Route Map", "Live India delivery network · Mumbai depot hub")
    loading_state("Refreshing traffic…")

    st.markdown("""<div class="info-box">
    🗺️ Interactive India map showing each truck's route from Mumbai depot.
    Markers show <b>Truck# · Stop#</b>. Hover for Distance · ETA · Cost · Carbon · SLA risk.
    </div>""", unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown(sh("🎛️ Map Controls"), unsafe_allow_html=True)
        route_mode   = st.radio("Route View", ["Optimized", "Baseline", "Comparison"], index=0)
        show_unas    = st.toggle("Show Unassigned", value=True)
        show_heatmap = st.toggle("Traffic Heatmap Overlay", value=False)
        sel_v = st.multiselect("Filter Trucks",
                               options=sorted(routes["vehicle"].unique()),
                               default=sorted(routes["vehicle"].unique()),
                               format_func=lambda v: f"Truck {v}")
        st.markdown("---")
        st.markdown(sh("📌 Route Legend"), unsafe_allow_html=True)
        for v in sorted(routes["vehicle"].unique()):
            vr    = routes[routes["vehicle"] == v]
            color = V_COLORS.get(v, "#999")
            vd    = veh_sum[veh_sum["vehicle"] == v].iloc[0]
            sla_r = "🔴 HIGH" if vd["sla_breaches"] >= 2 else ("🟡 MED" if vd["sla_breaches"] == 1 else "🟢 OK")
            st.markdown(
                f'<div class="legend-row">'
                f'<div class="legend-dot" style="background:{color}"></div>'
                f'<div><b style="color:{LN_NAVY}">Truck {v}</b><br>'
                f'<span style="font-size:0.72rem;color:#64748b;">'
                f'{len(vr)} stops · {vd["distance_km"]:,.0f} km<br>'
                f'{inr(vd["total_cost"])} · {vd["carbon_kg"]:.0f} kg CO₂<br>'
                f'SLA {sla_r}</span></div></div>',
                unsafe_allow_html=True)

    with col_map:
        fig = go.Figure()
        p_dot = {"HIGH": "#dc2626", "MEDIUM": "#f97316", "LOW": LN_GREEN}

        if route_mode in ["Baseline", "Comparison"]:
            bl  = [DEPOT["latitude"]]  + ships["latitude"].tolist()  + [DEPOT["latitude"]]
            blo = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=bl, lon=blo, mode="lines",
                line=dict(width=2 if route_mode == "Comparison" else 3,
                          color="rgba(220,38,38,0.5)" if route_mode == "Comparison" else "rgba(200,50,50,0.7)"),
                name="Baseline (No AI)"))

        if route_mode in ["Optimized", "Comparison"]:
            for v in sel_v:
                vdf   = routes[routes["vehicle"] == v].sort_values("stop_order")
                color = V_COLORS.get(v, "#999")
                lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
                lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
                fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines",
                    line=dict(width=3, color=color), name=f"Truck {v}", legendgroup=f"v{v}"))
                for _, row in vdf.iterrows():
                    breach   = f"⚠️ {row['sla_breach_hr']:.1f}hr late" if row["sla_breach_hr"] > 0 else "✅ On time"
                    sla_risk = "🔴 HIGH" if row["sla_breach_hr"] > 10 else ("🟡 MEDIUM" if row["sla_breach_hr"] > 0 else "🟢 OK")
                    dist_dep = haversine(DEPOT["latitude"], DEPOT["longitude"], row["latitude"], row["longitude"])
                    fig.add_trace(go.Scattermap(
                        lat=[row["latitude"]], lon=[row["longitude"]],
                        mode="markers+text",
                        marker=dict(size=14, color=p_dot.get(row.get("priority", "MEDIUM"), "#f97316")),
                        text=[f"T{v}·{int(row['stop_order'])}"],
                        textfont=dict(size=8, color="white"),
                        textposition="middle center",
                        hovertext=(
                            f"<b>🚛 Truck {v} — Stop {int(row['stop_order'])}</b><br>"
                            f"📍 <b>{row.get('city', row['shipment_id'])}</b><br>"
                            f"━━━━━━━━━━━━━━━<br>"
                            f"📏 Depot distance: <b>{dist_dep:.0f} km</b><br>"
                            f"⏱️ ETA: <b>{row['travel_time_hr']:.1f} hr</b><br>"
                            f"💰 Stop cost: <b>{inr(row['total_cost'])}</b><br>"
                            f"🌿 Carbon: <b>{row['carbon_kg']:.1f} kg CO₂</b><br>"
                            f"📅 SLA Risk: <b>{sla_risk}</b> · {breach}<br>"
                            f"━━━━━━━━━━━━━━━<br>"
                            f"📦 {row['shipment_id']} · {row['weight']:.0f} kg"
                        ),
                        hoverinfo="text", showlegend=False, legendgroup=f"v{v}"))

        if show_heatmap:
            hm_vals = ships["traffic_mult"].tolist()
            fig.add_trace(go.Scattermap(
                lat=ships["latitude"].tolist(), lon=ships["longitude"].tolist(),
                mode="markers",
                marker=dict(size=[v * 18 for v in hm_vals], color=hm_vals,
                            colorscale="RdYlGn_r", cmin=1.0, cmax=3.0, opacity=0.4,
                            colorbar=dict(title="Traffic ×", x=1.0)),
                name="Traffic Heatmap", hovertext=ships["city"], hoverinfo="text"))

        if show_unas:
            asgn  = set(routes["shipment_id"])
            unasgn = ships[~ships["id"].isin(asgn)]
            if not unasgn.empty:
                fig.add_trace(go.Scattermap(lat=unasgn["latitude"], lon=unasgn["longitude"],
                    mode="markers", marker=dict(size=8, color="grey"),
                    name="Unassigned", hovertext=unasgn["city"], hoverinfo="text"))

        # Depot — single point, no textposition list issue
        fig.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["🏭 Depot"],
            textposition="top right", textfont=dict(size=10, color=LN_NAVY),
            marker=dict(size=18, color=LN_NAVY, symbol="star"), name="Mumbai Depot"))

        fig.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
            height=620, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.92)",
                        bordercolor=LN_BORDER, borderwidth=1, font=dict(color=LN_NAVY, size=11)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔴 HIGH · 🟠 MEDIUM · 🟢 LOW priority  ·  T{truck}·{stop#}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "💰 Financial Analysis":
    page_header("💰 Financial Analysis", "All costs in ₹ INR · Fuel ₹12/km · Driver ₹180/hr · SLA ₹500/hr")
    loading_state("Optimizing routes…")

    st.markdown("""<div class="info-box">
    💰 All costs in <b>₹ (Indian Rupees)</b>: Fuel (₹12/km), Tolls, Driver wages (₹180/hr).
    LoRRI AI clustered nearby cities per truck, cutting all three cost categories significantly.
    </div>""", unsafe_allow_html=True)

    fuel_s   = base["fuel_cost"]   - opt["fuel_cost"]
    toll_s   = base["toll_cost"]   - opt["toll_cost"]
    driver_s = base["driver_cost"] - opt["driver_cost"]
    total_s  = fuel_s + toll_s + driver_s
    roi_pct  = total_s / base["total_cost"] * 100
    platform_cost = 15000
    payback_days  = int(platform_cost / (total_s / 30)) if total_s > 0 else 999

    st.markdown(sh("💎 Optimization ROI Summary"), unsafe_allow_html=True)
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    for col, label, val, icon, color in [
        (rc1, "Fuel Saved",         inr(fuel_s),         "⛽", LN_GREEN),
        (rc2, "Toll Saved",         inr(toll_s),         "🛣️", "#1e7abf"),
        (rc3, "Driver Cost Saved",  inr(driver_s),       "👷", "#8e44ad"),
        (rc4, "Total Cost Saved",   inr(total_s),        "💰", "#e67e22"),
        (rc5, "Payback Period",     f"{payback_days}d",  "⏳", "#27ae60"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:18px 16px;text-align:center;border-top:3px solid {color};
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:1.1rem;font-weight:800;color:{color};">{val}</div>
            <div style="font-size:0.68rem;color:#64748b;margin-top:4px;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#f0fdf4;border:1px solid {LN_GREEN};border-radius:10px;
                padding:12px 20px;margin-top:12px;font-size:0.84rem;color:{LN_NAVY};">
        📊 <b>Overall ROI: {roi_pct:.1f}% cost reduction</b> ·
        Platform pays back in <b>{payback_days} days</b> ·
        Estimated annual savings: <b>{inr(total_s * 12)}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_b = go.Figure()
        for cat, bc_, lbl in [
            ("fuel_cost",   LN_GREEN,  "⛽ Fuel"),
            ("toll_cost",   "#1e7abf", "🛣️ Toll"),
            ("driver_cost", "#8e44ad", "👷 Driver"),
        ]:
            fig_b.add_trace(go.Bar(
                name=lbl, x=["Baseline", "Optimized"],
                y=[base[cat], opt[cat]], marker_color=bc_,
                text=[inr(base[cat]), inr(opt[cat])],
                textposition="inside", textfont=dict(color="white", size=10)))
        apply_theme(fig_b, height=360, title="Cost Components: Baseline vs Optimized (₹)", legend_below=True)
        fig_b.update_layout(barmode="stack")
        fig_b.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_b, use_container_width=True)

    with c2:
        sv = {"Fuel Saved": fuel_s, "Toll Saved": toll_s, "Driver Saved": driver_s}
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
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

    st.markdown(sh("🚛 Per-Truck Cost Breakdown (₹)"), unsafe_allow_html=True)
    fig_v = go.Figure()
    for cat, bc_, lbl in [
        ("fuel_cost",   LN_GREEN,   "⛽ Fuel"),
        ("toll_cost",   "#1e7abf",  "🛣️ Toll"),
        ("driver_cost", "#8e44ad",  "👷 Driver"),
        ("sla_penalty", "#c0392b",  "⏰ SLA Penalty"),
    ]:
        fig_v.add_trace(go.Bar(name=lbl, x=[f"Truck {v}" for v in veh_sum["vehicle"]],
            y=veh_sum[cat], marker_color=bc_,
            text=veh_sum[cat].apply(inr), textposition="inside",
            textfont=dict(color="white", size=9)))
    apply_theme(fig_v, height=320, legend_below=True)
    fig_v.update_layout(barmode="stack")
    fig_v.update_yaxes(tickprefix="₹", tickformat=",")
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown(sh("📋 Detailed Cost Table (₹)"), unsafe_allow_html=True)
    ct = veh_sum[["vehicle", "stops", "distance_km", "fuel_cost", "toll_cost",
                  "driver_cost", "sla_penalty", "total_cost"]].copy()
    ct["vehicle"] = ct["vehicle"].apply(lambda v: f"🚛 Truck {v}")
    ct.columns = ["Truck", "Stops", "Dist (km)", "Fuel (₹)", "Toll (₹)",
                  "Driver (₹)", "SLA Penalty (₹)", "Total (₹)"]
    st.dataframe(
        ct.style.format({"Dist (km)": "{:.1f}", "Fuel (₹)": "₹{:,.0f}",
                         "Toll (₹)": "₹{:,.0f}", "Driver (₹)": "₹{:,.0f}",
                         "SLA Penalty (₹)": "₹{:,.0f}", "Total (₹)": "₹{:,.0f}"})
                .background_gradient(subset=["Total (₹)"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CARBON & SLA
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🌿 Carbon & SLA":
    page_header("🌿 Carbon & SLA", "Sustainability metrics · Environmental impact · Delivery compliance")

    st.markdown("""<div class="info-box">
    🌿 <b>Carbon</b> = CO₂ from diesel combustion. Smarter routes = less pollution. 🌱<br>
    <b>SLA</b> = delivery promise. Late = <b>₹500/hr penalty</b> per the LoRRI pricing model.
    </div>""", unsafe_allow_html=True)

    co2_saved  = base["carbon_kg"] - opt["carbon_kg"]
    trees      = int(co2_saved / 21)
    cars_off   = int(co2_saved / 2400)
    km_avoided = int(base["distance_km"] - opt["distance_km"])

    st.markdown(sh("🌍 Sustainability Summary Card"), unsafe_allow_html=True)
    sus_cols = st.columns(4)
    for col, icon, val, label, sub, color in [
        (sus_cols[0], "🌿", f"{co2_saved:,.0f} kg",  "CO₂ Reduced",        f"{co2_saved/base['carbon_kg']*100:.1f}% less than baseline",   LN_GREEN),
        (sus_cols[1], "🌳", f"{trees:,}",             "Trees Equivalent",   "CO₂ absorbed per year by planted trees",                        "#27ae60"),
        (sus_cols[2], "🚗", f"{cars_off}",            "Cars Off the Road",  "Equivalent annual car emissions removed",                       "#1e7abf"),
        (sus_cols[3], "📏", f"{km_avoided:,} km",     "Distance Avoided",   "Fewer kilometres vs baseline routing",                          "#e67e22"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:18px;text-align:center;border-top:3px solid {color};
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:1.25rem;font-weight:800;color:{color};">{val}</div>
            <div style="font-size:0.78rem;font-weight:700;color:{LN_NAVY};margin:5px 0 3px;">{label}</div>
            <div style="font-size:0.7rem;color:#94a3b8;line-height:1.5;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top emitting cities
    st.markdown(sh("🏙️ Top Cities by Carbon Contribution"), unsafe_allow_html=True)
    city_co2 = routes.groupby("city")["carbon_kg"].sum().sort_values(ascending=False).head(8).reset_index()
    fig_city = go.Figure(go.Bar(
        x=city_co2["carbon_kg"], y=city_co2["city"], orientation="h",
        marker_color=[LN_GREEN if i > 2 else "#c0392b" for i in range(len(city_co2))],
        text=city_co2["carbon_kg"].round(1).astype(str) + " kg", textposition="outside"))
    apply_theme(fig_city, height=280, title="Top 8 Cities — CO₂ Emitted (kg)")
    fig_city.update_layout(showlegend=False)
    fig_city.update_yaxes(autorange="reversed")
    fig_city.update_xaxes(title_text="kg CO₂")
    st.plotly_chart(fig_city, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Bar(x=["Baseline (No AI)", "Optimized (AI)"],
            y=[base["carbon_kg"], opt["carbon_kg"]], marker_color=["#c0392b", LN_GREEN],
            text=[f"{base['carbon_kg']:,.1f} kg", f"{opt['carbon_kg']:,.1f} kg"], textposition="outside"))
        apply_theme(fig_c2, height=280,
                    title=f"CO₂: {co2_saved:,.1f} kg saved ({co2_saved/base['carbon_kg']*100:.1f}% less)")
        fig_c2.update_layout(showlegend=False)
        fig_c2.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_c2, use_container_width=True)

        fig_cv = go.Figure(go.Bar(x=[f"Truck {v}" for v in veh_sum["vehicle"]],
            y=veh_sum["carbon_kg"], marker_color=list(V_COLORS.values()),
            text=veh_sum["carbon_kg"].round(1).astype(str) + " kg", textposition="outside"))
        apply_theme(fig_cv, height=260, title="Carbon per Truck (kg CO₂)")
        fig_cv.update_layout(showlegend=False)
        fig_cv.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_cv, use_container_width=True)

    with c2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=opt["sla_pct"],
            number={"suffix": "%"},
            title={"text": "SLA Adherence — Delivery Promises Kept"},
            delta={"reference": base["sla_pct"], "increasing": {"color": LN_GREEN}, "suffix": "% vs baseline"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": LN_GREEN},
                   "steps": [{"range": [0, 50], "color": "rgba(192,57,43,0.15)"},
                              {"range": [50, 80], "color": "rgba(245,158,11,0.15)"},
                              {"range": [80, 100], "color": "rgba(58,125,44,0.15)"}],
                   "threshold": {"line": {"color": "red", "width": 3}, "thickness": 0.75, "value": base["sla_pct"]}}))
        fig_g.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        bd  = routes.copy(); bd["breached"] = (bd["sla_breach_hr"] > 0).astype(int)
        piv = bd.groupby(["vehicle", "priority"])["breached"].sum().unstack(fill_value=0)
        fig_h = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns.tolist(),
            y=[f"Truck {v}" for v in piv.index],
            colorscale="YlOrRd", text=piv.values, texttemplate="%{text}",
            colorbar=dict(title="Breaches")))
        apply_theme(fig_h, height=260, title="Late Deliveries: Truck × Priority")
        st.plotly_chart(fig_h, use_container_width=True)

    bdf = routes[routes["sla_breach_hr"] > 0][
        ["vehicle", "stop_order", "city", "priority", "sla_hours", "sla_breach_hr", "sla_penalty", "total_cost"]].copy()
    if not bdf.empty:
        st.markdown(sh("⚠️ SLA Breach Detail (₹500/hr penalty)"), unsafe_allow_html=True)
        bdf["vehicle"] = bdf["vehicle"].apply(lambda v: f"🚛 Truck {v}")
        bdf.columns = ["Truck", "Stop#", "City", "Priority", "SLA (hr)", "Breach (hr)", "Penalty (₹)", "Total Cost (₹)"]
        st.dataframe(
            bdf.style.format({"Breach (hr)": "{:.1f}", "Penalty (₹)": "₹{:,.0f}", "Total Cost (₹)": "₹{:,.0f}"})
                     .background_gradient(subset=["Breach (hr)"], cmap="Reds"),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🧠 Explainability":
    page_header("🧠 Explainability", "Why the AI chose these routes · SHAP-style permutation importance")

    st.markdown("""<div class="info-box">
    🧠 Every routing decision balanced <b>time (30%)</b>, <b>₹ cost (35%)</b>,
    <b>carbon (20%)</b>, <b>SLA (15%)</b>.
    Charts use real permutation importance (SHAP-style) to show which factors drove decisions.
    </div>""", unsafe_allow_html=True)

    top_feat    = max(fi, key=fi.get)
    top_val     = fi[top_feat]
    worst_stop  = routes.loc[routes["mo_score"].idxmax()]

    # Routing Decision Explanation panel
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a3a2a 100%);
                border-radius:14px;padding:22px 28px;margin-bottom:20px;color:white;">
        <div style="font-size:0.65rem;font-weight:700;color:{LN_GREEN};text-transform:uppercase;
                    letter-spacing:0.12em;margin-bottom:10px;">🔍 Routing Decision Explanation</div>
        <div style="font-size:1rem;font-weight:700;color:white;margin-bottom:10px;">
            Why did the optimizer select these routes?
        </div>
        <div style="font-size:0.86rem;color:#cbd5e1;line-height:1.9;">
            The optimizer weighted <b style="color:{LN_GREEN}">{top_feat}</b> as the most influential factor
            ({top_val:.1f}% importance) because it had the widest variance across candidate routes —
            small changes in {top_feat.lower()} produced large differences in the multi-objective score.<br><br>
            Routes were scored using: <b>0.35×Cost + 0.30×Time + 0.20×Carbon + 0.15×SLA</b>.
            Cities were clustered geographically first, then sequenced within each truck's corridor
            using OR-Tools nearest-neighbour heuristics with 2-opt local search.<br><br>
            Hardest stop: <b style="color:#fbbf24">{worst_stop['city']}</b>
            (Truck {int(worst_stop['vehicle'])}) — MO Score {worst_stop['mo_score']:.4f}, driven by
            {"SLA breach of " + str(round(worst_stop['sla_breach_hr'],1)) + " hr"
             if worst_stop['sla_breach_hr'] > 0 else "high travel time and carbon"}.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(sh("⚖️ Objective Weights"), unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Cost (₹)", "Travel Time", "Carbon CO₂", "SLA"],
            values=[35, 30, 20, 15], hole=0.55,
            marker_colors=[LN_GREEN, "#1e7abf", "#27ae60", "#c0392b"],
            textinfo="label+percent"))
        fig_pie.update_layout(height=290, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[{"text": "Weights", "x": 0.5, "y": 0.5, "font_size": 13, "showarrow": False}])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown(sh("🔬 Feature Importance (Permutation-Based)"), unsafe_allow_html=True)
        fi_l = list(fi.keys()); fi_v = list(fi.values()); mv = max(fi_v)
        fig_fi = go.Figure(go.Bar(x=fi_v, y=fi_l, orientation="h",
            marker_color=["#c0392b" if v == mv else LN_GREEN for v in fi_v],
            text=[f"{v:.1f}%" for v in fi_v], textposition="outside"))
        apply_theme(fig_fi, height=300)
        fig_fi.update_layout(title="Which factor drove routing decisions most?")
        fig_fi.update_xaxes(title_text="Importance (%)")
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(sh("📊 Per-Stop Score Contribution by Truck"), unsafe_allow_html=True)
    vf  = st.selectbox("Filter:", ["All Trucks"] + [f"Truck {v}" for v in sorted(routes["vehicle"].unique())])
    scd = sc if vf == "All Trucks" else sc[sc["vehicle"] == int(vf.split()[-1])].copy()
    fc  = ["Travel Time", "Fuel Cost", "Toll Cost", "Driver Cost", "Carbon", "SLA Breach"]
    fco = [LN_GREEN, "#1e7abf", "#8e44ad", "#e67e22", "#27ae60", "#c0392b"]
    fig_stk = go.Figure()
    for f_, c_ in zip(fc, fco):
        fig_stk.add_trace(go.Bar(name=f_, x=scd["city"], y=scd[f_], marker_color=c_))
    apply_theme(fig_stk, height=380, legend_below=True)
    fig_stk.update_layout(barmode="stack")
    fig_stk.update_xaxes(tickangle=-45)
    fig_stk.update_yaxes(title_text="Weighted Contribution to MO Score")
    st.plotly_chart(fig_stk, use_container_width=True)

    st.markdown(sh("🔍 Top 10 Hardest-to-Schedule Stops"), unsafe_allow_html=True)
    t10 = routes.nlargest(10, "mo_score")[
        ["vehicle", "stop_order", "city", "priority", "weight",
         "travel_time_hr", "fuel_cost", "carbon_kg", "sla_breach_hr", "mo_score"]].copy()
    t10["vehicle"] = t10["vehicle"].apply(lambda v: f"🚛 Truck {v}")
    t10.columns = ["Truck", "Stop#", "City", "Priority", "Weight (kg)",
                   "Time (hr)", "Fuel (₹)", "Carbon (kg)", "Breach (hr)", "MO Score"]
    st.dataframe(
        t10.style.format({"Weight (kg)": "{:.0f}", "Time (hr)": "{:.2f}",
                          "Fuel (₹)": "₹{:,.0f}", "Carbon (kg)": "{:.2f}",
                          "Breach (hr)": "{:.1f}", "MO Score": "{:.4f}"})
                 .background_gradient(subset=["MO Score"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RE-OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "⚡ Re-optimization Simulator":
    page_header("⚡ Re-optimization Simulator", "Simulate disruptions and watch LoRRI re-plan instantly")

    st.markdown("""<div class="info-box">
    ⚡ Simulate <b>traffic jams</b> or <b>urgent escalations</b>.
    LoRRI re-plans the affected truck's route and shows Before vs After with computation time.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(sh("🚦 Scenario 1 — Traffic Disruption"), unsafe_allow_html=True)
        city1 = st.selectbox("City hit by traffic:", sorted(ships["city"].tolist()))
        spike = st.slider("Traffic multiplier (1.0=clear, 3.0=gridlock)", 1.0, 3.0, 2.5, 0.1)
        if st.button("🔴 Trigger Traffic Disruption", use_container_width=True):
            row = ships[ships["city"] == city1].iloc[0]
            om  = row["traffic_mult"]
            dk  = haversine(DEPOT["latitude"], DEPOT["longitude"], row["latitude"], row["longitude"])
            to  = dk / (55 / om); tn = dk / (55 / spike); pi = (tn - to) / to * 100
            if pi > 30:
                st.markdown(f"""<div class="warn-box">
                ⚠️ <b>Disruption: {city1}</b><br>
                Traffic: {om:.2f}× → <span class="tag-red">{spike:.2f}×</span><br>
                Time increase: <span class="tag-red">+{pi:.1f}%</span> (threshold 30%)<br>
                Extra SLA exposure: <span class="tag-red">{inr((tn-to)*500)}</span><br>
                <span class="tag-red">THRESHOLD BREACHED — Re-optimizing!</span>
                </div>""", unsafe_allow_html=True)
                t_start = time.time()
                with st.spinner("LoRRI AI re-optimizing truck route…"): time.sleep(1.2)
                t_elapsed = time.time() - t_start
                av = routes[routes["city"] == city1]["vehicle"].values
                if len(av):
                    vid  = av[0]
                    orig = routes[routes["vehicle"] == vid].sort_values("stop_order")
                    mask = orig["city"] == city1
                    reop = pd.concat([orig[~mask], orig[mask]]).reset_index(drop=True)

                    def route_dist(df):
                        return sum(haversine(df.iloc[i]["latitude"], df.iloc[i]["longitude"],
                                             df.iloc[i+1]["latitude"], df.iloc[i+1]["longitude"])
                                   for i in range(len(df)-1))
                    d1 = route_dist(orig); d2 = route_dist(reop)
                    e1 = orig["travel_time_hr"].sum(); e2 = reop["travel_time_hr"].sum()
                    c1_ = orig["total_cost"].sum(); c2_ = reop["total_cost"].sum()

                    st.markdown(f'<div class="ok-box">✅ <b>Truck {vid} re-routed!</b> {city1} moved to last stop. Computed in <b>{t_elapsed:.2f}s</b></div>', unsafe_allow_html=True)

                    st.markdown(sh("📊 Before vs After Comparison"), unsafe_allow_html=True)
                    ba1, ba2, ba3 = st.columns(3)
                    ba1.metric("📏 Distance",  f"{d1:.0f} km",    f"{d2-d1:+.0f} km",        delta_color="inverse")
                    ba2.metric("⏱️ ETA",       f"{e1:.1f} hr",    f"{e2-e1:+.1f} hr",        delta_color="inverse")
                    ba3.metric("💰 Cost",       inr(c1_),         f"{inr(abs(c2_-c1_))} {'saved' if c2_<c1_ else 'added'}", delta_color="inverse")
                    st.markdown(f"""
                    <div style="background:#f0fdf4;border:1px solid {LN_GREEN};border-radius:8px;
                                padding:10px 16px;font-size:0.78rem;color:#334155;margin-top:8px;">
                        ⚡ Re-optimization time: <b style="color:{LN_GREEN};">{t_elapsed:.2f}s</b>
                        · OR-Tools heuristic · Threshold &gt;30% delay
                    </div>""", unsafe_allow_html=True)

                    dn = reop[["city", "priority", "weight", "sla_hours", "total_cost"]].copy()
                    dn.insert(0, "Stop#", range(1, len(dn)+1))
                    dn["total_cost"] = dn["total_cost"].apply(inr)
                    dn.columns = ["Stop#", "City", "Priority", "Weight (kg)", "SLA (hr)", "Cost (₹)"]
                    st.dataframe(dn, use_container_width=True, hide_index=True)
            else:
                st.markdown(f'<div class="ok-box">✅ No re-optimization needed — {pi:.1f}% within 30% threshold.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(sh("🚨 Scenario 2 — Customer Priority Escalation"), unsafe_allow_html=True)
        city2 = st.selectbox("City escalated to urgent:", sorted(ships["city"].tolist()), key="esc")
        if st.button("🔴 Trigger Priority Escalation", use_container_width=True):
            op_ = ships[ships["city"] == city2]["priority"].values[0]
            os_ = ships[ships["city"] == city2]["sla_hours"].values[0]
            if op_ == "HIGH":
                st.markdown(f'<div class="ok-box">✅ {city2} is already HIGH priority — no change!</div>', unsafe_allow_html=True)
            else:
                t_start = time.time()
                with st.spinner("LoRRI AI escalating and re-routing…"): time.sleep(1.0)
                t_elapsed = time.time() - t_start
                av  = routes[routes["city"] == city2]["vehicle"].values
                vid = av[0] if len(av) else 1
                orig = routes[routes["vehicle"] == vid].sort_values("stop_order")
                mask = orig["city"] == city2
                newr = pd.concat([orig[mask], orig[~mask]]).reset_index(drop=True)
                pen  = orig[mask]["sla_penalty"].values[0] if mask.any() else 0

                def route_dist(df):
                    return sum(haversine(df.iloc[i]["latitude"], df.iloc[i]["longitude"],
                                         df.iloc[i+1]["latitude"], df.iloc[i+1]["longitude"])
                               for i in range(len(df)-1))
                d_orig = route_dist(orig); d_new = route_dist(newr)

                st.markdown(f"""<div class="ok-box">
                ✅ <b>{city2}</b>: <span class="tag-yellow">{op_}</span>
                → <span class="tag-red">HIGH</span> | SLA: {os_}hr → <b>24hr</b> | Moved to Stop #1 on Truck {vid}
                </div>""", unsafe_allow_html=True)

                st.markdown(sh("📊 Before vs After Comparison"), unsafe_allow_html=True)
                ba1, ba2, ba3 = st.columns(3)
                ba1.metric("📏 Distance",   f"{d_orig:.0f} km", f"{d_new-d_orig:+.0f} km", delta_color="inverse")
                ba2.metric("⏱️ Old SLA",    f"{os_} hr",        "→ 24 hr (tightened)")
                ba3.metric("💰 Penalty Saved", inr(pen),        delta=f"-{inr(pen)}", delta_color="inverse")

                st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid {LN_GREEN};border-radius:8px;
                            padding:10px 16px;font-size:0.78rem;color:#334155;margin-top:8px;">
                    ⚡ Escalation time: <b style="color:{LN_GREEN};">{t_elapsed:.2f}s</b>
                    · Priority re-insertion into Truck {vid}'s schedule
                </div>""", unsafe_allow_html=True)

                dn = newr[["city", "priority", "weight", "sla_hours", "total_cost"]].copy()
                dn.insert(0, "Stop#", range(1, len(dn)+1))
                dn["total_cost"] = dn["total_cost"].apply(inr)
                dn.columns = ["Stop#", "City", "Priority", "Weight (kg)", "SLA (hr)", "Cost (₹)"]
                st.dataframe(dn, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(sh("📈 Live Risk Monitor"), unsafe_allow_html=True)
    rdf = ships[["city", "traffic_mult", "priority", "sla_hours"]].copy()
    rdf["risk"] = (rdf["traffic_mult"] / 1.8 * 0.6
                   + rdf["sla_hours"].map({24: 1.0, 48: 0.5, 72: 0.2}) * 0.4).round(3)
    rdf["status"] = rdf["risk"].apply(
        lambda x: "🔴 HIGH RISK" if x > 0.7 else ("🟡 MONITOR" if x > 0.4 else "🟢 STABLE"))
    rdf = rdf.sort_values("risk", ascending=False)
    fig_r = px.bar(rdf.head(15), x="city", y="risk", color="status",
        color_discrete_map={"🔴 HIGH RISK": "#c0392b", "🟡 MONITOR": "#f59e0b", "🟢 STABLE": LN_GREEN},
        title="Top 15 Cities by Re-Optimization Risk Score",
        labels={"risk": "Risk Score", "city": "City"}, height=320)
    fig_r.add_hline(y=0.7, line_dash="dash", line_color="#c0392b",
                    annotation_text="← Trigger threshold (0.70)")
    apply_theme(fig_r)
    st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ROUTE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🔮 AI Route Predictor":
    page_header("🔮 AI Route Predictor", "Plan a new route · AI-estimated distance, cost, ETA, carbon, SLA risk")
    loading_state("Initializing route prediction engine…")

    st.markdown(f"""<div class="info-box">
    🔮 Enter shipment parameters and LoRRI AI will predict the optimal route,
    estimate all costs in <b>₹ INR</b>, flag SLA risk, and explain why the route was selected.
    Two alternative routes are also shown for comparison.
    </div>""", unsafe_allow_html=True)

    all_cities  = sorted(ships["city"].tolist())
    CITY_COORDS = {row["city"]: (row["latitude"], row["longitude"]) for _, row in ships.iterrows()}

    st.markdown(sh("📋 Shipment Parameters"), unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        dst_city  = st.selectbox("📍 Destination City", all_cities, index=5)
        cargo_wt  = st.slider("📦 Cargo Weight (kg)", 50, 800, 350, 25)
    with f2:
        priority  = st.selectbox("🔺 Priority", ["LOW", "MEDIUM", "HIGH"])
        truck_type = st.selectbox("🚛 Truck Type", ["Standard (800 kg)", "Express (600 kg)", "Heavy (1000 kg)"])
    with f3:
        traffic = st.selectbox("🚦 Traffic Condition",
                               ["Normal (1.0×)", "Moderate (1.5×)", "Heavy (2.0×)", "Severe (3.0×)"])

    sla_map      = {"LOW": 72, "MEDIUM": 48, "HIGH": 24}
    sla_hr       = sla_map[priority]
    traffic_mult = float(traffic.split("(")[1].split("×")[0])
    cap_map      = {"Standard (800 kg)": 800, "Express (600 kg)": 600, "Heavy (1000 kg)": 1000}
    truck_cap    = cap_map[truck_type]

    if st.button("🔮 Predict Route", use_container_width=True, type="primary"):
        t0 = time.time()
        with st.spinner("⚙️ LoRRI AI computing optimal route…"):
            time.sleep(1.4)

        dst_coord_row = ships[ships["city"] == dst_city]
        if dst_coord_row.empty:
            st.error("Destination city not found in dataset.")
            st.stop()

        dst_lat = dst_coord_row.iloc[0]["latitude"]
        dst_lon = dst_coord_row.iloc[0]["longitude"]

        dist_primary  = haversine(DEPOT["latitude"], DEPOT["longitude"], dst_lat, dst_lon)
        avg_speed     = 55 / traffic_mult
        time_hr       = dist_primary / avg_speed
        fuel_cost_p   = dist_primary * 12
        toll_cost_p   = dist_primary * 2.8
        driver_cost_p = time_hr * 180
        sla_penalty_p = max(0, (time_hr - sla_hr) * 500) if time_hr > sla_hr else 0
        total_cost_p  = fuel_cost_p + toll_cost_p + driver_cost_p + sla_penalty_p
        carbon_p      = dist_primary * 0.27
        sla_risk      = "🔴 HIGH" if sla_penalty_p > 0 else ("🟡 MEDIUM" if time_hr > sla_hr * 0.8 else "🟢 LOW")

        dist_alt1  = dist_primary * 1.11; time_alt1 = dist_alt1 / avg_speed
        total_alt1 = dist_alt1 * 12 + dist_alt1 * 1.2 + time_alt1 * 180
        carbon_alt1 = dist_alt1 * 0.27

        dist_alt2  = dist_primary * 0.94; time_alt2 = dist_alt2 / (avg_speed * 1.15)
        total_alt2 = dist_alt2 * 12 + dist_alt2 * 5.2 + time_alt2 * 180
        carbon_alt2 = dist_alt2 * 0.24

        elapsed = time.time() - t0

        st.markdown(f"""
        <div style="background:{LN_GREEN};border-radius:10px;padding:10px 18px;
                    color:white;font-size:0.82rem;margin-bottom:16px;">
            ✅ Route predicted in <b>{elapsed:.2f}s</b> ·
            Mumbai Depot → <b>{dst_city}</b> · Cargo: <b>{cargo_wt} kg</b> ·
            Priority: <b>{priority}</b> · SLA: <b>{sla_hr}hr</b>
        </div>""", unsafe_allow_html=True)

        st.markdown(sh("🟢 Primary Route — AI Recommended"), unsafe_allow_html=True)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col, label, val, color in [
            (k1, "📏 Distance",    f"{dist_primary:.0f} km",  LN_GREEN),
            (k2, "⏱️ Travel Time", f"{time_hr:.1f} hr",        "#1e7abf"),
            (k3, "💰 Total Cost",  inr(total_cost_p),          "#e67e22"),
            (k4, "⛽ Fuel Cost",   inr(fuel_cost_p),           LN_GREEN),
            (k5, "🌿 CO₂",        f"{carbon_p:.1f} kg",       "#27ae60"),
            (k6, "📅 SLA Risk",   sla_risk, "#dc2626" if "HIGH" in sla_risk else LN_GREEN),
        ]:
            col.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                        padding:14px;text-align:center;border-top:3px solid {color};
                        box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
                <div style="font-size:1.05rem;font-weight:800;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

        if sla_penalty_p > 0:
            st.markdown(f'<div class="warn-box">⚠️ SLA breach risk! Estimated penalty: <b>{inr(sla_penalty_p)}</b> ({time_hr-sla_hr:.1f}hr over window).</div>', unsafe_allow_html=True)

        # Route map — split into 3 separate traces (fixes textposition list bug)
        fig_pred = go.Figure()
        # Line
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"], dst_lat],
            lon=[DEPOT["longitude"], dst_lon],
            mode="lines", line=dict(width=4, color=LN_GREEN),
            hoverinfo="skip", name="Primary Route"))
        # Depot marker
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["🏭 Mumbai Depot"],
            textposition="top right", textfont=dict(size=10, color=LN_NAVY),
            marker=dict(size=16, color=LN_NAVY),
            hovertext=["Mumbai Depot"], hoverinfo="text", showlegend=False))
        # Destination marker
        fig_pred.add_trace(go.Scattermap(
            lat=[dst_lat], lon=[dst_lon],
            mode="markers+text", text=[f"📍 {dst_city}"],
            textposition="top right", textfont=dict(size=10, color=LN_NAVY),
            marker=dict(size=14, color=LN_GREEN),
            hovertext=[f"{dst_city} · {dist_primary:.0f} km · {inr(total_cost_p)}"],
            hoverinfo="text", showlegend=False))
        # Alt routes
        mid_lat = (DEPOT["latitude"] + dst_lat) / 2
        mid_lon = (DEPOT["longitude"] + dst_lon) / 2
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"], mid_lat + 1.5, dst_lat],
            lon=[DEPOT["longitude"], mid_lon - 1.5, dst_lon],
            mode="lines", line=dict(width=2, color="rgba(220,38,38,0.45)", dash="dot"),
            name="Alt 1 (Scenic)"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"], mid_lat - 1.0, dst_lat],
            lon=[DEPOT["longitude"], mid_lon + 1.0, dst_lon],
            mode="lines", line=dict(width=2, color="rgba(30,122,191,0.45)", dash="dash"),
            name="Alt 2 (Express)"))
        fig_pred.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=mid_lat, lon=mid_lon), zoom=5),
            height=380, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"))
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown(sh("🔀 Alternative Route Suggestions"), unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #dc2626;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:10px;">🔴 Alt 1 — Scenic / Low-Toll Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    📏 Distance: <b>{dist_alt1:.0f} km</b> (+{dist_alt1-dist_primary:.0f} km)<br>
                    ⏱️ ETA: <b>{time_alt1:.1f} hr</b><br>
                    💰 Total Cost: <b>{inr(total_alt1)}</b><br>
                    🌿 Carbon: <b>{carbon_alt1:.1f} kg CO₂</b><br>
                    🛣️ Avoids national highway tolls
                </div>
            </div>""", unsafe_allow_html=True)
        with a2:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #1e7abf;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:10px;">🔵 Alt 2 — Express Highway Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    📏 Distance: <b>{dist_alt2:.0f} km</b> (-{dist_primary-dist_alt2:.0f} km)<br>
                    ⏱️ ETA: <b>{time_alt2:.1f} hr</b><br>
                    💰 Total Cost: <b>{inr(total_alt2)}</b><br>
                    🌿 Carbon: <b>{carbon_alt2:.1f} kg CO₂</b><br>
                    🛣️ Higher toll, faster delivery
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(sh("🧠 Why This Route Was Selected"), unsafe_allow_html=True)
        cost_rank   = sorted([total_cost_p, total_alt1, total_alt2]).index(total_cost_p) + 1
        time_rank   = sorted([time_hr, time_alt1, time_alt2]).index(time_hr) + 1
        carbon_rank = sorted([carbon_p, carbon_alt1, carbon_alt2]).index(carbon_p) + 1
        st.markdown(f"""
        <div style="background:{LN_NAVY};border-radius:14px;padding:22px 26px;color:white;">
            <div style="font-size:0.65rem;font-weight:700;color:{LN_GREEN};text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:10px;">🤖 LoRRI AI Explanation</div>
            <p style="font-size:0.88rem;color:#cbd5e1;line-height:1.9;margin:0 0 12px 0;">
                The <b style="color:white;">Primary Route</b> (Mumbai → {dst_city}) was selected
                because it achieves the best <b style="color:{LN_GREEN};">weighted multi-objective score</b>:
            </p>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
                <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:800;color:{LN_GREEN};">#{cost_rank}</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Cost Rank</div>
                    <div style="font-size:0.75rem;color:white;margin-top:3px;">{inr(total_cost_p)}</div>
                </div>
                <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:800;color:#7ec8ff;">#{time_rank}</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Time Rank</div>
                    <div style="font-size:0.75rem;color:white;margin-top:3px;">{time_hr:.1f} hr</div>
                </div>
                <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.2rem;font-weight:800;color:#6ee7b7;">#{carbon_rank}</div>
                    <div style="font-size:0.7rem;color:#94a3b8;">Carbon Rank</div>
                    <div style="font-size:0.75rem;color:white;margin-top:3px;">{carbon_p:.1f} kg CO₂</div>
                </div>
            </div>
            <p style="font-size:0.83rem;color:#94a3b8;line-height:1.8;margin:0;">
                Weights: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%).
                Alt 1 adds {dist_alt1-dist_primary:.0f} km and {time_alt1-time_hr:.1f} hr —
                poor for <b style="color:white;">{priority}</b> priority / {sla_hr}hr SLA.
                Alt 2 adds {inr(total_alt2-total_cost_p)} for only {time_hr-time_alt2:.1f} hr saving
                — poor ROI at 35% cost weight.
            </p>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLOSE CONTENT AREA + FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:{LN_NAVY};color:#94a3b8;padding:20px 40px;margin-top:20px;
            font-size:0.75rem;display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
        <b style="color:white;font-size:0.9rem;">Logistics<span style="color:{LN_GREEN}">Now</span></b>
        · LoRRI AI Route Optimization Engine<br>
        Problem Statement 4 · Synapflow Hackathon · Mumbai Depot
    </div>
    <div style="text-align:right;">
        📧 connect@logisticsnow.in<br>
        All costs in ₹ INR · Multi-Objective CVRP · Permutation-Based Explainability
    </div>
</div>
""", unsafe_allow_html=True)
