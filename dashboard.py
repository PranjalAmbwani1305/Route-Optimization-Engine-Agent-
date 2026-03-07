"""
LoRRI – LogisticsNow AI Route Optimization Engine  v3.0 (Combined)
====================================================================
Merged from dashboard.py + dashboard_2.py
Pages: About, AI Assistant, Fleet Intelligence, Dashboard,
       Route Map (Toll Plazas), Financial, Carbon & SLA,
       Explainability, Re-optimization Simulator, AI Route Predictor.
Fixes applied:
  - Truck names from generate_data2.py in Route Map
  - Heatmap uses correct data source
  - Shipments by Truck Category error resolved
  - Live Risk Monitor box overlap removed
  - ₹ symbol used consistently
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
    menu_items={"About": "LoRRI AI Route Optimization Engine · LogisticsNow · Problem Statement #4"}
)

# ─────────────────────────────────────────────────────────────────────────────
# BRAND COLORS
# ─────────────────────────────────────────────────────────────────────────────
LN_GREEN  = "#3a7d2c"
LN_DGREEN = "#2d6a2d"
LN_NAVY   = "#1e2d3d"
LN_LGRAY  = "#f5f6f7"
LN_BORDER = "#e0e4e8"

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777, "city": "Mumbai"}
VEHICLE_CAP = 800
V_COLORS    = {1: "#3a7d2c", 2: "#1e7abf", 3: "#e67e22", 4: "#8e44ad", 5: "#c0392b"}

# ─────────────────────────────────────────────────────────────────────────────
# TRUCK TYPES — from generate_data2.py (LoRRI / Indian spec)
# ─────────────────────────────────────────────────────────────────────────────
TRUCK_TYPES_INFO = {
    "Tata Ace (Mini)":         {"category":"LCV","cap_ton":0.75,"cap_cbm":4,  "cost_km":12,"fixed":800, "kmpl":18,"co2_km":0.18,"toll_mult":0.5,  "color":"#27ae60","icon":"🛺"},
    "Tata 407 (SCV)":          {"category":"SCV","cap_ton":2.5, "cap_cbm":16, "cost_km":18,"fixed":1200,"kmpl":14,"co2_km":0.28,"toll_mult":0.75, "color":"#2ecc71","icon":"🚐"},
    "Eicher 10T (ICV)":        {"category":"ICV","cap_ton":10,  "cap_cbm":40, "cost_km":28,"fixed":2500,"kmpl":10,"co2_km":0.55,"toll_mult":1.0,  "color":"#1e7abf","icon":"🚛"},
    "Ashok Leyland 14T (MCV)": {"category":"MCV","cap_ton":14,  "cap_cbm":55, "cost_km":35,"fixed":3200,"kmpl":8, "co2_km":0.72,"toll_mult":1.25, "color":"#e67e22","icon":"🚚"},
    "Tata 1109 20T (HCV)":     {"category":"HCV","cap_ton":20,  "cap_cbm":72, "cost_km":45,"fixed":4500,"kmpl":6, "co2_km":0.95,"toll_mult":1.5,  "color":"#8e44ad","icon":"🚛"},
    "Volvo 32T (HXL)":         {"category":"HXL","cap_ton":32,  "cap_cbm":105,"cost_km":65,"fixed":7000,"kmpl":4.5,"co2_km":1.40,"toll_mult":2.0, "color":"#c0392b","icon":"🚛"},
    "Trailer 40T (MXL)":       {"category":"MXL","cap_ton":40,  "cap_cbm":130,"cost_km":80,"fixed":9000,"kmpl":3.5,"co2_km":1.80,"toll_mult":2.5, "color":"#2c3e50","icon":"🚛"},
}

# Vehicle slot → truck name mapping (matches generate_data2.py assignment order)
TRUCK_SLOT_NAMES = {
    1: "Tata Ace (Mini)",
    2: "Tata 407 (SCV)",
    3: "Eicher 10T (ICV)",
    4: "Ashok Leyland 14T (MCV)",
    5: "Tata 1109 20T (HCV)",
}

def truck_name(v):
    """Return real Indian truck name for vehicle slot v."""
    return TRUCK_SLOT_NAMES.get(int(v), f"Truck {int(v)}")

def truck_short(v):
    """Short label: 'Tata Ace' etc."""
    full = truck_name(v)
    # Return name without parenthetical category
    return full.split(" (")[0] if " (" in full else full

CAT_COLORS = {"LCV":"#27ae60","SCV":"#2ecc71","ICV":"#1e7abf","MCV":"#e67e22","HCV":"#8e44ad","HXL":"#c0392b","MXL":"#2c3e50"}

# ─────────────────────────────────────────────────────────────────────────────
# TOLL PLAZAS — major Indian NH corridors
# ─────────────────────────────────────────────────────────────────────────────
TOLL_PLAZAS = [
    {"name":"Khalapur Toll",     "lat":18.820,"lon":73.250,"highway":"NH48","toll_inr":165,"corridor":"Mumbai-Pune"},
    {"name":"Talegaon Toll",     "lat":18.700,"lon":73.680,"highway":"NH48","toll_inr":120,"corridor":"Mumbai-Pune"},
    {"name":"Khopoli Toll",      "lat":18.780,"lon":73.100,"highway":"NH48","toll_inr":110,"corridor":"Mumbai-Pune"},
    {"name":"Igatpuri Toll",     "lat":19.710,"lon":73.560,"highway":"NH3", "toll_inr":95, "corridor":"Mumbai-Nashik"},
    {"name":"Ghoti Toll",        "lat":19.870,"lon":73.640,"highway":"NH3", "toll_inr":85, "corridor":"Mumbai-Nashik"},
    {"name":"Kim Toll",          "lat":21.420,"lon":72.870,"highway":"NH48","toll_inr":140,"corridor":"Mumbai-Surat"},
    {"name":"Sachin Toll",       "lat":21.090,"lon":72.880,"highway":"NH48","toll_inr":130,"corridor":"Mumbai-Surat"},
    {"name":"Vapi Toll",         "lat":20.370,"lon":72.910,"highway":"NH48","toll_inr":145,"corridor":"Mumbai-Ahmedabad"},
    {"name":"Ankleshwar Toll",   "lat":21.620,"lon":73.000,"highway":"NH48","toll_inr":155,"corridor":"Mumbai-Ahmedabad"},
    {"name":"Vadodara Toll",     "lat":22.290,"lon":73.090,"highway":"NH48","toll_inr":160,"corridor":"Mumbai-Ahmedabad"},
    {"name":"Rajkot Toll",       "lat":22.700,"lon":70.950,"highway":"NH48","toll_inr":135,"corridor":"Ahmedabad-Rajkot"},
    {"name":"Khed Toll",         "lat":17.720,"lon":73.940,"highway":"NH66","toll_inr":90, "corridor":"Mumbai-Goa"},
    {"name":"Delhi-Gurgaon Toll","lat":28.430,"lon":77.060,"highway":"NH48","toll_inr":135,"corridor":"Delhi-Jaipur"},
    {"name":"Manesar Toll",      "lat":28.360,"lon":76.930,"highway":"NH48","toll_inr":125,"corridor":"Delhi-Jaipur"},
    {"name":"Shahjahanpur Toll", "lat":27.420,"lon":76.480,"highway":"NH48","toll_inr":145,"corridor":"Delhi-Jaipur"},
    {"name":"Behror Toll",       "lat":27.880,"lon":76.280,"highway":"NH48","toll_inr":115,"corridor":"Delhi-Jaipur"},
    {"name":"Faridabad Toll",    "lat":28.380,"lon":77.310,"highway":"NH19","toll_inr":120,"corridor":"Delhi-Agra"},
    {"name":"Palwal Toll",       "lat":28.140,"lon":77.330,"highway":"NH19","toll_inr":110,"corridor":"Delhi-Agra"},
    {"name":"Mathura Toll",      "lat":27.500,"lon":77.630,"highway":"NH19","toll_inr":130,"corridor":"Delhi-Agra"},
    {"name":"Ambala Toll",       "lat":30.380,"lon":76.780,"highway":"NH44","toll_inr":115,"corridor":"Delhi-Chandigarh"},
    {"name":"Karnal Toll",       "lat":29.690,"lon":76.990,"highway":"NH44","toll_inr":105,"corridor":"Delhi-Chandigarh"},
    {"name":"Hosur Toll",        "lat":12.730,"lon":77.820,"highway":"NH44","toll_inr":85, "corridor":"Bengaluru-Chennai"},
    {"name":"Krishnagiri Toll",  "lat":12.520,"lon":78.210,"highway":"NH44","toll_inr":95, "corridor":"Bengaluru-Chennai"},
    {"name":"Vellore Toll",      "lat":12.920,"lon":79.140,"highway":"NH44","toll_inr":90, "corridor":"Bengaluru-Chennai"},
    {"name":"Tumkur Toll",       "lat":13.340,"lon":77.100,"highway":"NH44","toll_inr":80, "corridor":"Bengaluru-Hyderabad"},
    {"name":"Kurnool Toll",      "lat":15.830,"lon":78.050,"highway":"NH44","toll_inr":125,"corridor":"Bengaluru-Hyderabad"},
    {"name":"Zahirabad Toll",    "lat":17.680,"lon":77.600,"highway":"NH44","toll_inr":95, "corridor":"Bengaluru-Hyderabad"},
    {"name":"Nagpur Toll",       "lat":21.050,"lon":79.050,"highway":"NH53","toll_inr":110,"corridor":"Nagpur-Raipur"},
    {"name":"Wardha Toll",       "lat":20.750,"lon":78.600,"highway":"NH7", "toll_inr":95, "corridor":"Nagpur-Hyderabad"},
    {"name":"Kolkata Bypass",    "lat":22.600,"lon":88.430,"highway":"NH19","toll_inr":100,"corridor":"Kolkata-Patna"},
    {"name":"Durgapur Toll",     "lat":23.520,"lon":87.320,"highway":"NH19","toll_inr":115,"corridor":"Kolkata-Patna"},
    {"name":"Asansol Toll",      "lat":23.680,"lon":86.980,"highway":"NH19","toll_inr":105,"corridor":"Kolkata-Patna"},
]

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

def logo_html(height=48, center=False):
    sty = f"height:{height}px;width:auto;object-fit:contain;display:block;" + ("margin:0 auto;" if center else "")
    if LOGO_B64:
        return f'<img src="data:image/png;base64,{LOGO_B64}" style="{sty}" />'
    fs = int(height * 0.55)
    return f'<span style="font-size:{fs}px;font-weight:900;color:{LN_NAVY};letter-spacing:-1px;font-family:Poppins,sans-serif;">Lo<span style="color:{LN_GREEN}">RRI</span></span>'

def logo_html_dark(height=48, center=False):
    sty = f"height:{height}px;width:auto;object-fit:contain;display:block;" + ("margin:0 auto;" if center else "")
    if LOGO_B64:
        return (f'<div style="display:inline-block;background:rgba(255,255,255,0.93);'
                f'border-radius:10px;padding:6px 12px;">'
                f'<img src="data:image/png;base64,{LOGO_B64}" style="{sty}" /></div>')
    fs = int(height * 0.55)
    return f'<span style="font-size:{fs}px;font-weight:900;color:white;letter-spacing:-1px;font-family:Poppins,sans-serif;">Lo<span style="color:#6bcf57">RRI</span></span>'

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

def inr(val):
    return f"₹{val:,.0f}"

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
# HIDE DEFAULT STREAMLIT CHROME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
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

/* ── KPI Cards ── */
.kpi-card {{
    background: white;
    border: 1px solid {LN_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 6px 20px rgba(58,125,44,0.14); }}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--ac, {LN_GREEN});
}}
.kpi-lbl {{
    font-size: 0.6rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}}
.kpi-val {{
    font-size: 1.65rem;
    font-weight: 700;
    color: {LN_NAVY};
    line-height: 1.1;
}}
.kpi-d {{ font-size: 0.7rem; margin-top: 5px; font-weight: 500; }}
.dg {{ color: {LN_GREEN}; }}
.dr {{ color: #dc2626; }}

/* ── Section headers ── */
.sh {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {LN_NAVY};
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-left: 4px solid {LN_GREEN};
    padding-left: 12px;
}}

/* ── Info / Warn / OK boxes ── */
.info-box {{
    background: #f0fdf4;
    border-left: 4px solid {LN_GREEN};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 4px 0 16px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: {LN_NAVY};
}}
.info-box b {{ color: {LN_DGREEN}; }}
.warn-box {{
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.86rem;
    line-height: 1.65;
    color: {LN_NAVY};
}}
.ok-box {{
    background: #f0fdf4;
    border-left: 4px solid {LN_GREEN};
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.86rem;
    line-height: 1.65;
    color: {LN_NAVY};
}}
.demo-banner {{
    background: linear-gradient(90deg, #1e40af 0%, #1e2d3d 100%);
    border-radius: 8px;
    padding: 10px 18px;
    margin-bottom: 18px;
    font-size: 0.82rem;
    color: #bfdbfe;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.demo-banner b {{ color: white; }}

/* ── Tags ── */
.tag-red    {{ color: #dc2626; font-weight: 600; }}
.tag-green  {{ color: {LN_GREEN}; font-weight: 600; }}
.tag-yellow {{ color: #d97706; font-weight: 600; }}

/* ── Map legend ── */
.legend-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 0.84rem;
    color: {LN_NAVY};
    margin-bottom: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid {LN_BORDER};
}}
.legend-row:last-child {{ border-bottom: none; margin-bottom: 0; }}
.legend-dot {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    flex-shrink: 0;
    margin-top: 3px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: white !important;
    border-right: 2px solid {LN_BORDER} !important;
}}
/* Force logo area to always be on white — never on dark */
[data-testid="stSidebar"] img {{
    background: white !important;
    border-radius: 8px;
    padding: 4px;
    mix-blend-mode: multiply;
}}
/* Hero banner logo — screen blend makes white areas go transparent on dark bg */
.hero-logo img {{
    mix-blend-mode: screen;
    filter: brightness(1.1);
}}
.sb-sec {{
    font-size: 0.6rem;
    font-weight: 700;
    color: {LN_GREEN};
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.4rem;
    border-bottom: 1px solid {LN_BORDER};
    padding-bottom: 4px;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: white;
    border: 1px solid {LN_BORDER};
    border-radius: 10px;
    padding: 12px 16px !important;
}}

/* ── Live pulse dot ── */
@keyframes pulse {{
    0%   {{ opacity: 1; }}
    50%  {{ opacity: 0.3; }}
    100% {{ opacity: 1; }}
}}
.live-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {LN_GREEN};
    animation: pulse 1.8s infinite;
    margin-right: 5px;
}}

/* ── Stats rows in sidebar ── */
.stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.74rem;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-label {{ color: #64748b; }}
.stat-val {{ font-weight: 700; color: {LN_NAVY}; }}

/* ── Truck cards ── */
.truck-card {{
    background:white;border:1px solid {LN_BORDER};border-radius:14px;
    padding:20px;position:relative;overflow:hidden;
    box-shadow:0 2px 10px rgba(0,0,0,0.06);transition:all 0.2s;
}}
.truck-card:hover {{box-shadow:0 8px 24px rgba(30,45,61,0.12);transform:translateY(-2px);}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    ships  = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")

    # Ensure required columns
    if "city" not in ships.columns:
        ships["city"] = "Unknown"
    if "latitude" not in ships.columns:
        ships["latitude"] = 19.076
    if "longitude" not in ships.columns:
        ships["longitude"] = 72.877
    if "weight" not in ships.columns:
        ships["weight"] = 300.0
    if "priority" not in ships.columns:
        ships["priority"] = "MEDIUM"
    if "sla_hours" not in ships.columns:
        ships["sla_hours"] = 48
    if "traffic_mult" not in ships.columns:
        np.random.seed(42)
        ships["traffic_mult"] = np.random.uniform(1.0, 2.8, len(ships)).round(2)
    if "truck_type" not in ships.columns:
        ships["truck_type"] = "Tata 1109 20T (HCV)"
    if "truck_category" not in ships.columns:
        ships["truck_category"] = ships["truck_type"].map(
            {k: v["category"] for k, v in TRUCK_TYPES_INFO.items()}
        ).fillna("HCV")

    if "city" not in routes.columns:
        routes["city"] = "Stop"
    if "latitude" not in routes.columns:
        routes["latitude"] = 19.076
    if "longitude" not in routes.columns:
        routes["longitude"] = 72.877
    if "weight" not in routes.columns:
        routes["weight"] = 300.0

    try:
        veh = pd.read_csv("vehicle_summary.csv")
    except FileNotFoundError:
        veh = routes.groupby("vehicle").agg(
            stops=("stop_order","max"),
            load_kg=("weight","sum"),
            distance_km=("route_distance_km","first"),
            time_hr=("travel_time_hr","sum"),
            fuel_cost=("fuel_cost","sum"),
            toll_cost=("toll_cost","sum"),
            driver_cost=("driver_cost","sum"),
            sla_penalty=("sla_penalty","sum"),
            total_cost=("total_cost","sum"),
            carbon_kg=("carbon_kg","sum"),
            sla_breaches=("sla_breach_hr",lambda x:(x>0).sum()),
        ).reset_index()
        veh["utilization_pct"] = (veh["load_kg"] / VEHICLE_CAP * 100).round(1)

    try:
        vehicles_master = pd.read_csv("vehicles.csv")
    except FileNotFoundError:
        vehicles_master = None

    try:
        lanes = pd.read_csv("lanes.csv")
    except FileNotFoundError:
        lanes = None

    try:
        metrics = pd.read_csv("metrics.csv")
        r = metrics.iloc[0]
        base = dict(
            distance_km  = float(r.get("baseline_distance_km",  53527)),
            time_hr      = float(r.get("baseline_time_hr",      973)),
            fuel_cost    = float(r.get("baseline_fuel_cost",    634949)),
            toll_cost    = float(r.get("baseline_toll_cost",    104570)),
            driver_cost  = float(r.get("baseline_driver_cost",  173168)),
            total_cost   = float(r.get("baseline_total_cost",   912688)),
            carbon_kg    = float(r.get("baseline_carbon_kg",    13077)),
            sla_pct      = float(r.get("baseline_sla_adherence_pct", 4.0)),
        )
    except Exception:
        base = dict(
            distance_km=53527, time_hr=973, fuel_cost=601808,
            toll_cost=112741,  driver_cost=197566, total_cost=912115,
            carbon_kg=13077,   sla_pct=4.0,
        )

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
    return ships, routes, veh, base, opt, vehicles_master, lanes

@st.cache_data
def perm_imp(routes_df):
    np.random.seed(42)
    feats = {
        "Travel Time":    "travel_time_hr",
        "Fuel Cost":      "fuel_cost",
        "Toll Cost":      "toll_cost",
        "Driver Cost":    "driver_cost",
        "Carbon Emitted": "carbon_kg",
        "SLA Breach":     "sla_breach_hr",
        "Package Weight": "weight",
    }
    available = {k: v for k, v in feats.items() if v in routes_df.columns}
    X = routes_df[list(available.values())].copy()
    y = routes_df["mo_score"].values
    base_mae = np.mean(np.abs(y - y.mean()))
    imp = {}
    for lbl, col in available.items():
        sh2 = X.copy()
        sh2[col] = np.random.permutation(sh2[col].values)
        proxy = sh2.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        imp[lbl] = abs(np.mean(np.abs(y - proxy.values)) - base_mae)
    tot = sum(imp.values()) + 1e-9
    return {k: round(v / tot * 100, 1) for k, v in sorted(imp.items(), key=lambda x: -x[1])}

@st.cache_data
def stop_cont(routes_df):
    cols    = [c for c in ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"] if c in routes_df.columns]
    labels  = {"travel_time_hr":"Travel Time","fuel_cost":"Fuel Cost","toll_cost":"Toll Cost",
               "driver_cost":"Driver Cost","carbon_kg":"Carbon","sla_breach_hr":"SLA Breach"}
    weights = {"travel_time_hr":0.30,"fuel_cost":0.20,"toll_cost":0.05,
               "driver_cost":0.15,"carbon_kg":0.20,"sla_breach_hr":0.10}
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max() - df[c].min()
        df[c] = (df[c] - df[c].min()) / (rng + 1e-9)
        df[c] *= weights.get(c, 0.1)
    df.columns = [labels[c] for c in cols]
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

ships, routes, veh_sum, base, opt, vehicles_master, lanes = load()
fi = perm_imp(routes)
sc = stop_cont(routes)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:white;padding:18px 16px 14px;text-align:center;
                border-bottom:2px solid {LN_BORDER};margin-bottom:10px;">
        <div style="background:white;border-radius:10px;padding:8px 14px;display:inline-block;">
            {logo_html(height=54, center=True)}
        </div>
        <div style="font-size:0.56rem;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.16em;margin-top:6px;font-weight:600;">
            AI Route Optimization Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-sec">📊 Analytics Suite</div>', unsafe_allow_html=True)

    pg = st.radio("nav", [
        "🏢 About LoRRI",
        "🤖 LoRRI AI Assistant",
        "🚛 Fleet Intelligence",
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
        <div class="stat-row"><span class="stat-label">Truck Types</span>
            <span class="stat-val">{len(TRUCK_TYPES_INFO)}</span></div>
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
        <div class="stat-row"><span class="stat-label">Toll Plazas</span>
            <span class="stat-val">{len(TOLL_PLAZAS)} mapped</span></div>
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
# DEMO MODE BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="demo-banner">
    🔵 <b>Demo Mode</b> — Running on India-realistic synthetic data
    (30 cities · 7 LoRRI truck categories · real NH toll corridors · INR pricing).
    Connect LoRRI's API to switch to live freight data.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div style="padding:0 32px 24px;background:{LN_LGRAY};">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT LORRI
# ══════════════════════════════════════════════════════════════════════════════
if pg == "🏢 About LoRRI":
    page_header("🏢 About LoRRI", "LogisticsNow · AI-Powered Logistics Intelligence Platform")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#2d4a6b 60%,#1a3a1a 100%);
                border-radius:14px;padding:32px 40px;margin-bottom:22px;
                display:flex;align-items:center;gap:32px;">
        <div style="flex-shrink:0;text-align:center;">
            <div style="font-size:2.8rem;font-weight:900;letter-spacing:-2px;
                        font-family:Poppins,sans-serif;line-height:1;">
                <span style="color:white;">Lo</span><span style="color:#6bcf57;">RRI</span>
            </div>
            <div style="font-size:0.55rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:0.18em;margin-top:4px;">Logistics Intelligence</div>
        </div>
        <div style="width:1px;height:64px;background:rgba(255,255,255,0.15);flex-shrink:0;"></div>
        <div>
            <div style="font-size:1.5rem;font-weight:700;color:white;margin-bottom:6px;">
                AI-Powered Route Optimization Engine
            </div>
            <div style="font-size:0.88rem;color:#94a3b8;margin-bottom:14px;">
                Logistics Rating &amp; Intelligence — India's Premier Platform
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <div style="display:inline-flex;align-items:center;gap:6px;
                            background:rgba(58,125,44,0.2);border:1px solid {LN_GREEN};
                            color:#6bcf57;border-radius:20px;padding:4px 14px;
                            font-size:0.72rem;font-weight:600;text-transform:uppercase;">● LIVE SYSTEM</div>
                <div style="display:inline-flex;align-items:center;gap:6px;
                            background:rgba(30,122,191,0.2);border:1px solid #1e7abf;
                            color:#60a5fa;border-radius:20px;padding:4px 14px;
                            font-size:0.72rem;font-weight:600;text-transform:uppercase;">7 TRUCK CLASSES</div>
                <div style="display:inline-flex;align-items:center;gap:6px;
                            background:rgba(230,126,34,0.2);border:1px solid #e67e22;
                            color:#fb923c;border-radius:20px;padding:4px 14px;
                            font-size:0.72rem;font-weight:600;text-transform:uppercase;">{len(TOLL_PLAZAS)} TOLL PLAZAS</div>
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
                    padding:20px;height:190px;border-top:3px solid {LN_GREEN};">
            <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
            <div style="font-size:0.85rem;font-weight:700;color:{LN_NAVY};margin-bottom:8px;">{title}</div>
            <div style="font-size:0.75rem;color:#64748b;line-height:1.65;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sh">🚛 Fleet Capability Matrix</div>', unsafe_allow_html=True)
    tcols = st.columns(len(TRUCK_TYPES_INFO))
    for col, (name, spec) in zip(tcols, TRUCK_TYPES_INFO.items()):
        cat_color = CAT_COLORS.get(spec["category"], LN_GREEN)
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                    padding:14px;text-align:center;border-top:3px solid {cat_color};height:230px;">
            <div style="font-size:1.6rem;">{spec['icon']}</div>
            <div style="font-size:0.7rem;font-weight:800;color:{LN_NAVY};margin:4px 0 2px;line-height:1.3;">{name}</div>
            <div style="background:{cat_color};color:white;border-radius:10px;padding:1px 8px;
                        font-size:0.6rem;font-weight:700;display:inline-block;margin-bottom:8px;">{spec['category']}</div>
            <div style="font-size:0.68rem;color:#64748b;line-height:1.8;">
                <b>{spec['cap_ton']}T</b> / {spec['cap_cbm']}m³<br>
                ₹{spec['cost_km']}/km<br>
                {spec['kmpl']} kmpl<br>
                {spec['co2_km']} kg CO₂/km
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns(4)
    for col, icon, val, lbl, color in [
        (dc1, "🏙️", "30 Cities",       "Across 15 Indian states",    LN_GREEN),
        (dc2, "🚛", "7 Truck Types",    "LCV to 40T MXL (LoRRI cats)","#1e7abf"),
        (dc3, "🛣️", f"{len(TOLL_PLAZAS)} Toll Plazas", "Real NH corridors", "#e67e22"),
        (dc4, "📦", "200 Shipments",    "90-day stochastic demand",    "#8e44ad"),
    ]:
        col.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                    padding:16px;text-align:center;border-top:3px solid {color};">
            <div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>
            <div style="font-size:1.0rem;font-weight:800;color:{color};">{val}</div>
            <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🤖 LoRRI AI Assistant":
    page_header("🤖 LoRRI AI Assistant",
                "Vectorless RAG · BM25-lite · Pandas Retrieval · Rule Router · Fully Offline")

    route_map_text = {
        v: f"Mumbai → {' → '.join(routes[routes['vehicle']==v].sort_values('stop_order')['city'].tolist())}"
        for v in sorted(routes["vehicle"].unique())
    }

    KNOWLEDGE_CHUNKS = [
        {"id":"company","triggers":["logisticsnow","company","about","who","contact","email","phone","website","lorri"],
         "text":(f"COMPANY: LogisticsNow (logisticsnow.in) | connect@logisticsnow.in | +91-9867773508\n"
                 f"LoRRI = Logistics Rating & Intelligence — India's premier logistics platform.\n"
                 f"7 truck categories: LCV, SCV, ICV, MCV, HCV, HXL, MXL.\n{len(TOLL_PLAZAS)} toll plazas mapped on national highways.")},
        {"id":"optimization","triggers":["cvrp","optimize","optimization","weighted","objective","score","solver","algorithm"],
         "text":("OPTIMIZATION ENGINE: CVRP framework.\n"
                 "Objective: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%).\n"
                 "Solver: Nearest-neighbour heuristic with multi-objective scoring.\n"
                 "Re-optimization triggers: traffic delay >30% OR priority escalation.\n"
                 "Explainability: permutation-based feature importance (SHAP-style).")},
        {"id":"trucks","triggers":["truck","vehicle","fleet","category","lcv","scv","icv","mcv","hcv","hxl","mxl","tata","eicher","ashok","volvo","trailer"],
         "text":(f"TRUCK CATEGORIES (7 types):\n"
                 + "\n".join([f"- {n}: {s['category']} | {s['cap_ton']}T | ₹{s['cost_km']}/km | {s['co2_km']}kg CO₂/km" for n,s in TRUCK_TYPES_INFO.items()]))},
        {"id":"tolls","triggers":["toll","plaza","highway","nh","corridor","expressway"],
         "text":(f"TOLL PLAZAS: {len(TOLL_PLAZAS)} plazas mapped across India.\n"
                 f"Key corridors: Mumbai-Pune (NH48), Delhi-Jaipur (NH48), Bengaluru-Chennai (NH44)\n"
                 f"Toll multipliers: LCV 0.5x | ICV 1.0x | HCV 1.5x | HXL 2.0x | MXL 2.5x")},
        {"id":"pricing","triggers":["fuel","cost","price","inr","rupee","penalty","wage"],
         "text":("PRICING MODEL (₹ INR):\n- Fuel: ₹12/km\n- Driver wages: ₹180/hr\n- SLA breach penalty: ₹500/hr late")},
        {"id":"sla","triggers":["sla","late","breach","delay","on time","delivery","adherence"],
         "text":(f"SLA PERFORMANCE:\n- Optimized: {opt['sla_pct']:.1f}% (baseline {base['sla_pct']:.0f}%)\n"
                 f"- Breaches: {opt['breaches']} cities\n- Penalty: ₹500/hr | Total: {inr(veh_sum['sla_penalty'].sum())}\n- Windows: HIGH=24hr, MEDIUM=48hr, LOW=72hr")},
        {"id":"carbon","triggers":["carbon","co2","emission","green","environment","sustainability","eco"],
         "text":(f"CARBON & SUSTAINABILITY:\n- Optimized: {opt['carbon_kg']:,.1f} kg | Baseline: {base['carbon_kg']:,.1f} kg\n"
                 f"- Saved: {base['carbon_kg']-opt['carbon_kg']:,.1f} kg ({(base['carbon_kg']-opt['carbon_kg'])/base['carbon_kg']*100:.1f}% reduction)\n"
                 f"- Trees equivalent: {int((base['carbon_kg']-opt['carbon_kg'])/21):,}")},
        {"id":"fleet_summary","triggers":["fleet","total","summary","overall","depot","mumbai","shipment","save","saving","baseline","optimized"],
         "text":(f"FLEET SUMMARY (Mumbai Depot):\n- {opt['n_ships']} shipments | {opt['n_vehicles']} trucks\n"
                 f"- Optimized: {inr(opt['total_cost'])} | Baseline: {inr(base['total_cost'])}\n"
                 f"- Saved: {inr(base['total_cost']-opt['total_cost'])} ({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}%)\n"
                 f"- Distance: {opt['distance_km']:,.0f} km (was {base['distance_km']:,.0f} km)")},
    ]

    def rag_retrieve(query, top_k=3):
        q = query.lower()
        scored = [(sum((2 if len(t.split())>1 else 1) for t in chunk["triggers"] if t in q), chunk) for chunk in KNOWLEDGE_CHUNKS]
        scored.sort(key=lambda x: -x[0])
        tops = [c["text"] for s,c in scored[:top_k] if s>0]
        return "\n\n---\n\n".join(tops) if tops else ""

    def pandas_retrieve(query):
        q = query.lower()
        results = []
        for v in sorted(routes["vehicle"].unique()):
            if f"truck {v}" in q or f"vehicle {v}" in q:
                row = veh_sum[veh_sum["vehicle"]==v]
                if not row.empty:
                    r  = row.iloc[0]
                    bc = routes[(routes["vehicle"]==v)&(routes["sla_breach_hr"]>0)]["city"].tolist()
                    results.append(
                        f"{truck_name(v)} (Truck {v}):\nRoute: {route_map_text.get(v,'?')}\n"
                        f"Stops:{int(r['stops'])} | Dist:{r['distance_km']:,.0f}km | Load:{r['load_kg']:.0f}kg ({r['utilization_pct']:.0f}%)\n"
                        f"Fuel:{inr(r['fuel_cost'])} | Toll:{inr(r['toll_cost'])} | Driver:{inr(r['driver_cost'])} | Penalty:{inr(r['sla_penalty'])}\n"
                        f"Total:{inr(r['total_cost'])} | CO₂:{r['carbon_kg']:.1f}kg\n"
                        f"Breaches:{int(r['sla_breaches'])} ({'cities:'+','.join(bc) if bc else 'none OK'})"
                    )
        if any(k in q for k in ["late","breach","which cities","missed sla"]):
            bd = routes[routes["sla_breach_hr"]>0][["vehicle","city","priority","sla_breach_hr","sla_penalty"]].copy()
            if not bd.empty:
                bd["vehicle"] = bd["vehicle"].apply(lambda v: truck_name(v))
                results.append("SLA BREACHES:\n" + bd.to_string(index=False))
            else:
                results.append("SLA BREACHES: None — all on time!")
        return "\n\n".join(results) if results else ""

    def rule_based_answer(query):
        q = query.lower().strip()
        if q in {"hi","hello","hey","namaste","hii"}:
            return (f"**Namaste!** I'm the LoRRI AI Assistant.\n\n"
                    f"Fleet: **{opt['n_ships']} shipments**, **{opt['n_vehicles']} trucks**, "
                    f"7 truck classes, {len(TOLL_PLAZAS)} toll plazas.\nTotal cost **{inr(opt['total_cost'])}**. Ask me anything!", 99)
        if any(k in q for k in ["contact","email","phone","reach"]):
            return ("**LogisticsNow Contact:**\n- 🌐 logisticsnow.in\n"
                    "- 📧 connect@logisticsnow.in\n- 📞 +91-9867773508", 99)
        if any(k in q for k in ["how much did we save","total saving","how much saved"]):
            saved = base["total_cost"] - opt["total_cost"]
            return (f"**Total Savings: {inr(saved)} ({saved/base['total_cost']*100:.1f}% reduction)**\n\n"
                    f"| Category | Baseline | Optimized | Saved |\n|---|---|---|---|\n"
                    f"| Fuel | {inr(base['fuel_cost'])} | {inr(opt['fuel_cost'])} | **{inr(base['fuel_cost']-opt['fuel_cost'])}** |\n"
                    f"| Toll | {inr(base['toll_cost'])} | {inr(opt['toll_cost'])} | **{inr(base['toll_cost']-opt['toll_cost'])}** |\n"
                    f"| Driver | {inr(base['driver_cost'])} | {inr(opt['driver_cost'])} | **{inr(base['driver_cost']-opt['driver_cost'])}** |\n"
                    f"| **Total** | {inr(base['total_cost'])} | {inr(opt['total_cost'])} | **{inr(saved)}** |", 98)
        if any(k in q for k in ["truck type","vehicle type","fleet type","categories"]):
            lines = ["**7 Truck Categories in LoRRI fleet:**\n"]
            for n, s in TRUCK_TYPES_INFO.items():
                lines.append(f"- {s['icon']} **{n}** ({s['category']}): {s['cap_ton']}T | ₹{s['cost_km']}/km | {s['co2_km']} kg CO₂/km")
            return "\n".join(lines), 99
        for v in sorted(routes["vehicle"].unique()):
            if f"truck {v}" in q and "route" in q:
                r = veh_sum[veh_sum["vehicle"]==v].iloc[0]
                return (f"**{truck_name(v)} Route:**\n🚛 {route_map_text.get(v,'N/A')}\n\n"
                        f"{int(r['stops'])} stops · {r['distance_km']:,.0f} km · {inr(r['total_cost'])}", 97)
        if any(k in q for k in ["which cities","cities late","missed sla","where late"]):
            bd = routes[routes["sla_breach_hr"]>0][["city","vehicle","sla_breach_hr","sla_penalty"]]
            if bd.empty:
                return "**No SLA breaches** — all on time!", 99
            lines = [f"**{len(bd)} cities late:**\n"]
            for _, r in bd.iterrows():
                lines.append(f"- **{r['city']}** ({truck_name(r['vehicle'])}) — {r['sla_breach_hr']:.1f}hr late · {inr(r['sla_penalty'])}")
            return "\n".join(lines), 97
        return None, None

    def call_rag_pipeline(query, _history):
        rule_reply, rule_conf = rule_based_answer(query)
        if rule_reply:
            return rule_reply, rule_conf, "Rule-based router", "Instant match"
        rag_ctx    = rag_retrieve(query, top_k=3)
        pandas_ctx = pandas_retrieve(query)
        sources    = []
        if rag_ctx:    sources.append("RAG chunks")
        if pandas_ctx: sources.append("Pandas DataFrames")
        src_label   = " + ".join(sources) if sources else "Synthesized"
        n_chunks    = sum(1 for c in KNOWLEDGE_CHUNKS if c["text"] in rag_ctx) if rag_ctx else 0
        chunks_info = f"{n_chunks} chunk(s) + {'DataFrame rows' if pandas_ctx else 'no structured data'}"
        ctx_parts   = [p for p in [pandas_ctx, rag_ctx] if p]
        reply = "\n\n".join(ctx_parts) if ctx_parts else (
            f"Fleet: **{opt['n_ships']} shipments** | **{opt['n_vehicles']} trucks** | "
            f"Cost **{inr(opt['total_cost'])}** | SLA **{opt['sla_pct']:.0f}%**\n\n"
            f"Ask about trucks, toll plazas, SLA, costs, carbon, or optimization."
        )
        conf = 92 if (rag_ctx and pandas_ctx) else (88 if (rag_ctx or pandas_ctx) else 72)
        return reply, conf, src_label, chunks_info

    # UI
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{LN_NAVY} 0%,#1a4d2e 100%);
                border-radius:14px;padding:20px 26px;margin-bottom:16px;
                display:flex;align-items:center;gap:18px;">
        <div style="flex-shrink:0;background:rgba(255,255,255,0.08);border-radius:10px;padding:8px 12px;">
            {logo_html_dark(height=44)}
        </div>
        <div>
            <div style="font-size:1.05rem;font-weight:700;color:white;">LoRRI Intelligence Assistant</div>
            <div style="font-size:0.76rem;color:#94a3b8;margin-top:3px;">
                <b style="color:{LN_GREEN}">Vectorless RAG</b> · BM25-lite · Pandas Retrieval · Rule Router · Fully Offline
            </div>
        </div>
        <div style="margin-left:auto;">
            <div style="background:rgba(34,197,94,0.15);border:1px solid {LN_GREEN};
                        border-radius:20px;padding:4px 14px;font-size:0.7rem;
                        color:{LN_GREEN};font-weight:600;">OFFLINE AI</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    CHIPS = [
        ("What is LoRRI?","🏢"),("How much did we save?","💰"),("Which cities were late?","⚠️"),
        ("Truck types & categories","🚛"),("Toll plazas info","🛣️"),("Carbon savings?","🌿"),
        ("Truck 1 route?","🗺️"),("Tomorrow traffic risk","🚦"),("Suggest cost saving","💡"),("Contact LogisticsNow","📞"),
    ]
    chip_cols = st.columns(5)
    chip_clicked = False
    for idx, (question, emoji) in enumerate(CHIPS):
        with chip_cols[idx % 5]:
            if st.button(f"{emoji} {question}", key=f"chip_{idx}", use_container_width=True):
                st.session_state.chip_prompt = question
                chip_clicked = True
    if chip_clicked:
        st.rerun()

    if not st.session_state.msgs:
        with st.chat_message("assistant", avatar="🚚"):
            st.markdown(
                f"**Namaste! Welcome to LoRRI Intelligence Assistant.**\n\n"
                f"Fleet: **{opt['n_ships']} shipments** · **{opt['n_vehicles']} trucks** · "
                f"**7 truck classes** · **{len(TOLL_PLAZAS)} toll plazas** · "
                f"Cost **{inr(opt['total_cost'])}** · SLA **{opt['sla_pct']:.0f}%**\n\nClick a chip or type your question."
            )

    for m in st.session_state.msgs:
        avatar = "🚚" if m["role"] == "assistant" else "👤"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("meta"):
                meta = m["meta"]
                cc = LN_GREEN if meta["conf"]>=90 else ("#f59e0b" if meta["conf"]>=80 else "#dc2626")
                st.markdown(f"""<div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                    border:1px solid {LN_BORDER};border-radius:20px;display:inline-flex;
                    align-items:center;gap:8px;font-size:0.7rem;color:#64748b;">
                    <span style="color:{cc};font-weight:700;">●</span>
                    <b style="color:{cc};">{meta['conf']}%</b>
                    &nbsp;·&nbsp; {meta['source']} &nbsp;·&nbsp; {meta['chunks']}
                </div>""", unsafe_allow_html=True)

    final_prompt = st.session_state.pop("chip_prompt", None)
    typed = st.chat_input("Ask about fleet, tolls, truck types, costs, routes, SLA, carbon...", key="lorri_chat_input")
    if typed:
        final_prompt = typed

    if final_prompt:
        st.session_state.msgs.append({"role": "user", "content": final_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(final_prompt)
        api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-10:]]
        with st.chat_message("assistant", avatar="🚚"):
            with st.spinner("Running RAG pipeline..."):
                reply, conf, source, chunks = call_rag_pipeline(final_prompt, api_msgs)
            st.markdown(reply)
            cc = LN_GREEN if conf>=90 else ("#f59e0b" if conf>=80 else "#dc2626")
            st.markdown(f"""<div style="margin-top:8px;padding:5px 12px;background:#f8fafc;
                border:1px solid {LN_BORDER};border-radius:20px;display:inline-flex;
                align-items:center;gap:8px;font-size:0.7rem;color:#64748b;">
                <span style="color:{cc};font-weight:700;">●</span>
                <b style="color:{cc};">{conf}%</b> &nbsp;·&nbsp; {source} &nbsp;·&nbsp; {chunks}
            </div>""", unsafe_allow_html=True)
        st.session_state.msgs.append({"role":"assistant","content":reply,"meta":{"conf":conf,"source":source,"chunks":chunks}})
        st.rerun()

    if st.session_state.msgs:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.msgs = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FLEET INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🚛 Fleet Intelligence":
    page_header("🚛 Fleet Intelligence", "Truck profiles · Route intelligence · Cargo analytics · Toll summary")
    loading_state("Loading fleet data...")

    tab1, tab2, tab3, tab4 = st.tabs(["🚛 Truck Types", "📋 Vehicle Master", "📦 Cargo & Routes", "🛣️ Toll Analysis"])

    with tab1:
        st.markdown(f'<div class="sh">Truck Category Comparison</div>', unsafe_allow_html=True)
        truck_df = pd.DataFrame([
            {"Truck Model": name, "Category": s["category"], "Capacity (T)": s["cap_ton"],
             "Volume (m³)": s["cap_cbm"], "Cost/km (₹)": s["cost_km"],
             "Fixed/day (₹)": s["fixed"], "Fuel (kmpl)": s["kmpl"],
             "CO₂ (kg/km)": s["co2_km"], "Toll Multiplier": s["toll_mult"]}
            for name, s in TRUCK_TYPES_INFO.items()
        ])
        st.dataframe(
            truck_df.style
                .background_gradient(subset=["Capacity (T)"], cmap="Blues")
                .background_gradient(subset=["CO₂ (kg/km)"], cmap="Reds")
                .background_gradient(subset=["Cost/km (₹)"], cmap="YlOrRd")
                .format({"Capacity (T)":"{:.2f}","CO₂ (kg/km)":"{:.2f}","Fuel (kmpl)":"{:.1f}"}),
            use_container_width=True, hide_index=True
        )

        c1, c2 = st.columns(2)
        with c1:
            names = list(TRUCK_TYPES_INFO.keys())
            caps  = [s["cap_ton"] for s in TRUCK_TYPES_INFO.values()]
            colors_list = [CAT_COLORS[s["category"]] for s in TRUCK_TYPES_INFO.values()]
            short_names = [n.split("(")[1].rstrip(")") if "(" in n else n.split()[0] for n in names]
            fig_cap = go.Figure(go.Bar(x=short_names, y=caps, marker_color=colors_list,
                text=[f"{c}T" for c in caps], textposition="outside"))
            apply_theme(fig_cap, height=300, title="Payload Capacity by Truck Type")
            fig_cap.update_yaxes(title_text="Tons")
            st.plotly_chart(fig_cap, use_container_width=True)

        with c2:
            co2s = [s["co2_km"] for s in TRUCK_TYPES_INFO.values()]
            fig_co2 = go.Figure(go.Scatter(
                x=short_names, y=co2s, mode="lines+markers",
                line=dict(color="#c0392b", width=2),
                marker=dict(size=10, color=colors_list),
                name="CO₂ kg/km"
            ))
            apply_theme(fig_co2, height=300, title="CO₂ Emissions by Truck Class")
            fig_co2.update_yaxes(title_text="kg CO₂/km")
            st.plotly_chart(fig_co2, use_container_width=True)

    with tab2:
        if vehicles_master is not None:
            st.markdown(f'<div class="sh">Vehicle Master Registry ({len(vehicles_master)} vehicles)</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Vehicles", len(vehicles_master))
            c2.metric("Truck Types", vehicles_master["truck_type"].nunique() if "truck_type" in vehicles_master.columns else 7)
            c3.metric("Home Hubs", vehicles_master["home_hub"].nunique() if "home_hub" in vehicles_master.columns else "—")
            c4.metric("Avg Utilization", f"{vehicles_master['current_utilization_pct'].mean():.1f}%" if "current_utilization_pct" in vehicles_master.columns else "—")
            st.dataframe(vehicles_master, use_container_width=True, hide_index=True)
        else:
            # Build vehicle master from route data + TRUCK_TYPES_INFO
            vm_rows = []
            hubs = ["Mumbai","Pune","Ahmedabad","Surat","Nashik"]
            for i, v in enumerate(sorted(routes["vehicle"].unique())):
                vr  = veh_sum[veh_sum["vehicle"]==v]
                name = truck_name(v)
                spec = TRUCK_TYPES_INFO.get(name, list(TRUCK_TYPES_INFO.values())[0])
                hub  = hubs[i % len(hubs)]
                dist = float(vr["distance_km"].iloc[0]) if len(vr) else 0
                load = float(vr["load_kg"].iloc[0]) if len(vr) else 0
                util = round(load / (spec["cap_ton"]*1000) * 100, 1) if spec["cap_ton"] > 0 else 0
                vm_rows.append({
                    "Vehicle ID":   f"VH-{1000+v}",
                    "Truck Name":   name,
                    "Category":     spec["category"],
                    "Capacity (T)": spec["cap_ton"],
                    "Home Hub":     hub,
                    "Distance (km)": round(dist, 1),
                    "Load (kg)":    round(load, 1),
                    "Utilization %": min(util, 100),
                    "Fuel (kmpl)":  spec["kmpl"],
                    "CO₂/km":       spec["co2_km"],
                    "Cost/km (₹)":  spec["cost_km"],
                    "Status":       "🟢 Active",
                })
            vm_df = pd.DataFrame(vm_rows)
            st.markdown(f'<div class="sh">Vehicle Master Registry ({len(vm_df)} active vehicles)</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Vehicles",  len(vm_df))
            c2.metric("Truck Types",     vm_df["Category"].nunique())
            c3.metric("Home Hubs",       vm_df["Home Hub"].nunique())
            c4.metric("Avg Utilization", f'{vm_df["Utilization %"].mean():.1f}%')
            st.dataframe(
                vm_df.style
                    .format({"Utilization %": "{:.1f}%", "Distance (km)": "{:,.1f}",
                             "Load (kg)": "{:,.0f}", "Cost/km (₹)": "₹{:,.0f}"})
                    .background_gradient(subset=["Utilization %"], cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, hide_index=True)

    with tab3:
        if "cargo_type" in ships.columns:
            st.markdown(f'<div class="sh">Cargo Type Distribution</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                cargo_cnt = ships.groupby("cargo_type").size().reset_index(name="count").sort_values("count", ascending=False)
                fig_cargo = go.Figure(go.Pie(
                    labels=cargo_cnt["cargo_type"], values=cargo_cnt["count"],
                    hole=0.45, textinfo="label+percent",
                    marker_colors=px.colors.qualitative.Bold
                ))
                fig_cargo.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Poppins", color=LN_NAVY), showlegend=False, title="Cargo Type Mix")
                st.plotly_chart(fig_cargo, use_container_width=True)
            with c2:
                if "truck_category" in ships.columns:
                    pivot = ships.groupby(["cargo_type","truck_category"]).size().unstack(fill_value=0)
                    fig_heat = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                        colorscale="Greens", text=pivot.values, texttemplate="%{text}",
                        colorbar=dict(title="Count")
                    ))
                    apply_theme(fig_heat, height=380, title="Cargo Type × Truck Category Matrix")
                    st.plotly_chart(fig_heat, use_container_width=True)
        else:
            # Generate cargo data from routes using TRUCK_TYPES_INFO categories
            np.random.seed(99)
            CARGO_TYPES = ["Electronics","Textiles","FMCG","Pharma","Auto Parts","Perishables","Industrial"]
            CARGO_TRUCK = {"Electronics":"ICV","Textiles":"SCV","FMCG":"HCV",
                           "Pharma":"ICV","Auto Parts":"MCV","Perishables":"LCV","Industrial":"HCV"}
            n = len(routes)
            cargo_assigned = np.random.choice(CARGO_TYPES, size=n,
                p=[0.18,0.15,0.22,0.12,0.13,0.10,0.10])
            ships_aug = routes.copy()
            ships_aug["cargo_type"]    = cargo_assigned
            ships_aug["truck_category"] = ships_aug["cargo_type"].map(CARGO_TRUCK).fillna("HCV")
            ships_aug["weight_kg"]     = ships_aug["weight"] if "weight" in ships_aug.columns else 300
            ships_aug["cargo_value_inr"] = (ships_aug["weight_kg"] *
                pd.Series(cargo_assigned).map(
                    {"Electronics":850,"Textiles":320,"FMCG":180,"Pharma":1200,
                     "Auto Parts":550,"Perishables":240,"Industrial":160}).values)

            st.markdown(f'<div class="sh">Cargo Type Distribution (derived from route data)</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                cargo_cnt = ships_aug.groupby("cargo_type").size().reset_index(name="count").sort_values("count", ascending=False)
                fig_cargo = go.Figure(go.Pie(
                    labels=cargo_cnt["cargo_type"], values=cargo_cnt["count"],
                    hole=0.45, textinfo="label+percent",
                    marker_colors=px.colors.qualitative.Bold))
                fig_cargo.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Poppins", color=LN_NAVY), showlegend=False, title="Cargo Type Mix")
                st.plotly_chart(fig_cargo, use_container_width=True)

                cargo_val = ships_aug.groupby("cargo_type")["cargo_value_inr"].sum().sort_values(ascending=False).reset_index()
                cargo_val.columns = ["Cargo Type","Total Value (₹)"]
                st.dataframe(cargo_val.style.format({"Total Value (₹)": "₹{:,.0f}"}),
                             use_container_width=True, hide_index=True)
            with c2:
                pivot = ships_aug.groupby(["cargo_type","truck_category"]).size().unstack(fill_value=0)
                fig_heat = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    colorscale=[[0,"rgba(255,220,220,1)"],[0.5,"rgba(220,38,38,0.7)"],[1,"rgba(153,0,0,1)"]],
                    text=pivot.values, texttemplate="%{text}",
                    colorbar=dict(title="Count")))
                apply_theme(fig_heat, height=340, title="Cargo Type × Truck Category Matrix")
                st.plotly_chart(fig_heat, use_container_width=True)

                # Route summary by cargo
                route_cargo = ships_aug.groupby("cargo_type").agg(
                    shipments=("cargo_type","count"),
                    avg_weight=("weight_kg","mean"),
                    total_value=("cargo_value_inr","sum"),
                ).reset_index()
                route_cargo.columns = ["Cargo","Shipments","Avg Wt (kg)","Total Value (₹)"]
                st.markdown('<div class="sh">Cargo Summary by Type</div>', unsafe_allow_html=True)
                st.dataframe(
                    route_cargo.style
                        .format({"Avg Wt (kg)": "{:.0f}", "Total Value (₹)": "₹{:,.0f}"})
                        .background_gradient(subset=["Shipments"], cmap="Reds"),
                    use_container_width=True, hide_index=True)

    with tab4:
        st.markdown(f'<div class="sh">{len(TOLL_PLAZAS)} Toll Plazas — India National Highway Network</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Plazas", len(TOLL_PLAZAS))
        c2.metric("Highways Covered", len(set(p["highway"] for p in TOLL_PLAZAS)))
        c3.metric("Corridors", len(set(p["corridor"] for p in TOLL_PLAZAS)))
        c4.metric("Avg Base Toll", f"₹{int(sum(p['toll_inr'] for p in TOLL_PLAZAS)/len(TOLL_PLAZAS)):,}")

        toll_df = pd.DataFrame(TOLL_PLAZAS)
        by_corridor = toll_df.groupby("corridor").agg(
            plazas=("name","count"), avg_toll=("toll_inr","mean"), total_toll=("toll_inr","sum")
        ).reset_index().sort_values("total_toll", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig_tc = go.Figure(go.Bar(
                x=by_corridor["total_toll"], y=by_corridor["corridor"], orientation="h",
                marker_color="#e67e22",
                text=by_corridor["total_toll"].apply(lambda x: f"₹{x:,.0f}"), textposition="outside"
            ))
            apply_theme(fig_tc, height=400, title="Total Toll by Corridor (Base ₹)")
            st.plotly_chart(fig_tc, use_container_width=True)
        with c2:
            categories = list({"LCV":"0.5x","SCV":"0.75x","ICV":"1.0x","MCV":"1.25x","HCV":"1.5x","HXL":"2.0x","MXL":"2.5x"}.items())
            avg_base = sum(p["toll_inr"] for p in TOLL_PLAZAS) / len(TOLL_PLAZAS)
            eff_tolls = [avg_base * float(mult[:-1]) for _, mult in categories]
            cat_names = [cat for cat, _ in categories]
            cat_col_list = ["#27ae60","#2ecc71","#1e7abf","#e67e22","#8e44ad","#c0392b","#2c3e50"]
            fig_tm = go.Figure(go.Bar(
                x=cat_names, y=eff_tolls, marker_color=cat_col_list,
                text=[f"₹{v:,.0f}" for v in eff_tolls], textposition="outside"
            ))
            apply_theme(fig_tm, height=400, title="Effective Toll by Truck Category (Avg Plaza)")
            fig_tm.update_yaxes(title_text="Effective Toll (₹)")
            st.plotly_chart(fig_tm, use_container_width=True)

        st.dataframe(
            toll_df[["name","highway","corridor","toll_inr"]].rename(
                columns={"name":"Plaza Name","highway":"Highway","corridor":"Corridor","toll_inr":"Base Toll (₹)"}
            ),
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "📊 Dashboard Summary":
    page_header("📊 Dashboard Summary", "Baseline vs AI-Optimized · All costs in ₹ INR")
    loading_state("Refreshing dashboard metrics...")

    st.markdown("""<div class="info-box">
    📋 <b>Report card for the full delivery run.</b>
    Baseline = trucks without AI. Optimized = LoRRI AI planner. All figures in <b>₹ INR</b>.
    </div>""", unsafe_allow_html=True)

    sc_ = base["total_cost"] - opt["total_cost"]
    sd_ = base["distance_km"] - opt["distance_km"]
    sco = base["carbon_kg"]   - opt["carbon_kg"]
    ss_ = opt["sla_pct"]      - base["sla_pct"]

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px;">
    {kpi_card("Total Cost Savings",  inr(sc_),                        f"↓ {sc_/base['total_cost']*100:.1f}% vs baseline",True,LN_GREEN)}
    {kpi_card("Optimized Distance",  f"{opt['distance_km']:,.0f} km", f"↓ {sd_:,.0f} km saved",                          True,"#1e7abf")}
    {kpi_card("SLA Adherence",       f"{opt['sla_pct']:.0f}%",        f"↑ +{ss_:.0f} pts (base {base['sla_pct']:.0f}%)", True,"#e67e22")}
    {kpi_card("Carbon Reduced",      f"{sco/1000:.1f}t CO₂",          f"↓ {sco/base['carbon_kg']*100:.1f}% cleaner",      True,"#27ae60")}
    {kpi_card("Fleet Utilization",   f"{veh_sum['utilization_pct'].mean():.1f}%", f"Avg across {opt['n_vehicles']} trucks", True,"#8e44ad")}
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fuel Saved",   inr(base["fuel_cost"]  -opt["fuel_cost"]),
              f"-{(base['fuel_cost']-opt['fuel_cost'])/base['fuel_cost']*100:.1f}%",   delta_color="inverse")
    c2.metric("Toll Saved",   inr(base["toll_cost"]  -opt["toll_cost"]),
              f"-{(base['toll_cost']-opt['toll_cost'])/base['toll_cost']*100:.1f}%",   delta_color="inverse")
    c3.metric("Driver Saved", inr(base["driver_cost"]-opt["driver_cost"]),
              f"-{(base['driver_cost']-opt['driver_cost'])/base['driver_cost']*100:.1f}%", delta_color="inverse")
    c4.metric("Time Saved",   f"{base['time_hr']-opt['time_hr']:,.1f} hr",
              f"-{(base['time_hr']-opt['time_hr'])/base['time_hr']*100:.1f}%",          delta_color="inverse")

    st.markdown(sh("Cost Reduction Over Optimization Iterations"), unsafe_allow_html=True)
    iters = list(range(1, 11))
    decay = [(base["total_cost"]-(base["total_cost"]-opt["total_cost"])*(1-np.exp(-0.4*i))) for i in iters]
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=iters, y=[base["total_cost"]]*10, mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=2), name="Baseline"))
    fig_t.add_trace(go.Scatter(x=iters, y=decay, mode="lines+markers",
        line=dict(color=LN_GREEN, width=3), marker=dict(size=7, color=LN_GREEN),
        fill="tonexty", fillcolor="rgba(58,125,44,0.08)", name="Optimized"))
    apply_theme(fig_t, height=280, title="Fleet Cost vs Optimization Iterations")
    fig_t.update_yaxes(tickprefix="₹", tickformat=",")
    fig_t.update_xaxes(title_text="Iteration")
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown(sh("Per-Truck Summary"), unsafe_allow_html=True)
    d = veh_sum.copy()
    # Use truck names from generate_data2.py
    d.insert(0, "Truck", d["vehicle"].apply(lambda v: truck_name(v)))
    d = d.drop(columns=["vehicle"])
    col_labels = ["Truck","Stops","Load (kg)","Dist (km)","Time (hr)",
                  "Fuel","Toll","Driver","SLA Penalty","Total","Carbon (kg)","SLA Breaches","Util %"]
    if len(d.columns) == len(col_labels):
        d.columns = col_labels
    st.dataframe(
        d.style
         .format({c:"{:.1f}" for c in d.columns if any(x in c.lower() for x in ["kg","km","hr","%"])})
         .format({c:"₹{:,.0f}" for c in d.columns if c in ["Fuel","Toll","Driver","SLA Penalty","Total"]})
         .background_gradient(subset=["Util %"] if "Util %" in d.columns else [], cmap="Greens")
         .background_gradient(subset=["Total"] if "Total" in d.columns else [], cmap="YlOrRd"),
        use_container_width=True, hide_index=True)

    # ── Shipments by Truck Category (FIX: guard against missing cat_colors) ──
    if "truck_type" in ships.columns or "truck_category" in ships.columns:
        st.markdown(sh("Shipments by Truck Category"), unsafe_allow_html=True)
        cat_col = "truck_category" if "truck_category" in ships.columns else "truck_type"
        cat_cnt = ships.groupby(cat_col).size().reset_index(name="count")
        cat_cnt_col = cat_cnt.columns[0]
        # Safe color lookup — fallback to LN_GREEN if category not found
        bar_colors = [CAT_COLORS.get(str(c), LN_GREEN) for c in cat_cnt[cat_cnt_col]]
        fig_cat = go.Figure(go.Bar(
            x=cat_cnt[cat_cnt_col], y=cat_cnt["count"],
            marker_color=bar_colors,
            text=cat_cnt["count"], textposition="outside"
        ))
        apply_theme(fig_cat, height=280, title="Shipments per Truck Category")
        fig_cat.update_yaxes(title_text="Shipment Count")
        st.plotly_chart(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROUTE MAP  (with Toll Plazas + truck names from generate_data2.py)
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Route Map":
    page_header("🗺️ Route Map", "Live India delivery network · Mumbai depot hub · Toll plazas overlay")
    loading_state("Refreshing traffic...")

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown(sh("Map Controls"), unsafe_allow_html=True)
        route_mode    = st.radio("Route View", ["Optimized","Baseline","Comparison"], index=0)
        show_heatmap  = st.toggle("Traffic Heatmap", value=False)
        show_tolls    = st.toggle("🛣️ Toll Plazas", value=True)
        toll_hw_filter = st.multiselect(
            "Filter by Highway",
            options=sorted(set(p["highway"] for p in TOLL_PLAZAS)),
            default=sorted(set(p["highway"] for p in TOLL_PLAZAS)),
        )
        # Use truck_name() for display — generate_data2.py truck names
        sel_v = st.multiselect(
            "Filter Trucks",
            options=sorted(routes["vehicle"].unique()),
            default=sorted(routes["vehicle"].unique()),
            format_func=lambda v: truck_name(v)   # ← truck names from generate_data2.py
        )
        st.markdown("---")
        for v in sorted(routes["vehicle"].unique()):
            vr    = routes[routes["vehicle"]==v]
            color = V_COLORS.get(v, "#999")
            vd    = veh_sum[veh_sum["vehicle"]==v].iloc[0]
            sla_r = "HIGH" if vd["sla_breaches"]>=2 else ("MED" if vd["sla_breaches"]==1 else "OK")
            st.markdown(
                f'<div class="legend-row">'
                f'<div class="legend-dot" style="background:{color}"></div>'
                f'<div><b style="color:{LN_NAVY}">{truck_name(v)}</b><br>'   # ← real truck name
                f'<span style="font-size:0.72rem;color:#64748b;">'
                f'{len(vr)} stops · {vd["distance_km"]:,.0f} km<br>'
                f'{inr(vd["total_cost"])} · SLA {sla_r}</span></div></div>',
                unsafe_allow_html=True)

        if show_tolls:
            st.markdown("---")
            st.markdown(f"""
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
                        padding:10px 12px;font-size:0.75rem;color:#92400e;">
                <b>🛣️ Toll Plazas</b><br>
                Showing {len([p for p in TOLL_PLAZAS if p['highway'] in toll_hw_filter])} plazas
            </div>""", unsafe_allow_html=True)

    with col_map:
        fig = go.Figure()
        p_dot = {"HIGH":"#dc2626","MEDIUM":"#f97316","LOW":LN_GREEN}

        if route_mode in ["Baseline","Comparison"]:
            bl  = [DEPOT["latitude"]]  + ships["latitude"].tolist()  + [DEPOT["latitude"]]
            blo = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=bl, lon=blo, mode="lines",
                line=dict(width=2, color="rgba(220,38,38,0.5)"), name="Baseline"))

        if route_mode in ["Optimized","Comparison"]:
            for v in sel_v:
                vdf   = routes[routes["vehicle"]==v].sort_values("stop_order")
                color = V_COLORS.get(v, "#999")
                lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
                lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
                # Legend shows real truck name from generate_data2.py
                fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines",
                    line=dict(width=3, color=color), name=truck_name(v), legendgroup=f"v{v}"))
                for _, row in vdf.iterrows():
                    breach = f"{row['sla_breach_hr']:.1f}hr late" if row["sla_breach_hr"]>0 else "On time"
                    dd = haversine(DEPOT["latitude"],DEPOT["longitude"],row["latitude"],row["longitude"])
                    fig.add_trace(go.Scattermap(
                        lat=[row["latitude"]], lon=[row["longitude"]],
                        mode="markers+text",
                        marker=dict(size=14, color=p_dot.get(row.get("priority","MEDIUM"),"#f97316")),
                        text=[f"T{v}.{int(row['stop_order'])}"],
                        textfont=dict(size=8, color="white"),
                        textposition="middle center",
                        hovertext=(
                            f"{truck_name(v)} Stop {int(row['stop_order'])}: "
                            f"{row.get('city',row['shipment_id'])} | "
                            f"{dd:.0f}km depot | ETA:{row['travel_time_hr']:.1f}hr | "
                            f"{inr(row['total_cost'])} | {row['carbon_kg']:.1f}kg CO₂ | {breach}"
                        ),
                        hoverinfo="text", showlegend=False, legendgroup=f"v{v}"))

        # ── Toll Plazas Overlay ─────────────────────────────────────────────
        if show_tolls:
            filtered_tolls = [p for p in TOLL_PLAZAS if p["highway"] in toll_hw_filter]
            if filtered_tolls:
                fig.add_trace(go.Scattermap(
                    lat=[p["lat"] for p in filtered_tolls],
                    lon=[p["lon"] for p in filtered_tolls],
                    mode="markers",
                    marker=dict(size=10, color="#f97316", symbol="square", opacity=0.85),
                    hovertext=[
                        f"🛣️ {p['name']}<br>{p['highway']} · {p['corridor']}<br>"
                        f"Base toll: ₹{p['toll_inr']:,}<br>"
                        f"LCV: ₹{int(p['toll_inr']*0.5):,} | HCV: ₹{int(p['toll_inr']*1.5):,}"
                        for p in filtered_tolls
                    ],
                    hoverinfo="text",
                    name=f"🛣️ Toll Plazas ({len(filtered_tolls)})",
                ))
                major_tolls = [p for p in filtered_tolls if p["toll_inr"] >= 150]
                if major_tolls:
                    fig.add_trace(go.Scattermap(
                        lat=[p["lat"] for p in major_tolls],
                        lon=[p["lon"] for p in major_tolls],
                        mode="markers+text",
                        marker=dict(size=12, color="#ea580c", symbol="square"),
                        text=[f"₹{p['toll_inr']}" for p in major_tolls],
                        textfont=dict(size=8, color="#7c2d12"),
                        textposition="top right",
                        hoverinfo="skip", showlegend=False,
                    ))

        # ── Traffic Heatmap ─────────────────────────────────────────────────
        if show_heatmap and "traffic_mult" in ships.columns:
            hm = ships["traffic_mult"].tolist()
            fig.add_trace(go.Scattermap(
                lat=ships["latitude"].tolist(), lon=ships["longitude"].tolist(),
                mode="markers",
                marker=dict(size=[v*18 for v in hm], color=hm,
                            colorscale="RdYlGn_r", cmin=1.0, cmax=3.0, opacity=0.4),
                name="Traffic", hovertext=ships["city"], hoverinfo="text"))

        fig.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["Depot"],
            textposition="top right", textfont=dict(size=10, color=LN_NAVY),
            marker=dict(size=18, color=LN_NAVY, symbol="star"), name="Mumbai Depot"))

        fig.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
            height=620, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.92)",
                        bordercolor=LN_BORDER, borderwidth=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="display:flex;gap:14px;flex-wrap:wrap;padding:10px 0;font-size:0.78rem;color:{LN_NAVY};">
            <span style="color:#dc2626;">🔴 HIGH priority stop</span>
            <span style="color:#f97316;">🟠 MEDIUM priority stop</span>
            <span style="color:{LN_GREEN};">🟢 LOW priority stop</span>
            <span style="color:#f97316;">⬡ Toll Plaza</span>
            <span>⭐ Mumbai Depot</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "💰 Financial Analysis":
    page_header("💰 Financial Analysis", "All costs in ₹ INR · Fuel ₹12/km · Driver ₹180/hr · SLA ₹500/hr")
    loading_state("Calculating savings...")

    fuel_s   = base["fuel_cost"]   - opt["fuel_cost"]
    toll_s   = base["toll_cost"]   - opt["toll_cost"]
    driver_s = base["driver_cost"] - opt["driver_cost"]
    total_s  = fuel_s + toll_s + driver_s
    roi_pct  = total_s / base["total_cost"] * 100
    payback  = int(15000 / (total_s / 30)) if total_s > 0 else 999

    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    for col, label, val, icon, color in [
        (rc1,"Fuel Saved",   inr(fuel_s),   "⛽",LN_GREEN),
        (rc2,"Toll Saved",   inr(toll_s),   "🛣️","#1e7abf"),
        (rc3,"Driver Saved", inr(driver_s), "👷","#8e44ad"),
        (rc4,"Total Saved",  inr(total_s),  "💰","#e67e22"),
        (rc5,"Payback",     f"{payback}d",  "⏳","#27ae60"),
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

    c1, c2 = st.columns(2)
    with c1:
        fig_b = go.Figure()
        for cat, bc_, lbl in [("fuel_cost",LN_GREEN,"Fuel"),("toll_cost","#1e7abf","Toll"),("driver_cost","#8e44ad","Driver")]:
            fig_b.add_trace(go.Bar(name=lbl, x=["Baseline","Optimized"],
                y=[base[cat], opt[cat]], marker_color=bc_,
                text=[inr(base[cat]), inr(opt[cat])], textposition="inside",
                textfont=dict(color="white", size=10)))
        apply_theme(fig_b, height=360, title="Cost Components: Baseline vs Optimized", legend_below=True)
        fig_b.update_layout(barmode="stack")
        fig_b.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_b, use_container_width=True)

    with c2:
        sv = {"Fuel Saved":fuel_s,"Toll Saved":toll_s,"Driver Saved":driver_s}
        fig_w = go.Figure(go.Waterfall(
            orientation="v", measure=["relative","relative","relative","total"],
            x=list(sv.keys())+["Total Saved"],
            y=list(sv.values())+[sum(sv.values())],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":LN_GREEN}},
            totals={"marker":{"color":"#1e7abf"}},
            text=[inr(v) for v in list(sv.values())+[sum(sv.values())]],
            textposition="outside"))
        apply_theme(fig_w, height=360, title="Savings Waterfall")
        fig_w.update_yaxes(tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_w, use_container_width=True)

    fig_v = go.Figure()
    for cat, bc_, lbl in [("fuel_cost",LN_GREEN,"Fuel"),("toll_cost","#1e7abf","Toll"),
                           ("driver_cost","#8e44ad","Driver"),("sla_penalty","#c0392b","SLA Penalty")]:
        fig_v.add_trace(go.Bar(name=lbl,
            x=[truck_name(v) for v in veh_sum["vehicle"]],  # ← truck names
            y=veh_sum[cat], marker_color=bc_,
            text=veh_sum[cat].apply(inr), textposition="inside",
            textfont=dict(color="white", size=9)))
    apply_theme(fig_v, height=320, legend_below=True)
    fig_v.update_layout(barmode="stack", title="Per-Truck Cost Breakdown")
    fig_v.update_yaxes(tickprefix="₹", tickformat=",")
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown(sh("Toll Cost Analysis by Truck Category"), unsafe_allow_html=True)
    toll_df_d = pd.DataFrame(TOLL_PLAZAS)
    avg_base_toll = toll_df_d["toll_inr"].mean()
    mult_map = {"LCV":0.5,"SCV":0.75,"ICV":1.0,"MCV":1.25,"HCV":1.5,"HXL":2.0,"MXL":2.5}
    cat_toll = pd.DataFrame([
        {"Category": cat, "Multiplier": m,
         "Avg Toll per Plaza": f"₹{round(avg_base_toll * m):,}",
         "Estimated Trip Toll (3 plazas)": f"₹{round(avg_base_toll * m * 3):,}"}
        for cat, m in mult_map.items()
    ])
    st.dataframe(cat_toll, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CARBON & SLA
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🌿 Carbon & SLA":
    page_header("🌿 Carbon & SLA", "Sustainability metrics · Delivery compliance")

    co2_saved  = base["carbon_kg"] - opt["carbon_kg"]
    trees      = int(co2_saved / 21)
    cars_off   = int(co2_saved / 2400)
    km_avoided = int(base["distance_km"] - opt["distance_km"])

    sus = st.columns(4)
    for col, icon, val, label, sub, color in [
        (sus[0],"🌿",f"{co2_saved:,.0f} kg","CO₂ Reduced",    f"{co2_saved/base['carbon_kg']*100:.1f}% less",LN_GREEN),
        (sus[1],"🌳",f"{trees:,}",           "Trees Equiv",    "CO₂ absorbed per year",                      "#27ae60"),
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
    c1, c2 = st.columns(2)
    with c1:
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=["Baseline","Optimized"],
            y=[base["carbon_kg"], opt["carbon_kg"]], marker_color=["#c0392b", LN_GREEN],
            text=[f"{base['carbon_kg']:,.1f} kg", f"{opt['carbon_kg']:,.1f} kg"],
            textposition="outside"))
        apply_theme(fig_c, height=280, title=f"CO₂ Saved: {co2_saved:,.1f} kg")
        fig_c.update_layout(showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

        city_co2 = routes.groupby("city")["carbon_kg"].sum().sort_values(ascending=False).head(8).reset_index()
        fig_city = go.Figure(go.Bar(x=city_co2["carbon_kg"], y=city_co2["city"], orientation="h",
            marker_color=[LN_GREEN if i>2 else "#c0392b" for i in range(len(city_co2))],
            text=city_co2["carbon_kg"].round(1).astype(str)+" kg", textposition="outside"))
        apply_theme(fig_city, height=280, title="Top Cities — CO₂ (kg)")
        fig_city.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_city, use_container_width=True)

    with c2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=opt["sla_pct"],
            number={"suffix":"%"}, title={"text":"SLA Adherence"},
            delta={"reference":base["sla_pct"],"increasing":{"color":LN_GREEN},"suffix":"% vs baseline"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":LN_GREEN},
                   "steps":[{"range":[0,50],"color":"rgba(192,57,43,0.15)"},
                             {"range":[50,80],"color":"rgba(245,158,11,0.15)"},
                             {"range":[80,100],"color":"rgba(58,125,44,0.15)"}],
                   "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":base["sla_pct"]}}))
        fig_g.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        # ── HEATMAP: SLA Breaches — Truck × Priority (from dashboard.py) ───
        bd = routes.copy()
        bd["breached"] = (bd["sla_breach_hr"] > 0).astype(int)
        piv = bd.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_h = go.Figure(go.Heatmap(
            z=piv.values,
            x=piv.columns.tolist(),
            y=[truck_name(v) for v in piv.index],
            colorscale=[[0,"rgba(255,220,220,1)"],[0.5,"rgba(220,38,38,0.8)"],[1,"rgba(153,0,0,1)"]],
            text=piv.values, texttemplate="%{text}",
            colorbar=dict(title="Breaches")))
        apply_theme(fig_h, height=260, title="SLA Breaches: Truck × Priority")
        st.plotly_chart(fig_h, use_container_width=True)

    bdf = routes[routes["sla_breach_hr"]>0][
        [c for c in ["vehicle","stop_order","city","priority","sla_hours","sla_breach_hr","sla_penalty","total_cost"] if c in routes.columns]].copy()
    if not bdf.empty:
        st.markdown(sh("SLA Breach Detail (₹500/hr penalty)"), unsafe_allow_html=True)
        bdf["vehicle"] = bdf["vehicle"].apply(lambda v: truck_name(v))
        st.dataframe(
            bdf.style
               .format({c:"{:.1f}" for c in bdf.columns if "hr" in c.lower()})
               .format({c:"₹{:,.0f}" for c in bdf.columns if "penalty" in c.lower() or "cost" in c.lower()})
               .background_gradient(subset=[c for c in bdf.columns if "breach" in c.lower()], cmap="Reds"),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🧠 Explainability":
    page_header("🧠 Explainability", "Why the AI chose these routes · SHAP-style permutation importance")

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
            <b>0.35×Cost + 0.30×Time + 0.20×Carbon + 0.15×SLA</b>.<br>
            Hardest stop: <b style="color:#fbbf24">{worst_stop['city']}</b>
            ({truck_name(int(worst_stop['vehicle']))}) — MO Score {worst_stop['mo_score']:.4f}.
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_pie = go.Figure(go.Pie(
            labels=["Cost","Travel Time","Carbon CO₂","SLA"],
            values=[35,30,20,15], hole=0.55,
            marker_colors=[LN_GREEN,"#1e7abf","#27ae60","#c0392b"],
            textinfo="label+percent"))
        fig_pie.update_layout(height=290, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
            annotations=[{"text":"Weights","x":0.5,"y":0.5,"font_size":13,"showarrow":False}])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fi_l = list(fi.keys()); fi_v = list(fi.values()); mv = max(fi_v)
        fig_fi = go.Figure(go.Bar(x=fi_v, y=fi_l, orientation="h",
            marker_color=["#c0392b" if v==mv else LN_GREEN for v in fi_v],
            text=[f"{v:.1f}%" for v in fi_v], textposition="outside"))
        apply_theme(fig_fi, height=300, title="Feature Importance (Permutation-Based)")
        fig_fi.update_xaxes(title_text="Importance (%)")
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi, use_container_width=True)

    vf  = st.selectbox("Filter:", ["All Trucks"]+[truck_name(v) for v in sorted(routes["vehicle"].unique())])
    if vf == "All Trucks":
        scd = sc
    else:
        # Find vehicle number from truck name
        v_num = next((v for v in sorted(routes["vehicle"].unique()) if truck_name(v) == vf), None)
        scd = sc[sc["vehicle"]==v_num].copy() if v_num else sc

    fc  = [c for c in ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"] if c in scd.columns]
    fco = [LN_GREEN,"#1e7abf","#8e44ad","#e67e22","#27ae60","#c0392b"][:len(fc)]
    fig_stk = go.Figure()
    for f_, c_ in zip(fc, fco):
        fig_stk.add_trace(go.Bar(name=f_, x=scd["city"], y=scd[f_], marker_color=c_))
    apply_theme(fig_stk, height=380, legend_below=True)
    fig_stk.update_layout(barmode="stack", title="Per-Stop Score Contribution by Truck")
    fig_stk.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_stk, use_container_width=True)

    t10 = routes.nlargest(10,"mo_score")[
        [c for c in ["vehicle","stop_order","city","priority","weight","travel_time_hr",
         "fuel_cost","carbon_kg","sla_breach_hr","mo_score"] if c in routes.columns]].copy()
    if "vehicle" in t10.columns:
        t10["vehicle"] = t10["vehicle"].apply(lambda v: truck_name(v))
    st.dataframe(t10, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RE-OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "⚡ Re-optimization Simulator":
    page_header("⚡ Re-optimization Simulator", "Simulate disruptions · Watch LoRRI re-plan instantly")

    st.markdown("""<div class="info-box">
    Simulate <b>traffic jams</b> or <b>priority escalations</b>.
    LoRRI re-plans the affected truck and shows Before vs After.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    all_cities_list = sorted(ships["city"].tolist()) if "city" in ships.columns else sorted(routes["city"].tolist())

    with c1:
        st.markdown(sh("Scenario 1 — Traffic Disruption"), unsafe_allow_html=True)
        city1  = st.selectbox("City hit by traffic:", all_cities_list)
        spike  = st.slider("Traffic multiplier", 1.0, 3.0, 2.5, 0.1)
        if st.button("Trigger Traffic Disruption", use_container_width=True):
            row_matches = ships[ships["city"]==city1] if "city" in ships.columns else routes[routes["city"]==city1]
            if not row_matches.empty:
                row = row_matches.iloc[0]
                om  = row.get("traffic_mult", 1.5)
                lat = row.get("latitude", DEPOT["latitude"])
                lon = row.get("longitude", DEPOT["longitude"])
                dk  = haversine(DEPOT["latitude"],DEPOT["longitude"],lat,lon)
                to  = dk/(55/om); tn=dk/(55/spike); pi=(tn-to)/to*100
                if pi > 30:
                    st.markdown(f"""<div class="warn-box">
                    <b>{city1}</b>: {om:.2f}x → <span class="tag-red">{spike:.2f}x</span> |
                    +{pi:.1f}% delay — THRESHOLD BREACHED!
                    </div>""", unsafe_allow_html=True)
                    t0 = time.time()
                    with st.spinner("Re-optimizing..."): time.sleep(1.2)
                    elapsed = time.time()-t0
                    av = routes[routes["city"]==city1]["vehicle"].values
                    if len(av):
                        vid  = av[0]
                        orig = routes[routes["vehicle"]==vid].sort_values("stop_order")
                        mask = orig["city"]==city1
                        reop = pd.concat([orig[~mask], orig[mask]]).reset_index(drop=True)
                        def rdist(df):
                            return sum(haversine(df.iloc[i]["latitude"],df.iloc[i]["longitude"],
                                                 df.iloc[i+1]["latitude"],df.iloc[i+1]["longitude"])
                                       for i in range(len(df)-1))
                        d1 = rdist(orig); d2 = rdist(reop)
                        st.markdown(f'<div class="ok-box">{truck_name(vid)} re-routed in <b>{elapsed:.2f}s</b></div>', unsafe_allow_html=True)
                        ba1,ba2,ba3 = st.columns(3)
                        ba1.metric("Distance",f"{d1:.0f} km",f"{d2-d1:+.0f} km",delta_color="inverse")
                        ba2.metric("ETA",f"{orig['travel_time_hr'].sum():.1f} hr")
                        ba3.metric("Cost",inr(orig["total_cost"].sum()))
                else:
                    st.markdown(f'<div class="ok-box">{pi:.1f}% — within 30% threshold. No re-opt needed.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown(sh("Scenario 2 — Priority Escalation"), unsafe_allow_html=True)
        city2 = st.selectbox("City escalated:", all_cities_list, key="esc")
        if st.button("Trigger Priority Escalation", use_container_width=True):
            row_matches2 = ships[ships["city"]==city2] if "city" in ships.columns else routes[routes["city"]==city2]
            if not row_matches2.empty:
                row2 = row_matches2.iloc[0]
                op_  = row2.get("priority","MEDIUM")
                os_  = row2.get("sla_hours", 48)
                if op_ == "HIGH":
                    st.markdown(f'<div class="ok-box">{city2} is already HIGH priority!</div>', unsafe_allow_html=True)
                else:
                    t0 = time.time()
                    with st.spinner("Escalating..."): time.sleep(1.0)
                    elapsed = time.time()-t0
                    av  = routes[routes["city"]==city2]["vehicle"].values
                    vid = av[0] if len(av) else 1
                    orig = routes[routes["vehicle"]==vid].sort_values("stop_order")
                    mask = orig["city"]==city2
                    pen  = orig[mask]["sla_penalty"].values[0] if mask.any() else 0
                    st.markdown(f"""<div class="ok-box">
                    <b>{city2}</b>: <span class="tag-yellow">{op_}</span> →
                    <span class="tag-red">HIGH</span> | SLA: {os_}hr → 24hr |
                    Moved to Stop 1 on {truck_name(vid)} in <b>{elapsed:.2f}s</b>
                    </div>""", unsafe_allow_html=True)
                    ba1,ba2,ba3 = st.columns(3)
                    ba1.metric("Old SLA",f"{os_} hr","→ 24 hr")
                    ba2.metric("Penalty Saved",inr(pen),delta=f"-{inr(pen)}",delta_color="inverse")
                    ba3.metric("Replan Time",f"{elapsed:.2f}s")

    # ── Live Risk Monitor — FIX: removed overlapping box, clean layout ──────
    st.markdown("---")
    st.markdown(sh("Live Risk Monitor"), unsafe_allow_html=True)

    if "traffic_mult" in ships.columns:
        rdf = ships[["city","traffic_mult","priority","sla_hours"]].copy()
        rdf["risk"] = (rdf["traffic_mult"]/1.8*0.6
                       + rdf["sla_hours"].map({24:1.0,48:0.5,72:0.2}).fillna(0.3)*0.4).round(3)
        rdf["status"] = rdf["risk"].apply(
            lambda x: "HIGH RISK" if x>0.7 else ("MONITOR" if x>0.4 else "STABLE"))
        rdf = rdf.sort_values("risk", ascending=False)
        # Chart only — no surrounding box div that would clash
        fig_r = px.bar(
            rdf.head(15), x="city", y="risk", color="status",
            color_discrete_map={"HIGH RISK":"#c0392b","MONITOR":"#f59e0b","STABLE":LN_GREEN},
            title="Top 15 Cities by Re-Optimization Risk",
            labels={"risk":"Risk Score","city":"City"},
            height=340
        )
        fig_r.add_hline(y=0.7, line_dash="dash", line_color="#c0392b",
                        annotation_text="Trigger threshold (0.70)")
        apply_theme(fig_r)
        st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.markdown("""<div class="warn-box">Traffic multiplier data not available in shipments.csv.
        Run generate_data2.py to add the traffic_mult column.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ROUTE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🔮 AI Route Predictor":
    page_header("🔮 AI Route Predictor", "Plan a new route · Estimated distance, cost, ETA, carbon, SLA risk, toll plazas")
    loading_state("Initializing prediction engine...")

    all_cities = sorted(routes["city"].tolist())

    f1, f2, f3 = st.columns(3)
    with f1:
        dst_city = st.selectbox("Destination City", all_cities, index=min(5,len(all_cities)-1))
        cargo_wt = st.slider("Cargo Weight (kg)", 50, 800, 350, 25)
    with f2:
        priority   = st.selectbox("Priority", ["LOW","MEDIUM","HIGH"])
        truck_sel  = st.selectbox("Truck Type", list(TRUCK_TYPES_INFO.keys()))
    with f3:
        traffic = st.selectbox("Traffic", ["Normal (1.0x)","Moderate (1.5x)","Heavy (2.0x)","Severe (3.0x)"])

    sla_map      = {"LOW":72,"MEDIUM":48,"HIGH":24}
    sla_hr       = sla_map[priority]
    traffic_mult = float(traffic.split("(")[1].split("x")[0])
    truck_spec   = TRUCK_TYPES_INFO[truck_sel]

    if st.button("Predict Route", use_container_width=True, type="primary"):
        t0 = time.time()
        with st.spinner("Computing optimal route..."): time.sleep(1.2)

        dst_row = routes[routes["city"]==dst_city]
        if dst_row.empty:
            st.error("City not found."); st.stop()

        dst_lat = dst_row.iloc[0]["latitude"]
        dst_lon = dst_row.iloc[0]["longitude"]
        dist_p  = haversine(DEPOT["latitude"],DEPOT["longitude"],dst_lat,dst_lon)
        avg_spd = 55/traffic_mult
        time_hr = dist_p/avg_spd
        fuel_p  = dist_p*truck_spec["cost_km"]
        toll_p  = dist_p*2.8*truck_spec["toll_mult"]
        drv_p   = time_hr*180
        pen_p   = max(0,(time_hr-sla_hr)*500) if time_hr>sla_hr else 0
        total_p = fuel_p+toll_p+drv_p+pen_p
        co2_p   = dist_p*truck_spec["co2_km"]
        sla_risk = "HIGH" if pen_p>0 else ("MEDIUM" if time_hr>sla_hr*0.8 else "LOW")
        elapsed  = time.time()-t0

        mid_lat = (DEPOT["latitude"]+dst_lat)/2
        mid_lon = (DEPOT["longitude"]+dst_lon)/2
        route_tolls = [p for p in TOLL_PLAZAS if
            haversine(mid_lat,mid_lon,p["lat"],p["lon"]) < dist_p*0.6]
        n_tolls = len(route_tolls)
        est_toll_collected = sum(p["toll_inr"]*truck_spec["toll_mult"] for p in route_tolls)

        st.markdown(f"""
        <div style="background:{LN_GREEN};border-radius:10px;padding:10px 18px;
                    color:white;font-size:0.82rem;margin-bottom:16px;">
            Route predicted in <b>{elapsed:.2f}s</b> — Mumbai to <b>{dst_city}</b> ·
            {cargo_wt}kg · {priority} priority · {truck_sel} · ~{n_tolls} toll plaza(s)
        </div>""", unsafe_allow_html=True)

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        for col, label, val, color in [
            (k1,"Distance",   f"{dist_p:.0f} km",  LN_GREEN),
            (k2,"ETA",        f"{time_hr:.1f} hr",  "#1e7abf"),
            (k3,"Total Cost", inr(total_p),          "#e67e22"),
            (k4,"Toll Est.",  inr(est_toll_collected),"#f97316"),
            (k5,"CO₂",        f"{co2_p:.1f} kg",    "#27ae60"),
            (k6,"SLA Risk",   sla_risk, "#dc2626" if sla_risk=="HIGH" else ("#f59e0b" if sla_risk=="MEDIUM" else LN_GREEN)),
        ]:
            col.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                        padding:14px;text-align:center;border-top:3px solid {color};">
                <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:6px;">{label}</div>
                <div style="font-size:1.05rem;font-weight:800;color:{color};">{val}</div>
            </div>""", unsafe_allow_html=True)

        if pen_p > 0:
            st.markdown(f'<div class="warn-box">⚠️ SLA breach risk! Penalty: <b>{inr(pen_p)}</b></div>',
                        unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:white;border:1px solid {LN_BORDER};border-radius:10px;
                    padding:12px 18px;margin:12px 0;display:flex;gap:24px;flex-wrap:wrap;
                    border-left:4px solid {truck_spec['color']};">
            <span style="font-size:0.8rem;color:{LN_NAVY};">{truck_spec['icon']} <b>{truck_sel}</b></span>
            <span style="font-size:0.8rem;color:#64748b;">Category: <b>{truck_spec['category']}</b></span>
            <span style="font-size:0.8rem;color:#64748b;">Capacity: <b>{truck_spec['cap_ton']}T</b></span>
            <span style="font-size:0.8rem;color:#64748b;">Fuel: <b>{truck_spec['kmpl']} kmpl</b></span>
            <span style="font-size:0.8rem;color:#64748b;">CO₂: <b>{truck_spec['co2_km']} kg/km</b></span>
            <span style="font-size:0.8rem;color:#f97316;">Toll mult: <b>{truck_spec['toll_mult']}x</b></span>
        </div>""", unsafe_allow_html=True)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],dst_lat], lon=[DEPOT["longitude"],dst_lon],
            mode="lines", line=dict(width=4,color=LN_GREEN), name="Primary Route"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],mid_lat+1.5,dst_lat],
            lon=[DEPOT["longitude"],mid_lon-1.5,dst_lon],
            mode="lines", line=dict(width=2,color="rgba(220,38,38,0.45)"),
            name="Alt 1 (Scenic)"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"],mid_lat-1.0,dst_lat],
            lon=[DEPOT["longitude"],mid_lon+1.0,dst_lon],
            mode="lines", line=dict(width=2,color="rgba(30,122,191,0.45)"),
            name="Alt 2 (Express)"))
        fig_pred.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["Mumbai Depot"],
            textposition="top right", textfont=dict(size=10,color=LN_NAVY),
            marker=dict(size=16,color=LN_NAVY), showlegend=False))
        fig_pred.add_trace(go.Scattermap(
            lat=[dst_lat], lon=[dst_lon],
            mode="markers+text", text=[dst_city],
            textposition="top right", textfont=dict(size=10,color=LN_NAVY),
            marker=dict(size=14,color=LN_GREEN), showlegend=False))
        if route_tolls:
            fig_pred.add_trace(go.Scattermap(
                lat=[p["lat"] for p in route_tolls],
                lon=[p["lon"] for p in route_tolls],
                mode="markers",
                marker=dict(size=11,color="#f97316",symbol="square",opacity=0.9),
                hovertext=[f"🛣️ {p['name']}<br>{p['highway']} · Base: ₹{p['toll_inr']:,}<br>"
                           f"Your category ({truck_spec['category']}): ₹{int(p['toll_inr']*truck_spec['toll_mult']):,}"
                           for p in route_tolls],
                hoverinfo="text",
                name=f"🛣️ Toll Plazas ({n_tolls})"))
        fig_pred.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=mid_lat,lon=mid_lon),zoom=5),
            height=420, margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.9)"))
        st.plotly_chart(fig_pred, use_container_width=True)

        if route_tolls:
            st.markdown(sh(f"Toll Plazas on Route ({n_tolls} plazas · Est. {inr(est_toll_collected)})"), unsafe_allow_html=True)
            toll_table = pd.DataFrame([{
                "Plaza": p["name"], "Highway": p["highway"],
                "Base Toll": f"₹{p['toll_inr']:,}",
                f"Your Toll ({truck_spec['category']}, {truck_spec['toll_mult']}x)": f"₹{int(p['toll_inr']*truck_spec['toll_mult']):,}"
            } for p in route_tolls])
            st.dataframe(toll_table, use_container_width=True, hide_index=True)

        dist_a1=dist_p*1.11; time_a1=dist_a1/avg_spd; total_a1=dist_a1*truck_spec["cost_km"]+dist_a1*1.2+time_a1*180
        dist_a2=dist_p*0.94; time_a2=dist_a2/(avg_spd*1.15); total_a2=dist_a2*truck_spec["cost_km"]+dist_a2*5.2+time_a2*180
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #dc2626;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:8px;">Alt 1 — Scenic Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    {dist_a1:.0f} km (+{dist_a1-dist_p:.0f} km) · {time_a1:.1f} hr · {inr(total_a1)}<br>
                    Avoids highway tolls · Fewer toll plazas
                </div>
            </div>""", unsafe_allow_html=True)
        with a2:
            st.markdown(f"""
            <div style="background:white;border:1px solid {LN_BORDER};border-radius:12px;
                        padding:18px 20px;border-top:3px solid #1e7abf;">
                <div style="font-weight:700;color:{LN_NAVY};margin-bottom:8px;">Alt 2 — Express Route</div>
                <div style="font-size:0.83rem;color:#475569;line-height:2;">
                    {dist_a2:.0f} km (-{dist_p-dist_a2:.0f} km) · {time_a2:.1f} hr · {inr(total_a2)}<br>
                    Higher toll, faster delivery · More toll plazas
                </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{LN_NAVY};color:#94a3b8;padding:20px 40px;margin-top:20px;
            font-size:0.75rem;display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:12px;">
    <div style="display:flex;align-items:center;gap:16px;">
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:8px 12px;">
            {logo_html_dark(height=32)}
        </div>
        <div>
            <b style="color:white;">LogisticsNow</b> · LoRRI AI Route Optimization Engine v3.0<br>
            <span style="color:#64748b;">7 truck classes · {len(TOLL_PLAZAS)} toll plazas · generate_data2.py schema · Problem Statement #4</span>
        </div>
    </div>
    <div style="text-align:right;">
        📧 connect@logisticsnow.in &nbsp;·&nbsp; 🌐 logisticsnow.in
    </div>
</div>
""", unsafe_allow_html=True)
