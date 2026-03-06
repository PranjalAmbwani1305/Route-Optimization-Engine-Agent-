"""
LoRRI · AI Route Optimization Engine
Production SaaS Dashboard — Sidebar Navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import re

from rag_engine import get_rag_response, set_hf_key

st.set_page_config(
    page_title="LoRRI · Route Intelligence",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
COLORS      = px.colors.qualitative.Bold
VEHICLE_CAP = 800

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
        return ships, routes, metrics, veh
    except FileNotFoundError:
        st.error("⚠️ CSV files not found. Run generate_data.py and route_solver.py first.")
        st.stop()

ships, routes, metrics, veh_summary = load_data()

@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {
        "Travel Time":"travel_time_hr","Fuel Cost":"fuel_cost","Toll Cost":"toll_cost",
        "Driver Cost":"driver_cost","Carbon Emitted":"carbon_kg",
        "SLA Breach":"sla_breach_hr","Package Weight":"weight",
    }
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    bm = np.mean(np.abs(y - y.mean()))
    imp = {}
    for label, col in features.items():
        s = X.copy(); s[col] = np.random.permutation(s[col].values)
        proxy = s.apply(lambda c:(c-c.mean())/(c.std()+1e-9)).mean(axis=1)
        imp[label] = abs(np.mean(np.abs(y - proxy.values)) - bm)
    total = sum(imp.values()) + 1e-9
    return {k: round(v/total*100,1) for k,v in sorted(imp.items(), key=lambda x:-x[1])}

@st.cache_data
def compute_stop_contributions(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    weights = [0.30,0.20,0.05,0.15,0.20,0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max()-df[c].min()
        df[c] = (df[c]-df[c].min())/(rng+1e-9)
    for i,c in enumerate(cols): df[c] = df[c]*weights[i]
    df.columns = labels
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

feature_importance = compute_feature_importance(routes)
stop_contrib       = compute_stop_contributions(routes)

# Session state
if "page"          not in st.session_state: st.session_state.page = "dashboard"
if "rag_messages"  not in st.session_state:
    st.session_state.rag_messages = [{"role":"assistant",
        "content":"Hello! I'm **LoRRI AI** — your Route Intelligence Analyst.\n\nAsk me anything about your deliveries: costs, carbon, SLA, vehicle performance, or how the AI made routing decisions."}]
if "rag_sources"   not in st.session_state: st.session_state.rag_sources = []

# Derived metrics
_ds  = metrics["baseline_distance_km"]  - metrics["opt_distance_km"]
_ts  = metrics["baseline_time_hr"]       - metrics["opt_time_hr"]
_cs  = metrics["baseline_total_cost"]    - metrics["opt_total_cost"]
_co2 = metrics["baseline_carbon_kg"]     - metrics["opt_carbon_kg"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium SaaS dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── Base ── */
html,body,[class*="css"]{
  font-family:'Plus Jakarta Sans',sans-serif;
  background:#050810!important;
  color:#c8d6ee!important;
}
.main .block-container{
  padding:0 2.2rem 3rem 2.2rem!important;
  max-width:1600px;
}

/* ── Sidebar complete override ── */
section[data-testid="stSidebar"]{
  background:#07091280!important;
  border-right:1px solid rgba(255,255,255,.06)!important;
  backdrop-filter:blur(20px)!important;
  width:240px!important;
  min-width:240px!important;
  padding:0!important;
}
section[data-testid="stSidebar"] > div:first-child{
  padding:0!important;
  overflow:hidden!important;
}
section[data-testid="stSidebar"] .block-container{
  padding:0!important;
}

/* hide streamlit sidebar controls */
[data-testid="stSidebarNavItems"]{display:none!important}
button[data-testid="baseButton-headerNoPadding"]{display:none!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:#07091280}
::-webkit-scrollbar-thumb{background:#1a2d47;border-radius:4px}

/* ── Top bar ── */
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:1.2rem 2.2rem 1rem 2.2rem;
  border-bottom:1px solid rgba(255,255,255,.05);
  background:rgba(5,8,16,.8);
  backdrop-filter:blur(12px);
  position:sticky;top:0;z-index:100;
  margin:-0rem -2.2rem 2rem -2.2rem;
}
.topbar-title{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:#f0f6ff;letter-spacing:-.02em}
.topbar-sub{font-family:'DM Mono',monospace;font-size:.6rem;color:#3a5070;letter-spacing:.1em;margin-top:2px}
.topbar-right{display:flex;gap:10px;align-items:center}
.status-dot{width:8px;height:8px;background:#3fb950;border-radius:50%;box-shadow:0 0 6px #3fb95060;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.status-lbl{font-family:'DM Mono',monospace;font-size:.62rem;color:#3fb950;letter-spacing:.08em}

/* ── Sidebar nav ── */
.nav-logo{
  padding:1.6rem 1.4rem 1.2rem 1.4rem;
  border-bottom:1px solid rgba(255,255,255,.05);
}
.nav-wordmark{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;letter-spacing:-.04em;color:#f0f6ff;line-height:1}
.nav-wordmark em{color:#3b82f6;font-style:normal}
.nav-tagline{font-family:'DM Mono',monospace;font-size:.58rem;color:#2a4060;letter-spacing:.12em;text-transform:uppercase;margin-top:5px}
.nav-version{display:inline-block;font-family:'DM Mono',monospace;font-size:.55rem;color:#3b82f6;background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);border-radius:4px;padding:2px 7px;margin-top:6px;letter-spacing:.06em}

.nav-section{
  padding:.8rem 1.4rem .35rem 1.4rem;
  font-family:'DM Mono',monospace;font-size:.55rem;color:#253a52;
  letter-spacing:.18em;text-transform:uppercase;
}
.nav-divider{height:1px;background:rgba(255,255,255,.04);margin:.6rem 1.4rem}

/* nav item via Streamlit button override */
[data-testid="stSidebar"] [data-testid="stButton"]>button{
  background:transparent!important;
  border:none!important;
  color:#5a7a9a!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  font-size:.82rem!important;
  font-weight:500!important;
  border-radius:8px!important;
  padding:.55rem .9rem!important;
  width:100%!important;
  text-align:left!important;
  justify-content:flex-start!important;
  margin:1px 0!important;
  transition:all .15s!important;
}
[data-testid="stSidebar"] [data-testid="stButton"]>button:hover{
  background:rgba(59,130,246,.08)!important;
  color:#c8d6ee!important;
}

/* Active nav — use a specific class trick via markdown */
.nav-active-btn button{
  background:rgba(59,130,246,.12)!important;
  color:#f0f6ff!important;
  border-left:2px solid #3b82f6!important;
}

/* Nav footer */
.nav-footer{
  position:absolute;bottom:0;left:0;right:0;
  padding:1rem 1.4rem;
  border-top:1px solid rgba(255,255,255,.05);
}
.nav-client{font-family:'DM Mono',monospace;font-size:.6rem;color:#2a4060;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
.nav-client-name{font-family:'Plus Jakarta Sans',sans-serif;font-size:.78rem;font-weight:600;color:#5a7a9a}

/* ── KPI cards ── */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:2rem}
.kpi-card{
  background:linear-gradient(145deg,#0a0f1e,#080c18);
  border:1px solid rgba(255,255,255,.06);
  border-radius:14px;padding:1.2rem 1.4rem;
  position:relative;overflow:hidden;
  transition:border-color .2s,transform .15s;
}
.kpi-card:hover{border-color:rgba(59,130,246,.2);transform:translateY(-1px)}
.kpi-card::after{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--ac,rgba(59,130,246,.4)),transparent);
}
.kpi-ico{font-size:1.1rem;margin-bottom:.5rem}
.kpi-lbl{font-family:'DM Mono',monospace;font-size:.58rem;color:#3a5070;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.3rem}
.kpi-val{font-family:'Syne',sans-serif;font-size:1.85rem;font-weight:700;color:#f0f6ff;line-height:1;letter-spacing:-.03em}
.kpi-val sub{font-size:.85rem;color:#3a5070;font-weight:400;letter-spacing:0}
.kpi-chg{font-family:'DM Mono',monospace;font-size:.63rem;margin-top:6px}
.up{color:#3fb950}.dn{color:#f85149}.nt{color:#3b82f6}

/* ── Section header ── */
.sec-hd{
  display:flex;align-items:center;gap:12px;
  margin:0 0 1.2rem 0;
}
.sec-hd-icon{
  width:32px;height:32px;border-radius:8px;
  background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.2);
  display:flex;align-items:center;justify-content:center;font-size:.9rem;flex-shrink:0;
}
.sec-hd-text{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#f0f6ff}
.sec-hd-sub{font-family:'DM Mono',monospace;font-size:.6rem;color:#3a5070;letter-spacing:.08em;margin-top:2px}

/* ── Card container ── */
.card{
  background:#080c18;border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:1.3rem 1.5rem;margin-bottom:1rem;
}

/* ── Group label ── */
.grp{
  font-family:'DM Mono',monospace;font-size:.58rem;color:#3a5070;
  letter-spacing:.18em;text-transform:uppercase;
  padding:.5rem .8rem;background:rgba(59,130,246,.04);
  border:1px solid rgba(59,130,246,.07);border-radius:7px;
  margin:1rem 0 .6rem 0;display:flex;align-items:center;gap:8px;
}
.grp::after{content:'';flex:1;height:1px;background:rgba(59,130,246,.1)}

/* ── Info callout ── */
.info-box{
  background:rgba(59,130,246,.06);
  border:1px solid rgba(59,130,246,.15);
  border-left:3px solid #3b82f6;
  border-radius:10px;padding:13px 16px;
  margin-bottom:1.4rem;
  font-size:.83rem;line-height:1.75;color:#7a9cbf;
}
.info-box .ib-head{
  font-family:'DM Mono',monospace;font-size:.58rem;color:#3b82f6;
  letter-spacing:.14em;text-transform:uppercase;margin-bottom:7px;
}
.info-box b,.info-box strong{color:#d4e4f7}
.info-box .eg{margin-top:9px;padding:8px 12px;background:rgba(59,130,246,.05);border-radius:7px;font-size:.79rem;color:#4a6a87}
.info-box .eg::before{content:'💡 ';font-weight:700;color:#e3b341}

/* ── Callout variants ── */
.cb{border-radius:8px;padding:10px 14px;margin:.4rem 0 .9rem 0;font-size:.82rem;line-height:1.65;border-left:3px solid}
.cb-b{background:rgba(59,130,246,.07);border-color:#3b82f6;color:#7a9cbf}
.cb-g{background:rgba(63,185,80,.07);border-color:#3fb950;color:#56d364}
.cb-y{background:rgba(227,179,65,.07);border-color:#e3b341;color:#e3b341}
.cb-r{background:rgba(248,81,73,.07);border-color:#f85149;color:#ff7b72}
.cb b{color:#f0f6ff}

/* ── Metric cards (Streamlit override) ── */
[data-testid="metric-container"]{
  background:#080c18!important;
  border:1px solid rgba(255,255,255,.06)!important;
  border-radius:10px!important;padding:.95rem 1.1rem!important;
  transition:border-color .2s!important;
}
[data-testid="metric-container"]:hover{border-color:rgba(59,130,246,.18)!important}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;font-size:1.45rem!important;color:#f0f6ff!important}
[data-testid="stMetricLabel"]{font-family:'DM Mono',monospace!important;font-size:.58rem!important;color:#3a5070!important;letter-spacing:.12em!important;text-transform:uppercase!important}
[data-testid="stMetricDelta"]{font-family:'DM Mono',monospace!important;font-size:.65rem!important}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.06)!important;border-radius:10px!important;overflow:hidden}
thead tr th{background:#090e1c!important;font-family:'DM Mono',monospace!important;font-size:.61rem!important;letter-spacing:.09em!important;text-transform:uppercase!important;color:#3a5070!important;border-bottom:1px solid rgba(255,255,255,.07)!important}

/* ── Form ── */
[data-baseweb="select"]>div{background:#080c18!important;border-color:rgba(255,255,255,.08)!important;border-radius:8px!important;color:#c8d6ee!important;font-family:'Plus Jakarta Sans',sans-serif!important;font-size:.84rem!important}
[data-testid="stTextInput"] input{background:#080c18!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:8px!important;color:#c8d6ee!important;font-family:'DM Mono',monospace!important;font-size:.82rem!important}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;box-shadow:0 0 0 2px rgba(59,130,246,.1)!important}
[data-testid="stToggle"] label{font-size:.8rem!important;color:#7a9cbf!important}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{background:#3b82f6!important}

/* ── Buttons (main area) ── */
.main [data-testid="stButton"]>button{
  background:rgba(59,130,246,.1)!important;
  border:1px solid rgba(59,130,246,.25)!important;
  color:#3b82f6!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  font-size:.81rem!important;font-weight:600!important;
  border-radius:8px!important;padding:.5rem 1.2rem!important;
  transition:all .2s!important;
}
.main [data-testid="stButton"]>button:hover{
  background:rgba(59,130,246,.18)!important;
  color:#f0f6ff!important;border-color:#3b82f6!important;
}

/* ── Chat ── */
.chat-wrap{min-height:440px;max-height:540px;overflow-y:auto;background:#050810;border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.2rem}
.msg-row{display:flex;gap:10px;margin-bottom:14px;align-items:flex-end}
.msg-row.you{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:.62rem;font-weight:500;flex-shrink:0}
.av-u{background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:#3b82f6}
.av-b{background:rgba(63,185,80,.12);border:1px solid rgba(63,185,80,.25);color:#3fb950}
.bub{max-width:78%;padding:10px 14px;font-size:.83rem;line-height:1.65;color:#c8d6ee;border:1px solid}
.bub.u{background:rgba(59,130,246,.08);border-color:rgba(59,130,246,.2);border-radius:12px 12px 2px 12px}
.bub.b{background:#080c18;border-color:rgba(255,255,255,.07);border-radius:12px 12px 12px 2px}
.mn{font-family:'DM Mono',monospace;font-size:.56rem;letter-spacing:.1em;margin-bottom:5px}
.mn-u{color:#3b82f6}.mn-b{color:#3fb950}
.src-row{margin-top:8px;padding-top:7px;border-top:1px solid rgba(255,255,255,.05)}
.chip{display:inline-block;font-family:'DM Mono',monospace;font-size:.56rem;background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.15);border-radius:4px;padding:2px 7px;margin:1px 3px 1px 0;color:#3a5070}
.chat-empty{text-align:center;padding:4rem 2rem;color:#1a2d3f;font-family:'DM Mono',monospace;font-size:.72rem;line-height:2.5}

/* ── KB list ── */
.kb-entry{font-family:'DM Mono',monospace;font-size:.68rem;color:#5a7a9a;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);line-height:1.8}
.kb-entry span{color:#2a4060;margin-right:5px}

/* ── Plotly transparent ── */
.js-plotly-plot .plotly,.js-plotly-plot .plotly .main-svg{background:transparent!important}

/* ── Misc ── */
hr{border-color:rgba(255,255,255,.05)!important}
[data-testid="stCaptionContainer"]{font-family:'DM Mono',monospace!important;font-size:.61rem!important;color:#1a2d3f!important}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly theme
# ─────────────────────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#7a9cbf", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — segmented navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo block
    st.markdown("""
    <div class="nav-logo">
      <div class="nav-wordmark">Lo<em>RRI</em></div>
      <div class="nav-tagline">Route Intelligence Platform</div>
      <span class="nav-version">v2.1 · Enterprise</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SEGMENT 1: Analytics ─────────────────────────────────────────────────
    st.markdown('<div class="nav-section">📊 Analytics</div>', unsafe_allow_html=True)

    if st.button("  🏠  Dashboard Overview",    key="nav_dash",   use_container_width=True):
        st.session_state.page = "dashboard"
    if st.button("  🗺️  Route Intelligence",    key="nav_map",    use_container_width=True):
        st.session_state.page = "route_map"
    if st.button("  💰  Financial Analysis",    key="nav_cost",   use_container_width=True):
        st.session_state.page = "cost"

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── SEGMENT 2: Sustainability ────────────────────────────────────────────
    st.markdown('<div class="nav-section">🌿 Sustainability</div>', unsafe_allow_html=True)

    if st.button("  🌍  Carbon & Emissions",    key="nav_carbon", use_container_width=True):
        st.session_state.page = "carbon"
    if st.button("  ✅  SLA Performance",       key="nav_sla",    use_container_width=True):
        st.session_state.page = "sla"

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── SEGMENT 3: AI Engine ─────────────────────────────────────────────────
    st.markdown('<div class="nav-section">🧠 AI Engine</div>', unsafe_allow_html=True)

    if st.button("  🔍  Explainability",         key="nav_expl",   use_container_width=True):
        st.session_state.page = "explainability"
    if st.button("  ⚡  Re-Optimization",         key="nav_reopt",  use_container_width=True):
        st.session_state.page = "reopt"

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── SEGMENT 4: Intelligence ──────────────────────────────────────────────
    st.markdown('<div class="nav-section">🤖 Intelligence</div>', unsafe_allow_html=True)

    if st.button("  💬  AI Assistant (RAG)",     key="nav_rag",    use_container_width=True):
        st.session_state.page = "rag"

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

    # ── Settings block ───────────────────────────────────────────────────────
    st.markdown('<div class="nav-section">⚙️ Settings</div>', unsafe_allow_html=True)
    hf_key = st.text_input("HuggingFace API Key", type="password", placeholder="hf_...",
                            help="Get yours free at huggingface.co/settings/tokens",
                            label_visibility="collapsed")
    if hf_key:
        set_hf_key(hf_key)
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:#3fb950;margin-top:3px;">✓ API key saved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:#2a4060;margin-top:3px;">Enter key for RAG Assistant</div>', unsafe_allow_html=True)

    # Client footer
    st.markdown("""
    <br><br>
    <div style="padding:.8rem 0;border-top:1px solid rgba(255,255,255,.04);">
      <div class="nav-client">Client Account</div>
      <div class="nav-client-name">India Logistics Corp.</div>
      <div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#1a2d3f;margin-top:3px;">
      Run · 3/6/2026 · Mumbai Depot
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
page_titles = {
    "dashboard":    ("Dashboard Overview",     "Baseline vs Optimized · Full Run Report"),
    "route_map":    ("Route Intelligence",     "Live delivery network · India"),
    "cost":         ("Financial Analysis",     "Fuel · Toll · Driver · Savings"),
    "carbon":       ("Carbon & Emissions",     "CO₂ tracking · Environmental impact"),
    "sla":          ("SLA Performance",        "Service Level Agreement · On-time delivery"),
    "explainability":("AI Explainability",     "Permutation importance · Decision transparency"),
    "reopt":        ("Re-Optimization Engine", "Live disruption response · Traffic & priority"),
    "rag":          ("AI Assistant",           "RAG-powered · Grounded on your data · Llama 3.1-8B"),
}
pg  = st.session_state.page
ttl, tsub = page_titles.get(pg, ("LoRRI", ""))

st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-title">{ttl}</div>
    <div class="topbar-sub">{tsub}</div>
  </div>
  <div class="topbar-right">
    <div class="status-dot"></div>
    <span class="status-lbl">Live · Mumbai Depot</span>
    &nbsp;&nbsp;
    <span style="font-family:'DM Mono',monospace;font-size:.6rem;color:#2a4060;">
      {int(metrics['num_shipments'])} shipments · {int(metrics['num_vehicles'])} vehicles
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: always-visible mini KPI strip
# ─────────────────────────────────────────────────────────────────────────────
def render_kpi_strip():
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card" style="--ac:rgba(59,130,246,.45)">
        <div class="kpi-ico">📏</div>
        <div class="kpi-lbl">Distance Optimized</div>
        <div class="kpi-val">{metrics['opt_distance_km']:,.0f}<sub> km</sub></div>
        <div class="kpi-chg up">▼ {_ds:,.0f} km saved &nbsp;·&nbsp; -{_ds/metrics['baseline_distance_km']*100:.1f}%</div>
      </div>
      <div class="kpi-card" style="--ac:rgba(227,179,65,.4)">
        <div class="kpi-ico">⏱️</div>
        <div class="kpi-lbl">Travel Time</div>
        <div class="kpi-val">{metrics['opt_time_hr']:,.1f}<sub> hr</sub></div>
        <div class="kpi-chg up">▼ {_ts:.1f} hr faster &nbsp;·&nbsp; -{_ts/metrics['baseline_time_hr']*100:.1f}%</div>
      </div>
      <div class="kpi-card" style="--ac:rgba(63,185,80,.4)">
        <div class="kpi-ico">💰</div>
        <div class="kpi-lbl">Cost Saved</div>
        <div class="kpi-val">₹{_cs:,.0f}</div>
        <div class="kpi-chg up">▼ -{_cs/metrics['baseline_total_cost']*100:.1f}% vs baseline</div>
      </div>
      <div class="kpi-card" style="--ac:rgba(86,211,100,.4)">
        <div class="kpi-ico">🌿</div>
        <div class="kpi-lbl">Carbon Reduced</div>
        <div class="kpi-val">{_co2:,.0f}<sub> kg</sub></div>
        <div class="kpi-chg up">▼ -{_co2/metrics['baseline_carbon_kg']*100:.1f}% CO₂</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard Overview
# ══════════════════════════════════════════════════════════════════════════════
if pg == "dashboard":
    render_kpi_strip()

    st.markdown("""
    <div class="info-box">
      <div class="ib-head">📋 What is this page?</div>
      This is the <strong>command centre</strong> for your entire delivery run. It compares what would
      have happened without AI (<strong>Baseline</strong>) versus what LoRRI's optimizer chose
      (<strong>Optimized</strong>). Every green number means money saved, time saved, or less pollution.
      <div class="eg">Think of it like a before/after report card: the AI grouped nearby cities per driver,
      so each truck takes a smart loop instead of zigzagging across India.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Run Info ──────────────────────────────────────────────────────────────
    st.markdown('<div class="grp">🏭 Run Information</div>', unsafe_allow_html=True)
    g1 = st.columns(4)
    g1[0].metric("📦 Shipments",    int(metrics["num_shipments"]))
    g1[1].metric("🚛 Vehicles",     int(metrics["num_vehicles"]))
    g1[2].metric("🏭 Depot",        "Mumbai")
    g1[3].metric("⚖️ Vehicle Cap",  f"{VEHICLE_CAP} kg")

    # ── Optimization results ──────────────────────────────────────────────────
    st.markdown('<div class="grp">⚡ Optimization Results — Baseline vs AI</div>', unsafe_allow_html=True)
    g2 = st.columns(4)
    g2[0].metric("📏 Distance (km)",   f"{metrics['opt_distance_km']:,.1f}",
                 delta=f"{metrics['opt_distance_km']-metrics['baseline_distance_km']:,.0f} km", delta_color="inverse")
    g2[1].metric("⏱️ Travel Time (hr)",f"{metrics['opt_time_hr']:,.1f}",
                 delta=f"{metrics['opt_time_hr']-metrics['baseline_time_hr']:,.1f} hr",         delta_color="inverse")
    g2[2].metric("💰 Total Cost (₹)",  f"₹{metrics['opt_total_cost']:,.0f}",
                 delta=f"₹{metrics['opt_total_cost']-metrics['baseline_total_cost']:,.0f}",     delta_color="inverse")
    g2[3].metric("🌿 Carbon (kg CO₂)", f"{metrics['opt_carbon_kg']:,.1f}",
                 delta=f"{metrics['opt_carbon_kg']-metrics['baseline_carbon_kg']:,.1f} kg",     delta_color="inverse")

    # ── Savings ───────────────────────────────────────────────────────────────
    st.markdown('<div class="grp">💵 Financial Savings Breakdown</div>', unsafe_allow_html=True)
    g3 = st.columns(4)
    g3[0].metric("⛽ Fuel Saved",
        f"₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f}",
        delta=f"-{(metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'])/metrics['baseline_fuel_cost']*100:.1f}%",
        delta_color="inverse")
    g3[1].metric("🛣️ Toll Saved",
        f"₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}",
        delta=f"-{(metrics['baseline_toll_cost']-metrics['opt_toll_cost'])/metrics['baseline_toll_cost']*100:.1f}%",
        delta_color="inverse")
    g3[2].metric("👷 Driver Saved",
        f"₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f}",
        delta=f"-{(metrics['baseline_driver_cost']-metrics['opt_driver_cost'])/metrics['baseline_driver_cost']*100:.1f}%",
        delta_color="inverse")
    g3[3].metric("✅ SLA Adherence",
        f"{metrics['opt_sla_adherence_pct']:.0f}%",
        delta=f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']:.0f} pts",
        delta_color="normal")

    # ── Per-vehicle table ──────────────────────────────────────────────────────
    st.markdown('<div class="grp">🚛 Per-Vehicle Performance Summary</div>', unsafe_allow_html=True)
    tc, bc = st.columns([4,1])
    with bc:
        show_all = st.toggle("All columns", value=False)
        st.download_button("⬇ Export CSV", data=veh_summary.to_csv(index=False),
                           file_name="lorri_vehicle_summary.csv", use_container_width=True)
    dv = veh_summary.copy()
    dv.columns = ["Vehicle","Stops","Load (kg)","Dist (km)","Time (hr)","Fuel ₹","Toll ₹","Driver ₹",
                  "SLA Penalty ₹","Total Cost ₹","Carbon kg","SLA Breaches","Util %"]
    show_cols = dv.columns.tolist() if show_all else \
                ["Vehicle","Stops","Dist (km)","Time (hr)","Total Cost ₹","Carbon kg","Util %","SLA Breaches"]
    fmt = {"Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}","Fuel ₹":"₹{:,.0f}",
           "Toll ₹":"₹{:,.0f}","Driver ₹":"₹{:,.0f}","SLA Penalty ₹":"₹{:,.0f}",
           "Total Cost ₹":"₹{:,.0f}","Carbon kg":"{:.1f}","Util %":"{:.1f}%"}
    with tc:
        st.dataframe(
            dv[show_cols].style
            .format({k:v for k,v in fmt.items() if k in show_cols})
            .background_gradient(subset=[c for c in ["Util %","Total Cost ₹"] if c in show_cols], cmap="Blues"),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Route Intelligence
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "route_map":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">🗺️ What is this page?</div>
      A <strong>real map of India</strong> showing every truck's optimized path. Each coloured line is a
      different vehicle. <strong>Red dots</strong> = HIGH priority, <strong>orange</strong> = medium,
      <strong>green</strong> = low urgency. Toggle baseline to see how much cleaner the AI route is.
      <div class="eg">Each driver takes a smooth geographic loop instead of zigzagging randomly across India.</div>
    </div>
    """, unsafe_allow_html=True)

    map_col, ctrl_col = st.columns([3,1])

    with ctrl_col:
        st.markdown('<div class="grp">Map Controls</div>', unsafe_allow_html=True)
        show_baseline   = st.toggle("Show Baseline Route", value=False)
        show_unassigned = st.toggle("Show Unassigned Stops", value=True)
        selected_v      = st.multiselect("Filter Vehicles",
                                         options=sorted(routes["vehicle"].unique()),
                                         default=sorted(routes["vehicle"].unique()))

        st.markdown('<div class="grp">Route Legend</div>', unsafe_allow_html=True)
        for v in sorted(routes["vehicle"].unique()):
            c   = COLORS[(v-1)%len(COLORS)]
            cnt = len(routes[routes["vehicle"]==v])
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.71rem;padding:3px 0;display:flex;align-items:center;gap:8px;"><span style="color:{c};font-size:.9rem;">━━</span><b style="color:#c8d6ee;">V{v}</b><span style="color:#3a5070;"> · {cnt} stops</span></div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:10px;font-family:'DM Mono',monospace;font-size:.65rem;line-height:2.2;color:#3a5070;">
        <span style="color:#f85149;">●</span> HIGH priority &nbsp;
        <span style="color:#e3b341;">●</span> MEDIUM &nbsp;
        <span style="color:#3fb950;">●</span> LOW
        </div>""", unsafe_allow_html=True)

    with map_col:
        fig_map = go.Figure()
        if show_baseline:
            fig_map.add_trace(go.Scattermap(
                lat=[DEPOT["latitude"]]+ships["latitude"].tolist()+[DEPOT["latitude"]],
                lon=[DEPOT["longitude"]]+ships["longitude"].tolist()+[DEPOT["longitude"]],
                mode="lines", line=dict(width=1.5, color="rgba(248,81,73,0.25)"),
                name="Baseline Route"))
        p_col = {"HIGH":"#f85149","MEDIUM":"#e3b341","LOW":"#3fb950"}
        for v in selected_v:
            vdf   = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
            lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            color = COLORS[(v-1)%len(COLORS)]
            fig_map.add_trace(go.Scattermap(lat=lats,lon=lons,mode="lines",
                                             line=dict(width=2.8,color=color),name=f"Vehicle {v}"))
            for _, row in vdf.iterrows():
                fig_map.add_trace(go.Scattermap(
                    lat=[row["latitude"]], lon=[row["longitude"]], mode="markers",
                    marker=dict(size=10,color=p_col.get(row.get("priority","MEDIUM"),"#e3b341")),
                    hovertext=(f"<b>{row.get('city','')}</b><br>Priority: {row.get('priority','')}<br>"
                               f"Weight: {row['weight']:.0f} kg<br>Cost: ₹{row['total_cost']:,.0f}<br>"
                               f"Carbon: {row['carbon_kg']:.1f} kg<br>SLA breach: {row['sla_breach_hr']:.1f} hr"),
                    hoverinfo="text", showlegend=False))
        fig_map.add_trace(go.Scattermap(lat=[DEPOT["latitude"]],lon=[DEPOT["longitude"]],
            mode="markers+text",text=["Mumbai Depot"],textposition="top right",
            marker=dict(size=18,color="#3b82f6",symbol="star"),name="Depot"))
        fig_map.update_layout(
            map_style="carto-darkmatter",
            map=dict(center=dict(lat=20.5,lon=78.9),zoom=4),
            margin=dict(l=0,r=0,t=0,b=0),height=620,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=.01,y=.99,bgcolor="rgba(7,9,18,.9)",bordercolor="rgba(255,255,255,.08)",
                        borderwidth=1,font=dict(color="#c8d6ee",family="DM Mono",size=11)))
        st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Financial Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "cost":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">💰 What is this page?</div>
      Delivering packages costs money in 3 main ways: <strong>fuel</strong> (petrol/diesel),
      <strong>tolls</strong> (highway booths), and <strong>driver wages</strong>. This page shows
      exactly how much was spent on each — and how much LoRRI saved.
      <div class="eg">Splitting deliveries into smart clusters means fewer total km, less fuel burned,
      fewer toll booths crossed, and drivers finish faster.</div>
    </div>
    """, unsafe_allow_html=True)

    render_kpi_strip()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">Total Cost Components (₹)</div>', unsafe_allow_html=True)
        fig_cost = go.Figure()
        for lbl, col, bk, ok in [
            ("Fuel",  "#3b82f6","baseline_fuel_cost","opt_fuel_cost"),
            ("Toll",  "#e3b341","baseline_toll_cost","opt_toll_cost"),
            ("Driver","#8957e5","baseline_driver_cost","opt_driver_cost")]:
            fig_cost.add_trace(go.Bar(name=lbl,x=["Baseline","Optimized"],
                y=[metrics[bk],metrics[ok]],marker_color=col,marker_line_width=0,
                text=[f"₹{metrics[bk]:,.0f}",f"₹{metrics[ok]:,.0f}"],
                textposition="inside",textfont=dict(color="#f0f6ff",family="DM Mono",size=10)))
        fig_cost.update_layout(barmode="stack",yaxis_title="₹",height=360,
            legend=dict(orientation="h",y=-.2,font=dict(color="#7a9cbf",family="DM Mono"),title_text=""),**PT)
        st.plotly_chart(fig_cost, use_container_width=True)

    with c2:
        st.markdown('<div class="grp">Savings Waterfall — How Much We Saved</div>', unsafe_allow_html=True)
        sav = {
            "Fuel Saved":   metrics["baseline_fuel_cost"]-metrics["opt_fuel_cost"],
            "Toll Saved":   metrics["baseline_toll_cost"]-metrics["opt_toll_cost"],
            "Driver Saved": metrics["baseline_driver_cost"]-metrics["opt_driver_cost"],
        }
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","total"],
            x=list(sav.keys())+["Total Saved"],
            y=list(sav.values())+[sum(sav.values())],
            connector={"line":{"color":"rgba(59,130,246,.2)","width":1}},
            decreasing={"marker":{"color":"#3fb950","line":{"width":0}}},
            totals={"marker":{"color":"#3b82f6","line":{"width":0}}},
            text=[f"₹{v:,.0f}" for v in list(sav.values())+[sum(sav.values())]],
            textposition="outside",textfont=dict(color="#c8d6ee",family="DM Mono",size=11)))
        fig_wf.update_layout(yaxis_title="₹ Saved",height=360,showlegend=False,**PT)
        st.plotly_chart(fig_wf, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="grp">Per-Vehicle Cost Composition</div>', unsafe_allow_html=True)
        fig_vc = go.Figure()
        for cat,col,lbl in [("fuel_cost","#3b82f6","Fuel"),("toll_cost","#e3b341","Toll"),
                             ("driver_cost","#8957e5","Driver"),("sla_penalty","#f85149","SLA Penalty")]:
            fig_vc.add_trace(go.Bar(name=lbl,x=["V"+str(v) for v in veh_summary["vehicle"]],
                y=veh_summary[cat],marker_color=col,marker_line_width=0))
        fig_vc.update_layout(barmode="stack",yaxis_title="₹",height=310,
            legend=dict(orientation="h",y=-.28,font=dict(color="#7a9cbf",family="DM Mono"),title_text=""),**PT)
        st.plotly_chart(fig_vc, use_container_width=True)

    with c4:
        st.markdown('<div class="grp">Savings % by Category</div>', unsafe_allow_html=True)
        cats_pct = ["Fuel","Toll","Driver","Total"]
        bvals = [metrics[k] for k in ["baseline_fuel_cost","baseline_toll_cost","baseline_driver_cost","baseline_total_cost"]]
        ovals = [metrics[k] for k in ["opt_fuel_cost","opt_toll_cost","opt_driver_cost","opt_total_cost"]]
        pcts  = [(b-o)/b*100 for b,o in zip(bvals,ovals)]
        fig_pct = go.Figure(go.Bar(
            y=cats_pct, x=pcts, orientation="h",
            marker_color=["#3b82f6","#e3b341","#8957e5","#3fb950"],marker_line_width=0,
            text=[f"{p:.1f}%" for p in pcts],textposition="outside",
            textfont=dict(color="#c8d6ee",family="DM Mono",size=11)))
        fig_pct.update_layout(xaxis=dict(range=[0,100],**PT["xaxis"]),height=310,**PT)
        st.plotly_chart(fig_pct, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Carbon & Emissions
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "carbon":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">🌿 What is this page?</div>
      <strong>Carbon emissions</strong> = how much CO₂ (pollution) the trucks release. Every litre of diesel
      burned by a truck produces ~2.7 kg CO₂. Shorter, smarter routes = less fuel = less pollution.
      <div class="eg">By grouping nearby stops, LoRRI cut total distance by 67%, which means 67% less fuel
      burned and 67% fewer emissions — equivalent to planting hundreds of trees.</div>
    </div>
    """, unsafe_allow_html=True)

    # Education cards
    ec1, ec2, ec3 = st.columns(3)
    ec1.markdown(f"""
    <div style="background:#080c18;border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.1rem 1.3rem;">
      <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3a5070;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">What causes CO₂?</div>
      <div style="font-size:.82rem;color:#7a9cbf;line-height:1.75;">Every km driven burns fuel and releases <b style="color:#d4e4f7;">~0.27 kg CO₂</b> per km. Total distance is the biggest lever.</div>
    </div>""", unsafe_allow_html=True)
    ec2.markdown(f"""
    <div style="background:#080c18;border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.1rem 1.3rem;">
      <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3a5070;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">How AI reduces it</div>
      <div style="font-size:.82rem;color:#7a9cbf;line-height:1.75;">Clustering nearby stops cuts distance by <b style="color:#3fb950;">67%</b>. Less km = less fuel = less CO₂. Same deliveries, cleaner air.</div>
    </div>""", unsafe_allow_html=True)
    ec3.markdown(f"""
    <div style="background:#080c18;border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.1rem 1.3rem;">
      <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#3a5070;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.6rem;">Our Impact This Run</div>
      <div style="font-size:.82rem;color:#3fb950;line-height:1.75;font-weight:600;"><b style="color:#f0f6ff;font-size:1.1rem;">{_co2:,.0f} kg</b> CO₂ saved<br>≈ {int(_co2/21)} trees planted<br>≈ {int(_co2/4600)} cars off road / year</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">CO₂ Emissions — Baseline vs Optimized</div>', unsafe_allow_html=True)
        fig_co2 = go.Figure(go.Bar(
            x=["Baseline (no AI)","Optimized (AI)"],
            y=[metrics["baseline_carbon_kg"],metrics["opt_carbon_kg"]],
            marker_color=["#f85149","#3fb950"],marker_line_width=0,
            text=[f"{metrics['baseline_carbon_kg']:,.1f} kg",f"{metrics['opt_carbon_kg']:,.1f} kg"],
            textposition="outside",textfont=dict(color="#c8d6ee",family="DM Mono",size=11)))
        fig_co2.add_annotation(x=.5,y=1.08,xref="paper",yref="paper",showarrow=False,
            text=f"<b>{_co2:,.0f} kg saved · {_co2/metrics['baseline_carbon_kg']*100:.1f}% reduction</b>",
            font=dict(color="#3fb950",family="DM Mono",size=11))
        fig_co2.update_layout(yaxis_title="kg CO₂",height=300,showlegend=False,**PT)
        st.plotly_chart(fig_co2, use_container_width=True)

    with c2:
        st.markdown('<div class="grp">Carbon per Vehicle (kg CO₂)</div>', unsafe_allow_html=True)
        fig_cv = go.Figure(go.Bar(
            x=["V"+str(v) for v in veh_summary["vehicle"]],
            y=veh_summary["carbon_kg"],
            marker_color=[COLORS[i%len(COLORS)] for i in range(len(veh_summary))],
            marker_line_width=0,
            text=veh_summary["carbon_kg"].round(1).astype(str)+" kg",
            textposition="outside",textfont=dict(color="#c8d6ee",family="DM Mono",size=10)))
        fig_cv.update_layout(yaxis_title="kg CO₂",height=300,showlegend=False,**PT)
        st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown('<div class="grp">Carbon vs Distance per Shipment</div>', unsafe_allow_html=True)
    fig_sc = px.scatter(routes,x="route_distance_km",y="carbon_kg",color="priority",size="weight",
        hover_name="city",
        color_discrete_map={"HIGH":"#f85149","MEDIUM":"#e3b341","LOW":"#3fb950"},
        labels={"route_distance_km":"Route Distance (km)","carbon_kg":"Carbon Emitted (kg CO₂)"},height=320)
    fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans",color="#7a9cbf"),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        legend=dict(font=dict(color="#7a9cbf",family="DM Mono"),title_text="Priority"))
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SLA Performance
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "sla":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">✅ What is this page?</div>
      <strong>SLA (Service Level Agreement)</strong> is the delivery promise made to each customer —
      e.g., "your package arrives within 48 hours." A breach means the truck arrived late.
      This page tracks how well we kept our promises.
      <div class="eg">Think of a school pick-up: you promise to arrive in 1 hour. If you take 2 hours,
      you breached the SLA. The gauge below shows how often our trucks kept their promise (higher = better).</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="grp">SLA Adherence Gauge</div>', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=metrics["opt_sla_adherence_pct"],
            title={"text":"Delivery Promise Kept","font":{"family":"Syne","color":"#f0f6ff","size":14}},
            number={"font":{"family":"Syne","color":"#f0f6ff","size":52},"suffix":"%"},
            delta={"reference":metrics["baseline_sla_adherence_pct"],
                   "increasing":{"color":"#3fb950"},
                   "font":{"family":"DM Mono","size":13},"suffix":"% vs baseline"},
            gauge={"axis":{"range":[0,100],"tickcolor":"#3a5070","tickfont":{"family":"DM Mono","size":10}},
                   "bar":{"color":"#3b82f6","thickness":.28},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                   "steps":[{"range":[0,60],"color":"rgba(248,81,73,.1)"},
                             {"range":[60,80],"color":"rgba(227,179,65,.1)"},
                             {"range":[80,100],"color":"rgba(63,185,80,.1)"}],
                   "threshold":{"line":{"color":"#f85149","width":2},"thickness":.7,
                                "value":metrics["baseline_sla_adherence_pct"]}}))
        fig_g.update_layout(height=320,paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Plus Jakarta Sans",color="#7a9cbf"))
        st.plotly_chart(fig_g, use_container_width=True)

    with c2:
        st.markdown('<div class="grp">Late Deliveries — Vehicle × Priority</div>', unsafe_allow_html=True)
        breach_df = routes.copy()
        breach_df["breached"] = (breach_df["sla_breach_hr"]>0).astype(int)
        pivot = breach_df.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_h = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(),
            y=["V"+str(v) for v in pivot.index],
            colorscale=[[0,"rgba(59,130,246,.05)"],[.5,"rgba(227,179,65,.5)"],[1,"rgba(248,81,73,.85)"]],
            text=pivot.values, texttemplate="%{text}",
            textfont=dict(color="#f0f6ff",family="DM Mono",size=14),
            colorbar=dict(title="Breaches",titlefont=dict(color="#7a9cbf"),tickfont=dict(color="#7a9cbf",family="DM Mono"))))
        fig_h.update_layout(xaxis_title="Priority",yaxis_title="Vehicle",height=320,**PT)
        st.plotly_chart(fig_h, use_container_width=True)

    # SLA summary table
    st.markdown('<div class="grp">SLA Breach Detail</div>', unsafe_allow_html=True)
    breached_df = routes[routes["sla_breach_hr"]>0][
        ["city","vehicle","priority","sla_hours","sla_breach_hr","weight","total_cost"]
    ].sort_values("sla_breach_hr",ascending=False).reset_index(drop=True)
    if len(breached_df):
        st.dataframe(breached_df.style.format({
            "sla_hours":"{:.0f} hr","sla_breach_hr":"{:.1f} hr",
            "weight":"{:.0f} kg","total_cost":"₹{:,.0f}"}
        ).background_gradient(subset=["sla_breach_hr"],cmap="YlOrRd"),
        use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="cb cb-g"><b>✓ Zero SLA breaches</b> — perfect delivery adherence!</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "explainability":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">🧠 What is this page?</div>
      Every time the AI picked the <em>next city to deliver to</em>, it scored each candidate on
      4 factors: <strong>time, cost, carbon, and SLA risk</strong>. This page reveals which factors
      actually drove the decisions — calculated by scrambling each factor and measuring score change
      (<strong>permutation importance</strong>, same technique used in production ML systems like SHAP).
      <div class="eg">You choose which homework to do first: time (30%), difficulty (35%), boredom (20%),
      deadline urgency (15%). That's exactly what our AI does — but for cities!</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown('<div class="grp">Objective Weights</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Cost (₹)","Travel Time","Carbon CO₂","SLA Adherence"],
            values=[35,30,20,15], hole=.62,
            marker_colors=["#3b82f6","#e3b341","#3fb950","#f85149"],
            textinfo="label+percent",
            textfont=dict(family="DM Mono",size=11,color="#c8d6ee")))
        fig_pie.update_layout(height=300,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text":"Weights","x":.5,"y":.5,
                "font":{"size":13,"color":"#f0f6ff","family":"Syne"},"showarrow":False}])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<div class="grp">Real Feature Importance (Permutation-Based)</div>', unsafe_allow_html=True)
        fi_l = list(feature_importance.keys())
        fi_v = list(feature_importance.values())
        mv   = max(fi_v)
        fig_fi = go.Figure(go.Bar(
            x=fi_v, y=fi_l, orientation="h",
            marker_color=["#f85149" if v==mv else "#3b82f6" for v in fi_v],
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in fi_v],textposition="outside",
            textfont=dict(color="#c8d6ee",family="DM Mono",size=11)))
        fig_fi.update_layout(title="Which factor drove routing decisions the most?",
            xaxis_title="Importance (%)",
            yaxis=dict(autorange="reversed",**PT["yaxis"]),height=300,**PT)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="grp">Per-Stop Score Contribution Breakdown</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cb cb-b">
    Each bar = one delivery stop. Colours show which factor added most to that stop's difficulty score.
    Tall red = SLA pressure. Tall blue = fuel cost dominated.
    </div>""", unsafe_allow_html=True)

    vcol_a, vcol_b = st.columns([3,1])
    with vcol_b:
        veh_filter = st.selectbox("Vehicle:", ["All"]+[f"V{v}" for v in sorted(routes["vehicle"].unique())])
    sc_df = stop_contrib if veh_filter=="All" else \
            stop_contrib[stop_contrib["vehicle"]==int(veh_filter[1:])]

    fig_st = go.Figure()
    fc_pairs = [("Travel Time","#e3b341"),("Fuel Cost","#3b82f6"),("Toll Cost","#8957e5"),
                ("Driver Cost","#bc8cff"),("Carbon","#3fb950"),("SLA Breach","#f85149")]
    for fc, col in fc_pairs:
        fig_st.add_trace(go.Bar(name=fc,x=sc_df["city"],y=sc_df[fc],marker_color=col,marker_line_width=0))
    fig_st.update_layout(barmode="stack",xaxis_tickangle=-45,yaxis_title="Weighted MO Contribution",height=360,
        legend=dict(orientation="h",y=-.35,font=dict(color="#7a9cbf",family="DM Mono",size=11),title_text=""),**PT)
    st.plotly_chart(fig_st, use_container_width=True)

    st.markdown('<div class="grp">Top 10 Hardest-to-Schedule Stops</div>', unsafe_allow_html=True)
    st.markdown('<div class="cb cb-b">High AI score = hard routing decision — due to distance, traffic, tight SLA, or high emissions.</div>', unsafe_allow_html=True)
    top10 = routes.nlargest(10,"mo_score")[
        ["city","vehicle","priority","weight","travel_time_hr","fuel_cost","toll_cost","carbon_kg","sla_breach_hr","mo_score"]
    ].reset_index(drop=True)
    st.dataframe(top10.style.format({
        "travel_time_hr":"{:.2f} hr","fuel_cost":"₹{:.0f}","toll_cost":"₹{:.0f}",
        "carbon_kg":"{:.2f} kg","sla_breach_hr":"{:.1f} hr","mo_score":"{:.4f}","weight":"{:.0f} kg"}
    ).background_gradient(subset=["mo_score"],cmap="YlOrRd"),
    use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Re-Optimization Engine
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "reopt":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">⚡ What is this page?</div>
      The real world is messy — traffic jams happen, customers demand urgent delivery.
      This page lets you <strong>simulate disruptions</strong> and watch LoRRI instantly re-plan
      the affected truck's route in real time.
      <div class="eg">Delhi → Agra → Jaipur. Traffic jam in Agra. AI: "Skip Agra, do Jaipur first,
      come back later." That's re-optimization — same stops, smarter order.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="grp">🚦 Scenario 1 — Traffic Jam</div>', unsafe_allow_html=True)
        st.markdown('<div class="cb cb-y">Pick a city and raise its traffic. If travel time increases &gt;30%, AI triggers a re-route.</div>', unsafe_allow_html=True)
        disrupted_city = st.selectbox("City hit by traffic jam:", options=sorted(ships["city"].tolist()))
        traffic_spike  = st.slider("Traffic multiplier (1.0 = clear · 3.0 = gridlock)", 1.0, 3.0, 2.5, 0.1)
        if st.button("🔴 Trigger Traffic Disruption", use_container_width=True):
            orig_row  = ships[ships["city"]==disrupted_city].iloc[0]
            orig_mult = orig_row["traffic_mult"]
            dist_to   = haversine(DEPOT["latitude"],DEPOT["longitude"],orig_row["latitude"],orig_row["longitude"])
            orig_time = dist_to/(55/orig_mult)
            new_time  = dist_to/(55/traffic_spike)
            time_inc  = (new_time-orig_time)/orig_time*100
            if time_inc > 30:
                st.markdown(f'<div class="cb cb-r"><b>⚠ Threshold Breached — Re-optimizing</b><br>{disrupted_city} · {orig_mult:.2f}x → <b>{traffic_spike:.2f}x</b> · +<b>{time_inc:.1f}%</b> travel time</div>', unsafe_allow_html=True)
                with st.spinner("Re-optimizing route…"):
                    time.sleep(1.0)
                affected = routes[routes["city"]==disrupted_city]["vehicle"].values
                if len(affected):
                    vid   = affected[0]
                    orig  = routes[routes["vehicle"]==vid].sort_values("stop_order")
                    mask  = orig["city"]==disrupted_city
                    reopt = pd.concat([orig[~mask],orig[mask]]).reset_index(drop=True)
                    def sd(df): return sum(haversine(df.iloc[i]["latitude"],df.iloc[i]["longitude"],df.iloc[i+1]["latitude"],df.iloc[i+1]["longitude"]) for i in range(len(df)-1))
                    od,nd = sd(orig),sd(reopt)
                    st.markdown(f'<div class="cb cb-g"><b>✓ Vehicle {vid} re-routed</b> — {disrupted_city} moved to last stop.</div>', unsafe_allow_html=True)
                    ca,cb2 = st.columns(2)
                    ca.metric("Original route",f"{od:.1f} km")
                    cb2.metric("Re-optimized",f"{nd:.1f} km",delta=f"{nd-od:+.1f} km",delta_color="inverse")
                    st.dataframe(reopt[["city","priority","weight","sla_hours"]].reset_index(drop=True),use_container_width=True,hide_index=True)
            else:
                st.markdown(f'<div class="cb cb-g"><b>✓ Within threshold</b> — no re-route needed (+{time_inc:.1f}%, threshold: 30%)</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="grp">🚨 Scenario 2 — Customer Escalation</div>', unsafe_allow_html=True)
        st.markdown('<div class="cb cb-y">A customer demands urgent delivery. AI promotes their city to stop #1.</div>', unsafe_allow_html=True)
        escalate_city = st.selectbox("Which city escalated to urgent?", options=sorted(ships["city"].tolist()), key="esc")
        if st.button("🔴 Trigger Priority Escalation", use_container_width=True):
            orig_p = ships[ships["city"]==escalate_city]["priority"].values[0]
            if orig_p=="HIGH":
                st.markdown(f'<div class="cb cb-g"><b>✓ Already HIGH priority</b> — no change needed.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Re-routing for urgent delivery…"):
                    time.sleep(0.8)
                affected = routes[routes["city"]==escalate_city]["vehicle"].values
                vid      = affected[0] if len(affected) else 1
                orig_r   = routes[routes["vehicle"]==vid].sort_values("stop_order")
                mask     = orig_r["city"]==escalate_city
                new_r    = pd.concat([orig_r[mask],orig_r[~mask]]).reset_index(drop=True)
                st.markdown(f'<div class="cb cb-g"><b>✓ {escalate_city} → stop #1 on Vehicle {vid}</b> · SLA tightened to 24h · Priority: {orig_p} → HIGH</div>', unsafe_allow_html=True)
                ca,cb2 = st.columns(2)
                ca.metric("Old SLA",f"{ships[ships['city']==escalate_city]['sla_hours'].values[0]}h")
                cb2.metric("New SLA","24h",delta="Urgent")
                st.dataframe(new_r[["city","priority","weight","sla_hours"]].reset_index(drop=True),use_container_width=True,hide_index=True)

    # Risk monitor
    st.markdown('<div class="grp">📊 Live Re-Optimization Risk Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="cb cb-b">Cities most likely to trigger a re-route right now, based on traffic level and delivery urgency. Red bars = immediate action required.</div>', unsafe_allow_html=True)
    tdf = ships[["city","traffic_mult","priority","sla_hours"]].copy()
    tdf["risk"] = (tdf["traffic_mult"]/1.8*0.6 + tdf["sla_hours"].map({24:1.0,48:0.5,72:0.2}).fillna(0.2)*0.4).round(3)
    tdf["status"] = tdf["risk"].apply(lambda x:"HIGH RISK" if x>.7 else ("MONITOR" if x>.4 else "STABLE"))
    tdf = tdf.sort_values("risk",ascending=False)
    fig_r = px.bar(tdf.head(15),x="city",y="risk",color="status",
        color_discrete_map={"HIGH RISK":"#f85149","MONITOR":"#e3b341","STABLE":"#3fb950"},
        labels={"risk":"Risk Score","city":"City"},height=320)
    fig_r.add_hline(y=.7,line_dash="dash",line_color="#f85149",line_width=1.5,
        annotation_text="Re-optimize triggers above this line",
        annotation_font=dict(color="#f85149",family="DM Mono",size=11))
    fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans",color="#7a9cbf",size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)",tickangle=-30),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        legend=dict(font=dict(color="#7a9cbf",family="DM Mono"),title_text=""))
    st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Assistant (RAG)
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "rag":
    st.markdown("""
    <div class="info-box">
      <div class="ib-head">🤖 What is this page?</div>
      Ask questions in plain English about your routes, costs, carbon, or vehicles —
      and get <strong>instant, data-grounded answers</strong>. The system retrieves the most relevant
      data chunks before answering, so responses are based on your <em>actual</em> CSV data — no guessing.
      <div class="eg">Ask: "Which vehicle has the most SLA breaches?" or "How much CO₂ did Vehicle 3 emit?"
      — LoRRI AI finds the answer in your data, not from its training.</div>
    </div>
    """, unsafe_allow_html=True)

    chat_col, side_col = st.columns([5,2])

    with side_col:
        st.markdown('<div class="grp">Knowledge Base</div>', unsafe_allow_html=True)
        from rag_engine import _build_kb
        kb_docs, _, _ = _build_kb()
        kb_html = "".join(f'<div class="kb-entry"><span>▸</span>{d["title"]}</div>' for d in kb_docs[:14])
        if len(kb_docs)>14:
            kb_html += f'<div style="font-family:DM Mono,monospace;font-size:.6rem;color:#1a2d3f;margin-top:4px;">+{len(kb_docs)-14} more chunks indexed</div>'
        st.markdown(kb_html, unsafe_allow_html=True)

        st.markdown('<div class="grp">Quick Questions</div>', unsafe_allow_html=True)
        show_sources = st.toggle("Show sources", value=True)
        quick_qs = [
            ("💰","What are total cost savings?"),
            ("🌿","How much CO₂ was saved?"),
            ("🚛","Which vehicle emits most carbon?"),
            ("⏰","Which vehicle has most SLA breaches?"),
            ("📍","Which cities are HIGH priority?"),
            ("⛽","What is fuel cost per vehicle?"),
            ("🧠","Explain how MO score works"),
            ("🏙️","Which cities are hardest to route?"),
        ]
        for icon, qq in quick_qs:
            if st.button(f"{icon} {qq}", key=f"qq_{qq[:12]}", use_container_width=True):
                if not hf_key:
                    st.warning("Enter your HuggingFace API key in the sidebar first.")
                else:
                    with st.spinner("Thinking…"):
                        answer, sources = get_rag_response(qq, history=st.session_state.rag_messages[1:])
                    st.session_state.rag_messages.append({"role":"user","content":qq})
                    st.session_state.rag_messages.append({"role":"assistant","content":answer})
                    st.session_state.rag_sources.append(sources)
                    st.rerun()

        if len(st.session_state.rag_messages) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear conversation", use_container_width=True):
                st.session_state.rag_messages = [st.session_state.rag_messages[0]]
                st.session_state.rag_sources  = []
                st.rerun()

    with chat_col:
        st.markdown('<div class="grp">Conversation</div>', unsafe_allow_html=True)

        chat_html = ""
        src_idx   = 0
        for msg in st.session_state.rag_messages:
            content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', msg['content'])
            content = content.replace('\n','<br>')
            if msg["role"]=="user":
                chat_html += f"""
                <div class="msg-row you">
                  <div class="av av-u">YOU</div>
                  <div class="bub u"><div class="mn mn-u">YOU</div>{content}</div>
                </div>"""
            else:
                src_html = ""
                if show_sources and src_idx < len(st.session_state.rag_sources):
                    chips = "".join(f'<span class="chip">{s}</span>' for s in st.session_state.rag_sources[src_idx])
                    src_html = f'<div class="src-row">{chips}</div>'
                    src_idx += 1
                chat_html += f"""
                <div class="msg-row">
                  <div class="av av-b">AI</div>
                  <div class="bub b"><div class="mn mn-b">LORRI AI · LLAMA 3.1-8B</div>{content}{src_html}</div>
                </div>"""

        if len(st.session_state.rag_messages) <= 1:
            chat_html = """
            <div class="chat-empty">
              <div style="font-size:3rem;margin-bottom:1rem;opacity:.4;">💬</div>
              Ask anything about your delivery run.<br>
              LoRRI AI retrieves your actual data before answering.<br>
              <span style="color:#3b82f6;opacity:.4;font-size:.68rem;">Powered by Llama 3.1-8B via HuggingFace</span>
            </div>"""

        st.markdown(f'<div class="chat-wrap">{chat_html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not hf_key:
            st.markdown('<div class="cb cb-y"><b>Enter your HuggingFace API key in the sidebar</b> to start chatting. Get one free at huggingface.co/settings/tokens</div>', unsafe_allow_html=True)
        else:
            user_input = st.chat_input("Ask about routes, costs, carbon, SLA, vehicles…")
            if user_input:
                with st.spinner("Retrieving context · Generating answer…"):
                    answer, sources = get_rag_response(user_input, history=st.session_state.rag_messages[1:])
                st.session_state.rag_messages.append({"role":"user","content":user_input})
                st.session_state.rag_messages.append({"role":"assistant","content":answer})
                st.session_state.rag_sources.append(sources)
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,.04);
     font-family:'DM Mono',monospace;font-size:.58rem;color:#1a2d3f;
     display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
  <span>LoRRI · Route Intelligence Platform · v2.1 Enterprise</span>
  <span>CVRP · Multi-Objective · TF-IDF RAG · Llama 3.1-8B · Streamlit + Plotly</span>
  <span>© 2026 LoRRI Technologies · Confidential</span>
</div>
""", unsafe_allow_html=True)
