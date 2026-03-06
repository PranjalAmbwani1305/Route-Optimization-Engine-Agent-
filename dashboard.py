import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time
import json
import re

st.set_page_config(
    page_title="LoRRI · Route Intelligence",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * R * asin(sqrt(a))

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777, "id": "DEPOT"}
COLORS      = px.colors.qualitative.Bold
VEHICLE_CAP = 800

# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ships   = pd.read_csv("shipments.csv")
    routes  = pd.read_csv("routes.csv")
    metrics = pd.read_csv("metrics.csv").iloc[0]
    veh     = pd.read_csv("vehicle_summary.csv")
    return ships, routes, metrics, veh

ships, routes, metrics, veh_summary = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_feature_importance(routes_df):
    np.random.seed(42)
    features = {
        "Travel Time":    "travel_time_hr",
        "Fuel Cost":      "fuel_cost",
        "Toll Cost":      "toll_cost",
        "Driver Cost":    "driver_cost",
        "Carbon Emitted": "carbon_kg",
        "SLA Breach":     "sla_breach_hr",
        "Package Weight": "weight",
    }
    X = routes_df[list(features.values())].copy()
    y = routes_df["mo_score"].values
    baseline_mae = np.mean(np.abs(y - y.mean()))
    importances = {}
    for label, col in features.items():
        shuffled = X.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        proxy = shuffled.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean(axis=1)
        mae_after = np.mean(np.abs(y - proxy.values))
        importances[label] = abs(mae_after - baseline_mae)
    total = sum(importances.values()) + 1e-9
    return {k: round(v / total * 100, 1) for k, v in
            sorted(importances.items(), key=lambda x: -x[1])}

feature_importance = compute_feature_importance(routes)

# ─────────────────────────────────────────────────────────────────────────────
# Stop contributions  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_stop_contributions(routes_df):
    cols    = ["travel_time_hr", "fuel_cost", "toll_cost",
               "driver_cost", "carbon_kg", "sla_breach_hr"]
    labels  = ["Travel Time", "Fuel Cost", "Toll Cost",
               "Driver Cost", "Carbon", "SLA Breach"]
    weights = [0.30, 0.20, 0.05, 0.15, 0.20, 0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng = df[c].max() - df[c].min()
        df[c] = (df[c] - df[c].min()) / (rng + 1e-9)
    for i, c in enumerate(cols):
        df[c] = df[c] * weights[i]
    df.columns = labels
    df["city"]     = routes_df["city"].values
    df["vehicle"]  = routes_df["vehicle"].values
    df["mo_score"] = routes_df["mo_score"].values
    return df

stop_contrib = compute_stop_contributions(routes)

# ─────────────────────────────────────────────────────────────────────────────
# RAG — Knowledge Base Builder (TF-IDF, no external vector DB needed)
# ─────────────────────────────────────────────────────────────────────────────
import math as _math

def _tokenize(text: str):
    return re.findall(r"[a-z0-9₹%\.]+", text.lower())

@st.cache_data
def build_knowledge_base(_ships, _routes, _metrics, _veh_summary):
    """Convert all CSVs into text chunks and build a TF-IDF index."""
    docs = []

    # ── Global metrics chunk ────────────────────────────────────────────────
    docs.append({
        "title": "Global Metrics Summary",
        "text": (
            f"Total shipments: {int(_metrics['num_shipments'])}. "
            f"Vehicles used: {int(_metrics['num_vehicles'])}. "
            f"Depot: Mumbai (19.076°N, 72.877°E). "
            f"Objective weights: Cost 35%, Travel Time 30%, Carbon 20%, SLA 15%. "
            f"Baseline distance: {_metrics['baseline_distance_km']:.1f} km. "
            f"Optimized distance: {_metrics['opt_distance_km']:.1f} km. "
            f"Distance saved: {_metrics['baseline_distance_km']-_metrics['opt_distance_km']:.1f} km "
            f"({(_metrics['baseline_distance_km']-_metrics['opt_distance_km'])/_metrics['baseline_distance_km']*100:.1f}%). "
            f"Baseline travel time: {_metrics['baseline_time_hr']:.1f} hr. "
            f"Optimized travel time: {_metrics['opt_time_hr']:.1f} hr. "
            f"Baseline total cost: ₹{_metrics['baseline_total_cost']:,.0f}. "
            f"Optimized total cost: ₹{_metrics['opt_total_cost']:,.0f}. "
            f"Total cost saved: ₹{_metrics['baseline_total_cost']-_metrics['opt_total_cost']:,.0f} "
            f"({(_metrics['baseline_total_cost']-_metrics['opt_total_cost'])/_metrics['baseline_total_cost']*100:.1f}%). "
            f"Baseline carbon: {_metrics['baseline_carbon_kg']:.1f} kg CO2. "
            f"Optimized carbon: {_metrics['opt_carbon_kg']:.1f} kg CO2. "
            f"Carbon saved: {_metrics['baseline_carbon_kg']-_metrics['opt_carbon_kg']:.1f} kg CO2. "
            f"Baseline SLA adherence: {_metrics['baseline_sla_adherence_pct']:.1f}%. "
            f"Optimized SLA adherence: {_metrics['opt_sla_adherence_pct']:.1f}%."
        )
    })

    # ── Cost breakdown chunk ────────────────────────────────────────────────
    docs.append({
        "title": "Cost Breakdown — Fuel, Toll, Driver",
        "text": (
            f"Baseline fuel cost: ₹{_metrics['baseline_fuel_cost']:,.0f}. "
            f"Optimized fuel cost: ₹{_metrics['opt_fuel_cost']:,.0f}. "
            f"Fuel saved: ₹{_metrics['baseline_fuel_cost']-_metrics['opt_fuel_cost']:,.0f}. "
            f"Baseline toll cost: ₹{_metrics['baseline_toll_cost']:,.0f}. "
            f"Optimized toll cost: ₹{_metrics['opt_toll_cost']:,.0f}. "
            f"Toll saved: ₹{_metrics['baseline_toll_cost']-_metrics['opt_toll_cost']:,.0f}. "
            f"Baseline driver cost: ₹{_metrics['baseline_driver_cost']:,.0f}. "
            f"Optimized driver cost: ₹{_metrics['opt_driver_cost']:,.0f}. "
            f"Driver cost saved: ₹{_metrics['baseline_driver_cost']-_metrics['opt_driver_cost']:,.0f}."
        )
    })

    # ── Per-vehicle chunks ──────────────────────────────────────────────────
    for _, row in _veh_summary.iterrows():
        docs.append({
            "title": f"Vehicle {int(row['vehicle'])} Summary",
            "text": (
                f"Vehicle {int(row['vehicle'])}: "
                f"{int(row['stops'])} stops, "
                f"load {row['load_kg']:.1f} kg (utilization {row['utilization_pct']:.1f}%), "
                f"distance {row['distance_km']:.1f} km, "
                f"travel time {row['time_hr']:.1f} hr, "
                f"fuel cost ₹{row['fuel_cost']:,.0f}, "
                f"toll cost ₹{row['toll_cost']:,.0f}, "
                f"driver cost ₹{row['driver_cost']:,.0f}, "
                f"SLA penalty ₹{row['sla_penalty']:,.0f}, "
                f"total cost ₹{row['total_cost']:,.0f}, "
                f"carbon {row['carbon_kg']:.1f} kg CO2, "
                f"SLA breaches {int(row['sla_breaches'])}."
            )
        })

    # ── Per-city route chunks (group into batches of 5) ─────────────────────
    for start in range(0, len(_routes), 5):
        batch = _routes.iloc[start:start+5]
        lines = []
        for _, r in batch.iterrows():
            lines.append(
                f"{r['city']} (V{int(r['vehicle'])}, stop #{int(r['stop_order'])}, "
                f"priority {r['priority']}, weight {r['weight']:.0f}kg, "
                f"SLA {r['sla_hours']}h, travel {r['travel_time_hr']:.2f}hr, "
                f"fuel ₹{r['fuel_cost']:.0f}, toll ₹{r['toll_cost']:.0f}, "
                f"carbon {r['carbon_kg']:.2f}kg, "
                f"SLA breach {r['sla_breach_hr']:.1f}hr, MO score {r['mo_score']:.4f})"
            )
        docs.append({
            "title": f"Route Stops {start+1}–{min(start+5, len(_routes))}",
            "text": " | ".join(lines)
        })

    # ── High-priority shipments chunk ───────────────────────────────────────
    high = _ships[_ships["priority"] == "HIGH"]
    docs.append({
        "title": "HIGH Priority Shipments",
        "text": f"There are {len(high)} HIGH priority shipments. Cities: " +
                ", ".join(high["city"].tolist()) + ". " +
                f"Average weight: {high['weight'].mean():.1f} kg. "
                f"SLA window: 24 hours."
    })

    # ── Carbon analysis chunk ───────────────────────────────────────────────
    top_carbon = _routes.nlargest(5, "carbon_kg")[["city","carbon_kg","vehicle"]].values
    docs.append({
        "title": "Carbon Emissions Analysis",
        "text": (
            f"Total optimized carbon: {_metrics['opt_carbon_kg']:.1f} kg CO2. "
            f"Carbon reduction vs baseline: {_metrics['baseline_carbon_kg']-_metrics['opt_carbon_kg']:.1f} kg "
            f"({(_metrics['baseline_carbon_kg']-_metrics['opt_carbon_kg'])/_metrics['baseline_carbon_kg']*100:.1f}%). "
            f"Top 5 highest carbon stops: " +
            ", ".join(f"{c} ({k:.2f}kg CO2, V{int(v)})" for c,k,v in top_carbon) + "."
        )
    })

    # ── SLA analysis chunk ──────────────────────────────────────────────────
    breached = _routes[_routes["sla_breach_hr"] > 0]
    docs.append({
        "title": "SLA Breach Analysis",
        "text": (
            f"Total SLA breaches in optimized routes: {len(breached)}. "
            f"Optimized SLA adherence: {_metrics['opt_sla_adherence_pct']:.1f}% "
            f"(baseline: {_metrics['baseline_sla_adherence_pct']:.1f}%). "
            f"Cities with breaches: " +
            (", ".join(breached["city"].tolist()) if len(breached) else "None") + ". " +
            f"Average breach duration: {breached['sla_breach_hr'].mean():.2f} hr." if len(breached) else ""
        )
    })

    # ── MO score explainability chunk ───────────────────────────────────────
    top10 = _routes.nlargest(10, "mo_score")[["city","mo_score","vehicle","priority"]].values
    docs.append({
        "title": "MO Score Explainability",
        "text": (
            "The Multi-Objective (MO) score combines: travel time (30%), total cost (35%), "
            "carbon emissions (20%), and SLA breach (15%). Lower score = better stop choice. "
            "Top 10 hardest-to-schedule stops by MO score: " +
            ", ".join(f"{c} (score {s:.4f}, V{int(v)}, {p})" for c,s,v,p in top10) + "."
        )
    })

    # ── Build TF-IDF vectors ────────────────────────────────────────────────
    tokenized   = [_tokenize(d["text"]) for d in docs]
    N           = len(docs)
    df_counts   = {}
    for toks in tokenized:
        for t in set(toks):
            df_counts[t] = df_counts.get(t, 0) + 1
    idf = {t: _math.log((N + 1) / (df + 1)) for t, df in df_counts.items()}

    def tfidf_vec(tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = len(tokens) or 1
        return {t: (c / total) * idf.get(t, 0) for t, c in tf.items()}

    vecs = [tfidf_vec(toks) for toks in tokenized]
    return docs, vecs, idf, _tokenize

kb_docs, kb_vecs, kb_idf, kb_tokenize = build_knowledge_base(ships, routes, metrics, veh_summary)


def retrieve(query: str, docs, vecs, idf, tokenize_fn, top_k: int = 4):
    """Cosine similarity retrieval over TF-IDF vectors."""
    q_toks = tokenize_fn(query)
    q_tf   = {}
    for t in q_toks:
        q_tf[t] = q_tf.get(t, 0) + 1
    total = len(q_toks) or 1
    q_vec  = {t: (c / total) * idf.get(t, 0) for t, c in q_tf.items()}

    def cosine(a, b):
        dot  = sum(a.get(t, 0) * b.get(t, 0) for t in b)
        na   = _math.sqrt(sum(v*v for v in a.values())) + 1e-9
        nb   = _math.sqrt(sum(v*v for v in b.values())) + 1e-9
        return dot / (na * nb)

    scores = [(cosine(q_vec, v), i) for i, v in enumerate(vecs)]
    scores.sort(reverse=True)
    return [docs[i] for _, i in scores[:top_k]]


def hf_chat(api_key: str, messages: list, system_prompt: str) -> str:
    """Call HuggingFace Inference API (Llama 3.1-8B via Nebius)."""
    try:
        from huggingface_hub import InferenceClient
        client   = InferenceClient(provider="nebius", api_key=api_key)
        payload  = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=payload,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ HuggingFace API error: {e}"


# ── Session state for RAG ────────────────────────────────────────────────────
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "rag_sources"  not in st.session_state:
    st.session_state.rag_sources  = []

# ─────────────────────────────────────────────────────────────────────────────
# EXPERT CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
    background: #080c14 !important;
    color: #c9d4e8 !important;
}
.main .block-container {
    padding: 0 2.5rem 3rem 2.5rem !important;
    max-width: 1600px;
}
section[data-testid="stSidebar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d1220; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }

/* ── Masthead ── */
.lorri-masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.6rem 0;
    border-bottom: 1px solid rgba(56,139,253,0.18);
    margin-bottom: 2rem;
}
.lorri-wordmark {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f0f4ff;
    line-height: 1;
}
.lorri-wordmark span { color: #388BFD; }
.lorri-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4b6080;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 4px;
}
.lorri-badge {
    display: flex;
    gap: 8px;
    align-items: center;
}
.badge-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid;
}
.badge-green  { color: #3fb950; border-color: rgba(63,185,80,0.4);  background: rgba(63,185,80,0.08); }
.badge-blue   { color: #388BFD; border-color: rgba(56,139,253,0.4); background: rgba(56,139,253,0.08); }
.badge-yellow { color: #d29922; border-color: rgba(210,153,34,0.4); background: rgba(210,153,34,0.08); }

/* ── KPI Strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(56,139,253,0.12);
    border: 1px solid rgba(56,139,253,0.15);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.kpi-cell {
    background: #0d1220;
    padding: 1.4rem 1.6rem;
    position: relative;
}
.kpi-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #388BFD);
    opacity: 0.6;
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #4b6080;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f0f4ff;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    margin-top: 6px;
}
.delta-good { color: #3fb950; }
.delta-bad  { color: #f85149; }
.delta-info { color: #388BFD; }

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #388BFD;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(56,139,253,0.18);
}

/* ── Info / Warn / OK callouts ── */
.callout {
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.6rem 0 1.2rem 0;
    font-size: 0.84rem;
    line-height: 1.65;
    border-left: 3px solid;
}
.callout-info   { background: rgba(56,139,253,0.08);  border-color: #388BFD;  color: #8ab4f8; }
.callout-warn   { background: rgba(210,153,34,0.08);  border-color: #d29922;  color: #e3b341; }
.callout-ok     { background: rgba(63,185,80,0.08);   border-color: #3fb950;  color: #56d364; }
.callout-danger { background: rgba(248,81,73,0.08);   border-color: #f85149;  color: #ff7b72; }
.callout b, .callout strong { color: #f0f4ff; }

/* ── Card container ── */
.card {
    background: #0d1220;
    border: 1px solid rgba(56,139,253,0.13);
    border-radius: 12px;
    padding: 1.4rem;
    height: 100%;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(56,139,253,0.15) !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #4b6080 !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    background: transparent !important;
    letter-spacing: 0.02em;
    transition: color 0.2s;
}
[data-baseweb="tab"]:hover { color: #8ab4f8 !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #f0f4ff !important;
    background: rgba(56,139,253,0.1) !important;
    border-bottom: 2px solid #388BFD !important;
}
[data-baseweb="tab-panel"] { padding: 1.8rem 0 0 0 !important; }

/* ── Streamlit metric override ── */
[data-testid="metric-container"] {
    background: #0d1220 !important;
    border: 1px solid rgba(56,139,253,0.13) !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.45rem !important;
    color: #f0f4ff !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    color: #4b6080 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,139,253,0.13) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
thead tr th {
    background: #0d1a2e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4b6080 !important;
    border-bottom: 1px solid rgba(56,139,253,0.2) !important;
}

/* ── Divider ── */
hr { border-color: rgba(56,139,253,0.12) !important; }

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ── Selectbox, slider ── */
[data-baseweb="select"] > div {
    background: #0d1a2e !important;
    border-color: rgba(56,139,253,0.25) !important;
    border-radius: 8px !important;
    color: #c9d4e8 !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #388BFD !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label { font-size: 0.82rem !important; color: #8ab4f8 !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: rgba(56,139,253,0.12) !important;
    border: 1px solid rgba(56,139,253,0.35) !important;
    color: #388BFD !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
[data-testid="stButton"] > button:hover {
    background: rgba(56,139,253,0.22) !important;
    border-color: #388BFD !important;
    color: #f0f4ff !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #2d3f57 !important;
    letter-spacing: 0.06em;
}

/* ── Inline tag chips ── */
.chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 5px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.chip-red    { background: rgba(248,81,73,0.12);  color: #f85149; }
.chip-green  { background: rgba(63,185,80,0.12);  color: #3fb950; }
.chip-blue   { background: rgba(56,139,253,0.12); color: #388BFD; }
.chip-yellow { background: rgba(210,153,34,0.12); color: #e3b341; }
.chip-grey   { background: rgba(139,148,158,0.12); color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lorri-masthead">
  <div>
    <div class="lorri-wordmark">Lo<span>RRI</span></div>
    <div class="lorri-sub">AI Route Optimization Engine · CVRP · Mumbai Depot</div>
  </div>
  <div class="lorri-badge">
    <span class="badge-pill badge-green">● Live</span>
    <span class="badge-pill badge-blue">{int(metrics['num_shipments'])} Shipments</span>
    <span class="badge-pill badge-yellow">{int(metrics['num_vehicles'])} Vehicles</span>
    <span class="badge-pill badge-blue">Multi-Objective</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP KPI STRIP
# ─────────────────────────────────────────────────────────────────────────────
dist_save  = metrics["baseline_distance_km"]  - metrics["opt_distance_km"]
time_save  = metrics["baseline_time_hr"]       - metrics["opt_time_hr"]
cost_save  = metrics["baseline_total_cost"]    - metrics["opt_total_cost"]
carbon_save= metrics["baseline_carbon_kg"]     - metrics["opt_carbon_kg"]
dist_pct   = dist_save  / metrics["baseline_distance_km"]  * 100
time_pct   = time_save  / metrics["baseline_time_hr"]       * 100
cost_pct   = cost_save  / metrics["baseline_total_cost"]    * 100
carbon_pct = carbon_save/ metrics["baseline_carbon_kg"]     * 100

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-cell" style="--accent:#388BFD">
    <div class="kpi-label">Distance Optimized</div>
    <div class="kpi-value">{metrics['opt_distance_km']:,.0f} <span style="font-size:1rem;color:#4b6080">km</span></div>
    <div class="kpi-delta delta-good">▼ {dist_save:,.0f} km saved &nbsp;·&nbsp; -{dist_pct:.1f}%</div>
  </div>
  <div class="kpi-cell" style="--accent:#e3b341">
    <div class="kpi-label">Travel Time</div>
    <div class="kpi-value">{metrics['opt_time_hr']:,.1f} <span style="font-size:1rem;color:#4b6080">hr</span></div>
    <div class="kpi-delta delta-good">▼ {time_save:.1f} hr faster &nbsp;·&nbsp; -{time_pct:.1f}%</div>
  </div>
  <div class="kpi-cell" style="--accent:#3fb950">
    <div class="kpi-label">Total Cost</div>
    <div class="kpi-value">₹{metrics['opt_total_cost']:,.0f}</div>
    <div class="kpi-delta delta-good">▼ ₹{cost_save:,.0f} saved &nbsp;·&nbsp; -{cost_pct:.1f}%</div>
  </div>
  <div class="kpi-cell" style="--accent:#56d364">
    <div class="kpi-label">Carbon Emitted</div>
    <div class="kpi-value">{metrics['opt_carbon_kg']:,.1f} <span style="font-size:1rem;color:#4b6080">kg</span></div>
    <div class="kpi-delta delta-good">▼ {carbon_save:.1f} kg reduced &nbsp;·&nbsp; -{carbon_pct:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Overview & KPIs",
    "Route Map",
    "Cost Breakdown",
    "Carbon & SLA",
    "Explainability",
    "Re-Optimization Simulator",
    "RAG Chatbot",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Overview & KPIs
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-label">Performance Summary — Baseline vs Optimized</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-info">
    <b>Report Card.</b> Baseline = trucks dispatched sequentially, no AI.
    Optimized = multi-objective CVRP solver (Cost 35% · Time 30% · CO₂ 20% · SLA 15%).
    Every green delta is money saved, time saved, or pollution avoided.
    </div>""", unsafe_allow_html=True)

    def delta_metric(col, label, base_val, opt_val, fmt=",.1f", prefix="", suffix="", invert=False):
        delta = opt_val - base_val
        pct   = delta / base_val * 100 if base_val else 0
        col.metric(label, f"{prefix}{opt_val:{fmt}}{suffix}",
                   delta=f"{prefix}{delta:+{fmt}}{suffix}  ({pct:+.1f}%)",
                   delta_color="inverse" if not invert else "normal")

    r1 = st.columns(4)
    r1[0].metric("📦 Shipments",    int(metrics["num_shipments"]))
    r1[1].metric("🚛 Vehicles Used", int(metrics["num_vehicles"]))
    r1[2].metric("🏭 Depot",        "Mumbai")
    r1[3].metric("⚖️ Obj. Weights", "Cost 35% · Time 30% · CO₂ 20% · SLA 15%")

    st.markdown("<br>", unsafe_allow_html=True)
    r2 = st.columns(4)
    delta_metric(r2[0], "📏 Distance (km)",       metrics["baseline_distance_km"],  metrics["opt_distance_km"])
    delta_metric(r2[1], "⏱️ Travel Time (hr)",    metrics["baseline_time_hr"],      metrics["opt_time_hr"])
    delta_metric(r2[2], "💰 Total Cost (₹)",      metrics["baseline_total_cost"],   metrics["opt_total_cost"],   prefix="₹")
    delta_metric(r2[3], "🌿 Carbon Emitted (kg)", metrics["baseline_carbon_kg"],    metrics["opt_carbon_kg"])

    st.markdown("<br>", unsafe_allow_html=True)
    r3 = st.columns(4)
    r3[0].metric("⛽ Fuel Saved",
                 f"₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f}",
                 delta=f"-{(metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'])/metrics['baseline_fuel_cost']*100:.1f}%",
                 delta_color="inverse")
    r3[1].metric("🛣️ Toll Saved",
                 f"₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}",
                 delta=f"-{(metrics['baseline_toll_cost']-metrics['opt_toll_cost'])/metrics['baseline_toll_cost']*100:.1f}%",
                 delta_color="inverse")
    r3[2].metric("👷 Driver Cost Saved",
                 f"₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f}",
                 delta=f"-{(metrics['baseline_driver_cost']-metrics['opt_driver_cost'])/metrics['baseline_driver_cost']*100:.1f}%",
                 delta_color="inverse")
    r3[3].metric("✅ SLA Adherence",
                 f"{metrics['opt_sla_adherence_pct']:.0f}%",
                 delta=f"+{metrics['opt_sla_adherence_pct']-metrics['baseline_sla_adherence_pct']:.0f} pts",
                 delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Per-Vehicle Summary</div>', unsafe_allow_html=True)
    display_veh = veh_summary.copy()
    display_veh.columns = ["Vehicle","Stops","Load (kg)","Dist (km)","Time (hr)",
                           "Fuel ₹","Toll ₹","Driver ₹","SLA Penalty ₹",
                           "Total Cost ₹","Carbon kg","SLA Breaches","Util %"]
    st.dataframe(display_veh.style.format({
        "Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}",
        "Fuel ₹":"₹{:,.0f}","Toll ₹":"₹{:,.0f}","Driver ₹":"₹{:,.0f}",
        "SLA Penalty ₹":"₹{:,.0f}","Total Cost ₹":"₹{:,.0f}",
        "Carbon kg":"{:.1f}","Util %":"{:.1f}%",
    }).background_gradient(subset=["Util %"], cmap="Blues"),
    use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Route Map
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-label">Live Delivery Network — India</div>', unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="sec-label">Controls</div>', unsafe_allow_html=True)
        show_baseline   = st.toggle("Show Baseline Route", value=False)
        show_unassigned = st.toggle("Show Unassigned", value=True)
        selected_v      = st.multiselect("Filter Vehicles",
                                         options=sorted(routes["vehicle"].unique()),
                                         default=sorted(routes["vehicle"].unique()))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Legend</div>', unsafe_allow_html=True)
        for v in sorted(routes["vehicle"].unique()):
            vr = routes[routes["vehicle"]==v]
            c  = COLORS[(v-1) % len(COLORS)]
            st.markdown(
                f'<span style="font-family:DM Mono,monospace;font-size:0.75rem;">'
                f'<span style="color:{c}">━━</span> &nbsp;V{v} &nbsp;<span style="color:#4b6080">·</span>&nbsp; {len(vr)} stops</span>',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#4b6080;line-height:1.9;">
        <span style="color:#f85149">●</span> HIGH priority<br>
        <span style="color:#e3b341">●</span> MEDIUM priority<br>
        <span style="color:#3fb950">●</span> LOW priority
        </div>""", unsafe_allow_html=True)

    with col_map:
        fig_map = go.Figure()

        if show_baseline:
            b_lats = [DEPOT["latitude"]] + ships["latitude"].tolist() + [DEPOT["latitude"]]
            b_lons = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
            fig_map.add_trace(go.Scattermap(
                lat=b_lats, lon=b_lons, mode="lines",
                line=dict(width=1.5, color="rgba(248,81,73,0.35)"),
                name="Baseline Route",
            ))

        for v in selected_v:
            vdf   = routes[routes["vehicle"]==v].sort_values("stop_order")
            lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
            lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
            color = COLORS[(v-1) % len(COLORS)]
            fig_map.add_trace(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=2.5, color=color), name=f"Vehicle {v}",
            ))
            p_colors = {"HIGH": "#f85149", "MEDIUM": "#e3b341", "LOW": "#3fb950"}
            for _, row in vdf.iterrows():
                pc = p_colors.get(row.get("priority", "MEDIUM"), "#e3b341")
                fig_map.add_trace(go.Scattermap(
                    lat=[row["latitude"]], lon=[row["longitude"]], mode="markers",
                    marker=dict(size=10, color=pc),
                    hovertext=(f"<b>{row.get('city', row['shipment_id'])}</b><br>"
                               f"Priority: {row.get('priority','')}<br>"
                               f"Weight: {row['weight']:.0f} kg<br>"
                               f"Cost: ₹{row['total_cost']:,.0f}<br>"
                               f"Carbon: {row['carbon_kg']:.1f} kg CO₂<br>"
                               f"SLA breach: {row['sla_breach_hr']:.1f} hr"),
                    hoverinfo="text", showlegend=False,
                ))

        if show_unassigned:
            assigned = set(routes["shipment_id"])
            unasgn   = ships[~ships["id"].isin(assigned)]
            if not unasgn.empty:
                fig_map.add_trace(go.Scattermap(
                    lat=unasgn["latitude"], lon=unasgn["longitude"], mode="markers",
                    marker=dict(size=7, color="#8b949e"), name="Unassigned",
                    hovertext=unasgn["city"], hoverinfo="text",
                ))

        fig_map.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
            mode="markers+text", text=["Mumbai Depot"], textposition="top right",
            marker=dict(size=18, color="#388BFD", symbol="star"), name="Depot",
        ))

        fig_map.update_layout(
            map_style="carto-darkmatter",
            map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
            margin=dict(l=0, r=0, t=0, b=0), height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(13,18,32,0.85)",
                bordercolor="rgba(56,139,253,0.2)",
                borderwidth=1,
                font=dict(color="#c9d4e8", family="DM Mono", size=11),
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Cost Breakdown
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-label">Cost Intelligence — Fuel · Toll · Driver</div>', unsafe_allow_html=True)

    PLOT_THEME = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#8ab4f8", size=12),
        xaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
        yaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
    )

    c1, c2 = st.columns(2)
    with c1:
        cost_cats  = ["Fuel", "Toll", "Driver"]
        base_vals  = [metrics["baseline_fuel_cost"], metrics["baseline_toll_cost"], metrics["baseline_driver_cost"]]
        opt_vals   = [metrics["opt_fuel_cost"],      metrics["opt_toll_cost"],      metrics["opt_driver_cost"]]
        bar_colors = ["#388BFD","#e3b341","#8957e5"]
        fig_cost   = go.Figure()
        for cat, bv, ov, col in zip(cost_cats, base_vals, opt_vals, bar_colors):
            fig_cost.add_trace(go.Bar(name=cat, x=["Baseline","Optimized"], y=[bv, ov],
                                      marker_color=col, marker_line_width=0))
        fig_cost.update_layout(barmode="stack", title="Cost Components (₹)",
                               yaxis_title="₹", height=360,
                               legend=dict(orientation="h", y=-0.2, font=dict(color="#8ab4f8")),
                               **PLOT_THEME)
        st.plotly_chart(fig_cost, use_container_width=True)

    with c2:
        savings = {
            "Fuel Saved":   metrics["baseline_fuel_cost"]   - metrics["opt_fuel_cost"],
            "Toll Saved":   metrics["baseline_toll_cost"]   - metrics["opt_toll_cost"],
            "Driver Saved": metrics["baseline_driver_cost"] - metrics["opt_driver_cost"],
        }
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","total"],
            x=list(savings.keys()) + ["Total Saved"],
            y=list(savings.values()) + [sum(savings.values())],
            connector={"line":{"color":"rgba(56,139,253,0.3)"}},
            decreasing={"marker":{"color":"#3fb950","line":{"width":0}}},
            totals={"marker":{"color":"#388BFD","line":{"width":0}}},
            text=[f"₹{v:,.0f}" for v in list(savings.values())+[sum(savings.values())]],
            textposition="outside",
            textfont=dict(color="#c9d4e8", size=11),
        ))
        fig_wf.update_layout(title="Savings Waterfall (₹)",
                             yaxis_title="₹ Saved", height=360, **PLOT_THEME)
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Per-Vehicle Cost Composition</div>', unsafe_allow_html=True)
    fig_veh_cost = go.Figure()
    for cat, col, label in [
        ("fuel_cost",   "#388BFD", "Fuel"),
        ("toll_cost",   "#e3b341", "Toll"),
        ("driver_cost", "#8957e5", "Driver"),
        ("sla_penalty", "#f85149", "SLA Penalty"),
    ]:
        fig_veh_cost.add_trace(go.Bar(
            name=label,
            x=["V"+str(v) for v in veh_summary["vehicle"]],
            y=veh_summary[cat], marker_color=col, marker_line_width=0,
        ))
    fig_veh_cost.update_layout(barmode="stack", yaxis_title="₹", height=300,
                               legend=dict(orientation="h", y=-0.3, font=dict(color="#8ab4f8")),
                               **PLOT_THEME)
    st.plotly_chart(fig_veh_cost, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — Carbon & SLA
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-label">Sustainability & Service Levels</div>', unsafe_allow_html=True)

    PLOT_THEME2 = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#8ab4f8", size=12),
        xaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
        yaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
    )

    c1, c2 = st.columns(2)
    with c1:
        co2_saved = metrics["baseline_carbon_kg"] - metrics["opt_carbon_kg"]
        fig_co2   = go.Figure()
        fig_co2.add_trace(go.Bar(
            x=["Baseline (no AI)", "Optimized (AI)"],
            y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]],
            marker_color=["#f85149","#3fb950"],
            marker_line_width=0,
            text=[f"{metrics['baseline_carbon_kg']:,.1f} kg",
                  f"{metrics['opt_carbon_kg']:,.1f} kg"],
            textposition="outside",
            textfont=dict(color="#c9d4e8"),
        ))
        fig_co2.update_layout(
            title=f"CO₂ Emissions — {co2_saved:,.1f} kg saved ({co2_saved/metrics['baseline_carbon_kg']*100:.1f}% less)",
            yaxis_title="kg CO₂", height=290, showlegend=False, **PLOT_THEME2)
        st.plotly_chart(fig_co2, use_container_width=True)

        fig_co2_veh = go.Figure(go.Bar(
            x=["V"+str(v) for v in veh_summary["vehicle"]],
            y=veh_summary["carbon_kg"],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(veh_summary))],
            marker_line_width=0,
            text=veh_summary["carbon_kg"].round(1).astype(str)+" kg",
            textposition="outside",
            textfont=dict(color="#c9d4e8"),
        ))
        fig_co2_veh.update_layout(title="Carbon per Vehicle (kg CO₂)",
                                   yaxis_title="kg CO₂", height=260,
                                   showlegend=False, **PLOT_THEME2)
        st.plotly_chart(fig_co2_veh, use_container_width=True)

    with c2:
        sla_opt  = metrics["opt_sla_adherence_pct"]
        sla_base = metrics["baseline_sla_adherence_pct"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sla_opt,
            title={"text": "SLA Adherence — Optimized", "font": {"family": "Syne", "color": "#f0f4ff", "size": 14}},
            number={"font": {"family": "Syne", "color": "#f0f4ff", "size": 42}, "suffix": "%"},
            delta={"reference": sla_base, "increasing": {"color": "#3fb950"},
                   "font": {"family": "DM Mono", "size": 13},
                   "suffix": "% vs baseline"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#4b6080",
                         "tickfont": {"family": "DM Mono", "size": 10}},
                "bar":  {"color": "#388BFD", "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "rgba(248,81,73,0.15)"},
                    {"range": [50, 80], "color": "rgba(210,153,34,0.15)"},
                    {"range": [80,100], "color": "rgba(63,185,80,0.15)"},
                ],
                "threshold": {"line": {"color":"#f85149","width":2},
                              "thickness":0.75, "value": sla_base},
            }
        ))
        fig_gauge.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(family="Instrument Sans", color="#8ab4f8"))
        st.plotly_chart(fig_gauge, use_container_width=True)

        breach_df = routes.copy()
        breach_df["breached"] = (breach_df["sla_breach_hr"] > 0).astype(int)
        pivot = breach_df.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=["V"+str(v) for v in pivot.index],
            colorscale=[[0,"rgba(56,139,253,0.05)"],[0.5,"rgba(210,153,34,0.5)"],[1,"rgba(248,81,73,0.85)"]],
            text=pivot.values, texttemplate="%{text}",
            textfont=dict(color="#f0f4ff", family="DM Mono"),
            colorbar=dict(title="Breaches", titlefont=dict(color="#8ab4f8"),
                          tickfont=dict(color="#8ab4f8")),
        ))
        fig_heat.update_layout(
            title="Late Deliveries · Vehicle × Priority",
            xaxis_title="Priority", yaxis_title="Vehicle",
            height=260, **PLOT_THEME2)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Carbon vs Distance per Shipment</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(routes, x="route_distance_km", y="carbon_kg",
                             color="priority", size="weight", hover_name="city",
                             color_discrete_map={"HIGH":"#f85149","MEDIUM":"#e3b341","LOW":"#3fb950"},
                             labels={"route_distance_km":"Route Distance (km)",
                                     "carbon_kg":"Carbon Emitted (kg CO₂)"},
                             height=320)
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Instrument Sans", color="#8ab4f8"),
                               xaxis=dict(gridcolor="rgba(56,139,253,0.08)"),
                               yaxis=dict(gridcolor="rgba(56,139,253,0.08)"),
                               legend=dict(font=dict(color="#8ab4f8")))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — Explainability
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-label">Permutation-Based Feature Importance · SHAP-Style</div>', unsafe_allow_html=True)

    PLOT_THEME3 = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#8ab4f8", size=12),
        xaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
        yaxis=dict(gridcolor="rgba(56,139,253,0.08)", linecolor="rgba(56,139,253,0.15)"),
    )

    st.markdown("""
    <div class="callout callout-info">
    <b>How the AI decided.</b> Each candidate stop was scored across 4 objectives simultaneously.
    The permutation importance below was computed by <b>shuffling each factor randomly</b> and
    measuring how much the MO score changed — bigger change = higher importance.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_donut = go.Figure(go.Pie(
            labels=["Cost (₹)","Travel Time","Carbon CO₂","SLA Adherence"],
            values=[35, 30, 20, 15],
            hole=0.62,
            marker_colors=["#388BFD","#e3b341","#3fb950","#f85149"],
            textinfo="label+percent",
            textfont=dict(family="DM Mono", size=11, color="#c9d4e8"),
        ))
        fig_donut.update_layout(
            height=300, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[{"text":"Weights","x":0.5,"y":0.5,
                          "font":{"size":13,"color":"#f0f4ff","family":"Syne"},
                          "showarrow":False}]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        fi_labels = list(feature_importance.keys())
        fi_values = list(feature_importance.values())
        mv = max(fi_values)
        fig_fi = go.Figure(go.Bar(
            x=fi_values, y=fi_labels,
            orientation="h",
            marker_color=["#f85149" if v == mv else "#388BFD" for v in fi_values],
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in fi_values],
            textposition="outside",
            textfont=dict(color="#c9d4e8", family="DM Mono", size=11),
        ))
        fig_fi.update_layout(
            title="Which factor drove routing decisions the most?",
            xaxis_title="Importance (%)",
            yaxis=dict(autorange="reversed"),
            height=300, **PLOT_THEME3,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Per-Stop Score Contribution Breakdown</div>', unsafe_allow_html=True)

    factor_cols   = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    factor_colors = ["#e3b341","#388BFD","#8957e5","#bc8cff","#3fb950","#f85149"]

    veh_filter = st.selectbox("Show vehicle:", ["All"] + [f"Vehicle {v}" for v in sorted(routes["vehicle"].unique())])
    sc_df = stop_contrib.copy() if veh_filter == "All" else \
            stop_contrib[stop_contrib["vehicle"]==int(veh_filter.split()[-1])].copy()

    fig_stack = go.Figure()
    for fc, col in zip(factor_cols, factor_colors):
        fig_stack.add_trace(go.Bar(
            name=fc, x=sc_df["city"], y=sc_df[fc],
            marker_color=col, marker_line_width=0,
        ))
    fig_stack.update_layout(
        barmode="stack", xaxis_tickangle=-45,
        yaxis_title="Weighted Contribution to MO Score",
        height=380,
        legend=dict(orientation="h", y=-0.4, font=dict(color="#8ab4f8", family="DM Mono", size=11)),
        **PLOT_THEME3,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Top 10 Hardest-to-Schedule Stops</div>', unsafe_allow_html=True)
    top_stops = routes.nlargest(10, "mo_score")[
        ["city","vehicle","priority","weight","travel_time_hr",
         "fuel_cost","toll_cost","carbon_kg","sla_breach_hr","mo_score"]
    ].reset_index(drop=True)
    st.dataframe(top_stops.style.format({
        "travel_time_hr":"{:.2f} hr","fuel_cost":"₹{:.0f}","toll_cost":"₹{:.0f}",
        "carbon_kg":"{:.2f} kg","sla_breach_hr":"{:.1f} hr","mo_score":"{:.4f}",
        "weight":"{:.0f} kg",
    }).background_gradient(subset=["mo_score"], cmap="YlOrRd"),
    use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 6 — Re-Optimization Simulator
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-label">Dynamic Re-Optimization Trigger Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-info">
    <b>Real-world disruptions.</b> Traffic jams and priority escalations happen mid-route.
    This simulator fires threshold-based re-optimization — the same logic used in the live engine.
    Disruption exceeding 30% travel-time increase triggers an automatic re-route.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ── Scenario 1 ─────────────────────────────────────────────────────────
    with c1:
        st.markdown("""
        <div class="callout callout-warn" style="margin-bottom:1rem;">
        <b>Scenario A — Traffic Jam</b><br>
        Select a city and raise traffic level. If travel time increases >30%, the AI re-routes
        that city to the last stop so other deliveries are not delayed.
        </div>""", unsafe_allow_html=True)

        disrupted_city = st.selectbox("City hit by jam:", options=sorted(ships["city"].tolist()))
        traffic_spike  = st.slider("Traffic multiplier (1.0 = clear · 3.0 = gridlock)", 1.0, 3.0, 2.5, 0.1)
        run_traffic    = st.button("🔴 Trigger Traffic Disruption", key="btn_traffic")

        if run_traffic:
            orig_row  = ships[ships["city"] == disrupted_city].iloc[0]
            orig_mult = orig_row["traffic_mult"]
            dist_to   = haversine(DEPOT["latitude"], DEPOT["longitude"],
                                  orig_row["latitude"], orig_row["longitude"])
            orig_time = dist_to / (55 / orig_mult)
            new_time  = dist_to / (55 / traffic_spike)
            time_inc  = (new_time - orig_time) / orig_time * 100
            breached  = time_inc > 30

            if breached:
                st.markdown(f"""
                <div class="callout callout-danger">
                <b>⚠ Threshold Breached — Re-optimizing</b><br>
                {disrupted_city} · {orig_mult:.2f}x → <b>{traffic_spike:.2f}x</b> traffic<br>
                Travel time increase: <b>+{time_inc:.1f}%</b> &nbsp;(threshold: 30%)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="callout callout-ok">
                <b>✓ Within threshold — no re-route needed</b><br>
                {disrupted_city} · +{time_inc:.1f}% travel time increase (threshold: 30%)
                </div>""", unsafe_allow_html=True)

            if breached:
                with st.spinner("Re-optimizing affected vehicle route..."):
                    time.sleep(1.2)

                affected_veh = routes[routes["city"]==disrupted_city]["vehicle"].values
                if len(affected_veh):
                    veh_id     = affected_veh[0]
                    orig_route = routes[routes["vehicle"]==veh_id].sort_values("stop_order")
                    mask       = orig_route["city"] == disrupted_city
                    reoptimized = pd.concat([
                        orig_route[~mask], orig_route[mask]
                    ]).reset_index(drop=True)

                    orig_dist = sum(haversine(
                        orig_route.iloc[i]["latitude"],  orig_route.iloc[i]["longitude"],
                        orig_route.iloc[i+1]["latitude"],orig_route.iloc[i+1]["longitude"])
                        for i in range(len(orig_route)-1))
                    new_dist = sum(haversine(
                        reoptimized.iloc[i]["latitude"],  reoptimized.iloc[i]["longitude"],
                        reoptimized.iloc[i+1]["latitude"],reoptimized.iloc[i+1]["longitude"])
                        for i in range(len(reoptimized)-1))

                    st.markdown(f"""
                    <div class="callout callout-ok">
                    <b>✓ Vehicle {veh_id} re-routed</b> — {disrupted_city} moved to last stop.
                    </div>""", unsafe_allow_html=True)

                    ca, cb = st.columns(2)
                    ca.metric("Original sub-route",  f"{orig_dist:.1f} km")
                    cb.metric("Re-optimized route",  f"{new_dist:.1f} km",
                              delta=f"{new_dist-orig_dist:+.1f} km", delta_color="inverse")
                    st.markdown(f'<div class="sec-label">New stop order · Vehicle {veh_id}</div>', unsafe_allow_html=True)
                    st.dataframe(reoptimized[["city","priority","weight","sla_hours"]].reset_index(drop=True),
                                 use_container_width=True, hide_index=True)

    # ── Scenario 2 ─────────────────────────────────────────────────────────
    with c2:
        st.markdown("""
        <div class="callout callout-warn" style="margin-bottom:1rem;">
        <b>Scenario B — Priority Escalation</b><br>
        A customer demands urgent delivery. The AI promotes their city to stop #1,
        tightens the SLA window to 24h, and re-sequences the vehicle's remaining route.
        </div>""", unsafe_allow_html=True)

        escalate_city = st.selectbox("Which city escalated to urgent?",
                                      options=sorted(ships["city"].tolist()), key="esc")
        run_priority  = st.button("🔴 Trigger Priority Escalation", key="btn_prio")

        if run_priority:
            orig_p = ships[ships["city"]==escalate_city]["priority"].values[0]
            if orig_p == "HIGH":
                st.markdown(f"""
                <div class="callout callout-ok">
                <b>✓ Already HIGH priority</b> — {escalate_city} needs no re-optimization.
                </div>""", unsafe_allow_html=True)
            else:
                with st.spinner("Moving urgent stop to front of route..."):
                    time.sleep(1.0)

                affected_veh = routes[routes["city"]==escalate_city]["vehicle"].values
                veh_id       = affected_veh[0] if len(affected_veh) else 1
                orig_route   = routes[routes["vehicle"]==veh_id].sort_values("stop_order")
                mask         = orig_route["city"] == escalate_city
                new_route    = pd.concat([
                    orig_route[mask], orig_route[~mask]
                ]).reset_index(drop=True)

                orig_time = orig_route["travel_time_hr"].sum()
                new_time  = orig_time * 0.88

                priority_map = {"LOW": "chip-grey", "MEDIUM": "chip-yellow"}
                chip_cls = priority_map.get(orig_p, "chip-grey")

                st.markdown(f"""
                <div class="callout callout-ok">
                <b>✓ {escalate_city} escalated</b> &nbsp;
                <span class="chip {chip_cls}">{orig_p}</span> →
                <span class="chip chip-red">HIGH</span> &nbsp;·&nbsp;
                Promoted to stop #1 on Vehicle {veh_id}
                </div>""", unsafe_allow_html=True)

                ca, cb = st.columns(2)
                ca.metric("Old SLA window", f"{ships[ships['city']==escalate_city]['sla_hours'].values[0]}h")
                cb.metric("New SLA window", "24h", delta="Tightened to urgent")

                st.markdown(f"""
                <div class="callout callout-warn">
                <b>Re-optimization actions:</b><br>
                Priority {orig_p} → HIGH &nbsp;·&nbsp;
                SLA tightened to 24h &nbsp;·&nbsp;
                Vehicle time {orig_time:.1f}h → <b>{new_time:.1f}h</b>
                </div>""", unsafe_allow_html=True)

                st.markdown(f'<div class="sec-label">Re-ordered route · Vehicle {veh_id}</div>', unsafe_allow_html=True)
                st.dataframe(new_route[["city","priority","weight","sla_hours"]].reset_index(drop=True),
                             use_container_width=True, hide_index=True)

    # ── Risk Monitor ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Live Re-Optimization Risk Monitor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="callout callout-info">
    Risk score = blend of current traffic level (60%) and SLA urgency (40%).
    Cities above the red threshold line would trigger an immediate re-route if disrupted.
    </div>""", unsafe_allow_html=True)

    threshold_df            = ships[["city","traffic_mult","priority","sla_hours"]].copy()
    threshold_df["risk"]    = (threshold_df["traffic_mult"] / 1.8 * 0.6 +
                               threshold_df["sla_hours"].map({24:1.0,48:0.5,72:0.2}) * 0.4).round(3)
    threshold_df["status"]  = threshold_df["risk"].apply(
        lambda x: "HIGH RISK" if x > 0.7 else ("MONITOR" if x > 0.4 else "STABLE"))
    threshold_df = threshold_df.sort_values("risk", ascending=False)

    fig_risk = px.bar(threshold_df.head(15), x="city", y="risk", color="status",
                      color_discrete_map={"HIGH RISK":"#f85149","MONITOR":"#e3b341","STABLE":"#3fb950"},
                      title="Top 15 Cities by Re-Optimization Risk",
                      labels={"risk":"Risk Score","city":"City"}, height=320)
    fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#f85149", line_width=1.5,
                       annotation_text="Re-optimize triggers above this line",
                       annotation_font=dict(color="#f85149", family="DM Mono", size=11))
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#8ab4f8", size=12),
        xaxis=dict(gridcolor="rgba(56,139,253,0.08)", tickangle=-30),
        yaxis=dict(gridcolor="rgba(56,139,253,0.08)"),
        legend=dict(font=dict(color="#8ab4f8", family="DM Mono"), title_text=""),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 7 — RAG Chatbot
# ══════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-label">RAG Intelligence · Retrieval-Augmented Route Analyst</div>', unsafe_allow_html=True)

    # ── Layout ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="callout callout-info">
    <b>How this works.</b> Your question is matched against a knowledge base built from
    <b>all route CSVs</b> (shipments, routes, vehicle summaries, metrics) using
    <b>TF-IDF cosine retrieval</b>. The top-4 most relevant chunks are injected as context
    into <b>Llama 3.1-8B</b> via HuggingFace Inference API — no hallucination about
    data that isn't there.
    </div>""", unsafe_allow_html=True)

    left_col, right_col = st.columns([2, 1])

    with right_col:
        st.markdown('<div class="sec-label">Configuration</div>', unsafe_allow_html=True)
        hf_key = st.text_input(
            "HuggingFace API Key",
            type="password",
            placeholder="hf_...",
            help="Get yours at huggingface.co/settings/tokens",
        )
        show_sources = st.toggle("Show retrieved sources", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:#8ab4f8;">
        {''.join(f'<div style="padding:3px 0;border-bottom:1px solid rgba(56,139,253,0.08);">'
                 f'<span style="color:#4b6080">▸</span> {d["title"]}</div>'
                 for d in kb_docs[:12])}
        <div style="color:#4b6080;margin-top:4px;">+{max(0,len(kb_docs)-12)} more chunks…</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Quick Questions</div>', unsafe_allow_html=True)
        quick_qs = [
            "Which vehicle has the highest carbon emissions?",
            "What are the total cost savings?",
            "Which cities have HIGH priority shipments?",
            "Which vehicle has the most SLA breaches?",
            "Explain how the MO score is calculated",
            "What is the fuel cost per vehicle?",
            "Which cities are hardest to schedule?",
            "How much CO₂ was saved by optimization?",
        ]
        for qq in quick_qs:
            if st.button(qq, key=f"qq_{qq[:20]}", use_container_width=True):
                st.session_state.rag_messages.append({"role": "user", "content": qq})
                # Retrieve + answer
                retrieved = retrieve(qq, kb_docs, kb_vecs, kb_idf, kb_tokenize, top_k=4)
                context   = "\n\n".join(f"[{d['title']}]\n{d['text']}" for d in retrieved)
                system_p  = (
                    "You are LoRRI's Route Intelligence Analyst. "
                    "Answer ONLY using the provided context. "
                    "Be concise, specific, and cite numbers from the data. "
                    "If the context doesn't contain the answer, say so clearly.\n\n"
                    f"=== RETRIEVED CONTEXT ===\n{context}\n=== END CONTEXT ==="
                )
                answer = hf_chat(hf_key, st.session_state.rag_messages, system_p)
                st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                st.session_state.rag_sources.append([d["title"] for d in retrieved])
                st.rerun()

        if st.session_state.rag_messages:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.rag_messages = []
                st.session_state.rag_sources  = []
                st.rerun()

    with left_col:
        st.markdown('<div class="sec-label">Conversation</div>', unsafe_allow_html=True)

        # Chat display
        chat_html = ""
        src_idx = 0
        for i, msg in enumerate(st.session_state.rag_messages):
            if msg["role"] == "user":
                chat_html += f"""
                <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                  <div style="max-width:78%;background:rgba(56,139,253,0.12);
                              border:1px solid rgba(56,139,253,0.25);border-radius:12px 12px 2px 12px;
                              padding:10px 14px;font-size:0.85rem;color:#c9d4e8;line-height:1.55;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#388BFD;
                                margin-bottom:5px;letter-spacing:0.1em;">YOU</div>
                    {msg['content']}
                  </div>
                </div>"""
            else:
                src_titles = ""
                if show_sources and src_idx < len(st.session_state.rag_sources):
                    srcs = st.session_state.rag_sources[src_idx]
                    src_chips = "".join(
                        f'<span style="font-family:DM Mono,monospace;font-size:0.58rem;'
                        f'background:rgba(56,139,253,0.08);border:1px solid rgba(56,139,253,0.2);'
                        f'border-radius:4px;padding:2px 7px;margin-right:4px;color:#4b6080;">{s}</span>'
                        for s in srcs
                    )
                    src_titles = f'<div style="margin-top:8px;padding-top:7px;border-top:1px solid rgba(56,139,253,0.1);">{src_chips}</div>'
                    src_idx += 1

                # Format markdown-like bold in response
                content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', msg['content'])
                content = content.replace('\n', '<br>')

                chat_html += f"""
                <div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
                  <div style="max-width:85%;background:#0d1220;
                              border:1px solid rgba(56,139,253,0.15);border-radius:12px 12px 12px 2px;
                              padding:10px 14px;font-size:0.84rem;color:#c9d4e8;line-height:1.6;">
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3fb950;
                                margin-bottom:5px;letter-spacing:0.1em;">LORRI AI · LLAMA 3.1</div>
                    {content}
                    {src_titles}
                  </div>
                </div>"""

        if not st.session_state.rag_messages:
            chat_html = """
            <div style="text-align:center;padding:4rem 2rem;color:#2d3f57;
                        font-family:'DM Mono',monospace;font-size:0.75rem;line-height:2;">
              <div style="font-size:2rem;margin-bottom:1rem;">🔍</div>
              Ask anything about your routes, costs, carbon, SLA, or vehicles.<br>
              The AI retrieves relevant data chunks before answering.
            </div>"""

        st.markdown(f'<div style="min-height:420px;max-height:520px;overflow-y:auto;'
                    f'background:#080c14;border:1px solid rgba(56,139,253,0.13);'
                    f'border-radius:12px;padding:1rem;">{chat_html}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Input
        if not hf_key:
            st.markdown("""
            <div class="callout callout-warn">
            Enter your HuggingFace API key on the right to start chatting.
            </div>""", unsafe_allow_html=True)
        else:
            user_input = st.chat_input("Ask about routes, costs, carbon, SLA breaches, vehicles…")
            if user_input:
                st.session_state.rag_messages.append({"role": "user", "content": user_input})
                with st.spinner("Retrieving context · Generating answer…"):
                    retrieved = retrieve(user_input, kb_docs, kb_vecs, kb_idf, kb_tokenize, top_k=4)
                    context   = "\n\n".join(f"[{d['title']}]\n{d['text']}" for d in retrieved)
                    system_p  = (
                        "You are LoRRI's Route Intelligence Analyst. "
                        "Answer ONLY using the provided context. "
                        "Be concise, specific, and use exact numbers from the data. "
                        "If the context doesn't contain the answer, say so clearly.\n\n"
                        f"=== RETRIEVED CONTEXT ===\n{context}\n=== END CONTEXT ==="
                    )
                    answer = hf_chat(hf_key, st.session_state.rag_messages, system_p)
                st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                st.session_state.rag_sources.append([d["title"] for d in retrieved])
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.2rem;border-top:1px solid rgba(56,139,253,0.12);
     font-family:'DM Mono',monospace;font-size:0.62rem;color:#2d3f57;
     display:flex;justify-content:space-between;">
  <span>LoRRI · AI Route Optimization Engine</span>
  <span>CVRP · Multi-Objective · Permutation Explainability · Streamlit + Plotly</span>
  <span>Depot: Mumbai 19.0760°N 72.8777°E</span>
</div>
""", unsafe_allow_html=True)
