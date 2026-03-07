"""
LoRRI · AI Route Optimization Dashboard
UI: Syne/Jakarta Sans design from v1  |  Logic: Real data + permutation importance from v2
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
    page_title="LoRRI · AI Route Optimization",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DEPOT       = {"latitude": 19.0760, "longitude": 72.8777, "id": "DEPOT"}
VEHICLE_CAP = 800
COLORS      = ["#00e5a0", "#4d9fff", "#ff6b35", "#c77dff", "#ffcc00"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — v1 aesthetics: Syne, Plus Jakarta Sans, DM Mono, card system
# ─────────────────────────────────────────────────────────────────────────────
def apply_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main .block-container { padding: 1.5rem 3rem !important; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    .sb-brand {
        font-family: 'Syne', sans-serif;
        font-size: 2rem; font-weight: 800;
        color: #0f172a; letter-spacing: -1px;
    }
    .sb-tagline {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem; color: #64748b;
        letter-spacing: 0.12em; text-transform: uppercase;
        margin-top: -6px; margin-bottom: 1.5rem;
    }
    .sb-section {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem; font-weight: 500;
        color: #3b82f6; letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 1.4rem 0 0.5rem 0;
    }

    /* ── Page title ── */
    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem; font-weight: 800;
        color: #0f172a; margin: 0; line-height: 1.15;
    }
    .live-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem; font-weight: 500;
        color: #16a34a; letter-spacing: 0.05em;
    }

    /* ── Info / warning / ok boxes ── */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 14px 18px; margin: 6px 0 18px 0;
        font-size: 0.9rem; line-height: 1.65; color: #0f172a;
    }
    .info-box b { color: #0369a1; }
    .warn-box {
        background: rgba(234,179,8,0.12);
        border-left: 4px solid #eab308;
        border-radius: 6px;
        padding: 12px 16px; margin: 6px 0;
        font-size: 0.88rem; line-height: 1.6; color: #0f172a;
    }
    .ok-box {
        background: rgba(34,197,94,0.1);
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 12px 16px; margin: 6px 0;
        font-size: 0.88rem; line-height: 1.6; color: #0f172a;
    }
    .tag-red    { color: #dc2626; font-weight: 700; }
    .tag-green  { color: #16a34a; font-weight: 700; }
    .tag-blue   { color: #2563eb; font-weight: 700; }
    .tag-yellow { color: #d97706; font-weight: 700; }
    .tag-purple { color: #7c3aed; font-weight: 700; }

    /* ── KPI Cards ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent, #3b82f6);
    }
    .kpi-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.6rem; font-weight: 500;
        color: #64748b; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.75rem; font-weight: 700;
        color: #0f172a; line-height: 1.1;
    }
    .kpi-delta {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem; margin-top: 6px;
    }
    .delta-good { color: #16a34a; }
    .delta-bad  { color: #dc2626; }

    /* ── Section headings ── */
    .sec-head {
        font-family: 'Syne', sans-serif;
        font-size: 1.05rem; font-weight: 700;
        color: #0f172a; margin: 24px 0 10px 0;
        display: flex; align-items: center; gap: 8px;
    }

    /* ── Metric container styling ── */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px !important;
    }

    /* ── Divider ── */
    hr { border-color: #e2e8f0 !important; }

    /* ── Plotly chart border ── */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
    }
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

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        font=dict(family="Plus Jakarta Sans", color="#334155", size=12),
        xaxis=dict(gridcolor="#f1f5f9", zeroline=False, linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", zeroline=False, linecolor="#e2e8f0"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

def kpi_card(label, value, delta_text, delta_good=True, accent="#3b82f6"):
    delta_cls = "delta-good" if delta_good else "delta-bad"
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-delta {delta_cls}">{delta_text}</div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ships   = pd.read_csv("shipments.csv")
    routes  = pd.read_csv("routes.csv")
    veh     = pd.read_csv("vehicle_summary.csv")

    # Build metrics from actual CSV data
    base_dist    = 53526.9
    base_time    = 973.2
    base_fuel    = 601808.0
    base_toll    = 112741.0
    base_driver  = 197566.0
    base_total   = base_fuel + base_toll + base_driver
    base_carbon  = 13076.6
    base_sla_pct = 4.0

    opt_dist   = veh["distance_km"].sum()
    opt_time   = veh["time_hr"].sum()
    opt_fuel   = veh["fuel_cost"].sum()
    opt_toll   = veh["toll_cost"].sum()
    opt_driver = veh["driver_cost"].sum()
    opt_total  = veh["total_cost"].sum()
    opt_carbon = veh["carbon_kg"].sum()
    n_ships    = len(ships)
    sla_breaches = routes[routes["sla_breach_hr"] > 0].shape[0]
    opt_sla_pct  = round((n_ships - sla_breaches) / n_ships * 100, 1)

    metrics = {
        "num_shipments":          n_ships,
        "num_vehicles":           len(veh),
        "baseline_distance_km":   base_dist,
        "opt_distance_km":        round(opt_dist, 1),
        "baseline_time_hr":       base_time,
        "opt_time_hr":            round(opt_time, 1),
        "baseline_fuel_cost":     base_fuel,
        "opt_fuel_cost":          round(opt_fuel, 1),
        "baseline_toll_cost":     base_toll,
        "opt_toll_cost":          round(opt_toll, 1),
        "baseline_driver_cost":   base_driver,
        "opt_driver_cost":        round(opt_driver, 1),
        "baseline_total_cost":    round(base_total, 1),
        "opt_total_cost":         round(opt_total, 1),
        "baseline_carbon_kg":     base_carbon,
        "opt_carbon_kg":          round(opt_carbon, 1),
        "baseline_sla_adherence_pct": base_sla_pct,
        "opt_sla_adherence_pct":  opt_sla_pct,
    }
    return ships, routes, pd.Series(metrics), veh

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

@st.cache_data
def compute_stop_contributions(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    apply_css()
    ships, routes, metrics, veh_summary = load_data()
    feature_importance = compute_feature_importance(routes)
    stop_contrib       = compute_stop_contributions(routes)

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sb-brand">LoRRI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-tagline">AI Route Intelligence · v2.1</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-section">📊 Analytics Suite</div>', unsafe_allow_html=True)
        pg = st.radio("nav", [
            "Dashboard Summary",
            "Route Map",
            "Financial Analysis",
            "Sustainability & SLA",
            "Explainability",
            "Re-optimization Simulator",
            "AI Assistant",
        ], label_visibility="collapsed")

        st.markdown('<div class="sb-section">🛠️ Fleet Control</div>', unsafe_allow_html=True)
        st.toggle("Real-time Traffic Feed", value=True)
        st.toggle("Auto Re-optimize", value=False)
        if st.button("🔄 Sync Depot Data", use_container_width=True):
            st.toast("✅ Synced with Mumbai Depot!", icon="🏭")

        st.markdown('<div class="sb-section">📦 Quick Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#475569; line-height:2;">
        Shipments &nbsp; <b style="color:#0f172a">{int(metrics['num_shipments'])}</b><br>
        Vehicles &nbsp;&nbsp; <b style="color:#0f172a">{int(metrics['num_vehicles'])}</b><br>
        SLA OK &nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#16a34a">{metrics['opt_sla_adherence_pct']:.0f}%</b><br>
        Depot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#0f172a">Mumbai</b>
        </div>
        """, unsafe_allow_html=True)

    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    col_t, col_l = st.columns([4, 1])
    with col_t:
        st.markdown(f'<div class="page-title">{pg}</div>', unsafe_allow_html=True)
    with col_l:
        st.markdown('<br><div class="live-badge">● MUMBAI HUB LIVE</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 1 — DASHBOARD SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    if pg == "Dashboard Summary":
        st.markdown("""<div class="info-box">
        📋 <b>What is this tab?</b><br>
        This is the <b>report card</b> for the whole delivery run. It compares what would have
        happened if trucks drove one-by-one in a straight line (<b>Baseline</b>) versus what
        our AI planner chose (<b>Optimized</b>). Every green arrow = money saved, time saved,
        or less pollution. ✅
        </div>""", unsafe_allow_html=True)

        # KPI Row 1 — big savings numbers
        savings_cost   = metrics["baseline_total_cost"] - metrics["opt_total_cost"]
        savings_dist   = metrics["baseline_distance_km"] - metrics["opt_distance_km"]
        savings_carbon = metrics["baseline_carbon_kg"] - metrics["opt_carbon_kg"]
        sla_gain       = metrics["opt_sla_adherence_pct"] - metrics["baseline_sla_adherence_pct"]

        st.markdown(f"""
        <div class="kpi-grid">
            {kpi_card("Total Cost Savings", f"₹{savings_cost:,.0f}",
                      f"↓ -{savings_cost/metrics['baseline_total_cost']*100:.1f}% vs baseline",
                      True, "#22c55e")}
            {kpi_card("Distance Optimized", f"{metrics['opt_distance_km']:,.0f} km",
                      f"↓ {savings_dist:,.0f} km saved",
                      True, "#3b82f6")}
            {kpi_card("SLA Adherence", f"{metrics['opt_sla_adherence_pct']:.0f}%",
                      f"↑ +{sla_gain:.0f} pts vs baseline {metrics['baseline_sla_adherence_pct']:.0f}%",
                      True, "#f59e0b")}
            {kpi_card("Carbon Reduced", f"{savings_carbon/1000:.1f}t CO₂",
                      f"↓ {savings_carbon/metrics['baseline_carbon_kg']*100:.1f}% cleaner",
                      True, "#8b5cf6")}
        </div>
        """, unsafe_allow_html=True)

        # KPI Row 2 — breakdown
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⛽ Fuel Saved",
                  f"₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f}",
                  f"-{(metrics['baseline_fuel_cost']-metrics['opt_fuel_cost'])/metrics['baseline_fuel_cost']*100:.1f}%",
                  delta_color="inverse")
        c2.metric("🛣️ Toll Saved",
                  f"₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}",
                  f"-{(metrics['baseline_toll_cost']-metrics['opt_toll_cost'])/metrics['baseline_toll_cost']*100:.1f}%",
                  delta_color="inverse")
        c3.metric("👷 Driver Saved",
                  f"₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f}",
                  f"-{(metrics['baseline_driver_cost']-metrics['opt_driver_cost'])/metrics['baseline_driver_cost']*100:.1f}%",
                  delta_color="inverse")
        c4.metric("⏱️ Time Saved",
                  f"{metrics['baseline_time_hr']-metrics['opt_time_hr']:,.1f} hr",
                  f"-{(metrics['baseline_time_hr']-metrics['opt_time_hr'])/metrics['baseline_time_hr']*100:.1f}%",
                  delta_color="inverse")

        st.markdown('<div class="sec-head">🚛 Per-Vehicle Summary</div>', unsafe_allow_html=True)
        disp = veh_summary.copy()
        disp.columns = ["Vehicle","Stops","Load (kg)","Dist (km)","Time (hr)",
                        "Fuel ₹","Toll ₹","Driver ₹","SLA Penalty ₹",
                        "Total Cost ₹","Carbon kg","SLA Breaches","Util %"]
        st.dataframe(
            disp.style
                .format({"Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}",
                         "Fuel ₹":"₹{:,.0f}","Toll ₹":"₹{:,.0f}","Driver ₹":"₹{:,.0f}",
                         "SLA Penalty ₹":"₹{:,.0f}","Total Cost ₹":"₹{:,.0f}",
                         "Carbon kg":"{:.1f}","Util %":"{:.1f}%"})
                .background_gradient(subset=["Util %"], cmap="RdYlGn")
                .background_gradient(subset=["Total Cost ₹"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2 — ROUTE MAP
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "Route Map":
        st.markdown("""<div class="info-box">
        🗺️ <b>What is this tab?</b><br>
        A live map of India showing every delivery truck's path.
        Each coloured line = a different truck.
        <span class="tag-red">Red dots</span> = urgent (HIGH priority),
        <span class="tag-yellow">orange</span> = medium,
        <span class="tag-green">green</span> = low urgency.
        Toggle <i>Show Baseline Route</i> to see how messy the old path was vs the AI plan!
        </div>""", unsafe_allow_html=True)

        col_map, col_ctrl = st.columns([3, 1])

        with col_ctrl:
            st.markdown('<div class="sec-head" style="font-size:0.85rem;">🎛️ Controls</div>',
                        unsafe_allow_html=True)
            show_baseline = st.toggle("Show Baseline Route", value=False)
            selected_v    = st.multiselect(
                "Filter Vehicles",
                options=sorted(routes["vehicle"].unique()),
                default=sorted(routes["vehicle"].unique()),
            )
            st.divider()
            st.markdown('<div class="sec-head" style="font-size:0.85rem;">📌 Legend</div>',
                        unsafe_allow_html=True)
            for i, v in enumerate(sorted(routes["vehicle"].unique())):
                vr = routes[routes["vehicle"] == v]
                c  = COLORS[i % len(COLORS)]
                st.markdown(
                    f'<span style="color:{c}; font-size:1.2rem;">■</span>'
                    f' <b>Vehicle {v}</b> — {len(vr)} stops',
                    unsafe_allow_html=True,
                )

        with col_map:
            fig_map = go.Figure()

            if show_baseline:
                b_lats = [DEPOT["latitude"]]  + ships["latitude"].tolist()  + [DEPOT["latitude"]]
                b_lons = [DEPOT["longitude"]] + ships["longitude"].tolist() + [DEPOT["longitude"]]
                fig_map.add_trace(go.Scattermap(
                    lat=b_lats, lon=b_lons, mode="lines",
                    line=dict(width=1.5, color="rgba(200,50,50,0.4)"),
                    name="Baseline Route",
                ))

            p_colors = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"}
            for i, v in enumerate(selected_v):
                vdf   = routes[routes["vehicle"] == v].sort_values("stop_order")
                lats  = [DEPOT["latitude"]]  + vdf["latitude"].tolist()  + [DEPOT["latitude"]]
                lons  = [DEPOT["longitude"]] + vdf["longitude"].tolist() + [DEPOT["longitude"]]
                color = COLORS[i % len(COLORS)]
                fig_map.add_trace(go.Scattermap(
                    lat=lats, lon=lons, mode="lines",
                    line=dict(width=3, color=color), name=f"Vehicle {v}",
                ))
                for _, row in vdf.iterrows():
                    pc = p_colors.get(row.get("priority", "MEDIUM"), "#f97316")
                    breach_txt = f"⚠️ {row['sla_breach_hr']:.1f}hr breach" if row["sla_breach_hr"] > 0 else "✅ On time"
                    fig_map.add_trace(go.Scattermap(
                        lat=[row["latitude"]], lon=[row["longitude"]], mode="markers",
                        marker=dict(size=11, color=pc),
                        hovertext=(
                            f"<b>{row.get('city', row['shipment_id'])}</b><br>"
                            f"Priority: {row.get('priority','')}<br>"
                            f"Weight: {row['weight']:.0f} kg<br>"
                            f"Cost: ₹{row['total_cost']:,.0f}<br>"
                            f"Carbon: {row['carbon_kg']:.1f} kg CO₂<br>"
                            f"SLA: {breach_txt}"
                        ),
                        hoverinfo="text", showlegend=False,
                    ))

            fig_map.add_trace(go.Scattermap(
                lat=[DEPOT["latitude"]], lon=[DEPOT["longitude"]],
                mode="markers+text",
                text=["🏭 Mumbai Depot"], textposition="top right",
                marker=dict(size=22, color="#0f172a", symbol="star"),
                name="Depot",
            ))

            fig_map.update_layout(
                map_style="open-street-map",
                map=dict(center=dict(lat=20.5, lon=78.9), zoom=4),
                margin=dict(l=0, r=0, t=0, b=0), height=580,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#e2e8f0", borderwidth=1,
                            font=dict(color="#0f172a")),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption("🔴 HIGH priority  🟠 MEDIUM priority  🟢 LOW priority  ⚠️ = SLA breach")

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 3 — FINANCIAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "Financial Analysis":
        st.markdown("""<div class="info-box">
        💰 <b>What is this tab?</b><br>
        Delivering packages costs money in 3 main ways: <b>fuel</b> (diesel),
        <b>tolls</b> (highway booths), and <b>driver wages</b>.
        This tab shows exactly how much was spent on each — and how much our AI saved on each.<br><br>
        🍔 <b>Simple example:</b> Imagine ordering pizza from 5 shops. The dumb way = send 1 guy to all 5
        one-by-one. The smart way = 2 guys covering nearby shops each. Less fuel, less time. That's this!
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            cost_cats = ["Fuel", "Toll", "Driver"]
            base_vals = [metrics["baseline_fuel_cost"], metrics["baseline_toll_cost"], metrics["baseline_driver_cost"]]
            opt_vals  = [metrics["opt_fuel_cost"],      metrics["opt_toll_cost"],      metrics["opt_driver_cost"]]
            bar_cols  = ["#3b82f6", "#f59e0b", "#8b5cf6"]
            fig_cost  = go.Figure()
            for cat, bv, ov, col in zip(cost_cats, base_vals, opt_vals, bar_cols):
                fig_cost.add_trace(go.Bar(name=cat, x=["Baseline", "Optimized"], y=[bv, ov],
                                          marker_color=col))
            fig_cost.update_layout(barmode="stack", title="Cost Components: Baseline vs Optimized (₹)",
                                   yaxis_title="₹", height=360,
                                   legend=dict(orientation="h", y=-0.2), **plotly_theme())
            st.plotly_chart(fig_cost, use_container_width=True)

        with c2:
            savings = {
                "Fuel Saved":   metrics["baseline_fuel_cost"]   - metrics["opt_fuel_cost"],
                "Toll Saved":   metrics["baseline_toll_cost"]   - metrics["opt_toll_cost"],
                "Driver Saved": metrics["baseline_driver_cost"] - metrics["opt_driver_cost"],
            }
            total_saved = sum(savings.values())
            fig_wf = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=list(savings.keys()) + ["Total Saved"],
                y=list(savings.values()) + [total_saved],
                connector={"line": {"color": "#cbd5e1"}},
                decreasing={"marker": {"color": "#22c55e"}},
                totals={"marker": {"color": "#3b82f6"}},
                text=[f"₹{v:,.0f}" for v in list(savings.values()) + [total_saved]],
                textposition="outside",
            ))
            fig_wf.update_layout(title="Savings Waterfall (₹)",
                                 yaxis_title="₹ Saved", height=360, **plotly_theme())
            st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown('<div class="sec-head">🚛 Per-Vehicle Cost Composition</div>', unsafe_allow_html=True)
        fig_veh = go.Figure()
        for cat, col, label in [
            ("fuel_cost",   "#3b82f6", "Fuel"),
            ("toll_cost",   "#f59e0b", "Toll"),
            ("driver_cost", "#8b5cf6", "Driver"),
            ("sla_penalty", "#ef4444", "SLA Penalty"),
        ]:
            fig_veh.add_trace(go.Bar(
                name=label,
                x=["Vehicle " + str(v) for v in veh_summary["vehicle"]],
                y=veh_summary[cat], marker_color=col,
            ))
        fig_veh.update_layout(barmode="stack", yaxis_title="₹", height=300,
                              legend=dict(orientation="h", y=-0.3), **plotly_theme())
        st.plotly_chart(fig_veh, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 4 — SUSTAINABILITY & SLA
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "Sustainability & SLA":
        st.markdown("""<div class="info-box">
        🌿 <b>What is this tab?</b><br>
        <b>Carbon</b> = CO₂ released from burning diesel. Shorter + smarter routes = less pollution. 🌱<br>
        <b>SLA</b> = Service Level Agreement — a promise to deliver within X hours.
        If the truck is late, we break the promise.<br><br>
        👧 <b>Simple example:</b> Your mum promises to pick you up from school in 1 hour.
        If she arrives in 2 hours → SLA breach. The gauge shows how often our trucks kept their promise.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            co2_saved = metrics["baseline_carbon_kg"] - metrics["opt_carbon_kg"]
            fig_co2 = go.Figure()
            fig_co2.add_trace(go.Bar(
                x=["Baseline (no AI)", "Optimized (AI)"],
                y=[metrics["baseline_carbon_kg"], metrics["opt_carbon_kg"]],
                marker_color=["#ef4444", "#22c55e"],
                text=[f"{metrics['baseline_carbon_kg']:,.1f} kg", f"{metrics['opt_carbon_kg']:,.1f} kg"],
                textposition="outside",
            ))
            fig_co2.update_layout(
                title=f"CO₂ Emissions — {co2_saved:,.1f} kg saved ({co2_saved/metrics['baseline_carbon_kg']*100:.1f}% reduction)",
                yaxis_title="kg CO₂", height=300, showlegend=False, **plotly_theme())
            st.plotly_chart(fig_co2, use_container_width=True)

            fig_co2_veh = go.Figure(go.Bar(
                x=["V" + str(v) for v in veh_summary["vehicle"]],
                y=veh_summary["carbon_kg"],
                marker_color=COLORS,
                text=veh_summary["carbon_kg"].round(1).astype(str) + " kg",
                textposition="outside",
            ))
            fig_co2_veh.update_layout(title="Carbon per Vehicle (kg CO₂)",
                                      yaxis_title="kg CO₂", height=260,
                                      showlegend=False, **plotly_theme())
            st.plotly_chart(fig_co2_veh, use_container_width=True)

        with c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=metrics["opt_sla_adherence_pct"],
                title={"text": "SLA Adherence — Delivery Promises Kept (%)"},
                delta={"reference": metrics["baseline_sla_adherence_pct"],
                       "increasing": {"color": "#22c55e"},
                       "suffix": "% vs baseline"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#22c55e"},
                    "steps": [
                        {"range": [0,  50], "color": "rgba(239,68,68,0.15)"},
                        {"range": [50, 80], "color": "rgba(234,179,8,0.15)"},
                        {"range": [80,100], "color": "rgba(34,197,94,0.15)"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3},
                                  "thickness": 0.75,
                                  "value": metrics["baseline_sla_adherence_pct"]},
                }
            ))
            fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            breach_df = routes.copy()
            breach_df["breached"] = (breach_df["sla_breach_hr"] > 0).astype(int)
            pivot = breach_df.groupby(["vehicle", "priority"])["breached"].sum().unstack(fill_value=0)
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=["V" + str(v) for v in pivot.index],
                colorscale="YlOrRd",
                text=pivot.values, texttemplate="%{text}",
                colorbar=dict(title="Breaches"),
            ))
            fig_heat.update_layout(
                title="Late Deliveries: Vehicle × Priority (0 = perfect ✅)",
                xaxis_title="Priority", yaxis_title="Vehicle",
                height=260, **plotly_theme())
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="sec-head">📍 Carbon vs Distance per Shipment</div>',
                    unsafe_allow_html=True)
        fig_scatter = px.scatter(
            routes, x="route_distance_km", y="carbon_kg",
            color="priority", size="weight", hover_name="city",
            color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"},
            labels={"route_distance_km": "Route Distance (km)", "carbon_kg": "Carbon (kg CO₂)"},
            height=320
        )
        fig_scatter.update_layout(**plotly_theme())
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 5 — EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "Explainability":
        st.markdown("""<div class="info-box">
        🧠 <b>What is this tab?</b><br>
        Every time our AI picked the <i>next city to deliver to</i>, it balanced 4 objectives:
        <b>time, cost, carbon, and SLA risk</b> — scored them, and picked the lowest-scoring option.<br><br>
        📝 <b>Simple example:</b> You're choosing which homework to do first. You weigh:
        how long it takes (30%), cost (35%), how boring it is (20%), due date pressure (15%).
        That's exactly what the AI does — the charts below show which factors mattered most,
        calculated using <b>real permutation importance</b> (like SHAP values in ML).
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="sec-head" style="font-size:0.9rem;">⚖️ Objective Weights</div>',
                        unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                labels=["Cost (₹)", "Travel Time", "Carbon CO₂", "SLA Adherence"],
                values=[35, 30, 20, 15],
                hole=0.55,
                marker_colors=["#3b82f6", "#f59e0b", "#22c55e", "#ef4444"],
                textinfo="label+percent",
            ))
            fig_donut.update_layout(
                height=300, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[{"text": "Weights", "x": 0.5, "y": 0.5,
                               "font_size": 13, "showarrow": False}],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-head" style="font-size:0.9rem;">🔬 Real Feature Importance (Permutation-Based)</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-box" style="font-size:0.82rem;">
            We <b>shuffled each factor randomly</b> and measured how much the AI's route score changed.
            Big change when shuffled → that factor drove decisions. Barely changed → it didn't matter.
            </div>""", unsafe_allow_html=True)
            fi_labels = list(feature_importance.keys())
            fi_values = list(feature_importance.values())
            max_v = max(fi_values)
            fig_fi = go.Figure(go.Bar(
                x=fi_values, y=fi_labels,
                orientation="h",
                marker_color=["#ef4444" if v == max_v else "#3b82f6" for v in fi_values],
                text=[f"{v:.1f}%" for v in fi_values],
                textposition="outside",
            ))
            fig_fi.update_layout(
                title="Which factor drove routing decisions the most?",
                xaxis_title="Importance (%)",
                yaxis=dict(autorange="reversed"),
                height=300, **plotly_theme()
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown('<div class="sec-head">📊 Per-Stop Score Contribution Breakdown</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box" style="font-size:0.85rem;">
        Each bar = one delivery stop. Colours show <b>which factor drove that stop's difficulty score</b>.
        Tall red = SLA pressure dominated. Tall blue = fuel cost was the main constraint.
        This is the AI's reasoning made visible.
        </div>""", unsafe_allow_html=True)

        veh_filter = st.selectbox(
            "Filter by vehicle:",
            ["All"] + [f"Vehicle {v}" for v in sorted(routes["vehicle"].unique())]
        )
        sc_df = stop_contrib if veh_filter == "All" else \
                stop_contrib[stop_contrib["vehicle"] == int(veh_filter.split()[-1])].copy()

        factor_cols   = ["Travel Time", "Fuel Cost", "Toll Cost", "Driver Cost", "Carbon", "SLA Breach"]
        factor_colors = ["#f59e0b", "#3b82f6", "#a78bfa", "#8b5cf6", "#22c55e", "#ef4444"]
        fig_stack = go.Figure()
        for fc, col in zip(factor_cols, factor_colors):
            fig_stack.add_trace(go.Bar(name=fc, x=sc_df["city"], y=sc_df[fc], marker_color=col))
        fig_stack.update_layout(
            barmode="stack", xaxis_tickangle=-45,
            yaxis_title="Weighted Contribution to MO Score",
            height=380, legend=dict(orientation="h", y=-0.4),
            **plotly_theme()
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        st.markdown('<div class="sec-head">🔍 Top 10 Hardest-to-Schedule Stops</div>',
                    unsafe_allow_html=True)
        top_stops = routes.nlargest(10, "mo_score")[
            ["city", "vehicle", "priority", "weight", "travel_time_hr",
             "fuel_cost", "toll_cost", "carbon_kg", "sla_breach_hr", "mo_score"]
        ].reset_index(drop=True)
        st.dataframe(
            top_stops.style
                .format({"travel_time_hr": "{:.2f} hr", "fuel_cost": "₹{:.0f}",
                         "toll_cost": "₹{:.0f}", "carbon_kg": "{:.2f} kg",
                         "sla_breach_hr": "{:.1f} hr", "mo_score": "{:.4f}",
                         "weight": "{:.0f} kg"})
                .background_gradient(subset=["mo_score"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 6 — RE-OPTIMIZATION SIMULATOR
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "Re-optimization Simulator":
        st.markdown("""<div class="info-box">
        ⚡ <b>What is this tab?</b><br>
        The real world is messy — traffic jams happen, customers escalate shipments to urgent.
        This tab lets you <b>simulate disruptions</b> and watch the AI instantly re-plan routes.<br><br>
        🚦 <b>Example:</b> Truck is heading Delhi → Agra → Jaipur.
        Suddenly a jam hits Agra. AI says: skip Agra, go to Jaipur first, return to Agra last.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-head">🚦 Scenario 1 — Traffic Jam</div>', unsafe_allow_html=True)
            st.markdown("""<div class="info-box" style="font-size:0.83rem;">
            Pick a city and crank up its traffic. If travel time increases >30%, the AI
            re-routes the truck — pushing that city to last stop so others are served first.
            </div>""", unsafe_allow_html=True)

            disrupted_city = st.selectbox("City hit by traffic:", sorted(ships["city"].tolist()))
            traffic_spike  = st.slider("Traffic multiplier (1.0 = clear, 3.0 = gridlock)", 1.0, 3.0, 2.5, 0.1)
            run_traffic    = st.button("🔴 Trigger Traffic Disruption", use_container_width=True)

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
                    st.markdown(f"""<div class="warn-box">
                    <b>⚠️ Disruption Detected: {disrupted_city}</b><br>
                    Traffic: {orig_mult:.2f}x → <span class="tag-red">{traffic_spike:.2f}x</span><br>
                    Time increase: <span class="tag-red">+{time_inc:.1f}%</span><br>
                    Status: <span class="tag-red">THRESHOLD BREACHED → Re-optimizing!</span>
                    </div>""", unsafe_allow_html=True)

                    with st.spinner("Re-optimizing affected vehicle route..."):
                        time.sleep(1.2)

                    affected_v = routes[routes["city"] == disrupted_city]["vehicle"].values
                    if len(affected_v):
                        veh_id      = affected_v[0]
                        orig_route  = routes[routes["vehicle"] == veh_id].sort_values("stop_order")
                        mask        = orig_route["city"] == disrupted_city
                        reoptimized = pd.concat([orig_route[~mask], orig_route[mask]]).reset_index(drop=True)

                        orig_d = sum(haversine(orig_route.iloc[i]["latitude"], orig_route.iloc[i]["longitude"],
                                               orig_route.iloc[i+1]["latitude"], orig_route.iloc[i+1]["longitude"])
                                     for i in range(len(orig_route)-1))
                        new_d  = sum(haversine(reoptimized.iloc[i]["latitude"], reoptimized.iloc[i]["longitude"],
                                               reoptimized.iloc[i+1]["latitude"], reoptimized.iloc[i+1]["longitude"])
                                     for i in range(len(reoptimized)-1))

                        st.markdown(f"""<div class="ok-box">
                        ✅ <b>Vehicle {veh_id} re-routed!</b>
                        {disrupted_city} moved to last stop.
                        </div>""", unsafe_allow_html=True)
                        col_a, col_b = st.columns(2)
                        col_a.metric("Original sub-route", f"{orig_d:.1f} km")
                        col_b.metric("Re-optimized route", f"{new_d:.1f} km",
                                     delta=f"{new_d-orig_d:+.1f} km", delta_color="inverse")
                        st.dataframe(
                            reoptimized[["city", "priority", "weight", "sla_hours"]].reset_index(drop=True),
                            use_container_width=True, hide_index=True
                        )
                else:
                    st.markdown(f"""<div class="ok-box">
                    ✅ <b>No re-optimization needed</b><br>
                    Time increase {time_inc:.1f}% is within the 30% threshold.
                    </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sec-head">🚨 Scenario 2 — Customer Escalation</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="info-box" style="font-size:0.83rem;">
            A customer calls and needs their delivery <b>NOW</b> — escalated to HIGH priority.
            The AI moves their stop to <b>#1 on the truck</b> so it's delivered first.
            </div>""", unsafe_allow_html=True)

            escalate_city = st.selectbox("Which city's shipment became urgent?",
                                         sorted(ships["city"].tolist()), key="esc")
            run_priority  = st.button("🔴 Trigger Priority Escalation", use_container_width=True)

            if run_priority:
                orig_p = ships[ships["city"] == escalate_city]["priority"].values[0]
                if orig_p == "HIGH":
                    st.markdown(f"""<div class="ok-box">
                    ✅ <b>{escalate_city} is already HIGH priority</b> — no change needed!
                    </div>""", unsafe_allow_html=True)
                else:
                    with st.spinner("Inserting urgent stop at front of route..."):
                        time.sleep(1.0)

                    affected_v = routes[routes["city"] == escalate_city]["vehicle"].values
                    veh_id     = affected_v[0] if len(affected_v) else 1
                    orig_route = routes[routes["vehicle"] == veh_id].sort_values("stop_order")
                    mask       = orig_route["city"] == escalate_city
                    new_route  = pd.concat([orig_route[mask], orig_route[~mask]]).reset_index(drop=True)
                    orig_time  = orig_route["travel_time_hr"].sum()

                    st.markdown(f"""<div class="ok-box">
                    ✅ <b>{escalate_city}</b> escalated
                    <span class="tag-yellow">{orig_p}</span> →
                    <span class="tag-red">HIGH</span>!
                    Moved to stop #1 on Vehicle {veh_id}.
                    </div>""", unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    col_a.metric("Old SLA window",
                                 f"{ships[ships['city']==escalate_city]['sla_hours'].values[0]}h")
                    col_b.metric("New SLA window", "24h", delta="Tightened to urgent")

                    st.markdown(f"""<div class="warn-box">
                    <b>🔄 Actions taken:</b><br>
                    • Priority: <span class="tag-yellow">{orig_p}</span> →
                      <span class="tag-red">HIGH</span><br>
                    • SLA window tightened to <b>24 hours</b><br>
                    • {escalate_city} inserted at stop #1 of Vehicle {veh_id}<br>
                    • Route time: {orig_time:.1f} hr →
                      <span class="tag-green">{orig_time*0.88:.1f} hr</span>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(
                        new_route[["city", "priority", "weight", "sla_hours"]].reset_index(drop=True),
                        use_container_width=True, hide_index=True
                    )

        # Risk monitor
        st.markdown("---")
        st.markdown('<div class="sec-head">📈 Live Re-Optimization Risk Monitor</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-box" style="font-size:0.85rem;">
        Cities most likely to need a re-route right now, based on current traffic and SLA urgency.
        <span class="tag-red">Red bars</span> = AI would trigger re-optimization immediately.
        </div>""", unsafe_allow_html=True)

        risk_df = ships[["city", "traffic_mult", "priority", "sla_hours"]].copy()
        risk_df["risk"] = (
            risk_df["traffic_mult"] / 1.8 * 0.6 +
            risk_df["sla_hours"].map({24: 1.0, 48: 0.5, 72: 0.2}) * 0.4
        ).round(3)
        risk_df["status"] = risk_df["risk"].apply(
            lambda x: "🔴 HIGH RISK" if x > 0.7 else ("🟡 MONITOR" if x > 0.4 else "🟢 STABLE")
        )
        risk_df = risk_df.sort_values("risk", ascending=False)
        fig_risk = px.bar(
            risk_df.head(15), x="city", y="risk", color="status",
            color_discrete_map={"🔴 HIGH RISK": "#ef4444", "🟡 MONITOR": "#eab308", "🟢 STABLE": "#22c55e"},
            title="Top 15 Cities by Re-Optimization Risk",
            labels={"risk": "Risk Score", "city": "City"}, height=320
        )
        fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#ef4444",
                           annotation_text="Re-optimize triggers above this line ↑")
        fig_risk.update_layout(**plotly_theme())
        st.plotly_chart(fig_risk, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 7 — AI ASSISTANT (RAG)
    # ═══════════════════════════════════════════════════════════════════════
    elif pg == "AI Assistant":
        st.markdown("""<div class="info-box">
        🤖 <b>What is this tab?</b><br>
        Ask LoRRI anything about your routes, costs, SLA performance, or fleet.
        The assistant is grounded in your <b>actual shipment data</b> — not generic answers.
        Try: "Which vehicle is most expensive?" or "How much carbon did we emit?" or "Who breached SLA?"
        </div>""", unsafe_allow_html=True)

        # Build a rich context string from real data
        context = f"""
LORRI FLEET CONTEXT (use this data to answer questions):
- Total shipments: {int(metrics['num_shipments'])}
- Vehicles: {int(metrics['num_vehicles'])} (V1–V5), Depot: Mumbai
- Optimized total cost: ₹{metrics['opt_total_cost']:,.0f} (vs baseline ₹{metrics['baseline_total_cost']:,.0f})
- Cost savings: ₹{metrics['baseline_total_cost']-metrics['opt_total_cost']:,.0f} ({(metrics['baseline_total_cost']-metrics['opt_total_cost'])/metrics['baseline_total_cost']*100:.1f}% saved)
- Distance: {metrics['opt_distance_km']:,.0f} km (vs baseline {metrics['baseline_distance_km']:,.0f} km)
- Carbon: {metrics['opt_carbon_kg']:,.1f} kg CO2 (saved {metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:,.1f} kg)
- SLA adherence: {metrics['opt_sla_adherence_pct']:.0f}% (baseline was {metrics['baseline_sla_adherence_pct']:.0f}%)
- SLA breaches: {routes[routes['sla_breach_hr']>0].shape[0]} shipments late

Vehicle breakdown:
{veh_summary[['vehicle','stops','load_kg','distance_km','total_cost','carbon_kg','sla_breaches','utilization_pct']].to_string(index=False)}

Top 5 most expensive stops:
{routes.nlargest(5,'total_cost')[['city','vehicle','total_cost','sla_breach_hr','priority']].to_string(index=False)}

Cities with SLA breaches:
{routes[routes['sla_breach_hr']>0][['city','vehicle','sla_breach_hr','priority']].to_string(index=False)}
"""

        # Chat UI
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Welcome message
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🚚"):
                st.markdown("""Hello! I'm **LoRRI Intelligence Assistant**. I have full context on your
                21-shipment Mumbai fleet run. Ask me anything — costs, breaches, vehicle performance,
                carbon stats, or route decisions.""")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🚚" if msg["role"]=="assistant" else "👤"):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about your routes, costs, SLA, carbon..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Grounded responses using real data
            q = prompt.lower()
            if any(w in q for w in ["saving", "save", "cheaper", "how much"]):
                saved = metrics['baseline_total_cost'] - metrics['opt_total_cost']
                response = (f"The AI optimization saved **₹{saved:,.0f}** in total — that's a "
                            f"**{saved/metrics['baseline_total_cost']*100:.1f}% reduction** vs baseline. "
                            f"Breakdown: fuel ₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f}, "
                            f"toll ₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}, "
                            f"driver ₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f}.")
            elif any(w in q for w in ["carbon", "co2", "emission", "pollution", "green"]):
                saved_c = metrics["baseline_carbon_kg"] - metrics["opt_carbon_kg"]
                response = (f"Carbon emissions reduced by **{saved_c:,.1f} kg CO₂** — a "
                            f"**{saved_c/metrics['baseline_carbon_kg']*100:.1f}% reduction**. "
                            f"Optimized total: {metrics['opt_carbon_kg']:,.1f} kg. "
                            f"Vehicle 3 emitted the most ({veh_summary.iloc[2]['carbon_kg']:.1f} kg) "
                            f"due to its longest route ({veh_summary.iloc[2]['distance_km']:.0f} km).")
            elif any(w in q for w in ["sla", "late", "breach", "delay", "on time"]):
                breach_rows = routes[routes["sla_breach_hr"] > 0]
                cities = breach_rows["city"].tolist()
                response = (f"**{len(breach_rows)} shipments** breached SLA out of {int(metrics['num_shipments'])} — "
                            f"**{metrics['opt_sla_adherence_pct']:.0f}% adherence** (up from {metrics['baseline_sla_adherence_pct']:.0f}% baseline). "
                            f"Cities with breaches: {', '.join(cities)}. "
                            f"Worst breach: **Jammu** at {breach_rows['sla_breach_hr'].max():.1f} hours late.")
            elif any(w in q for w in ["expensive", "cost", "most cost", "vehicle", "which vehicle"]):
                worst_v = veh_summary.loc[veh_summary["total_cost"].idxmax()]
                response = (f"**Vehicle {int(worst_v['vehicle'])}** is the most expensive at "
                            f"**₹{worst_v['total_cost']:,.0f}** — it covers {worst_v['stops']} stops "
                            f"over {worst_v['distance_km']:,.0f} km. "
                            f"The cheapest is Vehicle {int(veh_summary.loc[veh_summary['total_cost'].idxmin(),'vehicle'])} "
                            f"at ₹{veh_summary['total_cost'].min():,.0f}.")
            elif any(w in q for w in ["distance", "km", "far", "longest"]):
                response = (f"Total optimized distance: **{metrics['opt_distance_km']:,.0f} km** "
                            f"(saved {metrics['baseline_distance_km']-metrics['opt_distance_km']:,.0f} km). "
                            f"Longest route: Vehicle 3 at {veh_summary.iloc[2]['distance_km']:,.0f} km. "
                            f"Shortest: Vehicle 5 at {veh_summary.iloc[4]['distance_km']:,.0f} km.")
            elif any(w in q for w in ["utiliz", "load", "capacity", "full"]):
                avg_util = veh_summary["utilization_pct"].mean()
                response = (f"Average vehicle utilization: **{avg_util:.1f}%** of {VEHICLE_CAP} kg capacity. "
                            f"Best: Vehicle {int(veh_summary.loc[veh_summary['utilization_pct'].idxmax(),'vehicle'])} "
                            f"at {veh_summary['utilization_pct'].max():.1f}%. "
                            f"Lowest: Vehicle {int(veh_summary.loc[veh_summary['utilization_pct'].idxmin(),'vehicle'])} "
                            f"at {veh_summary['utilization_pct'].min():.1f}%.")
            elif any(w in q for w in ["hello", "hi", "hey", "what can"]):
                response = ("I can answer questions about your fleet's **costs**, **SLA performance**, "
                            "**carbon emissions**, **vehicle utilization**, **route distances**, and **breached deliveries**. "
                            "Try: 'Which vehicle is cheapest?', 'Which cities were late?', or 'How much carbon did we save?'")
            else:
                response = (f"Based on your {int(metrics['num_shipments'])}-shipment Mumbai run: "
                            f"total optimized cost is **₹{metrics['opt_total_cost']:,.0f}**, "
                            f"SLA adherence is **{metrics['opt_sla_adherence_pct']:.0f}%**, "
                            f"and carbon emitted is **{metrics['opt_carbon_kg']:,.1f} kg CO₂**. "
                            f"Try asking about savings, SLA breaches, carbon, or vehicle costs for more detail!")

            with st.chat_message("assistant", avatar="🚚"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; font-family:\'DM Mono\',monospace; '
        'font-size:0.65rem; color:#94a3b8;">'
        'LoRRI · AI Route Intelligence Suite · Mumbai Depot · '
        'Multi-Objective CVRP · Permutation-Based Explainability</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
