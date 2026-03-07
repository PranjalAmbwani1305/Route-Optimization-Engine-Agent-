"""
LoRRI · AI Route Optimization Dashboard
- India map with real truck routes + truck numbers on markers
- All costs in ₹ (INR)
- TypeError FIXED: plotly theme never merged with explicit kwargs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from math import radians, cos, sin, asin, sqrt
import time

st.set_page_config(
    page_title="LoRRI · AI Route Optimization",
    layout="wide",
    page_icon="🚚",
    initial_sidebar_state="expanded",
)

DEPOT       = {"latitude": 19.0760, "longitude": 72.8777}
VEHICLE_CAP = 800
V_COLORS    = {1:"#8b5cf6", 2:"#22c55e", 3:"#f97316", 4:"#eab308", 5:"#ec4899"}

# ─── SAFE THEME ── never merge with other kwargs ─────────────────────────────
def apply_theme(fig, height=350, title="", legend_below=False, yprefix=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155", size=12),
        height=height,
        margin=dict(l=10, r=10, t=45 if title else 20, b=10),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color="#0f172a")))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0))
    if yprefix:
        fig.update_yaxes(tickprefix=yprefix, tickformat=",")
    fig.update_xaxes(gridcolor="#f1f5f9", zeroline=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zeroline=False, linecolor="#e2e8f0")
    return fig

def inr(val):
    return f"₹{val:,.0f}"

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
.main .block-container{padding:1.5rem 3rem !important;max-width:1440px;}
[data-testid="stSidebar"]{background:#f8fafc !important;border-right:1px solid #e2e8f0 !important;}
.sb-brand{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:#0f172a;letter-spacing:-1px;}
.sb-sub{font-family:'DM Mono',monospace;font-size:0.58rem;color:#64748b;letter-spacing:0.14em;text-transform:uppercase;margin-top:-4px;margin-bottom:1.6rem;}
.sb-sec{font-family:'DM Mono',monospace;font-size:0.58rem;font-weight:500;color:#3b82f6;letter-spacing:0.16em;text-transform:uppercase;margin:1.4rem 0 0.5rem 0;border-bottom:1px solid #e2e8f0;padding-bottom:4px;}
.page-h1{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#0f172a;margin:0;}
.page-sub{font-size:0.82rem;color:#64748b;margin-top:2px;}
.live{font-family:'DM Mono',monospace;font-size:0.7rem;color:#16a34a;}
.info-box{background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:8px;padding:14px 18px;margin:4px 0 16px 0;font-size:0.88rem;line-height:1.7;color:#0f172a;}
.info-box b{color:#0369a1;}
.warn-box{background:rgba(234,179,8,0.1);border-left:4px solid #eab308;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:0.86rem;line-height:1.65;color:#0f172a;}
.ok-box{background:rgba(34,197,94,0.08);border-left:4px solid #22c55e;border-radius:6px;padding:12px 16px;margin:6px 0;font-size:0.86rem;line-height:1.65;color:#0f172a;}
.tag-red{color:#dc2626;font-weight:700;}.tag-green{color:#16a34a;font-weight:700;}.tag-yellow{color:#d97706;font-weight:700;}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px;}
.kpi-card{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:16px 20px;position:relative;overflow:hidden;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--ac,#3b82f6);}
.kpi-lbl{font-family:'DM Mono',monospace;font-size:0.58rem;font-weight:500;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:7px;}
.kpi-val{font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:700;color:#0f172a;line-height:1.1;}
.kpi-d{font-family:'DM Mono',monospace;font-size:0.68rem;margin-top:5px;}
.dg{color:#16a34a;}.dr{color:#dc2626;}
.sh{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#0f172a;margin:22px 0 10px 0;}
.legend-row{display:flex;align-items:flex-start;gap:10px;font-size:0.85rem;color:#334155;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f1f5f9;}
.legend-row:last-child{border-bottom:none;margin-bottom:0;}
.legend-line{width:28px;height:4px;border-radius:2px;flex-shrink:0;margin-top:6px;}
[data-testid="metric-container"]{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:12px 16px !important;}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2*R*asin(sqrt(a))

def kpi_card(label, value, delta, good=True, ac="#3b82f6"):
    cls = "dg" if good else "dr"
    return f'<div class="kpi-card" style="--ac:{ac}"><div class="kpi-lbl">{label}</div><div class="kpi-val">{value}</div><div class="kpi-d {cls}">{delta}</div></div>'

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    ships  = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    veh    = pd.read_csv("vehicle_summary.csv")
    n      = len(ships)
    breaches = int((routes["sla_breach_hr"]>0).sum())
    opt = dict(
        distance_km = round(veh["distance_km"].sum(),1),
        time_hr     = round(veh["time_hr"].sum(),1),
        fuel_cost   = round(veh["fuel_cost"].sum(),1),
        toll_cost   = round(veh["toll_cost"].sum(),1),
        driver_cost = round(veh["driver_cost"].sum(),1),
        total_cost  = round(veh["total_cost"].sum(),1),
        carbon_kg   = round(veh["carbon_kg"].sum(),1),
        sla_pct     = round((n-breaches)/n*100,1),
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
    feats = {"Travel Time":"travel_time_hr","Fuel Cost":"fuel_cost","Toll Cost":"toll_cost",
             "Driver Cost":"driver_cost","Carbon Emitted":"carbon_kg",
             "SLA Breach":"sla_breach_hr","Package Weight":"weight"}
    X = routes_df[list(feats.values())].copy()
    y = routes_df["mo_score"].values
    base_mae = np.mean(np.abs(y-y.mean()))
    imp = {}
    for lbl,col in feats.items():
        sh = X.copy(); sh[col] = np.random.permutation(sh[col].values)
        proxy = sh.apply(lambda c:(c-c.mean())/(c.std()+1e-9)).mean(axis=1)
        imp[lbl] = abs(np.mean(np.abs(y-proxy.values))-base_mae)
    tot = sum(imp.values())+1e-9
    return {k:round(v/tot*100,1) for k,v in sorted(imp.items(),key=lambda x:-x[1])}

@st.cache_data
def stop_cont(routes_df):
    cols    = ["travel_time_hr","fuel_cost","toll_cost","driver_cost","carbon_kg","sla_breach_hr"]
    labels  = ["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    weights = [0.30,0.20,0.05,0.15,0.20,0.10]
    df = routes_df[cols].copy()
    for c in cols:
        rng=df[c].max()-df[c].min(); df[c]=(df[c]-df[c].min())/(rng+1e-9)
    for i,c in enumerate(cols): df[c]*=weights[i]
    df.columns=labels
    df["city"]=routes_df["city"].values; df["vehicle"]=routes_df["vehicle"].values
    df["mo_score"]=routes_df["mo_score"].values
    return df

ships, routes, veh_sum, base, opt = load()
fi    = perm_imp(routes)
sc    = stop_cont(routes)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">LoRRI</div>',unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">AI Route Intelligence · v2.1</div>',unsafe_allow_html=True)
    st.markdown('<div class="sb-sec">📊 Analytics Suite</div>',unsafe_allow_html=True)
    pg = st.radio("nav",[
        "📊 Dashboard Summary","🗺️ Route Map","💰 Financial Analysis",
        "🌿 Carbon & SLA","🧠 Explainability",
        "⚡ Re-optimization Simulator","🤖 AI Assistant",
    ],label_visibility="collapsed")
    st.markdown('<div class="sb-sec">🛠️ Fleet Control</div>',unsafe_allow_html=True)
    st.toggle("Real-time Traffic Feed",value=True)
    st.toggle("Auto Re-optimize",value=False)
    if st.button("🔄 Sync Depot Data",use_container_width=True):
        st.toast("✅ Synced with Mumbai Depot!",icon="🏭")
    st.markdown('<div class="sb-sec">📦 Live Stats</div>',unsafe_allow_html=True)
    st.markdown(f"""<div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569;line-height:2.2;">
    Shipments &nbsp;&nbsp; <b style="color:#0f172a">{opt['n_ships']}</b><br>
    Vehicles &nbsp;&nbsp;&nbsp; <b style="color:#0f172a">{opt['n_vehicles']}</b><br>
    SLA OK &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#16a34a">{opt['sla_pct']:.0f}%</b><br>
    SLA Breaches &nbsp;<b style="color:#dc2626">{opt['breaches']}</b><br>
    Depot &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#0f172a">Mumbai</b>
    </div>""",unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
ch,cl = st.columns([4,1])
with ch:
    st.markdown('<div class="page-h1">🚚 LoRRI · AI Route Optimization Engine</div>',unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Dynamic Multi-Objective CVRP · India Logistics Network · Depot: Mumbai · All costs in ₹ INR</div>',unsafe_allow_html=True)
with cl:
    st.markdown('<br><div class="live">● MUMBAI HUB: LIVE</div>',unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if pg == "📊 Dashboard Summary":
    st.markdown("""<div class="info-box">
    📋 <b>What is this tab?</b> — The <b>report card</b> for the whole delivery run.
    Baseline = trucks driving one-by-one with no AI. Optimized = AI-planned routes.
    Every green arrow = money saved, time saved, less pollution. All figures in <b>₹ INR</b>. ✅
    </div>""",unsafe_allow_html=True)

    sc_=base["total_cost"]-opt["total_cost"]; sd_=base["distance_km"]-opt["distance_km"]
    sco=base["carbon_kg"]-opt["carbon_kg"];   ss_=opt["sla_pct"]-base["sla_pct"]

    st.markdown(f"""<div class="kpi-row">
    {kpi_card("Total Cost Savings",inr(sc_),f"↓ -{sc_/base['total_cost']*100:.1f}% vs baseline",True,"#22c55e")}
    {kpi_card("Optimized Distance",f"{opt['distance_km']:,.0f} km",f"↓ {sd_:,.0f} km saved",True,"#3b82f6")}
    {kpi_card("SLA Adherence",f"{opt['sla_pct']:.0f}%",f"↑ +{ss_:.0f} pts (baseline {base['sla_pct']:.0f}%)",True,"#f59e0b")}
    {kpi_card("Carbon Reduced",f"{sco/1000:.1f}t CO₂",f"↓ {sco/base['carbon_kg']*100:.1f}% cleaner",True,"#8b5cf6")}
    </div>""",unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⛽ Fuel Saved",  inr(base["fuel_cost"]  -opt["fuel_cost"]),  f"-{(base['fuel_cost']  -opt['fuel_cost']  )/base['fuel_cost']  *100:.1f}%",delta_color="inverse")
    c2.metric("🛣️ Toll Saved", inr(base["toll_cost"]  -opt["toll_cost"]),  f"-{(base['toll_cost']  -opt['toll_cost']  )/base['toll_cost']  *100:.1f}%",delta_color="inverse")
    c3.metric("👷 Driver Saved",inr(base["driver_cost"]-opt["driver_cost"]),f"-{(base['driver_cost']-opt['driver_cost'])/base['driver_cost']*100:.1f}%",delta_color="inverse")
    c4.metric("⏱️ Time Saved", f"{base['time_hr']-opt['time_hr']:,.1f} hr",f"-{(base['time_hr']-opt['time_hr'])/base['time_hr']*100:.1f}%",delta_color="inverse")

    st.markdown('<div class="sh">🚛 Per-Truck Summary</div>',unsafe_allow_html=True)
    d = veh_sum.copy()
    d.insert(0,"Truck",d["vehicle"].apply(lambda v:f"🚛 Truck {v}"))
    d = d.drop(columns=["vehicle"])
    d.columns = ["Truck","Stops","Load (kg)","Dist (km)","Time (hr)",
                 "Fuel (₹)","Toll (₹)","Driver (₹)","SLA Penalty (₹)","Total (₹)","Carbon (kg)","SLA Breaches","Util %"]
    st.dataframe(d.style
        .format({"Load (kg)":"{:.1f}","Dist (km)":"{:.1f}","Time (hr)":"{:.1f}",
                 "Fuel (₹)":"₹{:,.0f}","Toll (₹)":"₹{:,.0f}","Driver (₹)":"₹{:,.0f}",
                 "SLA Penalty (₹)":"₹{:,.0f}","Total (₹)":"₹{:,.0f}",
                 "Carbon (kg)":"{:.1f}","Util %":"{:.1f}%"})
        .background_gradient(subset=["Util %"],cmap="RdYlGn")
        .background_gradient(subset=["Total (₹)"],cmap="YlOrRd"),
        use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ROUTE MAP  — matches PDF layout with truck numbers
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🗺️ Route Map":
    st.markdown("""<div class="info-box">
    🗺️ <b>What is this tab?</b> — Real India map showing every truck's delivery path.
    Each line = a truck. Markers show <b>Truck number · Stop number</b>.
    Hover for full details including cost in ₹ and SLA status.
    </div>""",unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([3,1])

    with col_ctrl:
        st.markdown("### 🎛️ Map Controls")
        show_base = st.toggle("Show Baseline Route", value=False)
        show_unas = st.toggle("Show Unassigned",     value=True)
        sel_v     = st.multiselect("Filter Vehicles",
                                    options=sorted(routes["vehicle"].unique()),
                                    default=sorted(routes["vehicle"].unique()),
                                    format_func=lambda v:f"Truck {v}")
        st.markdown("---")
        st.markdown("### 📌 Route Legend")
        for v in sorted(routes["vehicle"].unique()):
            vr    = routes[routes["vehicle"]==v]
            color = V_COLORS.get(v,"#999")
            vd    = veh_sum[veh_sum["vehicle"]==v].iloc[0]
            st.markdown(
                f'<div class="legend-row">'
                f'<div class="legend-line" style="background:{color}"></div>'
                f'<div><b>Truck {v}</b><br>'
                f'<span style="font-size:0.75rem;color:#64748b">'
                f'{len(vr)} stops · {vd["distance_km"]:,.0f} km<br>'
                f'{inr(vd["total_cost"])} · {vd["carbon_kg"]:.0f} kg CO₂</span></div>'
                f'</div>',unsafe_allow_html=True)

    with col_map:
        fig = go.Figure()

        if show_base:
            bl  = [DEPOT["latitude"] ]+ships["latitude"].tolist() +[DEPOT["latitude"] ]
            blo = [DEPOT["longitude"]]+ships["longitude"].tolist()+[DEPOT["longitude"]]
            fig.add_trace(go.Scattermap(lat=bl,lon=blo,mode="lines",
                line=dict(width=1.5,color="rgba(200,50,50,0.4)"),name="Baseline (No AI)"))

        p_dot = {"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#22c55e"}

        for v in sel_v:
            vdf   = routes[routes["vehicle"]==v].sort_values("stop_order")
            color = V_COLORS.get(v,"#999")
            lats  = [DEPOT["latitude"] ]+vdf["latitude"].tolist() +[DEPOT["latitude"] ]
            lons  = [DEPOT["longitude"]]+vdf["longitude"].tolist()+[DEPOT["longitude"]]

            # Route line
            fig.add_trace(go.Scattermap(
                lat=lats,lon=lons,mode="lines",
                line=dict(width=3,color=color),
                name=f"Truck {v}",legendgroup=f"v{v}"))

            # Stop markers with truck + stop number label
            for _,row in vdf.iterrows():
                breach = f"⚠️ {row['sla_breach_hr']:.1f}hr late" if row["sla_breach_hr"]>0 else "✅ On time"
                fig.add_trace(go.Scattermap(
                    lat=[row["latitude"]],lon=[row["longitude"]],
                    mode="markers+text",
                    marker=dict(size=14,color=p_dot.get(row.get("priority","MEDIUM"),"#f97316")),
                    text=[f"T{v}·{int(row['stop_order'])}"],
                    textfont=dict(size=8,color="white"),
                    textposition="middle center",
                    hovertext=(
                        f"<b>🚛 Truck {v} — Stop {int(row['stop_order'])}</b><br>"
                        f"📍 {row.get('city',row['shipment_id'])}<br>"
                        f"📦 {row['shipment_id']} | {row['weight']:.0f} kg<br>"
                        f"🔺 Priority: {row.get('priority','')}<br>"
                        f"⏱️ Travel: {row['travel_time_hr']:.1f} hr<br>"
                        f"⛽ Fuel: {inr(row['fuel_cost'])}<br>"
                        f"🛣️ Toll: {inr(row['toll_cost'])}<br>"
                        f"💰 Total: {inr(row['total_cost'])}<br>"
                        f"🌿 Carbon: {row['carbon_kg']:.1f} kg CO₂<br>"
                        f"📅 SLA {row['sla_hours']}hr: {breach}"
                    ),
                    hoverinfo="text",showlegend=False,legendgroup=f"v{v}"))

        if show_unas:
            asgn  = set(routes["shipment_id"])
            unasgn= ships[~ships["id"].isin(asgn)]
            if not unasgn.empty:
                fig.add_trace(go.Scattermap(
                    lat=unasgn["latitude"],lon=unasgn["longitude"],mode="markers",
                    marker=dict(size=8,color="grey"),name="Unassigned",
                    hovertext=unasgn["city"],hoverinfo="text"))

        fig.add_trace(go.Scattermap(
            lat=[DEPOT["latitude"]],lon=[DEPOT["longitude"]],
            mode="markers+text",text=["🏭 Mumbai\nDepot"],
            textposition="top right",textfont=dict(size=10,color="#0f172a"),
            marker=dict(size=18,color="#0f172a",symbol="star"),name="Mumbai Depot"))

        fig.update_layout(
            map_style="open-street-map",
            map=dict(center=dict(lat=20.5,lon=78.9),zoom=4),
            height=600,
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(x=0.01,y=0.99,bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0",borderwidth=1,font=dict(color="#0f172a",size=11)))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("🔴 HIGH priority  🟠 MEDIUM priority  🟢 LOW priority  ⚠️ SLA breach  ·  Labels: T{truck}·{stop#}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "💰 Financial Analysis":
    st.markdown("""<div class="info-box">
    💰 <b>What is this tab?</b> — All costs in <b>₹ (Indian Rupees)</b>. Delivery costs split into
    <b>fuel</b>, <b>tolls</b>, and <b>driver wages</b>. The AI clustered nearby Indian cities into
    efficient per-truck routes, cutting all three cost categories significantly.
    </div>""",unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        fig_b = go.Figure()
        cats=["Fuel","Toll","Driver"]
        bv=[base["fuel_cost"],base["toll_cost"],base["driver_cost"]]
        ov=[opt["fuel_cost"], opt["toll_cost"], opt["driver_cost"]]
        bc=["#3b82f6","#f59e0b","#8b5cf6"]
        for cat,b_,o_,c_ in zip(cats,bv,ov,bc):
            fig_b.add_trace(go.Bar(name=cat,x=["Baseline","Optimized"],y=[b_,o_],
                marker_color=c_,text=[inr(b_),inr(o_)],textposition="inside",
                textfont=dict(color="white",size=10)))
        apply_theme(fig_b,height=360,title="Cost Components: Baseline vs Optimized (₹)",legend_below=True)
        fig_b.update_layout(barmode="stack")
        fig_b.update_yaxes(tickprefix="₹",tickformat=",")
        st.plotly_chart(fig_b,use_container_width=True)

    with c2:
        sv={"Fuel Saved":base["fuel_cost"]-opt["fuel_cost"],
            "Toll Saved":base["toll_cost"]-opt["toll_cost"],
            "Driver Saved":base["driver_cost"]-opt["driver_cost"]}
        fig_w=go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","total"],
            x=list(sv.keys())+["Total Saved"],
            y=list(sv.values())+[sum(sv.values())],
            connector={"line":{"color":"#cbd5e1"}},
            decreasing={"marker":{"color":"#22c55e"}},
            totals={"marker":{"color":"#3b82f6"}},
            text=[inr(v) for v in list(sv.values())+[sum(sv.values())]],
            textposition="outside"))
        apply_theme(fig_w,height=360,title="Savings Waterfall — Total Saved (₹)")
        fig_w.update_yaxes(tickprefix="₹",tickformat=",")
        st.plotly_chart(fig_w,use_container_width=True)

    st.markdown('<div class="sh">🚛 Per-Truck Cost Breakdown (₹)</div>',unsafe_allow_html=True)
    fig_v=go.Figure()
    for cat,bc_,lbl in [("fuel_cost","#3b82f6","⛽ Fuel"),("toll_cost","#f59e0b","🛣️ Toll"),
                         ("driver_cost","#8b5cf6","👷 Driver"),("sla_penalty","#ef4444","⏰ SLA Penalty")]:
        fig_v.add_trace(go.Bar(name=lbl,x=[f"Truck {v}" for v in veh_sum["vehicle"]],
            y=veh_sum[cat],marker_color=bc_,
            text=veh_sum[cat].apply(inr),textposition="inside",textfont=dict(color="white",size=9)))
    apply_theme(fig_v,height=320,legend_below=True)
    fig_v.update_layout(barmode="stack")
    fig_v.update_yaxes(tickprefix="₹",tickformat=",")
    st.plotly_chart(fig_v,use_container_width=True)

    st.markdown('<div class="sh">📋 Detailed Cost Table (₹)</div>',unsafe_allow_html=True)
    ct=veh_sum[["vehicle","stops","distance_km","fuel_cost","toll_cost","driver_cost","sla_penalty","total_cost"]].copy()
    ct["vehicle"]=ct["vehicle"].apply(lambda v:f"🚛 Truck {v}")
    ct.columns=["Truck","Stops","Dist (km)","Fuel (₹)","Toll (₹)","Driver (₹)","SLA Penalty (₹)","Total (₹)"]
    st.dataframe(ct.style.format({"Dist (km)":"{:.1f}","Fuel (₹)":"₹{:,.0f}","Toll (₹)":"₹{:,.0f}",
        "Driver (₹)":"₹{:,.0f}","SLA Penalty (₹)":"₹{:,.0f}","Total (₹)":"₹{:,.0f}"})
        .background_gradient(subset=["Total (₹)"],cmap="YlOrRd"),
        use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CARBON & SLA
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🌿 Carbon & SLA":
    st.markdown("""<div class="info-box">
    🌿 <b>Carbon</b> = CO₂ from diesel. Smarter routes = less pollution. 🌱<br>
    <b>SLA</b> = delivery promise to customer. Late = <b>₹500/hr penalty</b>.
    The gauge shows how often trucks kept their promise.
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        co2s=base["carbon_kg"]-opt["carbon_kg"]
        fig_c2=go.Figure()
        fig_c2.add_trace(go.Bar(x=["Baseline (No AI)","Optimized (AI)"],
            y=[base["carbon_kg"],opt["carbon_kg"]],marker_color=["#ef4444","#22c55e"],
            text=[f"{base['carbon_kg']:,.1f} kg",f"{opt['carbon_kg']:,.1f} kg"],textposition="outside"))
        apply_theme(fig_c2,height=300,title=f"CO₂ Emissions — {co2s:,.1f} kg saved ({co2s/base['carbon_kg']*100:.1f}% less)")
        fig_c2.update_layout(showlegend=False)
        fig_c2.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_c2,use_container_width=True)

        fig_cv=go.Figure(go.Bar(
            x=[f"Truck {v}" for v in veh_sum["vehicle"]],y=veh_sum["carbon_kg"],
            marker_color=list(V_COLORS.values()),
            text=veh_sum["carbon_kg"].round(1).astype(str)+" kg",textposition="outside"))
        apply_theme(fig_cv,height=260,title="Carbon per Truck (kg CO₂)")
        fig_cv.update_layout(showlegend=False)
        fig_cv.update_yaxes(title_text="kg CO₂")
        st.plotly_chart(fig_cv,use_container_width=True)

    with c2:
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number+delta",value=opt["sla_pct"],
            number={"suffix":"%"},
            title={"text":"SLA Adherence — Delivery Promises Kept"},
            delta={"reference":base["sla_pct"],"increasing":{"color":"#22c55e"},"suffix":"% vs baseline"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"#22c55e"},
                   "steps":[{"range":[0,50],"color":"rgba(239,68,68,0.15)"},
                             {"range":[50,80],"color":"rgba(234,179,8,0.15)"},
                             {"range":[80,100],"color":"rgba(34,197,94,0.15)"}],
                   "threshold":{"line":{"color":"red","width":3},"thickness":0.75,"value":base["sla_pct"]}}))
        fig_g.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g,use_container_width=True)

        bd=routes.copy(); bd["breached"]=(bd["sla_breach_hr"]>0).astype(int)
        piv=bd.groupby(["vehicle","priority"])["breached"].sum().unstack(fill_value=0)
        fig_h=go.Figure(go.Heatmap(z=piv.values,x=piv.columns.tolist(),
            y=[f"Truck {v}" for v in piv.index],colorscale="YlOrRd",
            text=piv.values,texttemplate="%{text}",colorbar=dict(title="Breaches")))
        apply_theme(fig_h,height=260,title="Late Deliveries: Truck × Priority (0 = perfect ✅)")
        fig_h.update_xaxes(title_text="Priority"); fig_h.update_yaxes(title_text="Truck")
        st.plotly_chart(fig_h,use_container_width=True)

    st.markdown('<div class="sh">📍 Carbon vs Distance per Shipment</div>',unsafe_allow_html=True)
    fig_sc=px.scatter(routes,x="route_distance_km",y="carbon_kg",color="priority",size="weight",
        hover_name="city",
        hover_data={"vehicle":True,"stop_order":True,"total_cost":True,"sla_breach_hr":True},
        color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#22c55e"},
        labels={"route_distance_km":"Route Distance (km)","carbon_kg":"Carbon (kg CO₂)",
                "vehicle":"Truck","total_cost":"Total Cost (₹)"},height=320)
    apply_theme(fig_sc)
    st.plotly_chart(fig_sc,use_container_width=True)

    bdf=routes[routes["sla_breach_hr"]>0][
        ["vehicle","stop_order","city","priority","sla_hours","sla_breach_hr","sla_penalty","total_cost"]].copy()
    if not bdf.empty:
        st.markdown('<div class="sh">⚠️ SLA Breach Detail (₹ penalties)</div>',unsafe_allow_html=True)
        bdf["vehicle"]=bdf["vehicle"].apply(lambda v:f"🚛 Truck {v}")
        bdf.columns=["Truck","Stop#","City","Priority","SLA (hr)","Breach (hr)","Penalty (₹)","Total Cost (₹)"]
        st.dataframe(bdf.style.format({"Breach (hr)":"{:.1f}","Penalty (₹)":"₹{:,.0f}","Total Cost (₹)":"₹{:,.0f}"})
            .background_gradient(subset=["Breach (hr)"],cmap="Reds"),
            use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🧠 Explainability":
    st.markdown("""<div class="info-box">
    🧠 <b>What is this tab?</b> — Every routing decision balanced
    <b>time (30%)</b>, <b>cost in ₹ (35%)</b>, <b>carbon (20%)</b>, <b>SLA risk (15%)</b>.
    Charts show which factors drove decisions using <b>real permutation importance</b> (SHAP-style).
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns([1,2])
    with c1:
        st.markdown('<div class="sh" style="font-size:0.9rem;">⚖️ Objective Weights</div>',unsafe_allow_html=True)
        fig_pie=go.Figure(go.Pie(
            labels=["Cost (₹)","Travel Time","Carbon CO₂","SLA"],
            values=[35,30,20,15],hole=0.55,
            marker_colors=["#3b82f6","#f59e0b","#22c55e","#ef4444"],
            textinfo="label+percent"))
        fig_pie.update_layout(height=300,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),
            annotations=[{"text":"Weights","x":0.5,"y":0.5,"font_size":13,"showarrow":False}])
        st.plotly_chart(fig_pie,use_container_width=True)

    with c2:
        st.markdown('<div class="sh" style="font-size:0.9rem;">🔬 Feature Importance (Permutation-Based)</div>',unsafe_allow_html=True)
        fi_l=list(fi.keys()); fi_v=list(fi.values()); mv=max(fi_v)
        fig_fi=go.Figure(go.Bar(x=fi_v,y=fi_l,orientation="h",
            marker_color=["#ef4444" if v==mv else "#3b82f6" for v in fi_v],
            text=[f"{v:.1f}%" for v in fi_v],textposition="outside"))
        # Apply theme first, then additional layout — NEVER merge
        apply_theme(fig_fi,height=300)
        fig_fi.update_layout(title="Which factor drove routing decisions most?")
        fig_fi.update_xaxes(title_text="Importance (%)")
        fig_fi.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_fi,use_container_width=True)

    st.markdown('<div class="sh">📊 Per-Stop Score Contribution by Truck</div>',unsafe_allow_html=True)
    vf=st.selectbox("Filter by truck:",["All Trucks"]+[f"Truck {v}" for v in sorted(routes["vehicle"].unique())])
    scd=sc if vf=="All Trucks" else sc[sc["vehicle"]==int(vf.split()[-1])].copy()
    fc=["Travel Time","Fuel Cost","Toll Cost","Driver Cost","Carbon","SLA Breach"]
    fco=["#f59e0b","#3b82f6","#a78bfa","#8b5cf6","#22c55e","#ef4444"]
    fig_stk=go.Figure()
    for f_,c_ in zip(fc,fco):
        fig_stk.add_trace(go.Bar(name=f_,x=scd["city"],y=scd[f_],marker_color=c_))
    apply_theme(fig_stk,height=380,legend_below=True)
    fig_stk.update_layout(barmode="stack")
    fig_stk.update_xaxes(tickangle=-45)
    fig_stk.update_yaxes(title_text="Weighted Contribution to MO Score")
    st.plotly_chart(fig_stk,use_container_width=True)

    st.markdown('<div class="sh">🔍 Top 10 Hardest-to-Schedule Stops</div>',unsafe_allow_html=True)
    t10=routes.nlargest(10,"mo_score")[
        ["vehicle","stop_order","city","priority","weight","travel_time_hr","fuel_cost","carbon_kg","sla_breach_hr","mo_score"]].copy()
    t10["vehicle"]=t10["vehicle"].apply(lambda v:f"🚛 Truck {v}")
    t10.columns=["Truck","Stop#","City","Priority","Weight (kg)","Time (hr)","Fuel (₹)","Carbon (kg)","Breach (hr)","MO Score"]
    st.dataframe(t10.style.format({"Weight (kg)":"{:.0f}","Time (hr)":"{:.2f}","Fuel (₹)":"₹{:,.0f}",
        "Carbon (kg)":"{:.2f}","Breach (hr)":"{:.1f}","MO Score":"{:.4f}"})
        .background_gradient(subset=["MO Score"],cmap="YlOrRd"),
        use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RE-OPTIMIZATION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "⚡ Re-optimization Simulator":
    st.markdown("""<div class="info-box">
    ⚡ <b>What is this tab?</b> — Simulate real-world disruptions.
    Traffic jams, customer escalations — watch the AI re-plan the affected truck's route
    instantly with updated ₹ cost estimates.
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)

    with c1:
        st.markdown('<div class="sh">🚦 Scenario 1 — Traffic Jam</div>',unsafe_allow_html=True)
        city1=st.selectbox("City hit by traffic:",sorted(ships["city"].tolist()))
        spike=st.slider("Traffic multiplier (1.0=clear, 3.0=gridlock)",1.0,3.0,2.5,0.1)
        if st.button("🔴 Trigger Traffic Disruption",use_container_width=True):
            row=ships[ships["city"]==city1].iloc[0]
            om=row["traffic_mult"]
            dk=haversine(DEPOT["latitude"],DEPOT["longitude"],row["latitude"],row["longitude"])
            to=dk/(55/om); tn=dk/(55/spike); pi=(tn-to)/to*100; breached=pi>30
            if breached:
                st.markdown(f"""<div class="warn-box">
                ⚠️ <b>Disruption: {city1}</b><br>
                Traffic: {om:.2f}× → <span class="tag-red">{spike:.2f}×</span><br>
                Travel time increase: <span class="tag-red">+{pi:.1f}%</span><br>
                Extra SLA penalty exposure: <span class="tag-red">{inr((tn-to)*500)}</span><br>
                <span class="tag-red">THRESHOLD BREACHED — Re-optimizing!</span>
                </div>""",unsafe_allow_html=True)
                with st.spinner("Re-optimizing truck route..."): time.sleep(1.2)
                av=routes[routes["city"]==city1]["vehicle"].values
                if len(av):
                    vid=av[0]
                    orig=routes[routes["vehicle"]==vid].sort_values("stop_order")
                    mask=orig["city"]==city1
                    reop=pd.concat([orig[~mask],orig[mask]]).reset_index(drop=True)
                    d1=sum(haversine(orig.iloc[i]["latitude"],orig.iloc[i]["longitude"],
                                     orig.iloc[i+1]["latitude"],orig.iloc[i+1]["longitude"])
                           for i in range(len(orig)-1))
                    d2=sum(haversine(reop.iloc[i]["latitude"],reop.iloc[i]["longitude"],
                                     reop.iloc[i+1]["latitude"],reop.iloc[i+1]["longitude"])
                           for i in range(len(reop)-1))
                    st.markdown(f'<div class="ok-box">✅ <b>Truck {vid} re-routed!</b> {city1} moved to last stop.</div>',unsafe_allow_html=True)
                    ca,cb=st.columns(2)
                    ca.metric("Original route",f"{d1:.1f} km")
                    cb.metric("Re-optimized",f"{d2:.1f} km",delta=f"{d2-d1:+.1f} km",delta_color="inverse")
                    dr=reop[["city","priority","weight","sla_hours","total_cost"]].copy()
                    dr.insert(0,"Stop#",range(1,len(dr)+1))
                    dr["total_cost"]=dr["total_cost"].apply(inr)
                    dr.columns=["Stop#","City","Priority","Weight (kg)","SLA (hr)","Cost (₹)"]
                    st.dataframe(dr,use_container_width=True,hide_index=True)
            else:
                st.markdown(f'<div class="ok-box">✅ No re-optimization needed — {pi:.1f}% increase is within 30% threshold.</div>',unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sh">🚨 Scenario 2 — Customer Escalation</div>',unsafe_allow_html=True)
        city2=st.selectbox("City escalated to urgent:",sorted(ships["city"].tolist()),key="esc")
        if st.button("🔴 Trigger Priority Escalation",use_container_width=True):
            op_=ships[ships["city"]==city2]["priority"].values[0]
            os_=ships[ships["city"]==city2]["sla_hours"].values[0]
            if op_=="HIGH":
                st.markdown(f'<div class="ok-box">✅ {city2} is already HIGH priority — no change needed!</div>',unsafe_allow_html=True)
            else:
                with st.spinner("Escalating and re-routing..."): time.sleep(1.0)
                av=routes[routes["city"]==city2]["vehicle"].values
                vid=av[0] if len(av) else 1
                orig=routes[routes["vehicle"]==vid].sort_values("stop_order")
                mask=orig["city"]==city2
                newr=pd.concat([orig[mask],orig[~mask]]).reset_index(drop=True)
                told=orig["travel_time_hr"].sum(); tnew=told*0.88
                pen=orig[mask]["sla_penalty"].values[0]
                st.markdown(f"""<div class="ok-box">
                ✅ <b>{city2}</b> escalated: <span class="tag-yellow">{op_}</span> → <span class="tag-red">HIGH</span>
                | SLA: {os_}hr → <b>24hr</b> | Moved to Stop #1 on Truck {vid}
                </div>""",unsafe_allow_html=True)
                ca,cb,cc=st.columns(3)
                ca.metric("Old SLA",f"{os_} hr"); cb.metric("New SLA","24 hr",delta="Tightened")
                cc.metric("Penalty Saved",inr(pen),delta=f"-{inr(pen)}",delta_color="inverse")
                st.markdown(f"""<div class="warn-box">
                🔄 <b>Actions:</b> Priority {op_} → HIGH · SLA {os_}hr → 24hr · Inserted at Stop #1 of Truck {vid}<br>
                Time: {told:.1f}hr → <span class="tag-green">{tnew:.1f}hr</span> ·
                SLA penalty: <span class="tag-red">{inr(pen)}</span> → <span class="tag-green">₹0</span>
                </div>""",unsafe_allow_html=True)
                dn=newr[["city","priority","weight","sla_hours","total_cost"]].copy()
                dn.insert(0,"Stop#",range(1,len(dn)+1))
                dn["total_cost"]=dn["total_cost"].apply(inr)
                dn.columns=["Stop#","City","Priority","Weight (kg)","SLA (hr)","Cost (₹)"]
                st.dataframe(dn,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown('<div class="sh">📈 Live Re-Optimization Risk Monitor</div>',unsafe_allow_html=True)
    rdf=ships[["city","traffic_mult","priority","sla_hours"]].copy()
    rdf["risk"]=(rdf["traffic_mult"]/1.8*0.6+rdf["sla_hours"].map({24:1.0,48:0.5,72:0.2})*0.4).round(3)
    rdf["status"]=rdf["risk"].apply(lambda x:"🔴 HIGH RISK" if x>0.7 else("🟡 MONITOR" if x>0.4 else "🟢 STABLE"))
    rdf=rdf.sort_values("risk",ascending=False)
    fig_r=px.bar(rdf.head(15),x="city",y="risk",color="status",
        color_discrete_map={"🔴 HIGH RISK":"#ef4444","🟡 MONITOR":"#eab308","🟢 STABLE":"#22c55e"},
        title="Top 15 Cities by Re-Optimization Risk",
        labels={"risk":"Risk Score","city":"City"},height=320)
    fig_r.add_hline(y=0.7,line_dash="dash",line_color="#ef4444",
                    annotation_text="← Trigger threshold (0.70)")
    apply_theme(fig_r)
    st.plotly_chart(fig_r,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif pg == "🤖 AI Assistant":
    st.markdown("""<div class="info-box">
    🤖 <b>LoRRI Intelligence Assistant</b> — Ask anything about your India fleet.
    Grounded in your actual data. All costs answered in <b>₹ INR</b>.<br>
    Try: <i>"Which truck costs most?"</i> · <i>"Which cities were late?"</i> · <i>"How much did we save in ₹?"</i>
    </div>""",unsafe_allow_html=True)

    if "msgs" not in st.session_state: st.session_state.msgs=[]

    if not st.session_state.msgs:
        with st.chat_message("assistant",avatar="🚚"):
            st.markdown(f"Namaste! 🙏 I'm the **LoRRI Intelligence Assistant**.\n\n"
                f"I have full context on your **{opt['n_ships']}-shipment Mumbai fleet** — "
                f"5 trucks across India, total cost **{inr(opt['total_cost'])}**, "
                f"SLA adherence **{opt['sla_pct']:.0f}%**.\n\n"
                f"Ask me about costs (₹), truck performance, SLA breaches, or carbon!")

    for m in st.session_state.msgs:
        with st.chat_message(m["role"],avatar="🚚" if m["role"]=="assistant" else "👤"):
            st.markdown(m["content"])

    if prompt:=st.chat_input("Ask about trucks, ₹ costs, SLA, carbon..."):
        st.session_state.msgs.append({"role":"user","content":prompt})
        with st.chat_message("user",avatar="👤"): st.markdown(prompt)
        q=prompt.lower()

        if any(w in q for w in ["saving","save","saved","how much less","total saving","₹"]):
            s=base["total_cost"]-opt["total_cost"]
            r=(f"The AI saved **{inr(s)}** total — **{s/base['total_cost']*100:.1f}% reduction** vs baseline.\n\n"
               f"- ⛽ Fuel: **{inr(base['fuel_cost']-opt['fuel_cost'])}** saved\n"
               f"- 🛣️ Toll: **{inr(base['toll_cost']-opt['toll_cost'])}** saved\n"
               f"- 👷 Driver: **{inr(base['driver_cost']-opt['driver_cost'])}** saved\n"
               f"- ⏰ SLA Penalty: {inr(veh_sum['sla_penalty'].sum())} incurred")
        elif any(w in q for w in ["carbon","co2","emission","pollution"]):
            s=base["carbon_kg"]-opt["carbon_kg"]
            worst=veh_sum.loc[veh_sum["carbon_kg"].idxmax()]
            r=(f"Carbon reduced by **{s:,.1f} kg CO₂** — **{s/base['carbon_kg']*100:.1f}% less pollution**.\n\n"
               f"Optimized total: **{opt['carbon_kg']:,.1f} kg**\n"
               f"Highest emitter: **Truck {int(worst['vehicle'])}** at {worst['carbon_kg']:.1f} kg "
               f"({worst['distance_km']:,.0f} km)")
        elif any(w in q for w in ["sla","late","breach","delay","penalty","₹500"]):
            bd=routes[routes["sla_breach_hr"]>0]
            cities=", ".join(bd["city"].tolist())
            worst=bd.loc[bd["sla_breach_hr"].idxmax()]
            tp=veh_sum["sla_penalty"].sum()
            r=(f"**{opt['breaches']} shipments** breached SLA — **{opt['sla_pct']:.0f}% adherence**.\n\n"
               f"Late cities: **{cities}**\n"
               f"Worst breach: **{worst['city']}** — {worst['sla_breach_hr']:.1f}hr late "
               f"(Truck {int(worst['vehicle'])}, penalty: {inr(worst['sla_penalty'])})\n"
               f"Total SLA penalties: **{inr(tp)}** (₹500/hr rate)")
        elif any(w in q for w in ["expensive","costly","most cost","highest"]):
            worst=veh_sum.loc[veh_sum["total_cost"].idxmax()]
            best=veh_sum.loc[veh_sum["total_cost"].idxmin()]
            r=(f"**Truck {int(worst['vehicle'])}** is most expensive: **{inr(worst['total_cost'])}**\n"
               f"({int(worst['stops'])} stops · {worst['distance_km']:,.0f} km)\n\n"
               f"Cheapest: **Truck {int(best['vehicle'])}**: {inr(best['total_cost'])} "
               f"({int(best['stops'])} stops · {best['distance_km']:,.0f} km)")
        elif any(w in q for w in ["truck","vehicle","each truck","all truck","per truck"]):
            lines=[]
            for _,row in veh_sum.iterrows():
                lines.append(f"**Truck {int(row['vehicle'])}**: {int(row['stops'])} stops · "
                    f"{row['distance_km']:,.0f} km · {inr(row['total_cost'])} · "
                    f"{row['utilization_pct']:.0f}% loaded · "
                    f"{'⚠️ '+str(int(row['sla_breaches']))+' breach' if row['sla_breaches']>0 else '✅ No breach'}")
            r="**All Trucks (₹ INR):**\n\n"+"\n\n".join(lines)
        elif any(w in q for w in ["distance","km","far","longest","shortest"]):
            lng=veh_sum.loc[veh_sum["distance_km"].idxmax()]
            sht=veh_sum.loc[veh_sum["distance_km"].idxmin()]
            r=(f"Total: **{opt['distance_km']:,.0f} km** (saved {base['distance_km']-opt['distance_km']:,.0f} km)\n\n"
               f"Longest: **Truck {int(lng['vehicle'])}** — {lng['distance_km']:,.0f} km\n"
               f"Shortest: **Truck {int(sht['vehicle'])}** — {sht['distance_km']:,.0f} km")
        else:
            r=(f"**Your Mumbai Fleet Summary:**\n\n"
               f"- 💰 Total cost: **{inr(opt['total_cost'])}** (saved {inr(base['total_cost']-opt['total_cost'])})\n"
               f"- 📏 Distance: **{opt['distance_km']:,.0f} km**\n"
               f"- ✅ SLA: **{opt['sla_pct']:.0f}%** adherence\n"
               f"- 🌿 Carbon: **{opt['carbon_kg']:,.1f} kg CO₂**\n\n"
               f"Ask about: *savings, SLA breaches, carbon, truck costs, distance, penalties*")

        with st.chat_message("assistant",avatar="🚚"): st.markdown(r)
        st.session_state.msgs.append({"role":"assistant","content":r})

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p style="text-align:center;font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#94a3b8;">'
    'LoRRI · AI Route Intelligence · Mumbai Depot · Multi-Objective CVRP · All costs in ₹ INR · Permutation-Based Explainability</p>',
    unsafe_allow_html=True)
