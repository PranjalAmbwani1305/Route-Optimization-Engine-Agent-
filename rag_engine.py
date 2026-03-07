"""
rag_engine.py  —  LoRRI Vectorless RAG Engine
Fully offline. No external API. No embeddings. No vectors.
Pipeline: Rule Router → BM25-lite Retriever → Pandas Query → Local Synthesizer
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Load data (called once from dashboard.py via st.cache_data)
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    ships  = pd.read_csv("shipments.csv")
    routes = pd.read_csv("routes.csv")
    veh    = pd.read_csv("vehicle_summary.csv")
    return ships, routes, veh


def inr(val):
    return f"Rs.{val:,.0f}"


ROUTE_MAP = {
    1: "Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada",
    2: "Mumbai → Surat → Vadodara → Raipur",
    3: "Mumbai → Aurangabad → Solapur → Madurai → Jammu",
    4: "Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram",
    5: "Mumbai → Hubli → Mangalore → Bengaluru",
}

# ─────────────────────────────────────────────────────────────────────────────
# ① KNOWLEDGE CHUNKS  (static text, no vectors)
# ─────────────────────────────────────────────────────────────────────────────
def build_knowledge_chunks(opt, base, veh_sum):
    return [
        {
            "id": "company",
            "triggers": ["logisticsnow","company","about","who","contact",
                         "email","phone","website","lorri"],
            "text": (
                "COMPANY: LogisticsNow (logisticsnow.in)\n"
                "Email: connect@logisticsnow.in\n"
                "Phone: +91-9867773508 / +91-9653620207\n"
                "LoRRI = Logistics Rating & Intelligence\n"
                "For Shippers: carrier profiles, ratings, cost savings.\n"
                "For Carriers: discoverability, business inquiries, reputation."
            ),
        },
        {
            "id": "optimization",
            "triggers": ["cvrp","optimize","optimization","weighted","objective",
                         "score","solver","ortools","heuristic","algorithm"],
            "text": (
                "OPTIMIZATION ENGINE: CVRP framework.\n"
                "Objective: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%).\n"
                "Solver: OR-Tools nearest-neighbour + 2-opt local search.\n"
                "Re-optimization triggers: traffic delay >30% OR priority escalation.\n"
                "Explainability: permutation-based feature importance (SHAP-style)."
            ),
        },
        {
            "id": "pricing",
            "triggers": ["fuel","toll","driver","cost","price","inr",
                         "rupee","penalty","wage","rs"],
            "text": (
                "PRICING (Rs. INR):\n"
                "Fuel: Rs.12/km | Driver: Rs.180/hr | SLA breach: Rs.500/hr | Toll: variable"
            ),
        },
        {
            "id": "sla",
            "triggers": ["sla","late","breach","delay","on time","delivery",
                         "promise","window","adherence"],
            "text": (
                f"SLA PERFORMANCE:\n"
                f"Optimized: {opt['sla_pct']:.1f}% (baseline {base['sla_pct']:.0f}%)\n"
                f"Breaches: {opt['breaches']} cities | Penalty: Rs.500/hr\n"
                f"Total penalties: {inr(veh_sum['sla_penalty'].sum())}\n"
                f"Windows: HIGH=24hr, MEDIUM=48hr, LOW=72hr"
            ),
        },
        {
            "id": "carbon",
            "triggers": ["carbon","co2","emission","green","environment",
                         "sustainability","eco","tree","pollution"],
            "text": (
                f"CARBON:\n"
                f"Optimized: {opt['carbon_kg']:,.1f} kg | Baseline: {base['carbon_kg']:,.1f} kg\n"
                f"Saved: {base['carbon_kg']-opt['carbon_kg']:,.1f} kg "
                f"({(base['carbon_kg']-opt['carbon_kg'])/base['carbon_kg']*100:.1f}%)\n"
                f"Trees equivalent: {int((base['carbon_kg']-opt['carbon_kg'])/21):,} | "
                f"Cars off road: {int((base['carbon_kg']-opt['carbon_kg'])/2400)}"
            ),
        },
        {
            "id": "fleet_summary",
            "triggers": ["fleet","total","summary","overall","depot","mumbai",
                         "shipment","truck","vehicle","save","saving","saved",
                         "baseline","optimized"],
            "text": (
                f"FLEET SUMMARY (Mumbai Depot):\n"
                f"{opt['n_ships']} shipments | {opt['n_vehicles']} trucks\n"
                f"Optimized: {inr(opt['total_cost'])} | Baseline: {inr(base['total_cost'])}\n"
                f"Saved: {inr(base['total_cost']-opt['total_cost'])} "
                f"({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}%)\n"
                f"Distance: {opt['distance_km']:,.0f} km (was {base['distance_km']:,.0f} km)"
            ),
        },
        {
            "id": "traffic",
            "triggers": ["traffic","jam","congestion","disruption",
                         "reoptimize","threshold","delay","multiplier"],
            "text": (
                "TRAFFIC & RE-OPTIMIZATION:\n"
                "Triggers when delay >30% or on priority escalation.\n"
                "Recomputes in ~1-2s via OR-Tools local search.\n"
                "Risk levels: HIGH >0.7, MONITOR >0.4, STABLE <=0.4"
            ),
        },
        {
            "id": "cost_saving",
            "triggers": ["suggest","recommendation","improve","reduce cost",
                         "save more","tip","advice","better","optimize further"],
            "text": (
                "COST SAVING RECOMMENDATIONS:\n"
                "1. Consolidate Truck 2 & 5 — both under 70% utilization\n"
                "2. Avoid HIGH-traffic corridors during peak hours\n"
                "3. Upgrade LOW-priority to 72hr SLA window\n"
                "4. Use express highway only when savings exceed toll premium\n"
                "5. Cluster Madurai + Thiruvananthapuram on one southern truck\n"
                "6. Pre-position trucks at Pune, Ahmedabad hubs"
            ),
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ② BM25-LITE RETRIEVER  (keyword scoring, no embeddings)
# ─────────────────────────────────────────────────────────────────────────────
def rag_retrieve(query: str, chunks: list, top_k: int = 3) -> str:
    q = query.lower()
    scored = []
    for chunk in chunks:
        score = sum(
            (2 if len(t.split()) > 1 else 1)
            for t in chunk["triggers"] if t in q
        )
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    tops = [c["text"] for s, c in scored[:top_k] if s > 0]
    return "\n\n---\n\n".join(tops) if tops else ""


# ─────────────────────────────────────────────────────────────────────────────
# ③ PANDAS RETRIEVAL  (structured DataFrame queries)
# ─────────────────────────────────────────────────────────────────────────────
def pandas_retrieve(query: str, routes, veh_sum) -> str:
    q = query.lower()
    results = []

    # Per-truck detail
    for v in [1, 2, 3, 4, 5]:
        if f"truck {v}" in q or f"truck{v}" in q:
            row = veh_sum[veh_sum["vehicle"] == v]
            if not row.empty:
                r  = row.iloc[0]
                bc = routes[
                    (routes["vehicle"] == v) & (routes["sla_breach_hr"] > 0)
                ]["city"].tolist()
                stops = routes[routes["vehicle"] == v].sort_values("stop_order")[
                    ["stop_order","city","weight","priority",
                     "travel_time_hr","fuel_cost","carbon_kg","sla_breach_hr"]
                ]
                results.append(
                    f"TRUCK {v}:\n"
                    f"Route: {ROUTE_MAP.get(v,'?')}\n"
                    f"Stops:{int(r['stops'])} | Dist:{r['distance_km']:,.0f}km | "
                    f"Time:{r['time_hr']:.1f}hr | Load:{r['load_kg']:.0f}kg ({r['utilization_pct']:.0f}%)\n"
                    f"Fuel:{inr(r['fuel_cost'])} | Toll:{inr(r['toll_cost'])} | "
                    f"Driver:{inr(r['driver_cost'])} | Penalty:{inr(r['sla_penalty'])}\n"
                    f"Total:{inr(r['total_cost'])} | CO2:{r['carbon_kg']:.1f}kg\n"
                    f"Breaches:{int(r['sla_breaches'])} "
                    f"({'cities: '+', '.join(bc) if bc else 'none — OK'})\n"
                    f"\nSTOPS:\n{stops.to_string(index=False)}"
                )

    # SLA breach cities
    if any(k in q for k in ["late","breach","which cities","missed sla","cities late"]):
        bd = routes[routes["sla_breach_hr"] > 0][
            ["vehicle","city","priority","sla_breach_hr","sla_penalty"]
        ].copy()
        if not bd.empty:
            bd["vehicle"] = bd["vehicle"].apply(lambda v: f"Truck {v}")
            results.append("SLA BREACHES:\n" + bd.to_string(index=False))
        else:
            results.append("SLA BREACHES: None — all on time!")

    # Most expensive truck
    if any(k in q for k in ["most expensive","highest cost","costs most","expensive truck"]):
        t = veh_sum.loc[veh_sum["total_cost"].idxmax()]
        results.append(
            f"MOST EXPENSIVE: Truck {int(t['vehicle'])} — {inr(t['total_cost'])}\n"
            f"Route: {ROUTE_MAP.get(int(t['vehicle']),'?')}\n"
            f"Fuel:{inr(t['fuel_cost'])} | Toll:{inr(t['toll_cost'])} | "
            f"Driver:{inr(t['driver_cost'])} | Penalty:{inr(t['sla_penalty'])}"
        )

    # Fleet utilization
    if any(k in q for k in ["utilization","capacity","load","util"]):
        u = veh_sum[["vehicle","load_kg","utilization_pct"]].copy()
        u["vehicle"] = u["vehicle"].apply(lambda v: f"Truck {v}")
        results.append(
            f"UTILIZATION:\n{u.to_string(index=False)}\n"
            f"Average: {veh_sum['utilization_pct'].mean():.1f}%"
        )

    # All trucks comparison
    if any(k in q for k in ["compare truck","all truck","each truck","per truck","breakdown"]):
        c = veh_sum[["vehicle","distance_km","total_cost","carbon_kg",
                     "sla_breaches","utilization_pct"]].copy()
        c["vehicle"]    = c["vehicle"].apply(lambda v: f"Truck {v}")
        c["total_cost"] = c["total_cost"].apply(lambda x: f"Rs.{x:,.0f}")
        results.append("ALL TRUCKS:\n" + c.to_string(index=False))

    return "\n\n".join(results) if results else ""


# ─────────────────────────────────────────────────────────────────────────────
# ④ RULE-BASED ROUTER  (instant answers, no retrieval needed)
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_answer(query: str, opt, base, veh_sum, routes, ships):
    q = query.lower().strip()

    if q in {"hi","hello","hey","namaste","hii","hlo"}:
        return (
            f"**Namaste!** I'm the LoRRI AI Assistant.\n\n"
            f"Fleet: **{opt['n_ships']} shipments**, **{opt['n_vehicles']} trucks**, "
            f"cost **{inr(opt['total_cost'])}**. Ask me anything!", 99
        )

    if any(k in q for k in ["contact","email","phone","reach","logisticsnow"]):
        return (
            "**LogisticsNow Contact:**\n"
            "- 🌐 logisticsnow.in\n"
            "- 📧 connect@logisticsnow.in\n"
            "- 📞 +91-9867773508 / +91-9653620207", 99
        )

    if any(k in q for k in ["how much did we save","total saving","how much saved","save in"]):
        saved = base["total_cost"] - opt["total_cost"]
        return (
            f"**Total Savings: {inr(saved)} ({saved/base['total_cost']*100:.1f}% reduction)**\n\n"
            f"| Category | Baseline | Optimized | Saved |\n|---|---|---|---|\n"
            f"| Fuel    | {inr(base['fuel_cost'])}   | {inr(opt['fuel_cost'])}   | **{inr(base['fuel_cost']-opt['fuel_cost'])}** |\n"
            f"| Toll    | {inr(base['toll_cost'])}   | {inr(opt['toll_cost'])}   | **{inr(base['toll_cost']-opt['toll_cost'])}** |\n"
            f"| Driver  | {inr(base['driver_cost'])} | {inr(opt['driver_cost'])} | **{inr(base['driver_cost']-opt['driver_cost'])}** |\n"
            f"| **Total** | {inr(base['total_cost'])} | {inr(opt['total_cost'])} | **{inr(saved)}** |", 98
        )

    if ("carbon" in q or "co2" in q) and any(k in q for k in ["saving","save","reduc","how much"]):
        s = base["carbon_kg"] - opt["carbon_kg"]
        return (
            f"**CO2 Saved: {s:,.1f} kg ({s/base['carbon_kg']*100:.1f}% reduction)**\n\n"
            f"- Baseline: {base['carbon_kg']:,.1f} kg\n"
            f"- Optimized: {opt['carbon_kg']:,.1f} kg\n"
            f"- 🌳 {int(s/21):,} trees equivalent per year\n"
            f"- 🚗 {int(s/2400)} cars removed from road", 98
        )

    for v in [1, 2, 3, 4, 5]:
        if f"truck {v}" in q and "route" in q:
            r = veh_sum[veh_sum["vehicle"] == v].iloc[0]
            return (
                f"**Truck {v} Route:**\n🚛 {ROUTE_MAP[v]}\n\n"
                f"{int(r['stops'])} stops · {r['distance_km']:,.0f} km · {inr(r['total_cost'])}", 97
            )

    if "truck 3" in q and any(k in q for k in ["change","why","explain","reason","chose"]):
        t3 = veh_sum[veh_sum["vehicle"] == 3].iloc[0]
        bc = routes[(routes["vehicle"]==3)&(routes["sla_breach_hr"]>0)]["city"].tolist()
        return (
            f"**Why Truck 3's Route:**\n{ROUTE_MAP[3]}\n\n"
            f"- Aurangabad & Solapur cluster on the Pune–Hyderabad axis\n"
            f"- Madurai anchors the deep south leg (no other truck covers it)\n"
            f"- Jammu: high-priority northern extension\n"
            f"- CVRP scored: Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%)\n\n"
            f"**Stats:** {t3['distance_km']:,.0f} km · {inr(t3['total_cost'])} · "
            f"{t3['carbon_kg']:.0f} kg CO2 · {int(t3['sla_breaches'])} breach\n"
            + (f"Breach cities: {', '.join(bc)}" if bc else "All SLA met ✅"), 96
        )

    if any(k in q for k in ["tomorrow","traffic risk","forecast","predict traffic","next day"]):
        hr = ships[ships["traffic_mult"] > 2.0]["city"].tolist()
        mr = ships[(ships["traffic_mult"]>1.4)&(ships["traffic_mult"]<=2.0)]["city"].tolist()
        return (
            f"**Traffic Risk Forecast:**\n\n"
            f"🔴 HIGH (>2.0x): {', '.join(hr) if hr else 'None'}\n"
            f"🟡 MEDIUM (1.4–2.0x): {', '.join(mr[:5]) if mr else 'None'}\n"
            f"🟢 STABLE: All other corridors\n\n"
            f"Auto re-optimize triggers at >30% delay threshold.", 90
        )

    if any(k in q for k in ["which cities","cities late","missed sla","where late","who was late"]):
        bd = routes[routes["sla_breach_hr"] > 0][
            ["city","vehicle","sla_breach_hr","sla_penalty"]]
        if bd.empty:
            return "**No SLA breaches** — all deliveries on time! ✅", 99
        lines = [f"**{len(bd)} cities were late:**\n"]
        for _, r in bd.iterrows():
            lines.append(
                f"- **{r['city']}** (Truck {int(r['vehicle'])}) — "
                f"{r['sla_breach_hr']:.1f}hr late · penalty {inr(r['sla_penalty'])}"
            )
        return "\n".join(lines), 97

    if any(k in q for k in ["suggest","recommendation","tip","advice","reduce cost","save more"]):
        return (
            "**Cost Saving Recommendations:**\n\n"
            "1. Consolidate Truck 2 & 5 — both under 70% utilization\n"
            "2. Avoid HIGH-traffic corridors during peak hours\n"
            "3. Upgrade LOW-priority shipments to 72hr SLA window\n"
            "4. Use express highway only when savings exceed toll premium\n"
            "5. Cluster Madurai + Thiruvananthapuram on one southern truck\n"
            "6. Pre-position trucks at Pune & Ahmedabad hubs to cut backtracking", 94
        )

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ LOCAL SYNTHESIZER  (no API call — assembles retrieved context directly)
# ─────────────────────────────────────────────────────────────────────────────
def synthesize_answer(query: str, rag_ctx: str, pandas_ctx: str, opt, base) -> str:
    parts = []

    if pandas_ctx:
        parts.append(pandas_ctx)
    if rag_ctx:
        parts.append(rag_ctx)

    if parts:
        return "\n\n".join(parts)

    # Generic fallback
    return (
        f"**Fleet Summary (Mumbai Depot):**\n\n"
        f"- **{opt['n_ships']} shipments** | **{opt['n_vehicles']} trucks**\n"
        f"- Optimized cost: **{inr(opt['total_cost'])}** "
        f"(saved {inr(base['total_cost']-opt['total_cost'])} vs baseline)\n"
        f"- SLA adherence: **{opt['sla_pct']:.0f}%**\n"
        f"- Carbon: **{opt['carbon_kg']:,.1f} kg CO2**\n\n"
        f"Try asking about a specific truck, SLA breaches, cost savings, "
        f"carbon, routes, or optimization methodology."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ MAIN PIPELINE  — no external API, fully local
# ─────────────────────────────────────────────────────────────────────────────
def call_rag_pipeline(query: str, ships, routes, veh_sum, opt, base):
    """
    4-step fully offline RAG pipeline.
    Returns: (reply, confidence_int, source_label, chunks_info)
    """
    chunks = build_knowledge_chunks(opt, base, veh_sum)

    # Step 1: Rule-based router (instant)
    rule_reply, rule_conf = rule_based_answer(query, opt, base, veh_sum, routes, ships)
    if rule_reply:
        return rule_reply, rule_conf, "⚡ Rule router", "Instant match"

    # Step 2: BM25-lite keyword retrieval
    rag_ctx = rag_retrieve(query, chunks, top_k=3)

    # Step 3: Pandas structured retrieval
    pandas_ctx = pandas_retrieve(query, routes, veh_sum)

    # Source label
    sources = []
    if rag_ctx:    sources.append("📚 RAG chunks")
    if pandas_ctx: sources.append("🐼 Pandas")
    src_label   = " + ".join(sources) if sources else "🤖 Synthesized"
    n_chunks    = sum(1 for c in chunks if c["text"] in rag_ctx) if rag_ctx else 0
    chunks_info = f"{n_chunks} chunk(s) + {'DataFrame rows' if pandas_ctx else 'no structured data'}"

    # Step 4: Local synthesis (NO API CALL)
    reply = synthesize_answer(query, rag_ctx, pandas_ctx, opt, base)
    conf  = 92 if (rag_ctx and pandas_ctx) else (88 if (rag_ctx or pandas_ctx) else 72)

    return reply, conf, src_label, chunks_info
