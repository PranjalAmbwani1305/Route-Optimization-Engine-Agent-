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
#    20 chunks covering: company, project definitions, algorithms, data,
#    metrics, fleet, SLA, carbon, costs, hackathon context
# ─────────────────────────────────────────────────────────────────────────────
def build_knowledge_chunks(opt, base, veh_sum):
    return [

        # ── COMPANY & PLATFORM ───────────────────────────────────────────────
        {
            "id": "company",
            "triggers": ["logisticsnow","company","about","who","contact",
                         "email","phone","website","platform","startup"],
            "text": (
                "COMPANY: LogisticsNow (logisticsnow.in)\n"
                "Email: connect@logisticsnow.in\n"
                "Phone: +91-9867773508 / +91-9653620207\n"
                "LoRRI = Logistics Rating & Intelligence — India's premier logistics AI platform.\n"
                "For Shippers: carrier profiles, ratings, cost savings, procurement insights.\n"
                "For Carriers: discoverability, business inquiries, reputation building.\n"
                "Hackathon: Synapflow — Problem Statement 4."
            ),
        },

        # ── WHAT IS LORRI ────────────────────────────────────────────────────
        {
            "id": "lorri_definition",
            "triggers": ["what is lorri","lorri mean","lorri stand","define lorri",
                         "lorri definition","what does lorri","lorri full form",
                         "what is this project","explain lorri","lorri intro",
                         "what lorri","lorri is"],
            "text": (
                "WHAT IS LoRRI:\n"
                "LoRRI = Logistics Rating & Intelligence.\n"
                "An AI-powered Route Optimization Engine built by LogisticsNow.\n"
                "Purpose: Solve the Capacitated Vehicle Routing Problem (CVRP) for India's "
                "logistics networks — minimizing total delivery cost, time, carbon emissions, "
                "and SLA violations simultaneously.\n"
                "Built for: Synapflow Hackathon, Problem Statement 4.\n"
                "Stack: Python, Streamlit, OR-Tools, Plotly, Pandas.\n"
                "Depot: Mumbai (19.0760°N, 72.8777°E).\n"
                "Fleet: 5 trucks, capacity 800 kg each, delivering across India."
            ),
        },

        # ── PROJECT OVERVIEW ─────────────────────────────────────────────────
        {
            "id": "project_overview",
            "triggers": ["project","overview","problem","hackathon","synapflow",
                         "problem statement","ps4","build","built","tech stack",
                         "technology","streamlit","python","plotly"],
            "text": (
                "PROJECT OVERVIEW — LoRRI:\n"
                "Event: Synapflow Hackathon | Problem Statement 4\n"
                "Team: LogisticsNow\n"
                "Goal: AI-driven multi-objective route optimization for Indian logistics.\n\n"
                "TECH STACK:\n"
                "- Frontend/UI: Streamlit (Python)\n"
                "- Optimization: OR-Tools (Google) — CVRP solver\n"
                "- Visualization: Plotly (maps, charts, heatmaps)\n"
                "- Data: Pandas, NumPy\n"
                "- AI Assistant: Vectorless RAG — BM25-lite + Pandas retrieval (fully offline)\n"
                "- Maps: OpenStreetMap via Plotly Scattermap\n\n"
                "FILES: dashboard.py, rag_engine.py, generate_data.py, route_solver.py,\n"
                "       shipments.csv, routes.csv, vehicle_summary.csv, metrics.csv, logo.png\n\n"
                "GitHub: github.com/PranjalAmbwani1305/Route-Optimization-Engine-Agent-"
            ),
        },

        # ── CVRP DEFINITION ──────────────────────────────────────────────────
        {
            "id": "cvrp_definition",
            "triggers": ["cvrp","capacitated","vehicle routing problem","vrp",
                         "routing problem","what is cvrp","define cvrp",
                         "explain cvrp","cvrp mean"],
            "text": (
                "CVRP — CAPACITATED VEHICLE ROUTING PROBLEM:\n"
                "Definition: An NP-hard combinatorial optimization problem.\n"
                "Goal: Find the optimal set of routes for a fleet of vehicles "
                "to deliver goods to a set of customers, subject to vehicle capacity constraints.\n\n"
                "LoRRI's CVRP setup:\n"
                "- Depot: Mumbai\n"
                "- Customers: Cities across India (shipment locations)\n"
                "- Vehicles: 5 trucks, 800 kg capacity each\n"
                "- Constraints: Weight capacity, SLA time windows, traffic multipliers\n"
                "- Objective: Minimize weighted score = Cost(35%) + Time(30%) + Carbon(20%) + SLA(15%)\n\n"
                "Solver: OR-Tools with nearest-neighbour heuristic + 2-opt local search improvement.\n"
                "Why CVRP: Classic VRP doesn't consider capacity. CVRP ensures no truck is overloaded."
            ),
        },

        # ── OPTIMIZATION ENGINE ───────────────────────────────────────────────
        {
            "id": "optimization",
            "triggers": ["optimize","optimization","weighted","objective","score",
                         "solver","ortools","or-tools","heuristic","algorithm",
                         "nearest neighbour","2-opt","local search","how optimize",
                         "how does it work","how it work","how route"],
            "text": (
                "OPTIMIZATION ENGINE:\n"
                "Algorithm: CVRP with multi-objective weighted scoring.\n\n"
                "OBJECTIVE FUNCTION (MO Score per stop):\n"
                "  MO_score = 0.35×norm(cost) + 0.30×norm(time) + 0.20×norm(carbon) + 0.15×norm(SLA)\n"
                "  Lower score = better stop assignment.\n\n"
                "SOLVER STEPS:\n"
                "1. Nearest-neighbour heuristic — greedy initial route construction\n"
                "2. 2-opt local search — swap pairs of edges to reduce total distance\n"
                "3. Capacity check — ensure no truck exceeds 800 kg\n"
                "4. SLA check — flag stops that breach time windows\n"
                "5. Re-optimization — triggered by traffic spike >30% or priority escalation\n\n"
                "WEIGHTS RATIONALE:\n"
                "Cost(35%): Primary business KPI\n"
                "Time(30%): Customer experience and SLA compliance\n"
                "Carbon(20%): ESG and sustainability mandate\n"
                "SLA(15%): Penalty avoidance and contract compliance"
            ),
        },

        # ── RAG / AI ASSISTANT ───────────────────────────────────────────────
        {
            "id": "rag_definition",
            "triggers": ["rag","retrieval","retrieval augmented","bm25","vectorless",
                         "how does assistant work","how ai works","how chatbot",
                         "nlp","natural language","ai assistant","intelligent",
                         "pipeline","offline ai","no api","knowledge base"],
            "text": (
                "LoRRI AI ASSISTANT — VECTORLESS RAG PIPELINE:\n\n"
                "RAG = Retrieval-Augmented Generation.\n"
                "LoRRI uses a vectorless variant — no embeddings, no vector database.\n\n"
                "4-STEP PIPELINE:\n"
                "Step 1 — Rule Router: Pattern-matches common questions for instant answers.\n"
                "Step 2 — BM25-lite Retriever: Scores knowledge chunks by keyword frequency.\n"
                "         Multi-word triggers score 2x, single-word triggers score 1x.\n"
                "Step 3 — Pandas Retriever: Queries live DataFrames for truck/route/SLA data.\n"
                "Step 4 — Local Synthesizer: Assembles retrieved context into a response.\n"
                "         NO external API call. Fully offline.\n\n"
                "WHY VECTORLESS: Deterministic, fast, interpretable, zero-cost, works offline.\n"
                "Confidence score: 99% rule match | 92% RAG+Pandas | 88% RAG only | 72% fallback."
            ),
        },

        # ── DATA & DATASETS ───────────────────────────────────────────────────
        {
            "id": "data_definition",
            "triggers": ["data","dataset","csv","shipments","shipment","routes.csv",
                         "vehicle_summary","metrics","generate","synthetic","columns",
                         "fields","schema","what data","input data","traffic_mult",
                         "mo_score","sla_hours","weight","priority"],
            "text": (
                "PROJECT DATA & SCHEMA:\n\n"
                "shipments.csv — One row per delivery city:\n"
                "  city, latitude, longitude, weight(kg), priority(HIGH/MEDIUM/LOW),\n"
                "  sla_hours(24/48/72), traffic_mult(1.0–3.0), demand\n\n"
                "routes.csv — One row per stop (stop-level detail):\n"
                "  shipment_id, vehicle(1-5), stop_order, city, latitude, longitude,\n"
                "  weight, priority, travel_time_hr, fuel_cost, toll_cost, driver_cost,\n"
                "  sla_hours, sla_breach_hr, sla_penalty, carbon_kg, total_cost, mo_score\n\n"
                "vehicle_summary.csv — One row per truck:\n"
                "  vehicle(1-5), stops, load_kg, distance_km, time_hr, fuel_cost,\n"
                "  toll_cost, driver_cost, sla_penalty, total_cost, carbon_kg,\n"
                "  sla_breaches, utilization_pct\n\n"
                "metrics.csv — Aggregated run-level KPIs.\n"
                "Data is synthetically generated via generate_data.py to simulate\n"
                "realistic Indian logistics conditions."
            ),
        },

        # ── PRICING MODEL ─────────────────────────────────────────────────────
        {
            "id": "pricing",
            "triggers": ["fuel","toll","driver","cost","price","inr","rupee",
                         "penalty","wage","rs","pricing","rate","per km",
                         "per hour","cost model","how cost","calculate cost"],
            "text": (
                "COST MODEL (Rs. INR):\n\n"
                "FUEL COST:    Rs.12 per km travelled\n"
                "DRIVER COST:  Rs.180 per hour of travel time\n"
                "TOLL COST:    Variable by corridor (highway premium)\n"
                "SLA PENALTY:  Rs.500 per hour of late delivery breach\n\n"
                "TOTAL COST per stop = fuel_cost + toll_cost + driver_cost + sla_penalty\n\n"
                "BASELINE vs OPTIMIZED:\n"
                f"Baseline total:  Rs.{base['total_cost']:,.0f}\n"
                f"Optimized total: Rs.{opt['total_cost']:,.0f}\n"
                f"Saved:           Rs.{base['total_cost']-opt['total_cost']:,.0f} "
                f"({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}%)"
            ),
        },

        # ── SLA ───────────────────────────────────────────────────────────────
        {
            "id": "sla",
            "triggers": ["sla","late","breach","delay","on time","delivery",
                         "promise","window","adherence","service level",
                         "sla definition","what is sla","explain sla"],
            "text": (
                "SLA — SERVICE LEVEL AGREEMENT:\n"
                "Definition: A contractual commitment on delivery time.\n"
                "A breach occurs when actual delivery time exceeds the promised window.\n\n"
                "LoRRI SLA WINDOWS:\n"
                "  HIGH priority   → must deliver within 24 hours\n"
                "  MEDIUM priority → must deliver within 48 hours\n"
                "  LOW priority    → must deliver within 72 hours\n\n"
                "BREACH PENALTY: Rs.500 per hour of breach\n"
                f"CURRENT RUN:\n"
                f"  Optimized SLA adherence: {opt['sla_pct']:.1f}% (baseline was {base['sla_pct']:.0f}%)\n"
                f"  Breached cities: {opt['breaches']}\n"
                f"  Total penalties: Rs.{veh_sum['sla_penalty'].sum():,.0f}"
            ),
        },

        # ── CARBON / SUSTAINABILITY ───────────────────────────────────────────
        {
            "id": "carbon",
            "triggers": ["carbon","co2","emission","green","environment",
                         "sustainability","eco","tree","pollution","climate",
                         "carbon footprint","emission factor","kg co2"],
            "text": (
                "CARBON & SUSTAINABILITY:\n"
                "Emission factor: 0.27 kg CO2 per km (standard diesel truck).\n\n"
                f"CURRENT RUN:\n"
                f"  Optimized: {opt['carbon_kg']:,.1f} kg CO2\n"
                f"  Baseline:  {base['carbon_kg']:,.1f} kg CO2\n"
                f"  Saved:     {base['carbon_kg']-opt['carbon_kg']:,.1f} kg "
                f"({(base['carbon_kg']-opt['carbon_kg'])/base['carbon_kg']*100:.1f}% reduction)\n\n"
                f"REAL-WORLD EQUIVALENTS:\n"
                f"  🌳 {int((base['carbon_kg']-opt['carbon_kg'])/21):,} trees absorbing CO2 for 1 year\n"
                f"  🚗 {int((base['carbon_kg']-opt['carbon_kg'])/2400)} cars removed from road for 1 year\n\n"
                "WHY IT MATTERS: Logistics accounts for ~11% of global CO2 emissions. "
                "LoRRI's carbon weight (20%) forces shorter, cleaner routes."
            ),
        },

        # ── FLEET SUMMARY ─────────────────────────────────────────────────────
        {
            "id": "fleet_summary",
            "triggers": ["fleet","total","summary","overall","depot","mumbai",
                         "shipment","truck","vehicle","save","saving","saved",
                         "baseline","optimized","run","how many","capacity",
                         "800 kg","utilization"],
            "text": (
                f"FLEET SUMMARY (Mumbai Depot):\n"
                f"Trucks: {opt['n_vehicles']} | Capacity: 800 kg each | Shipments: {opt['n_ships']}\n\n"
                f"OPTIMIZED RUN:\n"
                f"  Total cost:    {inr(opt['total_cost'])}\n"
                f"  Total distance:{opt['distance_km']:,.0f} km\n"
                f"  Total time:    {opt['time_hr']:,.1f} hr\n"
                f"  Carbon:        {opt['carbon_kg']:,.1f} kg CO2\n"
                f"  SLA adherence: {opt['sla_pct']:.1f}%\n\n"
                f"BASELINE (no AI):\n"
                f"  Total cost:    {inr(base['total_cost'])}\n"
                f"  Total distance:{base['distance_km']:,.0f} km\n"
                f"  Carbon:        {base['carbon_kg']:,.1f} kg CO2\n"
                f"  SLA adherence: {base['sla_pct']:.0f}%\n\n"
                f"NET IMPROVEMENT:\n"
                f"  Cost saved:    {inr(base['total_cost']-opt['total_cost'])} "
                f"({(base['total_cost']-opt['total_cost'])/base['total_cost']*100:.1f}%)\n"
                f"  CO2 saved:     {base['carbon_kg']-opt['carbon_kg']:,.1f} kg\n"
                f"  SLA gain:      +{opt['sla_pct']-base['sla_pct']:.0f} percentage points"
            ),
        },

        # ── ROUTES ────────────────────────────────────────────────────────────
        {
            "id": "routes",
            "triggers": ["route","all routes","truck routes","which route","route map",
                         "where go","cities covered","stops","delivery path","corridor",
                         "north route","south route","west route","east route"],
            "text": (
                "ALL TRUCK ROUTES (from Mumbai Depot):\n\n"
                "Truck 1 (North): Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada\n"
                "  Coverage: Central India + Delhi NCR + Andhra Pradesh\n\n"
                "Truck 2 (West): Mumbai → Surat → Vadodara → Raipur\n"
                "  Coverage: Gujarat corridor + Chhattisgarh\n\n"
                "Truck 3 (South+North): Mumbai → Aurangabad → Solapur → Madurai → Jammu\n"
                "  Coverage: Marathwada + Deep South + Jammu (high priority)\n\n"
                "Truck 4 (West+South): Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram\n"
                "  Coverage: Saurashtra + Rajasthan + Kerala\n\n"
                "Truck 5 (South-West): Mumbai → Hubli → Mangalore → Bengaluru\n"
                "  Coverage: Karnataka tech corridor"
            ),
        },

        # ── TRAFFIC / RE-OPT ─────────────────────────────────────────────────
        {
            "id": "traffic",
            "triggers": ["traffic","jam","congestion","disruption","reoptimize",
                         "re-optimize","threshold","delay","multiplier",
                         "traffic multiplier","how reoptimize","what triggers",
                         "dynamic","real time","live"],
            "text": (
                "TRAFFIC & DYNAMIC RE-OPTIMIZATION:\n\n"
                "TRAFFIC MULTIPLIER: A float (1.0x–3.0x) per city.\n"
                "  1.0x = free flow | 1.5x = moderate | 2.0x = heavy | 3.0x = severe jam\n"
                "  Effect: avg_speed = 55 km/h ÷ traffic_multiplier\n\n"
                "RE-OPTIMIZATION TRIGGER CONDITIONS:\n"
                "  1. Traffic delay exceeds 30% above planned ETA\n"
                "  2. Shipment priority escalated to HIGH\n\n"
                "RE-OPTIMIZATION PROCESS:\n"
                "  - OR-Tools re-runs 2-opt local search on affected truck only\n"
                "  - Completes in ~1–2 seconds\n"
                "  - Delayed city moved to end of route (de-prioritized) or promoted to front\n\n"
                "RISK SCORING:\n"
                "  risk = (traffic_mult/1.8 × 0.6) + (urgency_weight × 0.4)\n"
                "  HIGH >0.7 | MONITOR >0.4 | STABLE ≤0.4"
            ),
        },

        # ── EXPLAINABILITY / SHAP ─────────────────────────────────────────────
        {
            "id": "explainability",
            "triggers": ["explain","explainability","shap","feature importance",
                         "permutation","why this route","why chose","interpret",
                         "xai","transparent","black box","which feature",
                         "most important","importance"],
            "text": (
                "EXPLAINABILITY — SHAP-STYLE PERMUTATION IMPORTANCE:\n\n"
                "Definition: Permutation importance measures how much model performance "
                "degrades when a feature's values are randomly shuffled.\n\n"
                "LoRRI FEATURES RANKED (approximate):\n"
                "1. Travel Time    — highest influence on MO score\n"
                "2. Fuel Cost      — directly proportional to distance\n"
                "3. Carbon Emitted — correlated with distance\n"
                "4. SLA Breach     — binary penalty multiplier\n"
                "5. Driver Cost    — time-dependent\n"
                "6. Toll Cost      — corridor-dependent\n"
                "7. Package Weight — capacity constraint driver\n\n"
                "STOP-LEVEL CONTRIBUTION:\n"
                "Each stop's MO score decomposed into 6 components weighted by the "
                "objective function. Visualized as stacked bar chart per city.\n\n"
                "PURPOSE: Makes AI routing decisions auditable and trustworthy."
            ),
        },

        # ── BASELINE DEFINITION ───────────────────────────────────────────────
        {
            "id": "baseline",
            "triggers": ["baseline","what is baseline","without ai","before optimization",
                         "no optimization","naive","brute force","benchmark",
                         "original route","unoptimized"],
            "text": (
                "BASELINE — DEFINITION:\n"
                "The baseline represents routing WITHOUT any AI optimization.\n"
                "It simulates a naive sequential assignment: trucks visit cities "
                "in the order they appear in the dataset, ignoring capacity, traffic, "
                "SLA windows, or cost efficiency.\n\n"
                "BASELINE METRICS:\n"
                f"  Total cost:    Rs.{base['total_cost']:,.0f}\n"
                f"  Distance:      {base['distance_km']:,.0f} km\n"
                f"  Time:          {base['time_hr']:,.1f} hr\n"
                f"  Carbon:        {base['carbon_kg']:,.1f} kg CO2\n"
                f"  SLA adherence: {base['sla_pct']:.0f}%\n\n"
                "The baseline is the benchmark. LoRRI's AI optimization is measured "
                "as percentage improvement over this baseline."
            ),
        },

        # ── MO SCORE DEFINITION ───────────────────────────────────────────────
        {
            "id": "mo_score",
            "triggers": ["mo score","mo_score","multi objective","multi-objective",
                         "score meaning","what is mo","scoring","how scored",
                         "objective function","weighted score","normalize"],
            "text": (
                "MO SCORE — MULTI-OBJECTIVE SCORE:\n\n"
                "Definition: A single normalized score (0–1) that captures the "
                "combined cost, time, carbon, and SLA burden of each delivery stop.\n\n"
                "FORMULA:\n"
                "  MO_score = 0.35×norm(total_cost)\n"
                "           + 0.30×norm(travel_time_hr)\n"
                "           + 0.20×norm(carbon_kg)\n"
                "           + 0.15×norm(sla_breach_hr)\n\n"
                "NORMALIZATION: Each feature scaled to [0,1] using min-max normalization "
                "across all stops in the current run.\n\n"
                "INTERPRETATION:\n"
                "  Low score → efficient, fast, clean, on-time delivery\n"
                "  High score → expensive, slow, polluting, or late\n\n"
                "USE: Used by the CVRP solver to rank and assign stops to trucks."
            ),
        },

        # ── UTILIZATION DEFINITION ────────────────────────────────────────────
        {
            "id": "utilization",
            "triggers": ["utilization","utilisation","capacity used","load factor",
                         "truck load","how full","efficiency","under loaded",
                         "overloaded","800 kg","vehicle capacity"],
            "text": (
                "TRUCK UTILIZATION:\n\n"
                "Definition: Percentage of maximum 800 kg capacity used per truck.\n"
                "  utilization_pct = (total load_kg / 800) × 100\n\n"
                "TARGET: 70–90% utilization is ideal.\n"
                "  Below 70% = under-utilized (consolidation opportunity)\n"
                "  Above 90% = near-capacity (SLA risk if extra stop added)\n\n"
                f"CURRENT RUN AVERAGES:\n"
                f"  Fleet avg utilization: {veh_sum['utilization_pct'].mean():.1f}%\n"
                + "\n".join([
                    f"  Truck {int(r['vehicle'])}: {r['load_kg']:.0f} kg = {r['utilization_pct']:.1f}%"
                    for _, r in veh_sum.iterrows()
                ])
            ),
        },

        # ── COST SAVING RECOMMENDATIONS ───────────────────────────────────────
        {
            "id": "cost_saving",
            "triggers": ["suggest","recommendation","improve","reduce cost","save more",
                         "tip","advice","better","optimize further","action",
                         "what can we do","next step","improvement"],
            "text": (
                "COST SAVING RECOMMENDATIONS:\n\n"
                "1. CONSOLIDATE Truck 2 & 5 — both under 70% utilization.\n"
                "   Merging saves one truck's driver cost (~Rs.180/hr).\n\n"
                "2. AVOID HIGH-TRAFFIC corridors during peak hours (8–10am, 5–8pm).\n"
                "   Saves fuel and prevents SLA breaches.\n\n"
                "3. UPGRADE LOW-PRIORITY shipments to 72hr SLA window.\n"
                "   Reduces penalty exposure by removing tight time constraints.\n\n"
                "4. USE EXPRESS HIGHWAY only when time savings exceed toll premium.\n"
                "   Break-even: express saves >Rs.500 in driver cost per Rs.300 extra toll.\n\n"
                "5. CLUSTER southern cities — Madurai + Thiruvananthapuram on one truck.\n"
                "   Currently split across Truck 3 and Truck 4 (inefficient).\n\n"
                "6. PRE-POSITION trucks at Pune and Ahmedabad regional hubs.\n"
                "   Reduces Mumbai depot congestion and cuts first-leg distance by ~15%."
            ),
        },

        # ── HACKATHON / CONTEXT ───────────────────────────────────────────────
        {
            "id": "hackathon",
            "triggers": ["hackathon","synapflow","competition","problem statement",
                         "ps4","judge","submission","team","winner","event",
                         "challenge","why built","motivation"],
            "text": (
                "HACKATHON CONTEXT:\n\n"
                "Event: Synapflow Hackathon\n"
                "Problem Statement: #4 — AI-Powered Logistics Optimization\n"
                "Team / Company: LogisticsNow\n\n"
                "PROBLEM STATEMENT SUMMARY:\n"
                "Build an intelligent logistics platform that:\n"
                "1. Optimizes multi-stop delivery routes for a fleet of trucks\n"
                "2. Minimizes cost, time, carbon emissions, and SLA violations\n"
                "3. Handles real-world constraints: traffic, capacity, priorities\n"
                "4. Provides explainability for every routing decision\n"
                "5. Enables dynamic re-optimization on disruptions\n\n"
                "LoRRI SOLUTION:\n"
                "Full-stack Streamlit dashboard with CVRP optimizer, interactive maps, "
                "financial analytics, carbon tracking, explainability module, "
                "re-optimization simulator, AI route predictor, and an offline RAG chatbot."
            ),
        },

        # ── PAGES / FEATURES ──────────────────────────────────────────────────
        {
            "id": "features_pages",
            "triggers": ["features","pages","dashboard","what can you do","capabilities",
                         "modules","sections","navigation","menu","tabs",
                         "what pages","route predictor","simulator","financial",
                         "explainability page","carbon page"],
            "text": (
                "LoRRI DASHBOARD PAGES (9 modules):\n\n"
                "1. About LoRRI         — Company info, vision, platform capabilities\n"
                "2. AI Assistant        — Offline RAG chatbot (this page)\n"
                "3. Dashboard Summary   — KPIs: cost saved, SLA%, carbon, utilization\n"
                "4. Route Map           — Interactive India map with all truck routes\n"
                "5. Financial Analysis  — Cost breakdown, waterfall chart, per-truck table\n"
                "6. Carbon & SLA        — Sustainability metrics, breach heatmap, SLA gauge\n"
                "7. Explainability      — Feature importance, MO score decomposition\n"
                "8. Re-opt Simulator    — Simulate traffic jams + priority escalations\n"
                "9. AI Route Predictor  — Predict cost/ETA/risk for any new delivery\n\n"
                "Sidebar: Live fleet stats, toggle controls, depot sync button."
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

    # ── DEFINITION QUERIES ───────────────────────────────────────────────────

    if any(k in q for k in ["what is lorri","lorri mean","lorri stand","lorri full form",
                             "what is this project","explain lorri","define lorri"]):
        return (
            "**LoRRI = Logistics Rating & Intelligence**\n\n"
            "An AI-powered Route Optimization Engine built by LogisticsNow for the "
            "**Synapflow Hackathon (Problem Statement 4)**.\n\n"
            "**Purpose:** Solve the Capacitated Vehicle Routing Problem (CVRP) for "
            "India's logistics networks — minimizing cost, time, carbon emissions, "
            "and SLA violations simultaneously.\n\n"
            "**Stack:** Python · Streamlit · OR-Tools · Plotly · Pandas\n"
            "**Depot:** Mumbai · **Fleet:** 5 trucks · 800 kg each\n"
            "**Covers:** Pan-India deliveries across 5 optimized corridors", 99
        )

    if any(k in q for k in ["what is cvrp","cvrp mean","cvrp stand","define cvrp",
                             "capacitated vehicle","routing problem","vrp"]):
        return (
            "**CVRP — Capacitated Vehicle Routing Problem**\n\n"
            "An NP-hard optimization problem: find the cheapest set of routes for a "
            "fleet of vehicles to serve all customers, without exceeding each vehicle's capacity.\n\n"
            "**LoRRI's CVRP setup:**\n"
            "- Depot: Mumbai\n"
            "- Vehicles: 5 trucks · 800 kg capacity each\n"
            "- Customers: Cities across India\n"
            "- Constraints: weight capacity, SLA time windows, traffic multipliers\n"
            "- Objective: `0.35×Cost + 0.30×Time + 0.20×Carbon + 0.15×SLA`\n\n"
            "**Solver:** OR-Tools nearest-neighbour heuristic + 2-opt local search", 98
        )

    if any(k in q for k in ["what is rag","rag mean","how does assistant","how ai works",
                             "vectorless","bm25","pipeline","how chatbot","offline ai"]):
        return (
            "**LoRRI AI — Vectorless RAG Pipeline**\n\n"
            "RAG = Retrieval-Augmented Generation. LoRRI uses a **vectorless** variant — "
            "no embeddings, no vector database, no external API.\n\n"
            "**4-Step Pipeline:**\n"
            "1. ⚡ **Rule Router** — Pattern-matches common questions instantly\n"
            "2. 📚 **BM25-lite Retriever** — Scores 20 knowledge chunks by keyword frequency\n"
            "3. 🐼 **Pandas Retriever** — Queries live DataFrames for real fleet data\n"
            "4. 🤖 **Local Synthesizer** — Assembles answer from retrieved context\n\n"
            "**No API call. Fully offline. Deterministic. Zero-cost.**\n"
            "Confidence: 99% rule · 92% RAG+Pandas · 88% RAG only · 72% fallback", 99
        )

    if any(k in q for k in ["what is mo score","mo score mean","mo_score","multi objective",
                             "what is score","objective function","weighted score"]):
        return (
            "**MO Score — Multi-Objective Score**\n\n"
            "A normalized score (0–1) per delivery stop capturing combined burden:\n\n"
            "```\n"
            "MO_score = 0.35 × norm(total_cost)\n"
            "         + 0.30 × norm(travel_time_hr)\n"
            "         + 0.20 × norm(carbon_kg)\n"
            "         + 0.15 × norm(sla_breach_hr)\n"
            "```\n\n"
            "Features normalized via min-max scaling across all stops.\n\n"
            "**Interpretation:** Low score = efficient, fast, clean, on-time. "
            "High score = costly, slow, polluting, or late.\n\n"
            "Used by the CVRP solver to rank and assign stops to trucks optimally.", 98
        )

    if any(k in q for k in ["what is baseline","baseline mean","without ai","before optimization",
                             "unoptimized","no optimization","benchmark","naive route"]):
        return (
            "**Baseline — Definition**\n\n"
            "The baseline simulates routing **without any AI optimization**.\n"
            "Trucks visit cities in dataset order, ignoring capacity, traffic, SLA, or cost.\n\n"
            "**Baseline metrics:**\n"
            f"- Total cost: **Rs.{base['total_cost']:,.0f}**\n"
            f"- Distance: **{base['distance_km']:,.0f} km**\n"
            f"- Time: **{base['time_hr']:,.1f} hr**\n"
            f"- Carbon: **{base['carbon_kg']:,.1f} kg CO2**\n"
            f"- SLA adherence: **{base['sla_pct']:.0f}%**\n\n"
            "LoRRI's AI optimization is measured as percentage improvement over this baseline.", 98
        )

    if any(k in q for k in ["what is sla","sla definition","sla mean","sla stand",
                             "service level","explain sla","define sla"]):
        return (
            "**SLA — Service Level Agreement**\n\n"
            "A contractual promise on delivery time. Breach = actual time > promised window.\n\n"
            "**LoRRI SLA windows:**\n"
            "- 🔴 HIGH priority → deliver within **24 hours**\n"
            "- 🟡 MEDIUM priority → deliver within **48 hours**\n"
            "- 🟢 LOW priority → deliver within **72 hours**\n\n"
            f"**Breach penalty:** Rs.500 per hour late\n"
            f"**Current run:** {opt['sla_pct']:.1f}% adherence · "
            f"{opt['breaches']} cities breached · "
            f"Rs.{veh_sum['sla_penalty'].sum():,.0f} total penalties", 98
        )

    if any(k in q for k in ["what is utilization","utilization mean","truck capacity",
                             "how full","load factor","capacity used"]):
        return (
            "**Truck Utilization**\n\n"
            "Percentage of 800 kg maximum capacity used per truck.\n"
            "`utilization_pct = (load_kg / 800) × 100`\n\n"
            "**Target range:** 70–90% (optimal)\n"
            "- Below 70% = under-utilized → consolidation opportunity\n"
            "- Above 90% = near-capacity → SLA risk if stop added\n\n"
            f"**Current fleet average: {veh_sum['utilization_pct'].mean():.1f}%**\n"
            + "\n".join([
                f"- Truck {int(r['vehicle'])}: {r['load_kg']:.0f} kg = **{r['utilization_pct']:.1f}%**"
                for _, r in veh_sum.iterrows()
            ]), 97
        )

    if any(k in q for k in ["tech stack","technology used","built with","tools used",
                             "which library","python","streamlit","ortools","plotly"]):
        return (
            "**LoRRI Tech Stack:**\n\n"
            "- 🎨 **UI:** Streamlit (Python)\n"
            "- ⚙️ **Optimization:** OR-Tools (Google CVRP solver)\n"
            "- 📊 **Charts:** Plotly (maps, bar, gauge, waterfall, heatmap)\n"
            "- 🔢 **Data:** Pandas + NumPy\n"
            "- 🗺️ **Maps:** OpenStreetMap via Plotly Scattermap\n"
            "- 🤖 **AI:** Vectorless RAG (BM25-lite + Pandas, fully offline)\n"
            "- 📐 **Distance:** Haversine formula (great-circle)\n\n"
            "**Repo:** github.com/PranjalAmbwani1305/Route-Optimization-Engine-Agent-", 97
        )

    if any(k in q for k in ["what pages","features","what can you do","modules",
                             "dashboard sections","navigation","9 pages"]):
        return (
            "**LoRRI Dashboard — 9 Pages:**\n\n"
            "1. 🏢 About LoRRI — Company & platform overview\n"
            "2. 🤖 AI Assistant — This offline RAG chatbot\n"
            "3. 📊 Dashboard Summary — KPIs: savings, SLA, carbon, utilization\n"
            "4. 🗺️ Route Map — Interactive India delivery network\n"
            "5. 💰 Financial Analysis — Cost waterfall, per-truck breakdown\n"
            "6. 🌿 Carbon & SLA — Sustainability + breach heatmap\n"
            "7. 🧠 Explainability — Feature importance + MO score decomposition\n"
            "8. ⚡ Re-opt Simulator — Simulate traffic jams + priority escalations\n"
            "9. 🔮 AI Route Predictor — Cost/ETA/risk for new delivery", 97
        )

    if any(k in q for k in ["all routes","show routes","truck routes","which cities",
                             "where do trucks go","coverage","corridors"]):
        return (
            "**All 5 Truck Routes (from Mumbai Depot):**\n\n"
            "🚛 **Truck 1** — North: Mumbai → Pune → Nashik → Indore → Bhopal → Agra → Delhi → Vijayawada\n"
            "🚛 **Truck 2** — West: Mumbai → Surat → Vadodara → Raipur\n"
            "🚛 **Truck 3** — South+North: Mumbai → Aurangabad → Solapur → Madurai → Jammu\n"
            "🚛 **Truck 4** — West+South: Mumbai → Rajkot → Ahmedabad → Jodhpur → Thiruvananthapuram\n"
            "🚛 **Truck 5** — South-West: Mumbai → Hubli → Mangalore → Bengaluru\n\n"
            "Total coverage: Pan-India from Kashmir (Jammu) to Kerala (Thiruvananthapuram)", 97
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
