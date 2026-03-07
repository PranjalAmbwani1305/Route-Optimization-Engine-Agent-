# 🚚 LoRRI — AI-Powered Route Optimization Engine

> **LogisticsNow Hackathon Submission** · AI Route Optimization Engine · Problem Statement #4

---

<div align="center">

**Transforming static logistics planning into intelligent, adaptive, real-time route optimization**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![RAG](https://img.shields.io/badge/RAG-Enabled-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Our Solution](#-our-solution)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Tech Stack](#-tech-stack)
6. [Deliverables Checklist](#-deliverables-checklist)
7. [Dashboard Walkthrough](#-dashboard-walkthrough)
8. [India-Realistic Data Model](#-india-realistic-data-model)
9. [Performance Results](#-performance-results)
10. [Getting Started](#-getting-started)
11. [Project Structure](#-project-structure)
12. [Future Roadmap](#-future-roadmap)
13. [Team](#-team)

---

## 🔴 Problem Statement

Logistics companies execute **thousands of shipments daily** across complex transportation networks involving multiple origins, destinations, delivery windows, and vehicle constraints. Current routing systems are fundamentally broken:

| Pain Point | Impact |
|---|---|
| Static route planning, no real-time adaptation | Vehicles follow suboptimal routes even when conditions change |
| No consideration of delivery time windows or vehicle capacity | Missed SLAs, overloaded trucks, customer complaints |
| Inefficient delivery sequencing across multi-stop routes | Unnecessary backtracking, wasted kilometres |
| Rising fuel costs with no optimization lever | Operational costs spiral with no visibility |
| Zero predictive intelligence | Reactive decisions, not proactive planning |
| No AI-driven continuous improvement | Same mistakes repeated across thousands of shipments |

> **The core problem:** There is no AI-driven system capable of continuously optimizing routes based on shipment demand, network conditions, and operational constraints — at scale, in real time, with explainability.

---

## ✅ Our Solution

**LoRRI AI Route Optimization Engine** is a full-stack AI logistics intelligence platform that replaces static route planning with an intelligent, adaptive, and explainable optimization system.

The system operates on a four-layer pipeline:

```
📦 India-Realistic Shipment Data  (real cities, INR, LoRRI truck types)
           ↓
🧠 AI Route Optimization Engine   (VRPTW + Multi-objective + Stochastic Demand)
           ↓
🤖 RAG-Powered AI Assistant       (Natural language querying over logistics data)
           ↓
📊 Interactive Intelligence Dashboard  (Streamlit + Plotly, 9 pages)
```

**What makes our solution different:**

- Solves the **Vehicle Routing Problem with Time Windows (VRPTW)** — the gold standard in logistics optimization
- Incorporates **stochastic demand and delay probabilities** — not just shortest path
- Balances **five objectives simultaneously**: distance, cost, time, CO₂, and delivery priority
- Provides **Explainable AI (XAI)** — judges and operators can see *why* each routing decision was made
- Includes a **RAG-powered chatbot** that lets any user query the system in plain English
- Built on **India-realistic data**: 30 real cities, 21 real NH corridors, 7 LoRRI truck categories, INR pricing

---

## ✨ Key Features

### 1. 🗺️ Route Map & Visualization
Interactive Plotly map of India showing all optimized delivery routes. Color-coded by vehicle, with stop sequencing, origin/destination markers, and corridor-level detail (highway, road quality, distance, toll cost).

### 2. 📊 Dashboard Summary & KPIs
Executive-level headline metrics comparing baseline vs. optimized performance:
- Total distance saved (km and %)
- Cost savings (₹ and %)
- CO₂ reduction (kg)
- Fleet utilization improvement (%)
- On-time delivery rate (SLA %)

### 3. 💰 Financial Analysis
Deep-dive into freight economics broken down by truck category, lane, and highway corridor. Includes toll optimization analysis — scenarios where splitting into 2 smaller trucks saves money vs. 1 large truck (e.g., 2× Eicher 10T instead of 1× Volvo 32T to avoid HXL toll multiplier on NH48).

### 4. 🌿 Carbon & SLA Tracking
- CO₂ emissions per route and per truck type
- Emissions reduction quantified from optimization
- SLA compliance rate and time window adherence by shipment priority
- Carbon cost monetization (₹ per kg CO₂ equivalent)

### 5. 🧠 Explainability (XAI)
For every optimized route, the system explains in plain English:
- Why this route was chosen over alternatives
- Which constraints were binding (time window, vehicle capacity, toll avoidance)
- Trade-off breakdown: what was sacrificed vs. what was gained
- Confidence score of each optimization decision

### 6. ⚡ Re-optimization Simulator
Live scenario simulator — change parameters and see real-time re-optimization:
- Toggle real-time traffic feed on/off
- Add an urgent shipment mid-route
- Simulate vehicle breakdown and fleet re-assignment
- Test toll avoidance strategies

### 7. 🤖 LoRRI AI Assistant (RAG Agent)
Conversational AI interface powered by Retrieval-Augmented Generation (RAG):
- Ask questions in plain English over your entire logistics dataset
- Grounded in actual shipment, route, and metrics data — no hallucinations
- Example queries:
  - *"Which lanes have the highest delay risk this week?"*
  - *"What's the cheapest truck type for Mumbai to Hyderabad with a 12-ton load?"*
  - *"Show me all routes where CO₂ can be reduced by switching truck type"*

### 8. 🔮 AI Route Predictor
Predictive intelligence for future shipments:
- 90-day demand forecast by lane (weekly granularity)
- Seasonal spike detection (Diwali +45%, FY-end +25%, Monsoon −20%)
- Recommended fleet pre-positioning by hub city
- Stochastic delay risk scoring per corridor

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│  generate_data.py                                                    │
│  ├── shipments.csv       (200 shipments, real Indian cities + INR)  │
│  ├── vehicles.csv        (35 trucks, 7 LoRRI categories)            │
│  ├── lanes.csv           (21 NH corridors + state roads + tolls)    │
│  └── demand_forecast.csv (90-day stochastic demand, seasonal)       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     OPTIMIZATION ENGINE                              │
│                                                                      │
│  route_solver.py                                                     │
│  ├── VRPTW Algorithm     (OR-Tools / NumPy)                         │
│  ├── Multi-objective     (distance + cost + time + CO₂ + priority)  │
│  ├── Stochastic Module   (Monte Carlo delay simulation)             │
│  ├── Toll Optimizer      (fleet split vs. single truck decision)    │
│  └── Output → routes.csv + metrics.csv + vehicle_summary.csv        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG LAYER                                    │
│                                                                      │
│  rag_engine.py                                                       │
│  ├── Document Indexing   (shipments + routes + metrics → vector DB) │
│  ├── Query Processing    (natural language → retrieval → answer)    │
│  └── LLM Response        (grounded, citation-backed answers)        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DASHBOARD LAYER                                 │
│                                                                      │
│  dashboard.py  (Streamlit + Plotly)                                 │
│  ├── About LoRRI                                                    │
│  ├── LoRRI AI Assistant (RAG chatbot)                               │
│  ├── Dashboard Summary (KPIs + before/after)                        │
│  ├── Route Map (interactive India map)                              │
│  ├── Financial Analysis (INR cost breakdown)                        │
│  ├── Carbon & SLA                                                   │
│  ├── Explainability (XAI per-route reasoning)                       │
│  ├── Re-optimization Simulator                                      │
│  └── AI Route Predictor (90-day forecast)                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    🌐 Deployed on Streamlit Cloud
```

### How It Would Integrate with LoRRI Production

In a production deployment with LoRRI's platform, the architecture extends as:

```
LoRRI Platform (lorri.in)
        │
        │  REST API / Data Export
        ↓
  Data Ingestion Layer
  (replaces generate_data.py with live connector)
        │
        ├──→ Route Optimization Engine  (route_solver.py — unchanged)
        ├──→ RAG Agent  (re-indexes live LoRRI freight data)
        └──→ Dashboard  (Streamlit app or embedded iframe in LoRRI portal)
```

> **Important:** No access to LoRRI's private carrier data was used. The system is architected as a drop-in module — when LoRRI provides an API, `generate_data.py` is swapped for a live connector. Everything downstream (optimizer, RAG, dashboard) remains identical.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **Optimization** | OR-Tools, NumPy | VRPTW algorithm, Monte Carlo simulation |
| **Data** | Pandas | Data manipulation, CSV I/O |
| **Dashboard UI** | Streamlit | Multi-page interactive web interface |
| **Visualization** | Plotly | Route maps, bar/line charts, KPI gauges |
| **RAG Engine** | LangChain + FAISS / ChromaDB | Vector search, document retrieval |
| **LLM** | OpenAI GPT / Groq | Natural language responses in AI Assistant |
| **Embeddings** | OpenAI / HuggingFace | Semantic search over logistics data |
| **Deployment** | Streamlit Cloud | Zero-infra public URL deployment |
| **Data Model** | India-realistic synthetic | 30 cities, 7 truck types, real NH tolls, INR |

---

## 📦 Deliverables Checklist

### ✅ Deliverable 1 — AI-Powered Optimization Engine Prototype

| Requirement | Status | Implementation |
|---|---|---|
| Route optimization algorithm | ✅ Done | `route_solver.py` — VRPTW with OR-Tools |
| Multi-stop route planning logic | ✅ Done | Nearest-neighbor heuristic + 2-opt improvement |
| Delivery window constraints | ✅ Done | `earliest_pickup` + `latest_delivery` per shipment |
| Vehicle capacity constraints | ✅ Done | `capacity_ton` + `capacity_cbm` enforced per truck |
| Shipment priority handling | ✅ Done | Priority flag weighted in objective function |
| Dynamic re-routing | ✅ Done | Re-optimization Simulator page |
| Continuous learning signal | ✅ Done | RAG agent learns from historical shipment performance |

### ✅ Deliverable 2 — Visualization Dashboard

| Requirement | Status | Implementation |
|---|---|---|
| Route visualization interface | ✅ Done | Route Map page — interactive Plotly map of India |
| Current vs. optimized comparison | ✅ Done | Dashboard Summary — before/after KPI cards + charts |
| Fleet utilization view | ✅ Done | Vehicle Summary + utilization charts |
| Financial breakdown | ✅ Done | Financial Analysis page — INR cost per route/truck/lane |
| Carbon impact view | ✅ Done | Carbon & SLA page — CO₂ per route + savings |
| Explainability view | ✅ Done | Explainability page — per-route AI decision reasoning |

### ✅ Deliverable 3 — Performance Simulation (Measurable Improvements)

| Metric | What We Measure | Where to See It |
|---|---|---|
| Reduction in total travel distance | Baseline km vs. optimized km | Dashboard Summary — Distance Saved card |
| Reduction in transportation cost | Baseline ₹ vs. optimized ₹ | Dashboard Summary — Cost Savings card |
| Faster delivery times | Time window adherence rate | Carbon & SLA — SLA Compliance chart |
| Improved fleet efficiency | Truck utilization % before/after | Dashboard Summary — Fleet Utilization card |
| CO₂ emissions reduction | kg CO₂ before/after | Carbon & SLA — Emissions Reduction chart |

> All five metrics are computed live by `route_solver.py` and rendered in the dashboard with percentage improvement badges.

---

## 📊 Dashboard Walkthrough

### Page 1 — About LoRRI
Landing page with project vision, problem framing, and system overview. Sets context for judges before diving into the live demo.

### Page 2 — LoRRI AI Assistant *(Best demo moment)*
RAG-powered chatbot. Type any logistics question in plain English. The agent retrieves relevant data from shipments, routes, and metrics to give grounded, accurate answers.

**Suggested live demo queries for judges:**
- *"Which 3 routes have the highest toll costs?"*
- *"What truck should I use for a 15-ton steel shipment from Mumbai to Nagpur?"*
- *"How much CO₂ would we save by switching all HXL trucks to HCV on routes under 300km?"*
- *"Which lanes are most at risk of delays this week?"*

### Page 3 — Dashboard Summary
KPI scorecards: distance saved, cost saved, CO₂ reduced, SLA %, fleet utilization. Below: before/after comparison charts across all five performance metrics.

### Page 4 — Route Map
Interactive map of India with all optimized routes plotted. Click any route to inspect: truck type, distance, cost, highway used, toll amount, and delivery stop sequence.

### Page 5 — Financial Analysis
Cost breakdown by truck category, lane, and highway corridor. Toll optimization analysis highlighting where smaller truck combinations beat single large trucks on cost.

### Page 6 — Carbon & SLA
CO₂ emissions dashboard with route-level and fleet-level views. SLA compliance rate. Time window adherence by priority tier. Carbon savings quantified in both kg and ₹.

### Page 7 — Explainability (XAI)
For every optimized route: a plain-English explanation of why the AI chose this route, what alternatives existed, which constraints drove the decision, and a confidence score. This is critical for building trust with non-technical logistics managers.

### Page 8 — Re-optimization Simulator
Live controls: toggle traffic, add emergency shipments, simulate breakdowns, change fleet size. The optimizer re-solves and shows updated routes and metrics in real time — demonstrates the system's dynamic capability.

### Page 9 — AI Route Predictor
90-day demand forecast by lane. Seasonal spikes flagged. Fleet pre-positioning recommendations by hub city. Delay risk heatmap across all corridors.

---

## 🇮🇳 India-Realistic Data Model

All synthetic data is modelled on real Indian logistics conditions to demonstrate production readiness:

**Cities & Geography** — 30 real Indian cities across 15 states with accurate coordinates:
- Tier-1 hubs: Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Pune, Ahmedabad, Kolkata
- Tier-2/3: Nashik, Nagpur, Surat, Vadodara, Coimbatore, Kochi, Guwahati, Raipur, Varanasi, Amritsar, Jodhpur and more

**Truck Types (aligned with LoRRI's platform categories)**

| Category | Vehicle | Capacity | Cost/km (₹) | CO₂/km (kg) |
|---|---|---|---|---|
| LCV | Tata Ace Mini | 0.75T | ₹12 | 0.18 |
| SCV | Tata 407 | 2.5T | ₹18 | 0.28 |
| ICV | Eicher 10T | 10T | ₹28 | 0.55 |
| MCV | Ashok Leyland 14T | 14T | ₹35 | 0.72 |
| HCV | Tata 1109 20T | 20T | ₹45 | 0.95 |
| HXL | Volvo 32T | 32T | ₹65 | 1.40 |
| MXL | Trailer 40T | 40T | ₹80 | 1.80 |

**Highway Corridors** — 21 real NH corridors with INR toll amounts:
NH48 (Mumbai–Delhi Expressway), NH44 (Delhi–Bengaluru), NH19 (Golden Quadrilateral), NH65 (Pune–Hyderabad), NH3, NH47, NH53, NH62, NH544, and more.

**Seasonal Demand Model**
- Diwali / Festive Season (Oct–Nov): +45% demand spike
- Financial Year-End (Mar–Apr): +25% demand spike
- Monsoon Slowdown (Jun–Jul): −20% demand reduction

**Cargo Types** — FMCG, Automotive Parts, Electronics, Textiles, Pharmaceuticals, Steel/Metal, Chemicals, Food & Agri, E-commerce, Construction — each with realistic density, fragility, and hazmat flags.

---

## 📈 Performance Results

The optimization engine demonstrates measurable improvements across all five required KPIs. Computed live by `route_solver.py` and displayed in the Dashboard Summary page:

- **Distance Reduction** — Optimized routing eliminates backtracking and consolidates overlapping delivery lanes
- **Cost Savings** — Toll-aware fleet selection and load consolidation reduce per-shipment freight cost
- **CO₂ Reduction** — Fewer kilometres driven + right-sized trucks = lower emissions per ton delivered
- **SLA Improvement** — Time-window-constrained VRPTW increases on-time delivery rate
- **Fleet Utilization** — Better load matching increases average truck fill rate across the fleet

> Run `python route_solver.py` after generating data to see your exact numbers in the terminal and dashboard.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip
- Git
- VS Code (recommended)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/PranjalAmbwani1305/Route-Optimization-Engine-Agent-.git
cd Route-Optimization-Engine-Agent-
```

### Step 2 — Open in VS Code

```bash
code .
```

### Step 3 — Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

You'll know it's active when you see `(venv)` at the start of your terminal prompt.

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The RAG assistant requires an OpenAI API key. All other dashboard pages work without it.

### Step 6 — Run the Full Pipeline

```bash
# Generate India-realistic synthetic data
python generate_data.py

# Run the route optimization engine
python route_solver.py

# Launch the dashboard
streamlit run dashboard.py
```

The dashboard opens automatically at **http://localhost:8501**

To stop the server: `Ctrl + C`

---

## 📁 Project Structure

```
Route-Optimization-Engine-Agent-/
│
├── dashboard.py            # Streamlit multi-page dashboard (9 pages)
├── generate_data.py        # India-realistic data generator
├── route_solver.py         # VRPTW route optimization engine
├── rag_engine.py           # RAG agent — LangChain + vector search
│
├── shipments.csv           # Generated: 200 India shipments
├── vehicles.csv            # Generated: 35 trucks, 7 categories
├── lanes.csv               # Generated: NH corridors + tolls
├── demand_forecast.csv     # Generated: 90-day stochastic demand
├── routes.csv              # Generated: optimized route assignments
├── metrics.csv             # Generated: before vs. after metrics
├── vehicle_summary.csv     # Generated: per-vehicle utilization
│
├── logo.png                # LoRRI brand logo
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔮 Future Roadmap

**Phase 1 — Live Data Integration**
Replace `generate_data.py` with a LoRRI API connector pulling real shipment orders, carrier availability, and freight rates. Everything downstream — optimizer, RAG agent, dashboard — works unchanged.

**Phase 2 — Satellite & Weather Awareness**
Integrate weather APIs for monsoon/disruption-aware routing. Satellite data for road condition monitoring on state highways. Proactive rerouting before disruptions occur — not just reactive.

**Phase 3 — Continuous Learning**
Feedback loop: actual vs. predicted delivery times feeds model retraining. Driver behavior patterns incorporated into delay modeling. Lane-level performance history improves stochastic estimates over time.

**Phase 4 — Fleet Intelligence**
Real-time GPS integration for live vehicle tracking. Automatic re-optimization triggered by geofence events. Driver app integration for turn-by-turn optimized navigation.

---

## 👥 Team

Built for the **LogisticsNow Hackathon** · Problem Statement #4 · AI Route Optimization Engine

| Name | Role |
|---|---|
| Pranjal Ambwani | Backend / Optimization Engine / RAG |
| Swadha Singh | Data Pipeline / Dashboard |

---

<div align="center">

**Built with ❤️ for India's logistics ecosystem**

*LoRRI — Logistics Rating & Intelligence · India's Premier Freight Intelligence Platform*

</div>
