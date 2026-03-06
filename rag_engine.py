"""
rag_engine.py
─────────────────────────────────────────────────────────────────────────────
Standalone RAG engine for LoRRI Route Optimization Assistant.
Reads all four CSVs, builds a TF-IDF knowledge base, retrieves top-k chunks
per query, and calls HuggingFace Llama-3.1-8B for the final answer.

Usage:
    from rag_engine import get_rag_response, set_hf_key

    set_hf_key("hf_...")
    answer, sources = get_rag_response("Which vehicle has the most SLA breaches?")
"""

import re
import math
import pandas as pd
import numpy as np
import streamlit as st

# ── HuggingFace key (set at runtime from the UI) ──────────────────────────────
_HF_KEY: str = ""

def set_hf_key(key: str) -> None:
    global _HF_KEY
    _HF_KEY = key.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-base builder  (cached so it only runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _build_kb():
    """
    Load the four CSVs and chunk them into text documents.
    Returns (docs, tfidf_vecs, idf_dict).
    """
    try:
        ships   = pd.read_csv("shipments.csv")
        routes  = pd.read_csv("routes.csv")
        metrics = pd.read_csv("metrics.csv").iloc[0]
        veh     = pd.read_csv("vehicle_summary.csv")
    except FileNotFoundError:
        return [], [], {}

    VEHICLE_CAP = 800
    docs: list[dict] = []

    # ── 1. Global summary ─────────────────────────────────────────────────────
    docs.append({"title": "Global Optimization Metrics", "text": (
        f"LoRRI run: {int(metrics['num_shipments'])} shipments, "
        f"{int(metrics['num_vehicles'])} vehicles, depot Mumbai. "
        f"Objective weights: Cost 35%, Time 30%, Carbon 20%, SLA 15%. "
        f"Baseline distance {metrics['baseline_distance_km']:.1f} km → "
        f"optimized {metrics['opt_distance_km']:.1f} km "
        f"(saved {metrics['baseline_distance_km']-metrics['opt_distance_km']:.1f} km, "
        f"{(metrics['baseline_distance_km']-metrics['opt_distance_km'])/metrics['baseline_distance_km']*100:.1f}%). "
        f"Baseline cost ₹{metrics['baseline_total_cost']:,.0f} → "
        f"optimized ₹{metrics['opt_total_cost']:,.0f} "
        f"(saved ₹{metrics['baseline_total_cost']-metrics['opt_total_cost']:,.0f}, "
        f"{(metrics['baseline_total_cost']-metrics['opt_total_cost'])/metrics['baseline_total_cost']*100:.1f}%). "
        f"Carbon: baseline {metrics['baseline_carbon_kg']:.1f} kg → "
        f"optimized {metrics['opt_carbon_kg']:.1f} kg "
        f"(reduced {metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:.1f} kg). "
        f"SLA adherence: {metrics['baseline_sla_adherence_pct']:.1f}% → {metrics['opt_sla_adherence_pct']:.1f}%."
    )})

    # ── 2. Cost breakdown ─────────────────────────────────────────────────────
    docs.append({"title": "Cost Breakdown — Fuel, Toll, Driver", "text": (
        f"Fuel: baseline ₹{metrics['baseline_fuel_cost']:,.0f}, optimized ₹{metrics['opt_fuel_cost']:,.0f}, "
        f"saved ₹{metrics['baseline_fuel_cost']-metrics['opt_fuel_cost']:,.0f} at ₹12/km. "
        f"Toll: baseline ₹{metrics['baseline_toll_cost']:,.0f}, optimized ₹{metrics['opt_toll_cost']:,.0f}, "
        f"saved ₹{metrics['baseline_toll_cost']-metrics['opt_toll_cost']:,.0f}. "
        f"Driver: baseline ₹{metrics['baseline_driver_cost']:,.0f}, optimized ₹{metrics['opt_driver_cost']:,.0f}, "
        f"saved ₹{metrics['baseline_driver_cost']-metrics['opt_driver_cost']:,.0f} at ₹180/hr. "
        f"SLA penalty rate: ₹500/breach-hour."
    )})

    # ── 3. Per-vehicle summaries ──────────────────────────────────────────────
    for _, r in veh.iterrows():
        docs.append({"title": f"Vehicle {int(r['vehicle'])} Summary", "text": (
            f"Vehicle {int(r['vehicle'])}: {int(r['stops'])} stops, "
            f"load {r['load_kg']:.1f} kg ({r['utilization_pct']:.1f}% of {VEHICLE_CAP} kg capacity), "
            f"distance {r['distance_km']:.1f} km, time {r['time_hr']:.1f} hr, "
            f"fuel ₹{r['fuel_cost']:,.0f}, toll ₹{r['toll_cost']:,.0f}, "
            f"driver ₹{r['driver_cost']:,.0f}, SLA penalty ₹{r['sla_penalty']:,.0f}, "
            f"total ₹{r['total_cost']:,.0f}, carbon {r['carbon_kg']:.1f} kg CO2, "
            f"SLA breaches {int(r['sla_breaches'])}."
        )})

    # ── 4. Optimized route per vehicle ────────────────────────────────────────
    for v in routes["vehicle"].unique():
        vdf = routes[routes["vehicle"] == v].sort_values("stop_order")
        stops = []
        for _, r in vdf.iterrows():
            stops.append(
                f"stop {int(r['stop_order'])}: {r['city']} "
                f"(priority {r['priority']}, {r['weight']:.0f} kg, "
                f"travel {r['travel_time_hr']:.2f} hr, cost ₹{r['total_cost']:,.0f}, "
                f"carbon {r['carbon_kg']:.2f} kg, SLA breach {r['sla_breach_hr']:.1f} hr, "
                f"MO score {r['mo_score']:.4f})"
            )
        docs.append({"title": f"Vehicle {v} Optimized Route", "text":
                     f"Vehicle {v} route: " + " | ".join(stops)})

    # ── 5. Route stops in batches of 6 ────────────────────────────────────────
    for start in range(0, len(routes), 6):
        batch = routes.iloc[start:start+6]
        lines = [
            f"{r['city']} V{int(r['vehicle'])} stop#{int(r['stop_order'])} "
            f"prio={r['priority']} wt={r['weight']:.0f}kg "
            f"time={r['travel_time_hr']:.2f}hr cost=₹{r['total_cost']:.0f} "
            f"co2={r['carbon_kg']:.2f}kg breach={r['sla_breach_hr']:.1f}hr"
            for _, r in batch.iterrows()
        ]
        docs.append({"title": f"Route Stops {start+1}–{min(start+6, len(routes))}",
                     "text": " | ".join(lines)})

    # ── 6. HIGH-priority shipments ─────────────────────────────────────────────
    high = ships[ships["priority"] == "HIGH"]
    docs.append({"title": "HIGH Priority Shipments", "text":
        f"{len(high)} HIGH priority cities: " + ", ".join(high["city"].tolist()) +
        f". Avg weight {high['weight'].mean():.1f} kg. SLA window: 24 hours."
    })

    # ── 7. Carbon analysis ────────────────────────────────────────────────────
    top5 = routes.nlargest(5, "carbon_kg")[["city","carbon_kg","vehicle"]].values
    docs.append({"title": "Carbon Emissions Analysis", "text":
        f"Total optimized CO2: {metrics['opt_carbon_kg']:.1f} kg. "
        f"Reduction: {metrics['baseline_carbon_kg']-metrics['opt_carbon_kg']:.1f} kg "
        f"({(metrics['baseline_carbon_kg']-metrics['opt_carbon_kg'])/metrics['baseline_carbon_kg']*100:.1f}%). "
        f"Top 5 emitting stops: " +
        ", ".join(f"{c} ({k:.2f} kg, V{int(v)})" for c, k, v in top5) + "."
    })

    # ── 8. SLA analysis ───────────────────────────────────────────────────────
    breached = routes[routes["sla_breach_hr"] > 0]
    docs.append({"title": "SLA Breach Analysis", "text":
        f"{len(breached)} SLA breaches total. "
        f"Optimized adherence: {metrics['opt_sla_adherence_pct']:.1f}% "
        f"(baseline {metrics['baseline_sla_adherence_pct']:.1f}%). "
        f"Breached cities: " +
        (", ".join(breached["city"].tolist()) if len(breached) else "None") +
        (f". Avg breach: {breached['sla_breach_hr'].mean():.2f} hr." if len(breached) else "")
    })

    # ── 9. MO score / explainability ─────────────────────────────────────────
    top10 = routes.nlargest(10, "mo_score")[["city","mo_score","vehicle","priority"]].values
    docs.append({"title": "MO Score & Explainability", "text":
        "MO score = 0.30×(time/norm) + 0.35×(cost/norm) + 0.20×(carbon/norm) + 0.15×(SLA/norm). "
        "Lower = better routing choice. Re-opt triggers on >30% time increase or priority escalation. "
        "Top 10 hardest stops: " +
        ", ".join(f"{c} ({s:.4f}, V{int(v)}, {p})" for c, s, v, p in top10) + "."
    })

    # ── 10. System architecture doc ───────────────────────────────────────────
    docs.append({"title": "System Architecture & How CVRP Works", "text":
        "LoRRI uses a Capacitated Vehicle Routing Problem (CVRP) with nearest-neighbor heuristic. "
        "Each vehicle has 800 kg capacity. The solver departs from Mumbai depot, visits stops "
        "greedily by lowest MO score, and returns to depot. "
        "Re-optimization is threshold-based: triggered by traffic disruption (>30% time increase) "
        "or priority escalation (shipment upgraded to HIGH urgency). "
        "Features: TF-IDF RAG chatbot, permutation feature importance, SLA breach heatmap, "
        "carbon scatter analysis, cost waterfall chart, and live risk monitor."
    })

    # ── Build TF-IDF index ────────────────────────────────────────────────────
    N = len(docs)
    tokenized = [_tokenize(d["title"] + " " + d["text"]) for d in docs]

    df_counts: dict[str, int] = {}
    for toks in tokenized:
        for t in set(toks):
            df_counts[t] = df_counts.get(t, 0) + 1

    idf = {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df_counts.items()}

    def _tfidf(tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        return {t: (c / n) * idf.get(t, 1.0) for t, c in tf.items()}

    vecs = [_tfidf(toks) for toks in tokenized]
    return docs, vecs, idf


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def _cosine(a: dict, b: dict) -> float:
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    na  = math.sqrt(sum(x*x for x in a.values())) + 1e-9
    nb  = math.sqrt(sum(x*x for x in b.values())) + 1e-9
    return dot / (na * nb)


def _retrieve(query: str, docs, vecs, idf, top_k: int = 4) -> list[dict]:
    q_toks = _tokenize(query)
    q_tf: dict[str, int] = {}
    for t in q_toks:
        q_tf[t] = q_tf.get(t, 0) + 1
    n = len(q_toks) or 1
    q_vec = {t: (c / n) * idf.get(t, 1.0) for t, c in q_tf.items()}

    scored = [(c, i) for i, v in enumerate(vecs) if (c := _cosine(q_vec, v)) > 0]
    scored.sort(reverse=True)
    return [docs[i] for _, i in scored[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace LLM call
# ─────────────────────────────────────────────────────────────────────────────
def _hf_generate(messages: list[dict], system_prompt: str) -> str:
    if not _HF_KEY:
        return "⚠️ No HuggingFace API key set. Call `set_hf_key('hf_...')` first."
    try:
        from huggingface_hub import InferenceClient
        client   = InferenceClient(provider="nebius", api_key=_HF_KEY)
        payload  = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=payload,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ HuggingFace API error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_rag_response(
    query: str,
    history: list[dict] | None = None,
    top_k: int = 4,
) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks and generate an answer.

    Parameters
    ----------
    query   : user question string
    history : list of {"role": "user"|"assistant", "content": str}
    top_k   : number of chunks to retrieve

    Returns
    -------
    (answer_str, source_title_list)
    """
    docs, vecs, idf = _build_kb()
    if not docs:
        return ("⚠️ Knowledge base is empty — run generate_data.py and route_solver.py first.", [])

    retrieved   = _retrieve(query, docs, vecs, idf, top_k=top_k)
    context_str = "\n\n".join(f"[{d['title']}]\n{d['text']}" for d in retrieved)
    sources     = [d["title"] for d in retrieved]

    system_prompt = (
        "You are LoRRI's Route Intelligence Analyst — an expert in logistics, "
        "CVRP optimization, SLA management, and carbon footprint reduction. "
        "Answer using ONLY the provided context. Be concise and cite exact numbers. "
        "If the context doesn't contain enough information, say so clearly.\n\n"
        f"=== RETRIEVED CONTEXT ===\n{context_str}\n=== END CONTEXT ==="
    )

    msgs = list(history or []) + [{"role": "user", "content": query}]
    answer = _hf_generate(msgs[:-1] + [{"role": "user", "content": query}], system_prompt)
    return answer, sources
