# rag_engine.py
import re

# Knowledge base extracted directly from Synapflow_Problem4.pdf
KNOWLEDGE_BASE = [
    "The AI Route Optimization Engine transforms LoRRI from a retrospective diagnostic tool into a proactive, real-time decision intelligence ecosystem.",
    "It optimizes multi-stop delivery routes using a Capacitated Vehicle Routing (CVRP) framework.",
    "Rather than minimizing distance alone, it balances delivery time, transportation cost (including toll charges), SLA adherence, and carbon impact through a weighted objective scoring model.",
    "It dynamically re-optimizes routes when traffic disruptions or shipment priorities change.",
    "The system models routing using a heuristic OR-Tools solver with local search refinements to generate scalable multi-stop routes.",
    "Based on industry benchmarks, optimization yields an 8-20% reduction in travel distance, 5-15% cost savings, and measurable improvements in SLA adherence and fleet utilization.",
    "The Comprehensive Architecture includes a User Layer (LoRRI Dashboard), Application Layer (Route Optimization API, Trigger Service), Optimization Core (CVRP Model), and Intelligence & Output Layer."
]

def get_rag_response(query: str) -> str:
    """
    Lightweight keyword-based Retrieval-Augmented Generation (RAG) proxy.
    """
    query_words = re.findall(r'\w+', query.lower())
    
    # Score each chunk based on word overlap
    scores = []
    for chunk in KNOWLEDGE_BASE:
        score = sum(1 for word in query_words if word in chunk.lower())
        scores.append(score)
        
    best_idx = scores.index(max(scores))
    
    if max(scores) > 0:
        return KNOWLEDGE_BASE[best_idx]
    
    return "I am your LoRRI AI Assistant. I can answer questions about the CVRP architecture, multi-objective scoring, dynamic re-optimization, and expected business impact."
