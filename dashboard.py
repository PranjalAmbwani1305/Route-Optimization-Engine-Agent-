import streamlit as st
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LoRRI · AI Intelligence", layout="wide")

def apply_advanced_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@400;500;600&family=DM+Mono&display=swap');
    
    /* Base */
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background: #050810 !important; color: #c8d6ee !important; }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] { background: #070912 !important; border-right: 1px solid rgba(255,255,255,.05) !important; width: 280px !important; }
    
    /* Main Layout */
    .topbar { display: flex; align-items: center; justify-content: space-between; padding: 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,.05); margin-bottom: 2rem; }
    .topbar-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #f0f6ff; }
    
    /* Chat UI */
    .chat-container { background: #080c18; border: 1px solid rgba(255,255,255,.06); border-radius: 12px; height: 500px; overflow-y: auto; padding: 20px; margin-bottom: 20px; }
    .msg-bubble { padding: 12px 16px; border-radius: 12px; margin-bottom: 15px; font-size: 0.9rem; line-height: 1.5; max-width: 85%; }
    .msg-user { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); margin-left: auto; color: #f0f6ff; }
    .msg-ai { background: #0d111d; border: 1px solid rgba(255,255,255,0.05); color: #c8d6ee; }
    .source-chip { display: inline-block; font-family: 'DM Mono'; font-size: 0.65rem; background: rgba(59,130,246,0.1); color: #3b82f6; padding: 2px 8px; border-radius: 4px; margin-top: 8px; }
    
    /* Knowledge Base Panel */
    .kb-panel { background: #080c18; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 20px; }
    .kb-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
    .kb-name { font-family: 'DM Mono'; font-size: 0.75rem; color: #7a9cbf; }
    .kb-status { color: #3fb950; font-size: 0.7rem; font-weight: 600; }
    
    /* Input Area */
    .stChatInputContainer { background: #0d111d !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGIC (Mock RAG)
# ─────────────────────────────────────────────────────────────────────────────
def get_mock_response(query):
    query = query.lower()
    if "carbon" in query or "co2" in query:
        return "Based on **vehicle_summary.csv**, Vehicle 3 is your highest emitter at **1,126.5 kg CO2**. Switching its route to the optimized baseline would reduce this by 22%.", ["vehicle_summary.csv", "metrics.csv"]
    elif "cost" in query or "savings" in query:
        return "Total savings for this run are **₹601,807**. The largest contributor to savings was the reduction in total distance by **35,966 km**.", ["metrics.csv", "shipments.csv"]
    elif "sla" in query or "late" in query:
        return "Current SLA adherence is **90%**. There were 5 breaches recorded in the Mumbai Hub run, primarily due to traffic spikes in Agra.", ["routes.csv", "metrics.csv"]
    else:
        return "I've analyzed the current shipment data. Total distance is optimized at 17,560 km across 5 active vehicles.", ["shipments.csv"]

# ─────────────────────────────────────────────────────────────────────────────
# 3. UI RENDERING
# ─────────────────────────────────────────────────────────────────────────────
apply_advanced_styles()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<h1 style="font-family:Syne; color:#f0f6ff; font-size:1.8rem;">Lo<em>RRI</em></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#5a7a9a; font-family:DM Mono; font-size:0.65rem; margin-top:-15px;">v2.1 · Enterprise Edition</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Navigation Simulation
    st.markdown("### Navigation")
    st.markdown("""
    <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="color:#5a7a9a; padding:10px; border-radius:8px; cursor:pointer;">🏠 Dashboard</div>
        <div style="color:#5a7a9a; padding:10px; border-radius:8px; cursor:pointer;">🗺️ Route Map</div>
        <div style="color:#5a7a9a; padding:10px; border-radius:8px; cursor:pointer;">🧠 AI Explainability</div>
        <div style="background:rgba(59,130,246,0.12); color:#3b82f6; padding:10px; border-radius:8px; border-left:3px solid #3b82f6;">🤖 AI Assistant (RAG)</div>
    </div>
    """, unsafe_allow_html=True)

# --- TOPBAR ---
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-title">AI Assistant (RAG)</div>
    <div style="font-family:'DM Mono'; font-size:0.65rem; color:#3a5070;">Connected to: Local Intelligence Engine · No External Keys Required</div>
  </div>
  <div style="text-align:right;">
    <span style="color:#3fb950; font-family:'DM Mono'; font-size:0.75rem;">● LIVE RUN ACTIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT (Two Columns) ---
col_chat, col_kb = st.columns([2.5, 1])

with col_chat:
    st.markdown('<div class="grp-label" style="font-family:DM Mono; font-size:0.65rem; color:#3b82f6; text-transform:uppercase; margin-bottom:10px;">Grounded Intelligence Chat</div>', unsafe_allow_html=True)
    
    # Chat History Container
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": "Hello! I'm LoRRI AI. I have indexed your shipment and route data. How can I help you optimize your logistics today?", "sources": []}]
    
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        cls = "msg-user" if msg["role"] == "user" else "msg-ai"
        chat_html += f'<div class="msg-bubble {cls}">{msg["content"]}'
        if msg["sources"]:
            chips = "".join([f'<span class="source-chip">📄 {s}</span> ' for s in msg["sources"]])
            chat_html += f'<br>{chips}'
        chat_html += '</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input Area
    if prompt := st.chat_input("Ask about costs, carbon, or vehicle performance..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        
        # Thinking Animation Simulation
        with st.spinner("Analyzing data tables..."):
            time.sleep(1.2)
            response, sources = get_mock_response(prompt)
            st.session_state.messages.append({"role": "ai", "content": response, "sources": sources})
        st.rerun()

with col_kb:
    st.markdown('<div class="grp-label" style="font-family:DM Mono; font-size:0.65rem; color:#3b82f6; text-transform:uppercase; margin-bottom:10px;">Knowledge Base Status</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"""
        <div class="kb-panel">
            <div style="font-size:0.8rem; font-weight:600; margin-bottom:15px; color:#f0f6ff;">Indexed Data Sources</div>
            <div class="kb-item"><span class="kb-name">shipments.csv</span><span class="kb-status">SYNCED</span></div>
            <div class="kb-item"><span class="kb-name">routes.csv</span><span class="kb-status">SYNCED</span></div>
            <div class="kb-item"><span class="kb-name">vehicle_summary.csv</span><span class="kb-status">SYNCED</span></div>
            <div class="kb-item"><span class="kb-name">metrics.csv</span><span class="kb-status">SYNCED</span></div>
            <br>
            <div style="font-size:0.65rem; color:#5a7a9a; line-height:1.4;">
                <b>Total context:</b> 142kb<br>
                <b>Embedded vectors:</b> 1,024<br>
                <b>Model:</b> LoRRI Local v2
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="grp-label" style="font-family:DM Mono; font-size:0.65rem; color:#3b82f6; text-transform:uppercase; margin-bottom:10px;">Suggested Queries</div>', unsafe_allow_html=True)
    if st.button("💰 Total cost savings?", use_container_width=True): pass
    if st.button("🌿 Highest carbon emitter?", use_container_width=True): pass
    if st.button("⏰ SLA performance details?", use_container_width=True): pass

# Footer
st.markdown("""<hr><div style="text-align:center; font-family:'DM Mono'; font-size:0.7rem; color:#1a2d3f; margin-top:2rem;">
© 2026 LoRRI Technologies · Local Data Grounding · Enterprise v2.1</div>""", unsafe_allow_html=True)
