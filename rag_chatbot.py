"""
rag_chatbot.py  ·  LoRRI RAG Chatbot
Stateful AI assistant backed by live optimization data.
Called from dashboard.py via: from rag_chatbot import render_chatbot
"""

import streamlit as st
import requests

CLAUDE_URL   = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS   = 700

SUGGESTED = [
    "Which vehicle traveled the longest route?",
    "Which cities have SLA breaches?",
    "How much carbon did we save?",
    "Which city has the highest traffic impact?",
    "Break down fuel vs toll vs driver costs.",
    "What are the 3 hardest deliveries?",
]


def _call(system: str, messages: list) -> str:
    try:
        r = requests.post(CLAUDE_URL,
            headers={"Content-Type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": MAX_TOKENS,
                  "system": system, "messages": messages},
            timeout=40)
        d = r.json()
        if "content" in d and d["content"]:
            return d["content"][0]["text"]
        if "error" in d:
            return f"⚠️ API error: {d['error'].get('message','Unknown')}"
        return "⚠️ Unexpected response. Please retry."
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out — please retry."
    except Exception as e:
        return f"⚠️ Error: {e}"


def render_chatbot(rag_context: str):
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []
    if "chat_pending" not in st.session_state:
        st.session_state.chat_pending = False

    # Quick question buttons
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.58rem;
        font-weight:500;letter-spacing:.18em;text-transform:uppercase;
        color:#334155;margin-bottom:8px;">QUICK QUERIES</div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 3].button(q, key=f"q_{i}", use_container_width=True):
            st.session_state.chat_msgs.append({"role": "user", "content": q})
            st.session_state.chat_pending = True
            st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Resolve pending AI turn
    if st.session_state.chat_pending and st.session_state.chat_msgs:
        if st.session_state.chat_msgs[-1]["role"] == "user":
            with st.spinner("LoRRI Copilot analysing data…"):
                reply = _call(rag_context,
                              [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.chat_msgs])
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.session_state.chat_pending = False
            st.rerun()

    # Message history
    if not st.session_state.chat_msgs:
        st.markdown("""
        <div style="text-align:center;padding:36px 20px;
            border:1px dashed rgba(255,255,255,.07);border-radius:12px;margin-bottom:14px;">
            <div style="font-size:1.8rem;margin-bottom:8px;">🤖</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:.95rem;
                font-weight:700;color:#94A3B8;margin-bottom:4px;">LoRRI Copilot Ready</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:.67rem;color:#475569;">
                Full access to routes, costs, vehicles, carbon &amp; SLA data.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_msgs:
            with st.chat_message("user" if msg["role"] == "user" else "assistant",
                                 avatar=None if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # Input row
    ci, cc = st.columns([7, 1])
    with ci:
        user_input = st.chat_input("Ask about routes, costs, carbon, SLA, vehicles…")
    with cc:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="chat_clear", use_container_width=True):
            st.session_state.chat_msgs = []
            st.session_state.chat_pending = False
            st.rerun()

    if user_input:
        st.session_state.chat_msgs.append({"role": "user", "content": user_input})
        st.session_state.chat_pending = True
        st.rerun()
