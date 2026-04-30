"""
streamlit_app.py — Real-Time Research Copilot frontend

Run with:  streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Real-Time Research Copilot",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Real-Time Research Copilot")
st.caption("Local-first AI research assistant powered by Ollama — no cloud required.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Research Options")
    include_web = st.checkbox(
        "🌐 Web Search (Tavily)",
        value=False,
        help="Requires TAVILY_API_KEY in your .env file.",
    )
    include_wikipedia = st.checkbox("📖 Wikipedia", value=True)
    include_private_kb = st.checkbox("📁 Private Knowledge Base", value=True)
    max_results = st.slider("Max results per source", min_value=1, max_value=10, value=5)

    st.divider()
    st.caption(f"Backend: `{API_BASE_URL}`")

    if st.button("Check API connection"):
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
            st.success("✅ API is running") if resp.status_code == 200 else st.error(f"API returned {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach API. Is `uvicorn app.main:app --reload` running?")

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "Your research question",
    placeholder="e.g. What is quantum entanglement and how is it used in computing?",
    height=100,
)

run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.warning("Please enter a research question first.")
        st.stop()

    payload = {
        "query": query.strip(),
        "include_web": include_web,
        "include_wikipedia": include_wikipedia,
        "include_private_kb": include_private_kb,
        "max_results": max_results,
    }

    with st.spinner("Researching… local Ollama models can take 15–30 seconds on first call."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/research/",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the backend.\n\nRun: `uvicorn app.main:app --reload`")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ API error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()

    confidence = data.get("confidence_score", 0.0)
    tools_used = data.get("tools_used", [])
    sources = data.get("sources", [])
    key_points = data.get("key_points", [])
    answer = data.get("answer", "")

    # ── Top row: answer header + confidence badge ─────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("📝 Answer")
    with col2:
        if confidence >= 0.7:
            st.success(f"Confidence: {confidence:.0%}")
        elif confidence >= 0.4:
            st.warning(f"Confidence: {confidence:.0%}")
        else:
            st.error(f"Confidence: {confidence:.0%}")

    # ── Answer prose ──────────────────────────────────────────────────────
    # `answer` is now plain prose — no markdown section headers — so it
    # renders cleanly without duplicating the Key Points section.
    st.markdown(answer if answer else "_No answer returned._")

    # ── Key points ────────────────────────────────────────────────────────
    if key_points:
        st.subheader("🔑 Key Points")
        for point in key_points:
            st.markdown(f"- {point}")

    # ── Meta row: tools + source count ────────────────────────────────────
    if tools_used or sources:
        meta_parts = []
        if tools_used:
            meta_parts.append(f"Sources searched: **{', '.join(tools_used)}**")
        if sources:
            meta_parts.append(f"**{len(sources)}** snippet(s) retrieved")
        st.caption("  ·  ".join(meta_parts))

    # ── Sources (expandable) ──────────────────────────────────────────────
    if sources:
        st.subheader(f"📚 Sources ({len(sources)})")
        for i, src in enumerate(sources, start=1):
            label = f"{i}. [{src['source_type'].upper()}] {src['title']}"
            with st.expander(label):
                st.write(src["content"])
                if src.get("url"):
                    st.markdown(f"[Open source ↗]({src['url']})")
                if src.get("score") is not None:
                    st.caption(f"Relevance score: {src['score']:.2f}")
    else:
        st.info("No sources were retrieved for this query.")
