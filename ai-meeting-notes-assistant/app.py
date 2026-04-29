from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meeting_notes.evaluation import evaluate_notes
from meeting_notes.exporter import notes_to_markdown, notes_to_pdf
from meeting_notes.llm import generate_meeting_notes
from meeting_notes.models import MeetingNotes

load_dotenv()

SAMPLE_PATH = ROOT / "data" / "sample_transcript.txt"


st.set_page_config(
    page_title="AI Meeting Notes Assistant",
    layout="wide",
)

st.title("AI Meeting Notes Assistant")
st.caption("Turn transcripts into summaries, decisions, owners, deadlines, JSON, Markdown, and PDF.")

with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox(
        "Provider",
        ["Demo heuristic", "OpenAI", "Ollama"],
        help="Demo heuristic runs offline. OpenAI and Ollama call real LLMs.",
    )

    if provider == "OpenAI":
        model_name = st.text_input("OpenAI model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        st.info("Set OPENAI_API_KEY in your .env file or environment.")
    elif provider == "Ollama":
        model_name = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.1"))
        st.text_input("Ollama base URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), key="ollama_base_url")
    else:
        model_name = "demo-heuristic"

    st.divider()
    st.write("Quality target")
    st.caption("Useful meeting notes should have clear owners and deadlines for most action items.")


uploaded_file = st.file_uploader("Upload a TXT meeting transcript", type=["txt"])
sample_text = SAMPLE_PATH.read_text(encoding="utf-8")

if uploaded_file:
    transcript = uploaded_file.read().decode("utf-8", errors="replace")
else:
    transcript = st.text_area("Transcript", sample_text, height=280)

col1, col2 = st.columns([1, 4])
with col1:
    generate = st.button("Generate Notes", type="primary", use_container_width=True)
with col2:
    st.caption(f"Characters: {len(transcript):,}")

if generate:
    if not transcript.strip():
        st.error("Please upload or paste a transcript first.")
    else:
        with st.spinner("Generating structured meeting notes..."):
            try:
                notes = generate_meeting_notes(
                    transcript=transcript,
                    provider=provider,
                    model_name=model_name,
                    ollama_base_url=st.session_state.get("ollama_base_url"),
                )
                st.session_state["notes"] = notes
                st.session_state["transcript"] = transcript
            except Exception as exc:
                st.error(f"Could not generate notes: {exc}")

notes: MeetingNotes | None = st.session_state.get("notes")

if notes:
    markdown = notes_to_markdown(notes)
    json_text = json.dumps(notes.model_dump(), indent=2, ensure_ascii=False)
    evaluation = evaluate_notes(notes)

    tab_summary, tab_json, tab_eval, tab_export = st.tabs(["Notes", "JSON", "Evaluation", "Export"])

    with tab_summary:
        st.subheader("Summary")
        st.write(notes.summary)

        st.subheader("Decisions")
        if notes.decisions:
            for decision in notes.decisions:
                st.markdown(f"- {decision}")
        else:
            st.caption("No decisions detected.")

        st.subheader("Action Items")
        if notes.action_items:
            st.dataframe(
                [item.model_dump() for item in notes.action_items],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No action items detected.")

        if notes.follow_ups:
            st.subheader("Follow-ups")
            for follow_up in notes.follow_ups:
                st.markdown(f"- {follow_up}")

    with tab_json:
        st.code(json_text, language="json")

    with tab_eval:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Action items", evaluation.action_item_count)
        metric_cols[1].metric("Owner coverage", f"{evaluation.owner_coverage:.0%}")
        metric_cols[2].metric("Deadline coverage", f"{evaluation.deadline_coverage:.0%}")
        metric_cols[3].metric("Completeness", f"{evaluation.completeness_score:.0%}")

        if evaluation.warnings:
            st.warning("\n".join(evaluation.warnings))
        else:
            st.success("The notes have owners and deadlines for every action item.")

    with tab_export:
        pdf_bytes = notes_to_pdf(notes)
        st.download_button(
            "Download Markdown",
            data=markdown,
            file_name="meeting_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "Download JSON",
            data=json_text,
            file_name="meeting_notes.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="meeting_notes.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
