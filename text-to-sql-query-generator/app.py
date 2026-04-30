from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sql_generator.database import (  # noqa: E402
    DB_PATH,
    get_schema_text,
    initialize_database,
    preview_tables,
    run_select_query,
)
from sql_generator.datasets import (  # noqa: E402
    column_profile,
    read_uploaded_dataset,
    summarize_dataset,
)
from sql_generator.llm import generate_sql  # noqa: E402
from sql_generator.safety import validate_select_query  # noqa: E402


EXAMPLE_QUESTIONS = [
    "Show total sales by month",
    "Which products have the highest sales?",
    "Show total revenue by customer",
    "Show sales by product category",
    "Show the latest orders",
    "Drop the customers table",
]


st.set_page_config(
    page_title="AI SQL Query Generator",
    page_icon="SQL",
    layout="wide",
)

initialize_database()

st.title("AI-Powered SQL Query Generator")
st.caption("Convert natural language questions into safe SQLite SELECT queries.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox(
        "Provider",
        ["Demo rules", "OpenAI", "Ollama"],
        help="Demo rules works without any API key. OpenAI and Ollama use the schema prompt.",
    )
    model = ""
    ollama_url = "http://localhost:11434/api/generate"
    if provider == "OpenAI":
        model = st.text_input("OpenAI model", value="gpt-4o-mini")
        st.info("Set OPENAI_API_KEY in your terminal before starting Streamlit.")
    elif provider == "Ollama":
        model = st.text_input("Ollama model", value="llama3.1")
        ollama_url = st.text_input("Ollama generate URL", value=ollama_url)

    st.header("Database")
    st.write(f"SQLite file: `{DB_PATH.name}`")
    if st.button("Reset sample database", width="stretch"):
        initialize_database()
        st.success("Sample database reset.")

schema_text = get_schema_text()

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Ask a question")
    selected_example = st.selectbox("Try an example", EXAMPLE_QUESTIONS)
    question = st.text_area(
        "Natural language request",
        value=selected_example,
        height=110,
        placeholder="Example: Show total sales by month",
    )

    generate_clicked = st.button("Generate and run SQL", type="primary")

    st.subheader("Schema sent to the model")
    st.code(schema_text, language="text")

with right:
    data_tab, upload_tab = st.tabs(["Sample database", "Upload dataset"])

    with data_tab:
        st.subheader("Sample data")
        for table_name, frame in preview_tables().items():
            with st.expander(table_name, expanded=table_name == "orders"):
                st.dataframe(frame, width="stretch", hide_index=True)

    with upload_tab:
        st.subheader("Dataset schemas")
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            uploaded_summaries = []
            parsed_files = []
            for uploaded_file in uploaded_files:
                try:
                    uploaded_frame = read_uploaded_dataset(
                        uploaded_file,
                        uploaded_file.name,
                    )
                except Exception as exc:
                    st.error(f"Could not read {uploaded_file.name}: {exc}")
                    continue

                summary = summarize_dataset(uploaded_frame)
                uploaded_summaries.append(
                    {
                        "file": uploaded_file.name,
                        "rows": summary.rows,
                        "columns": summary.columns,
                    }
                )
                parsed_files.append((uploaded_file.name, uploaded_frame))

            if uploaded_summaries:
                st.dataframe(uploaded_summaries, width="stretch", hide_index=True)

            for file_name, uploaded_frame in parsed_files:
                with st.expander(file_name, expanded=len(parsed_files) == 1):
                    schema_frame = column_profile(uploaded_frame)[["column", "type"]]
                    st.dataframe(
                        schema_frame,
                        width="stretch",
                        hide_index=True,
                    )

if generate_clicked:
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    result = generate_sql(
        question=question,
        schema_text=schema_text,
        provider=provider,
        model=model,
        ollama_url=ollama_url,
    )

    st.divider()
    st.subheader("Generated SQL")
    if result.notes:
        st.caption(f"{result.provider}: {result.notes}")

    if not result.sql:
        st.error("No SQL was generated.")
        st.stop()

    st.code(result.sql, language="sql")
    is_safe, cleaned_sql, message = validate_select_query(result.sql)

    if not is_safe:
        st.error(message)
        st.info("Unsafe SQL was blocked before it reached the database.")
        st.stop()

    st.success(message)

    try:
        output = run_select_query(cleaned_sql)
    except Exception as exc:  # pragma: no cover - displayed in the Streamlit app
        st.error(f"SQLite execution failed: {exc}")
        st.stop()

    st.subheader("Results")
    st.dataframe(output, width="stretch", hide_index=True)
    st.caption(f"{len(output)} row(s) returned.")
