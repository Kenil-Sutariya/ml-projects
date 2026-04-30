# Real-Time Research Copilot

A **local-first AI research assistant** that takes a user question, searches
multiple sources, synthesizes an answer using a local Ollama LLM, and returns
source references with a confidence score — all without sending data to any
cloud AI service.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────┐
│          Streamlit Frontend                 │
│        frontend/streamlit_app.py            │
└────────────────────┬────────────────────────┘
                     │ HTTP POST /research/
                     ▼
┌─────────────────────────────────────────────┐
│           FastAPI Backend                   │
│           app/main.py                       │
│           app/routers/research.py           │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           ResearchAgent                     │
│         app/agents/research_agent.py        │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Wikipedia │  │ Tavily   │  │PrivateKB │  │
│  │  Tool    │  │  Tool    │  │  Tool    │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────┬────────────────────────┘
                     │  sources collected
                     ▼
┌─────────────────────────────────────────────┐
│           OllamaService                     │
│        app/services/ollama_service.py       │
│                                             │
│   Ollama local LLM (llama3.2 / gemma / …)  │
└────────────────────┬────────────────────────┘
                     │  structured answer
                     ▼
┌─────────────────────────────────────────────┐
│         ConfidenceService                   │
│      app/services/confidence_service.py     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
              ResearchResponse
    (answer + key points + sources + score)
```

---

## Features

- **Multi-source research** — Wikipedia (live), optional Tavily web search, private local docs
- **Local LLM inference** — Ollama only; works with llama3.2, gemma2, phi3, mistral, and more
- **No OpenAI, no cloud AI** — all inference runs on your machine
- **Confidence scoring** — 0–1 score based on source count, type diversity, and content richness
- **Private knowledge base** — drop `.txt` files into `app/data/private_docs/` and they get searched
- **Modular tool system** — add a new data source by creating one new class; the agent needs no changes
- **Clean structured output** — separate answer, key points, sources, and confidence badge
- **Streamlit frontend** — interactive UI with expandable source cards
- **Full offline test suite** — 31 tests, no real Ollama or internet connection required

---

## Tech Stack

| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| Frontend     | Streamlit                                    |
| Backend      | FastAPI + Uvicorn                            |
| LLM          | Ollama (`llama3.2`, `gemma2`, `phi3`, …)    |
| Embeddings   | Ollama `nomic-embed-text` *(Milestone 7)*    |
| Vector DB    | FAISS (local) *(Milestone 7)*               |
| Web search   | Tavily API (optional)                        |
| Wikipedia    | Wikipedia REST API (direct, no broken pkg)   |
| Validation   | Pydantic v2                                  |
| Config       | pydantic-settings + python-dotenv            |
| Tests        | pytest + httpx                               |

---

## Setup

### 1. Install Ollama

Download from [ollama.com](https://ollama.com) and install for your OS.

```bash
# Pull at least one chat model (pick what fits your RAM)
ollama pull llama3.2          # 2 GB — good balance of speed and quality
ollama pull gemma2:2b         # 1.6 GB — fastest
ollama pull phi3              # 2.2 GB — good for reasoning

# Pull the embedding model (needed for Milestone 7 vector search)
ollama pull nomic-embed-text

# Confirm what you have installed
ollama list
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set `OLLAMA_CHAT_MODEL` to one of your installed models.
Run `ollama list` to see what you have. All other values work out of the box.

```env
OLLAMA_CHAT_MODEL=llama3.2          # change to gemma2:2b, phi3, etc.
TAVILY_API_KEY=                     # optional — leave blank to skip web search
```

---

## Environment Variables

| Variable                 | Required | Default                   | Description                                           |
|--------------------------|----------|---------------------------|-------------------------------------------------------|
| `OLLAMA_BASE_URL`        | No       | `http://localhost:11434`  | Where the Ollama server is running                    |
| `OLLAMA_CHAT_MODEL`      | **Yes**  | `llama3.2`                | Model used for answer synthesis (`ollama list`)       |
| `OLLAMA_EMBEDDING_MODEL` | No       | `nomic-embed-text`        | Embedding model for vector search (Milestone 7)       |
| `TAVILY_API_KEY`         | No       | *(empty)*                 | Enables live web search — free key at tavily.com      |
| `VECTORSTORE_PATH`       | No       | `app/data/vectorstore`    | Where the FAISS index is persisted on disk            |
| `API_BASE_URL`           | No       | `http://localhost:8000`   | Backend URL used by the Streamlit frontend            |

---

## Running the Application

### Terminal 1 — FastAPI backend

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

- API root: http://localhost:8000
- Swagger UI (interactive docs): http://localhost:8000/docs
- Health check: http://localhost:8000/health → `{"status": "ok"}`

> **Port conflict?** If you see `[Errno 48] Address already in use`, run:
> ```bash
> kill -9 $(lsof -ti :8000)
> ```

### Terminal 2 — Streamlit frontend

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

- Frontend: http://localhost:8501

---

## Running Tests

```bash
pytest tests/ -v
```

All 31 tests run fully offline. No Ollama server, no Tavily key, and no internet
connection are needed — tools and LLM calls are mocked.

```
31 passed in ~46s
```

---

## Example Queries

| Query | Recommended sources |
|---|---|
| *What is quantum entanglement?* | Wikipedia |
| *Who are the best performing LLMs right now?* | Wikipedia + Web (Tavily) |
| *What is our company's AI usage policy?* | Private KB |
| *Explain the transformer architecture* | Wikipedia + Web |

**Steps:**
1. Open http://localhost:8501
2. Type your question in the text area
3. Choose sources in the sidebar (Wikipedia is on by default)
4. Click **Run Research**
5. View the answer, key points, confidence score, and source cards

---

## Project Structure

```
research_copilot/
├── app/
│   ├── main.py                    ← FastAPI app, CORS middleware, routers
│   ├── core/
│   │   ├── config.py              ← Pydantic-settings; loads .env
│   │   └── prompts.py             ← System prompt + user prompt builder
│   ├── models/
│   │   └── schemas.py             ← ResearchRequest / SourceResult / ResearchResponse
│   ├── routers/
│   │   └── research.py            ← POST /research endpoint
│   ├── agents/
│   │   └── research_agent.py      ← Orchestrates tools → LLM → confidence score
│   ├── tools/
│   │   ├── base_tool.py           ← Abstract BaseResearchTool (interface)
│   │   ├── wikipedia_tool.py      ← Wikipedia REST API search ✅
│   │   ├── tavily_tool.py         ← Tavily live web search ✅
│   │   └── private_kb_tool.py     ← Local .txt file search (stub → Milestone 3)
│   ├── services/
│   │   ├── ollama_service.py      ← ollama.chat() + response parser ✅
│   │   ├── confidence_service.py  ← Source-count + diversity scoring ✅
│   │   ├── embedding_service.py   ← Ollama embeddings (stub → Milestone 7)
│   │   └── vector_store.py        ← FAISS index (stub → Milestone 7)
│   └── data/
│       └── private_docs/          ← Drop your .txt files here
│           └── sample_company_policy.txt
├── frontend/
│   └── streamlit_app.py           ← Full Streamlit UI ✅
├── tests/
│   ├── test_research_api.py       ← API endpoint tests (13 tests)
│   ├── test_confidence_service.py ← Confidence scoring tests (6 tests)
│   └── test_tools.py              ← Tool interface + mock tests (12 tests)
├── .env.example                   ← Copy to .env and configure
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Development Milestones

| # | What was built | Status |
|---|---|---|
| 1 | Project structure, FastAPI scaffold, Pydantic schemas, health endpoint | ✅ Done |
| 2 | WikipediaTool — live Wikipedia REST API search | ✅ Done |
| 3 | PrivateKBTool — keyword search over local `.txt` files | 🔲 Next |
| 4 | OllamaService — real local LLM inference via `ollama.chat()` | ✅ Done |
| 5 | ConfidenceService — source count + diversity + richness scoring | ✅ Done |
| 6 | Streamlit frontend — clean UI with answer, key points, source cards | ✅ Done |
| 7 | FAISS vector search for private knowledge base | 🔲 |
| 8 | TavilyTool — optional live web search via Tavily API | ✅ Done |
| 9 | Complete test suite polish | 🔲 |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `[Errno 48] Address already in use` | `kill -9 $(lsof -ti :8000)` then restart |
| `Model 'llama3.2' not found` | `ollama pull llama3.2` |
| Ollama connection refused | Open the Ollama app or run `ollama serve` |
| Tavily returns no results | Check your `TAVILY_API_KEY` in `.env` |
| Wikipedia returns empty | Check your internet connection |
| Very slow first response | Normal — Ollama loads the model into memory on the first call |

---

## Future Improvements

- Milestone 3 — PrivateKBTool with keyword matching
- Milestone 7 — FAISS semantic vector search for private KB
- ArxivTool — search scientific papers
- PDF ingestion for private knowledge base
- Streaming responses (Server-Sent Events) so answers appear word-by-word
- Conversation history / multi-turn research sessions
- Result caching to avoid re-running identical queries
- Docker Compose for one-command local startup
