# ⚡ LLM Cost Autopilot

> **Reduced LLM API costs by 96.6% while maintaining 96.4% quality parity** across 1,049+ requests — by routing each prompt to the cheapest model capable of handling it at acceptable quality.

An intelligent routing layer that sits in front of multiple LLM providers. It classifies incoming prompt complexity with a trained ML classifier, routes to the right model, and runs an async LLM-as-judge verification loop that feeds failures back into the classifier — a self-improving system that gets cheaper over time.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Cost reduction vs. GPT-4o | **96.6%** |
| Quality pass rate | **96.4%** |
| Classifier accuracy | **88.9%** |
| Escalation rate | **1.6%** |
| Load test: 60 req, 0 errors | **100% cost saved** vs. GPT-4o baseline |
| Load test median latency | **2,421 ms** |

**The core idea:** Not every prompt needs GPT-4o. "What is the capital of France?" is $0.00 on a local model. "Design a distributed caching system" legitimately needs the 70B. The router figures out which is which — automatically and continuously.

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [Phase 1 — Unified Model Interface](#phase-1--unified-model-interface)
  - [Phase 2 — Complexity Classifier](#phase-2--complexity-classifier)
  - [Phase 3 — Async Quality Verification Loop](#phase-3--async-quality-verification-loop)
  - [Phase 4 — Logging and Cost Dashboard](#phase-4--logging-and-cost-dashboard)
  - [Phase 5 — FastAPI Service](#phase-5--fastapi-service)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Option A — Local](#option-a--local-no-docker)
  - [Option B — Docker](#option-b--docker)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Classifier Deep Dive](#classifier-deep-dive)
- [Quality Verification Deep Dive](#quality-verification-deep-dive)
- [Model Registry](#model-registry)
- [Running the Load Test](#running-the-load-test)
- [Adding a New Provider](#adding-a-new-provider)
- [Reproducing the Results](#reproducing-the-results)
- [FAQ](#faq)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Client / Caller                            │
│                  (curl, Python SDK, your application)                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  POST /v1/completions
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI Router  (port 8000)                       │
│                                                                      │
│   ┌──────────────────────────┐      ┌────────────────────────────┐  │
│   │   Complexity Classifier  │─────▶│     Routing Decision       │  │
│   │                          │      │                            │  │
│   │  Input: 15 features from │      │  tier 1 → llama3.2 (local) │  │
│   │  the prompt text         │      │  tier 2 → groq-llama3-8b   │  │
│   │                          │      │  tier 3 → groq-llama3-70b  │  │
│   │  Model: Random Forest    │      │                            │  │
│   │  Accuracy: 88.9%         │      │  Configurable via YAML —   │  │
│   │  Training: 177 samples   │      │  no redeploy needed        │  │
│   └──────────────────────────┘      └─────────────┬──────────────┘  │
│                                                   │                  │
│   ┌───────────────────────────────────────────────▼──────────────┐  │
│   │                  Unified Model Client                         │  │
│   │                                                               │  │
│   │   OpenAI SDK  │  Anthropic SDK  │  httpx/Ollama  │  Groq     │  │
│   │   One interface, any provider, standardized Response object   │  │
│   └───────────────────────────────────┬───────────────────────────┘  │
│                                       │                              │
└───────────────────────────────────────┼──────────────────────────────┘
                                        │  Response returned to caller
                                        │  immediately (non-blocking)
                                        ▼
               ┌────────────────────────────────────────────┐
               │          Async Verification Queue           │
               │                                            │
               │  Background worker picks up the job        │
               │  Sends prompt + response to judge (70B)    │
               │  Judge scores quality 1–5 per task type    │
               │                                            │
               │  Score ≥ threshold → log PASS              │
               │  Score < threshold → log FAIL              │
               │     → escalate to next-tier model          │
               │     → record to failures.csv               │
               └──────────────────┬─────────────────────────┘
                                  │
               ┌──────────────────▼─────────────────────────┐
               │            SQLite Audit DB                  │
               │                                            │
               │  Every request row contains:               │
               │  timestamp, prompt_hash, complexity_tier,  │
               │  routed_model, cost_usd, latency_ms,       │
               │  quality_score, escalated, cost_delta      │
               └──────────────────┬─────────────────────────┘
                                  │
               ┌──────────────────▼─────────────────────────┐
               │       Streamlit Dashboard  (port 8501)      │
               │                                            │
               │  💰 $X saved vs. GPT-4o baseline           │
               │  Routing pie · Quality histogram           │
               │  Daily cost chart · Escalation trend       │
               │  Full audit log table                      │
               └────────────────────────────────────────────┘
                                  │
               ┌──────────────────▼─────────────────────────┐
               │        Feedback → Retraining Loop           │
               │                                            │
               │  failures.csv ──▶ retrain classifier       │
               │  Weekly: prompts.csv + failures → model.pkl│
               │  Routing gets smarter over time            │
               └────────────────────────────────────────────┘
```

---

## How It Works

### Phase 1 — Unified Model Interface

All LLM provider calls go through a single `send_request(prompt, model_config)` function in [`src/models/client.py`](src/models/client.py). It handles provider-specific SDK differences internally and always returns a standardised `LLMResponse` object containing:

- `text` — the model's output
- `input_tokens` / `output_tokens` — token usage
- `latency_ms` — wall-clock time for the API call
- `cost_usd` — calculated from the model's per-token pricing
- `model_id` / `provider` — which model actually answered

The model registry in [`src/models/registry.py`](src/models/registry.py) holds a `ModelConfig` dataclass for each model with real pricing, average latency, quality tier, and context window. Adding a new model is one dictionary entry.

**Supported providers:** OpenAI, Anthropic, Groq (OpenAI-compatible), Ollama (local REST).

---

### Phase 2 — Complexity Classifier

Every incoming prompt is scored before routing. The classifier lives in [`src/classifier/`](src/classifier/) and works in two steps:

**Step 1 — Feature extraction** ([`features.py`](src/classifier/features.py))

15 numerical features are extracted from the raw prompt text:

| Feature | What it captures |
|---------|-----------------|
| `token_count` | Raw token length via tiktoken |
| `word_count` | Word count |
| `char_count` | Character count |
| `sentence_count` | Number of sentences |
| `avg_word_length` | Vocabulary sophistication |
| `simple_verb_count` | Signals like "what is", "convert", "extract" |
| `moderate_verb_count` | Signals like "summarize", "classify", "explain" |
| `complex_verb_count` | Signals like "design", "architect", "tradeoff" |
| `constraint_count` | Words like "must", "ensure", "without", "while" |
| `output_format_signals` | Requests for JSON, table, essay, report, spec |
| `multi_step_signals` | "step-by-step", "phase", "comprehensive", "detailed plan" |
| `question_count` | Number of `?` characters |
| `has_code_or_data` | Presence of backticks, `def`, `SELECT`, etc. |
| `has_context_provided` | Long quoted text or colon-prefixed context |
| `unique_word_ratio` | Vocabulary richness |

**Step 2 — Classification** ([`train.py`](src/classifier/train.py))

Three models are trained and compared by 5-fold cross-validation. The winner is saved:

| Model | CV Accuracy |
|-------|------------|
| **Random Forest** | **88.65%** ← selected |
| Logistic Regression | 86.45% |
| Gradient Boosting | 85.81% |

**Test set results (held-out 20%):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Tier 1 — Simple | 0.92 | 0.92 | 0.92 |
| Tier 2 — Moderate | 0.86 | 0.86 | 0.86 |
| Tier 3 — Complex | 0.89 | 0.89 | 0.89 |
| **Overall** | **0.89** | **0.89** | **0.89** |

**Top 5 most important features:**

```
char_count              ████████████████████  24.8%
word_count              █████████████████     22.0%
token_count             ██████████            13.1%
moderate_verb_count     ███████                8.7%
avg_word_length         █████                  7.0%
```

Length-based features dominate — simple prompts are short, complex ones are long. The verb-signal features add precision at the decision boundary between tiers 2 and 3.

---

### Phase 3 — Async Quality Verification Loop

After the response is returned to the caller (non-blocking), a background `asyncio` worker picks up the request and runs verification:

**Judge prompt structure:**

```
System: You are a strict quality evaluator. Rate on 1–5:
  5 = Perfect: accurate, complete, well-structured
  4 = Good: mostly correct, minor omissions
  3 = Acceptable: covers basics, misses some aspects
  2 = Poor: significant errors or major omissions
  1 = Unacceptable: wrong, harmful, or off-topic

User: [prompt text]
Response to evaluate: [model output]
Rate the quality. Respond: SCORE: X\nRATIONALE: ...
```

**Quality thresholds by task type** ([`thresholds.py`](src/verifier/thresholds.py)):

| Task Type | Min Score | Rationale |
|-----------|-----------|-----------|
| Extraction | 4.0 | All fields must be present — no hallucination |
| Factual Q&A | 4.0 | Incorrect facts are worse than silence |
| Classification | 4.0 | Label accuracy matters |
| Grammar fix | 4.0 | All errors must be caught |
| Translation | 3.5 | Minor phrasing variance acceptable |
| Summarization | 3.5 | Key points covered, distortion-free |
| Code generation | 3.5 | Functional code, edge cases handled |
| Analysis | 3.0 | Balanced coverage, no major omissions |
| Reasoning | 3.5 | Logical steps present |
| Creative | 3.0 | Meets the brief and is coherent |

**When a failure is detected:**

1. `record_failure(prompt, correct_tier)` appends to `data/labeled_prompts/failures.csv`
2. The request is re-run with the next-tier model (auto-escalation)
3. The escalated response, cost delta, and quality gap are all logged
4. `retrain_from_failures()` merges failure data into the main dataset and retrains — closing the flywheel

---

### Phase 4 — Logging and Cost Dashboard

Every request writes a row to SQLite with full metadata. After the verifier finishes, quality columns are updated in place.

**Schema columns:** `id`, `timestamp`, `prompt_hash`, `prompt_preview`, `complexity_tier`, `tier_confidence`, `routed_model`, `provider`, `input_tokens`, `output_tokens`, `latency_ms`, `cost_usd`, `quality_score`, `quality_threshold`, `quality_passed`, `judge_rationale`, `escalated`, `escalation_model`, `cost_delta_usd`, `quality_gap`

**Dashboard panels** (Streamlit + Plotly, port 8501):

- **Headline metric** — `$X saved (Y% reduction) vs. GPT-4o` — the portfolio number
- **Daily cost chart** — actual cost (green) vs. GPT-4o baseline (pink), area chart
- **Routing distribution** — donut chart showing % of requests per model
- **Quality score histogram** — distribution of 1–5 judge scores
- **Escalation rate over time** — bar chart, total vs. escalated per day
- **Savings by complexity tier** — grouped bars, baseline vs. actual per tier
- **Tier breakdown table** — count, avg latency, cost, savings, quality per tier
- **Recent requests log** — scrollable audit table, 100 most recent rows

---

### Phase 5 — FastAPI Service

The API is an OpenAI-compatible drop-in. Callers don't choose the model — the router does.

**Startup sequence:**

1. `load_dotenv()` — loads API keys
2. `start_worker()` — launches the async verification worker
3. Routes are registered and the server starts

**Request lifecycle:**

```
POST /v1/completions
    │
    ├── Validate request (Pydantic)
    ├── Extract last user message as routing prompt
    ├── route(prompt) → RoutingDecision
    ├── send_request(prompt, model) → LLMResponse
    ├── log_request_with_prompt() → SQLite row
    ├── enqueue_verification() → asyncio.Queue (non-blocking)
    └── Return CompletionResponse with router_metadata
```

---

## Tech Stack

| Component | Tool / Library | Why |
|-----------|---------------|-----|
| Language | Python 3.11+ | Async-native, ecosystem compatibility |
| API Framework | FastAPI + Uvicorn | Async, automatic OpenAPI docs, production-grade |
| LLM Providers | Groq, Ollama | Free tier cloud + free local — zero cost to run |
| Classifier | scikit-learn Random Forest | Lightweight, fast inference, interpretable features |
| Token counting | tiktoken | Accurate token estimates for OpenAI tokenizer |
| Quality eval | LLM-as-judge (Groq 70B) | No human labels needed for ongoing validation |
| Database | SQLite + SQLAlchemy Core | Zero-infrastructure, full audit trail |
| Dashboard | Streamlit + Plotly | Interactive, zero frontend code |
| HTTP client | httpx | Async, used for Ollama REST calls |
| Retry logic | tenacity | Exponential backoff on transient API errors |
| Config | PyYAML | Human-editable routing map, no redeploy needed |
| Containerization | Docker + docker-compose | Multi-service orchestration |

---

## Project Structure

```
cost-autopilot/
│
├── src/
│   ├── models/
│   │   ├── registry.py       ← ModelConfig dataclass + 5-model REGISTRY with real pricing
│   │   ├── client.py         ← send_request() unified interface + per-provider handlers
│   │   └── response.py       ← LLMResponse standardised return object
│   │
│   ├── classifier/
│   │   ├── features.py       ← extract_features() → PromptFeatures (15 signals)
│   │   ├── train.py          ← trains 3 models, picks best by CV, saves model.pkl
│   │   └── predict.py        ← predict_tier() and predict_tier_proba() for the router
│   │
│   ├── router/
│   │   └── router.py         ← route() → RoutingDecision (tier + model + confidence)
│   │
│   ├── verifier/
│   │   ├── thresholds.py     ← QualityThreshold per task type, auto-detected from prompt
│   │   ├── judge.py          ← LLM-as-judge: sends to 70B, parses SCORE + RATIONALE
│   │   ├── verifier.py       ← verify() + asyncio queue + background worker
│   │   └── feedback.py       ← record_failure() + retrain_from_failures()
│   │
│   ├── api/
│   │   ├── app.py            ← FastAPI app, lifespan, CORS, error handler
│   │   ├── schemas.py        ← Pydantic models for all request/response shapes
│   │   └── routes/
│   │       ├── completions.py  ← POST /v1/completions
│   │       ├── models.py       ← GET  /v1/models
│   │       ├── stats.py        ← GET  /v1/stats
│   │       └── config.py       ← GET/PUT /v1/routing-config
│   │
│   ├── dashboard/
│   │   └── app.py            ← Streamlit dashboard, 6 charts + audit table
│   │
│   └── db/
│       ├── schema.py         ← SQLite table definition via SQLAlchemy Core
│       ├── logger.py         ← log_request_with_prompt() + log_verification()
│       └── queries.py        ← 7 SQL queries for dashboard + stats endpoint
│
├── config/
│   └── routing.yaml          ← Tier → model mapping, editable without redeploy
│
├── data/
│   ├── labeled_prompts/
│   │   ├── prompts.csv       ← 177 hand-labeled training examples
│   │   └── failures.csv      ← routing failures (auto-populated by verifier)
│   ├── autopilot.db          ← SQLite audit database
│   ├── baseline_results.json ← Phase 1 provider comparison output
│   ├── classifier_report.json← Training metrics, confusion matrix, feature importance
│   └── load_test_report.json ← Phase 6 load test results
│
├── scripts/
│   ├── load_test.py          ← Async load test with report generation
│   └── seed_data.py          ← Seed DB with 14 days of synthetic history
│
├── tests/
│   ├── test_baseline.py      ← Phase 1: sends 10 prompts to all providers
│   └── test_verifier.py      ← Phase 3: end-to-end verify loop test
│
├── .env.example              ← API key template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local free model)
- A free [Groq API key](https://console.groq.com) (takes 30 seconds to get)

### Option A — Local (no Docker)

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd cost-autopilot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Open .env and set:
#   GROQ_API_KEY=gsk_...
#   OLLAMA_BASE_URL=http://localhost:11434

# 5. Pull local model
ollama serve                        # terminal 1 — keep running
ollama pull llama3.2               # terminal 2 — ~2GB download

# 6. Train the complexity classifier
python -m src.classifier.train
# Expected: Test accuracy: 0.889

# 7. (Optional) Seed the dashboard with 14 days of history
python scripts/seed_data.py

# 8. Start the API
uvicorn src.api.app:app --port 8000 --reload

# 9. Start the dashboard (new terminal)
streamlit run src/dashboard/app.py
```

Services are now live:
- **API** → `http://localhost:8000`
- **API docs** → `http://localhost:8000/docs`
- **Dashboard** → `http://localhost:8501`

### Option B — Docker

```bash
cp .env.example .env    # add your GROQ_API_KEY

docker-compose up --build
```

The compose file starts three services:

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | FastAPI router + verifier worker |
| `dashboard` | 8501 | Streamlit cost dashboard |
| `ollama` | 11434 | Local model server (pulls llama3.2 on first start) |

---

## API Reference

### `POST /v1/completions`

Main endpoint. The router selects the model — callers don't specify one.

**Request:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the CAP theorem?"}
  ],
  "verify_quality": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | required | Chat history. Must include at least one `user` message. |
| `model` | string | null | Ignored — router always selects. |
| `verify_quality` | boolean | true | Whether to run async quality verification. |

**Response:**

```json
{
  "id": "autopilot-afb452a04f85",
  "object": "chat.completion",
  "model": "groq-llama3-8b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "The CAP theorem states..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 52,
    "completion_tokens": 310,
    "total_tokens": 362,
    "latency_ms": 1338.0
  },
  "router_metadata": {
    "selected_model": "groq-llama3-8b",
    "provider": "groq",
    "complexity_tier": 2,
    "routing_confidence": 0.67,
    "tier_probabilities": {"1": 0.08, "2": 0.67, "3": 0.25},
    "estimated_cost_usd": 0.0,
    "routing_reason": "Moderate complexity — routed to mid-tier model for balanced quality/cost"
  }
}
```

---

### `GET /v1/models`

List all registered models with pricing.

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.3-70b-versatile",
      "provider": "groq",
      "quality_tier": "high",
      "cost_per_input_token_usd": 0.0,
      "cost_per_output_token_usd": 0.0,
      "avg_latency_ms": 400,
      "context_window": 128000,
      "display_name": "Llama 3.3 70B (Groq)"
    }
  ]
}
```

---

### `GET /v1/stats`

Live cost savings summary.

```bash
curl http://localhost:8000/v1/stats
```

```json
{
  "total_requests": 1049,
  "actual_cost_usd": 0.082,
  "gpt4o_baseline_cost_usd": 2.379,
  "saved_usd": 2.297,
  "pct_saved": 96.6,
  "avg_latency_ms": 1969.8,
  "avg_quality_score": 4.42,
  "quality_pass_rate": 96.4,
  "escalation_rate": 1.6,
  "escalation_count": 17
}
```

---

### `GET /v1/routing-config`

Read the current tier-to-model mapping.

```bash
curl http://localhost:8000/v1/routing-config
```

---

### `PUT /v1/routing-config`

Swap which model handles which tier — live, no redeploy.

```bash
curl -X PUT http://localhost:8000/v1/routing-config \
  -H "Content-Type: application/json" \
  -d '{"tier_1": "groq-llama3-8b", "tier_3": "groq-llama3-70b"}'
```

Only the fields you send are updated. Model keys must exist in the registry.

---

### `GET /health`

```json
{"status": "ok"}
```

---

## Configuration

All routing behaviour is controlled by [`config/routing.yaml`](config/routing.yaml):

```yaml
routing:
  tier_1: llama3.2          # Simple tasks → local Ollama (free, zero cloud cost)
  tier_2: groq-llama3-8b    # Moderate tasks → Groq 8B (~680ms avg)
  tier_3: groq-llama3-70b   # Complex tasks → Groq 70B (~1500ms avg)

fallback:                   # Used when primary model is unavailable
  tier_1: groq-llama3-8b
  tier_2: groq-llama3-70b
  tier_3: groq-llama3-70b

quality:
  min_agreement_score: 0.75   # Below this → escalate
  judge_model: groq-llama3-70b
```

Changes take effect on the next request. No restart required.

**Environment variables** (`.env`):

```env
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...           # Optional
ANTHROPIC_API_KEY=sk-ant-...    # Optional
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Classifier Deep Dive

### Training data

177 manually labeled prompts in [`data/labeled_prompts/prompts.csv`](data/labeled_prompts/prompts.csv):

- **Tier 1 (61 examples):** Factual Q&A, unit conversions, translations, grammar fixes, simple extractions
- **Tier 2 (70 examples):** Summarization, classification, comparisons, short code tasks, simple analysis
- **Tier 3 (46 examples):** System design, multi-step reasoning, complex code with explanation, philosophical analysis

### Retraining

Every routing failure is appended to `data/labeled_prompts/failures.csv` with the corrected tier. Run this to retrain:

```bash
python -c "from src.verifier.feedback import retrain_from_failures; retrain_from_failures()"
```

The new `model.pkl` takes effect immediately (the predict module uses `lru_cache` — call `reload_model()` to clear it).

### Adding training examples

Add rows directly to `data/labeled_prompts/prompts.csv`:

```csv
"Your prompt text here",2
```

Then retrain:

```bash
python -m src.classifier.train
```

---

## Quality Verification Deep Dive

### Task type detection

Task type is auto-detected from the prompt using keyword regex patterns in [`thresholds.py`](src/verifier/thresholds.py). The detection order matters — more specific patterns are checked first.

### Two verification modes

| Mode | How to use | When to use |
|------|-----------|-------------|
| **Background** (default) | `verify_quality: true` in request | Production — response returned immediately |
| **Synchronous** | Call `verify()` directly | Testing, debugging, guarded pipelines |

### Escalation behaviour

When quality score < threshold:
1. The prompt is re-run on the next-tier model
2. Both responses are logged (original + escalated)
3. `cost_delta_usd` shows the extra cost of escalation
4. The failure is added to `failures.csv` for the next retraining cycle

If the failed request is already at tier 3, no escalation happens — it's logged as a failure only.

---

## Model Registry

Defined in [`src/models/registry.py`](src/models/registry.py). All pricing is real (mid-2025):

| Key | Model | Provider | Tier | Input $/M | Output $/M | Avg Latency |
|-----|-------|----------|------|-----------|-----------|-------------|
| `groq-llama3-70b` | Llama 3.3 70B | Groq | High | $0.00* | $0.00* | 400ms |
| `groq-llama3-8b` | Llama 3.1 8B Instant | Groq | Medium | $0.00* | $0.00* | 150ms |
| `llama3.2` | Llama 3.2 3B | Ollama (local) | Low | $0.00 | $0.00 | ~2500ms† |
| `gpt-4o` | GPT-4o | OpenAI | High | $2.50 | $10.00 | 1800ms |
| `gpt-4o-mini` | GPT-4o-mini | OpenAI | Medium | $0.15 | $0.60 | 800ms |
| `claude-sonnet-3-5` | Claude Sonnet 4.5 | Anthropic | High | $3.00 | $15.00 | 1500ms |
| `claude-haiku-3` | Claude Haiku 4.5 | Anthropic | Low | $0.80 | $4.00 | 500ms |

\* Groq free tier — rate limited but no per-token charge.
† Depends on local hardware. Apple M-series chips handle this well.

---

## Running the Load Test

```bash
# Default: 60 requests, concurrency 3 (safe for Groq free tier)
python scripts/load_test.py

# Custom
python scripts/load_test.py --count 200 --concurrency 5
```

**Load test results (60 requests):**

```
================================================================
  LLM COST AUTOPILOT — LOAD TEST RESULTS
================================================================

  Requests                  60/60 successful
  Duration                  95.0s  (0.63 req/s)
  Errors                    0

  ─────────────────────────────────────────────
  💰 COST SAVINGS
  ─────────────────────────────────────────────
  Actual cost               $0.000000
  GPT-4o baseline           $0.286550
  Saved                     $0.286550
  Reduction                 ✅ 100.0% cheaper than GPT-4o

  ─────────────────────────────────────────────
  ⚡ LATENCY
  ─────────────────────────────────────────────
  Median                    2421ms
  p95                       20779ms
  p99                       28550ms
  Max                       27831ms

  ─────────────────────────────────────────────
  🧭 ROUTING
  ─────────────────────────────────────────────
  Tier accuracy             85.0%
  llama3.2                  29 (48.3%) ████████████████
  groq-llama3-8b            25 (41.7%) █████████████
  groq-llama3-70b            6 (10.0%) ███
================================================================
```

> **Note on p95/p99 latency:** The high values are from Ollama running on a local CPU (MacBook Air). Tier 1 prompts routed to Ollama take 2–30s locally. Groq responses are consistently under 2s. In a deployment where Ollama runs on a GPU or is replaced by a fast API, p95 would be under 3s.

---

## Adding a New Provider

1. **Add to the enum** in [`src/models/registry.py`](src/models/registry.py):
   ```python
   class Provider(str, Enum):
       OPENAI = "openai"
       ANTHROPIC = "anthropic"
       OLLAMA = "ollama"
       GROQ = "groq"
       MISTRAL = "mistral"    # ← new
   ```

2. **Add model configs** to `REGISTRY`:
   ```python
   "mistral-large": ModelConfig(
       provider=Provider.MISTRAL,
       model_id="mistral-large-latest",
       cost_per_input_token=2.0 / 1_000_000,
       cost_per_output_token=6.0 / 1_000_000,
       avg_latency_ms=1200,
       quality_tier=QualityTier.HIGH,
       context_window=128_000,
       display_name="Mistral Large",
   ),
   ```

3. **Add a handler** in [`src/models/client.py`](src/models/client.py):
   ```python
   async def _call_mistral(prompt, config, system):
       # your implementation
       ...

   _DISPATCH[Provider.MISTRAL] = _call_mistral
   ```

4. **Update `config/routing.yaml`** to use the new key — no restart needed.

---

## Reproducing the Results

```bash
# 1. Train classifier from scratch
python -m src.classifier.train

# 2. Run Phase 1 baseline (compare providers)
SKIP_OPENAI=1 SKIP_ANTHROPIC=1 python tests/test_baseline.py

# 3. Run Phase 3 verification test (6 prompts end-to-end)
SKIP_OPENAI=1 SKIP_ANTHROPIC=1 python tests/test_verifier.py

# 4. Seed 14 days of dashboard history
python scripts/seed_data.py

# 5. Run the load test
python scripts/load_test.py --count 60 --concurrency 3

# 6. View the report
cat data/load_test_report.json | python -m json.tool

# 7. Open the dashboard
streamlit run src/dashboard/app.py
```

---

## FAQ

**Q: Why not use LangChain or LiteLLM for provider abstraction?**

This project builds the abstraction layer from scratch intentionally — it's the engineering exercise that demonstrates understanding of the primitives. In production you'd evaluate LiteLLM as a drop-in.

**Q: Why SQLite instead of Postgres?**

Zero infrastructure for a portfolio project. The schema and queries are standard SQL — swapping to Postgres is a one-line change in `src/db/schema.py` (change the connection URL).

**Q: The p95 latency is very high. Is this production-ready?**

The high p95 (20s) is entirely from Ollama on a CPU-only MacBook Air. Tier 1 requests on Groq take under 500ms. In production, Ollama would run on a GPU machine or be replaced by a fast small-model API like Groq's 8B endpoint.

**Q: What happens if a provider goes down?**

The fallback mapping in `routing.yaml` catches unavailable models. The `tenacity` retry decorator in `client.py` retries 3 times with exponential backoff before raising. Provider outages surface as HTTP 502 from the API.

**Q: How do I disable quality verification for latency-sensitive endpoints?**

Pass `"verify_quality": false` in the request body. The response is returned without queuing a verification job.

**Q: Can I use this with my existing OpenAI SDK calls?**

The `/v1/completions` endpoint accepts the same message format as OpenAI's chat completions API. Point your `base_url` at `http://localhost:8000` and set any dummy API key.

---

## License

MIT — use freely, attribution appreciated.
