"""
main.py — FastAPI application entry point

Creates the app, wires up middleware, and registers all routers.
Run with:  uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import research

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Real-Time Research Copilot API started.")
    logger.info("Swagger docs: http://localhost:8000/docs")
    yield  # app runs here
    logger.info("Real-Time Research Copilot API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Research Copilot",
    description=(
        "A local-first AI research assistant powered by Ollama. "
        "Ask a question, get a synthesized answer with sources and confidence score."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the Streamlit frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # open for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(research.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"], summary="Check if the API is running")
def health() -> dict[str, str]:
    """Returns {"status": "ok"} when the server is up."""
    return {"status": "ok"}
