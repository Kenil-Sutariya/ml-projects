"""
LLM Cost Autopilot — FastAPI service

Run:
    cd cost-autopilot
    uvicorn src.api.app:app --reload --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from src.api.routes import completions, config, models, stats
from src.verifier.verifier import start_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("autopilot.api")


# ---------------------------------------------------------------------------
# Lifespan — start background verifier worker on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting verifier background worker...")
    await start_worker()
    logger.info("API ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Cost Autopilot",
    description=(
        "Intelligent routing layer that classifies prompt complexity and routes "
        "each request to the cheapest model capable of handling it at acceptable quality."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": str(exc)}},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(completions.router, prefix="/v1", tags=["Completions"])
app.include_router(models.router,      prefix="/v1", tags=["Models"])
app.include_router(stats.router,       prefix="/v1", tags=["Stats"])
app.include_router(config.router,      prefix="/v1", tags=["Config"])


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "LLM Cost Autopilot",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
