"""
schemas.py — Pydantic v2 request/response models

These models define the shape of data flowing into and out of the API.
Pydantic validates them automatically — if a client sends wrong types,
FastAPI returns a clear 422 error before your code even runs.
"""

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """What the client sends when asking a research question."""

    query: str = Field(
        ...,
        min_length=3,
        description="The research question to investigate.",
        examples=["What is quantum entanglement?"],
    )

    # Which data sources should the agent search?
    include_web: bool = Field(
        default=False,
        description="Search the web via Tavily (requires TAVILY_API_KEY in .env).",
    )
    include_wikipedia: bool = Field(
        default=True,
        description="Search Wikipedia for background information.",
    )
    include_private_kb: bool = Field(
        default=True,
        description="Search your private local knowledge base (text files).",
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum results to fetch from each source.",
    )


# ── Building block ───────────────────────────────────────────────────────────

class SourceResult(BaseModel):
    """A single retrieved source snippet from any tool."""

    title: str = Field(description="Title of the source document or page.")
    url: str | None = Field(
        default=None,
        description="URL of the source, if available.",
    )
    content: str = Field(description="The relevant text snippet from this source.")
    source_type: str = Field(
        description="Which tool retrieved this: 'wikipedia', 'web', or 'private_kb'."
    )
    score: float | None = Field(
        default=None,
        description="Relevance score (0–1) if the tool provides one.",
    )


# ── Response ─────────────────────────────────────────────────────────────────

class ResearchResponse(BaseModel):
    """Everything the API returns after completing a research query."""

    query: str = Field(description="The original question that was asked.")
    answer: str = Field(description="The synthesized answer from the local LLM.")
    key_points: list[str] = Field(
        default_factory=list,
        description="Bullet-point highlights extracted from the answer.",
    )
    sources: list[SourceResult] = Field(
        default_factory=list,
        description="All source snippets that were retrieved and used.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the system is in the answer (0 = low, 1 = high).",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of the research tools that returned results.",
    )
