"""
Pydantic models for request/response validation.
Request shape mirrors OpenAI's chat completion API so the router is a drop-in.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
import uuid


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[ChatMessage]
    # Ignored fields kept for OpenAI-compat (we always route, never trust these)
    model: Optional[str] = Field(default=None, description="Ignored — router selects model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Autopilot-specific
    verify_quality: bool = Field(default=True, description="Run async quality verification")


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class RouterMetadata(BaseModel):
    selected_model: str
    provider: str
    complexity_tier: int
    routing_confidence: float
    tier_probabilities: dict[str, float]
    estimated_cost_usd: float
    routing_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


class CompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"autopilot-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    model: str                          # actual model used
    choices: list[CompletionChoice]
    usage: UsageInfo
    router_metadata: RouterMetadata


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    provider: str
    quality_tier: str
    cost_per_input_token_usd: float
    cost_per_output_token_usd: float
    avg_latency_ms: int
    context_window: int
    display_name: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# /v1/stats
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    total_requests: int
    actual_cost_usd: float
    gpt4o_baseline_cost_usd: float
    saved_usd: float
    pct_saved: float
    avg_latency_ms: float
    avg_quality_score: float
    quality_pass_rate: float
    escalation_rate: float
    escalation_count: int


# ---------------------------------------------------------------------------
# /v1/routing-config
# ---------------------------------------------------------------------------

class RoutingConfigUpdate(BaseModel):
    tier_1: Optional[str] = Field(default=None, description="Model key for simple requests")
    tier_2: Optional[str] = Field(default=None, description="Model key for moderate requests")
    tier_3: Optional[str] = Field(default=None, description="Model key for complex requests")


class RoutingConfigResponse(BaseModel):
    routing: dict[str, str]
    fallback: dict[str, str]
    quality: dict[str, Any]
    message: str = "OK"


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
