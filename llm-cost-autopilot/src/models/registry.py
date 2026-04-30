from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class ModelConfig:
    provider: Provider
    model_id: str
    cost_per_input_token: float   # USD per token
    cost_per_output_token: float  # USD per token
    avg_latency_ms: int           # rough baseline latency
    quality_tier: QualityTier
    context_window: int           # max input tokens
    display_name: Optional[str] = None

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.model_id

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


# ---------------------------------------------------------------------------
# Registry — real pricing as of mid-2025 (USD per token)
# OpenAI:    https://openai.com/api/pricing
# Anthropic: https://www.anthropic.com/pricing
# Ollama:    free (local compute)
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelConfig] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_input_token=2.50 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        avg_latency_ms=1800,
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        display_name="GPT-4o",
    ),
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_input_token=0.15 / 1_000_000,
        cost_per_output_token=0.60 / 1_000_000,
        avg_latency_ms=800,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        display_name="GPT-4o-mini",
    ),
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-sonnet-3-5": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-5",
        cost_per_input_token=3.00 / 1_000_000,
        cost_per_output_token=15.00 / 1_000_000,
        avg_latency_ms=1500,
        quality_tier=QualityTier.HIGH,
        context_window=200_000,
        display_name="Claude Sonnet 3.5",
    ),
    "claude-haiku-3": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5",
        cost_per_input_token=0.80 / 1_000_000,
        cost_per_output_token=4.00 / 1_000_000,
        avg_latency_ms=500,
        quality_tier=QualityTier.LOW,
        context_window=200_000,
        display_name="Claude Haiku 3",
    ),
    # ── Groq (free cloud tier — OpenAI-compatible) ───────────────────────────
    "groq-llama3-70b": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-versatile",
        cost_per_input_token=0.0,   # free tier
        cost_per_output_token=0.0,
        avg_latency_ms=400,         # Groq is extremely fast (custom LPU hardware)
        quality_tier=QualityTier.HIGH,
        context_window=128_000,
        display_name="Llama 3.3 70B (Groq)",
    ),
    "groq-llama3-8b": ModelConfig(
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=150,
        quality_tier=QualityTier.MEDIUM,
        context_window=128_000,
        display_name="Llama 3.1 8B Instant (Groq)",
    ),
    # ── Ollama (local — zero cloud cost) ─────────────────────────────────────
    "llama3.2": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="llama3.2",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        avg_latency_ms=2500,   # depends on local hardware
        quality_tier=QualityTier.LOW,
        context_window=128_000,
        display_name="Llama 3.2 (local)",
    ),
}


def get_model(key: str) -> ModelConfig:
    if key not in REGISTRY:
        raise KeyError(f"Model '{key}' not found. Available: {list(REGISTRY)}")
    return REGISTRY[key]


def list_models() -> list[ModelConfig]:
    return list(REGISTRY.values())


def models_by_tier(tier: QualityTier) -> list[ModelConfig]:
    return [m for m in REGISTRY.values() if m.quality_tier == tier]
