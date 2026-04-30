from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMResponse:
    text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: dict = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> dict:
        return {
            "model": self.model_id,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 8),
            "text_preview": self.text[:120] + ("..." if len(self.text) > 120 else ""),
        }
