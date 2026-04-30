"""
Reads routing.yaml and maps classifier tier → ModelConfig.
This is the single decision point the API calls.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.classifier.predict import predict_tier, predict_tier_proba
from src.models.registry import REGISTRY, ModelConfig

CONFIG_PATH = Path("config/routing.yaml")


@dataclass
class RoutingDecision:
    tier: int
    model_key: str
    model: ModelConfig
    confidence: float          # probability assigned to predicted tier
    tier_probabilities: dict   # full distribution


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def route(prompt: str) -> RoutingDecision:
    config = _load_config()
    routing_map = config["routing"]
    fallback_map = config.get("fallback", {})

    tier = predict_tier(prompt)
    proba = predict_tier_proba(prompt)
    confidence = proba.get(tier, 0.0)

    tier_key = f"tier_{tier}"
    model_key = routing_map.get(tier_key)

    # Use fallback if primary model not in registry
    if model_key not in REGISTRY:
        model_key = fallback_map.get(tier_key)

    if not model_key or model_key not in REGISTRY:
        raise ValueError(f"No valid model found for tier {tier}. Check config/routing.yaml.")

    return RoutingDecision(
        tier=tier,
        model_key=model_key,
        model=REGISTRY[model_key],
        confidence=confidence,
        tier_probabilities=proba,
    )
