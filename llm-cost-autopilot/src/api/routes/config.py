"""
GET/PUT /v1/routing-config — read or update tier-to-model mappings live.
Changes write directly to config/routing.yaml without redeployment.
"""

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from src.api.schemas import RoutingConfigResponse, RoutingConfigUpdate
from src.models.registry import REGISTRY

router = APIRouter()
CONFIG_PATH = Path("config/routing.yaml")


def _load() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _save(config: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


@router.get("/routing-config", response_model=RoutingConfigResponse)
def get_routing_config():
    config = _load()
    return RoutingConfigResponse(
        routing=config.get("routing", {}),
        fallback=config.get("fallback", {}),
        quality=config.get("quality", {}),
    )


@router.put("/routing-config", response_model=RoutingConfigResponse)
def update_routing_config(update: RoutingConfigUpdate):
    config = _load()
    routing = config.setdefault("routing", {})

    changes = {
        "tier_1": update.tier_1,
        "tier_2": update.tier_2,
        "tier_3": update.tier_3,
    }

    for tier_key, model_key in changes.items():
        if model_key is None:
            continue
        if model_key not in REGISTRY:
            raise HTTPException(
                status_code=422,
                detail=f"Model '{model_key}' not in registry. Available: {list(REGISTRY)}",
            )
        routing[tier_key] = model_key

    _save(config)

    return RoutingConfigResponse(
        routing=config.get("routing", {}),
        fallback=config.get("fallback", {}),
        quality=config.get("quality", {}),
        message=f"Updated: {[k for k, v in changes.items() if v]}",
    )
