from fastapi import APIRouter

from src.api.schemas import ModelInfo, ModelsResponse
from src.models.registry import list_models

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def get_models():
    return ModelsResponse(
        data=[
            ModelInfo(
                id=m.model_id,
                provider=m.provider.value,
                quality_tier=m.quality_tier.value,
                cost_per_input_token_usd=m.cost_per_input_token,
                cost_per_output_token_usd=m.cost_per_output_token,
                avg_latency_ms=m.avg_latency_ms,
                context_window=m.context_window,
                display_name=m.display_name,
            )
            for m in list_models()
        ]
    )
