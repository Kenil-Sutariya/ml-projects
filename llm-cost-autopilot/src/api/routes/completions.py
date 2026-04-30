"""
POST /v1/completions — the main routing endpoint.

The caller sends a standard chat message. The router classifies complexity,
picks the cheapest capable model, calls it, and returns the response with
routing metadata. Quality verification runs in the background.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    RouterMetadata,
    UsageInfo,
)
from src.db.logger import log_request_with_prompt
from src.models.client import send_request
from src.router.router import route
from src.verifier.verifier import enqueue_verification

router = APIRouter()

_TIER_REASONS = {
    1: "Simple request — routed to fastest/cheapest model",
    2: "Moderate complexity — routed to mid-tier model for balanced quality/cost",
    3: "Complex request — routed to highest-capability model",
}


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    # Extract the user-facing prompt (last user message)
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=422, detail="At least one user message is required.")

    prompt = user_messages[-1].content
    system_prompt = next(
        (m.content for m in request.messages if m.role == "system"), None
    )

    # Route
    try:
        decision = route(prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Routing error: {exc}")

    # Call model
    try:
        response = await send_request(prompt, decision.model, system=system_prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {exc}")

    # Log to DB (non-blocking — don't fail request if DB write fails)
    try:
        log_request_with_prompt(prompt, response, decision)
    except Exception:
        pass

    # Queue background verification
    if request.verify_quality:
        try:
            await enqueue_verification(
                prompt=prompt,
                original_response=response,
                decision=decision,
            )
        except Exception:
            pass

    # Build response
    tier_proba_str = {str(k): v for k, v in decision.tier_probabilities.items()}

    return CompletionResponse(
        model=decision.model_key,
        choices=[
            CompletionChoice(
                message=ChatMessage(role="assistant", content=response.text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            latency_ms=round(response.latency_ms, 1),
        ),
        router_metadata=RouterMetadata(
            selected_model=decision.model_key,
            provider=decision.model.provider.value,
            complexity_tier=decision.tier,
            routing_confidence=decision.confidence,
            tier_probabilities=tier_proba_str,
            estimated_cost_usd=round(response.cost_usd, 8),
            routing_reason=_TIER_REASONS[decision.tier],
        ),
    )
