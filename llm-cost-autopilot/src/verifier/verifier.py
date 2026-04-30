"""
Async quality verification loop.

Flow:
  1. Cheap model returns response to user immediately.
  2. Background task sends same prompt to judge (70B).
  3. If score < threshold → log failure + escalate.
  4. Escalated response is stored in DB for audit (not re-sent to user
     unless run in guarded mode).
  5. Routing failure becomes a training example via feedback.py.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.models.client import send_request
from src.models.registry import REGISTRY, ModelConfig, QualityTier
from src.models.response import LLMResponse
from src.router.router import RoutingDecision
from src.verifier.feedback import record_failure
from src.verifier.judge import JudgeResult, judge_response
from src.verifier.thresholds import get_threshold

logger = logging.getLogger(__name__)

# Tier → next tier up for escalation
_ESCALATION_MAP: dict[int, int] = {1: 2, 2: 3, 3: 3}

# Model key to use for each escalation tier (matches routing.yaml defaults)
_ESCALATION_MODELS: dict[int, str] = {
    2: "groq-llama3-8b",
    3: "groq-llama3-70b",
}


@dataclass
class VerificationResult:
    prompt: str
    original_response: LLMResponse
    decision: RoutingDecision
    judge_result: JudgeResult
    threshold: float
    passed: bool
    escalated: bool
    escalated_response: LLMResponse | None = None
    escalation_model_key: str | None = None
    cost_delta_usd: float = 0.0
    quality_gap: float = 0.0         # how far below threshold
    timestamp: datetime = field(default_factory=datetime.utcnow)


async def verify(
    prompt: str,
    original_response: LLMResponse,
    decision: RoutingDecision,
    escalate_on_failure: bool = True,
) -> VerificationResult:
    """
    Run quality verification for a completed request.
    Called as a background task — does NOT block the user response.
    """
    threshold_config = get_threshold(prompt)
    judge_result = await judge_response(prompt, original_response.text)

    passed = judge_result.score >= threshold_config.min_score
    quality_gap = max(0.0, threshold_config.min_score - judge_result.score)

    escalated = False
    escalated_response = None
    escalation_model_key = None
    cost_delta = 0.0

    if not passed:
        logger.warning(
            "Routing failure | model=%s tier=%d score=%.1f threshold=%.1f | %s",
            decision.model_key,
            decision.tier,
            judge_result.score,
            threshold_config.min_score,
            judge_result.rationale,
        )

        # Record failure for classifier retraining
        escalated_tier = _ESCALATION_MAP[decision.tier]
        record_failure(prompt, correct_tier=escalated_tier)

        if escalate_on_failure:
            escalation_model_key = _ESCALATION_MODELS.get(escalated_tier)
            if escalation_model_key and escalation_model_key != decision.model_key:
                escalation_model = REGISTRY[escalation_model_key]
                try:
                    escalated_response = await send_request(prompt, escalation_model)
                    cost_delta = escalated_response.cost_usd - original_response.cost_usd
                    escalated = True
                    logger.info(
                        "Escalated | %s → %s | cost_delta=$%.6f",
                        decision.model_key,
                        escalation_model_key,
                        cost_delta,
                    )
                except Exception as exc:
                    logger.error("Escalation failed: %s", exc)
    else:
        logger.debug(
            "Quality OK | model=%s score=%.1f/%.1f",
            decision.model_key,
            judge_result.score,
            threshold_config.min_score,
        )

    return VerificationResult(
        prompt=prompt,
        original_response=original_response,
        decision=decision,
        judge_result=judge_result,
        threshold=threshold_config.min_score,
        passed=passed,
        escalated=escalated,
        escalated_response=escalated_response,
        escalation_model_key=escalation_model_key,
        cost_delta_usd=cost_delta,
        quality_gap=quality_gap,
    )


# ---------------------------------------------------------------------------
# Background queue — fire-and-forget verification without blocking the caller
# ---------------------------------------------------------------------------

_queue: asyncio.Queue | None = None
_worker_task: asyncio.Task | None = None


def get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


async def _worker():
    """Drains the verification queue indefinitely."""
    q = get_queue()
    while True:
        job = await q.get()
        try:
            result = await verify(**job)
            # Persist to DB (Phase 4 wires this in)
            try:
                from src.db.logger import log_verification
                await asyncio.to_thread(log_verification, result)
            except Exception:
                pass  # DB not yet initialised — safe to skip
        except Exception as exc:
            logger.error("Verifier worker error: %s", exc)
        finally:
            q.task_done()


async def start_worker():
    """Launch the background worker. Call once at app startup."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())
        logger.info("Verifier worker started.")


async def enqueue_verification(
    prompt: str,
    original_response: LLMResponse,
    decision: RoutingDecision,
    escalate_on_failure: bool = True,
) -> None:
    """Non-blocking: push a verification job onto the queue."""
    q = get_queue()
    await q.put({
        "prompt": prompt,
        "original_response": original_response,
        "decision": decision,
        "escalate_on_failure": escalate_on_failure,
    })
