"""
Write request and verification results to SQLite.
"""

import hashlib
from datetime import datetime

from sqlalchemy import insert, update

from src.db.schema import engine, requests_table
from src.models.response import LLMResponse
from src.router.router import RoutingDecision


def log_request(response: LLMResponse, decision: RoutingDecision) -> int:
    """Insert a new request row. Returns the generated row id."""
    prompt_hash = hashlib.sha256(decision.model_key.encode() + response.text.encode()).hexdigest()[:16]

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_hash": prompt_hash,
        "prompt_preview": "",   # set by caller if needed
        "complexity_tier": decision.tier,
        "tier_confidence": decision.confidence,
        "routed_model": decision.model_key,
        "provider": decision.model.provider.value,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "cost_usd": response.cost_usd,
    }

    with engine().begin() as conn:
        result = conn.execute(insert(requests_table).values(**row))
        return result.inserted_primary_key[0]


def log_request_with_prompt(
    prompt: str,
    response: LLMResponse,
    decision: RoutingDecision,
) -> int:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_hash": prompt_hash,
        "prompt_preview": prompt[:200],
        "complexity_tier": decision.tier,
        "tier_confidence": decision.confidence,
        "routed_model": decision.model_key,
        "provider": decision.model.provider.value,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
        "cost_usd": response.cost_usd,
    }

    with engine().begin() as conn:
        result = conn.execute(insert(requests_table).values(**row))
        return result.inserted_primary_key[0]


def log_verification(verification_result) -> None:
    """
    Update an existing row with verifier results.
    If the row_id isn't available, insert a standalone verification record.
    """
    from src.verifier.verifier import VerificationResult
    vr: VerificationResult = verification_result

    prompt_hash = hashlib.sha256(vr.prompt.encode()).hexdigest()[:16]

    values = {
        "quality_score": vr.judge_result.score,
        "quality_threshold": vr.threshold,
        "quality_passed": int(vr.passed),
        "judge_rationale": vr.judge_result.rationale,
        "escalated": int(vr.escalated),
        "escalation_model": vr.escalation_model_key,
        "cost_delta_usd": vr.cost_delta_usd,
        "quality_gap": vr.quality_gap,
    }

    with engine().begin() as conn:
        # Update the most recent row matching this prompt hash
        conn.execute(
            update(requests_table)
            .where(requests_table.c.prompt_hash == prompt_hash)
            .values(**values)
        )
