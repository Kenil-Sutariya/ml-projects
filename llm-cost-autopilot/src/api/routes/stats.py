from fastapi import APIRouter

from src.api.schemas import StatsResponse
from src.db.queries import get_summary_stats

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    s = get_summary_stats()
    return StatsResponse(
        total_requests=s["total_requests"],
        actual_cost_usd=s["actual_cost"],
        gpt4o_baseline_cost_usd=s["baseline_cost"],
        saved_usd=s["saved_usd"],
        pct_saved=s["pct_saved"],
        avg_latency_ms=s["avg_latency_ms"],
        avg_quality_score=s["avg_quality_score"],
        quality_pass_rate=s["quality_pass_rate"],
        escalation_rate=s["escalation_rate"],
        escalation_count=s["escalation_count"],
    )
