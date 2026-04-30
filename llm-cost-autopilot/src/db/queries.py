"""
All SQL queries used by the dashboard and API stats endpoint.
GPT-4o baseline cost is calculated per-row using token counts and real GPT-4o pricing.
"""

from datetime import datetime, timedelta

from sqlalchemy import text

from src.db.schema import engine

# GPT-4o pricing (USD per token) — used only for baseline comparison
GPT4O_INPUT_COST = 2.50 / 1_000_000
GPT4O_OUTPUT_COST = 10.00 / 1_000_000


def _conn():
    return engine().connect()


# ---------------------------------------------------------------------------
# Summary / headline metrics
# ---------------------------------------------------------------------------

def get_summary_stats() -> dict:
    sql = text("""
        SELECT
            COUNT(*)                                    AS total_requests,
            SUM(cost_usd)                               AS actual_cost,
            SUM(input_tokens  * :in_price
              + output_tokens * :out_price)             AS baseline_cost,
            AVG(latency_ms)                             AS avg_latency_ms,
            AVG(quality_score)                          AS avg_quality_score,
            SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END) AS escalation_count,
            SUM(CASE WHEN quality_passed = 1 THEN 1 ELSE 0 END) AS passed_count,
            COUNT(quality_score)                        AS verified_count
        FROM requests
    """)
    with _conn() as conn:
        row = conn.execute(sql, {"in_price": GPT4O_INPUT_COST, "out_price": GPT4O_OUTPUT_COST}).fetchone()

    actual      = row.actual_cost or 0.0
    baseline    = row.baseline_cost or 0.0
    saved       = baseline - actual
    pct_saved   = (saved / baseline * 100) if baseline > 0 else 0.0
    verified    = row.verified_count or 0
    passed      = row.passed_count or 0
    quality_pct = (passed / verified * 100) if verified > 0 else 0.0

    return {
        "total_requests":    row.total_requests or 0,
        "actual_cost":       round(actual, 6),
        "baseline_cost":     round(baseline, 6),
        "saved_usd":         round(saved, 6),
        "pct_saved":         round(pct_saved, 1),
        "avg_latency_ms":    round(row.avg_latency_ms or 0, 1),
        "avg_quality_score": round(row.avg_quality_score or 0, 2),
        "escalation_count":  row.escalation_count or 0,
        "escalation_rate":   round((row.escalation_count or 0) / max(row.total_requests or 1, 1) * 100, 1),
        "quality_pass_rate": round(quality_pct, 1),
    }


# ---------------------------------------------------------------------------
# Cost over time (daily)
# ---------------------------------------------------------------------------

def get_daily_costs(days: int = 14) -> list[dict]:
    sql = text("""
        SELECT
            DATE(timestamp)                             AS day,
            SUM(cost_usd)                               AS actual_cost,
            SUM(input_tokens  * :in_price
              + output_tokens * :out_price)             AS baseline_cost,
            COUNT(*)                                    AS requests
        FROM requests
        WHERE timestamp >= :since
        GROUP BY DATE(timestamp)
        ORDER BY day
    """)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _conn() as conn:
        rows = conn.execute(sql, {
            "in_price": GPT4O_INPUT_COST,
            "out_price": GPT4O_OUTPUT_COST,
            "since": since,
        }).fetchall()

    return [
        {
            "day": r.day,
            "actual_cost": round(r.actual_cost or 0, 6),
            "baseline_cost": round(r.baseline_cost or 0, 6),
            "saved": round((r.baseline_cost or 0) - (r.actual_cost or 0), 6),
            "requests": r.requests,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Routing distribution
# ---------------------------------------------------------------------------

def get_routing_distribution() -> list[dict]:
    sql = text("""
        SELECT
            routed_model,
            provider,
            complexity_tier,
            COUNT(*)        AS request_count,
            SUM(cost_usd)   AS total_cost,
            AVG(latency_ms) AS avg_latency
        FROM requests
        GROUP BY routed_model, provider, complexity_tier
        ORDER BY request_count DESC
    """)
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()

    return [
        {
            "model": r.routed_model,
            "provider": r.provider,
            "tier": r.complexity_tier,
            "count": r.request_count,
            "total_cost": round(r.total_cost or 0, 6),
            "avg_latency": round(r.avg_latency or 0, 1),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Quality score distribution
# ---------------------------------------------------------------------------

def get_quality_distribution() -> list[dict]:
    sql = text("""
        SELECT
            quality_score,
            COUNT(*) AS count
        FROM requests
        WHERE quality_score IS NOT NULL
        GROUP BY quality_score
        ORDER BY quality_score
    """)
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [{"score": r.quality_score, "count": r.count} for r in rows]


# ---------------------------------------------------------------------------
# Escalation events over time
# ---------------------------------------------------------------------------

def get_escalation_trend(days: int = 14) -> list[dict]:
    sql = text("""
        SELECT
            DATE(timestamp)                                         AS day,
            COUNT(*)                                                AS total,
            SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END)         AS escalated,
            SUM(CASE WHEN quality_passed = 0 THEN 1 ELSE 0 END)    AS failed
        FROM requests
        WHERE timestamp >= :since AND quality_score IS NOT NULL
        GROUP BY DATE(timestamp)
        ORDER BY day
    """)
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with _conn() as conn:
        rows = conn.execute(sql, {"since": since}).fetchall()
    return [
        {
            "day": r.day,
            "total": r.total,
            "escalated": r.escalated or 0,
            "failed": r.failed or 0,
            "escalation_rate": round((r.escalated or 0) / max(r.total, 1) * 100, 1),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Recent requests log
# ---------------------------------------------------------------------------

def get_recent_requests(limit: int = 50) -> list[dict]:
    sql = text("""
        SELECT
            id, timestamp, prompt_preview, complexity_tier,
            routed_model, latency_ms, cost_usd,
            quality_score, quality_passed, escalated, escalation_model,
            (input_tokens * :in_price + output_tokens * :out_price) AS baseline_cost
        FROM requests
        ORDER BY id DESC
        LIMIT :limit
    """)
    with _conn() as conn:
        rows = conn.execute(sql, {
            "in_price": GPT4O_INPUT_COST,
            "out_price": GPT4O_OUTPUT_COST,
            "limit": limit,
        }).fetchall()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp,
            "prompt": r.prompt_preview,
            "tier": r.complexity_tier,
            "model": r.routed_model,
            "latency_ms": round(r.latency_ms or 0, 0),
            "cost_usd": round(r.cost_usd or 0, 6),
            "baseline_cost": round(r.baseline_cost or 0, 6),
            "quality_score": r.quality_score,
            "passed": r.quality_passed,
            "escalated": r.escalated,
            "escalation_model": r.escalation_model,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Tier breakdown
# ---------------------------------------------------------------------------

def get_tier_breakdown() -> list[dict]:
    sql = text("""
        SELECT
            complexity_tier,
            COUNT(*)                                    AS count,
            AVG(latency_ms)                             AS avg_latency,
            SUM(cost_usd)                               AS total_cost,
            AVG(quality_score)                          AS avg_quality,
            SUM(input_tokens  * :in_price
              + output_tokens * :out_price)             AS baseline_cost
        FROM requests
        GROUP BY complexity_tier
        ORDER BY complexity_tier
    """)
    with _conn() as conn:
        rows = conn.execute(sql, {
            "in_price": GPT4O_INPUT_COST,
            "out_price": GPT4O_OUTPUT_COST,
        }).fetchall()
    return [
        {
            "tier": r.complexity_tier,
            "count": r.count,
            "avg_latency": round(r.avg_latency or 0, 0),
            "total_cost": round(r.total_cost or 0, 6),
            "baseline_cost": round(r.baseline_cost or 0, 6),
            "saved": round((r.baseline_cost or 0) - (r.total_cost or 0), 6),
            "avg_quality": round(r.avg_quality or 0, 2) if r.avg_quality else None,
        }
        for r in rows
    ]
