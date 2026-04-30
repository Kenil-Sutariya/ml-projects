"""
SQLite schema via SQLAlchemy Core (no ORM — keeps it simple).
One table: requests — one row per routed request with full audit trail.
"""

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)
from pathlib import Path

DB_PATH = Path("data/autopilot.db")
DB_URL = f"sqlite:///{DB_PATH}"

metadata = MetaData()

requests_table = Table(
    "requests",
    metadata,
    Column("id",                Integer, primary_key=True, autoincrement=True),
    Column("timestamp",         String(32)),
    Column("prompt_hash",       String(64)),
    Column("prompt_preview",    String(200)),
    Column("complexity_tier",   Integer),
    Column("tier_confidence",   Float),
    Column("routed_model",      String(64)),
    Column("provider",          String(32)),
    Column("input_tokens",      Integer),
    Column("output_tokens",     Integer),
    Column("latency_ms",        Float),
    Column("cost_usd",          Float),
    # Verifier columns (NULL until background job completes)
    Column("quality_score",     Float,  nullable=True),
    Column("quality_threshold", Float,  nullable=True),
    Column("quality_passed",    Integer, nullable=True),   # 0/1
    Column("judge_rationale",   Text,   nullable=True),
    Column("escalated",         Integer, nullable=True),   # 0/1
    Column("escalation_model",  String(64), nullable=True),
    Column("cost_delta_usd",    Float,  nullable=True),
    Column("quality_gap",       Float,  nullable=True),
)


def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    metadata.create_all(engine)
    return engine


# Module-level singleton
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine
