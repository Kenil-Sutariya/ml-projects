"""
confidence_service.py — Confidence score calculator (Milestone 5)

Scores how well-supported an answer is based on:
  - Number of sources retrieved
  - Diversity of source types (wikipedia vs web vs private_kb)
  - Average content length (longer snippets = more context)

Score table (approximate):
  0 sources                             → 0.10
  1 source                              → 0.40
  2–3 sources, 1 type                   → 0.60
  2–3 sources, 2+ types                 → 0.70
  4+ sources, 2+ types                  → 0.80
  4+ sources, 3 types, rich content     → 0.90+
"""

import logging

from app.models.schemas import SourceResult

logger = logging.getLogger(__name__)

# Tuning constants — adjust these to recalibrate the scoring
_BASE_NO_SOURCES = 0.10
_SOURCE_COUNT_WEIGHT = 0.08   # added per source (capped)
_MAX_SOURCE_BONUS = 0.40      # maximum bonus from source count alone
_DIVERSITY_BONUS = 0.10       # added per unique source type beyond the first
_CONTENT_BONUS_MAX = 0.10     # bonus for having rich, lengthy content


class ConfidenceService:
    """Calculates a 0–1 confidence score based on the retrieved sources."""

    def calculate(self, sources: list[SourceResult]) -> float:
        """
        Score how trustworthy the answer is.

        Args:
            sources: All source snippets gathered during the research run.

        Returns:
            A float clamped to [0.0, 1.0].
        """
        if not sources:
            logger.debug("ConfidenceService: no sources → score=%.2f", _BASE_NO_SOURCES)
            return _BASE_NO_SOURCES

        # ── Component 1: source count ────────────────────────────────────
        count_bonus = min(len(sources) * _SOURCE_COUNT_WEIGHT, _MAX_SOURCE_BONUS)

        # ── Component 2: source type diversity ───────────────────────────
        unique_types = {s.source_type for s in sources}
        diversity_bonus = max(0, len(unique_types) - 1) * _DIVERSITY_BONUS

        # ── Component 3: content richness ────────────────────────────────
        avg_length = sum(len(s.content) for s in sources) / len(sources)
        # Full bonus at 800+ chars average; scales linearly below that
        content_bonus = min(avg_length / 800, 1.0) * _CONTENT_BONUS_MAX

        score = _BASE_NO_SOURCES + count_bonus + diversity_bonus + content_bonus
        score = round(min(score, 1.0), 3)

        logger.debug(
            "ConfidenceService: sources=%d, types=%s, avg_len=%.0f → score=%.2f",
            len(sources), unique_types, avg_length, score,
        )
        return score
