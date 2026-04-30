"""
private_kb_tool.py — Private knowledge base search tool (Milestone 3: stub)

Reads local .txt files from app/data/private_docs/.
Milestone 1: stub that returns empty list.
Milestone 3: keyword matching implementation.
Milestone 7: upgraded to FAISS vector search.
"""

import logging

from app.models.schemas import SourceResult
from app.tools.base_tool import BaseResearchTool

logger = logging.getLogger(__name__)


class PrivateKBTool(BaseResearchTool):
    """Searches your private local document collection."""

    @property
    def name(self) -> str:
        return "private_kb"

    def run(self, query: str, max_results: int = 5) -> list[SourceResult]:
        # ── Milestone 1 stub ──────────────────────────────────────────────
        # Keyword matching added in Milestone 3.
        # Vector search added in Milestone 7.
        logger.info("PrivateKBTool: stub — no results returned (Milestone 1)")
        return []
