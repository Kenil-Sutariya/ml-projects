"""
vector_store.py — FAISS vector index for private knowledge base (Milestone 7: stub)

Stores and searches document embeddings on disk so the private KB
can do semantic (meaning-based) search instead of keyword matching.
Milestone 1: stub only.
Milestone 7: full FAISS implementation.
"""

import logging

from app.models.schemas import SourceResult

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages a FAISS index for semantic search over local documents."""

    def build_index_from_documents(self, docs_path: str) -> None:
        """
        Read all .txt files in docs_path, embed them, and save a FAISS index.
        Called once at startup (or whenever docs change).
        """
        # ── Milestone 7 stub ──────────────────────────────────────────────
        logger.info("VectorStoreService.build_index_from_documents: stub (Milestone 7)")

    def search(self, query: str, top_k: int = 5) -> list[SourceResult]:
        """
        Find the top_k most semantically similar document chunks.

        Returns an empty list until Milestone 7 is implemented.
        """
        # ── Milestone 7 stub ──────────────────────────────────────────────
        logger.info("VectorStoreService.search: stub (Milestone 7)")
        return []
