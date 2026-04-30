"""
embedding_service.py — Text embeddings via Ollama (Milestone 7: stub)

Uses the nomic-embed-text model running locally in Ollama to convert
text into numeric vectors for semantic similarity search.
Milestone 1: stub only.
Milestone 7: real embedding calls + FAISS integration.
"""

import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Converts text to embedding vectors using a local Ollama model."""

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string into a float vector.

        Install the model first:  ollama pull nomic-embed-text
        """
        # ── Milestone 7 stub ──────────────────────────────────────────────
        logger.info("EmbeddingService.embed_text: stub (Milestone 7)")
        return []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings — more efficient than calling embed_text in a loop."""
        # ── Milestone 7 stub ──────────────────────────────────────────────
        logger.info("EmbeddingService.embed_texts: stub (Milestone 7)")
        return [[] for _ in texts]
