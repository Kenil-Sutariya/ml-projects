"""
research_agent.py — Orchestrates the full research workflow

The ResearchAgent is the brain of the application. It:
  1. Receives the user's question and selected options.
  2. Calls each enabled research tool to gather source snippets.
  3. Deduplicates the results.
  4. Passes everything to OllamaService for answer synthesis.
  5. Calculates a confidence score.
  6. Returns a complete ResearchResponse.

Design principle (Dependency Injection):
  Tools and services are passed in from outside — the agent has no
  hard-coded dependencies. This makes it trivially testable (just pass
  mock tools) and extensible (add a new tool without touching this file).
"""

import logging

from app.models.schemas import ResearchRequest, ResearchResponse, SourceResult
from app.services.confidence_service import ConfidenceService
from app.services.ollama_service import OllamaService
from app.tools.base_tool import BaseResearchTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Coordinates tools and services to answer a research question.

    Args:
        tools: A list of BaseResearchTool instances. Order does not matter.
        ollama_service: The service that calls the local LLM.
        confidence_service: The service that scores answer quality.
    """

    def __init__(
        self,
        tools: list[BaseResearchTool],
        ollama_service: OllamaService,
        confidence_service: ConfidenceService,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._ollama = ollama_service
        self._confidence = confidence_service

    def run(self, request: ResearchRequest) -> ResearchResponse:
        """
        Execute the full research workflow for a given request.

        Args:
            request: Validated ResearchRequest from the API endpoint.

        Returns:
            A complete ResearchResponse ready to send back to the client.
        """
        logger.info("ResearchAgent: starting research for query='%s'", request.query)

        # ── Step 1: determine which tools to run ──────────────────────────
        tools_to_run: list[BaseResearchTool] = []

        if request.include_wikipedia and "wikipedia" in self._tools:
            tools_to_run.append(self._tools["wikipedia"])

        if request.include_private_kb and "private_kb" in self._tools:
            tools_to_run.append(self._tools["private_kb"])

        if request.include_web and "tavily_web" in self._tools:
            tools_to_run.append(self._tools["tavily_web"])

        # ── Step 2: run each tool and collect results ─────────────────────
        all_sources: list[SourceResult] = []
        tools_used: list[str] = []

        for tool in tools_to_run:
            try:
                results = tool.run(query=request.query, max_results=request.max_results)
                if results:
                    all_sources.extend(results)
                    tools_used.append(tool.name)
                    logger.info("Tool '%s' returned %d results", tool.name, len(results))
                else:
                    logger.info("Tool '%s' returned 0 results", tool.name)
            except Exception as exc:
                # One broken tool should not kill the whole research run
                logger.error("Tool '%s' raised an error: %s", tool.name, exc)

        # ── Step 3: deduplicate by content ────────────────────────────────
        all_sources = self._deduplicate(all_sources)

        # ── Step 4: generate answer via Ollama ────────────────────────────
        raw_answer = self._ollama.generate_answer(
            query=request.query, sources=all_sources
        )

        # ── Step 5: parse the model's structured response ─────────────────
        # extract_answer_text pulls only the prose answer (not the ## headers)
        # so the frontend doesn't double-render Key Points and Limitations.
        answer_text = self._ollama.extract_answer_text(raw_answer)
        key_points = self._ollama.extract_key_points(raw_answer)

        # ── Step 6: score confidence ──────────────────────────────────────
        confidence = self._confidence.calculate(all_sources)

        logger.info(
            "ResearchAgent: done. sources=%d, tools=%s, confidence=%.2f",
            len(all_sources),
            tools_used,
            confidence,
        )

        return ResearchResponse(
            query=request.query,
            answer=answer_text,
            key_points=key_points,
            sources=all_sources,
            confidence_score=confidence,
            tools_used=tools_used,
        )

    @staticmethod
    def _deduplicate(sources: list[SourceResult]) -> list[SourceResult]:
        """Remove sources with identical content (e.g. same Wikipedia article fetched twice)."""
        seen: set[str] = set()
        unique: list[SourceResult] = []
        for source in sources:
            # Use the first 200 chars of content as the deduplication key
            key = source.content[:200].strip()
            if key not in seen:
                seen.add(key)
                unique.append(source)
        return unique
