"""
tavily_tool.py — Tavily web search tool (Milestone 8)

Performs real-time web search using the Tavily API.
Requires TAVILY_API_KEY to be set in .env — skips silently without it.
Get a free key at: https://tavily.com

The TavilyClient is instantiated lazily (on first use) so the app
still starts cleanly even if the key is not set.
"""

import logging

from app.core.config import settings
from app.models.schemas import SourceResult
from app.tools.base_tool import BaseResearchTool

logger = logging.getLogger(__name__)

# Optional import — app starts even if tavily-python isn't installed
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore[assignment,misc]


class TavilyTool(BaseResearchTool):
    """Searches the live web using the Tavily API."""

    @property
    def name(self) -> str:
        return "tavily_web"

    def run(self, query: str, max_results: int = 5) -> list[SourceResult]:
        """
        Search the web for `query` using Tavily.

        Returns an empty list (never raises) if the key is missing,
        the API is down, or no results are found.
        """
        if not settings.TAVILY_API_KEY:
            logger.info("TavilyTool: TAVILY_API_KEY not set — skipping web search")
            return []

        try:
            if TavilyClient is None:
                logger.error("TavilyTool: tavily-python not installed. Run: pip install tavily-python")
                return []

            client = TavilyClient(api_key=settings.TAVILY_API_KEY)

            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",   # "advanced" is slower and costs more credits
            )

            raw_results: list[dict] = response.get("results", [])
            if not raw_results:
                logger.info("TavilyTool: no results returned for query='%s'", query)
                return []

            sources: list[SourceResult] = []
            for item in raw_results:
                content = item.get("content", "").strip()
                if not content:
                    continue
                # Truncate long snippets
                if len(content) > 1500:
                    content = content[:1500] + "…"

                sources.append(
                    SourceResult(
                        title=item.get("title", "Web Result"),
                        url=item.get("url"),
                        content=content,
                        source_type="web",
                        score=item.get("score"),
                    )
                )

            logger.info("TavilyTool: returned %d results for query='%s'", len(sources), query)
            return sources

        except Exception as exc:
            # Never crash the whole research run because of one tool
            logger.error("TavilyTool: search failed: %s", exc)
            return []
