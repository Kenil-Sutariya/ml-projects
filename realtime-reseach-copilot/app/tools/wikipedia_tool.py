"""
wikipedia_tool.py — Wikipedia search tool (Milestone 2)

Uses the Wikipedia REST API directly (no third-party wikipedia package)
because the `wikipedia` package is broken on Python 3.14+.

Flow:
  1. POST to Wikipedia's search API to find relevant article titles.
  2. GET the REST summary endpoint for each title to fetch the extract.
  3. Return each article as a SourceResult.

Errors (disambiguation, page not found, network issues) are caught
per-article so one bad result doesn't block the others.
"""

import logging

import requests

from app.models.schemas import SourceResult
from app.tools.base_tool import BaseResearchTool

logger = logging.getLogger(__name__)

# Wikipedia requires a descriptive User-Agent (403 without it)
_HEADERS = {"User-Agent": "ResearchCopilot/1.0 (local-dev; open-source project)"}

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaTool(BaseResearchTool):
    """Searches Wikipedia and returns article summaries as source snippets."""

    @property
    def name(self) -> str:
        return "wikipedia"

    def run(self, query: str, max_results: int = 5) -> list[SourceResult]:
        """
        Search Wikipedia for `query` and return up to `max_results` summaries.

        Returns an empty list (never raises) if Wikipedia is unreachable or
        returns no relevant articles.
        """
        titles = self._search_titles(query, max_results)
        if not titles:
            logger.info("WikipediaTool: no results for query='%s'", query)
            return []

        sources: list[SourceResult] = []
        for title in titles:
            result = self._fetch_summary(title)
            if result:
                sources.append(result)

        logger.info("WikipediaTool: returned %d results for query='%s'", len(sources), query)
        return sources

    # ── Private helpers ──────────────────────────────────────────────────────

    def _search_titles(self, query: str, limit: int) -> list[str]:
        """Ask Wikipedia's search API for article titles matching the query."""
        try:
            resp = requests.get(
                _SEARCH_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                    "utf8": 1,
                },
                headers=_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            return [h["title"] for h in hits]
        except Exception as exc:
            logger.warning("WikipediaTool: search failed: %s", exc)
            return []

    def _fetch_summary(self, title: str) -> SourceResult | None:
        """Fetch the plain-text summary for a single Wikipedia article."""
        try:
            url_title = title.replace(" ", "_")
            resp = requests.get(
                _SUMMARY_URL.format(title=url_title),
                headers=_HEADERS,
                timeout=10,
            )
            # 404 means the title doesn't exist (can happen after search)
            if resp.status_code == 404:
                logger.debug("WikipediaTool: page not found — '%s'", title)
                return None
            resp.raise_for_status()

            data = resp.json()
            extract: str = data.get("extract", "").strip()
            if not extract:
                return None

            # Truncate very long summaries so the LLM prompt stays manageable
            if len(extract) > 1500:
                extract = extract[:1500] + "…"

            return SourceResult(
                title=data.get("title", title),
                url=data.get("content_urls", {}).get("desktop", {}).get("page"),
                content=extract,
                source_type="wikipedia",
                score=None,
            )
        except Exception as exc:
            logger.warning("WikipediaTool: failed to fetch '%s': %s", title, exc)
            return None
