"""
test_tools.py — Tests for research tools

All tests run fully offline using monkeypatching — no real network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import SourceResult
from app.tools.private_kb_tool import PrivateKBTool
from app.tools.tavily_tool import TavilyTool
from app.tools.wikipedia_tool import WikipediaTool


# ── Wikipedia ────────────────────────────────────────────────────────────────

class TestWikipediaTool:
    def setup_method(self):
        self.tool = WikipediaTool()

    def test_name(self):
        assert self.tool.name == "wikipedia"

    def test_returns_source_results_on_success(self):
        # Mock both the search and summary HTTP calls
        mock_search = {"query": {"search": [{"title": "Large language model"}]}}
        mock_summary = {
            "title": "Large language model",
            "extract": "An LLM is a neural network trained on vast text.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Large_language_model"}},
        }

        with patch("app.tools.wikipedia_tool.requests.get") as mock_get:
            # First call = search, second call = summary
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: mock_search, raise_for_status=lambda: None),
                MagicMock(status_code=200, json=lambda: mock_summary, raise_for_status=lambda: None),
            ]
            results = self.tool.run("large language models", max_results=1)

        assert len(results) == 1
        assert results[0].source_type == "wikipedia"
        assert results[0].title == "Large language model"
        assert "LLM" in results[0].content

    def test_returns_empty_list_on_network_error(self):
        with patch("app.tools.wikipedia_tool.requests.get", side_effect=Exception("network down")):
            results = self.tool.run("anything")
        assert results == []

    def test_skips_404_pages(self):
        mock_search = {"query": {"search": [{"title": "NonexistentPageXYZ"}]}}
        with patch("app.tools.wikipedia_tool.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: mock_search, raise_for_status=lambda: None),
                MagicMock(status_code=404, json=lambda: {}, raise_for_status=lambda: None),
            ]
            results = self.tool.run("nonexistent topic")
        assert results == []

    def test_truncates_long_content(self):
        long_text = "A" * 2000
        mock_search = {"query": {"search": [{"title": "Long Article"}]}}
        mock_summary = {"title": "Long Article", "extract": long_text, "content_urls": {}}

        with patch("app.tools.wikipedia_tool.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: mock_search, raise_for_status=lambda: None),
                MagicMock(status_code=200, json=lambda: mock_summary, raise_for_status=lambda: None),
            ]
            results = self.tool.run("long topic")

        assert len(results[0].content) <= 1510  # 1500 + "…"

    def test_does_not_raise(self):
        with patch("app.tools.wikipedia_tool.requests.get", side_effect=Exception("boom")):
            try:
                self.tool.run("any query")
            except Exception as exc:
                pytest.fail(f"WikipediaTool.run raised unexpectedly: {exc}")


# ── Tavily ───────────────────────────────────────────────────────────────────

class TestTavilyTool:
    def setup_method(self):
        self.tool = TavilyTool()

    def test_name(self):
        assert self.tool.name == "tavily_web"

    def test_returns_empty_when_no_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "TAVILY_API_KEY", "")
        results = self.tool.run("machine learning")
        assert results == []

    def test_returns_source_results_with_valid_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "TAVILY_API_KEY", "fake-key-for-test")

        mock_response = {
            "results": [
                {
                    "title": "What is ML?",
                    "url": "https://example.com/ml",
                    "content": "Machine learning is a subset of AI.",
                    "score": 0.95,
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.search.return_value = mock_response

        with patch("app.tools.tavily_tool.TavilyClient", return_value=mock_client):
            results = self.tool.run("machine learning", max_results=3)

        assert len(results) == 1
        assert results[0].source_type == "web"
        assert results[0].title == "What is ML?"
        assert results[0].score == 0.95

    def test_returns_empty_on_api_error(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "TAVILY_API_KEY", "fake-key")

        with patch("app.tools.tavily_tool.TavilyClient", side_effect=Exception("API down")):
            results = self.tool.run("test query")
        assert results == []

    def test_does_not_raise_when_no_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "TAVILY_API_KEY", "")
        try:
            self.tool.run("any query")
        except Exception as exc:
            pytest.fail(f"TavilyTool.run raised unexpectedly: {exc}")


# ── Private KB ────────────────────────────────────────────────────────────────

class TestPrivateKBTool:
    def setup_method(self):
        self.tool = PrivateKBTool()

    def test_name(self):
        assert self.tool.name == "private_kb"

    def test_stub_returns_empty_list(self):
        results = self.tool.run("company policy")
        assert results == []

    def test_does_not_raise(self):
        try:
            self.tool.run("any query")
        except Exception as exc:
            pytest.fail(f"PrivateKBTool.run raised unexpectedly: {exc}")


# ── Shared interface ──────────────────────────────────────────────────────────

class TestBaseToolInterface:
    def test_all_tools_have_name(self):
        for tool in [WikipediaTool(), TavilyTool(), PrivateKBTool()]:
            assert isinstance(tool.name, str) and len(tool.name) > 0

    def test_all_tools_return_list(self):
        with patch("app.tools.wikipedia_tool.requests.get", side_effect=Exception("offline")):
            from app.core import config
            tools = [WikipediaTool(), PrivateKBTool()]
            for tool in tools:
                assert isinstance(tool.run("test"), list)
