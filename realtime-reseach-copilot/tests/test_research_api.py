"""
test_research_api.py — Tests for FastAPI endpoints

Uses httpx's TestClient so no real server needs to be running.
All LLM and tool calls are stubbed out — these tests work fully offline.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# A TestClient wraps the FastAPI app and sends requests in-process.
client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_content_type_is_json(self):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


class TestResearchEndpoint:
    def test_research_returns_200_with_valid_query(self):
        payload = {
            "query": "What is machine learning?",
            "include_web": False,
            "include_wikipedia": False,  # stub returns empty list
            "include_private_kb": False,
            "max_results": 3,
        }
        response = client.post("/research/", json=payload)
        assert response.status_code == 200

    def test_research_response_has_required_fields(self):
        payload = {"query": "Test query for field validation"}
        response = client.post("/research/", json=payload)
        data = response.json()

        assert "query" in data
        assert "answer" in data
        assert "key_points" in data
        assert "sources" in data
        assert "confidence_score" in data
        assert "tools_used" in data

    def test_research_echoes_query(self):
        question = "What is the speed of light?"
        response = client.post("/research/", json={"query": question})
        assert response.json()["query"] == question

    def test_research_confidence_is_between_0_and_1(self):
        response = client.post("/research/", json={"query": "Test confidence range"})
        score = response.json()["confidence_score"]
        assert 0.0 <= score <= 1.0

    def test_research_rejects_empty_query(self):
        # query must be at least 3 characters (enforced by Pydantic schema)
        response = client.post("/research/", json={"query": "hi"})
        assert response.status_code == 422  # Unprocessable Entity

    def test_research_rejects_missing_query(self):
        response = client.post("/research/", json={})
        assert response.status_code == 422

    def test_research_with_all_sources_disabled(self):
        payload = {
            "query": "No sources query",
            "include_web": False,
            "include_wikipedia": False,
            "include_private_kb": False,
        }
        response = client.post("/research/", json=payload)
        # Should still return 200 with empty sources
        assert response.status_code == 200
        assert response.json()["sources"] == []
        assert response.json()["tools_used"] == []
