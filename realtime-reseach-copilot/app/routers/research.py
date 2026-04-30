"""
research.py — FastAPI router for the /research endpoint

Responsibilities (single responsibility principle):
  - Parse and validate the incoming HTTP request.
  - Build the ResearchAgent with its tools and services.
  - Call the agent.
  - Return the HTTP response.

No business logic lives here — that all belongs in agents/ and services/.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.agents.research_agent import ResearchAgent
from app.models.schemas import ResearchRequest, ResearchResponse
from app.services.confidence_service import ConfidenceService
from app.services.ollama_service import OllamaService
from app.tools.private_kb_tool import PrivateKBTool
from app.tools.tavily_tool import TavilyTool
from app.tools.wikipedia_tool import WikipediaTool

logger = logging.getLogger(__name__)

# Create a router — this gets registered in app/main.py
router = APIRouter(prefix="/research", tags=["research"])


def _build_agent() -> ResearchAgent:
    """
    Construct a ResearchAgent with all available tools and services.

    Called fresh for each request so there is no shared mutable state.
    In a production app you might use FastAPI's dependency injection
    system to cache long-lived services (e.g. a loaded FAISS index).
    """
    tools = [
        WikipediaTool(),
        TavilyTool(),
        PrivateKBTool(),
    ]
    return ResearchAgent(
        tools=tools,
        ollama_service=OllamaService(),
        confidence_service=ConfidenceService(),
    )


@router.post("/", response_model=ResearchResponse, summary="Run a research query")
def research(request: ResearchRequest) -> ResearchResponse:
    """
    Main research endpoint.

    Accepts a question + source preferences, runs all selected tools,
    synthesizes an answer with a local Ollama model, and returns a
    structured response with sources and a confidence score.
    """
    logger.info("POST /research — query='%s'", request.query)

    try:
        agent = _build_agent()
        response = agent.run(request)
        return response
    except Exception as exc:
        logger.exception("Unhandled error during research: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {exc}",
        )
