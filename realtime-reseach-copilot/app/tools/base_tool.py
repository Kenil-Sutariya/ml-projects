"""
base_tool.py — Abstract base class for all research tools

Every tool (Wikipedia, Tavily, PrivateKB, future ArxivTool, etc.) must
inherit from BaseResearchTool and implement the `run` method.

Why an abstract base class?
- Forces a consistent interface across all tools.
- The ResearchAgent can work with any list of tools without knowing
  their internals — classic Dependency Inversion (SOLID).
- Adding a new tool never requires changing the agent code.
"""

from abc import ABC, abstractmethod

from app.models.schemas import SourceResult


class BaseResearchTool(ABC):
    """
    All research tools must implement this interface.

    Usage:
        class MyTool(BaseResearchTool):
            @property
            def name(self) -> str:
                return "my_tool"

            def run(self, query: str, max_results: int = 5) -> list[SourceResult]:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short identifier for this tool, e.g. 'wikipedia' or 'tavily'."""
        ...

    @abstractmethod
    def run(self, query: str, max_results: int = 5) -> list[SourceResult]:
        """
        Search this source and return relevant snippets.

        Args:
            query: The user's research question.
            max_results: Maximum number of results to return.

        Returns:
            A list of SourceResult objects. Return an empty list (not an
            exception) when no results are found or the tool is unavailable.
        """
        ...
