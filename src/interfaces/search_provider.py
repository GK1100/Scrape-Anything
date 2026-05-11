"""
Interface — Search Provider (ISP: narrow contract).

Any search backend (Serper, SerpAPI, Brave, etc.) MUST implement this
interface so the orchestrator can swap providers without code changes.
"""

from abc import ABC, abstractmethod


class ISearchProvider(ABC):
    """Contract for URL discovery from a user query."""

    @abstractmethod
    def search(self, query: str, num_results: int = 10, time_filter: str | None = None) -> list[dict]:
        """
        Return a list of search results.

        Each result is a dict with at least:
            - "title": str
            - "link": str
            - "snippet": str  (optional)

        Parameters
        ----------
        time_filter : str | None
            Optional time-range filter (e.g. "qdr:d2" for past 2 days).
        """
        ...
