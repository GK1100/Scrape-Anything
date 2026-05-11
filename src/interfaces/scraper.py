"""
Interface — Web Scraper.

Responsible ONLY for fetching raw page content.
"""

from abc import ABC, abstractmethod


class IScraper(ABC):
    """Contract for web content fetching."""

    @abstractmethod
    def scrape(self, url: str) -> str | None:
        """
        Fetch and return the main text content of a URL.

        Returns None if the page could not be fetched or parsed.
        """
        ...
