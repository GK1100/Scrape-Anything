"""
Interface — Content Extractor.

Given raw page text + a schema, extracts structured data.
"""

from abc import ABC, abstractmethod
from typing import Any


class IContentExtractor(ABC):
    """Contract for structured content extraction."""

    @abstractmethod
    def extract(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """
        Extract structured data from raw_content according to schema_fields.

        Parameters
        ----------
        raw_content : str
            The raw text/HTML of the page.
        source_url : str
            The URL that was scraped (mandatory output field).
        schema_fields : list[dict]
            List of field definitions, e.g.
            [{"name": "price", "type": "string", "description": "..."}]
        user_query : str
            Original user query for context.

        Returns
        -------
        dict or None
            Extracted data conforming to schema, or None on failure.
        """
        ...
