"""
Pydantic data models for pipeline I/O.

These models ensure type-safety and validation at the boundaries.
"""

from pydantic import BaseModel, Field
from typing import Any


class SearchResult(BaseModel):
    """A single search engine result."""

    title: str = ""
    link: str = ""
    snippet: str = ""


class ScrapedItem(BaseModel):
    """A single item of structured scraped data (all fields dynamic)."""

    # All fields are now dynamically generated based on query semantics
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamically generated fields based on query semantics",
    )


class PipelineResult(BaseModel):
    """Full output of a scraping pipeline run."""

    query: str = Field(..., description="Original user query")
    optimized_query: str = Field("", description="Search-optimized query")
    schema_fields: list[dict] = Field(
        default_factory=list, description="Dynamic schema used for extraction"
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted data items"
    )
    total_results: int = 0
    urls_scraped: int = 0
    urls_failed: int = 0
