"""
Concrete implementation — LLM Content Extractor.

Given raw scraped text and a schema, uses the LLM to extract structured data.
Single Responsibility: structured extraction only.

Fix P1: Content truncation increased from 6000 to 12000 chars.
Fix P2: Multi-item extraction for list/ranking pages.
"""

import json
from typing import Any
from openai import OpenAI

from src.interfaces.content_extractor import IContentExtractor
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("llm_extractor")

# Fix P1: increased from 6000 to 12000
_MAX_CONTENT_CHARS = 12000

_SYSTEM_PROMPT = """\
You are a precise data extraction engine. You will be given:
1. Raw text content scraped from a web page.
2. A JSON schema describing the fields to extract.
3. The user's original query for context.

Your task:
- Extract the relevant data from the raw text according to the schema.
- For each field in the schema, provide the best matching value from the text.
- If a field cannot be determined from the text, set it to null.
- The "source_link" field is pre-filled; keep it as-is.
- The "main_content" field should capture the key details: a summary for articles,
  specs for products, requirements for jobs, ingredients for recipes, etc.
  Aim for 2-5 sentences that capture the most useful information.
- Match the expected data type for each field (string, number, boolean, array).
- Respond with ONLY a valid JSON object, no markdown fences, no explanation.
"""

_MULTI_ITEM_SYSTEM_PROMPT = """\
You are a precise data extraction engine. The web page contains MULTIPLE items
(e.g., a list of products, news articles, job postings, properties, recipes,
research papers, or any collection of similar entities).

You will be given:
1. Raw text content scraped from a web page containing multiple items.
2. A JSON schema describing the fields to extract for EACH item.
3. The user's original query for context.

Your task:
- Extract ALL items from the text that are relevant to the user's query.
- Items may appear as numbered lists, bullet points, table rows, cards,
  repeated sections, or any other repeated structure.
- For each item and each field in the schema, provide the best matching value.
- If a field cannot be determined for an item, set it to null.
- The "source_link" field should be the provided URL for all items.
- Skip items that are clearly irrelevant to the user's query (e.g., ads, navigation).
- Match the expected data type for each field (string, number, boolean, array).
- Respond with ONLY a JSON array of objects, no markdown fences, no explanation.
- Extract up to 15 items maximum.
"""


class LLMContentExtractor(IContentExtractor):
    """Uses OpenRouter LLM to extract structured data from raw page text."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

    def extract(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """Extract a single structured item from raw_content."""
        truncated = raw_content[:_MAX_CONTENT_CHARS]
        schema_description = json.dumps(schema_fields, indent=2)

        user_message = (
            f"## User Query\n{user_query}\n\n"
            f"## Schema\n```json\n{schema_description}\n```\n\n"
            f"## Source URL\n{source_url}\n\n"
            f"## Raw Page Content\n{truncated}"
        )

        log.info("Extracting data from %s (%d chars)", source_url, len(truncated))

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)
            result: dict[str, Any] = json.loads(raw)
            result["source_link"] = source_url
            log.info("Successfully extracted %d fields", len(result))
            return result
        except Exception as exc:
            log.error("Extraction failed for %s: %s", source_url, exc)
            return None

    def extract_multiple(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """
        Fix P2: Extract MULTIPLE items from a single page.
        Used for list/ranking/comparison pages.
        """
        truncated = raw_content[:_MAX_CONTENT_CHARS]
        schema_description = json.dumps(schema_fields, indent=2)

        user_message = (
            f"## User Query\n{user_query}\n\n"
            f"## Schema (apply to EACH item)\n```json\n{schema_description}\n```\n\n"
            f"## Source URL\n{source_url}\n\n"
            f"## Raw Page Content\n{truncated}"
        )

        log.info("Multi-item extraction from %s (%d chars)", source_url, len(truncated))

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": _MULTI_ITEM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)
            items: list[dict[str, Any]] = json.loads(raw)

            if not isinstance(items, list):
                items = [items]

            # Ensure source_link on every item
            for item in items:
                item["source_link"] = source_url

            log.info("Extracted %d items from single page", len(items))
            return items
        except Exception as exc:
            log.error("Multi-item extraction failed for %s: %s", source_url, exc)
            return []

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences from LLM output."""
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        return raw.strip()
