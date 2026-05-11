"""
Concrete implementation — Dynamic Schema Generator (LLM-powered).

Analyses user query semantics to produce a tailored JSON schema.
Mandatory fields are always present; extra fields are inferred.
Open/Closed: new mandatory fields can be added to MANDATORY_FIELDS
without changing the LLM prompt logic.
"""

import json
from openai import OpenAI

from src.interfaces.schema_generator import ISchemaGenerator
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("dynamic_schema")

# These fields are ALWAYS present in every output
MANDATORY_FIELDS = [
    {
        "name": "source_link",
        "type": "string",
        "description": "The URL of the page this data was scraped from.",
    },
    {
        "name": "title",
        "type": "string",
        "description": "The title or headline of the content.",
    },
    {
        "name": "main_content",
        "type": "string",
        "description": "Key details, summary, or description of the item.",
    },
]

_SYSTEM_PROMPT = """\
You are a schema architect. Given a user's data-retrieval query, you must 
propose ADDITIONAL fields (beyond the mandatory ones) that would be useful 
for structuring the scraped results.

Mandatory fields that are ALWAYS included (do NOT repeat these):
- source_link (string): URL of the scraped page
- title (string): headline / title
- main_content (string): key details or summary

Respond with a JSON array of objects. Each object has:
  {"name": "<field_name>", "type": "<string|number|boolean|array>", "description": "<what this field captures>"}

Rules:
1. Field names must be snake_case.
2. Only add fields that are genuinely relevant to the query.
3. Keep it concise — typically 2-8 extra fields.
4. For numeric data (prices, ratings, counts), use type "number".
5. For lists (tags, features, requirements), use type "array".
6. If the user explicitly requested specific fields, include all of them.
7. Respond with ONLY the JSON array, no markdown fences, no explanation.
"""


class DynamicSchemaGenerator(ISchemaGenerator):
    """Uses an LLM to infer additional schema fields from the query."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

    def generate_schema(self, user_query: str) -> list[dict]:
        """Return mandatory + dynamically inferred fields."""
        log.info("Generating dynamic schema for query: %s", user_query)

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Clean markdown fences if the model wraps them anyway
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            dynamic_fields: list[dict] = json.loads(raw)
            log.info(
                "LLM proposed %d dynamic fields: %s",
                len(dynamic_fields),
                [f["name"] for f in dynamic_fields],
            )
        except Exception as exc:
            log.warning("Schema generation failed (%s); using mandatory only", exc)
            dynamic_fields = []

        # Merge: mandatory first, then dynamic
        all_fields = list(MANDATORY_FIELDS) + dynamic_fields
        return all_fields
