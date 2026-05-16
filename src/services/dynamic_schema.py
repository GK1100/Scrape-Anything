"""
Concrete implementation — Dynamic Schema Generator (LLM-powered).

Analyses user query semantics to produce a fully dynamic JSON schema.
All fields are inferred by the LLM based on the user's query — there
are no hard-coded mandatory fields.
"""

import json
from openai import OpenAI

from src.interfaces.schema_generator import ISchemaGenerator
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("dynamic_schema")



_SYSTEM_PROMPT = """\
You are a schema architect. Given a user's data-retrieval query, you must 
propose ALL fields that would be useful for structuring the scraped results.

There are NO mandatory fields — you decide the complete schema based on the query.
Always include a "source_link" (string) field for the URL of the scraped page,
and consider including fields like "title" or "name" when appropriate.

Respond with a JSON array of objects. Each object has:
  {"name": "<field_name>", "type": "<string|number|boolean|array>", "description": "<what this field captures>"}

Rules:
1. Field names must be snake_case.
2. Only add fields that are genuinely relevant to the query.
3. Keep it concise — typically 3-10 fields total.
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
        """Return dynamically inferred fields (no mandatory fields)."""
        log.info("Generating dynamic schema for query: %s", user_query)

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.2,
                max_tokens=512,
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

            fields: list[dict] = json.loads(raw)
            log.info(
                "LLM proposed %d fields: %s",
                len(fields),
                [f["name"] for f in fields],
            )
        except Exception as exc:
            log.warning("Schema generation failed (%s); returning empty schema", exc)
            fields = []

        return fields
