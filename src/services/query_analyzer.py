"""
Concrete implementation — LLM-powered Query Analyzer.

Rewrites the user's natural-language query into an optimized
Google search query, AND extracts a time-range filter.
Single Responsibility.
"""

import re
import json
from openai import OpenAI

from src.config import settings
from src.utils.logger import get_logger

log = get_logger("query_analyzer")

_SYSTEM_PROMPT = """\
You are a search query optimizer. Given a user's natural-language data request,
produce TWO things:

1. A concise Google search query (5-12 words) that will find relevant web pages
   containing the requested data — these could be articles, product listings,
   job boards, directories, databases, forums, academic pages, government sites,
   or any other page type that matches the user's intent.
2. A time filter code if the user mentions any time range.

Time filter codes:
  "qdr:h"   = past hour
  "qdr:d"   = past 24 hours / past day / today
  "qdr:d2"  = past 2 days
  "qdr:d3"  = past 3 days
  "qdr:w"   = past week / past 7 days
  "qdr:w2"  = past 2 weeks
  "qdr:m"   = past month / past 30 days
  "qdr:m3"  = past 3 months
  "qdr:m6"  = past 6 months
  "qdr:y"   = past year
  null       = no time constraint mentioned

Rules:
- Do NOT put time/date words in the search query if you set a time filter.
- Do NOT use unmatched quotation marks in the query.
- NEVER add site: operators. Country/region context should be expressed as
  natural-language keywords (e.g. "India", "UK") — NOT as site:in or site:co.uk.
- For product/shopping queries, include the country/currency naturally and
  add terms like "best", "buy", "price" to find review/listing pages.
- For job queries, include "jobs", "careers", "openings" and relevant location.
- For data/list queries, use terms that find compilation or listing pages.
- Respond with ONLY a JSON object like: {"query": "...", "time_filter": "qdr:d2"}
- No markdown fences, no explanation.

Example:
  User: "Give me top 10 AI news in the past 2 days"
  Output: {"query": "top AI news articles", "time_filter": "qdr:d2"}

Example:
  User: "Best budget laptops 2026"
  Output: {"query": "best budget laptops 2026 reviews", "time_filter": null}

Example:
  User: "Find top 10 gaming laptops under 90,000 from Indian e-commerce sites"
  Output: {"query": "best gaming laptops under 90000 INR India 2026", "time_filter": null}

Example:
  User: "Remote software engineer jobs paying over $150k"
  Output: {"query": "remote software engineer jobs $150k salary", "time_filter": null}

Example:
  User: "3BHK apartments for rent in Mumbai under 50000 per month"
  Output: {"query": "3BHK apartment rent Mumbai under 50000 monthly", "time_filter": null}

Example:
  User: "Recent breakthroughs in cancer immunotherapy research"
  Output: {"query": "cancer immunotherapy research breakthroughs 2026", "time_filter": null}
"""


class QueryAnalyzerResult:
    """Result from query analysis: optimized query + optional time filter."""

    def __init__(self, query: str, time_filter: str | None = None) -> None:
        self.query = query
        self.time_filter = time_filter

    def __repr__(self) -> str:
        return f"QueryAnalyzerResult(query={self.query!r}, time_filter={self.time_filter!r})"


class QueryAnalyzer:
    """Converts user intent into an optimised search query + time filter."""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )

    def analyze(self, user_query: str) -> QueryAnalyzerResult:
        """Return an optimized query and time filter."""
        log.info("Analyzing query: %s", user_query)

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.2,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Clean markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            parsed = json.loads(raw)
            query = self._sanitize(parsed.get("query", user_query))
            time_filter = parsed.get("time_filter")

            # Validate time_filter format
            if time_filter and not time_filter.startswith("qdr:"):
                log.warning("Invalid time_filter '%s', ignoring", time_filter)
                time_filter = None

            log.info("Optimized query: %s | time_filter: %s", query, time_filter)
            return QueryAnalyzerResult(query=query, time_filter=time_filter)

        except Exception as exc:
            log.warning("Query analysis failed (%s); using raw query", exc)
            # Fallback: try basic time detection from the raw query
            time_filter = self._fallback_time_detection(user_query)
            return QueryAnalyzerResult(query=user_query, time_filter=time_filter)

    # Keep the old method name as a compatibility alias
    def optimize_query(self, user_query: str) -> str:
        """Legacy method — returns only the query string."""
        return self.analyze(user_query).query

    @staticmethod
    def _sanitize(query: str) -> str:
        """Remove malformed quotes and clean up the search query."""
        if query.count('"') % 2 != 0:
            query = query.replace('"', '')
        if query.count("'") % 2 != 0:
            query = query.replace("'", '')
        query = query.strip('`').strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    @staticmethod
    def _fallback_time_detection(query: str) -> str | None:
        """Basic regex fallback to detect time ranges in user queries."""
        q = query.lower()
        if any(w in q for w in ["past hour", "last hour"]):
            return "qdr:h"
        if any(w in q for w in ["today", "past day", "last day", "past 24"]):
            return "qdr:d"
        if any(w in q for w in ["past 2 day", "last 2 day"]):
            return "qdr:d2"
        if any(w in q for w in ["past 3 day", "last 3 day"]):
            return "qdr:d3"
        if any(w in q for w in ["past week", "last week", "this week", "past 7 day"]):
            return "qdr:w"
        if any(w in q for w in ["past 2 week", "last 2 week"]):
            return "qdr:w2"
        if any(w in q for w in ["past month", "last month", "this month", "past 30 day"]):
            return "qdr:m"
        if any(w in q for w in ["past year", "last year", "this year"]):
            return "qdr:y"
        return None
