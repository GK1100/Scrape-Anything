"""
Pipeline Orchestrator — ties all components together.

Dependency Inversion: depends on abstractions (interfaces), not concretions.
Single Responsibility: only orchestrates the pipeline stages.

Flow:
  1. Optimize user query (LLM) → get search query + time filter  ─┐
  2. Generate dynamic schema (LLM)                                 ├── parallel
                                                                    ─┘
  3. Search for URLs (Search API) — filters non-article URLs
  4. Scrape each URL (HTTP → Playwright fallback)  ── concurrent (ThreadPool)
     4a. If index page → extract article links → scrape those
     4b. If list page with multiple items → extract all items (multi-item)
  5. Extract structured data per URL (LLM)         ── concurrent (ThreadPool)
  6. Post-validate: skip results with >60% null fields (Fix P2)
  7. Aggregate and save results (JSON)
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from src.interfaces.search_provider import ISearchProvider
from src.interfaces.scraper import IScraper
from src.interfaces.content_extractor import IContentExtractor
from src.interfaces.schema_generator import ISchemaGenerator
from src.services.query_analyzer import QueryAnalyzer
from src.services.requests_scraper import RequestsScraper
from src.services.llm_extractor import LLMContentExtractor
from src.models.schemas import PipelineResult
from src.utils.logger import get_logger

log = get_logger("pipeline")

# Minimum ratio of non-null DYNAMIC fields to accept a result.
# Only dynamic (non-mandatory) fields count — mandatory fields
# (source_link, title, main_content) are always filled.
_MIN_FIELD_FILL_RATIO = 0.30

# Mandatory field names that are always present and always filled
_MANDATORY_FIELD_NAMES = {"source_link", "title", "main_content"}

# Concurrency settings
_SCRAPE_WORKERS = 6   # parallel HTTP fetches
_EXTRACT_WORKERS = 4  # parallel LLM extraction calls


class ScrapingPipeline:
    """
    Orchestrates the full scraping flow.

    All heavy-lifting services are injected via constructor (DI).
    """

    def __init__(
        self,
        search_provider: ISearchProvider,
        scraper: IScraper,
        content_extractor: IContentExtractor,
        schema_generator: ISchemaGenerator,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self._search = search_provider
        self._scraper = scraper
        self._extractor = content_extractor
        self._schema_gen = schema_generator
        self._query_analyzer = query_analyzer or QueryAnalyzer()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, user_query: str, max_urls: int = 10) -> PipelineResult:
        """Execute the full scraping pipeline and return structured results."""
        log.info("=" * 60)
        log.info("PIPELINE START - query: %s", user_query)
        log.info("=" * 60)

        # ── Step 1 & 2: Run query analysis + schema generation in parallel ─
        with ThreadPoolExecutor(max_workers=2) as tp:
            fut_analysis = tp.submit(self._query_analyzer.analyze, user_query)
            fut_schema = tp.submit(self._schema_gen.generate_schema, user_query)

            analysis = fut_analysis.result()
            schema_fields = fut_schema.result()

        optimized_query = analysis.query
        time_filter = analysis.time_filter
        log.info(
            "Schema fields: %s",
            [f["name"] for f in schema_fields],
        )

        # ── Step 3: Search for URLs (pre-filtered by serper_search) ────────
        search_count = min(max_urls * 3, 30)
        search_results = self._search.search(
            optimized_query, num_results=search_count, time_filter=time_filter,
        )
        if not search_results:
            log.warning("No search results found. Aborting.")
            return PipelineResult(
                query=user_query,
                optimized_query=optimized_query,
                schema_fields=schema_fields,
            )

        # ── Step 4: Scrape URLs concurrently ───────────────────────────────
        # Deduplicate URLs first
        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for result in search_results:
            url = result["link"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_urls.append(url)

        # Scrape in parallel using ThreadPool
        scraped_map: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=_SCRAPE_WORKERS) as tp:
            future_to_url = {
                tp.submit(self._scraper.scrape, url): url
                for url in unique_urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    scraped_map[url] = future.result()
                except Exception as exc:
                    log.warning("Scrape exception for %s: %s", url, exc)
                    scraped_map[url] = None

        # ── Step 5: Extract & validate (process in order, extract in parallel)
        extracted_items: list[dict[str, Any]] = []
        urls_attempted = 0
        urls_failed = 0

        # Categorize scraped URLs into: list pages, index pages, normal pages
        # so we can batch the LLM extraction calls.
        normal_tasks: list[tuple[str, str]] = []  # (url, content)

        for url in unique_urls:
            if len(extracted_items) >= max_urls and not normal_tasks:
                break

            raw_content = scraped_map.get(url)
            if not raw_content:
                urls_attempted += 1
                urls_failed += 1
                continue

            urls_attempted += 1

            # --- List page: multi-item extraction (single LLM call) ---
            if self._looks_like_list_page(raw_content, user_query):
                log.info("List page detected at %s - extracting multiple items", url)
                multi_items = self._extract_multiple_items(
                    raw_content, url, schema_fields, user_query
                )
                for item in multi_items:
                    if len(extracted_items) >= max_urls:
                        break
                    if self._validate_result(item, schema_fields):
                        extracted_items.append(item)
                if multi_items:
                    continue  # Skip index and single extraction

            # --- Index page: follow article links ---
            is_index = (
                isinstance(self._scraper, RequestsScraper)
                and self._scraper.is_index_page(raw_content, url)
            )

            if is_index:
                log.info("Index page detected at %s - following article links", url)
                article_links = self._scraper.extract_article_links(url)

                # Scrape article links in parallel
                article_scraped: dict[str, str | None] = {}
                links_to_scrape = [
                    u for u in article_links
                    if u not in seen_urls
                ]
                for u in links_to_scrape:
                    seen_urls.add(u)

                with ThreadPoolExecutor(max_workers=_SCRAPE_WORKERS) as tp:
                    afut = {
                        tp.submit(self._scraper.scrape, u): u
                        for u in links_to_scrape
                    }
                    for future in as_completed(afut):
                        u = afut[future]
                        try:
                            article_scraped[u] = future.result()
                        except Exception as exc:
                            log.warning("Article scrape failed %s: %s", u, exc)
                            article_scraped[u] = None

                for article_url in links_to_scrape:
                    if len(extracted_items) >= max_urls:
                        break
                    urls_attempted += 1
                    article_content = article_scraped.get(article_url)
                    if not article_content:
                        urls_failed += 1
                        continue
                    normal_tasks.append((article_url, article_content))
                continue

            # --- Normal page: queue for batch extraction ---
            normal_tasks.append((url, raw_content))

        # ── Batch LLM extraction for normal pages (concurrent) ─────────────
        if normal_tasks:
            # Only process as many as we still need
            slots_left = max_urls - len(extracted_items)
            tasks_to_process = normal_tasks[:slots_left + 5]  # slight over-fetch for failures

            with ThreadPoolExecutor(max_workers=_EXTRACT_WORKERS) as tp:
                future_to_url = {
                    tp.submit(
                        self._extractor.extract,
                        raw_content=content,
                        source_url=url,
                        schema_fields=schema_fields,
                        user_query=user_query,
                    ): url
                    for url, content in tasks_to_process
                }
                for future in as_completed(future_to_url):
                    if len(extracted_items) >= max_urls:
                        break
                    url = future_to_url[future]
                    try:
                        structured = future.result()
                        if structured and self._validate_result(structured, schema_fields):
                            extracted_items.append(structured)
                        elif structured:
                            log.warning(
                                "Result from %s rejected: too many null fields (<%d%% filled)",
                                url, int(_MIN_FIELD_FILL_RATIO * 100),
                            )
                            urls_failed += 1
                        else:
                            urls_failed += 1
                    except Exception as exc:
                        log.warning("Extraction exception for %s: %s", url, exc)
                        urls_failed += 1

        # ── Step 7: Build output ───────────────────────────────────────────
        pipeline_result = PipelineResult(
            query=user_query,
            optimized_query=optimized_query,
            schema_fields=schema_fields,
            results=extracted_items,
            total_results=len(extracted_items),
            urls_scraped=urls_attempted - urls_failed,
            urls_failed=urls_failed,
        )

        log.info("=" * 60)
        log.info(
            "PIPELINE COMPLETE - %d items extracted, %d failed",
            pipeline_result.total_results,
            pipeline_result.urls_failed,
        )
        log.info("=" * 60)

        return pipeline_result

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_and_validate(
        self,
        raw_content: str,
        url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """Extract a single item and validate field completeness (Fix P2)."""
        structured = self._extractor.extract(
            raw_content=raw_content,
            source_url=url,
            schema_fields=schema_fields,
            user_query=user_query,
        )
        if structured and self._validate_result(structured, schema_fields):
            return structured
        elif structured:
            log.warning(
                "Result from %s rejected: too many null fields (<%d%% filled)",
                url, int(_MIN_FIELD_FILL_RATIO * 100),
            )
        return None

    def _extract_multiple_items(
        self,
        raw_content: str,
        url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """Fix P2: Use multi-item extraction for list pages."""
        if isinstance(self._extractor, LLMContentExtractor):
            return self._extractor.extract_multiple(
                raw_content=raw_content,
                source_url=url,
                schema_fields=schema_fields,
                user_query=user_query,
            )
        return []

    @staticmethod
    def _validate_result(item: dict[str, Any], schema_fields: list[dict]) -> bool:
        """
        Reject results where too many DYNAMIC fields are null/empty.
        Mandatory fields (source_link, title, main_content) are excluded
        from the ratio because they are always filled.
        """
        if not item:
            return False

        # Must have at least title or main_content with real data
        title = item.get("title", "")
        content = item.get("main_content", "")
        if (not title or title == "null") and (not content or content == "null"):
            return False

        # Count only dynamic (non-mandatory) fields
        dynamic_fields = [f for f in schema_fields if f["name"] not in _MANDATORY_FIELD_NAMES]
        if not dynamic_fields:
            return True  # No dynamic fields = nothing to validate

        filled = 0
        for field in dynamic_fields:
            name = field["name"]
            value = item.get(name)
            if value is not None and value != "" and value != [] and value != "null":
                filled += 1

        ratio = filled / len(dynamic_fields)
        return ratio >= _MIN_FIELD_FILL_RATIO

    @staticmethod
    def _looks_like_list_page(content: str, user_query: str) -> bool:
        """
        Detect if content is a list/ranking page that should use
        multi-item extraction. Works across all domains.
        """
        q = user_query.lower()

        # Query signals: user is asking for multiple items
        list_keywords = [
            "top ", "best ", "list of", "ranking", "rated",
            "compare", "comparison", "vs ",
            " 5 ", " 10 ", " 15 ", " 20 ", " 25 ",
            "find ", "all ",
            # Domain-agnostic multi-item signals
            "apartments", "houses", "properties", "listings",
            "jobs", "openings", "vacancies", "positions",
            "scholarships", "grants", "fellowships",
            "restaurants", "hotels", "flights",
            "recipes", "courses", "programs",
            "stocks", "cryptocurrenc", "coins",
            "laptops", "phones", "cameras", "cars",
        ]
        query_wants_list = any(kw in q for kw in list_keywords)

        # Content signals: page has list-like structure
        # (numbered items, bullet lists, repeated patterns)
        lines = content.split("\n")
        numbered_lines = sum(
            1 for l in lines
            if l.strip() and (
                l.strip()[:3].rstrip(".).").isdigit()
                or l.strip().startswith("- ")
                or l.strip().startswith("* ")
                or l.strip().startswith("• ")
            )
        )
        has_list_structure = numbered_lines >= 3

        # Strong content signal: many numbered/bulleted items = list page
        # regardless of query keywords
        if numbered_lines >= 5:
            return True

        # Combined: query wants list AND content looks list-like
        return query_wants_list and has_list_structure

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_to_json(result: PipelineResult, output_dir: str = "output") -> str:
        """Save pipeline results to a timestamped JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scrape_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        log.info("Results saved to %s", filepath)
        return filepath
