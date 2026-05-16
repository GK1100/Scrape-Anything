"""
Pipeline Orchestrator — ties all components together.

Dependency Inversion: depends on abstractions (interfaces), not concretions.
Single Responsibility: only orchestrates the pipeline stages.

Flow:
  1. Optimize user query (LLM) → get search query + time filter  ─┐
  2. Generate dynamic schema (LLM)                                 ├── parallel
                                                                    ─┘
  3. (Optional) If target_url provided:
     3a. Check robots.txt to verify scraping is allowed
     3b. Scrape target page → extract all links
     3c. Use LLM to filter relevant links (matching user query)
     3d. Follow + scrape those relevant linked pages
  4. Otherwise: Search for URLs (Search API) — filters non-article URLs
  5. Scrape each URL (HTTP → Playwright fallback)  ── concurrent (ThreadPool)
     5a. If index page → extract article links → scrape those
     5b. If list page with multiple items → extract all items (multi-item)
  6. Extract structured data per URL (LLM)         ── concurrent (ThreadPool)
  7. Post-validate: skip results with >60% null fields
  8. Aggregate and save results (JSON)
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from openai import OpenAI

from src.interfaces.search_provider import ISearchProvider
from src.interfaces.scraper import IScraper
from src.interfaces.content_extractor import IContentExtractor
from src.interfaces.schema_generator import ISchemaGenerator
from src.services.query_analyzer import QueryAnalyzer
from src.services.requests_scraper import RequestsScraper
from src.services.llm_extractor import LLMContentExtractor
from src.models.schemas import PipelineResult
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("pipeline")

# Minimum ratio of non-null fields to accept a result.
_MIN_FIELD_FILL_RATIO = 0.30

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

    def run(
        self, user_query: str, max_urls: int = 10, target_url: str | None = None,
    ) -> PipelineResult:
        """Execute the full scraping pipeline and return structured results.

        Args:
            user_query: The user's data retrieval query.
            max_urls: Maximum number of URLs to scrape.
            target_url: Optional specific URL to scrape instead of searching.
        """
        log.info("=" * 60)
        log.info("PIPELINE START - query: %s", user_query)
        if target_url:
            log.info("Target URL provided: %s", target_url)
        log.info("=" * 60)

        # ── Step 0: If target_url, check robots.txt ────────────────────────
        if target_url:
            if not self._is_scraping_allowed(target_url):
                raise PermissionError("Website doesn't allow scraping")

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

        # ── Step 3: Get URLs (search or discover from target_url) ─────────
        if target_url:
            # Scrape the target page, extract all links, filter by relevance
            search_results = self._discover_links_from_target(
                target_url, user_query, max_urls,
            )
        else:
            search_count = min(max_urls * 3, 30)
            search_results = self._search.search(
                optimized_query, num_results=search_count, time_filter=time_filter,
            )
        if not search_results:
            log.warning("No URLs found to scrape. Aborting.")
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
        # Fetch raw HTML (not trafilatura text) so the selector-based
        # extractor can generate CSS selectors and use BS4 to extract.
        scraped_map: dict[str, str | None] = {}
        fetch_fn = (
            self._scraper.fetch_html
            if isinstance(self._scraper, RequestsScraper)
            else self._scraper.scrape
        )
        with ThreadPoolExecutor(max_workers=_SCRAPE_WORKERS) as tp:
            future_to_url = {
                tp.submit(fetch_fn, url): url
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

            # --- List / Index / Aggregator page detection ---
            # When a page looks like a list or index, fetch its raw HTML,
            # extract all clickable elements, use LLM to find relevant links,
            # then follow those links to scrape actual content pages.
            is_list = self._looks_like_list_page(raw_content, user_query)
            is_index = (
                isinstance(self._scraper, RequestsScraper)
                and self._scraper.is_index_page(raw_content, url)
            )

            # URL-pattern based aggregator detection (catches Medium /tag/,
            # Reddit subreddits, topic/category listing pages, etc.)
            if not is_list and not is_index:
                from urllib.parse import urlparse as _urlparse
                _path = _urlparse(url).path.rstrip("/").lower()
                _aggregator_patterns = [
                    "/tag/", "/tags/", "/topic/", "/topics/",
                    "/category/", "/categories/", "/hashtags/",
                    "/search", "/explore", "/collections/",
                    "/trending", "/popular", "/latest",
                ]
                if any(pat in _path or _path.endswith(pat.rstrip("/"))
                       for pat in _aggregator_patterns):
                    is_list = True
                    log.info("URL pattern indicates aggregator page: %s", url)

            if (is_list or is_index) and isinstance(self._scraper, RequestsScraper):
                log.info(
                    "%s page detected at %s — discovering relevant links",
                    "List" if is_list else "Index", url,
                )

                # Fetch raw HTML to access all clickable elements
                page_html = self._scraper.fetch_html(url)
                if page_html:
                    all_links = self._scraper.extract_all_links(url, html=page_html)
                    if all_links:
                        log.info("Found %d clickable elements on %s", len(all_links), url)

                        # LLM filters relevant links
                        relevant_urls = self._filter_relevant_links(
                            all_links, user_query,
                            max_count=max_urls - len(extracted_items),
                        )

                        if relevant_urls:
                            log.info("Following %d relevant links from %s", len(relevant_urls), url)

                            # Scrape relevant linked pages in parallel
                            links_to_scrape = [
                                u for u in relevant_urls if u not in seen_urls
                            ]
                            for u in links_to_scrape:
                                seen_urls.add(u)

                            linked_scraped: dict[str, str | None] = {}
                            with ThreadPoolExecutor(max_workers=_SCRAPE_WORKERS) as tp:
                                lfut = {
                                    tp.submit(fetch_fn, u): u
                                    for u in links_to_scrape
                                }
                                for future in as_completed(lfut):
                                    u = lfut[future]
                                    try:
                                        linked_scraped[u] = future.result()
                                    except Exception as exc:
                                        log.warning("Link scrape failed %s: %s", u, exc)
                                        linked_scraped[u] = None

                            for linked_url in links_to_scrape:
                                if len(extracted_items) >= max_urls:
                                    break
                                urls_attempted += 1
                                linked_content = linked_scraped.get(linked_url)
                                if not linked_content:
                                    urls_failed += 1
                                    continue
                                normal_tasks.append((linked_url, linked_content))
                            continue  # Done with this URL, move on

                # Fallback: if link discovery failed, try multi-item extraction
                if is_list:
                    log.info("Fallback: multi-item extraction from %s", url)
                    multi_items = self._extract_multiple_items(
                        raw_content, url, schema_fields, user_query
                    )
                    for item in multi_items:
                        if len(extracted_items) >= max_urls:
                            break
                        if self._validate_result(item, schema_fields):
                            extracted_items.append(item)
                    if multi_items:
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

    def _discover_links_from_target(
        self,
        target_url: str,
        user_query: str,
        max_urls: int,
    ) -> list[dict]:
        """
        Scrape the target page, extract all outgoing links, use the LLM to
        filter the ones relevant to the user's query, and return them in
        search-result format [{"title": ..., "link": ..., "snippet": ...}].
        """
        log.info("Discovering relevant links from target URL: %s", target_url)

        if not isinstance(self._scraper, RequestsScraper):
            # Fallback: just use the target URL itself
            return [{"title": "", "link": target_url, "snippet": ""}]

        # 1. Fetch raw HTML of the target page
        html = self._scraper.fetch_html(target_url)
        if not html:
            log.warning("Failed to fetch target URL: %s", target_url)
            return [{"title": "", "link": target_url, "snippet": ""}]

        # 2. Extract ALL outgoing links from the page
        all_links = self._scraper.extract_all_links(target_url, html=html)
        if not all_links:
            log.warning("No links found on target page: %s", target_url)
            return [{"title": "", "link": target_url, "snippet": ""}]

        log.info("Found %d total links on target page", len(all_links))

        # 3. Use LLM to filter relevant links
        relevant_urls = self._filter_relevant_links(
            all_links, user_query, max_count=max_urls,
        )

        if not relevant_urls:
            log.warning("LLM found no relevant links — falling back to target URL")
            return [{"title": "", "link": target_url, "snippet": ""}]

        log.info("LLM selected %d relevant links", len(relevant_urls))

        # 4. Build search-result format
        return [
            {"title": "", "link": url, "snippet": ""}
            for url in relevant_urls
        ]

    def _filter_relevant_links(
        self,
        links: list[dict],
        user_query: str,
        max_count: int = 10,
    ) -> list[str]:
        """
        Use the LLM to pick which links on a page are relevant to the
        user's query.

        Parameters
        ----------
        links : list[dict]
            Each dict has {"url": str, "text": str} (anchor text).
        user_query : str
            The original user query.
        max_count : int
            Maximum number of links to return.

        Returns
        -------
        list[str]
            Filtered and ranked list of URLs.
        """
        # Build a compact representation for the LLM
        # Limit to first 100 links to stay within context window
        link_candidates = links[:100]
        numbered_links = "\n".join(
            f"{i+1}. [{l['text']}] → {l['url']}"
            for i, l in enumerate(link_candidates)
        )

        system_prompt = (
            "You are a link-relevance filter. Given a user's query and a list of "
            "links (with anchor text) found on a web page, return ONLY the numbers "
            "of links that are RELEVANT to the user's query.\n\n"
            "Rules:\n"
            "- Select links that point to content pages matching the query topic.\n"
            "- Skip navigation, ads, social media, login, and generic links.\n"
            "- Return at most {max_count} link numbers.\n"
            "- Respond with ONLY a JSON array of integers (the link numbers), "
            "no explanation.\n"
            "- Example: [1, 3, 7, 12]\n"
        ).format(max_count=max_count)

        user_message = (
            f"## User Query\n{user_query}\n\n"
            f"## Links on the page\n{numbered_links}"
        )

        try:
            client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
            response = client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.1,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Clean markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            indices: list[int] = json.loads(raw)
            relevant_urls = []
            for idx in indices:
                # LLM returns 1-indexed numbers
                pos = idx - 1
                if 0 <= pos < len(link_candidates):
                    relevant_urls.append(link_candidates[pos]["url"])

            log.info(
                "LLM filtered %d relevant links from %d candidates",
                len(relevant_urls), len(link_candidates),
            )
            return relevant_urls[:max_count]
        except Exception as exc:
            log.error("LLM link filtering failed: %s", exc)
            return []

    @staticmethod
    def _validate_result(item: dict[str, Any], schema_fields: list[dict]) -> bool:
        """
        Reject results where too many fields are null/empty.
        """
        if not item:
            return False

        if not schema_fields:
            return True  # No schema = nothing to validate

        filled = 0
        for field in schema_fields:
            name = field["name"]
            value = item.get(name)
            if value is not None and value != "" and value != [] and value != "null":
                filled += 1

        ratio = filled / len(schema_fields)
        return ratio >= _MIN_FIELD_FILL_RATIO

    @staticmethod
    def _is_scraping_allowed(url: str) -> bool:
        """
        Check robots.txt to determine if scraping is allowed for the URL.
        Returns True if allowed, False if disallowed.
        """
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            allowed = rp.can_fetch("*", url)
            if not allowed:
                log.warning("robots.txt disallows scraping: %s", url)
            else:
                log.info("robots.txt allows scraping: %s", url)
            return allowed
        except Exception as exc:
            log.warning("Could not fetch robots.txt for %s: %s — assuming allowed", url, exc)
            return True  # If we can't read robots.txt, assume allowed

    @staticmethod
    def _looks_like_list_page(content: str, user_query: str) -> bool:
        """
        Detect if content is a list/ranking/aggregator page that should use
        link-following (not direct extraction). Works across all domains.

        Covers:
          - Traditional list pages (numbered/bulleted items)
          - Card-based aggregator pages (Medium tags, Substack topics, etc.)
          - Search result pages
          - Category/tag/topic listing pages
        """
        q = user_query.lower()

        # ── 1. URL-pattern signals (aggregator/listing page structures) ───
        # These patterns strongly indicate an aggregator page even if
        # the extracted text doesn't look "list-like" in a traditional sense.
        # (URL is not directly available here, but we check content patterns)

        # ── 2. Query signals: user is asking for multiple items ───────────
        list_keywords = [
            "top ", "best ", "list of", "ranking", "rated",
            "compare", "comparison", "vs ",
            " 5 ", " 10 ", " 15 ", " 20 ", " 25 ",
            "find ", "all ",
            "blogs", "articles", "posts", "stories",
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

        # ── 3. Content signals ────────────────────────────────────────────
        lines = content.split("\n")
        non_empty = [l.strip() for l in lines if l.strip()]

        # 3a. Traditional list structure (numbered/bulleted items)
        numbered_lines = sum(
            1 for l in non_empty
            if (l[:3].rstrip(".).").isdigit()
                or l.startswith("- ")
                or l.startswith("* ")
                or l.startswith("• "))
        )

        # Strong signal: many numbered/bulleted items
        if numbered_lines >= 5:
            return True

        # 3b. Card/tile aggregator detection:
        # Pages like Medium /tag/ render as cards. When extracted to text,
        # they produce many short lines (titles, author names, snippets)
        # with very few long paragraphs. Detect this pattern.
        if len(non_empty) > 8:
            short_lines = sum(1 for l in non_empty if len(l) < 100)
            short_ratio = short_lines / len(non_empty)

            # Many long paragraphs = real article, not a listing
            long_paragraphs = sum(1 for l in non_empty if len(l) > 300)

            # Card-based listing: mostly short lines, few long paragraphs
            if short_ratio > 0.7 and long_paragraphs < 3:
                if query_wants_list:
                    return True
                # Even without query signal, very high short ratio + many
                # lines = aggregator page (e.g. tag/topic pages)
                if short_ratio > 0.85 and len(non_empty) > 15:
                    return True

        # 3c. Repetitive pattern detection:
        # Aggregator pages often have repeated structural patterns like
        # "Author · Date" or "Read more" appearing many times
        if len(content) > 500:
            lower_content = content.lower()
            repeat_markers = [
                "read more", "min read", "read time",
                "ago", "·",
                "recommended", "related",
            ]
            repeat_count = sum(
                1 for marker in repeat_markers
                if lower_content.count(marker) >= 3
            )
            if repeat_count >= 2 and query_wants_list:
                return True

        # Combined: query wants list AND content has some list structure
        return query_wants_list and numbered_lines >= 3

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
