"""
AI Scraping Agent — Main Entry Point.

Wires up all dependencies and runs the pipeline.
Usage:
    python -m src.main
    python -m src.main "Give me top 10 AI news in the past 2 days"
"""

import sys
import json

from src.config import Settings
from src.utils.logger import get_logger

# Concrete implementations (injected into the pipeline)
from src.services.serper_search import SerperSearchProvider
from src.services.requests_scraper import RequestsScraper
from src.services.llm_extractor import LLMContentExtractor
from src.services.dynamic_schema import DynamicSchemaGenerator
from src.services.query_analyzer import QueryAnalyzer

# Pipeline orchestrator
from src.orchestrator.pipeline import ScrapingPipeline

log = get_logger("main")


def main() -> None:
    """Entry point — validates config, accepts query, runs pipeline."""

    # ── Validate API Keys ─────────────────────────────────────────────
    missing = Settings.validate()
    if missing:
        log.error("Missing required API keys in .env: %s", missing)
        log.error("Please fill in your .env file. See .env.example for reference.")
        sys.exit(1)

    # ── Get user query ────────────────────────────────────────────────
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        print("\n" + "=" * 56)
        print("       AI SCRAPING AGENT")
        print("=" * 56)
        print("  Enter your data query below.")
        print("  Example: Give me top 10 AI news in the past 2 days")
        print("=" * 56 + "\n")
        user_query = input("Your query: ").strip()

    if not user_query:
        log.error("No query provided. Exiting.")
        sys.exit(1)

    # ── Wire dependencies (Dependency Injection) ──────────────────────
    search_provider = SerperSearchProvider()
    scraper = RequestsScraper()
    extractor = LLMContentExtractor()
    schema_generator = DynamicSchemaGenerator()
    query_analyzer = QueryAnalyzer()

    pipeline = ScrapingPipeline(
        search_provider=search_provider,
        scraper=scraper,
        content_extractor=extractor,
        schema_generator=schema_generator,
        query_analyzer=query_analyzer,
    )

    # ── Execute ───────────────────────────────────────────────────────
    result = pipeline.run(user_query, max_urls=Settings.MAX_URLS)

    # ── Save output ───────────────────────────────────────────────────
    output_path = ScrapingPipeline.save_to_json(result)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [OK]  Scraping complete!")
    print(f"  Items extracted:  {result.total_results}")
    print(f"  URLs failed:      {result.urls_failed}")
    print(f"  Output saved to:  {output_path}")
    print("=" * 60)

    # Print a preview of the first result
    if result.results:
        print("\nPreview of first result:")
        print(json.dumps(result.results[0], indent=2, ensure_ascii=True)[:1000])
        print()


if __name__ == "__main__":
    main()
