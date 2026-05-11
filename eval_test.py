"""
Evaluation Test Suite — runs 10 diverse queries through the pipeline,
captures metrics, and generates an evaluation matrix.
"""

import json
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Settings
from src.services.serper_search import SerperSearchProvider
from src.services.requests_scraper import RequestsScraper
from src.services.llm_extractor import LLMContentExtractor
from src.services.dynamic_schema import DynamicSchemaGenerator
from src.services.query_analyzer import QueryAnalyzer
from src.orchestrator.pipeline import ScrapingPipeline
from openai import OpenAI

# ── Test Queries (diverse domains & time ranges) ──────────────────────
TEST_QUERIES = [
    {
        "id": 1,
        "query": "Give me top 5 AI news in the past 2 days",
        "domain": "AI/Tech News",
        "expected": "Recent AI news articles with dates within last 2 days",
    },
    {
        "id": 2,
        "query": "Latest SpaceX rocket launches this week",
        "domain": "Space/Aerospace",
        "expected": "SpaceX launch info with mission names, dates, outcomes",
    },
    {
        "id": 3,
        "query": "Top 5 best budget smartphones under $300 in 2026",
        "domain": "Consumer Electronics",
        "expected": "Phone names, prices, specs, ratings",
    },
    {
        "id": 4,
        "query": "Recent breakthroughs in cancer research this month",
        "domain": "Medical/Science",
        "expected": "Research findings, institutions, treatment types",
    },
    {
        "id": 5,
        "query": "What are the top trending GitHub repositories this week",
        "domain": "Software/Dev",
        "expected": "Repo names, descriptions, star counts, languages",
    },
    {
        "id": 6,
        "query": "Latest cryptocurrency market news and Bitcoin price updates",
        "domain": "Finance/Crypto",
        "expected": "Price data, market movements, analysis",
    },
    {
        "id": 7,
        "query": "Best new movies released in May 2026",
        "domain": "Entertainment",
        "expected": "Movie titles, release dates, genres, ratings",
    },
    {
        "id": 8,
        "query": "Climate change policy updates from the past week",
        "domain": "Environment/Policy",
        "expected": "Policy decisions, countries, agreements, dates",
    },
    {
        "id": 9,
        "query": "Top remote job openings for software engineers",
        "domain": "Jobs/Careers",
        "expected": "Job titles, companies, salary ranges, requirements",
    },
    {
        "id": 10,
        "query": "Recent cybersecurity breaches and data leaks in 2026",
        "domain": "Cybersecurity",
        "expected": "Breach details, affected companies, data types, dates",
    },
]


def evaluate_single_result(client, query_info, pipeline_result):
    """Use LLM to evaluate the quality of a single pipeline result."""
    
    results_summary = json.dumps(pipeline_result.results[:3], indent=2, ensure_ascii=False)
    schema_fields = [f["name"] for f in pipeline_result.schema_fields]
    
    eval_prompt = f"""You are a strict evaluator. Rate the following scraping results on a scale of 1-10 for each criterion.

## Query
{query_info["query"]}

## Expected Output
{query_info["expected"]}

## Dynamic Schema Fields Generated
{schema_fields}

## Results (first 3 shown)
{results_summary}

## Total Results: {pipeline_result.total_results}
## URLs Failed: {pipeline_result.urls_failed}

Rate each criterion (1-10) and respond with ONLY a JSON object:
{{
  "relevance": <1-10, how relevant are results to the query>,
  "freshness": <1-10, how recent/timely are the results>,
  "completeness": <1-10, how complete are the extracted fields (non-null)>,
  "schema_quality": <1-10, how well does the dynamic schema fit the query>,
  "content_depth": <1-10, how detailed/useful is main_content>,
  "overall": <1-10, overall quality score>,
  "issues": "<brief description of any problems found>"
}}
No markdown fences."""

    try:
        response = client.chat.completions.create(
            model=Settings.OPENROUTER_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a strict, objective evaluator. Be honest and critical."},
                {"role": "user", "content": eval_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"  [WARN] Evaluation failed: {exc}")
        return {
            "relevance": 0, "freshness": 0, "completeness": 0,
            "schema_quality": 0, "content_depth": 0, "overall": 0,
            "issues": f"Evaluation error: {exc}"
        }


def calculate_field_completeness(results):
    """Calculate what % of fields are non-null across all results."""
    if not results:
        return 0.0
    total_fields = 0
    non_null = 0
    for item in results:
        for key, value in item.items():
            total_fields += 1
            if value is not None and value != "" and value != []:
                non_null += 1
    return round((non_null / total_fields * 100) if total_fields > 0 else 0, 1)


def main():
    # Validate
    missing = Settings.validate()
    if missing:
        print(f"Missing API keys: {missing}")
        sys.exit(1)

    # Wire up pipeline
    pipeline = ScrapingPipeline(
        search_provider=SerperSearchProvider(),
        scraper=RequestsScraper(),
        content_extractor=LLMContentExtractor(),
        schema_generator=DynamicSchemaGenerator(),
        query_analyzer=QueryAnalyzer(),
    )

    eval_client = OpenAI(
        api_key=Settings.OPENROUTER_API_KEY,
        base_url=Settings.OPENROUTER_BASE_URL,
    )

    # ── Run all queries ───────────────────────────────────────────────
    all_evaluations = []

    for tq in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"  TEST {tq['id']}/10: {tq['query']}")
        print(f"  Domain: {tq['domain']}")
        print(f"{'='*60}")

        start = time.time()
        result = pipeline.run(tq["query"], max_urls=5)  # Cap at 5 for speed
        elapsed = round(time.time() - start, 1)

        # Save individual result
        output_path = pipeline.save_to_json(result, output_dir="output/eval")

        # Calculate metrics
        field_completeness = calculate_field_completeness(result.results)

        # LLM evaluation
        print(f"  [TIME] Took {elapsed}s | Results: {result.total_results} | Failed: {result.urls_failed}")
        print(f"  [STATS] Field completeness: {field_completeness}%")
        print(f"  [EVAL] Evaluating quality...")

        scores = evaluate_single_result(eval_client, tq, result)

        eval_entry = {
            "id": tq["id"],
            "query": tq["query"],
            "domain": tq["domain"],
            "time_seconds": elapsed,
            "total_results": result.total_results,
            "urls_failed": result.urls_failed,
            "field_completeness_pct": field_completeness,
            "optimized_query": result.optimized_query,
            "schema_fields": [f["name"] for f in result.schema_fields],
            "scores": scores,
        }
        all_evaluations.append(eval_entry)

        print(f"  [OK] Scores: relevance={scores.get('relevance')}, "
              f"freshness={scores.get('freshness')}, "
              f"completeness={scores.get('completeness')}, "
              f"overall={scores.get('overall')}")
        if scores.get("issues"):
            print(f"  [WARN] Issues: {scores['issues']}")

    # ── Save full evaluation ──────────────────────────────────────────
    eval_path = "output/eval/evaluation_matrix.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}")
    print(f"  EVALUATION COMPLETE - saved to {eval_path}")
    print(f"{'='*70}")

    # ── Print summary table ───────────────────────────────────────────
    print(f"\n{'-'*120}")
    print(f"{'#':<4} {'Domain':<22} {'Results':<9} {'Failed':<8} "
          f"{'Fields%':<9} {'Relev':<7} {'Fresh':<7} {'Compl':<7} "
          f"{'Schema':<8} {'Depth':<7} {'Overall':<8} {'Time':<6}")
    print(f"{'-'*120}")

    totals = {"relevance": 0, "freshness": 0, "completeness": 0,
              "schema_quality": 0, "content_depth": 0, "overall": 0}

    for e in all_evaluations:
        s = e["scores"]
        print(f"{e['id']:<4} {e['domain']:<22} {e['total_results']:<9} "
              f"{e['urls_failed']:<8} {e['field_completeness_pct']:<9} "
              f"{s.get('relevance','?'):<7} {s.get('freshness','?'):<7} "
              f"{s.get('completeness','?'):<7} {s.get('schema_quality','?'):<8} "
              f"{s.get('content_depth','?'):<7} {s.get('overall','?'):<8} "
              f"{e['time_seconds']}s")
        for k in totals:
            totals[k] += s.get(k, 0)

    n = len(all_evaluations)
    print(f"{'-'*120}")
    print(f"{'AVG':<4} {'':<22} {'':<9} {'':<8} {'':<9} "
          f"{totals['relevance']/n:<7.1f} {totals['freshness']/n:<7.1f} "
          f"{totals['completeness']/n:<7.1f} {totals['schema_quality']/n:<8.1f} "
          f"{totals['content_depth']/n:<7.1f} {totals['overall']/n:<8.1f}")
    print(f"{'-'*120}")


if __name__ == "__main__":
    main()
