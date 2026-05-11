# 🔍 AI Scraping Agent

An intelligent web scraping agent that takes natural-language queries, finds relevant URLs, scrapes them, and outputs structured JSON with a **dynamically generated schema**.

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Edit the `.env` file and add your keys:

| Key | Where to get it | Purpose |
|-----|-----------------|---------|
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) (free: 2,500 queries) | Google Search |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | LLM (query analysis, schema gen, extraction) |

### 3. Run it
```bash
# Interactive mode
python -m src.main

# With inline query
python -m src.main "Give me top 10 AI news in the past 2 days"
```

## 📂 Output
Results are saved to `output/scrape_YYYYMMDD_HHMMSS.json`.

### Example output structure:
```json
{
  "query": "Give me top 10 AI news in the past 2 days",
  "optimized_query": "top AI news 2024",
  "schema_fields": [
    {"name": "source_link", "type": "string", "description": "..."},
    {"name": "title", "type": "string", "description": "..."},
    {"name": "main_content", "type": "string", "description": "..."},
    {"name": "published_date", "type": "string", "description": "..."},
    {"name": "author", "type": "string", "description": "..."},
    {"name": "category", "type": "string", "description": "..."}
  ],
  "results": [
    {
      "source_link": "https://example.com/article",
      "title": "OpenAI Releases GPT-5",
      "main_content": "OpenAI announced...",
      "published_date": "2024-05-08",
      "author": "John Doe",
      "category": "AI Research"
    }
  ],
  "total_results": 5,
  "urls_scraped": 5,
  "urls_failed": 0
}
```

## 🏗 Architecture (SOLID)

```
src/
├── interfaces/           # Abstract contracts (D, I, L)
│   ├── search_provider   # ISearchProvider
│   ├── scraper           # IScraper
│   ├── content_extractor # IContentExtractor
│   └── schema_generator  # ISchemaGenerator
├── services/             # Concrete implementations (S, O)
│   ├── serper_search     # Google search via Serper
│   ├── requests_scraper  # HTTP scraper (trafilatura + BS4)
│   ├── llm_extractor     # GPT content extraction
│   ├── dynamic_schema    # Dynamic schema generation
│   └── query_analyzer    # Query optimization
├── models/               # Pydantic data models
├── orchestrator/         # Pipeline (DI wiring)
├── config.py             # Environment config
└── main.py               # CLI entry point
```

| SOLID Principle | How It's Applied |
|---|---|
| **S** – Single Responsibility | Each service handles exactly one concern |
| **O** – Open/Closed | Add new schema fields or swap providers without modifying core |
| **L** – Liskov Substitution | All implementations satisfy their interface contracts |
| **I** – Interface Segregation | Narrow, focused interfaces per concern |
| **D** – Dependency Inversion | Pipeline depends on abstractions, not concretions |

## ⚙️ Optional Settings (.env)

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | LLM model (OpenRouter format) |
| `MAX_URLS` | `5` | Max URLs to scrape per query |
| `REQUEST_TIMEOUT` | `15` | HTTP timeout in seconds |
