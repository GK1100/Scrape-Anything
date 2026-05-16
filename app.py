"""
AI Scraping Agent — Streamlit Frontend.

Provides a polished UI for:
  1. Running new scrape queries via the pipeline
  2. Browsing previously saved scrape results
  3. Viewing structured data in tables, cards, and detail views
"""

import streamlit as st
import json
import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Scraping Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ── Helpers ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def list_output_files() -> list[dict]:
    """Scan the output directory for JSON result files."""
    pattern = os.path.join(OUTPUT_DIR, "scrape_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    result = []
    for fp in files:
        fname = os.path.basename(fp)
        # Parse timestamp from filename: scrape_YYYYMMDD_HHMMSS.json
        try:
            ts_str = fname.replace("scrape_", "").replace(".json", "")
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            label = ts.strftime("%b %d, %Y  •  %I:%M %p")
        except ValueError:
            label = fname
        result.append({"path": fp, "filename": fname, "label": label})
    return result


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_field_icon(field_type: str) -> str:
    icons = {
        "string": "📝", "number": "🔢", "boolean": "✅",
        "array": "📋", "object": "📦",
    }
    return icons.get(field_type, "•")


def compute_completeness(results: list[dict], schema_fields: list[dict]) -> float:
    """Compute overall data completeness percentage."""
    if not results or not schema_fields:
        return 0.0
    total_cells = len(results) * len(schema_fields)
    filled = 0
    for item in results:
        for field in schema_fields:
            val = item.get(field["name"])
            if val is not None and val != "" and val != [] and val != "null":
                filled += 1
    return (filled / total_cells) * 100 if total_cells else 0.0


def run_pipeline(query: str, target_url: str | None = None):
    """Run the scraping pipeline and return the result dict."""
    from src.config import Settings
    from src.services.serper_search import SerperSearchProvider
    from src.services.requests_scraper import RequestsScraper
    from src.services.llm_extractor import LLMContentExtractor
    from src.services.dynamic_schema import DynamicSchemaGenerator
    from src.services.query_analyzer import QueryAnalyzer
    from src.orchestrator.pipeline import ScrapingPipeline

    missing = Settings.validate()
    if missing:
        st.error(f"Missing API keys: {', '.join(missing)}. Check your `.env` file.")
        return None

    pipeline = ScrapingPipeline(
        search_provider=SerperSearchProvider(),
        scraper=RequestsScraper(),
        content_extractor=LLMContentExtractor(),
        schema_generator=DynamicSchemaGenerator(),
        query_analyzer=QueryAnalyzer(),
    )
    result = pipeline.run(query, max_urls=Settings.MAX_URLS, target_url=target_url or None)
    ScrapingPipeline.save_to_json(result)
    return result.model_dump()


# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header gradient */
.hero-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 60%),
                radial-gradient(circle at 70% 30%, rgba(168,85,247,0.1) 0%, transparent 50%);
}
.hero-header h1 {
    color: #fff; font-size: 2rem; font-weight: 800; margin: 0;
    position: relative; z-index: 1;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-header p {
    color: rgba(255,255,255,0.6); font-size: 0.95rem; margin-top: 0.4rem;
    position: relative; z-index: 1;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; padding: 1.2rem 1.4rem; border-radius: 12px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(99,102,241,0.2);
}
.metric-card .label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: rgba(255,255,255,0.5); margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.7rem; font-weight: 700; color: #a5b4fc;
}
.metric-card.success .value { color: #86efac; }
.metric-card.warning .value { color: #fbbf24; }
.metric-card.info .value    { color: #93c5fd; }

/* Schema pill */
.schema-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.35rem 0.85rem; border-radius: 20px; margin: 0.25rem;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
    font-size: 0.82rem; color: #c7d2fe;
}

/* Result card */
.result-card {
    background: linear-gradient(145deg, #1e1b4b, #1a1744);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.result-card:hover {
    border-color: rgba(129,140,248,0.4);
    box-shadow: 0 4px 24px rgba(99,102,241,0.1);
}
.result-card .rc-title {
    font-size: 1.05rem; font-weight: 700; color: #e0e7ff; margin-bottom: 0.4rem;
}
.result-card .rc-source {
    font-size: 0.78rem; color: #818cf8; word-break: break-all; margin-bottom: 0.6rem;
}
.result-card .rc-content {
    font-size: 0.88rem; color: rgba(255,255,255,0.7); line-height: 1.5;
    margin-bottom: 0.8rem;
}
.result-card .rc-field {
    display: inline-flex; align-items: baseline; gap: 4px;
    margin: 0.2rem 0; font-size: 0.82rem;
}
.rc-field .fl { color: rgba(255,255,255,0.45); }
.rc-field .fv { color: #c7d2fe; }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1744 100%) !important;
}
section[data-testid="stSidebar"] .stMarkdown p { color: rgba(255,255,255,0.7); }

/* Badge */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.72rem; font-weight: 600;
}
.badge-true  { background: rgba(34,197,94,0.2); color: #86efac; }
.badge-false { background: rgba(239,68,68,0.2); color: #fca5a5; }
.badge-null  { background: rgba(100,116,139,0.2); color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ── Hero header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔍 AI Scraping Agent</h1>
    <p>Intelligent web scraping powered by LLMs — structured data at your fingertips</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Actions")
    mode = st.radio(
        "Choose mode",
        ["📂 Browse Results", "🚀 New Scrape"],
        label_visibility="collapsed",
    )

    st.divider()

    if mode == "📂 Browse Results":
        st.markdown("### 📁 Saved Results")
        files = list_output_files()
        if not files:
            st.info("No scrape results found in `output/`.")
            st.stop()

        selected_label = st.selectbox(
            "Select a result file",
            [f["label"] for f in files],
            label_visibility="collapsed",
        )
        selected_file = next(f for f in files if f["label"] == selected_label)
        st.caption(f"`{selected_file['filename']}`")

    else:
        st.markdown("### 🔎 Enter Query")
        st.caption("Describe what data you want to scrape.")


# ── Main content ─────────────────────────────────────────────────────────

if mode == "🚀 New Scrape":
    # ── New Scrape tab ───────────────────────────────────────────────
    query = st.text_area(
        "Your scraping query",
        placeholder="e.g. Find top 10 AI startups with name, funding, description, and founder",
        height=100,
    )
    target_url_input = st.text_input(
        "🔗 Target URL (optional)",
        placeholder="e.g. https://example.com/page-to-scrape",
        help="Provide a specific URL to scrape. Leave empty to let the agent search automatically.",
    )
    run_btn = st.button("🚀 Run Scrape", type="primary", use_container_width=True)

    if run_btn and query.strip():
        url_val = target_url_input.strip() if target_url_input else None
        with st.spinner("Running pipeline… this may take a minute."):
            try:
                data = run_pipeline(query.strip(), target_url=url_val)
            except PermissionError as e:
                st.error(f"🚫 {e}")
                data = None
        if data:
            st.success(f"Done! Extracted **{data['total_results']}** items.")
            st.cache_data.clear()
            st.session_state["live_data"] = data
    elif run_btn:
        st.warning("Please enter a query first.")

    if "live_data" in st.session_state:
        data = st.session_state["live_data"]
    else:
        st.info("Enter a query above and click **Run Scrape** to get started.")
        st.stop()

else:
    # ── Browse mode ──────────────────────────────────────────────────
    data = load_json(selected_file["path"])


# ── Display loaded data ─────────────────────────────────────────────────

query_text = data.get("query", "—")
optimized = data.get("optimized_query", "—")
schema_fields = data.get("schema_fields", [])
results = data.get("results", [])

# ── Query info ───────────────────────────────────────────────────────
with st.container():
    st.markdown("#### 💬 Query")
    st.markdown(f"> {query_text}")
    with st.expander("Optimized search query"):
        st.code(optimized, language=None)

# ── Metrics row ──────────────────────────────────────────────────────
completeness = compute_completeness(results, schema_fields)
comp_class = "success" if completeness >= 70 else ("warning" if completeness >= 40 else "")

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card info">
        <div class="label">Results Extracted</div>
        <div class="value">{data.get('total_results', 0)}</div>
    </div>
    <div class="metric-card success">
        <div class="label">URLs Scraped</div>
        <div class="value">{data.get('urls_scraped', 0)}</div>
    </div>
    <div class="metric-card warning">
        <div class="label">URLs Failed</div>
        <div class="value">{data.get('urls_failed', 0)}</div>
    </div>
    <div class="metric-card {comp_class}">
        <div class="label">Data Completeness</div>
        <div class="value">{completeness:.0f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Schema fields ────────────────────────────────────────────────────
with st.expander(f"📐 Dynamic Schema  ({len(schema_fields)} fields)", expanded=False):
    pills_html = ""
    for f in schema_fields:
        icon = get_field_icon(f.get("type", "string"))
        pills_html += f'<span class="schema-pill">{icon} <b>{f["name"]}</b> <span style="opacity:0.5">({f.get("type","str")})</span></span>'
    st.markdown(pills_html, unsafe_allow_html=True)

    schema_df = pd.DataFrame(schema_fields)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)


# ── View toggle ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Results")

if not results:
    st.warning("No results to display.")
    st.stop()

view = st.radio(
    "View mode",
    ["🃏 Cards", "📋 Table", "📄 Raw JSON"],
    horizontal=True,
    label_visibility="collapsed",
)

# All fields are dynamic now
all_fields = schema_fields

if view == "🃏 Cards":
    for idx, item in enumerate(results):
        # Try to find a title-like field for display
        title = (
            item.get("title")
            or item.get("name")
            or item.get("headline")
            or "Untitled"
        )
        source = item.get("source_link", "")

        # Build field HTML for all schema fields (skip source_link since shown separately)
        fields_html = ""
        for f in all_fields:
            if f["name"] == "source_link":
                continue  # shown as the card link already
            val = item.get(f["name"])
            if val is None or val == "null":
                rendered = '<span class="badge badge-null">null</span>'
            elif isinstance(val, bool):
                cls = "badge-true" if val else "badge-false"
                rendered = f'<span class="badge {cls}">{"Yes" if val else "No"}</span>'
            elif isinstance(val, list):
                rendered = ", ".join(str(v) for v in val) if val else '<span class="badge badge-null">empty</span>'
                rendered = f'<span class="fv">{rendered}</span>'
            else:
                rendered = f'<span class="fv">{val}</span>'

            fields_html += f'<div class="rc-field"><span class="fl">{f["name"]}:</span> {rendered}</div><br>'

        st.markdown(f"""
        <div class="result-card">
            <div class="rc-title">{idx + 1}. {title}</div>
            {'<div class="rc-source">🔗 <a href="' + source + '" target="_blank" style="color:#818cf8">' + source + '</a></div>' if source else ''}
            {fields_html}
        </div>
        """, unsafe_allow_html=True)

elif view == "📋 Table":
    # Build a clean DataFrame from results
    display_cols = [f["name"] for f in schema_fields]
    df = pd.DataFrame(results)

    # Reorder columns to match schema, add any missing
    for col in display_cols:
        if col not in df.columns:
            df[col] = None
    df = df[display_cols]

    # Truncate long text for readability
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: (str(v)[:120] + "…") if isinstance(v, str) and len(str(v)) > 120 else v
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "source_link": st.column_config.LinkColumn("Source", display_text="🔗 Link"),
        },
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        csv,
        file_name="scrape_results.csv",
        mime="text/csv",
    )

else:  # Raw JSON
    st.json(data, expanded=2)


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit  •  AI Scraping Agent")
