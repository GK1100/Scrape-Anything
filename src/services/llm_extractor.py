"""
Concrete implementation — LLM Content Extractor.

Architecture: "LLM-as-Manager"
  - The LLM only generates CSS selectors (brain/eyes) once per domain.
  - BeautifulSoup does the actual extraction using those selectors (hands).
  - ~98% token savings compared to sending full page content to the LLM.

Fallback: If selector-based extraction returns sparse results, falls back
to the original LLM-based extraction approach.
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag
from openai import OpenAI

from src.interfaces.content_extractor import IContentExtractor
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("llm_extractor")

# ── Selector generation prompt (compact — ~200 tokens) ──────────────────
_SELECTOR_SYSTEM_PROMPT = """\
You are a CSS selector generation engine. Given a DOM skeleton of a web page
and a list of data fields to extract, return the best CSS selector for each field.

Rules:
- Return ONLY a valid JSON object: {"field_name": "css_selector", ...}
- Each selector must be a valid CSS selector that targets the element
  containing the desired text.
- For "content" or body fields, select the main article/content container
  (e.g., "article", "div.post-body", "div.entry-content"). The selector
  should capture the ENTIRE article body, not individual paragraphs.
- For fields that don't exist on the page, set the selector to null.
- Prefer selectors using class/id over tag-only selectors for reliability.
- No explanation. Only JSON.
"""

# ── Fallback LLM extraction prompt (kept for fallback cases) ────────────
_FALLBACK_SYSTEM_PROMPT = """\
You are a precise data extraction engine. You will be given:
1. Raw text content scraped from a web page.
2. A JSON schema describing the fields to extract.
3. The user's original query for context.

Your task:
- Extract the relevant data from the raw text according to the schema.
- For each field in the schema, provide the best matching value from the text.
- For "content" or body fields: return the FULL, COMPLETE text — include ALL
  paragraphs, details, code snippets, and sections from the page. Do NOT
  summarize, truncate, or abbreviate. Preserve the original content as-is.
- If a field cannot be determined from the text, set it to null.
- If there is a "source_link" field, it is pre-filled; keep it as-is.
- Match the expected data type for each field (string, number, boolean, array).
- Respond with ONLY a valid JSON object, no markdown fences, no explanation.
"""

# Maximum content sent to fallback LLM extraction
_MAX_FALLBACK_CHARS = 50000


class LLMContentExtractor(IContentExtractor):
    """
    Selector-based extractor: LLM generates CSS selectors, BS4 extracts.

    Token usage:
      - Selector generation: ~500 tokens (once per domain)
      - Actual extraction: 0 tokens (pure Python/BS4)
      - Fallback: ~8K tokens (only if selectors fail)
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        # Cache: domain -> {field_name: css_selector}
        self._selector_cache: dict[str, dict[str, str | None]] = {}

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def extract(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """
        Extract structured data from a page.

        Strategy:
          1. If raw_content looks like HTML, try selector-based extraction.
          2. If selectors return too few fields, fallback to LLM extraction.
          3. If raw_content is plain text (trafilatura output), use LLM fallback.
        """
        is_html = bool(raw_content and "<html" in raw_content[:500].lower())

        if is_html:
            result = self._extract_with_selectors(
                raw_content, source_url, schema_fields, user_query,
            )
            if result and self._is_sufficient(result, schema_fields):
                log.info(
                    "Selector extraction succeeded for %s (%d fields filled)",
                    source_url, sum(1 for v in result.values() if v is not None),
                )
                return result
            log.info(
                "Selector extraction insufficient for %s, falling back to LLM",
                source_url,
            )

        # Fallback: LLM-based extraction
        return self._extract_with_llm(raw_content, source_url, schema_fields, user_query)

    def extract_multiple(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """Extract multiple items — delegates to LLM (listing pages are rare)."""
        return self._extract_multiple_with_llm(
            raw_content, source_url, schema_fields, user_query,
        )

    # ================================================================== #
    #  Selector-based extraction (zero tokens for actual extraction)        #
    # ================================================================== #

    def _extract_with_selectors(
        self,
        html: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """Use LLM-generated CSS selectors + BS4 to extract data."""
        domain = urlparse(source_url).netloc.replace("www.", "")

        # Get or generate selectors for this domain
        selectors = self._selector_cache.get(domain)
        if selectors is None:
            selectors = self._generate_selectors(html, schema_fields, user_query, domain)
            if selectors is None:
                return None
            self._selector_cache[domain] = selectors
            log.info("Generated and cached selectors for domain: %s", domain)
        else:
            log.info("Using cached selectors for domain: %s", domain)

        # Apply selectors using BeautifulSoup
        return self._apply_selectors(html, selectors, schema_fields, source_url)

    def _generate_selectors(
        self,
        html: str,
        schema_fields: list[dict],
        user_query: str,
        domain: str,
    ) -> dict[str, str | None] | None:
        """Ask LLM to generate CSS selectors from a compact DOM skeleton."""
        skeleton = self._build_dom_skeleton(html)
        field_descriptions = json.dumps(
            [{"name": f["name"], "type": f["type"], "description": f["description"]}
             for f in schema_fields if f["name"] != "source_link"],
            indent=2,
        )

        user_message = (
            f"## Website domain\n{domain}\n\n"
            f"## User intent\n{user_query}\n\n"
            f"## Fields to extract\n```json\n{field_descriptions}\n```\n\n"
            f"## DOM skeleton\n```html\n{skeleton}\n```"
        )

        log.info(
            "Generating CSS selectors for %s (skeleton: %d chars)",
            domain, len(skeleton),
        )

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.0,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)
            selectors: dict = json.loads(raw)
            log.info("LLM returned selectors: %s", selectors)
            return selectors
        except Exception as exc:
            log.error("Selector generation failed for %s: %s", domain, exc)
            return None

    def _apply_selectors(
        self,
        html: str,
        selectors: dict[str, str | None],
        schema_fields: list[dict],
        source_url: str,
    ) -> dict[str, Any] | None:
        """Use BeautifulSoup to extract data using CSS selectors."""
        soup = BeautifulSoup(html, "lxml")
        result: dict[str, Any] = {}

        for field in schema_fields:
            name = field["name"]
            field_type = field.get("type", "string")

            # source_link is always the URL itself
            if name == "source_link":
                result[name] = source_url
                continue

            selector = selectors.get(name)
            if not selector:
                result[name] = None
                continue

            try:
                # For content/body fields, get ALL text from the container
                if name in ("content", "body", "full_text", "article_text", "description"):
                    element = soup.select_one(selector)
                    if element:
                        # Get full text with paragraph separation
                        result[name] = self._extract_full_text(element)
                    else:
                        result[name] = None

                # For array fields (like tags), get all matching elements
                elif field_type == "array":
                    elements = soup.select(selector)
                    result[name] = [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]
                    if not result[name]:
                        result[name] = None

                # For date fields, try datetime attribute first
                elif "date" in name.lower():
                    element = soup.select_one(selector)
                    if element:
                        # Prefer datetime attribute, fallback to text
                        result[name] = (
                            element.get("datetime")
                            or element.get("content")
                            or element.get_text(strip=True)
                        )
                    else:
                        result[name] = None

                # For all other fields, get text content
                else:
                    element = soup.select_one(selector)
                    if element:
                        result[name] = element.get_text(strip=True)
                    else:
                        result[name] = None

            except Exception as exc:
                log.warning("Selector '%s' failed for field '%s': %s", selector, name, exc)
                result[name] = None

        return result

    @staticmethod
    def _extract_full_text(element: Tag) -> str:
        """
        Extract full text from a container element, preserving paragraph structure.
        Handles deduplication and noise removal for clean output.
        """
        import re as _re

        # ── 1. Remove junk elements ──────────────────────────────────────
        # Remove script/style/nav and common UI noise elements
        junk_selectors = [
            "script", "style", "nav", "footer", "aside",
            "button", "form", "input", "select", "textarea",
            "[class*='share']", "[class*='social']",
            "[class*='subscribe']", "[class*='signup']",
            "[class*='paywall']", "[class*='gate']",
            "[class*='cta']", "[class*='newsletter']",
            "[class*='sidebar']", "[class*='related']",
            "[class*='comments']", "[class*='comment-']",
            "[class*='ad-']", "[class*='advertisement']",
            "[class*='promo']", "[class*='popup']",
            "[class*='modal']", "[class*='cookie']",
            "[class*='nav']", "[class*='menu']",
            "[class*='breadcrumb']",
        ]
        for sel in junk_selectors:
            try:
                for junk in element.select(sel):
                    junk.decompose()
            except Exception:
                pass

        # ── 2. Extract text from block-level elements only ───────────────
        # This avoids the duplication issue where descendants yields both
        # the text node AND the parent element's get_text()
        block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6",
                       "li", "blockquote", "pre", "div", "section",
                       "figcaption", "td", "th", "dt", "dd"}

        paragraphs: list[str] = []
        seen: set[str] = set()

        for child in element.descendants:
            if not hasattr(child, "name") or child.name not in block_tags:
                continue

            # Skip if this element contains other block-level elements
            # (to avoid getting parent + child text = duplication)
            has_block_child = any(
                hasattr(c, "name") and c.name in block_tags
                for c in child.children
            )
            if has_block_child:
                continue

            text = child.get_text(strip=True)
            if not text or len(text) < 3:
                continue

            # Deduplicate
            if text in seen:
                continue
            seen.add(text)
            paragraphs.append(text)

        # Fallback if block-level extraction yields little
        if len("\n".join(paragraphs)) < 200:
            full = element.get_text(separator="\n", strip=True)
            # Split into lines and deduplicate
            lines = full.split("\n")
            paragraphs = []
            seen = set()
            for line in lines:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    paragraphs.append(line)

        # ── 3. Post-processing: remove noise lines ───────────────────────
        noise_patterns = _re.compile(
            r"^("
            r"share|subscribe|sign\s*in|sign\s*up|log\s*in|register|"
            r"upgrade|previous|next|loading\.{0,3}|"
            r"already\s+a\s+(paid\s+)?subscriber\??|"
            r"this\s+post\s+is\s+for\s+paid\s+subscribers?|"
            r"keep\s+reading\s+with\s+a\s+paid|"
            r"💎\s*keep\s+reading|"
            r"\d+\s+(comments?|likes?|shares?|views?|claps?)|"
            r"©\s*\d{4}|all\s+rights\s+reserved|"
            r"terms\s+of\s+service|privacy\s+policy|"
            r"cookie\s+(policy|settings)|"
            r"∙\s*paid|"
            r"\d+\s*$"  # pure numbers (like "228", "41")
            r")$",
            _re.IGNORECASE,
        )

        cleaned: list[str] = []
        for para in paragraphs:
            stripped = para.strip()
            # Skip noise patterns
            if noise_patterns.match(stripped):
                continue
            # Skip very short fragments that are likely UI noise
            if len(stripped) <= 5 and not stripped[0].isalpha():
                continue
            cleaned.append(stripped)

        return "\n\n".join(cleaned)

    @staticmethod
    def _build_dom_skeleton(html: str, max_chars: int = 3000) -> str:
        """
        Build a compact DOM skeleton showing structure + first few chars of text.
        This is what the LLM sees — just the page structure, NOT the content.

        Typical size: 1000–3000 chars vs 50,000+ for full page.
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "noscript", "svg", "link", "meta"]):
            tag.decompose()

        skeleton_parts: list[str] = []
        char_count = 0

        def _walk(element: Tag, depth: int = 0) -> None:
            nonlocal char_count
            if char_count >= max_chars:
                return

            indent = "  " * depth
            tag_name = element.name

            # Build attribute string (only class, id, role, datetime, type)
            attrs = []
            for attr_name in ("class", "id", "role", "datetime", "type", "name"):
                val = element.get(attr_name)
                if val:
                    if isinstance(val, list):
                        val = " ".join(val)
                    attrs.append(f'{attr_name}="{val}"')
            attr_str = " " + " ".join(attrs) if attrs else ""

            # Get direct text (not from children), truncated
            direct_text = element.find(string=True, recursive=False)
            text_preview = ""
            if direct_text and direct_text.strip():
                text_preview = direct_text.strip()[:40]
                if len(direct_text.strip()) > 40:
                    text_preview += "..."

            # Count children of same type (to show repeated patterns)
            child_tags = [c for c in element.children if isinstance(c, Tag)]
            child_count = len(child_tags)

            line = f"{indent}<{tag_name}{attr_str}>"
            if text_preview:
                line += text_preview
            if child_count > 3:
                # Summarize repeated children instead of listing all
                tag_counts: dict[str, int] = {}
                for c in child_tags:
                    tag_counts[c.name] = tag_counts.get(c.name, 0) + 1
                counts_str = ", ".join(f"{n}×{k}" for k, n in tag_counts.items() if n > 1)
                if counts_str:
                    line += f" [{counts_str}]"

            skeleton_parts.append(line)
            char_count += len(line)

            # Recurse into children (limit breadth)
            for child in child_tags[:6]:
                if char_count >= max_chars:
                    break
                _walk(child, depth + 1)

        body = soup.find("body")
        if body and isinstance(body, Tag):
            _walk(body, 0)
        else:
            # Fallback: walk the whole document
            for tag in soup.find_all(True)[:50]:
                if isinstance(tag, Tag) and tag.parent and tag.parent.name in ("[document]", "html"):
                    _walk(tag, 0)

        return "\n".join(skeleton_parts)

    @staticmethod
    def _is_sufficient(result: dict[str, Any], schema_fields: list[dict]) -> bool:
        """Check if selector extraction returned enough non-null fields."""
        if not result:
            return False

        # Exclude source_link from the count
        fields = [f["name"] for f in schema_fields if f["name"] != "source_link"]
        if not fields:
            return True

        filled = sum(1 for f in fields if result.get(f) is not None)
        fill_ratio = filled / len(fields)

        # Also check: if content field exists, it must have substantial text
        content_field = result.get("content") or result.get("body") or result.get("full_text")
        if content_field is not None and len(str(content_field)) < 100:
            return False

        return fill_ratio >= 0.40

    # ================================================================== #
    #  LLM fallback (used when selectors fail or content is plain text)    #
    # ================================================================== #

    def _extract_with_llm(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> dict[str, Any] | None:
        """Fallback: send content to LLM for extraction (token-heavy)."""
        truncated = raw_content[:_MAX_FALLBACK_CHARS]
        schema_description = json.dumps(schema_fields, indent=2)

        user_message = (
            f"## User Query\n{user_query}\n\n"
            f"## Schema\n```json\n{schema_description}\n```\n\n"
            f"## Source URL\n{source_url}\n\n"
            f"## Raw Page Content\n{truncated}"
        )

        log.info("LLM fallback extraction from %s (%d chars)", source_url, len(truncated))

        try:
            response = self._client.chat.completions.create(
                model=settings.OPENROUTER_MODEL,
                temperature=0.1,
                max_tokens=10000,
                messages=[
                    {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)
            result: dict[str, Any] = json.loads(raw)
            if any(f["name"] == "source_link" for f in schema_fields):
                result["source_link"] = source_url
            log.info("LLM fallback extracted %d fields", len(result))
            return result
        except Exception as exc:
            log.error("LLM fallback extraction failed for %s: %s", source_url, exc)
            return None

    def _extract_multiple_with_llm(
        self,
        raw_content: str,
        source_url: str,
        schema_fields: list[dict],
        user_query: str,
    ) -> list[dict[str, Any]]:
        """Extract multiple items from a listing page (LLM-based)."""
        _MULTI_PROMPT = """\
You are a precise data extraction engine. The web page MAY contain MULTIPLE
separate items (e.g., a list of products, news articles, job postings, etc.).

CRITICAL DISTINCTION:
- If the page is a SINGLE article/blog post with multiple sections/headings,
  treat it as ONE item. Do NOT split sections into separate items.
- If the page is a LISTING page with genuinely SEPARATE items, extract each
  as a separate item.

Your task:
- For "content" or body fields: return the FULL, COMPLETE text. Do NOT
  summarize, truncate, or abbreviate.
- If a field cannot be determined for an item, set it to null.
- If there is a "source_link" field, set it to the provided URL for all items.
- Skip items that are clearly irrelevant (ads, navigation).
- Respond with ONLY a JSON array of objects, no markdown fences.
- Extract up to 15 items maximum.
"""
        truncated = raw_content[:_MAX_FALLBACK_CHARS]
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
                max_tokens=12000,
                messages=[
                    {"role": "system", "content": _MULTI_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)
            items: list[dict[str, Any]] = json.loads(raw)

            if not isinstance(items, list):
                items = [items]

            has_source_link = any(f["name"] == "source_link" for f in schema_fields)
            if has_source_link:
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
