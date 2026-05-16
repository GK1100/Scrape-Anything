"""
Concrete implementation — HTTP-based web scraper.

Uses requests + trafilatura for clean text extraction.
Falls back to BeautifulSoup, then to Playwright headless browser.

Fix P0: Structural index page detection (not just char count).
Fix P1: Playwright headless browser fallback for 403/JS-rendered sites.
"""

import re
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from src.interfaces.scraper import IScraper
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("requests_scraper")

# ── Playwright lazy-loaded singleton ──────────────────────────────────
_playwright_instance = None
_playwright_browser = None


def _get_playwright_page():
    """Lazily initialise Playwright and return a fresh page."""
    global _playwright_instance, _playwright_browser
    try:
        from playwright.sync_api import sync_playwright
        if _playwright_instance is None:
            _playwright_instance = sync_playwright().start()
            _playwright_browser = _playwright_instance.chromium.launch(headless=True)
            log.info("Playwright headless browser initialised")
        page = _playwright_browser.new_page()
        page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.5",
        })
        return page
    except Exception as exc:
        log.warning("Playwright not available: %s", exc)
        return None


# Rotate through user-agents on retries
_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Safari/605.1.15"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
]

_BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Domains known to hard-block even headless browsers
_HARD_BLOCKED_DOMAINS = {
    "wsj.com", "ft.com", "bloomberg.com",
}


# Persistent session with connection pooling for faster repeated requests
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=0,  # we handle retries manually via UA rotation
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


class RequestsScraper(IScraper):
    """HTTP scraper with trafilatura extraction, UA rotation, and Playwright fallback."""

    def scrape(self, url: str) -> str | None:
        """Fetch URL and extract main text content."""
        domain = urlparse(url).netloc.replace("www.", "")

        # Skip known hard-blocked paywalled sites
        if any(blocked in domain for blocked in _HARD_BLOCKED_DOMAINS):
            log.warning("Skipping hard-blocked domain: %s", domain)
            return None

        log.info("Scraping: %s", url)

        # Strategy 1: requests with UA rotation
        html = self._fetch_with_retry(url)

        # Strategy 2: Playwright fallback if requests failed (403/401)
        if html is None:
            log.info("Trying Playwright fallback for: %s", url)
            html = self._fetch_with_playwright(url)

        if not html:
            return None

        return self._extract_text(html, url)

    def fetch_html(self, url: str) -> str | None:
        """
        Fetch the raw HTML of a URL without text extraction.

        This preserves the full HTML structure (links, cards, tiles, etc.)
        so downstream code can extract clickable elements.
        """
        domain = urlparse(url).netloc.replace("www.", "")
        if any(blocked in domain for blocked in _HARD_BLOCKED_DOMAINS):
            log.warning("Skipping hard-blocked domain: %s", domain)
            return None

        log.info("Fetching raw HTML: %s", url)
        html = self._fetch_with_retry(url)
        if html is None:
            log.info("Trying Playwright fallback for raw HTML: %s", url)
            html = self._fetch_with_playwright(url)
        return html

    def extract_article_links(self, url: str, html: str | None = None) -> list[str]:
        """
        Extract article links from an index/category page.
        Prioritises links that look like article URLs over navigation links.
        """
        if html is None:
            html = self._fetch_with_retry(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        base_domain = urlparse(url).netloc
        links = []

        for a_tag in soup.find_all("a", href=True):
            href = urljoin(url, a_tag["href"])
            parsed = urlparse(href)

            # Only same domain
            if parsed.netloc != base_domain:
                continue

            path = parsed.path.rstrip("/")
            segments = [s for s in path.split("/") if s]

            # Heuristic: article URLs have 2+ path segments and look like slugs
            if len(segments) >= 2 and self._looks_like_content_url(path):
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if clean not in links and clean != url:
                    links.append(clean)

        log.info("Extracted %d article links from index page %s", len(links), url)
        return links[:10]

    def extract_all_links(self, url: str, html: str | None = None) -> list[dict]:
        """
        Extract ALL clickable/navigable elements from a page.

        Goes beyond simple <a> tags to capture the full range of clickable
        UI elements on modern web pages:
          - Links / Hyperlinks (<a> tags)
          - Cards / Tiles (elements with data-href, data-url, data-link)
          - Onclick handlers with URLs (onclick="window.location=...")
          - Elements with role="link"
          - Clickable list items (<li> wrapping <a>)
          - Interactive containers / CTA components

        Returns a list of {"url": str, "text": str} dicts so the LLM can
        decide which links are relevant to the user's query.
        """
        if html is None:
            html = self._fetch_with_retry(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        links: list[dict] = []
        seen: set[str] = set()

        # Skip obvious non-content anchor text patterns
        _skip_texts = {
            "", "home", "about", "contact", "login", "signup", "sign up",
            "register", "privacy", "terms", "cookie", "menu", "close",
            "search", "subscribe", "newsletter", "log in", "sign in",
            "×", "x", "#", "...", "more", "skip to content",
        }

        def _add_link(raw_url: str, text: str) -> None:
            """Normalise, deduplicate, and add a link."""
            if not raw_url:
                return
            resolved = urljoin(url, raw_url)
            parsed = urlparse(resolved)
            if parsed.scheme not in ("http", "https"):
                return
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            if clean in seen or clean == url.rstrip("/"):
                return
            text = (text or "").strip()
            if text.lower() in _skip_texts:
                return
            seen.add(clean)
            links.append({"url": clean, "text": text})

        # ── 1. Standard <a href="..."> tags ────────────────────────────────
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            anchor_text = a_tag.get_text(strip=True)
            _add_link(href, anchor_text)

        # ── 2. Elements with data-href / data-url / data-link attrs ───────
        #    (Cards, tiles, clickable containers)
        for attr in ("data-href", "data-url", "data-link", "data-target-url"):
            for el in soup.find_all(attrs={attr: True}):
                href = el[attr]
                text = el.get_text(strip=True)[:120]  # cap long card text
                _add_link(href, text)

        # ── 3. onclick handlers containing URLs ───────────────────────────
        _url_in_js = re.compile(
            r"""(?:window\.location|location\.href|window\.open)\s*[=(]\s*['"]([^'"]+)['"]"""
        )
        for el in soup.find_all(attrs={"onclick": True}):
            match = _url_in_js.search(el["onclick"])
            if match:
                text = el.get_text(strip=True)[:120]
                _add_link(match.group(1), text)

        # ── 4. Elements with role="link" ───────────────────────────────────
        for el in soup.find_all(attrs={"role": "link"}):
            # Check for a nested <a> first
            nested_a = el.find("a", href=True)
            if nested_a:
                _add_link(nested_a["href"], el.get_text(strip=True)[:120])
                continue
            # Check data attrs
            for attr in ("data-href", "data-url", "href"):
                val = el.get(attr)
                if val:
                    _add_link(val, el.get_text(strip=True)[:120])
                    break

        log.info("Extracted %d clickable links from %s", len(links), url)
        return links

    def is_index_page(self, content: str, url: str = "") -> bool:
        """
        Fix P0: Structural detection of index/listing pages.
        Uses multiple signals instead of just char count.
        """
        if not content:
            return True

        # Content-rich pages are NEVER index pages — they have useful content
        # even if they look "list-like" (e.g. product reviews with specs)
        if len(content) > 1500:
            return False

        # Signal 1: Very short content is likely an index/nav page
        if len(content) < 300:
            return True

        lines = content.strip().split("\n")
        non_empty_lines = [l for l in lines if l.strip()]

        # Signal 2: Ratio of short lines (< 80 chars) to total.
        # Index pages have many short headlines; articles have paragraphs.
        if len(non_empty_lines) > 5:
            short_lines = sum(1 for l in non_empty_lines if len(l.strip()) < 80)
            short_ratio = short_lines / len(non_empty_lines)
            if short_ratio > 0.85:
                return True

        # Signal 3: Very few long paragraphs (> 150 chars).
        # Articles typically have multiple meaty paragraphs.
        long_paragraphs = sum(1 for l in non_empty_lines if len(l.strip()) > 150)
        if long_paragraphs < 2 and len(content) < 800:
            return True

        # Signal 4: URL path analysis — known listing patterns
        if url:
            path = urlparse(url).path.rstrip("/").lower()
            listing_patterns = [
                r"^/category/", r"^/tag/", r"^/topics?/?$",
                r"^/section/", r"^/archive", r"^/hub/",
                r"^/$",  # Homepages
            ]
            if any(re.match(p, path) for p in listing_patterns):
                # But only if content is also thin
                if len(content) < 2000:
                    return True

        return False

    # ------------------------------------------------------------------ #
    #  Private: Fetching strategies                                        #
    # ------------------------------------------------------------------ #

    def _fetch_with_retry(self, url: str) -> str | None:
        """Try fetching with multiple user-agents on 401/403."""
        for i, ua in enumerate(_USER_AGENTS):
            headers = {**_BASE_HEADERS, "User-Agent": ua}
            try:
                resp = _session.get(
                    url, headers=headers, timeout=settings.REQUEST_TIMEOUT,
                    allow_redirects=True,
                )
                if resp.status_code in (401, 403):
                    if i < len(_USER_AGENTS) - 1:
                        log.info("Got %d, retrying with different UA...", resp.status_code)
                        continue
                    log.warning("All UAs returned %d for %s", resp.status_code, url)
                    return None
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                if i < len(_USER_AGENTS) - 1:
                    continue
                log.warning("Failed to fetch %s after %d attempts: %s", url, i + 1, exc)
                return None
        return None

    def _fetch_with_playwright(self, url: str) -> str | None:
        """Fix P1: Headless browser fallback for JS-rendered / 403 pages."""
        page = _get_playwright_page()
        if page is None:
            return None
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1000)  # Let JS render
            html = page.content()
            log.info("Playwright fetched %d chars from %s", len(html), url)
            return html
        except Exception as exc:
            log.warning("Playwright failed for %s: %s", url, exc)
            return None
        finally:
            try:
                page.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Private: Content extraction                                         #
    # ------------------------------------------------------------------ #

    def _extract_text(self, html: str, url: str) -> str | None:
        """Extract main text from HTML using trafilatura with BS4 fallback."""
        # Primary: trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        if text and len(text.strip()) > 100:
            log.info("Extracted %d chars via trafilatura", len(text))
            return text.strip()

        # Fallback: BeautifulSoup
        log.info("Trafilatura returned little content; falling back to BS4")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if text and len(text.strip()) > 100:
            log.info("Extracted %d chars via BS4", len(text))
            return text.strip()

        log.warning("Could not extract meaningful content from %s", url)
        return None

    @staticmethod
    def _looks_like_content_url(path: str) -> bool:
        """Heuristic: does this URL path look like content vs. navigation/infrastructure."""
        # Block known non-content paths (navigation, auth, infrastructure)
        non_content_patterns = [
            r"^/category/", r"^/tag/", r"^/topics?/",
            r"^/about", r"^/contact", r"^/privacy",
            r"^/terms", r"^/help", r"^/faq",
            r"^/login", r"^/signup", r"^/register",
            r"^/search", r"^/account", r"/press/?$",
            r"/copyright/?$", r"/advertis",
            # E-commerce navigation
            r"/gp/cart", r"/gp/site-directory", r"/gp/bestsellers",
            r"/gp/help", r"/gp/css", r"/gp/new-releases",
            r"/gp/goldbox", r"/gp/redirect",
            r"/wishlist", r"/order", r"/checkout",
            r"/mobile-phones/b/?$", r"^/b/\d+$",
            r"/customer-preferences", r"/ref=",
            # Generic infrastructure
            r"/sitemap", r"/feed/?$", r"/rss/?$",
            r"/cdn-cgi/", r"/api/", r"/wp-admin",
            r"/wp-login", r"/wp-json",
            r"^/page/\d+/?$",  # Pagination pages
        ]
        return not any(re.search(p, path, re.IGNORECASE) for p in non_content_patterns)
