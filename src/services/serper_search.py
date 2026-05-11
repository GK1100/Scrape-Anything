"""
Concrete implementation — Serper.dev Google Search.

Single Responsibility: convert a user query into a ranked list of URLs.
Liskov Substitution: fully satisfies ISearchProvider contract.

Fix P0: Filters out non-article URLs (YouTube, social media, etc.)
"""

import re
import json as _json
from urllib.parse import urlparse
import requests

from src.interfaces.search_provider import ISearchProvider
from src.config import settings
from src.utils.logger import get_logger

log = get_logger("serper_search")

# Domains/patterns that almost never yield scrapable text content
_SKIP_DOMAINS = {
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com",
    "pinterest.com",
    "play.google.com", "apps.apple.com",
}

# URL path patterns that indicate non-article pages
_SKIP_PATH_PATTERNS = [
    r"/video/",
    r"/watch\?",
    r"/playlist",
    r"/shorts/",
    r"/reel/",
    r"/status/",
]


def _is_scrapable_url(url: str) -> bool:
    """Return True if the URL is likely to yield article content."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "").lower()

        # Check blocked domains
        if any(blocked in domain for blocked in _SKIP_DOMAINS):
            return False

        # Check path patterns
        for pattern in _SKIP_PATH_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True
    except Exception:
        return True


class SerperSearchProvider(ISearchProvider):
    """Google Search via the Serper.dev REST API."""

    API_URL = "https://google.serper.dev/search"

    # site: TLD → Serper gl (geo-location) code mapping
    _SITE_TO_GL = {
        "in": "in", ".in": "in",
        "uk": "uk", ".uk": "uk", "co.uk": "uk",
        "au": "au", ".au": "au",
        "ca": "ca", ".ca": "ca",
        "de": "de", ".de": "de",
        "fr": "fr", ".fr": "fr",
        "jp": "jp", ".jp": "jp",
    }

    @classmethod
    def _sanitize_query(cls, query: str) -> tuple[str, str | None]:
        """
        Strip site: operators from the query and convert them to a
        Serper `gl` (geo-location) parameter.

        Returns (cleaned_query, gl_code_or_None).
        """
        gl_code = None
        # Match patterns like site:in, site:.in, site:co.uk
        site_match = re.search(r'\bsite:([\.\w]+)', query, re.IGNORECASE)
        if site_match:
            site_val = site_match.group(1).lower().strip(".")
            gl_code = cls._SITE_TO_GL.get(site_val)
            # Remove the site: operator from query text
            query = re.sub(r'\bsite:[\w.]+\s*', '', query, flags=re.IGNORECASE).strip()
            query = re.sub(r'\s+', ' ', query)  # collapse extra spaces
            if gl_code:
                log.info("Converted site:%s → gl=%s", site_val, gl_code)
            else:
                log.warning("Unknown site: TLD '%s', stripping operator", site_val)
        return query, gl_code

    def search(
        self,
        query: str,
        num_results: int = 10,
        time_filter: str | None = None,
    ) -> list[dict]:
        """
        Execute a Google search and return top results.

        Automatically filters out non-scrapable URLs (YouTube, social media, etc.)
        and requests extra results to compensate.
        """
        # Sanitize: convert site: operators → gl parameter
        query, gl_code = self._sanitize_query(query)

        headers = {
            "X-API-KEY": settings.SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        # Request more to compensate for filtering
        request_count = min(num_results * 2, 30)
        payload: dict = {"q": query, "num": request_count}

        if gl_code:
            payload["gl"] = gl_code
        if time_filter:
            payload["tbs"] = time_filter
            log.info(
                "Searching Google for: %s (top %d, time_filter=%s, gl=%s)",
                query, num_results, time_filter, gl_code,
            )
        else:
            log.info("Searching Google for: %s (top %d, gl=%s)", query, num_results, gl_code)

        try:
            resp = requests.post(
                self.API_URL,
                json=payload,
                headers=headers,
                timeout=settings.REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.error("Serper API error: %s", exc)
            # Log full details for debugging
            if hasattr(exc, 'response') and exc.response is not None:
                log.error("Response body: %s", exc.response.text[:500])
            log.error("Payload was: %s", _json.dumps(payload, ensure_ascii=False))
            return []

        results = []
        skipped = 0
        for item in data.get("organic", []):
            link = item.get("link", "")

            # Fix P0: Filter non-article URLs
            if not _is_scrapable_url(link):
                skipped += 1
                log.info("Filtered out non-article URL: %s", link)
                continue

            results.append(
                {
                    "title": item.get("title", ""),
                    "link": link,
                    "snippet": item.get("snippet", ""),
                }
            )
            if len(results) >= num_results:
                break

        log.info("Found %d results (filtered %d non-article URLs)", len(results), skipped)
        return results
