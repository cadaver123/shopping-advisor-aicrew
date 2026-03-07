"""Custom tools for the Shopping Advisor crew."""

import atexit
import logging
import math
import os
import re
import requests
import yaml
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger("shopping_advisor.tools")


# ---------------------------------------------------------------------------
# Shopping Search Tool — Serper Shopping API (returns structured product data)
# ---------------------------------------------------------------------------

class ShoppingSearchTool(BaseTool):
    name: str = "Shopping Search"
    description: str = (
        "Search Google Shopping for products matching a query. "
        "Returns up to 50 results sorted by user rating (best first), "
        "with product name, rating, and review count. "
        "Input: a product search query including any budget constraint "
        "(e.g. 'gaming mouse under 200 PLN', 'best headphones below 300 USD')."
    )

    class _Input(BaseModel):
        query: str = Field(..., description="Product search query with budget if applicable.")

    args_schema: Type[BaseModel] = _Input

    def _run(self, query: str) -> str:
        api_key = os.environ.get("SERPER_API_KEY", "")
        try:
            resp = requests.get(
                "https://google.serper.dev/shopping",
                params={"q": query, "num": 50, "apiKey": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"[ShoppingSearch] Request failed: {exc}"

        items = data.get("shopping", [])
        if not items:
            return "[ShoppingSearch] No results found."

        # Keep only items with a rating; sort by confidence score to penalise
        # products with very few reviews (rating * log(1 + review_count))
        rated = [i for i in items if i.get("rating")]
        rated.sort(
            key=lambda i: i["rating"] * math.log1p(i.get("ratingCount", 0)),
            reverse=True,
        )

        rows = []
        for i, item in enumerate(rated, 1):
            title = item.get("title", "?")
            rating = item["rating"]
            count = item.get("ratingCount", 0)
            price = item.get("price", "—")
            rows.append(f"| {i} | {title} | {price} | {rating} | {count} |")

        logger.info("[ShoppingSearch] %d rated results for: %s", len(rated), query)
        if not rows:
            return "[ShoppingSearch] No rated products found."

        header = "| # | Product | Price | Rating | Reviews |\n|---|---------|-------|--------|---------|"
        return header + "\n" + "\n".join(rows)

# ---------------------------------------------------------------------------
# Playwright browser singleton
# ---------------------------------------------------------------------------

_playwright = None
_browser = None


def _get_browser():
    """Return the cached Playwright Browser, launching it on first call."""
    global _playwright, _browser
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright().__enter__()
        _browser = _playwright.chromium.launch(headless=True)
        atexit.register(_shutdown_browser)
        logger.info("[WebPageReader] Chromium browser started.")
    return _browser


def _shutdown_browser():
    global _playwright, _browser
    if _browser is not None:
        try:
            _browser.close()
        except Exception:
            pass
        _browser = None
    if _playwright is not None:
        try:
            _playwright.__exit__(None, None, None)
        except Exception:
            pass
        _playwright = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEBPAGE_CFG = yaml.safe_load(
    (Path(__file__).parent / "config" / "webpage_reader.yaml").read_text(encoding="utf-8")
)["webpage_reader"]


class WebPageReaderTool(BaseTool):
    name: str = "Web Page Reader"
    description: str = (
        "Fetch the content of a specific URL and return clean Markdown text. "
        "Uses a real headless browser so JavaScript-rendered pages work."
    )

    class _Input(BaseModel):
        url: str = Field(..., description="Full URL to fetch and read.")

    args_schema: Type[BaseModel] = _Input

    def _fetch_html(self, url: str) -> str:
        browser = _get_browser()
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        page.route("**/*", lambda route, req: route.abort()
                   if req.resource_type in {"image", "stylesheet", "font", "media", "other"}
                   else route.continue_())
        try:
            page.goto(url, wait_until="load", timeout=5000)
        except Exception:
            pass
        html = page.content()
        context.close()
        return html

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        import trafilatura
        from markdownify import markdownify
        md = trafilatura.extract(html, output_format="markdown",
                                 include_formatting=True, include_links=False, include_images=False)
        return md.strip() if md and md.strip() else markdownify(html, heading_style="ATX", strip=["script", "style"])

    @staticmethod
    def _split_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split *text* into overlapping chunks, preferring paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Prefer cutting at a paragraph boundary in the second half of the window
            if end < len(text):
                boundary = text.rfind("\n\n", start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - chunk_overlap
        return chunks

    def _summarize_chunk(self, chunk: str, api_key: str) -> str:
        """Send a single chunk to the summarizer LLM. Returns the chunk on failure."""
        try:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": _WEBPAGE_CFG["model"],
                    "temperature": _WEBPAGE_CFG["temperature"],
                    "messages": [
                        {"role": "system", "content": _WEBPAGE_CFG.get("system_prompt", "")},
                        {"role": "user", "content": chunk},
                    ],
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or chunk
        except Exception as exc:
            logger.warning("[WebPageReader] chunk summarization failed (%s) — keeping original.", exc)
            return chunk

    def _summarize(self, text: str) -> str:
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            return text

        chunk_size = _WEBPAGE_CFG.get("chunk_size", 8000)
        chunk_overlap = _WEBPAGE_CFG.get("chunk_overlap", 800)
        chunks = self._split_chunks(text, chunk_size, chunk_overlap)

        logger.info("[WebPageReader] summarizing %d chunk(s) with %s",
                    len(chunks), _WEBPAGE_CFG["model"])

        results = [self._summarize_chunk(chunk, api_key) for chunk in chunks]
        return "\n\n".join(results)

    _FAKE_DOMAINS = {"example.com", "example.org", "example.net", "test.com", "localhost"}

    def _run(self, url: str) -> str:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lstrip("www.")
        if domain in self._FAKE_DOMAINS:
            msg = f"[WebPageReader] REJECTED constructed URL: {url}. Only use URLs returned verbatim by Web Search."
            logger.warning(msg)
            return msg

        logger.info("[WebPageReader] FETCH_URL: %s", url)
        try:
            html = self._fetch_html(url)
        except Exception as exc:
            return f"Could not fetch {url}: {exc}"

        cleaned = re.sub(r"\n{3,}", "\n\n", self._html_to_markdown(html)).strip()
        before = len(cleaned)
        result = self._summarize(cleaned)
        logger.info("[WebPageReader] %d → %d chars (%.0f%%): %s",
                    before, len(result), (len(result) / before * 100) if before else 0, url)

        max_chars = _WEBPAGE_CFG.get("max_chars", 20_000)
        if len(result) > max_chars:
            result = result[:max_chars] + f"\n\n---\n*Content truncated at {max_chars:,} characters.*"

        return result
