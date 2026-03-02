"""Custom tools for the Shopping Advisor crew."""

import logging
import os
import json
import requests
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logging.basicConfig(
    filename="tool_urls.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serper_search(query: str, num: int = 8, site: str = None) -> dict:
    """Execute a search via the Serper.dev Google Search API."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("SERPER_API_KEY environment variable is not set.")

    search_query = f"site:{site} {query}" if site else query
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": search_query, "num": num}
    response = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def _format_results(data: dict, max_items: int = 8, tool_name: str = "unknown") -> str:
    """Convert Serper JSON results into a readable text block."""
    lines = []

    # Answer box (featured snippet)
    if answer := data.get("answerBox", {}).get("answer"):
        lines.append(f"[Featured Answer] {answer}\n")

    # Organic results
    for item in data.get("organic", [])[:max_items]:
        url = item.get("link", "N/A")
        logging.info("[%s] TOOL_URL: %s", tool_name, url)
        lines.append(f"Title   : {item.get('title', 'N/A')}")
        lines.append(f"URL     : {url}")
        lines.append(f"Snippet : {item.get('snippet', 'N/A')}")
        lines.append("-" * 60)

    return "\n".join(lines) if lines else "No results found."


# ---------------------------------------------------------------------------
# Input schema shared by most tools
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to execute.")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ReviewSearchTool(BaseTool):
    """Search expert review sites (RTINGS, TechRadar, Wirecutter, etc.)."""

    name: str = "Expert Review Search"
    description: str = (
        "Search for expert reviews, test results, and ratings for a product. "
        "Returns review snippets and links from authoritative review websites."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        data = _serper_search(f"{query} review rating pros cons best", num=8)
        return _format_results(data, tool_name="ReviewSearch")


class RedditSearchTool(BaseTool):
    """Search Reddit for community opinions and recommendations."""

    name: str = "Reddit Community Search"
    description: str = (
        "Search Reddit for real-user discussions, recommendations, and opinions "
        "about a product. Returns Reddit thread titles, URLs and comment snippets."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        data = _serper_search(f"{query} recommendations opinions", num=8, site="reddit.com")
        return _format_results(data, tool_name="RedditSearch")


class AllegroSearchTool(BaseTool):
    """Search Allegro.pl for product listings and current prices."""

    name: str = "Allegro.pl Offer Search"
    description: str = (
        "Search Allegro.pl (major Polish e-commerce marketplace) for product listings. "
        "Returns product names, prices and direct Allegro.pl links."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        # Allegro uses Polish; add buying-intent terms
        data = _serper_search(f"{query} kup cena oferta", num=8, site="allegro.pl")
        return _format_results(data, tool_name="AllegroSearch")


class AliExpressSearchTool(BaseTool):
    """Search AliExpress for product listings and prices."""

    name: str = "AliExpress Offer Search"
    description: str = (
        "Search AliExpress for product listings and prices. "
        "Returns product names, prices and direct AliExpress links."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        data = _serper_search(f"{query} buy price free shipping", num=8, site="aliexpress.com")
        return _format_results(data, tool_name="AliExpressSearch")


class WebPageReaderTool(BaseTool):
    """Fetch and extract readable text from a specific URL."""

    name: str = "Web Page Reader"
    description: str = (
        "Fetch the content of a specific URL and return readable text. "
        "Useful for reading full review articles or product detail pages."
    )

    class _Input(BaseModel):
        url: str = Field(..., description="Full URL to fetch and read.")

    args_schema: Type[BaseModel] = _Input

    def _run(self, url: str) -> str:
        logging.info("[WebPageReader] FETCH_URL: %s", url)
        try:
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Keep first 4 000 characters to stay within context limits
            return text[:4000]
        except Exception as exc:
            return f"Could not fetch {url}: {exc}"
