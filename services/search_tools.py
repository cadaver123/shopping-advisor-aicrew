"""URL domain blocklist and Serper search tool."""

import logging
import os
import requests
import yaml
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "search_tools.yaml").read_text(encoding="utf-8")
)
_DOMAIN_BLOCKLIST = set(_CFG["source_discovery"]["domain_blocklist"])


def _is_blocked(domain: str) -> bool:
    """Return True if *domain* matches any blocklisted domain or its subdomain."""
    return any(
        domain == blocked or domain.endswith("." + blocked)
        for blocked in _DOMAIN_BLOCKLIST
    )


class _SerperInput(BaseModel):
    search_query: str = Field(..., description="Search query to look up on Google.")


class SerperSearchTool:
    """Organic Google search via Serper.dev — drop-in replacement for SerperDevTool."""

    name: str = "Search the internet"
    description: str = (
        "Search Google for current information. "
        "Returns a list of organic search results with titles, URLs, and snippets."
    )
    args_schema = _SerperInput

    def run(self, search_query: str) -> str:
        api_key = os.environ.get("SERPER_API_KEY", "")
        try:
            resp = requests.get(
                "https://google.serper.dev/search",
                params={"q": search_query, "num": 5, "apiKey": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"[SerperSearch] Request failed: {exc}"

        results = []
        for item in data.get("organic", []):
            results.append(f"- {item.get('title', '')}\n  {item.get('link', '')}\n  {item.get('snippet', '')}")
        return "\n\n".join(results) if results else "No results found."
