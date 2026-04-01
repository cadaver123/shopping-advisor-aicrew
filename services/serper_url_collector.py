"""SerperUrlCollector — runs Google Dorks via Serper and returns deduplicated forum URLs."""

import logging
import os
import requests
import yaml
from pathlib import Path
from urllib.parse import urlparse

from services.search_tools import _is_blocked

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "discovery.yaml").read_text(encoding="utf-8")
)["scraping"]


class SerperUrlCollector:
    """Execute Google Dorks via Serper and return deduplicated forum URLs."""

    def run(self, dorks: list[str]) -> list[str]:
        api_key = os.environ.get("SERPER_API_KEY", "")
        seen: set[str] = set()
        urls: list[str] = []

        for i, dork in enumerate(dorks, 1):
            print(f"\n[Scraper] Serper query {i}: {dork!r}")
            try:
                resp = requests.get(
                    "https://google.serper.dev/search",
                    params={
                        "q": dork,
                        "num": _CFG["num_results_per_query"],
                        "tbs": _CFG["time_filter"],
                        "apiKey": api_key,
                    },
                    timeout=_CFG["timeout"],
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"  ✗ Serper failed: {exc}")
                continue

            added = 0
            for item in data.get("organic", []):
                url = item.get("link", "")
                if not url or url in seen:
                    continue
                domain = urlparse(url).netloc.lower().removeprefix("www.")
                if _is_blocked(domain):
                    print(f"  ✗ blocked: {url}")
                    continue
                seen.add(url)
                urls.append(url)
                added += 1
                print(f"  ✓ {url}")
                if len(urls) >= _CFG["max_total_urls"]:
                    return urls
            if not added:
                print("  (no results)")

        print(f"\n[Scraper] Found {len(urls)} unique URLs")
        return urls
