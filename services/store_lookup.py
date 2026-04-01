"""Pricing & availability tool — uses DuckDuckGo (free, no API key) to find store links and prices."""

import logging
import os
import re
import requests
import yaml
from pathlib import Path
from urllib.parse import urlparse

from ddgs import DDGS

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "store_lookup.yaml").read_text(encoding="utf-8")
)["pricing"]

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "models.yaml").read_text(encoding="utf-8")
)

# Regex: match prices like $39.99, €29, 1 299 zł, 1299 PLN, 1 299,00 zł
# Decimal part is either fully present (separator + 1-2 digits) or fully absent.
_PRICE_RE = re.compile(
    r"(\$|€|£|zł|PLN)\s?\d[\d\s]{0,5}(?:[.,]\d{1,2})?"
    r"|\d[\d\s]{0,5}[.,]\d{1,2}\s?(zł|PLN|USD|EUR)",
    re.IGNORECASE,
)

# Search keyword for "price" varies by language — used in DDG site: queries
_PRICE_KEYWORD = {
    "pl": "cena", "de": "preis", "fr": "prix", "it": "prezzo",
    "es": "precio", "pt": "preço", "nl": "prijs",
}


def _preferred_domains(market_scope: str, language: str) -> list[str]:
    pref = _CFG.get("preferred_domains", {})
    if market_scope == "local":
        return pref.get(language, pref.get("default", []))
    return pref.get("global", pref.get("default", []))


def _ddg_queries(product_name: str, market_scope: str, language: str) -> list[str]:
    """Return one DDG query per preferred domain (single site: per query — DDG limitation)."""
    domains = _preferred_domains(market_scope, language)
    keyword = _PRICE_KEYWORD.get(language, "price") if market_scope == "local" else "price"
    return [f'"{product_name}" {keyword} site:{domain}' for domain in domains]


def _extract_price(text: str) -> str | None:
    match = _PRICE_RE.search(text)
    return match.group().strip() if match else None


def _store_name(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _lowest_price(prices: list[str]) -> str | None:
    """Return the lowest price string >= 5, or None if none pass the threshold."""
    def _numeric(p: str) -> float:
        digits = re.sub(r"[^\d.,]", "", p).replace(",", ".").replace(" ", "")
        try:
            return float(digits)
        except ValueError:
            return 0.0

    valid = [p for p in prices if _numeric(p) >= 5]
    return min(valid, key=_numeric) if valid else None


def _browser_scrape_price(url: str, product_name: str) -> str | None:
    """Fetch *url* with a stealth browser, then ask small_llm to extract the price."""
    from tools.browser import fetch_as_markdown

    try:
        text = fetch_as_markdown(url)
        text = text[:3000]
    except Exception as exc:
        logger.debug("[Pricing] Browser fetch failed for %s: %s", url, exc)
        return None

    if not text:
        return None

    llm_cfg = _MODELS_CFG["small_llm"]
    extraction_cfg = _CFG["price_extraction"]
    api_key = os.environ.get("TOGETHER_API_KEY", "")

    body = {
        "model": llm_cfg["model"],
        "temperature": extraction_cfg["temperature"],
        "max_tokens": 32,
        "messages": [
            {"role": "system", "content": extraction_cfg["system_prompt"]},
            {"role": "user", "content": f"Product: {product_name}\n\nPage text:\n{text}"},
        ],
    }
    if llm_cfg.get("reasoning") is False:
        body["reasoning"] = {"enabled": False}

    try:
        from tools.llm import call as llm_call
        data = llm_call(body, label="PriceScrape", timeout=llm_cfg.get("timeout", 30))
        content = data["choices"][0]["message"].get("content", "").strip()
        if content and content.lower() not in ("null", "none", ""):
            return _extract_price(content) or content
    except Exception as exc:
        logger.debug("[Pricing] LLM price extraction failed for %s: %s", url, exc)
    return None


def get_product_pricing_and_links(
    product_name: str,
    market_scope: str,
    language: str = "en",
) -> dict:
    """Find prices and store links for *product_name*.

    For each DDG result, collects both the snippet price (fast, from search result)
    and a scraped price (browser + LLM). estimated_price_range is the lowest valid
    price across all sources.

    Args:
        product_name:  e.g. "Sony WH-1000XM5"
        market_scope:  "global" or "local"
        language:      language code from DorkGenerator (e.g. "pl", "de", "en")

    Returns:
        {
            "product": str,
            "estimated_price_range": str | null,
            "store_links": [
                {"store": str, "url": str, "snippet_price": str | null, "scraped_price": str | null},
                ...
            ]
        }
    """
    queries = _ddg_queries(product_name, market_scope, language)
    logger.info("[Pricing] Running %d store queries for %r", len(queries), product_name)

    raw_results: list[dict] = []
    for query in queries:
        try:
            hits = list(DDGS().text(query, max_results=2, backend="duckduckgo"))
            raw_results.extend(hits)
            if hits:
                for h in hits:
                    logger.info("[DDG] %-50s → %s", query[:50], h.get("href", "?"))
            else:
                logger.info("[DDG] %-50s → no results", query[:50])
        except Exception as exc:
            logger.info("[DDG] %-50s → failed: %s", query[:50], exc)

    allowed = set(_preferred_domains(market_scope, language))

    # Deduplicate by store domain, prefer entries that have any price
    by_domain: dict[str, dict] = {}
    for r in raw_results:
        url = r.get("href", "")
        if not url:
            continue
        domain = _store_name(url)
        if allowed and domain not in allowed:
            continue
        snippet_price = _extract_price(r.get("body", ""))
        scraped_price = _browser_scrape_price(url, product_name)
        entry = {
            "store":         domain,
            "url":           url,
            "snippet_price": snippet_price,
            "scraped_price": scraped_price,
        }
        existing = by_domain.get(domain)
        if not existing or (not existing["snippet_price"] and not existing["scraped_price"]):
            by_domain[domain] = entry

    store_links = list(by_domain.values())
    all_prices = [
        price
        for entry in store_links
        for price in (entry["snippet_price"], entry["scraped_price"])
        if price
    ]

    lowest = _lowest_price(all_prices)
    return {
        "product":               product_name,
        "estimated_price_range": f"~{lowest}" if lowest else None,
        "store_links":           store_links,
    }


if __name__ == "__main__":
    import sys
    import json

    # Usage: python store_lookup.py "Product Name" <scope> <lang>
    # e.g.:  python store_lookup.py "Samsung WW90T" local pl
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python store_lookup.py \"Product Name\" [global|local] [lang]")
        sys.exit(1)
    product  = args[0]
    scope    = args[1] if len(args) > 1 else "global"
    language = args[2] if len(args) > 2 else "en"

    result = get_product_pricing_and_links(product, scope, language)
    print(json.dumps(result, indent=2, ensure_ascii=False))
