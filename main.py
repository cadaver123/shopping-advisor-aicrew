"""
Shopping Advisor — Phase 1: Consensus Discovery + Pricing

Steps:
  1 — DorkGenerator:     query → search queries + market scope
  2 — SerperUrlCollector: dorks → Serper → forum URLs
  3 — ProductsExtractor: URLs → scrape + LLM → top 10 products
  4 — PricingEnricher:   products → parallel DDG → prices + store links

Usage:
    python main.py
    python main.py "Best IEM headphones under $50"
    python main.py "Best washing machine under 2000 PLN"
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s %(name)s %(message)s",
)

from services.dork_generator import DorkGenerator
from services.serper_url_collector import SerperUrlCollector
from services.products_extractor import ProductsExtractor
from services.rag_store import RagStore
from services.rag_enricher import RagEnricher

logger = logging.getLogger("shopping_advisor")


def _get_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    print('Describe what you want to buy, e.g. "gaming mouse under 200 PLN"')
    query = input("\nYour query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(0)
    return query


def main() -> None:
    missing = [v for v in ("TOGETHER_API_KEY", "SERPER_API_KEY") if not os.environ.get(v)]
    if missing:
        print("[ERROR] Missing env vars:", ", ".join(missing))
        sys.exit(1)

    query = _get_query()
    print(f"\nQuery: {query!r}\n")

    dorks_result = DorkGenerator().generate(query)
    print(f"Scope:       {dorks_result['market_scope']}")
    print(f"User lang:   {dorks_result.get('user_language', '?')}")
    print(f"Search lang: {dorks_result['search_language']}")
    print(f"Category:    {dorks_result['detected_category']}")

    urls = SerperUrlCollector().run(dorks_result["search_queries"])

    extractor = ProductsExtractor()
    products, pages = extractor.run(
        urls,
        query=query,
        budget=dorks_result.get("bucketed_budget_usd", dorks_result.get("converted_budget_usd", "none")),
        original_budget=dorks_result.get("original_budget", "none"),
    )

    if not products:
        print("\nNo products found within budget.")
        return

    store = RagStore()
    store.add(pages)

    enriched = RagEnricher().run(products, store)

    print(f"\n{'='*60}")
    for item in enriched:
        print(f"\n  {item['name']}")
        if item.get("price_mentioned"):
            print(f"    Price: {item['price_mentioned']}")
        for pro in item.get("pros", []):
            print(f"    + {pro}")
        for con in item.get("cons", []):
            print(f"    - {con}")
        if item.get("verdict"):
            print(f"    → {item['verdict']}")
    print(f"\nDone. {len(urls)} URLs scraped, {len(products)} products analyzed.")

    logger.info(
        "[Result] %s",
        json.dumps(
            {"query": query, "urls_scraped": urls, "products": enriched},
            indent=2,
            ensure_ascii=False,
        ),
    )



if __name__ == "__main__":
    main()
