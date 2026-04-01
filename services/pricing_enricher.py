"""PricingEnricher — fetches prices and store links for a list of products in parallel."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.store_lookup import get_product_pricing_and_links

logger = logging.getLogger("shopping_advisor.tools")

MAX_WORKERS = 5


class PricingEnricher:
    """Fetch price and store links for each product in parallel via DuckDuckGo."""

    def run(self, products: list[str], market_scope: str, user_language: str) -> list[dict]:
        pricing_scope = "local" if user_language != "en" else market_scope

        print(f"\n[Pricing] Fetching prices for {len(products)} products "
              f"(parallel, {MAX_WORKERS} workers, scope={pricing_scope}/{user_language})...")

        results: dict[str, dict] = {}

        def _fetch(name: str) -> tuple[str, dict]:
            data = get_product_pricing_and_links(name, pricing_scope, user_language)
            return name, data

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch, name): name for name in products}
            for future in as_completed(futures):
                name, data = future.result()
                print(f"  ✓ {name}: {data['estimated_price_range'] or 'not found'}")
                results[name] = data

        return [
            {
                "name":            name,
                "estimated_price": results[name]["estimated_price_range"],
                "store_links":     results[name]["store_links"],
            }
            for name in products
        ]
