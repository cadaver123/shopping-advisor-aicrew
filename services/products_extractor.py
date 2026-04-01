"""ProductsExtractor — scrapes a list of forum URLs and extracts recommended products."""

import json
import logging
import re
import yaml
from pathlib import Path

from tools.browser import fetch_as_markdown
from tools.llm import call as llm_call

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "products_extractor.yaml").read_text(encoding="utf-8")
)["products_extractor"]

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "models.yaml").read_text(encoding="utf-8")
)


class ProductsExtractor:
    """Scrape a list of forum URLs and extract the top recommended products.

    Usage::

        products = ProductsExtractor().run(urls, query="Best IEMs under $50", budget="$50", original_budget="250 PLN")
    """

    def run(
        self,
        urls: list[str],
        query: str = "",
        budget: str = "none",
        original_budget: str = "none",
    ) -> tuple[list[str], list[str]]:
        """Scrape *urls*, extract product names via LLM CoT.

        Returns:
            (products, pages) — product name strings and raw scraped page texts.
        """
        pages = self.scrape(urls)
        products = self.extract(pages, query, budget, original_budget)
        return products, pages

    # ------------------------------------------------------------------

    def scrape(self, urls: list[str]) -> list[str]:
        """Fetch and parse each URL. Returns list of page texts (one per URL)."""
        max_chars = _CFG["max_chars_per_page"]
        pages = []
        for url in urls:
            print(f"\n[Extractor] Fetching: {url}")
            try:
                text = fetch_as_markdown(url)
            except Exception as exc:
                print(f"  ✗ Failed: {exc}")
                continue
            if not text:
                print(f"  ✗ No content extracted")
                continue
            text = text[:max_chars]
            print(f"  ✓ {len(text):,} chars")
            pages.append(text)

        total = sum(len(p) for p in pages)
        print(f"\n[Extractor] Total: {total:,} chars from {len(pages)} pages")
        return pages

    def extract(self, pages: list[str], query: str = "", budget: str = "none", original_budget: str = "none") -> list[str]:
        """Extract product names from pre-scraped *pages* via LLM CoT."""
        text = "\n\n---\n\n".join(pages)
        return self._extract(text, query, budget, original_budget)

    def _extract(self, text: str, query: str, budget: str, original_budget: str) -> list[str]:
        cfg = _MODELS_CFG["strong_llm"]
        extraction_cfg = _CFG["extraction"]

        system_prompt = extraction_cfg["system_prompt"].format(
            converted_budget_usd=budget,
            original_budget=original_budget,
        )
        print(f"\n[Extractor] Extracting products from {len(text):,} chars "
              f"(budget={budget} / {original_budget})...")

        user_message = f"User query: {query}\n\nForum text:\n{text}" if query else text

        data = llm_call(
            {
                "model": cfg["model"],
                "temperature": extraction_cfg["temperature"],
                "max_tokens": 2048,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
            label="ProductsExtractor",
            timeout=extraction_cfg.get("timeout", 120),
        )
        content = data["choices"][0]["message"].get("content", "")
        products = self._parse_cot_response(content)

        if products is None:
            raise RuntimeError(f"[Extractor] Could not parse CoT JSON:\n{content[:400]}")

        if not products:
            print("[Extractor] No products fit the budget — returning empty list.")
            return []

        print(f"[Extractor] Extracted {len(products)} products:")
        for i, p in enumerate(products, 1):
            print(f"  {i}. {p}")

        return products

    @staticmethod
    def _parse_cot_response(text: str) -> list[str] | None:
        """Parse the audit_log JSON response and return recommended_products.

        Returns None if JSON cannot be found/parsed, empty list if budget filtered all out.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        # Log audit decisions so budget filtering is visible
        for entry in obj.get("audit_log", []):
            decision = entry.get("decision", "?")
            name = entry.get("name", "?")
            msrp = entry.get("estimated_msrp", "?")
            reason = entry.get("reason", "")
            marker = "✓" if decision == "KEEP" else "✗"
            print(f"  {marker} [{decision}] {name} ({msrp}) — {reason}")

        products = obj.get("recommended_products", [])
        return [str(p) for p in products if p]
