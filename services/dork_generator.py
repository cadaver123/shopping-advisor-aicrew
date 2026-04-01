"""DorkGenerator — generates 3 optimized search queries with global/local market scope routing."""

import json
import logging
import os
import re
import requests
import yaml
from pathlib import Path

logger = logging.getLogger("shopping_advisor.tools")

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "models.yaml").read_text(encoding="utf-8")
)
_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "dork_generator.yaml").read_text(encoding="utf-8")
)["dork_generator"]


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from *text* — works even if model wraps it in reasoning."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


class DorkGenerator:
    def generate(self, query: str) -> dict:
        """Generate 3 search queries for *query* (any language).

        The LLM determines market scope (global/local) and generates
        queries in the appropriate language.

        Returns a dict with keys:
          - market_scope     (str): "global" or "local"
          - search_language  (str): language code, e.g. "en", "pl"
          - detected_category (str)
          - search_queries   (list of 3 strings)
        Raises RuntimeError if LLM call fails or JSON cannot be parsed.
        """
        cfg = _MODELS_CFG["strong_llm"]
        api_key = os.environ.get("TOGETHER_API_KEY", "")

        body = {
            "model": cfg["model"],
            "temperature": _CFG["temperature"],
            "max_tokens": cfg["max_tokens"],
            "messages": [
                {"role": "system", "content": _CFG["system_prompt"]},
                {"role": "user", "content": query},
            ],
        }
        if cfg.get("reasoning") is False:
            body["reasoning"] = {"enabled": False}

        from tools.llm import call as llm_call
        data = llm_call(body, label="DorkGenerator", timeout=120)
        content = data["choices"][0]["message"].get("content", "")
        result = _extract_json(content)

        if not result:
            raise RuntimeError(f"[DorkGenerator] Could not parse JSON from LLM response:\n{content[:500]}")

        if len(result.get("search_queries", [])) != 3:
            raise RuntimeError(f"[DorkGenerator] Expected 3 queries, got: {result.get('search_queries')}")

        logger.info("[DorkGenerator] Scope: %s | Category: %s | User lang: %s | Search lang: %s",
                    result.get("market_scope"), result.get("detected_category"),
                    result.get("user_language"), result.get("search_language"))
        for i, q in enumerate(result["search_queries"], 1):
            logger.info("[DorkGenerator] Query %d: %s", i, q)

        return result


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    if len(sys.argv) < 2:
        print("Usage: python dork_generator.py \"your shopping query\"")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query!r}\n")

    result = DorkGenerator().generate(query)

    print(f"Scope:        {result['market_scope']}")
    print(f"User lang:    {result['user_language']}")
    print(f"Search lang:  {result['search_language']}")
    print(f"Category:     {result['detected_category']}")
    print(f"\nGenerated dorks:")
    for i, q in enumerate(result["search_queries"], 1):
        print(f"  {i}. {q}")
