"""RagEnricher — retrieves per-product context from RagStore and synthesizes analysis."""

import json
import logging
import re
import yaml
from pathlib import Path

from services.rag_store import RagStore
from tools.llm import call as llm_call

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "rag_enricher.yaml").read_text(encoding="utf-8")
)["rag_enricher"]

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "models.yaml").read_text(encoding="utf-8")
)


class RagEnricher:
    """Retrieve relevant forum passages per product and generate pros/cons analysis."""

    def run(self, products: list[str], store: RagStore) -> list[dict]:
        """Analyze each product using retrieved context from *store*.

        Returns list of dicts: name, pros, cons, price_mentioned, verdict.
        """
        cfg = _MODELS_CFG["medium_llm"]
        analysis_cfg = _CFG["analysis"]
        results = []

        for product in products:
            print(f"\n[RAG] Analyzing: {product}")
            chunks = store.query(f"review opinion pros cons {product}", top_k=6)
            if not chunks:
                print("  ✗ No relevant chunks found.")
                results.append(self._empty(product))
                continue

            context = "\n\n---\n\n".join(chunks)
            data = llm_call(
                {
                    "model": cfg["model"],
                    "temperature": analysis_cfg["temperature"],
                    "max_tokens": analysis_cfg["max_tokens"],
                    "messages": [
                        {"role": "system", "content": analysis_cfg["system_prompt"]},
                        {"role": "user", "content": f"Product: {product}\n\nCommunity text:\n{context}"},
                    ],
                },
                label=f"RagEnricher/{product[:20]}",
                timeout=cfg.get("timeout", 60),
            )
            content = data["choices"][0]["message"].get("content", "")
            analysis = self._parse(content)
            analysis["name"] = product
            results.append(analysis)

            for pro in analysis.get("pros", []):
                print(f"  + {pro}")
            for con in analysis.get("cons", []):
                print(f"  - {con}")
            if analysis.get("price_mentioned"):
                print(f"  $ {analysis['price_mentioned']}")
            if analysis.get("verdict"):
                print(f"  → {analysis['verdict']}")

        return results

    @staticmethod
    def _parse(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"pros": [], "cons": [], "price_mentioned": None, "verdict": ""}
        try:
            obj = json.loads(match.group())
            return {
                "pros": [str(p) for p in obj.get("pros", [])],
                "cons": [str(c) for c in obj.get("cons", [])],
                "price_mentioned": obj.get("price_mentioned"),
                "verdict": str(obj.get("verdict", "")),
            }
        except json.JSONDecodeError:
            return {"pros": [], "cons": [], "price_mentioned": None, "verdict": ""}

    @staticmethod
    def _empty(name: str) -> dict:
        return {"name": name, "pros": [], "cons": [], "price_mentioned": None, "verdict": ""}
