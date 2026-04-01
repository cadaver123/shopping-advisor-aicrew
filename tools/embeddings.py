"""Together.ai embeddings — batched, with token/time logging.

Uses intfloat/multilingual-e5-large-instruct (1024-dim, multilingual).
"""

import logging
import os
import time

import requests

logger = logging.getLogger("shopping_advisor.tools")

_MODEL = "intfloat/multilingual-e5-large-instruct"
_BATCH = 32  # texts per request


def embed(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for *texts* via Together.ai.

    Splits into batches of up to _BATCH. Logs elapsed time.
    Returns list of float vectors, one per input text.
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    vectors: list[list[float]] = []
    total_tokens = 0
    t0 = time.monotonic()

    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        for attempt in range(5):
            if attempt:
                time.sleep(2 ** attempt)
            resp = requests.post(
                "https://api.together.xyz/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": _MODEL, "input": batch},
                timeout=60,
            )
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        vectors.extend(item["embedding"] for item in items)
        total_tokens += (data.get("usage") or {}).get("total_tokens", 0)

    elapsed = time.monotonic() - t0
    logger.info("[Embed] %d texts | %d tok | %.1fs", len(texts), total_tokens, elapsed)
    return vectors
