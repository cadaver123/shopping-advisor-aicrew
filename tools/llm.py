"""Shared Together.ai API helper — logs token usage and wall-clock time per call."""

import logging
import os
import time
import requests

logger = logging.getLogger("shopping_advisor.tools")

_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BACKOFF_BASE = 2  # seconds


def call(body: dict, label: str, timeout: int = 60) -> dict:
    """POST *body* to Together.ai chat completions and return the parsed JSON response.

    Retries up to _MAX_RETRIES times on 5xx / 429 with exponential backoff.
    Logs model name, prompt/completion token counts, and elapsed time at INFO level.
    Raises requests.HTTPError on non-retryable errors or after all retries exhausted.
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    model_short = body.get("model", "?").split("/")[-1]

    t0 = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt:
            wait = _BACKOFF_BASE ** attempt
            print(f"  [LLM] {label} retry {attempt}/{_MAX_RETRIES} in {wait}s...")
            time.sleep(wait)

        try:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            continue

        if resp.status_code not in _RETRYABLE:
            resp.raise_for_status()
            break

        last_exc = requests.HTTPError(response=resp)

    else:
        raise last_exc  # type: ignore[misc]

    data = resp.json()
    elapsed = time.monotonic() - t0

    usage = data.get("usage") or {}
    logger.info(
        "[LLM] %-20s | %-30s | %dp + %dc = %d tok | %.1fs",
        label,
        model_short,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("total_tokens", 0),
        elapsed,
    )
    return data
