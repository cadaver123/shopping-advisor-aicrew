"""LLM-powered web page summarizer — fetch a URL and summarize it."""

import logging
import yaml
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from tools.browser import fetch_as_markdown

logger = logging.getLogger("shopping_advisor.tools")

_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "tools" / "page_summarizer.yaml").read_text(encoding="utf-8")
)["page_summarizer"]

_MODELS_CFG = yaml.safe_load(
    (Path(__file__).parent.parent / "config" / "models.yaml").read_text(encoding="utf-8")
)

_FAKE_DOMAINS = {"example.com", "example.org", "example.net", "test.com", "localhost"}


class _Input(BaseModel):
    url: str = Field(..., description="Full URL to fetch and summarize.")


class WebPageSummarizerTool:
    name: str = "Web Page Summarizer"
    description: str = (
        "Fetch the content of a specific URL, extract the text, and summarize it using an LLM. "
        "Uses a real headless browser so JavaScript-rendered pages work."
    )
    args_schema = _Input

    def run(self, url: str) -> str:
        domain = urlparse(url).netloc.removeprefix("www.")
        if domain in _FAKE_DOMAINS:
            msg = f"[WebPageSummarizer] REJECTED constructed URL: {url}."
            logger.warning(msg)
            return msg

        logger.info("[WebPageSummarizer] FETCH_URL: %s", url)
        try:
            text = fetch_as_markdown(url)
        except Exception as exc:
            return f"Could not fetch {url}: {exc}"

        if not text:
            return f"[WebPageSummarizer] No content extracted from {url}"

        max_chars = _CFG.get("max_chars", 20_000)
        text = text[:max_chars]

        cfg = _MODELS_CFG["strong_llm"]
        try:
            from tools.llm import call as llm_call
            data = llm_call(
                {
                    "model": cfg["model"],
                    "temperature": _CFG.get("temperature", 0.1),
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "system", "content": _CFG.get("system_prompt", "Summarize the following web page.")},
                        {"role": "user", "content": text},
                    ],
                },
                label="PageSummarizer",
                timeout=cfg.get("timeout", 60),
            )
            return data["choices"][0]["message"]["content"] or text
        except Exception as exc:
            logger.warning("[WebPageSummarizer] LLM failed (%s) — returning raw text.", exc)
            return text
