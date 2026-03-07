"""
Shopping Advisor — CrewAI multi-agent system

Usage:
    python main.py
    python main.py "gaming mouse under 200 PLN"
"""

import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Logging — shopping_advisor.tools writes to tool_urls.log + stdout
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = logging.Formatter("%(asctime)s %(message)s")
    logger = logging.getLogger("shopping_advisor.tools")
    logger.setLevel(level)
    logger.propagate = False
    fh = logging.FileHandler("tool_urls.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


_setup_logging()


# ---------------------------------------------------------------------------
# Token tracking via litellm success callback
# ---------------------------------------------------------------------------

_token_totals: dict = {"prompt": 0, "completion": 0, "requests": 0}


def _setup_token_tracking() -> None:
    import litellm

    def _on_success(kwargs, response_obj, start_time, end_time):
        try:
            usage = response_obj.usage
            _token_totals["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
            _token_totals["completion"] += getattr(usage, "completion_tokens", 0) or 0
            _token_totals["requests"] += 1
        except Exception:
            pass

    litellm.success_callback = [_on_success]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_env() -> list[str]:
    return [v for v in ("TOGETHER_API_KEY", "SERPER_API_KEY") if not os.environ.get(v)]


def _get_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    print("Describe what you want to buy and your budget.")
    print('Example: "gaming mouse under 200 PLN"')
    query = input("\nYour query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(0)
    return query


def _print_token_usage(output) -> None:
    prompt = _token_totals["prompt"]
    completion = _token_totals["completion"]
    requests = _token_totals["requests"]

    if prompt == 0:
        metrics = getattr(output, "token_usage", None)
        if metrics:
            prompt = getattr(metrics, "prompt_tokens", 0) or 0
            completion = getattr(metrics, "completion_tokens", 0) or 0
            requests = getattr(metrics, "successful_requests", 0) or 0

    cfg = yaml.safe_load(
        (Path(__file__).parent / "config" / "models.yaml").read_text(encoding="utf-8")
    )
    model = f"together_ai/{cfg['brain_llm']['model']}"

    try:
        import litellm
        cost = litellm.completion_cost(
            completion_response=None,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
    except Exception:
        cost = None

    print("=" * 50)
    print(f"  Prompt    : {prompt:>10,} tokens")
    print(f"  Completion: {completion:>10,} tokens")
    print(f"  Total     : {prompt + completion:>10,} tokens")
    print(f"  Requests  : {requests:>10,}")
    if cost is not None:
        print(f"  Est. cost : ${cost:>8.4f}")
    print("=" * 50)


def _print_result(result: str) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        Console().print(Markdown(result))
    except ImportError:
        print(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    missing = _check_env()
    if missing:
        print("[ERROR] Missing environment variables:")
        for v in missing:
            print(f"  • {v}")
        sys.exit(1)

    _setup_token_tracking()

    from crew import ShoppingAdvisorCrew

    query = _get_query()
    print(f"\nSearching: {query!r}\n")

    try:
        output = ShoppingAdvisorCrew().crew().kickoff(inputs={"query": query})
        result = str(output)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise

    print()
    _print_token_usage(output)
    print()
    _print_result(result)

    out_file = Path(__file__).parent / "last_report.md"
    out_file.write_text(result, encoding="utf-8")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
