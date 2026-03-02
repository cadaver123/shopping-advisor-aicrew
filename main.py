"""
Shopping Advisor — CrewAI multi-agent system
=============================================
Uses Together.ai (DeepSeek-V3) + Serper.dev to research products across
expert reviews, Reddit, Allegro.pl and AliExpress, then returns a ranked
Markdown report.

Usage
-----
    python main.py
    # or pass a query directly:
    python main.py "Best noise-cancelling headphones under $250"
"""

import re
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables from .env (if present)
load_dotenv(Path(__file__).parent / ".env")


def _check_env() -> list[str]:
    """Return a list of missing required environment variables."""
    required = ["TOGETHER_API_KEY", "SERPER_API_KEY"]
    return [v for v in required if not os.environ.get(v)]


def _print_banner() -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(
            Panel.fit(
                "[bold cyan]🛍  Shopping Advisor[/bold cyan]\n"
                "[dim]Powered by CrewAI · Together.ai · DeepSeek-V3[/dim]",
                border_style="cyan",
            )
        )
    except ImportError:
        print("=" * 60)
        print("  Shopping Advisor — CrewAI + Together.ai / DeepSeek-V3")
        print("=" * 60)


def _check_urls(text: str) -> None:
    """Print HTTP status for every URL found in the report."""
    urls = re.findall(r'https?://[^\s\)\]"<>]+', text)
    if not urls:
        print("No URLs found in the report.")
        return

    print(f"Checking {len(urls)} URL(s) found in the report…")
    for url in urls:
        try:
            r = requests.head(url, timeout=5, allow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0"})
            status = r.status_code
        except Exception as exc:
            status = f"ERROR: {exc}"
        marker = "OK " if status == 200 else "BAD"
        print(f"  [{marker}] {status}  {url}")


def _print_result(result: str) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(result))
    except ImportError:
        print(result)


def _get_query() -> str:
    """Return the user's shopping query from CLI arg or interactive prompt."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])

    _print_banner()
    print()
    print("Describe what you want to buy and your budget.")
    print('Example: "Best headphones between $200 and $250"')
    print()
    query = input("Your query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(0)
    return query


def main() -> None:
    # ------------------------------------------------------------------ env
    missing = _check_env()
    if missing:
        print("[ERROR] The following environment variables are not set:")
        for var in missing:
            print(f"  • {var}")
        print()
        print("Copy .env.example to .env and fill in your API keys.")
        sys.exit(1)

    # ------------------------------------------------------------------ query
    query = _get_query()

    print()
    print(f"Researching: {query!r}")
    print("This may take a minute or two — multiple agents are working in parallel…")
    print()

    # ------------------------------------------------------------------ run
    # Import here so env vars are loaded before any module-level LLM init
    from crew import ShoppingAdvisorCrew

    try:
        crew_instance = ShoppingAdvisorCrew().crew()
        output = crew_instance.kickoff(inputs={"query": query})
        result = str(output)

        # Dump each task's raw LLM output for URL debugging
        print("\n" + "=" * 60)
        print("DEBUG: Raw task outputs (compare URLs against tool_urls.log)")
        print("=" * 60)
        for task in crew_instance.tasks:
            print(f"\n--- {task.name} ---")
            if task.output:
                print(task.output.raw)
        print("=" * 60 + "\n")
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] The crew encountered an error:\n  {exc}")
        raise

    # ------------------------------------------------------------------ output
    print()
    _print_banner()
    print()
    _print_result(result)

    # Optionally save to file
    out_file = Path(__file__).parent / "last_report.md"
    out_file.write_text(result, encoding="utf-8")
    print()
    print(f"Report saved to: {out_file}")

    # URL validation
    print()
    _check_urls(result)


if __name__ == "__main__":
    main()
