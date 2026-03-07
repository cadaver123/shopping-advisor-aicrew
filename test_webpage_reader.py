"""Standalone tester for WebPageReaderTool — no project imports needed.

Usage:
    python test_webpage_reader.py <url>

Example:
    python test_webpage_reader.py https://www.rtings.com/headphones/reviews/sony/wh-1000xm4
"""

import logging
import re
import sys
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tools import WebPageReaderTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Ensure the tool's named logger is visible
logging.getLogger("shopping_advisor.tools").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]

    print(f"URL: {url}")
    print("-" * 60)

    result = WebPageReaderTool()._run(url=url)
    print(result)


if __name__ == "__main__":
    main()
