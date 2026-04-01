"""Shared Playwright browser and HTML extraction helpers.

Playwright's sync API is tied to the thread that created it (greenlet-based).
All browser operations are dispatched to a single dedicated worker thread via a
queue, making ``fetch_html`` / ``fetch_as_markdown`` safe to call from any thread.
"""

import atexit
import logging
import queue
import re
import threading

logger = logging.getLogger("shopping_advisor.tools")

# ---------------------------------------------------------------------------
# Dedicated browser thread
# ---------------------------------------------------------------------------

_req_queue: queue.Queue = queue.Queue()
_worker_thread: threading.Thread | None = None
_start_lock = threading.Lock()


def _browser_worker() -> None:
    """Runs in a single dedicated thread — owns all Playwright state."""
    from playwright.sync_api import sync_playwright
    from playwright_stealth import Stealth

    stealth = Stealth()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        logger.info("[Browser] Chromium browser started.")

        while True:
            item = _req_queue.get()
            if item is None:           # shutdown sentinel
                break

            url, result_holder, done_event = item
            try:
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                )
                page = context.new_page()
                stealth.apply_stealth_sync(page)
                page.route(
                    "**/*",
                    lambda route, req: route.abort()
                    if req.resource_type in {"image", "stylesheet", "font", "media", "other"}
                    else route.continue_(),
                )
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=5000)
                    except Exception:
                        pass

                # page.content() can raise if the page is still navigating — retry briefly
                html = ""
                for _ in range(5):
                    try:
                        html = page.content()
                        break
                    except Exception:
                        import time as _time
                        _time.sleep(0.5)
                result_holder["html"] = html
                context.close()
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                done_event.set()

        browser.close()


def _ensure_worker() -> None:
    global _worker_thread
    with _start_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_browser_worker, daemon=True, name="browser-worker")
            _worker_thread.start()


def _shutdown() -> None:
    if _worker_thread and _worker_thread.is_alive():
        _req_queue.put(None)
        _worker_thread.join(timeout=5)


atexit.register(_shutdown)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> str:
    """Fetch raw HTML from *url* via the stealth browser worker thread.

    Thread-safe — may be called from any thread including ThreadPoolExecutor workers.
    """
    _ensure_worker()
    result: dict = {}
    done = threading.Event()
    _req_queue.put((url, result, done))
    done.wait()
    if "error" in result:
        raise result["error"]
    return result.get("html", "")


def fetch_as_markdown(url: str) -> str:
    """Fetch *url* and return clean Markdown text.

    Uses trafilatura for boilerplate removal. Returns an empty string if
    extraction yields nothing — callers should handle this case explicitly
    rather than receiving noisy full-DOM output.
    """
    import trafilatura

    html = fetch_html(url)
    if not html:
        return ""

    md = trafilatura.extract(
        html,
        output_format="markdown",
        include_formatting=True,
        include_links=False,
        include_images=False,
    )

    if not md or not md.strip():
        return ""

    return re.sub(r"\n{3,}", "\n\n", md).strip()
