"""Capture screenshots of the local Streamlit app using Playwright."""
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

APP_FILE = PROJECT_ROOT / "app.py"
PORT = int(os.environ.get("STREAMLIT_SCREENSHOT_PORT", "8501"))
BASE_URL = f"http://localhost:{PORT}"


@dataclass(frozen=True)
class ScreenshotRoute:
    query: str
    filename: str
    wait_for_text: str
    click_tab: str | None = None
    scroll_to_text: str | None = None

ROUTES = [
    # These query-param routes match the public Streamlit presentation links.
    ScreenshotRoute("?view=showcase", "home.png", "Required Campaign"),
    ScreenshotRoute("?view=lab&room=room1", "room1_value_policy.png", "Start Value", click_tab="Policy Grid"),
    ScreenshotRoute("?view=lab&room=room2", "room2_training.png", "Greedy Replay Return"),
    ScreenshotRoute("?view=lab&room=room3", "room3_policy_no_key.png", "Greedy Replay Return", click_tab="Policy (No Key)"),
    ScreenshotRoute(
        "?view=showcase&room=room4",
        "room4_trajectory.png",
        "Continuous Trajectory Playback",
        scroll_to_text="Continuous Trajectory Playback",
    ),
    ScreenshotRoute("?view=comparison", "comparison.png", "Matched Comparison"),
]


def wait_for_server(timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL, timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Streamlit did not start at {BASE_URL} within {timeout_s}s")


def fail_if_error_page(page) -> None:
    error_markers = ["NameError", "Traceback:", "ModuleNotFoundError", "FileNotFoundError"]
    body_text = page.locator("body").inner_text(timeout=5000)
    for marker in error_markers:
        if marker in body_text:
            raise RuntimeError(f"Refusing to save screenshot because page contains {marker}")


def main():
    # Start a temporary Streamlit server, capture each page, then always stop
    # the server in the finally block.
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_FILE), "--server.port", str(PORT)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        },
    )
    print("Waiting for Streamlit to start...", flush=True)
    wait_for_server()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            for route in ROUTES:
                context = None
                try:
                    context = browser.new_context(viewport={"width": 1280, "height": 900})
                    page = context.new_page()
                    url = f"{BASE_URL}/{route.query}"
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    page.get_by_text(route.wait_for_text).first.wait_for(timeout=90000)
                    if route.click_tab is not None:
                        page.get_by_role("tab", name=route.click_tab).click(timeout=30000)
                    if route.scroll_to_text is not None:
                        page.get_by_text(route.scroll_to_text).first.scroll_into_view_if_needed(timeout=30000)
                    page.wait_for_timeout(3000)
                    fail_if_error_page(page)
                    path = SCREENSHOT_DIR / route.filename
                    page.screenshot(path=str(path))
                    print(f"Saved {route.filename}", flush=True)
                except Exception as e:
                    print(f"Failed {route.filename}: {e}", flush=True)
                finally:
                    if context is not None:
                        context.close()

            browser.close()
    finally:
        proc.terminate()
        proc.wait()
        print("Streamlit stopped.", flush=True)


if __name__ == "__main__":
    main()
