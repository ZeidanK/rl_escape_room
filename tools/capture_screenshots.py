"""Capture screenshots of the local Streamlit app using Playwright (query-param navigation)."""
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

APP_FILE = "app.py"
BASE_URL = "http://localhost:8501"

MODES = [
    # These correspond to the screenshots referenced in README/docs.
    ("Home", "home.png"),
    ("Room 1 \u2014 DP", "room1_value_policy.png"),
    ("Room 2 \u2014 SARSA", "room2_training.png"),
    ("Room 3 \u2014 Q-Learning", "room3_policy_no_key.png"),
    ("Room 4 \u2014 Function Approximation", "room4_trajectory.png"),
    ("Algorithm Comparison", "comparison.png"),
]


def main():
    # Start a temporary Streamlit server, capture each page, then always stop
    # the server in the finally block.
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP_FILE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "STREAMLIT_SERVER_HEADLESS": "true",
             "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"},
    )
    print("Waiting for Streamlit to start...", flush=True)
    time.sleep(20)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})

            for mode_label, filename in MODES:
                try:
                    url = f"{BASE_URL}/?mode={quote(mode_label)}"
                    page.goto(url, timeout=30000)
                    page.wait_for_load_state("networkidle")
                    time.sleep(4)
                    path = SCREENSHOT_DIR / filename
                    page.screenshot(path=str(path), full_page=True)
                    print(f"Saved {filename}", flush=True)
                except Exception as e:
                    print(f"Failed {filename}: {e}", flush=True)

            browser.close()
    finally:
        proc.terminate()
        proc.wait()
        print("Streamlit stopped.", flush=True)


if __name__ == "__main__":
    main()
