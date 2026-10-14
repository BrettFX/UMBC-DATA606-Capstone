"""Regenerate the README showcase screenshots in app/docs/screenshots/ (starts the app, drives it with Playwright).

Run from the repo root:  python scripts/capture_app_screenshots.py
(The throwaway screenshots written by the e2e tests go to app/tests/e2e/shots/, which is git-ignored.)
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "docs" / "screenshots"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.headless", "true", "--server.port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(url + "/_stcore/health", timeout=1).read() == b"ok":
                    break
            except Exception:
                time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
            pg.goto(url)
            btn = pg.get_by_role("button", name="Enter ClearanceIQ")
            btn.wait_for(timeout=30000)
            pg.wait_for_timeout(4500)
            pg.screenshot(path=OUT / "01-splash.png")

            btn.click()
            pg.locator(".ciq-live .ciq-w").first.wait_for(timeout=120000)
            pg.wait_for_timeout(2500)
            # park the playhead in the middle of the transcript so the word highlight is visible
            pg.evaluate("document.querySelector('audio').muted = true")
            t = pg.evaluate("(() => { const w = document.querySelectorAll('.ciq-live .ciq-w[data-s]')[3]; return (+w.dataset.s + +w.dataset.e) / 2 })()")
            pg.evaluate(f"document.querySelector('audio').currentTime = {t}")
            pg.wait_for_timeout(1200)
            pg.screenshot(path=OUT / "02-audio-analysis.png")

            pg.get_by_role("tab", name="Transcript & entities").click()
            pg.get_by_text("Highlighted transcript").wait_for(timeout=30000)
            pg.wait_for_timeout(1500)
            pg.evaluate("document.querySelector('[data-testid=stMain]').scrollTop = 330")
            pg.wait_for_timeout(500)
            pg.screenshot(path=OUT / "03-entities.png")

            pg.get_by_role("tab", name="Models").click()
            pg.get_by_text("Compare entity models").click()
            pg.get_by_text("Which should I pick?").wait_for(timeout=30000)
            pg.wait_for_timeout(800)
            pg.evaluate("document.querySelector('[data-testid=stMain]').scrollTop = 420")
            pg.wait_for_timeout(500)
            pg.screenshot(path=OUT / "04-model-card.png")
            b.close()
    finally:
        proc.terminate()
    print("wrote", *sorted(f.name for f in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
