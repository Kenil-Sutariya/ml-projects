"""
Capture polished screenshots of every page for documentation.

Pre-requisite: the Streamlit app must already be running. Run:
    streamlit run app.py --server.port 8600
then in another terminal:
    python tests/capture_screenshots.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "assets" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8600"
WIDTH, HEIGHT = 1600, 1000


# Each entry: (filename, page_label, optional pre-action callback)
PAGES: list[tuple[str, str]] = [
    ("01_home.png",                "🏠 Home"),
    ("02_upload_configure.png",    "📤 Upload & Configure"),
    ("03_run_monitoring.png",      "▶️ Run Monitoring"),
    ("04_dashboard.png",           "📊 Dashboard"),
    ("05_feature_drift.png",       "🌊 Feature Drift"),
    ("06_error_analysis.png",      "❌ Error Analysis"),
    ("07_evidently_reports.png",   "📋 Evidently Reports"),
    ("08_ai_insights.png",         "🤖 AI Insights"),
    ("09_settings.png",            "⚙️ Settings"),
]


async def navigate_to(page, label: str) -> None:
    """Click the sidebar radio item that matches `label`."""
    # Streamlit radio labels are inside <label> elements with the exact text
    await page.locator(f'label:has-text("{label}")').first.click()
    await page.wait_for_timeout(2500)


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,   # retina-quality screenshots
        )
        page = await context.new_page()

        # Initial visit + click "Load Demo Project" so the dashboard has data
        print(f"→ {URL}")
        await page.goto(URL, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(3000)

        # Make sure demo data is loaded so dashboards have content
        await navigate_to(page, "📤 Upload & Configure")
        try:
            await page.locator('button:has-text("Load Demo Project")').first.click(timeout=5_000)
            await page.wait_for_timeout(2000)
            print("  ✓ Demo project loaded")
        except Exception:
            print("  (Demo Project button not found — workspace already populated)")

        # Capture each page
        for fname, label in PAGES:
            print(f"→ {label}")
            await navigate_to(page, label)
            await page.wait_for_timeout(1500)

            out = OUT / fname
            await page.screenshot(path=str(out), full_page=True)
            kb = out.stat().st_size // 1024
            print(f"  ✓ saved {out.name}  ({kb} KB)")

        # Bonus: capture the BYO Model wizard branch
        print("→ Upload & Configure → Bring Your Own Model")
        await navigate_to(page, "📤 Upload & Configure")
        await page.locator('label:has-text("Bring Your Own Model")').first.click()
        await page.wait_for_timeout(1500)
        out = OUT / "02b_byo_model_mode.png"
        await page.screenshot(path=str(out), full_page=True)
        print(f"  ✓ saved {out.name}")

        # Bonus: AI Insights → Cloud LLM branch with Groq preset
        print("→ AI Insights → Cloud LLM (Groq preset)")
        await navigate_to(page, "🤖 AI Insights")
        await page.locator('label:has-text("Cloud LLM with my own API key")').first.click()
        await page.wait_for_timeout(1500)
        # Switch preset to Groq if possible
        try:
            await page.locator('div[data-baseweb="select"]:has-text("OpenAI")').first.click()
            await page.wait_for_timeout(500)
            await page.locator('li:has-text("Groq")').first.click()
            await page.wait_for_timeout(2000)
        except Exception as exc:
            print(f"  (preset switch skipped: {exc})")
        out = OUT / "08b_ai_insights_cloud.png"
        await page.screenshot(path=str(out), full_page=True)
        print(f"  ✓ saved {out.name}")

        await browser.close()

    print(f"\nAll screenshots saved to: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
