"""
Capture key pages in BOTH Streamlit themes (dark + light) to verify the
new theme-aware CSS works in either mode.

Output:
    assets/screenshots/theme_dark/*.png
    assets/screenshots/theme_light/*.png
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
URL  = "http://localhost:8600"
WIDTH, HEIGHT = 1600, 1000


PAGES = [
    ("01_home.png",       "🏠 Home"),
    ("04_dashboard.png",  "📊 Dashboard"),
    ("05_drift.png",      "🌊 Feature Drift"),
    ("08_ai.png",         "🤖 AI Insights"),
    ("09_settings.png",   "⚙️ Settings"),
]


async def capture_for_scheme(theme: str) -> None:
    """theme = 'dark' or 'light'"""
    out_dir = ROOT / "assets" / "screenshots" / f"theme_{theme}"
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # `color_scheme` switches the OS-level prefers-color-scheme media query
        context = await browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
            color_scheme=theme,         # ← drives our @media queries + Streamlit's auto theme
        )
        page = await context.new_page()
        print(f"\n=== Capturing {theme.upper()} theme ===")
        await page.goto(URL, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(3000)

        # Make sure demo data is loaded so dashboard pages have content
        await page.locator('label:has-text("📤 Upload & Configure")').first.click()
        await page.wait_for_timeout(2000)
        try:
            await page.locator('button:has-text("Load Demo Project")').first.click(timeout=4_000)
            await page.wait_for_timeout(2000)
            print("  · demo project loaded")
        except Exception:
            print("  · workspace already populated")

        for fname, label in PAGES:
            await page.locator(f'label:has-text("{label}")').first.click()
            await page.wait_for_timeout(2000)
            out = out_dir / fname
            await page.screenshot(path=str(out), full_page=True)
            print(f"  ✓ {out.relative_to(ROOT)}  ({out.stat().st_size//1024} KB)")

        await browser.close()


async def main() -> None:
    await capture_for_scheme("dark")
    await capture_for_scheme("light")


if __name__ == "__main__":
    asyncio.run(main())
