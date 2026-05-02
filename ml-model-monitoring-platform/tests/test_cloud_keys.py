"""
Quick-and-dirty live test for whichever cloud providers you have configured
in `.env`. Run AFTER pasting your keys into .env:

    python tests/test_cloud_keys.py

For each provider whose API key is detected, this script:
  - Confirms the configured base URL and model name
  - Sends a tiny "say ok" prompt
  - Reports response time and any error
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils         import load_env
from src.llm_providers import PRESETS, detected_providers
from src.llm_router    import generate_explanation, PROVIDER_CLOUD


def main() -> int:
    load_env()
    found = detected_providers()
    if not found:
        print("⚠️  No API keys detected in .env.")
        print("    Edit .env and paste any of:")
        print("      OPENAI_API_KEY=…   GROQ_API_KEY=…   OPENROUTER_API_KEY=…   GEMINI_API_KEY=…")
        return 1

    print(f"Detected keys for: {', '.join(found)}\n")

    fails = 0
    for name in found:
        preset = PRESETS[name]
        print(f"── {name}")
        print(f"   base_url : {preset.base_url}")
        print(f"   model    : {preset.model_name}")

        cfg = {
            "api_key":    preset.api_key,
            "base_url":   preset.base_url,
            "model_name": preset.model_name,
        }
        t0 = time.time()
        resp = generate_explanation(PROVIDER_CLOUD, "Reply with only the word: ok", cfg)
        dt = time.time() - t0

        if resp.startswith("[ERROR]"):
            print(f"   ✗ FAILED in {dt:5.1f}s — {resp[:200]}")
            fails += 1
        else:
            preview = resp.replace("\n", " ").strip()[:80]
            print(f"   ✓ OK in {dt:5.1f}s — response: {preview!r}")
        print()

    print(f"Result: {len(found) - fails}/{len(found)} cloud providers working.")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
