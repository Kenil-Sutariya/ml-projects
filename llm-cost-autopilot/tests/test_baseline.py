"""
Phase 1 baseline test — sends 10 prompts to every registered model,
logs results to data/baseline_results.json, and prints a cost/latency table.

Run:
    cd cost-autopilot
    python -m pytest tests/test_baseline.py -v -s
    # or directly:
    python tests/test_baseline.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.client import send_request
from src.models.registry import REGISTRY, Provider

# ---------------------------------------------------------------------------
# 10 diverse prompts covering simple → complex
# ---------------------------------------------------------------------------
BASELINE_PROMPTS = [
    # Tier 1 — simple
    "What is the capital of France?",
    "Convert 100 Fahrenheit to Celsius.",
    "Extract all email addresses from: 'Contact us at info@example.com or support@acme.org'",
    "Fix the grammar: 'She dont like going to the store'",
    # Tier 2 — moderate
    "Summarize the following in 3 bullet points: The transformer architecture revolutionized NLP by replacing recurrent networks with self-attention mechanisms, enabling parallelization and better long-range dependency modeling.",
    "Classify this review as Positive, Neutral, or Negative: 'The product arrived on time but the packaging was damaged and the instructions were unclear.'",
    "List pros and cons of using microservices architecture vs monolithic architecture.",
    # Tier 3 — complex
    "A company has 3 factories and 4 warehouses. Factory costs and warehouse demands are given. Explain step by step how you would set up a transportation problem to minimize cost.",
    "Write a Python function that finds all prime numbers up to N using the Sieve of Eratosthenes, then explain its time complexity.",
    "Compare and contrast the philosophical positions of Kant and Mill on the ethics of lying, and provide your own reasoned position.",
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "baseline_results.json"


async def run_baseline(skip_providers: set[Provider] | None = None) -> list[dict]:
    skip_providers = skip_providers or set()
    results = []

    for model_key, config in REGISTRY.items():
        if config.provider in skip_providers:
            print(f"\n[SKIP] {config.display_name} (provider excluded)")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {config.display_name}  ({config.provider.value})")
        print(f"{'='*60}")

        model_results = []
        total_cost = 0.0
        total_latency = 0.0

        for i, prompt in enumerate(BASELINE_PROMPTS, 1):
            try:
                response = await send_request(prompt, config)
                total_cost += response.cost_usd
                total_latency += response.latency_ms

                row = {
                    "model_key": model_key,
                    "model_id": config.model_id,
                    "provider": config.provider.value,
                    "prompt_index": i,
                    "prompt": prompt,
                    **response.summary(),
                    "timestamp": response.timestamp.isoformat(),
                }
                model_results.append(row)
                results.append(row)

                print(
                    f"  [{i:02d}] latency={response.latency_ms:>7.0f}ms  "
                    f"cost=${response.cost_usd:.6f}  "
                    f"tokens={response.total_tokens}"
                )

            except Exception as exc:
                print(f"  [{i:02d}] ERROR — {exc}")
                results.append({
                    "model_key": model_key,
                    "model_id": config.model_id,
                    "provider": config.provider.value,
                    "prompt_index": i,
                    "prompt": prompt,
                    "error": str(exc),
                })

        if model_results:
            print(
                f"\n  TOTAL  cost=${total_cost:.6f}  "
                f"avg_latency={total_latency/len(model_results):.0f}ms"
            )

    return results


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_model[r["model_key"]].append(r)

    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'Provider':<12} {'Total Cost':>12} {'Avg Latency':>14}")
    print(f"{'-'*70}")

    for key, rows in by_model.items():
        total_cost = sum(r["cost_usd"] for r in rows)
        avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
        provider = rows[0]["provider"]
        model_label = rows[0].get("display_name", key)
        print(
            f"{key:<25} {provider:<12} ${total_cost:>11.6f} {avg_latency:>12.0f}ms"
        )

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Skip Ollama if not running locally; set SKIP_OLLAMA=1 to exclude
    skip = set()
    if os.getenv("SKIP_OLLAMA", "0") == "1":
        skip.add(Provider.OLLAMA)
    if os.getenv("SKIP_OPENAI", "0") == "1":
        skip.add(Provider.OPENAI)
    if os.getenv("SKIP_ANTHROPIC", "0") == "1":
        skip.add(Provider.ANTHROPIC)
    if os.getenv("SKIP_GROQ", "0") == "1":
        skip.add(Provider.GROQ)

    results = asyncio.run(run_baseline(skip_providers=skip))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUTPUT_PATH}")

    print_summary(results)
