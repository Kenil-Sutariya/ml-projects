"""
Phase 6 load test — sends diverse prompts through the live API,
respects Groq free-tier rate limits, and produces a final report.

Run:
    python scripts/load_test.py
    python scripts/load_test.py --count 200 --concurrency 4
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, quantiles

import httpx

API_BASE = "http://localhost:8000"
REPORT_PATH = Path("data/load_test_report.json")

# Diverse prompt pool — roughly 1/3 per tier
PROMPTS = [
    # Tier 1 — simple
    ("What is the capital of Brazil?", 1),
    ("Convert 37 degrees Celsius to Fahrenheit.", 1),
    ("What does URL stand for?", 1),
    ("Fix the spelling: 'definately recieve tommorow'", 1),
    ("What is the square root of 169?", 1),
    ("Translate 'thank you' to German.", 1),
    ("What is the chemical formula for table salt?", 1),
    ("Extract emails from: 'ping us at dev@startup.io or hr@startup.io'", 1),
    ("Who invented the telephone?", 1),
    ("What is 256 in binary?", 1),
    ("What does CPU stand for?", 1),
    ("How many hours are in a week?", 1),
    ("Translate 'good night' to Italian.", 1),
    ("What is the plural of 'datum'?", 1),
    ("Fix grammar: 'They was going to the store'", 1),
    ("What is the boiling point of ethanol in Celsius?", 1),
    ("What does JSON stand for?", 1),
    ("Convert 1 mile to meters.", 1),
    ("What is the opposite of 'verbose'?", 1),
    ("Name the three branches of the US government.", 1),
    # Tier 2 — moderate
    ("Summarize the concept of machine learning in 3 bullet points.", 2),
    ("Classify this review: 'Fast shipping but the item was broken on arrival.'", 2),
    ("List the pros and cons of using TypeScript over JavaScript.", 2),
    ("Explain what a REST API is with a simple real-world analogy.", 2),
    ("Write a Python function to reverse a linked list.", 2),
    ("What are the SOLID principles? Give a one-line description for each.", 2),
    ("What metrics would you track for a SaaS product's health?", 2),
    ("Explain the difference between SQL JOINs: INNER, LEFT, RIGHT, FULL.", 2),
    ("Write a professional email asking for a 2-week project deadline extension.", 2),
    ("What is the difference between a mutex and a semaphore?", 2),
    ("Explain what Docker is and why developers use it.", 2),
    ("List 5 best practices for writing clean Python code.", 2),
    ("Write a SQL query to find the top 5 customers by total purchase amount.", 2),
    ("Explain the event loop in JavaScript in simple terms.", 2),
    ("What is the difference between horizontal and vertical database scaling?", 2),
    ("Write a brief product description for a standing desk converter.", 2),
    ("Classify the intent: 'I want to cancel my subscription immediately.'", 2),
    ("What is memoization? Give an example in Python.", 2),
    ("Summarize: The OSI model defines 7 layers of network communication, from physical transmission to application-level protocols, providing a framework for understanding how data moves across networks.", 2),
    ("Explain the producer-consumer problem and a common solution.", 2),
    # Tier 3 — complex
    ("Design a database schema for a Twitter-like social platform. Include tables, indexes, and justify your choices.", 3),
    ("Explain the tradeoffs between strong consistency and eventual consistency in distributed systems with real examples.", 3),
    ("Write a Python implementation of a trie data structure with insert, search, and startswith methods. Analyze complexity.", 3),
    ("You are an engineering manager. One of your best engineers wants to leave because of lack of growth. How do you handle this conversation?", 3),
    ("Design a CI/CD pipeline for a microservices application with 10 services. Cover testing, deployment, and rollback.", 3),
    ("Analyze why most A/B tests in product companies fail to produce actionable insights. Propose improvements.", 3),
    ("Explain how garbage collection works in Python and Java. Compare their approaches and tradeoffs.", 3),
    ("Build a mental model for understanding Kubernetes: what problem it solves, its core abstractions, and how they relate.", 3),
    ("Design an API rate limiting system that supports per-user, per-endpoint, and global limits with burst capacity.", 3),
    ("Compare and contrast React, Vue, and Svelte for building a large-scale enterprise dashboard application.", 3),
]


async def send_one(
    client: httpx.AsyncClient,
    prompt: str,
    expected_tier: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{API_BASE}/v1/completions",
                json={"messages": [{"role": "user", "content": prompt}], "verify_quality": False},
                timeout=60.0,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                meta = data["router_metadata"]
                return {
                    "status": "ok",
                    "prompt": prompt[:60],
                    "expected_tier": expected_tier,
                    "actual_tier": meta["complexity_tier"],
                    "model": meta["selected_model"],
                    "confidence": meta["routing_confidence"],
                    "cost_usd": meta["estimated_cost_usd"],
                    "latency_ms": elapsed,
                    "tokens": data["usage"]["total_tokens"],
                    "correct_tier": meta["complexity_tier"] == expected_tier,
                }
            else:
                return {"status": "error", "code": resp.status_code, "prompt": prompt[:60],
                        "expected_tier": expected_tier, "latency_ms": (time.perf_counter() - t0) * 1000}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "prompt": prompt[:60],
                    "expected_tier": expected_tier, "latency_ms": (time.perf_counter() - t0) * 1000}


def build_report(results: list[dict], elapsed_total: float) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]

    latencies = [r["latency_ms"] for r in ok]
    costs = [r["cost_usd"] for r in ok]
    tokens = [r["tokens"] for r in ok]

    # GPT-4o baseline estimate (avg token count × GPT-4o price)
    gpt4o_baseline = sum(t * (2.50 + 10.0) / 1_000_000 for t in tokens)
    actual_cost = sum(costs)
    saved = gpt4o_baseline - actual_cost
    pct_saved = (saved / gpt4o_baseline * 100) if gpt4o_baseline > 0 else 0

    # Tier accuracy
    tier_correct = sum(1 for r in ok if r.get("correct_tier", False))
    tier_accuracy = (tier_correct / len(ok) * 100) if ok else 0

    # Model distribution
    from collections import Counter
    model_dist = dict(Counter(r["model"] for r in ok))
    tier_dist  = dict(Counter(r["actual_tier"] for r in ok))

    q = quantiles(latencies, n=100) if len(latencies) >= 2 else [0] * 100

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_requests": len(results),
        "successful": len(ok),
        "errors": len(errors),
        "total_elapsed_s": round(elapsed_total, 1),
        "throughput_rps": round(len(ok) / elapsed_total, 2),
        "cost": {
            "actual_usd": round(actual_cost, 6),
            "gpt4o_baseline_usd": round(gpt4o_baseline, 6),
            "saved_usd": round(saved, 6),
            "pct_saved": round(pct_saved, 1),
        },
        "latency_ms": {
            "min":    round(min(latencies), 1) if latencies else 0,
            "median": round(median(latencies), 1) if latencies else 0,
            "mean":   round(mean(latencies), 1) if latencies else 0,
            "p95":    round(q[94], 1) if len(q) >= 95 else 0,
            "p99":    round(q[98], 1) if len(q) >= 99 else 0,
            "max":    round(max(latencies), 1) if latencies else 0,
        },
        "routing": {
            "tier_accuracy_pct": round(tier_accuracy, 1),
            "model_distribution": model_dist,
            "tier_distribution": {str(k): v for k, v in sorted(tier_dist.items())},
        },
        "tokens": {
            "total": sum(tokens),
            "avg_per_request": round(mean(tokens), 1) if tokens else 0,
        },
    }


def print_report(report: dict):
    c = report["cost"]
    l = report["latency_ms"]
    r = report["routing"]

    print("\n" + "=" * 65)
    print("  LLM COST AUTOPILOT — LOAD TEST RESULTS")
    print("=" * 65)
    print(f"\n  {'Requests':<28} {report['successful']}/{report['total_requests']} successful")
    print(f"  {'Duration':<28} {report['total_elapsed_s']}s  ({report['throughput_rps']} req/s)")
    print(f"  {'Errors':<28} {report['errors']}")

    print(f"\n  {'─'*60}")
    print(f"  💰 COST SAVINGS")
    print(f"  {'─'*60}")
    print(f"  {'Actual cost':<28} ${c['actual_usd']:.6f}")
    print(f"  {'GPT-4o baseline':<28} ${c['gpt4o_baseline_usd']:.6f}")
    print(f"  {'Saved':<28} ${c['saved_usd']:.6f}")
    print(f"  {'Reduction':<28} ✅ {c['pct_saved']}% cheaper than GPT-4o")

    print(f"\n  {'─'*60}")
    print(f"  ⚡ LATENCY")
    print(f"  {'─'*60}")
    print(f"  {'Median':<28} {l['median']}ms")
    print(f"  {'p95':<28} {l['p95']}ms")
    print(f"  {'p99':<28} {l['p99']}ms")
    print(f"  {'Max':<28} {l['max']}ms")

    print(f"\n  {'─'*60}")
    print(f"  🧭 ROUTING")
    print(f"  {'─'*60}")
    print(f"  {'Tier accuracy':<28} {r['tier_accuracy_pct']}%")
    for model, count in sorted(r["model_distribution"].items(), key=lambda x: -x[1]):
        pct = count / report["successful"] * 100
        bar = "█" * int(pct / 3)
        print(f"  {model:<28} {count:>4} requests ({pct:.1f}%) {bar}")

    print(f"\n  {'─'*60}")
    print(f"  📄 Report saved → {REPORT_PATH}")
    print("=" * 65 + "\n")


async def main(count: int, concurrency: int):
    # Repeat prompt pool to reach target count
    import itertools
    pool = list(itertools.islice(itertools.cycle(PROMPTS), count))

    semaphore = asyncio.Semaphore(concurrency)

    print(f"\nLoad test: {count} requests | concurrency={concurrency} | target={API_BASE}")
    print("Sending", end="", flush=True)

    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [send_one(client, p, t, semaphore) for p, t in pool]
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            print("." if result["status"] == "ok" else "E", end="", flush=True)
    elapsed = time.perf_counter() - t0

    print()
    report = build_report(results, elapsed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100,
                        help="Number of requests to send (default: 100)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent requests (default: 4 — safe for Groq free tier)")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.concurrency))
