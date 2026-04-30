"""
Seed the DB with 14 days of realistic synthetic request history.
Simulates a production system handling ~75 requests/day across all tiers.

Run:
    python scripts/seed_data.py
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import insert

from src.db.schema import engine, requests_table

random.seed(42)

MODELS = [
    # (model_key, provider, tier, input_cost, output_cost, avg_latency, avg_tokens_in, avg_tokens_out)
    ("llama3.2",        "ollama", 1, 0.0,               0.0,               3000, 30,  80),
    ("groq-llama3-8b",  "groq",   2, 0.0,               0.0,               700,  80,  250),
    ("groq-llama3-70b", "groq",   3, 0.59/1_000_000,    0.79/1_000_000,    1500, 150, 600),
]

# Tier distribution: 50% tier1, 35% tier2, 15% tier3
TIER_WEIGHTS = [0.50, 0.35, 0.15]

SAMPLE_PROMPTS = {
    1: [
        "What is the capital of Germany?",
        "Convert 72°F to Celsius.",
        "Fix grammar: 'She don't want to go'",
        "Extract emails from: 'reach us at hello@co.com'",
        "Translate 'goodbye' to French.",
        "What does API stand for?",
        "What is 8 * 7?",
        "Define the word 'verbose' briefly.",
    ],
    2: [
        "Summarize this article in 3 bullet points: [text]",
        "Classify this review as positive/neutral/negative: [review]",
        "Explain the difference between TCP and UDP.",
        "Write a short product description for wireless earbuds.",
        "List pros and cons of remote work.",
        "What is the difference between authentication and authorization?",
        "Write a Python function to check if a string is a palindrome.",
        "Explain what a webhook is with a use case.",
    ],
    3: [
        "Design a system architecture for a ride-sharing app.",
        "Explain the CAP theorem and give a real-world example.",
        "Write a binary search implementation and analyze complexity.",
        "Compare transformer models vs graph neural networks for drug discovery.",
        "Design a rate-limiting system for 100k req/s.",
        "Write a post-mortem for a 4-hour database outage.",
        "Analyze second-order effects of UBI on AI-driven job displacement.",
        "Design a multi-tenant SaaS database schema with audit logging.",
    ],
}

JUDGE_RATIONALES = [
    "The response is accurate, complete, and well-structured.",
    "Correct answer with appropriate detail level.",
    "Covers the key points without unnecessary padding.",
    "Good analysis but missed one minor aspect.",
    "Accurate and concise, directly addresses the prompt.",
    "Well-reasoned with clear examples provided.",
    "Mostly correct with a slight omission in edge cases.",
    "Perfect response — accurate, complete, no issues.",
]


def generate_rows(n_days: int = 14, requests_per_day: int = 75) -> list[dict]:
    rows = []
    now = datetime.utcnow()

    for day_offset in range(n_days, 0, -1):
        day = now - timedelta(days=day_offset)
        # Vary volume slightly per day
        daily_count = int(requests_per_day * random.uniform(0.7, 1.3))

        for _ in range(daily_count):
            tier_idx = random.choices([0, 1, 2], weights=TIER_WEIGHTS)[0]
            model_key, provider, tier, in_cost, out_cost, base_latency, avg_in, avg_out = MODELS[tier_idx]

            in_tokens  = max(10, int(random.gauss(avg_in,  avg_in  * 0.3)))
            out_tokens = max(10, int(random.gauss(avg_out, avg_out * 0.4)))
            latency    = max(100, int(random.gauss(base_latency, base_latency * 0.25)))
            cost       = in_tokens * in_cost + out_tokens * out_cost

            # 92% of requests are verified (background worker catches up)
            verified = random.random() < 0.92
            quality_score = None
            quality_passed = None
            judge_rationale = None
            escalated = None
            escalation_model = None
            cost_delta = None
            quality_gap = None

            if verified:
                # Tier 1 scores higher on average (simpler tasks)
                mean_score = {1: 4.7, 2: 4.3, 3: 3.9}[tier]
                score = round(min(5.0, max(1.0, random.gauss(mean_score, 0.4))), 1)
                threshold = {1: 4.0, 2: 3.5, 3: 3.5}[tier]
                passed = score >= threshold
                quality_score = score
                quality_passed = int(passed)
                judge_rationale = random.choice(JUDGE_RATIONALES)

                if not passed:
                    # ~5% failure rate → escalated
                    if random.random() < 0.85 and tier < 3:
                        escalated = 1
                        escalation_model = MODELS[tier_idx + 1][0]
                        cost_delta = random.uniform(0.00001, 0.0005)
                        quality_gap = threshold - score
                    else:
                        escalated = 0
                else:
                    escalated = 0

            # Random timestamp within this day
            ts = day + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            prompt = random.choice(SAMPLE_PROMPTS[tier])

            rows.append({
                "timestamp":        ts.isoformat(),
                "prompt_hash":      f"{hash(prompt + ts.isoformat()) & 0xFFFFFFFF:08x}",
                "prompt_preview":   prompt,
                "complexity_tier":  tier,
                "tier_confidence":  round(random.uniform(0.60, 0.99), 2),
                "routed_model":     model_key,
                "provider":         provider,
                "input_tokens":     in_tokens,
                "output_tokens":    out_tokens,
                "latency_ms":       latency,
                "cost_usd":         round(cost, 8),
                "quality_score":    quality_score,
                "quality_threshold": {1: 4.0, 2: 3.5, 3: 3.5}[tier] if verified else None,
                "quality_passed":   quality_passed,
                "judge_rationale":  judge_rationale,
                "escalated":        escalated,
                "escalation_model": escalation_model,
                "cost_delta_usd":   cost_delta,
                "quality_gap":      quality_gap,
            })

    return rows


if __name__ == "__main__":
    rows = generate_rows(n_days=14, requests_per_day=75)

    with engine().begin() as conn:
        conn.execute(insert(requests_table), rows)

    print(f"Seeded {len(rows)} rows across 14 days.")

    # Quick sanity check
    from src.db.queries import get_summary_stats
    stats = get_summary_stats()
    print(f"\nSummary after seeding:")
    print(f"  Total requests : {stats['total_requests']}")
    print(f"  Actual cost    : ${stats['actual_cost']:.4f}")
    print(f"  GPT-4o baseline: ${stats['baseline_cost']:.4f}")
    print(f"  Saved          : ${stats['saved_usd']:.4f} ({stats['pct_saved']}%)")
    print(f"  Quality pass   : {stats['quality_pass_rate']}%")
    print(f"  Escalation rate: {stats['escalation_rate']}%")
