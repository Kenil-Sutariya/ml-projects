"""
Phase 3 integration test — routes 6 prompts (mix of tiers), runs the
full verification loop, and prints a quality audit table.

Run:
    SKIP_OPENAI=1 SKIP_ANTHROPIC=1 python tests/test_verifier.py
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.logger import log_request_with_prompt, log_verification
from src.db.schema import engine  # initialises DB
from src.models.client import send_request
from src.router.router import route
from src.verifier.verifier import verify

TEST_PROMPTS = [
    # Tier 1 — expect cheap model to pass quality check
    "What is the capital of Japan?",
    "Fix the grammar: 'He don't like pizza'",
    # Tier 2
    "Summarize in 3 bullet points: Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. Common algorithms include decision trees, neural networks, and support vector machines.",
    "Classify the sentiment: 'The delivery was late but the product quality is outstanding.'",
    # Tier 3
    "Explain the CAP theorem and describe a real-world scenario where you would choose availability over consistency.",
    "Write a Python function implementing binary search. Explain its time complexity and edge cases.",
]


async def main():
    print("\n" + "=" * 100)
    print("PHASE 3 — QUALITY VERIFICATION LOOP TEST")
    print("=" * 100)

    results = []

    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        # Step 1: route + call cheap model
        decision = route(prompt)
        response = await send_request(prompt, decision.model)

        # Step 2: log to DB
        log_request_with_prompt(prompt, response, decision)

        print(f"  Routed to : {decision.model_key} (tier {decision.tier}, conf={decision.confidence:.2f})")
        print(f"  Latency   : {response.latency_ms:.0f}ms")
        print(f"  Response  : {response.text[:120]}...")

        # Step 3: verify (synchronous for test visibility)
        print(f"  Verifying with judge ({decision.model_key} → judge)...")
        vr = await verify(
            prompt=prompt,
            original_response=response,
            decision=decision,
            escalate_on_failure=True,
        )

        status = "PASS" if vr.passed else "FAIL → ESCALATED" if vr.escalated else "FAIL"
        print(f"  Quality   : {vr.judge_result.score:.1f}/{vr.threshold:.1f} [{status}]")
        print(f"  Rationale : {vr.judge_result.rationale}")

        log_verification(vr)

        if vr.escalated and vr.escalated_response:
            print(f"  Escalated : {vr.escalation_model_key} | cost_delta=${vr.cost_delta_usd:.6f}")
            print(f"  Better ans: {vr.escalated_response.text[:120]}...")

        results.append({
            "prompt": prompt[:55],
            "tier": decision.tier,
            "model": decision.model_key,
            "score": vr.judge_result.score,
            "threshold": vr.threshold,
            "passed": vr.passed,
            "escalated": vr.escalated,
        })

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Prompt':<56} {'Tier':>5} {'Model':<20} {'Score':>7} {'Thresh':>7} {'Pass':>5} {'Esc':>5}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['prompt']:<56} {r['tier']:>5} {r['model']:<20} "
            f"{r['score']:>7.1f} {r['threshold']:>7.1f} "
            f"{'Y' if r['passed'] else 'N':>5} {'Y' if r['escalated'] else 'N':>5}"
        )

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    escalated = sum(1 for r in results if r["escalated"])
    print(f"\nSummary: {passed}/{total} passed quality check | {escalated} escalated")
    print(f"DB written to: data/autopilot.db")


if __name__ == "__main__":
    asyncio.run(main())
