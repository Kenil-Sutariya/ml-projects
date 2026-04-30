"""
LLM-as-judge: asks the highest-tier model to score a cheaper model's response.
Returns a float score 1.0–5.0 and a short rationale.
"""

import re
from dataclasses import dataclass

from src.models.client import send_request
from src.models.registry import get_model

JUDGE_MODEL_KEY = "groq-llama3-70b"

_JUDGE_SYSTEM = """You are a strict but fair quality evaluator for LLM responses.
Your job is to score a response on a scale from 1 to 5 based on accuracy, completeness, and helpfulness.

Scoring rubric:
5 - Perfect: accurate, complete, well-structured, no issues
4 - Good: mostly correct with minor omissions or imprecision
3 - Acceptable: covers the basics but misses important aspects
2 - Poor: significant errors, major omissions, or misleading content
1 - Unacceptable: wrong, harmful, or completely off-topic

You MUST respond in exactly this format (no other text):
SCORE: <number 1-5>
RATIONALE: <one sentence explanation>"""

_JUDGE_PROMPT = """Evaluate this response to the given prompt.

PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}

Rate the quality of the response above."""


@dataclass
class JudgeResult:
    score: float           # 1.0 – 5.0
    rationale: str
    judge_model: str
    raw_output: str


async def judge_response(prompt: str, response_text: str) -> JudgeResult:
    judge_config = get_model(JUDGE_MODEL_KEY)

    judge_prompt = _JUDGE_PROMPT.format(prompt=prompt, response=response_text)

    judge_response = await send_request(
        prompt=judge_prompt,
        config=judge_config,
        system=_JUDGE_SYSTEM,
    )

    raw = judge_response.text.strip()
    score, rationale = _parse_judge_output(raw)

    return JudgeResult(
        score=score,
        rationale=rationale,
        judge_model=JUDGE_MODEL_KEY,
        raw_output=raw,
    )


def _parse_judge_output(raw: str) -> tuple[float, str]:
    score_match = re.search(r"SCORE:\s*([1-5](?:\.\d+)?)", raw, re.IGNORECASE)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", raw, re.IGNORECASE)

    score = float(score_match.group(1)) if score_match else 3.0
    rationale = rationale_match.group(1).strip() if rationale_match else raw[:200]

    # Clamp to valid range
    score = max(1.0, min(5.0, score))
    return score, rationale
