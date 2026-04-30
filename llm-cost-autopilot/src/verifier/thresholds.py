"""
Quality thresholds per task type.
Score is 1–5 (LLM-as-judge). Below min_score → routing failure → escalate.
"""

from dataclasses import dataclass
from enum import Enum
import re


class TaskType(str, Enum):
    EXTRACTION = "extraction"
    FACTUAL_QA = "factual_qa"
    TRANSLATION = "translation"
    GRAMMAR_FIX = "grammar_fix"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    REASONING = "reasoning"
    GENERAL = "general"


@dataclass
class QualityThreshold:
    task_type: TaskType
    min_score: float        # 1–5 scale; below this → escalate
    description: str


THRESHOLDS: dict[TaskType, QualityThreshold] = {
    TaskType.EXTRACTION: QualityThreshold(
        TaskType.EXTRACTION, min_score=4.0,
        description="All key fields extracted correctly with no hallucination",
    ),
    TaskType.FACTUAL_QA: QualityThreshold(
        TaskType.FACTUAL_QA, min_score=4.0,
        description="Factually correct, concise, no fabrication",
    ),
    TaskType.TRANSLATION: QualityThreshold(
        TaskType.TRANSLATION, min_score=3.5,
        description="Accurate meaning, natural phrasing in target language",
    ),
    TaskType.GRAMMAR_FIX: QualityThreshold(
        TaskType.GRAMMAR_FIX, min_score=4.0,
        description="All errors corrected, meaning preserved",
    ),
    TaskType.SUMMARIZATION: QualityThreshold(
        TaskType.SUMMARIZATION, min_score=3.5,
        description="Key points covered, no distortion, appropriate length",
    ),
    TaskType.CLASSIFICATION: QualityThreshold(
        TaskType.CLASSIFICATION, min_score=4.0,
        description="Correct label assigned with valid reasoning",
    ),
    TaskType.CODE_GENERATION: QualityThreshold(
        TaskType.CODE_GENERATION, min_score=3.5,
        description="Code is correct, readable, and handles edge cases",
    ),
    TaskType.ANALYSIS: QualityThreshold(
        TaskType.ANALYSIS, min_score=3.0,
        description="Covers main dimensions, balanced, no major omissions",
    ),
    TaskType.CREATIVE: QualityThreshold(
        TaskType.CREATIVE, min_score=3.0,
        description="Meets the brief, coherent, appropriately creative",
    ),
    TaskType.REASONING: QualityThreshold(
        TaskType.REASONING, min_score=3.5,
        description="Logical steps present, conclusion follows from premises",
    ),
    TaskType.GENERAL: QualityThreshold(
        TaskType.GENERAL, min_score=3.5,
        description="Helpful, accurate, and appropriately detailed",
    ),
}

# Keyword patterns to auto-detect task type from prompt
_TASK_PATTERNS: list[tuple[TaskType, re.Pattern]] = [
    (TaskType.EXTRACTION,      re.compile(r"\b(extract|find all|list all.*from|pull out)\b", re.I)),
    (TaskType.TRANSLATION,     re.compile(r"\b(translate|in (spanish|french|german|japanese|portuguese|chinese))\b", re.I)),
    (TaskType.GRAMMAR_FIX,     re.compile(r"\b(fix|correct|grammar|spelling|punctuation|rewrite formally)\b", re.I)),
    (TaskType.SUMMARIZATION,   re.compile(r"\b(summarize|summary|bullet points|key points|tldr)\b", re.I)),
    (TaskType.CLASSIFICATION,  re.compile(r"\b(classify|categorize|label|sentiment|spam|positive|negative)\b", re.I)),
    (TaskType.CODE_GENERATION, re.compile(r"\b(write a (function|class|script|query|code)|implement|sql|python|regex)\b", re.I)),
    (TaskType.ANALYSIS,        re.compile(r"\b(analyze|analyse|pros and cons|compare|tradeoff|evaluate)\b", re.I)),
    (TaskType.REASONING,       re.compile(r"\b(step.by.step|walk through|explain.*how|design|architect|strategy)\b", re.I)),
    (TaskType.CREATIVE,        re.compile(r"\b(write a (story|essay|email|bio|description|blog|paragraph))\b", re.I)),
    (TaskType.FACTUAL_QA,      re.compile(r"\b(what is|who is|when|where|how many|what does)\b", re.I)),
]


def detect_task_type(prompt: str) -> TaskType:
    for task_type, pattern in _TASK_PATTERNS:
        if pattern.search(prompt):
            return task_type
    return TaskType.GENERAL


def get_threshold(prompt: str) -> QualityThreshold:
    task_type = detect_task_type(prompt)
    return THRESHOLDS[task_type]
