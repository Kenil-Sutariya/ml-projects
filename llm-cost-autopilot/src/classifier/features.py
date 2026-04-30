import re
from dataclasses import dataclass, asdict

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

# Keywords that signal increasing complexity
_SIMPLE_VERBS = re.compile(
    r"\b(what is|who is|when|where|define|name|list|convert|translate|fix|correct|"
    r"spell|count|extract|format|reverse|find)\b",
    re.IGNORECASE,
)
_MODERATE_VERBS = re.compile(
    r"\b(summarize|summary|classify|classify|compare|explain|describe|identify|"
    r"analyze|analyse|evaluate|rewrite|paraphrase|categorize|draft|write a|create a)\b",
    re.IGNORECASE,
)
_COMPLEX_VERBS = re.compile(
    r"\b(design|architect|develop|implement|build|debate|argue|critique|"
    r"strategy|tradeoff|trade-off|step.by.step|walk through|reason|justify|"
    r"evaluate.*and|compare.*contrast|implications|framework)\b",
    re.IGNORECASE,
)
_CONSTRAINT_WORDS = re.compile(
    r"\b(must|should|ensure|require|constraint|without|while|but|however|"
    r"consider|including|covering|also|and then|additionally)\b",
    re.IGNORECASE,
)
_OUTPUT_FORMAT = re.compile(
    r"\b(json|yaml|table|bullet|list|paragraph|essay|report|document|"
    r"diagram|schema|prd|spec|specification|code)\b",
    re.IGNORECASE,
)
_MULTI_STEP = re.compile(
    r"\b(step.by.step|first.*then|phase|stage|finally|in addition|"
    r"part \d|section|detailed plan|comprehensive)\b",
    re.IGNORECASE,
)


@dataclass
class PromptFeatures:
    token_count: int
    word_count: int
    char_count: int
    sentence_count: int
    avg_word_length: float
    simple_verb_count: int
    moderate_verb_count: int
    complex_verb_count: int
    constraint_count: int
    output_format_signals: int
    multi_step_signals: int
    question_count: int          # number of '?' in prompt
    has_code_or_data: int        # backtick, indented block, or explicit data
    has_context_provided: int    # colon followed by long text, or quotes
    unique_word_ratio: float     # vocabulary richness

    def to_array(self) -> list[float]:
        return list(asdict(self).values())

    @staticmethod
    def feature_names() -> list[str]:
        return list(PromptFeatures.__dataclass_fields__.keys())


def extract_features(prompt: str) -> PromptFeatures:
    tokens = _ENC.encode(prompt)
    words = prompt.split()
    sentences = re.split(r"[.!?]+", prompt)
    sentences = [s.strip() for s in sentences if s.strip()]

    unique_words = set(w.lower().strip(".,!?;:") for w in words)
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    unique_ratio = len(unique_words) / max(len(words), 1)

    has_code = int(bool(re.search(r"`|def |class |SELECT |import |{.*}|\[.*\]", prompt)))
    # Context = a colon followed by 20+ chars, or content in quotes longer than 30 chars
    has_context = int(
        bool(re.search(r":\s.{20,}", prompt))
        or bool(re.search(r"['\"].{30,}['\"]", prompt))
    )

    return PromptFeatures(
        token_count=len(tokens),
        word_count=len(words),
        char_count=len(prompt),
        sentence_count=len(sentences),
        avg_word_length=round(avg_word_len, 2),
        simple_verb_count=len(_SIMPLE_VERBS.findall(prompt)),
        moderate_verb_count=len(_MODERATE_VERBS.findall(prompt)),
        complex_verb_count=len(_COMPLEX_VERBS.findall(prompt)),
        constraint_count=len(_CONSTRAINT_WORDS.findall(prompt)),
        output_format_signals=len(_OUTPUT_FORMAT.findall(prompt)),
        multi_step_signals=len(_MULTI_STEP.findall(prompt)),
        question_count=prompt.count("?"),
        has_code_or_data=has_code,
        has_context_provided=has_context,
        unique_word_ratio=round(unique_ratio, 3),
    )
