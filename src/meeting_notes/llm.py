from __future__ import annotations

import os
import re
from typing import Literal

from meeting_notes.models import ActionItem, MeetingNotes
from meeting_notes.parser import parse_meeting_notes


Provider = Literal["Demo heuristic", "OpenAI", "Ollama"]


SYSTEM_PROMPT = """You are an AI meeting notes assistant.
Return only valid JSON with this shape:
{
  "summary": "short but useful meeting summary",
  "decisions": ["decision 1"],
  "action_items": [
    {
      "task": "detailed task description with enough context to execute",
      "owner": "person or team responsible",
      "deadline": "deadline exactly as stated, or a careful inference",
      "evidence": "short reason from the transcript"
    }
  ],
  "follow_ups": ["open question or dependency"]
}

Rules:
- Make action item tasks specific enough for a real project tracker.
- If the deadline is relative, keep the relative wording and include any date mentioned.
- If an owner or deadline is missing, use "Unassigned" or "Not specified".
- Do not invent decisions that are not supported by the transcript.
"""


def generate_meeting_notes(
    transcript: str,
    provider: Provider,
    model_name: str,
    ollama_base_url: str | None = None,
) -> MeetingNotes:
    if provider == "Demo heuristic":
        return demo_heuristic_notes(transcript)
    if provider == "OpenAI":
        return openai_notes(transcript, model_name)
    if provider == "Ollama":
        return ollama_notes(transcript, model_name, ollama_base_url)
    raise ValueError(f"Unsupported provider: {provider}")


def openai_notes(transcript: str, model_name: str) -> MeetingNotes:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the openai package with `pip install -r requirements.txt`.") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return parse_meeting_notes(content)


def ollama_notes(transcript: str, model_name: str, ollama_base_url: str | None = None) -> MeetingNotes:
    import requests

    base_url = (ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_name,
            "format": "json",
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return parse_meeting_notes(content)


def demo_heuristic_notes(transcript: str) -> MeetingNotes:
    """Small offline baseline so the project works without paid APIs."""
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    speaker_lines = [line for line in lines if ":" in line]
    summary = _heuristic_summary(speaker_lines or lines)
    decisions = _find_decisions(lines)
    action_items = _find_action_items(lines)
    follow_ups = _find_follow_ups(lines)

    return MeetingNotes(
        summary=summary,
        decisions=decisions,
        action_items=action_items,
        follow_ups=follow_ups,
    )


def _heuristic_summary(lines: list[str]) -> str:
    if not lines:
        return "No transcript content was provided."
    first_topics = " ".join(_remove_speaker(line) for line in lines[:4])
    return (
        "The meeting covered project progress, decisions, and next steps. "
        f"Key discussion included: {first_topics[:420].strip()}"
    )


def _find_decisions(lines: list[str]) -> list[str]:
    decision_markers = ("decided", "decision", "approved", "agreed", "confirmed", "final")
    decisions: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(marker in lower for marker in decision_markers):
            decisions.append(_remove_speaker(line))
    return decisions[:8]


def _find_action_items(lines: list[str]) -> list[ActionItem]:
    action_words = ("will", "need to", "needs to", "please", "prepare", "send", "share", "follow up", "complete")
    items: list[ActionItem] = []

    for line in lines:
        lower = line.lower()
        if not any(word in lower for word in action_words):
            continue
        owner = _extract_owner(line)
        deadline = _extract_deadline(line)
        task = _task_from_line(line)
        items.append(ActionItem(task=task, owner=owner, deadline=deadline, evidence=_remove_speaker(line)))

    return _dedupe_actions(items)[:10]


def _find_follow_ups(lines: list[str]) -> list[str]:
    markers = ("follow up", "open question", "blocked", "waiting", "check with", "confirm")
    follow_ups = [_remove_speaker(line) for line in lines if any(marker in line.lower() for marker in markers)]
    return follow_ups[:8]


def _extract_owner(line: str) -> str:
    speaker_match = re.match(r"^\s*([^:]{2,40}):", line)
    if speaker_match:
        speaker = speaker_match.group(1).strip()
        if speaker.lower() not in {"team", "all", "everyone"}:
            return speaker

    patterns = [
        r"\b([A-Z][a-z]+)\s+will\b",
        r"\b([A-Z][a-z]+)\s+needs?\s+to\b",
        r"\bassign(?:ed)?\s+to\s+([A-Z][a-z]+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return "Unassigned"


def _extract_deadline(line: str) -> str:
    date_patterns = [
        r"\b(?:by|before|on|due)\s+([A-Z][a-z]+day(?:,\s+[A-Z][a-z]+\s+\d{1,2})?)",
        r"\b(?:by|before|on|due)\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)",
        r"\b(?:by|before|on|due)\s+(tomorrow|today|next week|end of week|EOD|Friday|Monday|Tuesday|Wednesday|Thursday)",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "Not specified"


def _task_from_line(line: str) -> str:
    text = _remove_speaker(line)
    text = re.sub(r"\b(I|we)\s+will\b", "Will", text, flags=re.IGNORECASE)
    return text[0].upper() + text[1:] if text else "Review transcript item."


def _remove_speaker(line: str) -> str:
    return re.sub(r"^\s*[^:]{2,40}:\s*", "", line).strip()


def _dedupe_actions(items: list[ActionItem]) -> list[ActionItem]:
    seen: set[str] = set()
    deduped: list[ActionItem] = []
    for item in items:
        key = item.task.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
