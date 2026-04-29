from __future__ import annotations

from meeting_notes.models import EvaluationResult, MeetingNotes


MISSING_OWNER_VALUES = {"", "unassigned", "unknown", "not specified", "none"}
MISSING_DEADLINE_VALUES = {"", "not specified", "unknown", "none", "tbd"}


def evaluate_notes(notes: MeetingNotes) -> EvaluationResult:
    total = len(notes.action_items)
    if total == 0:
        return EvaluationResult(
            action_item_count=0,
            owner_coverage=0.0,
            deadline_coverage=0.0,
            completeness_score=0.0,
            warnings=["No action items were extracted. Check whether the transcript contains tasks."],
        )

    owner_count = sum(1 for item in notes.action_items if item.owner.strip().lower() not in MISSING_OWNER_VALUES)
    deadline_count = sum(1 for item in notes.action_items if item.deadline.strip().lower() not in MISSING_DEADLINE_VALUES)
    owner_coverage = owner_count / total
    deadline_coverage = deadline_count / total
    completeness_score = (owner_coverage + deadline_coverage) / 2

    warnings: list[str] = []
    if owner_coverage < 1:
        warnings.append("Some action items are missing a responsible owner.")
    if deadline_coverage < 1:
        warnings.append("Some action items are missing a deadline.")

    return EvaluationResult(
        action_item_count=total,
        owner_coverage=owner_coverage,
        deadline_coverage=deadline_coverage,
        completeness_score=completeness_score,
        warnings=warnings,
    )

