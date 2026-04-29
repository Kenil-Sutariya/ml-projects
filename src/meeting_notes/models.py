from __future__ import annotations

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str = Field(..., description="Detailed task description with enough context to execute.")
    owner: str = Field(..., description="Responsible person or team.")
    deadline: str = Field(..., description="Deadline as stated or inferred from the meeting.")
    evidence: str = Field(
        default="",
        description="Short transcript-based reason for the extracted action item.",
    )


class MeetingNotes(BaseModel):
    summary: str
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    action_item_count: int
    owner_coverage: float
    deadline_coverage: float
    completeness_score: float
    warnings: list[str] = Field(default_factory=list)

