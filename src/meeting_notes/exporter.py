from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from meeting_notes.models import MeetingNotes


def notes_to_markdown(notes: MeetingNotes) -> str:
    lines = [
        "# Meeting Notes",
        "",
        "## Summary",
        notes.summary,
        "",
        "## Decisions",
    ]

    if notes.decisions:
        lines.extend(f"- {decision}" for decision in notes.decisions)
    else:
        lines.append("- No decisions captured.")

    lines.extend(["", "## Action Items"])
    if notes.action_items:
        for index, item in enumerate(notes.action_items, start=1):
            lines.extend(
                [
                    f"### {index}. {item.task}",
                    f"- Owner: {item.owner}",
                    f"- Deadline: {item.deadline}",
                    f"- Evidence: {item.evidence or 'Not provided'}",
                    "",
                ]
            )
    else:
        lines.append("- No action items captured.")

    lines.extend(["", "## Follow-ups"])
    if notes.follow_ups:
        lines.extend(f"- {follow_up}" for follow_up in notes.follow_ups)
    else:
        lines.append("- No follow-ups captured.")

    return "\n".join(lines).strip() + "\n"


def notes_to_pdf(notes: MeetingNotes) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, title="Meeting Notes")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Meeting Notes", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Summary", styles["Heading2"]),
        Paragraph(notes.summary, styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Decisions", styles["Heading2"]),
        _bullet_list(notes.decisions or ["No decisions captured."], styles),
        Spacer(1, 12),
        Paragraph("Action Items", styles["Heading2"]),
    ]

    if notes.action_items:
        for index, item in enumerate(notes.action_items, start=1):
            story.append(Paragraph(f"{index}. {item.task}", styles["Heading3"]))
            story.append(Paragraph(f"<b>Owner:</b> {item.owner}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Deadline:</b> {item.deadline}", styles["BodyText"]))
            if item.evidence:
                story.append(Paragraph(f"<b>Evidence:</b> {item.evidence}", styles["BodyText"]))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No action items captured.", styles["BodyText"]))

    story.extend(
        [
            Spacer(1, 12),
            Paragraph("Follow-ups", styles["Heading2"]),
            _bullet_list(notes.follow_ups or ["No follow-ups captured."], styles),
        ]
    )

    doc.build(story)
    return buffer.getvalue()


def _bullet_list(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyText"])) for item in items],
        bulletType="bullet",
    )
