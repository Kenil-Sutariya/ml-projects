from meeting_notes.evaluation import evaluate_notes
from meeting_notes.exporter import notes_to_markdown
from meeting_notes.models import ActionItem, MeetingNotes


def test_markdown_export_contains_owner_and_deadline():
    notes = MeetingNotes(
        summary="Team reviewed launch tasks.",
        decisions=["Use OpenAI and Ollama."],
        action_items=[
            ActionItem(
                task="Prepare the sales report and submit it to Mr. Kenil.",
                owner="Rahul",
                deadline="Friday",
                evidence="Rahul committed to the report.",
            )
        ],
    )

    markdown = notes_to_markdown(notes)

    assert "Owner: Rahul" in markdown
    assert "Deadline: Friday" in markdown
    assert "Prepare the sales report" in markdown


def test_evaluation_warns_about_missing_deadline():
    notes = MeetingNotes(
        summary="Team reviewed launch tasks.",
        action_items=[
            ActionItem(
                task="Prepare a launch checklist.",
                owner="Priya",
                deadline="Not specified",
            )
        ],
    )

    result = evaluate_notes(notes)

    assert result.deadline_coverage == 0
    assert result.warnings == ["Some action items are missing a deadline."]

