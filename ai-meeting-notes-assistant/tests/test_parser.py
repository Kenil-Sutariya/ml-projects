from meeting_notes.parser import parse_meeting_notes


def test_parse_json_inside_code_fence():
    raw = """```json
{
  "summary": "Launch plan reviewed.",
  "decisions": ["Keep launch date."],
  "action_items": [
    {
      "task": "Prepare sales report and submit it to Mr. Kenil.",
      "owner": "Rahul",
      "deadline": "Friday",
      "evidence": "Rahul agreed in the meeting."
    }
  ],
  "follow_ups": []
}
```"""

    notes = parse_meeting_notes(raw)

    assert notes.summary == "Launch plan reviewed."
    assert notes.action_items[0].owner == "Rahul"


def test_parse_json_with_intro_text():
    raw = """Here is the JSON:
{
  "summary": "Team discussed PDF export.",
  "decisions": [],
  "action_items": [],
  "follow_ups": ["Retest PDF export."]
}
"""

    notes = parse_meeting_notes(raw)

    assert notes.follow_ups == ["Retest PDF export."]

