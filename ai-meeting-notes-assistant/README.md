# AI Meeting Notes Assistant

A beginner-friendly AI Engineer portfolio project that turns raw meeting transcripts into structured notes:

- Meeting summary
- Decisions
- Action items
- Responsible person
- Deadline
- JSON, Markdown, and PDF exports

The project is intentionally practical. It shows data handling, LLM prompting, structured output parsing, validation, evaluation checks, and a simple Streamlit deployment.

## Demo Flow

1. Upload a `.txt` meeting transcript or use the sample transcript.
2. Choose a provider:
   - `Demo heuristic` works offline and is useful for testing.
   - `OpenAI` uses the OpenAI API.
   - `Ollama` uses a locally running Ollama model.
3. Generate structured meeting notes.
4. Review JSON, Markdown, quality checks, and exports.

## Project Structure

```text
ai-meeting-notes-assistant/
├── app.py
├── data/
│   └── sample_transcript.txt
├── examples/
│   ├── sample_output.json
│   └── sample_output.md
├── src/
│   └── meeting_notes/
│       ├── exporter.py
│       ├── evaluation.py
│       ├── llm.py
│       ├── models.py
│       └── parser.py
└── tests/
    ├── test_exporter.py
    └── test_parser.py
```

## Setup

```bash
cd ai-meeting-notes-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For OpenAI, add your API key to `.env`:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

For Ollama, install Ollama, pull a model, and keep the local server running:

```bash
ollama pull llama3.1
ollama serve
```

## Run the App

```bash
streamlit run app.py
```

## Run Tests

```bash
pytest
```

## Why This Is a Good AI Engineer Project

This is more than a summarizer. It demonstrates the workflow used in real AI products:

- Input handling: upload and validate transcript text.
- Prompt design: ask for a strict JSON schema.
- Model integration: support cloud and local LLM providers.
- Structured output: parse and validate model responses with Pydantic.
- Evaluation: measure whether action items have owners and deadlines.
- Deployment: package the workflow in a Streamlit interface.
- Export: convert generated notes into Markdown and PDF for real use.

## Expected JSON Shape

```json
{
  "summary": "The team reviewed launch progress, assigned final tasks, and confirmed deadlines.",
  "decisions": [
    "Launch date remains Friday.",
    "Rahul will own the sales report."
  ],
  "action_items": [
    {
      "task": "Prepare the sales report, include regional numbers, and submit it to Mr. Kenil for review.",
      "owner": "Rahul",
      "deadline": "Friday, May 3",
      "evidence": "Rahul confirmed he would prepare the sales report by Friday."
    }
  ],
  "follow_ups": [
    "Confirm final design approval after QA feedback."
  ]
}
```

## Portfolio Talking Points

- The app uses a schema-first approach so the output can feed downstream systems like task trackers or CRMs.
- Validation protects the user from malformed model output.
- Evaluation metrics reveal whether the generated notes are operationally useful, not only fluent.
- The provider abstraction makes it easy to compare OpenAI and local Ollama models.

