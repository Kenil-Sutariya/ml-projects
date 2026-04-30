"""
prompts.py — Prompt templates for the local Ollama LLM

Design notes:
- Local models are less reliable at returning strict JSON, so we ask for
  a simple markdown structure instead. The OllamaService parses it with
  plain string splitting.
- The system prompt instructs the model to stay grounded in the provided
  context, reducing hallucination.
"""


# ── System prompt ────────────────────────────────────────────────────────────
# Sent as the "system" role message — sets the model's overall behaviour.
RESEARCH_SYSTEM_PROMPT = """You are a careful, honest research assistant.

Your job:
1. Answer the user's question using ONLY the context snippets provided below.
2. If the context is weak or missing, say so clearly — do NOT make things up.
3. Structure your response exactly as shown in the format below.
4. Keep your answer concise and factual.
5. Cite which source type (Wikipedia / Web / Private KB) each key point comes from.

Response format (use exactly these markdown headers):

## Answer
<Write a clear, direct answer in 2-4 sentences>

## Key Points
- <Point 1 — cite source type>
- <Point 2 — cite source type>
- <Point 3 — cite source type>

## Limitations
<Note any gaps, missing context, or uncertainty. Write "None" if context is strong.>
"""


def build_user_prompt(query: str, context_block: str) -> str:
    """
    Builds the user-role message that is sent to the model.

    Args:
        query: The user's original research question.
        context_block: All retrieved source snippets, pre-formatted as text.

    Returns:
        A formatted string ready to pass as the user message.
    """
    return f"""Research Question:
{query}

--- Retrieved Context ---
{context_block}
--- End of Context ---

Please answer the research question using only the context above.
Follow the required response format exactly.
"""
