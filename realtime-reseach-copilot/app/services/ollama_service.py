"""
ollama_service.py — Local LLM inference via Ollama (Milestone 4)

Calls the local Ollama server to generate a research answer.
The model is set by OLLAMA_CHAT_MODEL in your .env file.

If Ollama isn't running or the model isn't pulled, the service returns
a clear error message instead of crashing the whole request.
"""

import logging

import ollama

from app.core.config import settings
from app.core.prompts import RESEARCH_SYSTEM_PROMPT, build_user_prompt
from app.models.schemas import SourceResult

logger = logging.getLogger(__name__)


def _build_context_block(sources: list[SourceResult]) -> str:
    """Format all retrieved source snippets into a single context string for the prompt."""
    if not sources:
        return "No sources were retrieved. Answer based on general knowledge only, and be explicit about uncertainty."

    parts: list[str] = []
    for i, src in enumerate(sources, start=1):
        parts.append(
            f"[Source {i} — {src.source_type.upper()}] {src.title}\n"
            f"{src.content}"
        )
    return "\n\n---\n\n".join(parts)


class OllamaService:
    """Sends prompts to a local Ollama model and parses the response."""

    def generate_answer(self, query: str, sources: list[SourceResult]) -> str:
        """
        Ask the local Ollama LLM to synthesize an answer from retrieved sources.

        Returns the full raw markdown response from the model.
        Falls back to a user-friendly error string if Ollama is unavailable.
        """
        context_block = _build_context_block(sources)
        user_prompt = build_user_prompt(query=query, context_block=context_block)

        logger.info(
            "OllamaService: calling model='%s' with %d sources",
            settings.OLLAMA_CHAT_MODEL,
            len(sources),
        )

        try:
            response = ollama.chat(
                model=settings.OLLAMA_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0.2,   # low temperature = more factual, less creative
                    "num_predict": 1024,  # max tokens to generate
                },
            )
            raw_text: str = response["message"]["content"].strip()
            logger.info("OllamaService: generation complete (%d chars)", len(raw_text))
            return raw_text

        except ollama.ResponseError as exc:
            # Model not found — give the user an actionable fix
            if "not found" in str(exc).lower() or "pull" in str(exc).lower():
                model = settings.OLLAMA_CHAT_MODEL
                msg = (
                    f"Model '{model}' is not installed in Ollama.\n"
                    f"Fix: run  `ollama pull {model}`  in your terminal."
                )
            else:
                msg = f"Ollama API error: {exc}"
            logger.error("OllamaService: %s", msg)
            return f"## Answer\n{msg}\n\n## Key Points\n\n## Limitations\nOllama error — see above."

        except Exception as exc:
            # Ollama server not running
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                msg = (
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running: open the Ollama app or run `ollama serve`."
                )
            else:
                msg = f"Unexpected error calling Ollama: {exc}"
            logger.error("OllamaService: %s", msg)
            return f"## Answer\n{msg}\n\n## Key Points\n\n## Limitations\nOllama error — see above."

    def extract_answer_text(self, raw_response: str) -> str:
        """
        Pull out only the prose answer (the ## Answer section).

        The model returns a multi-section markdown response. This extracts
        just the answer paragraph so the UI doesn't duplicate the other sections.
        """
        lines = raw_response.splitlines()
        answer_lines: list[str] = []
        in_answer = False

        for line in lines:
            stripped = line.strip()
            if stripped in ("## Answer", "**Answer**", "# Answer"):
                in_answer = True
                continue
            # Stop at the next section header
            if in_answer and stripped.startswith("##"):
                break
            if in_answer:
                answer_lines.append(line)

        result = "\n".join(answer_lines).strip()
        # Fall back to the full response if parsing fails (model ignored the format)
        return result if result else raw_response

    def extract_key_points(self, raw_response: str) -> list[str]:
        """
        Pull out bullet points from the ## Key Points section.

        Returns an empty list if the section is absent or empty.
        """
        points: list[str] = []
        in_key_points = False

        for line in raw_response.splitlines():
            stripped = line.strip()
            if stripped in ("## Key Points", "**Key Points**", "# Key Points"):
                in_key_points = True
                continue
            if in_key_points and stripped.startswith("##"):
                break
            if in_key_points and stripped.startswith("- "):
                points.append(stripped[2:].strip())
            # Also handle numbered lists: "1. point"
            elif in_key_points and len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".):":
                points.append(stripped[2:].strip())

        return points
