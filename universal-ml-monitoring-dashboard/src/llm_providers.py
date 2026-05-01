"""
Cloud LLM provider presets.

All listed providers expose an OpenAI-compatible /chat/completions endpoint, so
the same `cloud_llm_client.generate_cloud_response` works for all of them.

API keys, base URLs, and default model names are read from environment
variables (typically loaded from .env at app start).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# python-dotenv is in requirements.txt
try:
    from dotenv import load_dotenv
except ImportError:  # graceful fallback if dotenv is missing
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False


def load_env_file(env_path: Path | None = None) -> bool:
    """Load .env from project root (or the given path). Returns True if loaded."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        return True
    return False


@dataclass(frozen=True)
class CloudProviderPreset:
    name:           str
    env_key:        str
    env_base_url:   str
    env_model:      str
    default_url:    str
    default_model:  str
    docs_url:       str

    @property
    def api_key(self) -> str:
        # Strip whitespace AND leading/trailing quotes that users sometimes paste
        raw = os.getenv(self.env_key, "").strip()
        return raw.strip('"').strip("'").strip()

    @property
    def base_url(self) -> str:
        return os.getenv(self.env_base_url, "").strip() or self.default_url

    @property
    def model_name(self) -> str:
        return os.getenv(self.env_model, "").strip() or self.default_model

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


# Order matters — shown in the dropdown in this order.
PRESETS: Dict[str, CloudProviderPreset] = {
    "OpenAI": CloudProviderPreset(
        name="OpenAI",
        env_key="OPENAI_API_KEY",
        env_base_url="OPENAI_BASE_URL",
        env_model="OPENAI_DEFAULT_MODEL",
        default_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        docs_url="https://platform.openai.com/api-keys",
    ),
    "Groq": CloudProviderPreset(
        name="Groq",
        env_key="GROQ_API_KEY",
        env_base_url="GROQ_BASE_URL",
        env_model="GROQ_DEFAULT_MODEL",
        default_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        docs_url="https://console.groq.com/keys",
    ),
    "OpenRouter": CloudProviderPreset(
        name="OpenRouter",
        env_key="OPENROUTER_API_KEY",
        env_base_url="OPENROUTER_BASE_URL",
        env_model="OPENROUTER_DEFAULT_MODEL",
        default_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        docs_url="https://openrouter.ai/keys",
    ),
    "Gemini (OpenAI-compatible)": CloudProviderPreset(
        name="Gemini (OpenAI-compatible)",
        env_key="GEMINI_API_KEY",
        env_base_url="GEMINI_BASE_URL",
        env_model="GEMINI_DEFAULT_MODEL",
        default_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",   # gemini-2.0-flash often hits free-tier quota
        docs_url="https://aistudio.google.com/apikey",
    ),
    "Custom (OpenAI-compatible)": CloudProviderPreset(
        name="Custom (OpenAI-compatible)",
        env_key="CUSTOM_API_KEY",
        env_base_url="CUSTOM_BASE_URL",
        env_model="CUSTOM_DEFAULT_MODEL",
        default_url="",
        default_model="",
        docs_url="",
    ),
}


def get_preset(name: str) -> CloudProviderPreset:
    return PRESETS.get(name, PRESETS["Custom (OpenAI-compatible)"])


def list_presets() -> List[str]:
    return list(PRESETS.keys())


def detected_providers() -> List[str]:
    """Names of providers that have an API key set in the environment."""
    return [name for name, p in PRESETS.items() if p.has_key]
