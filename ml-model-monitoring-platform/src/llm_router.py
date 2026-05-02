"""
Route LLM explanation requests to the configured provider.

Providers:
- "disabled"      → no LLM at all
- "local_ollama"  → src/ollama_client.py
- "cloud"         → src/cloud_llm_client.py (user-supplied API key in config)
"""

from __future__ import annotations

from typing import Optional

from .cloud_llm_client import generate_cloud_response
from .ollama_client    import generate_ollama_response

PROVIDER_DISABLED = "disabled"
PROVIDER_OLLAMA   = "local_ollama"
PROVIDER_CLOUD    = "cloud"

ALL_PROVIDERS = [PROVIDER_DISABLED, PROVIDER_OLLAMA, PROVIDER_CLOUD]


def generate_explanation(
    provider: str,
    prompt:   str,
    config:   Optional[dict] = None,
) -> str:
    """
    Generate an explanation using the chosen provider.

    `config` shape:
        { "ollama_model":  "llama3.2:latest",          # for local_ollama
          "ollama_base_url": "http://localhost:11434", # optional
          "api_key":       "sk-...",                   # for cloud
          "base_url":      "https://api.openai.com/v1",# for cloud
          "model_name":    "gpt-4o-mini" }             # for cloud
    """
    cfg = config or {}
    provider = (provider or PROVIDER_DISABLED).lower()

    if provider == PROVIDER_DISABLED:
        return "AI explanations are disabled."

    if provider == PROVIDER_OLLAMA:
        model_name = cfg.get("ollama_model")
        if not model_name:
            return "[ERROR] No Ollama model selected."
        return generate_ollama_response(
            prompt, model_name,
            base_url=cfg.get("ollama_base_url"),
        )

    if provider == PROVIDER_CLOUD:
        return generate_cloud_response(
            prompt,
            api_key   = cfg.get("api_key", ""),
            base_url  = cfg.get("base_url"),
            model_name= cfg.get("model_name"),
        )

    return f"[ERROR] Unknown LLM provider: {provider}"
