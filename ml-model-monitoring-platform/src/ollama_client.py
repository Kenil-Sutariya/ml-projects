"""Local Ollama HTTP API client. Fully optional — fails silently if unavailable."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import requests

from .utils import load_config

# Default request timeout. Override via OLLAMA_TIMEOUT env var (in seconds) —
# useful for large models (e.g. qwen3.5 / llama3.3-70b) whose first response
# can take several minutes due to model loading.
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))


def _base_url(override: Optional[str] = None) -> str:
    if override:
        return override.rstrip("/")
    cfg = load_config().get("llm", {})
    return cfg.get("local_ollama_url", "http://localhost:11434").rstrip("/")


def _priority_models(override: Optional[List[str]] = None) -> List[str]:
    if override:
        return list(override)
    cfg = load_config().get("llm", {})
    return list(cfg.get("default_ollama_model_priority", []))


def check_ollama_connection(base_url: Optional[str] = None) -> bool:
    """Return True if local Ollama is reachable."""
    try:
        resp = requests.get(f"{_base_url(base_url)}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_available_ollama_models(base_url: Optional[str] = None) -> List[str]:
    """Return list of installed Ollama model names (empty if unreachable)."""
    try:
        resp = requests.get(f"{_base_url(base_url)}/api/tags", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def select_best_available_model(
    available_models: Optional[List[str]] = None,
    priority_list:    Optional[List[str]] = None,
    base_url:         Optional[str]       = None,
) -> Optional[str]:
    """Pick the highest-priority available model."""
    available = available_models if available_models is not None else list_available_ollama_models(base_url)
    if not available:
        return None
    available_lower = [m.lower().strip() for m in available]
    for preferred in _priority_models(priority_list):
        key = preferred.lower().strip()
        if key in available_lower:
            return available[available_lower.index(key)]
    return available[0]


def generate_ollama_response(
    prompt:     str,
    model_name: str,
    base_url:   Optional[str] = None,
) -> str:
    """
    POST a single chat message to Ollama and return the assistant text.
    Returns a descriptive fallback string on any failure.
    """
    if not check_ollama_connection(base_url):
        return (
            "**Ollama is not running.**\n\n"
            "Start it with:  `ollama serve`\n\n"
            "Then re-run AI Insights generation."
        )
    payload = {
        "model":    model_name,
        "stream":   False,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(f"{_base_url(base_url)}/api/chat", json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "[Empty response]")
    except requests.exceptions.Timeout:
        return f"[ERROR] Ollama request timed out after {DEFAULT_TIMEOUT}s."
    except requests.exceptions.RequestException as exc:
        return f"[ERROR] Ollama request failed: {exc}"
    except (json.JSONDecodeError, KeyError) as exc:
        return f"[ERROR] Could not parse Ollama response: {exc}"
