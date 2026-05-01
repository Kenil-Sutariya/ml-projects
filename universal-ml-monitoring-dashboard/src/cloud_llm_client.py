"""
Optional cloud LLM client using OpenAI-compatible chat-completion endpoints.

Privacy:
- API key is never persisted; it lives only in the caller's session state.
- This module never logs the key.
- It is the caller's responsibility to ensure no sensitive raw data is sent.
"""

from __future__ import annotations

import json
from typing import Optional

import requests

DEFAULT_BASE_URL = "https://api.openai.com/v1"
TIMEOUT          = 120

SYSTEM_PROMPT = (
    "You are a careful ML monitoring analyst. Use only the provided metrics. "
    "Do not invent numbers or causes."
)


def _normalise_base_url(base_url: Optional[str]) -> str:
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if not url:
        url = DEFAULT_BASE_URL
    return url


def generate_cloud_response(
    prompt:     str,
    api_key:    str,
    base_url:   Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Send `prompt` to an OpenAI-compatible chat endpoint.
    Returns response text, or a descriptive error message starting with `[ERROR]`.
    """
    if not api_key or not api_key.strip():
        return "[ERROR] No API key provided."
    if not model_name or not model_name.strip():
        return "[ERROR] No model name provided."

    url = _normalise_base_url(base_url) + "/chat/completions"
    payload = {
        "model":    model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        return f"[ERROR] Cloud LLM request timed out after {TIMEOUT}s."
    except requests.exceptions.ConnectionError as exc:
        return f"[ERROR] Could not reach cloud LLM endpoint ({base_url or DEFAULT_BASE_URL}): {exc}"
    except requests.exceptions.RequestException as exc:
        return f"[ERROR] Cloud LLM request failed: {exc}"

    if resp.status_code != 200:
        # Surface a concise error without leaking the key
        try:
            body = resp.json()
            msg  = body.get("error", {}).get("message") or body
        except Exception:
            msg = resp.text[:300]
        return f"[ERROR] Cloud LLM HTTP {resp.status_code}: {msg}"

    try:
        data    = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return "[ERROR] Cloud LLM returned no choices."
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not content:
            return "[ERROR] Cloud LLM returned empty content."
        return content
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        return f"[ERROR] Could not parse cloud LLM response: {exc}"
