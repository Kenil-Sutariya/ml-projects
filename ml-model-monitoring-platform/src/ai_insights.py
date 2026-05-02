"""Generate AI monitoring insights from monitoring summary + drift details.

Routes through `llm_router` so callers can choose Local Ollama, Cloud LLM,
or Disabled. Saves output to workspace/ai_insights/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .llm_router    import (
    PROVIDER_DISABLED, PROVIDER_OLLAMA, PROVIDER_CLOUD,
    generate_explanation,
)
from .ollama_client import (
    check_ollama_connection,
    list_available_ollama_models,
    select_best_available_model,
)
from .utils         import PATHS, ensure_dirs

PLACEHOLDER_MD = (
    "# AI Insight Unavailable\n\n"
    "No LLM provider was available when insights were generated.\n\n"
    "- **Local Ollama:** start it with `ollama serve` and re-run.\n"
    "- **Cloud LLM:** provide an API key on the AI Insights page.\n"
)


def load_prompt_template() -> str:
    p = PATHS["prompts"] / "monitoring_analyst_prompt.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "You are an ML monitoring analyst. Use only the provided metrics.\n\n"
        "Monitoring Summary:\n{monitoring_summary}\n\n"
        "Feature Drift Details:\n{feature_drift_details}\n\n"
        "Selected Batch: {selected_batch}\n"
    )


def build_monitoring_prompt(
    summary_df: pd.DataFrame,
    drift_df:   pd.DataFrame,
    selected_batch: Optional[str] = None,
) -> str:
    template = load_prompt_template()
    if selected_batch and selected_batch != "ALL":
        s_df = summary_df[summary_df["batch_name"] == selected_batch]
        d_df = drift_df[drift_df["batch_name"] == selected_batch]
        scope = selected_batch
    else:
        s_df = summary_df
        d_df = drift_df
        scope = "ALL"
    return template.format(
        monitoring_summary=s_df.to_string(index=False),
        feature_drift_details=d_df.to_string(index=False),
        selected_batch=scope,
    )


def _save_md(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_ollama_config(model_name: Optional[str]) -> dict:
    if model_name is None:
        if not check_ollama_connection():
            return {"ollama_model": None}
        model_name = select_best_available_model()
    return {"ollama_model": model_name}


def generate_overall_insight(
    summary_df: pd.DataFrame,
    drift_df:   pd.DataFrame,
    model_name: Optional[str] = None,
    *,
    provider: str = PROVIDER_OLLAMA,
    config:   Optional[dict] = None,
) -> tuple[str, Path]:
    """Generate the overall (all-batches) insight."""
    ensure_dirs()
    out_path = PATHS["ai_insights"] / "overall_monitoring_insight.md"

    if provider == PROVIDER_DISABLED:
        _save_md(PLACEHOLDER_MD, out_path)
        return PLACEHOLDER_MD, out_path

    cfg = config or {}
    if provider == PROVIDER_OLLAMA:
        cfg = {**cfg, **_resolve_ollama_config(model_name)}
        if not cfg.get("ollama_model"):
            _save_md(PLACEHOLDER_MD, out_path)
            return PLACEHOLDER_MD, out_path

    prompt   = build_monitoring_prompt(summary_df, drift_df, "ALL")
    response = generate_explanation(provider, prompt, cfg)
    label    = cfg.get("model_name") or cfg.get("ollama_model") or provider
    content  = f"# Overall Monitoring Insight\n\n*Provider: {provider} · Model: {label}*\n\n{response}"
    _save_md(content, out_path)
    return content, out_path


def generate_batch_insight(
    summary_df: pd.DataFrame,
    drift_df:   pd.DataFrame,
    batch_name: str,
    model_name: Optional[str] = None,
    *,
    provider: str = PROVIDER_OLLAMA,
    config:   Optional[dict] = None,
) -> tuple[str, Path]:
    """Generate one batch's insight."""
    ensure_dirs()
    out_path = PATHS["ai_insights"] / f"{batch_name}_insight.md"

    if provider == PROVIDER_DISABLED:
        _save_md(PLACEHOLDER_MD, out_path)
        return PLACEHOLDER_MD, out_path

    cfg = config or {}
    if provider == PROVIDER_OLLAMA:
        cfg = {**cfg, **_resolve_ollama_config(model_name)}
        if not cfg.get("ollama_model"):
            _save_md(PLACEHOLDER_MD, out_path)
            return PLACEHOLDER_MD, out_path

    prompt   = build_monitoring_prompt(summary_df, drift_df, batch_name)
    response = generate_explanation(provider, prompt, cfg)
    label    = cfg.get("model_name") or cfg.get("ollama_model") or provider
    content  = f"# Monitoring Insight — {batch_name}\n\n*Provider: {provider} · Model: {label}*\n\n{response}"
    _save_md(content, out_path)
    return content, out_path


def generate_all_batch_insights(
    summary_df: pd.DataFrame,
    drift_df:   pd.DataFrame,
    model_name: Optional[str] = None,
    *,
    provider: str = PROVIDER_OLLAMA,
    config:   Optional[dict] = None,
) -> dict:
    """Generate overall + per-batch insights."""
    paths = {}
    _, p = generate_overall_insight(summary_df, drift_df, model_name, provider=provider, config=config)
    paths["overall"] = p
    for name in summary_df["batch_name"].tolist():
        _, p = generate_batch_insight(summary_df, drift_df, name, model_name, provider=provider, config=config)
        paths[name] = p
    return paths
