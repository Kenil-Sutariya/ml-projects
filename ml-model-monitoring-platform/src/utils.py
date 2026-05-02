"""Shared utilities: paths, config loader, file helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATHS: Dict[str, Path] = {
    "config":            PROJECT_ROOT / "config",
    "prompts":           PROJECT_ROOT / "prompts",
    "sample_data":       PROJECT_ROOT / "sample_project" / "data",
    "sample_models":     PROJECT_ROOT / "sample_project" / "models",
    "workspace":         PROJECT_ROOT / "workspace",
    "uploads":           PROJECT_ROOT / "workspace" / "uploads",
    "uploads_reference": PROJECT_ROOT / "workspace" / "uploads" / "reference",
    "uploads_current":   PROJECT_ROOT / "workspace" / "uploads" / "current_batches",
    "processed":         PROJECT_ROOT / "workspace" / "processed",
    "reports":           PROJECT_ROOT / "workspace" / "reports",
    "reports_drift":     PROJECT_ROOT / "workspace" / "reports" / "data_drift",
    "reports_perf":      PROJECT_ROOT / "workspace" / "reports" / "model_performance",
    "reports_summary":   PROJECT_ROOT / "workspace" / "reports" / "data_summary",
    "summaries":         PROJECT_ROOT / "workspace" / "summaries",
    "ai_insights":       PROJECT_ROOT / "workspace" / "ai_insights",
}


def ensure_dirs() -> None:
    """Create all standard project folders if they don't exist."""
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# .env loading — loads on import so subsequent os.getenv() calls work
# ---------------------------------------------------------------------------
_ENV_LOADED = False


def load_env(env_path: Path | None = None, override: bool = False) -> bool:
    """Load .env from project root. Returns True if a file was found."""
    global _ENV_LOADED
    fp = env_path or (PROJECT_ROOT / ".env")
    if fp.exists():
        load_dotenv(fp, override=override)
        _ENV_LOADED = True
        return True
    return False


# Auto-load on first import
load_env()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
_CONFIG_CACHE: Dict[str, Any] | None = None


def load_config() -> Dict[str, Any]:
    """Load config/app_config.yaml (cached)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg_path = PATHS["config"] / "app_config.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            _CONFIG_CACHE = yaml.safe_load(fh) or {}
    else:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def safe_read_csv(filepath: str | Path) -> pd.DataFrame | None:
    fp = Path(filepath)
    if not fp.exists():
        return None
    try:
        return pd.read_csv(fp)
    except Exception:
        return None


def safe_save_csv(df: pd.DataFrame, filepath: str | Path) -> bool:
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(fp, index=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Workspace cleanup helpers
# ---------------------------------------------------------------------------

def clear_workspace() -> None:
    """Remove all files inside workspace/ (keeps the folder structure)."""
    if PATHS["workspace"].exists():
        shutil.rmtree(PATHS["workspace"])
    ensure_dirs()


def list_workspace_summary_paths() -> Dict[str, Path]:
    """Convenient access to monitoring output CSV paths."""
    return {
        "monitoring_summary":    PATHS["summaries"] / "monitoring_summary.csv",
        "feature_drift_details": PATHS["summaries"] / "feature_drift_details.csv",
    }


# ---------------------------------------------------------------------------
# Batch name helpers
# ---------------------------------------------------------------------------

def derive_batch_name(filename: str) -> str:
    """Convert 'batch_1_normal.csv' → 'batch_1_normal'."""
    return Path(filename).stem
