"""
Trusted-only model loader for .pkl / .joblib files.

⚠️  Pickle / joblib files can execute arbitrary Python code on load. This loader
will ONLY be invoked from the UI after the user explicitly opts in via a
"trust this file" checkbox.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib


@dataclass
class ModelInfo:
    loaded:           bool
    model:            Any = None
    model_type:       str = ""
    has_predict:      bool = False
    has_predict_proba: bool = False
    error:            str = ""

    def to_dict(self) -> dict:
        return {
            "loaded":            self.loaded,
            "model_type":        self.model_type,
            "has_predict":       self.has_predict,
            "has_predict_proba": self.has_predict_proba,
            "error":             self.error,
        }


def check_model_capabilities(model: Any) -> dict:
    """Inspect a loaded model object."""
    return {
        "model_type":        type(model).__name__,
        "has_predict":       callable(getattr(model, "predict",       None)),
        "has_predict_proba": callable(getattr(model, "predict_proba", None)),
    }


def get_model_info(model: Any) -> ModelInfo:
    caps = check_model_capabilities(model)
    return ModelInfo(
        loaded=True,
        model=model,
        model_type=caps["model_type"],
        has_predict=caps["has_predict"],
        has_predict_proba=caps["has_predict_proba"],
    )


def load_model(model_path: str | Path) -> ModelInfo:
    """
    Load a trusted .pkl or .joblib model.

    Tries joblib.load first; falls back to pickle.load.
    Never raises — returns a ModelInfo with `loaded=False` and an error string
    on failure.
    """
    fp = Path(model_path)
    if not fp.exists():
        return ModelInfo(loaded=False, error=f"File not found: {fp}")

    suffix = fp.suffix.lower()
    if suffix not in {".pkl", ".joblib"}:
        return ModelInfo(
            loaded=False,
            error=f"Unsupported file type '{suffix}'. Only .pkl and .joblib are accepted.",
        )

    # 1) Try joblib (handles most Scikit-learn pipelines and is the recommended format)
    try:
        model = joblib.load(fp)
        return get_model_info(model)
    except Exception as joblib_exc:
        # 2) Fall back to pickle
        try:
            with open(fp, "rb") as fh:
                model = pickle.load(fh)
            return get_model_info(model)
        except Exception as pickle_exc:
            return ModelInfo(
                loaded=False,
                error=(
                    f"joblib.load failed: {joblib_exc}; "
                    f"pickle.load failed: {pickle_exc}"
                ),
            )
