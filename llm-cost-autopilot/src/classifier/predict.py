"""
Loads the trained classifier and exposes predict() for use by the router.
"""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.classifier.features import extract_features

MODEL_PATH = Path("src/classifier/model.pkl")


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.classifier.train"
        )
    return joblib.load(MODEL_PATH)


def predict_tier(prompt: str) -> int:
    """Return complexity tier (1, 2, or 3) for the given prompt."""
    model = _load_model()
    features = extract_features(prompt).to_array()
    X = np.array([features])
    return int(model.predict(X)[0])


def predict_tier_proba(prompt: str) -> dict[int, float]:
    """Return probability distribution across tiers."""
    model = _load_model()
    features = extract_features(prompt).to_array()
    X = np.array([features])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return {int(cls): round(float(p), 4) for cls, p in zip(classes, proba)}


def reload_model() -> None:
    """Force reload after retraining — clears the lru_cache."""
    _load_model.cache_clear()
