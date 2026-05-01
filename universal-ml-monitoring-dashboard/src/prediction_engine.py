"""
Generate predictions from an uploaded model.

Designed to work with full Scikit-learn Pipelines (which handle their own
preprocessing) — we never try to manually preprocess uploaded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    success:               bool
    df:                    Optional[pd.DataFrame] = None
    used_predict_proba:    bool = False
    is_binary:             bool = False
    warnings:              List[str] = field(default_factory=list)
    error:                 str = ""


def generate_predictions(
    model: Any,
    df:    pd.DataFrame,
    feature_columns: List[str],
    *,
    prediction_col:       str = "prediction",
    prediction_proba_col: str = "prediction_proba",
) -> PredictionResult:
    """
    Run model.predict (and predict_proba when available) on `df[feature_columns]`.

    The returned DataFrame is a copy of `df` with `prediction` (and optionally
    `prediction_proba`) columns added. Original input columns are preserved.
    """
    result = PredictionResult(success=False)

    # --- Validate inputs ---
    if model is None:
        result.error = "No model provided."
        return result
    if df is None or df.empty:
        result.error = "Input DataFrame is empty."
        return result

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        result.error = f"Missing feature columns in data: {missing}"
        return result

    # --- Build feature matrix ---
    # Pass a DataFrame (not numpy) so column-name-aware Pipelines work.
    X = df[feature_columns].copy()

    # --- Predict ---
    try:
        y_pred = model.predict(X)
    except Exception as exc:
        result.error = f"model.predict failed: {exc}"
        return result

    out = df.copy()
    out[prediction_col] = y_pred

    # --- Predict_proba (optional) ---
    proba_fn = getattr(model, "predict_proba", None)
    if callable(proba_fn):
        try:
            proba = proba_fn(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                # Binary: take probability of class 1
                out[prediction_proba_col] = proba[:, 1]
                result.is_binary = True
                result.used_predict_proba = True
            elif proba.ndim == 2 and proba.shape[1] > 2:
                # Multiclass: probability of the predicted class
                pred_idx = np.argmax(proba, axis=1)
                out[prediction_proba_col] = proba[np.arange(len(proba)), pred_idx]
                result.used_predict_proba = True
                result.warnings.append(
                    "Multiclass model: probability column stores the probability of "
                    "the predicted class (not class 1)."
                )
            else:
                result.warnings.append("predict_proba returned an unexpected shape; column omitted.")
        except Exception as exc:
            result.warnings.append(f"predict_proba call failed: {exc}")
    else:
        result.warnings.append(
            "Model has no predict_proba method. ROC-AUC and probability charts may be unavailable."
        )

    # Heuristic: detect binary even without predict_proba
    classes = pd.unique(out[prediction_col])
    if len(classes) == 2 and not result.is_binary:
        result.is_binary = True

    result.success = True
    result.df = out
    return result


def predict_for_all(
    model:           Any,
    reference_df:    pd.DataFrame,
    current_batches: dict,
    feature_columns: List[str],
    *,
    prediction_col:       str = "prediction",
    prediction_proba_col: str = "prediction_proba",
) -> Tuple[Optional[pd.DataFrame], Optional[dict], List[str], str]:
    """
    Generate predictions for the reference dataset and every current batch.

    Returns:
        (reference_with_preds, {batch_name: batch_with_preds}, warnings, error)

    On error, the first three return values are None / {} / warnings.
    """
    warnings: List[str] = []

    ref_res = generate_predictions(
        model, reference_df, feature_columns,
        prediction_col=prediction_col,
        prediction_proba_col=prediction_proba_col,
    )
    if not ref_res.success:
        return None, None, ref_res.warnings, f"Reference prediction failed: {ref_res.error}"
    warnings.extend(ref_res.warnings)

    new_batches: dict = {}
    for name, batch_df in current_batches.items():
        batch_res = generate_predictions(
            model, batch_df, feature_columns,
            prediction_col=prediction_col,
            prediction_proba_col=prediction_proba_col,
        )
        if not batch_res.success:
            return None, None, warnings, f"Batch '{name}' prediction failed: {batch_res.error}"
        new_batches[name] = batch_res.df
        warnings.extend([f"[{name}] {w}" for w in batch_res.warnings])

    return ref_res.df, new_batches, warnings, ""
