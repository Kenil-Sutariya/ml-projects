"""Classification metrics for monitoring."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


def _safe(value, default=0.0):
    """Round to 4 dp, replace NaN/inf with default."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return round(v, 4)
    except Exception:
        return default


def calculate_classification_metrics(
    df: pd.DataFrame,
    target_col: str,
    prediction_col: str,
    prediction_proba_col: Optional[str] = None,
) -> dict:
    """
    Compute the full set of classification metrics for one batch.

    Handles:
      - missing probability column
      - non-binary labels
      - zero division
      - missing values (rows are dropped where target/prediction is NaN)
    """
    # Drop rows with missing target / prediction
    cols = [target_col, prediction_col]
    if prediction_proba_col and prediction_proba_col in df.columns:
        cols.append(prediction_proba_col)
    work = df[cols].dropna(subset=[target_col, prediction_col]).copy()

    if work.empty:
        return _empty_metrics()

    y_true = work[target_col]
    y_pred = work[prediction_col]
    total  = len(y_true)
    correct = int((y_true == y_pred).sum())
    wrong   = total - correct

    # Detect binary
    classes = sorted(set(y_true.unique()) | set(y_pred.unique()))
    is_binary = len(classes) == 2

    avg = "binary" if is_binary else "macro"

    accuracy  = _safe(accuracy_score(y_true, y_pred))
    try:
        precision = _safe(precision_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception:
        precision = 0.0
    try:
        recall    = _safe(recall_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception:
        recall = 0.0
    try:
        f1        = _safe(f1_score(y_true, y_pred, average=avg, zero_division=0))
    except Exception:
        f1 = 0.0

    # ROC-AUC needs probabilities
    roc_auc = 0.5
    avg_proba: Optional[float] = None
    if prediction_proba_col and prediction_proba_col in work.columns:
        proba = pd.to_numeric(work[prediction_proba_col], errors="coerce")
        if proba.notna().any():
            avg_proba = _safe(proba.mean())
            if is_binary:
                try:
                    roc_auc = _safe(roc_auc_score(y_true, proba))
                except Exception:
                    roc_auc = 0.5

    error_rate = _safe(wrong / total) if total > 0 else 0.0
    pos_rate   = _safe((y_pred == (1 if 1 in classes else classes[-1])).mean()) if classes else 0.0

    metrics = {
        "accuracy":                       accuracy,
        "precision":                      precision,
        "recall":                         recall,
        "f1_score":                       f1,
        "roc_auc":                        roc_auc,
        "total_predictions":              total,
        "correct_predictions":            correct,
        "wrong_predictions":              wrong,
        "error_rate":                     error_rate,
        "positive_prediction_rate":       pos_rate,
        "average_prediction_probability": avg_proba,
        "is_binary":                      is_binary,
    }

    # Confusion matrix entries when binary
    if is_binary:
        labels = sorted(classes)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # treat the larger label as the positive class (typical 0/1 convention)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            metrics.update({
                "true_negative":  tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive":  tp,
            })
    return metrics


def _empty_metrics() -> dict:
    return {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
        "f1_score": 0.0, "roc_auc": 0.5,
        "total_predictions": 0, "correct_predictions": 0, "wrong_predictions": 0,
        "error_rate": 0.0, "positive_prediction_rate": 0.0,
        "average_prediction_probability": None,
        "is_binary": False,
    }


# ---------------------------------------------------------------------------
# Model health labelling
# ---------------------------------------------------------------------------

def determine_model_health(
    error_rate: float,
    f1: float,
    drift_detected: str,
    *,
    warning_error_rate: float = 0.25,
    critical_error_rate: float = 0.35,
    warning_f1_threshold: float = 0.70,
    critical_f1_threshold: float = 0.60,
) -> str:
    if error_rate > critical_error_rate or f1 < critical_f1_threshold:
        return "Critical"
    if drift_detected == "Yes" or error_rate > warning_error_rate or f1 < warning_f1_threshold:
        return "Warning"
    return "Healthy"
