"""Custom drift detection for numerical and categorical features."""

from __future__ import annotations

from typing import List

import pandas as pd


def numerical_drift(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    numerical_features: List[str],
    threshold: float = 0.5,
) -> List[dict]:
    """
    Mean-shift drift score per numerical feature.

    score = abs(current_mean - reference_mean) / max(reference_std, 1e-9)
    """
    rows: List[dict] = []
    for feat in numerical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref = pd.to_numeric(reference_df[feat], errors="coerce")
        cur = pd.to_numeric(current_df[feat],   errors="coerce")
        ref_mean = float(ref.mean())  if ref.notna().any() else 0.0
        ref_std  = float(ref.std())   if ref.notna().any() else 0.0
        cur_mean = float(cur.mean())  if cur.notna().any() else 0.0
        denom    = ref_std if ref_std > 1e-9 else 1.0
        score    = abs(cur_mean - ref_mean) / denom
        rows.append({
            "feature_name":    feat,
            "feature_type":    "numerical",
            "reference_value": round(ref_mean, 4),
            "current_value":   round(cur_mean, 4),
            "drift_score":     round(score, 4),
            "drift_detected":  "Yes" if score > threshold else "No",
        })
    return rows


def categorical_drift(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    categorical_features: List[str],
    threshold: float = 0.25,
) -> List[dict]:
    """
    Total-variation distance between normalised category distributions.
    """
    rows: List[dict] = []
    for feat in categorical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref_dist = reference_df[feat].astype("string").value_counts(normalize=True)
        cur_dist = current_df[feat].astype("string").value_counts(normalize=True)
        all_cats = set(ref_dist.index) | set(cur_dist.index)
        score = sum(abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_cats)
        rows.append({
            "feature_name":    feat,
            "feature_type":    "categorical",
            "reference_value": str(ref_dist.round(3).to_dict()),
            "current_value":   str(cur_dist.round(3).to_dict()),
            "drift_score":     round(float(score), 4),
            "drift_detected":  "Yes" if score > threshold else "No",
        })
    return rows


def summarise_batch_drift(drift_rows: List[dict]) -> dict:
    """Aggregate per-feature drift rows into a batch-level summary."""
    drifted = [r["feature_name"] for r in drift_rows if r["drift_detected"] == "Yes"]
    return {
        "number_of_drifted_features": len(drifted),
        "drift_detected":             "Yes" if drifted else "No",
        "drifted_features":           ", ".join(drifted) if drifted else "None",
    }
