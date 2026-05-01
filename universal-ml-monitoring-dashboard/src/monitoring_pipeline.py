"""End-to-end monitoring pipeline orchestration.

Supports two prediction sources:
  1. CSVs already contain a `prediction_col` (Stage 1 — BYO Predictions).
  2. A trusted model object is supplied via `model=` (Stage 2 — BYO Model);
     predictions are generated for the reference + every current batch before
     the rest of the pipeline runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .drift_analyzer       import numerical_drift, categorical_drift, summarise_batch_drift
from .evidently_runner     import (
    run_data_drift_report, run_classification_report, run_data_summary_report,
)
from .metrics_calculator   import calculate_classification_metrics, determine_model_health
from .prediction_engine    import predict_for_all
from .schema_validator     import validate_reference_and_current
from .utils                import PATHS, ensure_dirs, safe_save_csv, load_config


ProgressFn = Callable[[str, float], None]   # (label, fraction 0–1)


def run_monitoring_pipeline(
    reference_df: pd.DataFrame,
    current_batches: Dict[str, pd.DataFrame],
    target_col: str,
    prediction_col: Optional[str] = None,
    prediction_proba_col: Optional[str] = None,
    *,
    model: Any = None,
    numerical_features:   Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    feature_columns:      Optional[List[str]] = None,
    run_id: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """
    Runs the complete monitoring pipeline.

    Returns a dict with output paths and DataFrames.
    """
    ensure_dirs()

    cfg = load_config().get("monitoring", {})
    if config_overrides:
        cfg = {**cfg, **config_overrides}

    drift_num_thr   = cfg.get("drift_numeric_threshold",     0.5)
    drift_cat_thr   = cfg.get("drift_categorical_threshold", 0.25)
    warn_err        = cfg.get("warning_error_rate",          0.25)
    crit_err        = cfg.get("critical_error_rate",         0.35)
    warn_f1         = cfg.get("warning_f1_threshold",        0.70)
    crit_f1         = cfg.get("critical_f1_threshold",       0.60)

    numerical_features   = list(numerical_features   or [])
    categorical_features = list(categorical_features or [])
    feature_columns      = feature_columns or (numerical_features + categorical_features)

    pipeline_warnings: List[str] = []

    def _progress(label: str, frac: float) -> None:
        if progress:
            progress(label, max(0.0, min(1.0, frac)))

    # ----------------------------------------------------------------
    # 0. Generate predictions from uploaded model (BYO Model mode)
    # ----------------------------------------------------------------
    if model is not None and (
        prediction_col is None
        or prediction_col not in (reference_df.columns if reference_df is not None else [])
    ):
        if not feature_columns:
            return {
                "status":            "failed",
                "validation_report": {"errors": ["No feature columns selected for model prediction."]},
            }
        _progress("Loading model", 0.02)
        _progress("Generating reference predictions", 0.04)
        ref_with, batches_with, warns, err = predict_for_all(
            model, reference_df, current_batches,
            feature_columns=feature_columns,
            prediction_col="prediction",
            prediction_proba_col="prediction_proba",
        )
        if err:
            return {
                "status":            "failed",
                "validation_report": {"errors": [err]},
            }
        reference_df    = ref_with
        current_batches = batches_with
        prediction_col  = "prediction"
        # Detect whether predict_proba succeeded (column added)
        if "prediction_proba" in reference_df.columns:
            prediction_proba_col = "prediction_proba"
        else:
            prediction_proba_col = None
            pipeline_warnings.append(
                "Model has no usable predict_proba — ROC-AUC and probability charts may be unavailable."
            )
        pipeline_warnings.extend(warns)
        _progress("Generating batch predictions", 0.08)
    elif prediction_col is None:
        return {
            "status":            "failed",
            "validation_report": {
                "errors": ["No prediction column provided and no model uploaded."]
            },
        }

    # ----------------------------------------------------------------
    # 1. Validate
    # ----------------------------------------------------------------
    _progress("Validating data", 0.10)
    report = validate_reference_and_current(
        reference_df, current_batches,
        target_col=target_col,
        prediction_col=prediction_col,
        prediction_proba_col=prediction_proba_col,
        feature_columns=feature_columns,
    )
    if report.status == "failed":
        return {
            "status":            "failed",
            "validation_report": report.to_dict(),
        }

    # ----------------------------------------------------------------
    # 2. Per-batch processing
    # ----------------------------------------------------------------
    summary_rows:      List[dict] = []
    feature_drift_rows: List[dict] = []

    n_batches = max(len(current_batches), 1)
    for idx, (batch_name, cur_df) in enumerate(current_batches.items()):
        base_frac = 0.10 + 0.75 * (idx / n_batches)
        _progress(f"Processing {batch_name}", base_frac)

        # --- Save processed batch (full uploaded data) ---
        processed_path = PATHS["processed"] / f"{batch_name}.csv"
        safe_save_csv(cur_df, processed_path)

        # --- Metrics ---
        _progress(f"Calculating metrics for {batch_name}", base_frac + 0.02)
        metrics = calculate_classification_metrics(
            cur_df, target_col, prediction_col, prediction_proba_col,
        )

        # --- Custom drift ---
        _progress(f"Detecting drift for {batch_name}", base_frac + 0.05)
        num_rows = numerical_drift(reference_df, cur_df, numerical_features, drift_num_thr)
        cat_rows = categorical_drift(reference_df, cur_df, categorical_features, drift_cat_thr)
        all_drift = num_rows + cat_rows
        for r in all_drift:
            feature_drift_rows.append({"batch_name": batch_name, **r})
        drift_summary = summarise_batch_drift(all_drift)

        # --- Evidently reports ---
        _progress(f"Generating Evidently reports for {batch_name}", base_frac + 0.08)
        drift_html_ok, drift_html_path = run_data_drift_report(
            reference_df, cur_df, numerical_features, categorical_features,
            PATHS["reports_drift"] / f"{batch_name}_data_drift.html",
        )
        perf_html_ok, perf_html_path = run_classification_report(
            reference_df, cur_df, target_col, prediction_col,
            numerical_features, categorical_features,
            PATHS["reports_perf"] / f"{batch_name}_performance.html",
        )
        summary_html_ok, summary_html_path = run_data_summary_report(
            reference_df, cur_df,
            PATHS["reports_summary"] / f"{batch_name}_data_summary.html",
        )

        # --- Model health ---
        health = determine_model_health(
            error_rate=metrics["error_rate"],
            f1=metrics["f1_score"],
            drift_detected=drift_summary["drift_detected"],
            warning_error_rate=warn_err, critical_error_rate=crit_err,
            warning_f1_threshold=warn_f1, critical_f1_threshold=crit_f1,
        )

        row = {
            "batch_name":                     batch_name,
            "accuracy":                       metrics["accuracy"],
            "precision":                      metrics["precision"],
            "recall":                         metrics["recall"],
            "f1_score":                       metrics["f1_score"],
            "roc_auc":                        metrics["roc_auc"],
            "total_predictions":              metrics["total_predictions"],
            "correct_predictions":            metrics["correct_predictions"],
            "wrong_predictions":              metrics["wrong_predictions"],
            "error_rate":                     metrics["error_rate"],
            "positive_prediction_rate":       metrics["positive_prediction_rate"],
            "average_prediction_probability": metrics["average_prediction_probability"],
            "is_binary":                      metrics["is_binary"],
            "true_positive":                  metrics.get("true_positive"),
            "false_positive":                 metrics.get("false_positive"),
            "true_negative":                  metrics.get("true_negative"),
            "false_negative":                 metrics.get("false_negative"),
            **drift_summary,
            "model_health":                   health,
            "data_drift_report_path":         drift_html_path  if drift_html_ok  else "N/A",
            "performance_report_path":        perf_html_path   if perf_html_ok   else "N/A",
            "data_summary_report_path":       summary_html_path if summary_html_ok else "N/A",
        }
        summary_rows.append(row)

    # ----------------------------------------------------------------
    # 3. Save summary CSVs
    # ----------------------------------------------------------------
    _progress("Saving monitoring summary", 0.92)
    summary_df = pd.DataFrame(summary_rows)
    drift_df   = pd.DataFrame(feature_drift_rows)

    summary_path = PATHS["summaries"] / "monitoring_summary.csv"
    drift_path   = PATHS["summaries"] / "feature_drift_details.csv"

    safe_save_csv(summary_df, summary_path)
    safe_save_csv(drift_df,   drift_path)

    _progress("Finished", 1.0)

    return {
        "status":                   "ok",
        "validation_report":        report.to_dict(),
        "monitoring_summary_path":  str(summary_path),
        "feature_drift_path":       str(drift_path),
        "monitoring_summary":       summary_df,
        "feature_drift_details":    drift_df,
        "n_batches":                len(current_batches),
        "pipeline_warnings":        pipeline_warnings,
        "prediction_col":           prediction_col,
        "prediction_proba_col":     prediction_proba_col,
    }
