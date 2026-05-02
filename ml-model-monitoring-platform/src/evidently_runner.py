"""Generate Evidently AI HTML reports — fault-tolerant wrappers."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Optional, Tuple


# Per-preset lazy imports so a single missing class can't disable other reports.

def _import_report():
    from evidently.report import Report
    return Report


def _import_column_mapping():
    from evidently import ColumnMapping
    return ColumnMapping


def _build_column_mapping(
    target_col: str,
    prediction_col: str,
    numerical_features: List[str],
    categorical_features: List[str],
):
    ColumnMapping = _import_column_mapping()
    cm = ColumnMapping()
    cm.target               = target_col
    cm.prediction           = prediction_col
    cm.numerical_features   = numerical_features
    cm.categorical_features = categorical_features
    return cm


def run_data_drift_report(
    reference_df,
    current_df,
    numerical_features: List[str],
    categorical_features: List[str],
    out_path: Path,
) -> Tuple[bool, str]:
    """Generate a data drift HTML report. Returns (success, message)."""
    try:
        Report = _import_report()
        from evidently.metric_preset import DataDriftPreset
        cols = numerical_features + categorical_features
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[cols],
            current_data=current_df[cols],
            column_mapping=None,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        return True, str(out_path)
    except Exception as exc:
        return False, f"Data drift report failed: {exc}"


def run_classification_report(
    reference_df,
    current_df,
    target_col: str,
    prediction_col: str,
    numerical_features: List[str],
    categorical_features: List[str],
    out_path: Path,
) -> Tuple[bool, str]:
    """Generate a classification performance HTML report."""
    try:
        Report = _import_report()
        from evidently.metric_preset import ClassificationPreset
        cm = _build_column_mapping(
            target_col, prediction_col, numerical_features, categorical_features,
        )
        cols = numerical_features + categorical_features + [target_col, prediction_col]
        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=reference_df[cols],
            current_data=current_df[cols],
            column_mapping=cm,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        return True, str(out_path)
    except Exception as exc:
        return False, f"Classification report failed: {exc}"


def run_data_summary_report(
    reference_df,
    current_df,
    out_path: Path,
) -> Tuple[bool, str]:
    """
    Generate a data summary / quality HTML report.

    Tries DataSummaryPreset first (newer Evidently versions), then falls back
    to DataQualityPreset (Evidently 0.6.x). Reports gracefully if neither is
    available.
    """
    try:
        Report = _import_report()
        try:
            from evidently.metric_preset import DataSummaryPreset as _Preset
        except ImportError:
            from evidently.metric_preset import DataQualityPreset as _Preset
        report = Report(metrics=[_Preset()])
        report.run(reference_data=reference_df, current_data=current_df)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        return True, str(out_path)
    except Exception as exc:
        return False, f"Data summary report failed: {exc}"
