"""Schema validation for reference + current CSV files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column type inference
# ---------------------------------------------------------------------------

def infer_column_types(df: pd.DataFrame, max_unique_for_categorical: int = 20) -> dict:
    """
    Infer numerical / categorical / id-like columns from a DataFrame.

    Returns:
        {
            "numerical":   [...],
            "categorical": [...],
            "boolean":     [...],
            "datetime":    [...],
            "other":       [...],
        }
    """
    numerical:   List[str] = []
    categorical: List[str] = []
    boolean:     List[str] = []
    datetime:    List[str] = []
    other:       List[str] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Numeric with very few unique values is likely categorical (e.g. encoded labels)
            nunique = series.nunique(dropna=True)
            if nunique <= 5 and series.dropna().apply(lambda x: float(x).is_integer()).all():
                categorical.append(col)
            else:
                numerical.append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            nunique = series.nunique(dropna=True)
            if nunique <= max_unique_for_categorical:
                categorical.append(col)
            else:
                other.append(col)
        else:
            other.append(col)

    return {
        "numerical":   numerical,
        "categorical": categorical,
        "boolean":     boolean,
        "datetime":    datetime,
        "other":       other,
    }


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    status:                  str = "passed"   # passed | warning | failed
    errors:                  List[str] = field(default_factory=list)
    warnings:                List[str] = field(default_factory=list)
    detected_numerical:      List[str] = field(default_factory=list)
    detected_categorical:    List[str] = field(default_factory=list)
    suggested_features:      List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.status = "failed"

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        if self.status == "passed":
            self.status = "warning"

    def to_dict(self) -> dict:
        return {
            "status":               self.status,
            "errors":               self.errors,
            "warnings":             self.warnings,
            "detected_numerical":   self.detected_numerical,
            "detected_categorical": self.detected_categorical,
            "suggested_features":   self.suggested_features,
        }


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_prediction_columns(
    df: pd.DataFrame,
    target_col: str,
    prediction_col: str,
    prediction_proba_col: Optional[str] = None,
    df_label: str = "DataFrame",
) -> List[str]:
    """Return a list of error strings (empty = OK)."""
    errors: List[str] = []
    if target_col not in df.columns:
        errors.append(f"{df_label} is missing target column '{target_col}'.")
    if prediction_col not in df.columns:
        errors.append(f"{df_label} is missing prediction column '{prediction_col}'.")
    if prediction_proba_col and prediction_proba_col not in df.columns:
        errors.append(f"{df_label} is missing prediction probability column '{prediction_proba_col}'.")
    return errors


def validate_reference_and_current(
    reference_df: pd.DataFrame,
    current_dfs: dict,                # {batch_name: DataFrame}
    target_col: str,
    prediction_col: str,
    prediction_proba_col: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> ValidationReport:
    """
    Run the full set of validation checks for monitoring.

    `current_dfs` is a mapping of batch name → DataFrame.
    """
    report = ValidationReport()

    # --- Empty checks ---
    if reference_df is None or reference_df.empty:
        report.add_error("Reference dataset is empty or missing.")
    if not current_dfs:
        report.add_error("No current batch dataset provided.")

    if report.status == "failed":
        return report

    # --- Reference column checks ---
    ref_errors = validate_prediction_columns(
        reference_df, target_col, prediction_col, prediction_proba_col,
        df_label="Reference",
    )
    for err in ref_errors:
        report.add_error(err)

    # --- Per-batch checks ---
    for batch_name, df in current_dfs.items():
        if df is None or df.empty:
            report.add_error(f"Batch '{batch_name}' is empty.")
            continue
        for err in validate_prediction_columns(
            df, target_col, prediction_col, prediction_proba_col,
            df_label=f"Batch '{batch_name}'",
        ):
            report.add_error(err)

    # --- Feature column checks ---
    if feature_columns:
        missing_in_ref = [c for c in feature_columns if c not in reference_df.columns]
        if missing_in_ref:
            report.add_error(
                f"Feature columns missing in reference: {', '.join(missing_in_ref)}"
            )
        for batch_name, df in current_dfs.items():
            if df is None:
                continue
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                report.add_error(
                    f"Feature columns missing in batch '{batch_name}': {', '.join(missing)}"
                )

    # --- Class label compatibility ---
    if target_col in reference_df.columns and prediction_col in reference_df.columns:
        ref_classes = set(reference_df[target_col].dropna().unique())
        for batch_name, df in current_dfs.items():
            if df is None or target_col not in df.columns:
                continue
            cur_classes = set(df[target_col].dropna().unique())
            extra = cur_classes - ref_classes
            if extra:
                report.add_warning(
                    f"Batch '{batch_name}' has unseen target classes: {sorted(map(str, extra))}"
                )

    # --- Missing values warning ---
    ref_missing = reference_df.isna().sum().sum() if reference_df is not None else 0
    if ref_missing > 0:
        report.add_warning(f"Reference has {ref_missing} missing values across all columns.")

    for batch_name, df in current_dfs.items():
        if df is None:
            continue
        n_missing = df.isna().sum().sum()
        if n_missing > 0:
            report.add_warning(f"Batch '{batch_name}' has {n_missing} missing values.")

    # --- Extra columns warning ---
    ref_cols = set(reference_df.columns) if reference_df is not None else set()
    for batch_name, df in current_dfs.items():
        if df is None:
            continue
        extra = set(df.columns) - ref_cols
        missing = ref_cols - set(df.columns)
        if extra:
            report.add_warning(
                f"Batch '{batch_name}' has extra columns not in reference: {sorted(extra)}"
            )
        if missing:
            report.add_warning(
                f"Batch '{batch_name}' is missing columns present in reference: {sorted(missing)}"
            )

    # --- Suggested feature columns from reference ---
    inferred = infer_column_types(reference_df) if reference_df is not None else {}
    report.detected_numerical   = [
        c for c in inferred.get("numerical", []) if c not in {target_col, prediction_col, prediction_proba_col or ""}
    ]
    report.detected_categorical = [
        c for c in inferred.get("categorical", []) if c not in {target_col, prediction_col, prediction_proba_col or ""}
    ]
    report.suggested_features   = report.detected_numerical + report.detected_categorical

    return report
