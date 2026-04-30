from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".xlsx", ".xls"}


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    numeric_columns: int
    text_columns: int


def read_uploaded_dataset(file_obj: BinaryIO, filename: str) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Use one of: {supported}")

    file_obj.seek(0)
    if extension == ".csv":
        return pd.read_csv(file_obj)
    if extension == ".tsv":
        return pd.read_csv(file_obj, sep="\t")
    if extension == ".json":
        return pd.read_json(file_obj)
    return pd.read_excel(file_obj)


def summarize_dataset(frame: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        rows=len(frame),
        columns=len(frame.columns),
        missing_values=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        numeric_columns=len(frame.select_dtypes(include="number").columns),
        text_columns=len(frame.select_dtypes(include=["object", "string"]).columns),
    )


def column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_rows = max(len(frame), 1)
    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "type": str(series.dtype),
                "missing": missing_count,
                "missing_%": round((missing_count / total_rows) * 100, 2),
                "unique": int(series.nunique(dropna=True)),
                "example": _format_example(series.dropna().head(1)),
            }
        )
    return pd.DataFrame(rows)


def numeric_profile(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "column"})


def styled_numeric_preview(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Add a lightweight numeric heatmap without matplotlib."""
    return frame.style.apply(_numeric_heatmap_styles, axis=None)


def _numeric_heatmap_styles(frame: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    numeric_columns = frame.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        low = non_null.min()
        high = non_null.max()
        span = high - low
        if span == 0:
            normalized = series.where(series.isna(), 0.45)
        else:
            normalized = (series - low) / span

        styles[column] = normalized.map(_heatmap_cell_style)

    return styles


def _heatmap_cell_style(value: float) -> str:
    if pd.isna(value):
        return ""
    alpha = 0.10 + (float(value) * 0.28)
    return f"background-color: rgba(37, 99, 235, {alpha:.2f}); color: #0f172a;"


def _format_example(series: pd.Series) -> str:
    if series.empty:
        return ""
    value = series.iloc[0]
    return str(value)[:80]
