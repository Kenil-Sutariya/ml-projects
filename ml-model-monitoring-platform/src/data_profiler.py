"""Lightweight per-DataFrame profiling — used in the Upload & Configure UI."""

from __future__ import annotations

import pandas as pd


def basic_profile(df: pd.DataFrame) -> dict:
    """Return a small summary about a DataFrame for UI display."""
    if df is None:
        return {"rows": 0, "cols": 0, "missing": 0, "duplicate_rows": 0}
    return {
        "rows":           int(df.shape[0]),
        "cols":           int(df.shape[1]),
        "missing":        int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns":        list(df.columns),
        "dtypes":         {c: str(t) for c, t in df.dtypes.items()},
    }


def column_stats(df: pd.DataFrame, column: str) -> dict:
    """Return stats for one column (numeric or categorical)."""
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return {
            "type":   "numerical",
            "mean":   round(float(s.mean()), 4) if len(s.dropna()) else None,
            "std":    round(float(s.std()), 4)  if len(s.dropna()) else None,
            "min":    round(float(s.min()), 4)  if len(s.dropna()) else None,
            "max":    round(float(s.max()), 4)  if len(s.dropna()) else None,
            "missing": int(s.isna().sum()),
        }
    return {
        "type":     "categorical",
        "unique":   int(s.nunique()),
        "top":      str(s.mode().iloc[0]) if not s.mode().empty else None,
        "top_freq": int((s == s.mode().iloc[0]).sum()) if not s.mode().empty else 0,
        "missing":  int(s.isna().sum()),
    }
