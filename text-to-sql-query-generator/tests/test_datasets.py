import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sql_generator.datasets import (  # noqa: E402
    column_profile,
    read_uploaded_dataset,
    styled_numeric_preview,
    summarize_dataset,
)


def test_read_uploaded_csv_dataset():
    dataset = io.BytesIO(b"name,sales\nAva,120\nLiam,90\n")

    frame = read_uploaded_dataset(dataset, "sales.csv")

    assert list(frame.columns) == ["name", "sales"]
    assert frame["sales"].sum() == 210


def test_summarize_dataset_counts_quality_stats():
    dataset = io.BytesIO(b"name,sales\nAva,120\nAva,120\nMia,\n")
    frame = read_uploaded_dataset(dataset, "sales.csv")

    summary = summarize_dataset(frame)

    assert summary.rows == 3
    assert summary.columns == 2
    assert summary.missing_values == 1
    assert summary.duplicate_rows == 1
    assert summary.numeric_columns == 1
    assert summary.text_columns == 1


def test_column_profile_includes_missing_and_examples():
    dataset = io.BytesIO(b"name,sales\nAva,120\nMia,\n")
    frame = read_uploaded_dataset(dataset, "sales.csv")

    profile = column_profile(frame)

    sales_row = profile[profile["column"] == "sales"].iloc[0]
    assert sales_row["missing"] == 1
    assert sales_row["example"] == "120.0"


def test_styled_numeric_preview_does_not_require_matplotlib():
    dataset = io.BytesIO(b"name,sales\nAva,120\nMia,90\n")
    frame = read_uploaded_dataset(dataset, "sales.csv")

    styler = styled_numeric_preview(frame)

    styler._compute()
