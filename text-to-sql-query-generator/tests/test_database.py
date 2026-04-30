import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sql_generator.database import get_schema_text, initialize_database, run_select_query
from sql_generator.llm import generate_sql
from sql_generator.safety import validate_select_query


def test_database_has_expected_schema(tmp_path):
    db_path = tmp_path / "sales.db"
    initialize_database(db_path)

    schema = get_schema_text(db_path)

    assert "customers(customer_id INTEGER" in schema
    assert "orders(order_id INTEGER" in schema
    assert "products(product_id INTEGER" in schema


def test_monthly_sales_demo_query_runs(tmp_path):
    db_path = tmp_path / "sales.db"
    initialize_database(db_path)

    result = generate_sql("Show total sales by month", get_schema_text(db_path))
    is_safe, sql, message = validate_select_query(result.sql)
    output = run_select_query(sql, db_path)

    assert is_safe, message
    assert list(output.columns) == ["month", "total_sales"]
    assert len(output) == 8


def test_read_only_execution_blocks_write_even_if_called_directly(tmp_path):
    db_path = tmp_path / "sales.db"
    initialize_database(db_path)

    try:
        run_select_query("DELETE FROM customers", db_path)
    except Exception as exc:
        assert "not authorized" in str(exc).lower() or "readonly" in str(exc).lower()
    else:
        raise AssertionError("Write query should not execute through read-only runner.")

