import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sql_generator.safety import validate_select_query


def test_allows_simple_select():
    is_safe, cleaned_sql, message = validate_select_query("SELECT * FROM orders;")

    assert is_safe is True
    assert cleaned_sql == "SELECT * FROM orders;"
    assert "passed" in message.lower()


def test_blocks_drop_table():
    is_safe, _, message = validate_select_query("DROP TABLE customers;")

    assert is_safe is False
    assert "select" in message.lower() or "blocked" in message.lower()


def test_blocks_multiple_statements():
    is_safe, _, message = validate_select_query(
        "SELECT * FROM orders; DELETE FROM orders;"
    )

    assert is_safe is False
    assert "one sql statement" in message.lower()


def test_blocks_update_hidden_after_select():
    is_safe, _, message = validate_select_query(
        "SELECT * FROM orders WHERE order_id IN (UPDATE orders SET quantity = 0)"
    )

    assert is_safe is False
    assert "update" in message.lower()

