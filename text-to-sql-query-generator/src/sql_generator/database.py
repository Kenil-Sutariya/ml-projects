from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sales.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);
"""


CUSTOMERS = [
    (1, "Ava Patel", "Austin", "2024-01-12"),
    (2, "Liam Chen", "Seattle", "2024-02-05"),
    (3, "Mia Johnson", "Chicago", "2024-03-18"),
    (4, "Noah Smith", "Boston", "2024-04-22"),
    (5, "Sophia Garcia", "Denver", "2024-05-11"),
]

PRODUCTS = [
    (1, "Laptop Stand", "Office", 45.00),
    (2, "Wireless Mouse", "Electronics", 28.50),
    (3, "Noise Cancelling Headphones", "Electronics", 120.00),
    (4, "Desk Lamp", "Office", 35.75),
    (5, "Notebook Pack", "Stationery", 14.99),
]

ORDERS = [
    (1, 1, 1, "2025-01-07", 2, 45.00),
    (2, 2, 2, "2025-01-15", 3, 28.50),
    (3, 3, 3, "2025-02-03", 1, 120.00),
    (4, 1, 5, "2025-02-20", 5, 14.99),
    (5, 4, 4, "2025-03-08", 2, 35.75),
    (6, 5, 1, "2025-03-19", 1, 45.00),
    (7, 2, 3, "2025-04-01", 2, 120.00),
    (8, 3, 2, "2025-04-14", 4, 28.50),
    (9, 4, 5, "2025-05-10", 6, 14.99),
    (10, 5, 4, "2025-05-28", 1, 35.75),
    (11, 1, 3, "2025-06-06", 1, 120.00),
    (12, 2, 1, "2025-06-17", 3, 45.00),
    (13, 3, 4, "2025-07-09", 2, 35.75),
    (14, 4, 2, "2025-07-21", 5, 28.50),
    (15, 5, 5, "2025-08-13", 10, 14.99),
]


def initialize_database(db_path: Path = DB_PATH) -> Path:
    """Create the SQLite sales database and seed deterministic sample data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM products")
        conn.execute("DELETE FROM customers")
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
            ORDERS,
        )
        conn.commit()
    return db_path


def get_connection(db_path: Path = DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        if not db_path.exists():
            initialize_database(db_path)
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(db_path)


def get_schema_text(db_path: Path = DB_PATH) -> str:
    if not db_path.exists():
        initialize_database(db_path)

    with get_connection(db_path, read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT m.name, p.name, p.type
            FROM sqlite_master AS m
            JOIN pragma_table_info(m.name) AS p
            WHERE m.type = 'table'
              AND m.name NOT LIKE 'sqlite_%'
            ORDER BY m.name, p.cid;
            """
        ).fetchall()

    tables: dict[str, list[str]] = {}
    for table_name, column_name, column_type in rows:
        tables.setdefault(table_name, []).append(f"{column_name} {column_type}")

    return "\n".join(
        f"- {table_name}({', '.join(columns)})"
        for table_name, columns in tables.items()
    )


def run_select_query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        initialize_database(db_path)

    with get_connection(db_path, read_only=True) as conn:
        conn.set_authorizer(_read_only_authorizer)
        return pd.read_sql_query(sql, conn)


def preview_tables(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    if not db_path.exists():
        initialize_database(db_path)

    previews: dict[str, pd.DataFrame] = {}
    with get_connection(db_path, read_only=True) as conn:
        for table_name in ("customers", "products", "orders"):
            previews[table_name] = pd.read_sql_query(
                f"SELECT * FROM {table_name} LIMIT 5",
                conn,
            )
    return previews


def _read_only_authorizer(
    action_code: int,
    arg1: str | None,
    arg2: str | None,
    db_name: str | None,
    trigger_name: str | None,
) -> int:
    del arg1, arg2, db_name, trigger_name
    allowed_actions: set[int] = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    return sqlite3.SQLITE_OK if action_code in allowed_actions else sqlite3.SQLITE_DENY


def query_to_records(sql: str) -> list[dict[str, Any]]:
    return run_select_query(sql).to_dict(orient="records")

