from __future__ import annotations

import re


FORBIDDEN_KEYWORDS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "replace",
    "truncate",
    "update",
    "vacuum",
}


def clean_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = re.sub(r"```$", "", sql).strip()
    return sql


def validate_select_query(raw_sql: str) -> tuple[bool, str, str]:
    """Return (is_valid, cleaned_sql, message)."""
    sql = clean_sql(raw_sql)
    if not sql:
        return False, sql, "SQL is empty."

    if "--" in sql or "/*" in sql or "*/" in sql:
        return False, sql, "Comments are blocked to keep validation simple and safe."

    sql_without_final_semicolon = sql[:-1].strip() if sql.endswith(";") else sql
    if ";" in sql_without_final_semicolon:
        return False, sql, "Only one SQL statement is allowed."

    normalized = sql_without_final_semicolon.strip()
    if not re.match(r"^select\b", normalized, flags=re.IGNORECASE):
        return False, sql, "Only SELECT queries are allowed."

    tokens = set(re.findall(r"\b[a-z_]+\b", normalized.lower()))
    blocked = sorted(tokens & FORBIDDEN_KEYWORDS)
    if blocked:
        return False, sql, f"Blocked unsafe keyword(s): {', '.join(blocked)}."

    return True, normalized + ";", "Query passed safety checks."

