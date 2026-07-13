"""Read-only SQL enforcement.

The agent is free to reason about whatever SQL it likes, but nothing it
writes should ever reach the database unless it is a single SELECT (or a
read-only CTE chain starting with WITH). This is the hard backstop behind
the "use a least-privilege, read-only DB user" advice in the README: even
if the DB user *did* have write access, or a future tool/agent misbehaves,
queries never get past this check.

Kept dependency-light (sqlparse only) and framework-agnostic so it can be
reused by any tool that touches the database (sql_db_query, plot_chart, …).
"""

import sqlparse

# Statement types sqlparse can identify that we never allow, regardless of
# keyword casing or leading whitespace/comments.
_BLOCKED_TYPES = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "GRANT", "REVOKE",
}

# Keywords that indicate a mutation even when sqlparse's coarse
# get_type() misclassifies the statement as "UNKNOWN" (this happens for
# some dialect-specific or multi-clause statements).
_BLOCKED_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "GRANT", "REVOKE", "MERGE", "CALL", "EXEC", "EXECUTE",
    "VACUUM", "COPY", "INTO",
}


class UnsafeQueryError(ValueError):
    """Raised when a query fails the read-only guard."""


def assert_select_only(sql: str) -> None:
    """Raise UnsafeQueryError unless `sql` is exactly one read-only query.

    Read-only means: a single statement, of type SELECT (or a WITH…
    starting CTE that itself only selects). Stacked statements
    (`SELECT 1; DROP TABLE x;`) are rejected even if the first statement
    is harmless, since a stacked-query injection is exactly the attack
    this guard exists to stop.
    """
    if not sql or not sql.strip():
        raise UnsafeQueryError("Empty query.")

    statements = [s for s in sqlparse.parse(sql) if s.token_first(skip_cm=True)]

    if len(statements) == 0:
        raise UnsafeQueryError("No executable statement found.")
    if len(statements) > 1:
        raise UnsafeQueryError(
            "Only a single SELECT statement is allowed — multiple "
            "statements in one query are blocked."
        )

    stmt = statements[0]
    stmt_type = stmt.get_type()  # "SELECT", "UNKNOWN", "INSERT", ...
    first_keyword = stmt.token_first(skip_cm=True)
    first_word = (first_keyword.value or "").strip().upper() if first_keyword else ""

    is_select = stmt_type == "SELECT"
    is_cte = stmt_type == "UNKNOWN" and first_word == "WITH"

    if not (is_select or is_cte):
        raise UnsafeQueryError(
            f"Only read-only SELECT queries are allowed (got: {stmt_type or first_word or 'unknown'})."
        )

    upper_sql = sql.upper()
    for keyword in _BLOCKED_KEYWORDS:
        # Word-boundary-ish check without pulling in `re` for a simple case:
        # pad with spaces so keyword can't match inside a longer identifier.
        if f" {keyword} " in f" {upper_sql} " or upper_sql.strip().startswith(keyword):
            raise UnsafeQueryError(
                f"Query contains a disallowed keyword ('{keyword}') — only read-only SELECTs are permitted."
            )


def is_select_only(sql: str) -> bool:
    """Boolean convenience wrapper around assert_select_only."""
    try:
        assert_select_only(sql)
        return True
    except UnsafeQueryError:
        return False
