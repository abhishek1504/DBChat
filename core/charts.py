"""Chart tool — lets the agent turn a query result into a chart spec.

The tool runs its own guarded, read-only SELECT (never trusts a prior
tool's text output) and returns a small JSON envelope describing the
chart. The frontend is what actually draws it (Recharts) — this file's
job stops at producing clean `{x, y}` rows plus a chart type, not at
rendering.
"""

import json
from typing import Literal

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from sqlalchemy import text

from core.sql_guard import UnsafeQueryError, assert_select_only

MAX_ROWS = 200  # keep chart payloads (and the NDJSON stream) small


def _rows_for(db: SQLDatabase, query: str) -> list[dict]:
    """Execute `query` against the DB's own engine and return list[dict]
    rows. Goes around SQLDatabase.run() (which formats results as a
    string for the LLM) since we need structured data for the frontend."""
    with db._engine.connect() as conn:
        result = conn.execute(text(query))
        columns = list(result.keys())
        rows = result.fetchmany(MAX_ROWS)
        return [dict(zip(columns, row)) for row in rows]


def make_chart_tool(db: SQLDatabase):
    """Build a `plot_chart` tool bound to this specific DB connection."""

    @tool("plot_chart")
    def plot_chart(
        query: str,
        x: str,
        y: str,
        chart_type: Literal["bar", "line", "pie"] = "bar",
        title: str = "",
    ) -> str:
        """Run a read-only SQL SELECT and return chart-ready data.

        Use this tool instead of sql_db_query whenever the user asks to
        see a chart, graph, plot, or trend. `query` must be a single
        SELECT whose result includes a column named exactly `x` and a
        column named exactly `y` (use SQL `AS` aliases if the source
        columns are named differently). `chart_type` is "bar", "line",
        or "pie".
        """
        try:
            assert_select_only(query)
        except UnsafeQueryError as exc:
            return f"Query rejected: {exc}"

        try:
            rows = _rows_for(db, query)
        except Exception as exc:  # surfaced to the agent as tool output
            return f"Chart query failed: {exc}"

        if not rows or x not in rows[0] or y not in rows[0]:
            return (
                f"Chart query ran but didn't return both '{x}' and '{y}' "
                f"columns — got: {list(rows[0].keys()) if rows else 'no rows'}."
            )

        spec = {
            "type": "chart",
            "chart_type": chart_type,
            "x": x,
            "y": y,
            "title": title,
            "data": rows,
        }
        return json.dumps(spec, default=str)

    return plot_chart
