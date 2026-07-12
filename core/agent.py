from typing import Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, Tool

from core.charts import make_chart_tool
from core.sql_guard import UnsafeQueryError, assert_select_only

QUERY_TOOL_NAME = "sql_db_query"


def _guard_query_tool(original: BaseTool) -> Tool:
    """Return a read-only-guarded replacement for the toolkit's SQL
    execution tool.

    The toolkit's tools are BaseTool subclasses (Pydantic models with a
    `_run` method), not plain function tools, so we don't mutate them —
    we wrap `original.run(...)` (the public entry point, same one the
    agent itself would call) inside a fresh function-based Tool. This is
    the actual enforcement point: whatever the agent generates, it passes
    through `assert_select_only` before it ever reaches the database.
    """

    def guarded(query: str) -> str:
        try:
            assert_select_only(query)
        except UnsafeQueryError as exc:
            # Returned as the tool's *output*, not raised — the agent sees
            # a normal tool result and can react (e.g. rewrite the query)
            # instead of the whole chain crashing.
            return f"Query rejected: {exc}"
        return original.run(query)

    # Reuse the original tool's args_schema (a pydantic model with a
    # `query: str` field). Without this, a bare Tool(func=...) falls back
    # to a generic single-string schema keyed "__arg1" instead of "query"
    # — the LLM keeps calling with "query" (matching the description and
    # its training on the standard SQL toolkit schema), the schema expects
    # "__arg1", and every call fails validation before guarded() ever runs.
    return Tool(
        name=original.name,
        description=original.description,
        func=guarded,
        args_schema=original.args_schema,
    )


class GuardedSQLDatabaseToolkit(SQLDatabaseToolkit):
    """SQLDatabaseToolkit whose SQL-execution tool is read-only-guarded.

    `create_sql_agent` doesn't accept a pre-built tool list — it only
    takes `toolkit=` or `db=` and calls `toolkit.get_tools()` internally
    to decide what the agent gets. So the guard has to live in an
    overridden `get_tools()`, not in a list we hand to `create_sql_agent`
    ourselves (that list would just be ignored).
    """

    def get_tools(self):
        tools = super().get_tools()
        return [
            _guard_query_tool(t) if t.name == QUERY_TOOL_NAME else t
            for t in tools
        ]


def build_agent(llm, db: SQLDatabase, verbose: bool = True, include_chart_tool: bool = True):
    """Assemble the SQL agent from its parts.

    Every SQL-executing tool is read-only-guarded before the agent is
    assembled — see core/sql_guard.py and GuardedSQLDatabaseToolkit above.
    Pair this with a least-privilege, SELECT-only DB user (see README) for
    defense in depth: the guard stops the agent from *attempting* writes,
    the DB user stops it from *succeeding* even if the guard were ever
    bypassed.
    """
    toolkit = GuardedSQLDatabaseToolkit(db=db, llm=llm)
    extra_tools = [make_chart_tool(db)] if include_chart_tool else []

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        extra_tools=extra_tools,
        verbose=verbose,
        agent_type="tool-calling",
    )


def ask(agent, question: str, callbacks: Optional[list] = None) -> str:
    """Run one question through the agent and return the answer text.
    Keeping this thin wrapper means the UI never needs to know about
    LangChain's invoke/run API details (which have changed before and will
    change again e.g. when we migrate to LangGraph)"""

    result = agent.invoke({"input": question}, config={"callbacks": callbacks or []})
    return result["output"]
