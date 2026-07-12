from typing import Optional

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
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

    return Tool(name=original.name, description=original.description, func=guarded)


def build_agent(llm, db: SQLDatabase, verbose: bool = True, include_chart_tool: bool = True):
    """Assemble the SQL agent from its parts.

    Every SQL-executing tool is read-only-guarded before the agent is
    assembled — see core/sql_guard.py. Pair this with a least-privilege,
    SELECT-only DB user (see README) for defense in depth: the guard stops
    the agent from *attempting* writes, the DB user stops it from
    *succeeding* even if the guard were ever bypassed.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = [
        _guard_query_tool(t) if t.name == QUERY_TOOL_NAME else t
        for t in toolkit.get_tools()
    ]

    if include_chart_tool:
        tools.append(make_chart_tool(db))

    return create_sql_agent(
        llm=llm,
        tools=tools,
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
