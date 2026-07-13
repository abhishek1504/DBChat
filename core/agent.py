from typing import Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from core.charts import make_chart_tool
from core.sql_guard import UnsafeQueryError, assert_select_only

QUERY_TOOL_NAME = "sql_db_query"
DEFAULT_THREAD_ID = "default"

# The default SQL_PREFIX tells the model not to SELECT * and to return
# "the answer" — it says nothing about *how* to present rows, so the
# model tends to narrate results in prose instead of a table, especially
# for wide tables (it'll happily describe 30 columns in a paragraph
# rather than lay them out). A soft "format nicely" instruction wasn't
# concrete enough to reliably override that; a hard column cap plus a
# worked example gives it a much harder edge to follow. The frontend
# already renders Markdown tables (react-markdown + remark-gfm), so this
# is purely about what the model produces, not how it's rendered.
_TABLE_FORMAT_INSTRUCTIONS = """
When your final answer includes more than one row, you MUST present it as
a GitHub-flavored Markdown table — never as a paragraph describing the
columns. Pick at most 5 columns: an identifying column (name and/or id)
plus only the columns the question is actually about. Never include every
column from the table just because the query returned them — write a
narrower SELECT instead.

Example — question: "list active employees for employer 123"

| employee_id | name       | status |
|-------------|------------|--------|
| 501         | Jane Doe   | active |
| 502         | John Smith | active |

Do not add a paragraph enumerating every other field on the row; if the
user wants more columns they will ask for them by name.
"""

# {dialect} / {top_k} placeholders are filled in by build_agent() below —
# create_sql_agent used to do this substitution for us; create_react_agent
# has no notion of a SQL toolkit at all, so it's on us now.
AGENT_PREFIX_TEMPLATE = SQL_PREFIX + "\n" + _TABLE_FORMAT_INSTRUCTIONS

# Kept as an alias for anything importing the pre-migration name.
AGENT_PREFIX = AGENT_PREFIX_TEMPLATE


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

    Unlike `create_sql_agent` (which only accepted `toolkit=`/`db=` and
    called `.get_tools()` internally), `create_react_agent` just takes a
    plain `tools=` list — so this subclass is no longer load-bearing the
    way it was pre-migration, but it's kept as-is since `.get_tools()` is
    still the one place the guard has to be applied before tools are
    handed to *any* agent constructor.
    """

    def get_tools(self):
        tools = super().get_tools()
        return [
            _guard_query_tool(t) if t.name == QUERY_TOOL_NAME else t
            for t in tools
        ]


def build_agent(llm, db: SQLDatabase, verbose: bool = True, include_chart_tool: bool = True):
    """Assemble the SQL agent from its parts.

    Built on LangGraph's `create_react_agent` (a compiled StateGraph)
    rather than the legacy `AgentExecutor` — see `ask()` below for what
    that changes call-site-wise. A fresh `MemorySaver` checkpointer is
    attached per agent (i.e. per session, since `server/main.py` calls
    this once per `/api/connect`), which is what gives conversational
    memory: follow-up questions in the same session now see prior turns,
    keyed by `thread_id` (server/main.py uses the session_id).

    Every SQL-executing tool is read-only-guarded before the agent is
    assembled — see core/sql_guard.py and GuardedSQLDatabaseToolkit above.
    Pair this with a least-privilege, SELECT-only DB user (see README) for
    defense in depth: the guard stops the agent from *attempting* writes,
    the DB user stops it from *succeeding* even if the guard were ever
    bypassed.
    """
    toolkit = GuardedSQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    if include_chart_tool:
        tools.append(make_chart_tool(db))

    prompt = AGENT_PREFIX_TEMPLATE.format(dialect=db.dialect, top_k=10)
    checkpointer = MemorySaver()

    return create_react_agent(
        llm,
        tools,
        prompt=prompt,
        checkpointer=checkpointer,
        debug=verbose,
    )


# Extra attempts (beyond the first) for a provider-side "malformed tool
# call" error — e.g. Groq's tool_use_failed ("Failed to call a function.
# Please adjust your prompt."). These are usually a one-off generation
# glitch rather than a real problem with the question, and since every
# retried call still goes through the same read-only guard, retrying
# costs nothing but a few seconds.
MAX_TOOL_CALL_RETRIES = 1


def _is_transient_tool_call_failure(exc: Exception) -> bool:
    """Best-effort, provider-agnostic detection of a malformed-tool-call
    error worth retrying. Checks the structured error body first (how
    Groq's SDK — and most OpenAI-compatible ones — report this), falling
    back to a text match so this still degrades gracefully for providers
    that raise something else entirely."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("code") == "tool_use_failed":
            return True
    return "failed to call a function" in str(exc).lower()


def _failed_generation(exc: Exception) -> Optional[str]:
    """Pull the model's raw malformed output out of a Groq-style error
    body, if present, so it's actually visible instead of just the
    generic top-level message ("...See 'failed_generation' for more
    details.") which points at data callers don't otherwise get."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return error.get("failed_generation")
    return None


def ask(
    agent,
    question: str,
    callbacks: Optional[list] = None,
    thread_id: str = DEFAULT_THREAD_ID,
) -> str:
    """Run one question through the agent and return the answer text.

    Keeping this thin wrapper means the UI never needs to know about the
    underlying framework's invoke API — which is exactly why this
    survived the LangGraph migration unchanged in shape: callers still
    just call `ask(agent, question)`. What changed underneath:
      - input is now `{"messages": [...]}` instead of `{"input": ...}`,
        and the answer comes from the last message instead of ["output"].
      - `thread_id` selects which conversation history (if any) this
        question continues — pass the session id to get real multi-turn
        memory; omit it and every call is independent, as before.

    Retries once on a transient malformed-tool-call error from the
    provider (see MAX_TOOL_CALL_RETRIES above). If it still fails, the
    model's raw failed generation is appended to the error when the
    provider makes it available, instead of surfacing only the generic
    "adjust your prompt" message.
    """
    last_exc: Optional[Exception] = None
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks or [],
    }

    for attempt in range(MAX_TOOL_CALL_RETRIES + 1):
        try:
            result = agent.invoke({"messages": [("user", question)]}, config=config)
            return result["messages"][-1].content
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_TOOL_CALL_RETRIES and _is_transient_tool_call_failure(exc):
                continue
            failed_generation = _failed_generation(exc)
            if failed_generation:
                raise RuntimeError(
                    f"{exc} | Model's malformed tool call: {failed_generation}"
                ) from exc
            raise

    # Unreachable — the loop above always returns or raises — but keeps
    # type checkers happy and guards against a future refactor slipping up.
    assert last_exc is not None
    raise last_exc
