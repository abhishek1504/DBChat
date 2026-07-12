# DBChat — Chat with Your Database

Ask your database questions in plain English. A LangChain SQL agent inspects the schema, writes and runs read-only SQL, and streams its reasoning and answer back live — including a transaction-log view of every tool the agent used, and inline charts when you ask for one.

Originally built as a single Streamlit script, DBChat is now a proper two-tier app: a **FastAPI backend** around a UI-agnostic core, and a **React frontend**.

## Architecture

```
React (Vite, :5173)  ──HTTP/NDJSON stream──►  FastAPI (:8000)  ──►  core/  ──►  PostgreSQL / MySQL
                                                                      │
                                                                 LangChain SQL agent
                                                                 (Groq · OpenAI · Anthropic · Ollama)
```

```
DBChat/
├── core/                     # business logic — no UI imports, fully reusable
│   ├── llm.py                #   LLM factory — groq / openai / anthropic / ollama
│   ├── database.py           #   DBConfig + SQLDatabase factory
│   ├── sql_guard.py          #   read-only SQL enforcement (SELECT/WITH-only, single statement)
│   ├── charts.py             #   plot_chart tool — guarded SELECT -> chart JSON
│   └── agent.py              #   agent assembly (guarded query tool + chart tool) + ask()
├── server/
│   └── main.py               # FastAPI: /api/connect, /api/chat (streaming, incl. chart events)
├── frontend/
│   ├── index.html, vite.config.js, package.json
│   └── src/
│       ├── main.jsx          # React entry point
│       ├── App.jsx           # state + streaming orchestration + shell layout
│       ├── api.js            # fetch client, NDJSON stream reader
│       └── components/       # ConnectionPanel · ChatWindow · ChatInput · ChartCard
├── app.py                    # legacy Streamlit UI (kept as a thin dev shell)
└── requirements.txt
```

The design rule that makes this work: **nothing in `core/` imports any UI framework.** The same core is consumed by FastAPI today, was consumed by Streamlit yesterday, and could be consumed by a CLI or tests tomorrow.

## Layout

The sidebar holds the connection form (DB fields, model provider, API key/model). Once connected it collapses to a compact status card — database, table count, provider/model — with a "Change connection" link back to the form. The center stage always shows a status bar plus the chat window and input, so there's no full-page swap between "connecting" and "chatting" (this mirrors the original Streamlit `st.sidebar` + main-page layout).

## How a question flows

1. React posts your question to `/api/chat`.
2. The agent runs in a worker thread; a callback handler pushes lifecycle events into a queue.
3. FastAPI streams those events back as NDJSON — one JSON object per line: `tool_start`, `tool_end`, `token`, `chart`, `final`.
4. React folds the events into the UI: tool events render as the monospace **agent log**, tokens type out live, `chart` events render a Recharts card (bar/line/pie) under the answer, and the final answer is rendered as Markdown (tables included).

## Read-only safety

Every SQL-executing tool the agent can call is wrapped by `core/sql_guard.py` before the agent is assembled (see `core/agent.py`). A query is only allowed through if it is a single `SELECT` (or a `WITH …` CTE chain that only selects) — no `INSERT`/`UPDATE`/`DELETE`/`DDL`, and no stacked statements (`SELECT 1; DROP TABLE x;` is rejected as a whole, even though the first half is harmless). Rejections come back as a normal tool result, so the agent can see what happened and try a different query rather than crashing the chain.

This is a second line of defense, not a replacement for **using a read-only database user** — do both. Example for Postgres:

```sql
CREATE ROLE dbchat_reader LOGIN PASSWORD 'change-me';
GRANT CONNECT ON DATABASE shopdb TO dbchat_reader;
GRANT USAGE ON SCHEMA public TO dbchat_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dbchat_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dbchat_reader;
```

`core/database.py` also sets `postgresql_readonly=True` on the SQLAlchemy engine's execution options for Postgres connections as a best-effort third layer (silently ignored if the driver doesn't support it — the SQL-level guard above is the real backstop across every dialect).

## Multi-LLM support

`core/llm.py` is a small provider registry — `get_llm(provider=..., api_key=..., model_name=..., base_url=...)` builds a Groq, OpenAI, Anthropic, or local Ollama chat model behind the same interface. Ollama needs no API key; point `base_url` at your local (or remote) Ollama server (defaults to `http://localhost:11434`). Pick the provider per-connection from the sidebar.

## Prerequisites

- Python 3.9+ (3.11 recommended)
- Node.js 18+
- A PostgreSQL or MySQL database you can reach (a read-only user — see above)
- An API key for whichever provider you pick (Groq, OpenAI, or Anthropic), or a running Ollama instance for a local model

## Running it

Two terminals.

**Terminal 1 — backend** (project root):

```bash
python3 -m venv venv && source venv/bin/activate   # first time only
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" psycopg2-binary

uvicorn server.main:app --reload --port 8000
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:5173**, fill in the connection form in the sidebar (DB host, user, password, database, model provider, API key), and start asking. The Vite dev server proxies `/api/*` to the backend, so there is no CORS setup to worry about.

The legacy Streamlit UI still works as a quick dev harness: `streamlit run app.py` (Groq-only, no chart tool wired in yet).

## Example questions

- "What tables are in this database?"
- "How many orders were placed last month?"
- "Show the 10 most recently joined active employees for employer 88901."
- "Chart total orders by month for this year."

## Security notes

- **Use a read-only database user** (see SQL snippet above) — the SQL guard is defense in depth, not a substitute.
- API keys and DB passwords are held in backend memory per session and are never written to disk. Don't commit `.env`.
- Sessions live in an in-memory dict — a backend restart clears them. Fine for local use; swap for Redis before multi-user deployment.

## Troubleshooting

- **`Error loading ASGI app`** — run uvicorn from the project root, and confirm `server/main.py` defines `app = FastAPI(...)`.
- **401 / LLM setup failed at connect** — the backend validates the key (or, for Ollama, reachability) with a one-token ping during `/api/connect`; double check the key is pasted whole, or that the Ollama server is running and reachable at the base URL you gave.
- **"Query rejected: ..." in the chat** — the agent tried to run something other than a single SELECT; this is the read-only guard working as intended, not a bug.
- **Raw `|` pipes during streaming** — expected; Markdown (tables) renders once the answer finalizes.
- **Agent feels slow** — check the `[timing]` lines in the uvicorn console for true agent latency before blaming the UI.
- **`model_decommissioned`** — providers rotate models; override the model in the sidebar's optional "Model" field, or update `DEFAULT_MODELS` in `core/llm.py`.
- **`Tool call validation failed: ... 'x<|channel|>commentary' which was not in request.tools`** — this is a known upstream bug in Groq's `openai/gpt-oss-20b`/`gpt-oss-120b` models: special "harmony" format tokens leak into tool names during function calling (affects LangChain, vLLM, and LM Studio alike, not specific to this app). The Groq default here is `llama-3.3-70b-versatile`, which doesn't have this issue — if you've overridden the model to a gpt-oss variant in the sidebar, switch back.
- **`Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.`** — the model attempted a tool call and generated malformed arguments; this happens occasionally with any tool-calling model, not just gpt-oss. `core/agent.py`'s `ask()` automatically retries once (`MAX_TOOL_CALL_RETRIES`), since it's usually a one-off generation glitch — every retried call still goes through the same read-only guard, so retrying is free. If it fails twice in a row, the error message now includes the model's actual malformed output (`Model's malformed tool call: ...`) instead of just the generic top-level message, which is the actual diagnostic signal Groq's API buries in the response body and doesn't surface by default.

## Roadmap

1. ~~Modularise into `core/`~~ ✅
2. ~~FastAPI backend + React frontend~~ ✅
3. ~~Read-only safety guard (least-privilege DB user, SELECT-only enforcement)~~ ✅
4. ~~Multi-LLM support (OpenAI, Anthropic, Ollama) via `core/llm.py`~~ ✅
5. ~~Charts and visual reporting~~ ✅
6. Conversation memory (follow-up questions) — LangGraph migration
7. Multi-database dialects (MySQL, SQLite) via `core/database.py` — MySQL URI already templated, needs end-to-end testing
8. RAG over schema docs / few-shot NL→SQL examples

## Tech Stack

[React](https://react.dev) · [Vite](https://vitejs.dev) · [Recharts](https://recharts.org) · [FastAPI](https://fastapi.tiangolo.com) · [LangChain](https://python.langchain.com) · [Groq](https://groq.com) · [OpenAI](https://openai.com) · [Anthropic](https://anthropic.com) · [Ollama](https://ollama.com) · [SQLAlchemy](https://www.sqlalchemy.org) · PostgreSQL · MySQL
