# DBChat — Chat with Your Database

Ask your PostgreSQL database questions in plain English. A LangChain SQL agent inspects the schema, writes and runs the SQL, and streams its reasoning and answer back live — including a transaction-log view of every tool the agent used.

Originally built as a single Streamlit script, DBChat is now a proper two-tier app: a **FastAPI backend** around a UI-agnostic core, and a **React frontend**.

## Architecture

```
React (Vite, :5173)  ──HTTP/NDJSON stream──►  FastAPI (:8000)  ──►  core/  ──►  PostgreSQL
                                                                      │
                                                                 LangChain SQL agent
                                                                 (Groq · gpt-oss-20b)
```

```
DBChat/
├── core/                     # business logic — no UI imports, fully reusable
│   ├── llm.py                #   LLM factory (Groq today, multi-provider later)
│   ├── database.py           #   DBConfig + SQLDatabase factory
│   └── agent.py              #   agent assembly + ask()
├── server/
│   └── main.py               # FastAPI: /api/connect, /api/chat (streaming)
├── frontend/
│   ├── index.html, vite.config.js, package.json
│   └── src/
│       ├── main.jsx          # React entry point
│       ├── App.jsx           # state + streaming orchestration
│       ├── api.js            # fetch client, NDJSON stream reader
│       └── components/       # ConnectionPanel · ChatWindow · ChatInput
├── app.py                    # legacy Streamlit UI (kept as a thin dev shell)
└── requirements.txt
```

The design rule that makes this work: **nothing in `core/` imports any UI framework.** The same core is consumed by FastAPI today, was consumed by Streamlit yesterday, and could be consumed by a CLI or tests tomorrow.

## How a question flows

1. React posts your question to `/api/chat`.
2. The agent runs in a worker thread; a callback handler pushes lifecycle events into a queue.
3. FastAPI streams those events back as NDJSON — one JSON object per line: `tool_start`, `tool_end`, `token`, `final`.
4. React folds the events into the UI: tool events render as the monospace **agent log**, tokens type out live, and the final answer is rendered as Markdown (tables included).

## Prerequisites

- Python 3.9+ (3.11 recommended)
- Node.js 18+
- A PostgreSQL database you can reach
- A free Groq API key — [console.groq.com](https://console.groq.com)

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

Open **http://localhost:5173**, fill in the connection card (DB host, user, password, database, Groq API key), and start asking. The Vite dev server proxies `/api/*` to the backend, so there is no CORS setup to worry about.

The legacy Streamlit UI still works as a quick dev harness: `streamlit run app.py`.

## Example questions

- "What tables are in this database?"
- "How many orders were placed last month?"
- "Show the 10 most recently joined active employees for employer 88901."

## Security notes

- **Use a read-only database user.** The agent executes whatever SQL it writes; a least-privilege login (`GRANT SELECT` only) is the difference between a wrong answer and a lost table.
- API keys and DB passwords are held in backend memory per session and are never written to disk. Don't commit `.env`.
- Sessions live in an in-memory dict — a backend restart clears them. Fine for local use; swap for Redis before multi-user deployment.

## Troubleshooting

- **`Error loading ASGI app`** — run uvicorn from the project root, and confirm `server/main.py` defines `app = FastAPI(...)`.
- **401 Invalid API key at connect** — the backend validates the Groq key with a one-token ping during `/api/connect`; get a fresh key and paste it whole (it starts with `gsk_`).
- **Raw `|` pipes during streaming** — expected; Markdown (tables) renders once the answer finalizes.
- **Agent feels slow** — check the `[timing]` lines in the uvicorn console for true agent latency before blaming the UI.
- **`model_decommissioned`** — Groq rotates models; update `DEFAULT_MODEL` in `core/llm.py` (single source of truth) per [their model list](https://console.groq.com/docs/models).

## Roadmap

1. ~~Modularise into `core/`~~ ✅
2. ~~FastAPI backend + React frontend~~ ✅
3. Read-only safety guard (least-privilege DB user, SELECT-only enforcement)
4. Multi-LLM support (OpenAI, Anthropic, Ollama) via `core/llm.py`
5. Charts and visual reporting
6. Conversation memory (follow-up questions) — LangGraph migration
7. Multi-database dialects (MySQL, SQLite) via `core/database.py`

## Tech Stack

[React](https://react.dev) · [Vite](https://vitejs.dev) · [FastAPI](https://fastapi.tiangolo.com) · [LangChain](https://python.langchain.com) · [Groq](https://groq.com) · [SQLAlchemy](https://www.sqlalchemy.org) · PostgreSQL