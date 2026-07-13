"""DBChat — FastAPI backend.

Reuses core/ unchanged. Exposes:
  POST /api/connect   -> create a session (DB + LLM + agent), return session_id
  POST /api/chat      -> stream agent events as NDJSON (one JSON object per line)
  GET  /api/health    -> liveness check

Run:  uvicorn server.main:app --reload --port 8000
(from the project root, venv active)
"""

import json
import queue
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from core.agent import ask, build_agent
from core.database import DBConfig, get_database
from core.llm import get_llm

app = FastAPI(title="DBChat API")

# CORS: allows the Vite dev server to call us directly if you skip the
# proxy. With the proxy configured in vite.config.js this is belt-and-braces.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------- sessions
# In-memory session store: fine for local/dev, swap for Redis in prod.
# Each session holds the constructed agent + visible table names.
SESSIONS: Dict[str, Dict[str, Any]] = {}


class ConnectRequest(BaseModel):
    host: str
    user: str
    password: str
    database: str
    provider: str = "groq"          # "groq" | "openai" | "anthropic" | "ollama"
    api_key: str = ""               # ignored for provider="ollama"
    model_name: Optional[str] = None
    base_url: Optional[str] = None  # ollama only, e.g. http://localhost:11434


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/connect")
def connect(req: ConnectRequest):
    # Server-side trim as well: never trust the client to have cleaned input.
    config = DBConfig(
        host=req.host.strip(),
        user=req.user.strip(),
        password=req.password.strip(),
        database=req.database.strip(),
    )
    provider = req.provider.strip().lower()
    api_key = req.api_key.strip()
    model_name = req.model_name.strip() if req.model_name else None
    base_url = req.base_url.strip() if req.base_url else None

    try:
        db = get_database(config)
        tables = list(db.get_usable_table_names())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {exc}")

    try:
        llm = get_llm(
            api_key=api_key,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
        )
        # Constructing the chat model never contacts the provider — a bad
        # key (or unreachable Ollama server) would only surface later, mid-
        # chat, as a confusing error. This one-token ping validates it NOW
        # so the user gets the error on this screen.
        llm.invoke("ping")
        agent = build_agent(llm, db, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"LLM setup failed: {exc}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "agent": agent,
        "tables": tables,
        "provider": provider,
        "model_name": model_name or "",
    }
    return {
        "session_id": session_id,
        "tables": tables,
        "provider": provider,
        "model_name": model_name or "",
    }


# ------------------------------------------------------- event streaming
class QueueCallbackHandler(BaseCallbackHandler):
    """Forwards agent lifecycle events into a queue.

    The agent runs synchronously in a worker thread; the HTTP response
    generator drains the queue and streams each event as one NDJSON line.
    This is the plain-Python equivalent of StreamlitCallbackHandler.

    Tool names aren't passed to on_tool_end by LangChain's callback API,
    so we track name-by-run_id from on_tool_start — that's what lets us
    tell a plot_chart result apart from a regular sql_db_query result and
    emit a dedicated "chart" event for it.
    """

    def __init__(self, q: "queue.Queue[Optional[dict]]"):
        self.q = q
        self._tool_names: Dict[str, str] = {}

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        name = (serialized or {}).get("name", "tool")
        if run_id is not None:
            self._tool_names[str(run_id)] = name
        self.q.put({"type": "tool_start", "tool": name, "input": str(input_str)[:400]})

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        name = self._tool_names.pop(str(run_id), "tool") if run_id is not None else "tool"
        # Since the LangGraph migration, tool output arrives wrapped in a
        # ToolMessage (str(ToolMessage(...)) is a Python repr like
        # "content='...' name='...' tool_call_id='...'", not the raw
        # string) — unwrap .content when present so this still works the
        # same as it did with the old AgentExecutor, which passed the raw
        # string straight through.
        text_output = str(getattr(output, "content", output))

        if name == "plot_chart":
            try:
                spec = json.loads(text_output)
            except (TypeError, ValueError):
                spec = None
            if isinstance(spec, dict) and spec.get("type") == "chart":
                self.q.put({"type": "chart", "spec": spec})
                self.q.put(
                    {
                        "type": "tool_end",
                        "output": f"chart ready: {spec.get('chart_type')} of "
                        f"{spec.get('y')} by {spec.get('x')}",
                    }
                )
                return

        self.q.put({"type": "tool_end", "output": text_output[:400]})

    def on_llm_new_token(self, token, **kwargs):
        if token:
            self.q.put({"type": "token", "token": token})


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session — connect first.")

    agent = session["agent"]
    q: "queue.Queue[Optional[dict]]" = queue.Queue()
    handler = QueueCallbackHandler(q)

    def run_agent():
        try:
            # thread_id=session_id is what gives this conversation
            # memory across turns (LangGraph's MemorySaver, attached in
            # build_agent) — same session id the frontend already holds,
            # no new plumbing needed. ask() also carries the transient
            # tool-call retry/diagnostics from core/agent.py.
            answer = ask(
                agent, req.message, callbacks=[handler], thread_id=req.session_id
            )
            q.put({"type": "final", "answer": answer})
        except Exception as exc:  # surfaced to the client as an event
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(None)  # sentinel: stream is done

    threading.Thread(target=run_agent, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
