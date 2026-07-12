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

from core.agent import build_agent
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
    groq_api_key: str


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
    api_key = req.groq_api_key.strip()

    try:
        db = get_database(config)
        tables = list(db.get_usable_table_names())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Database connection failed: {exc}")

    try:
        llm = get_llm(api_key=api_key)
        # Constructing ChatGroq never contacts Groq — a bad key would only
        # surface later, mid-chat, as a confusing 401. This one-token ping
        # validates the key NOW so the user gets the error on this screen.
        llm.invoke("ping")
        agent = build_agent(llm, db, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"LLM setup failed: {exc}")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"agent": agent, "tables": tables}
    return {"session_id": session_id, "tables": tables}


# ------------------------------------------------------- event streaming
class QueueCallbackHandler(BaseCallbackHandler):
    """Forwards agent lifecycle events into a queue.

    The agent runs synchronously in a worker thread; the HTTP response
    generator drains the queue and streams each event as one NDJSON line.
    This is the plain-Python equivalent of StreamlitCallbackHandler.
    """

    def __init__(self, q: "queue.Queue[Optional[dict]]"):
        self.q = q

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.q.put(
            {
                "type": "tool_start",
                "tool": (serialized or {}).get("name", "tool"),
                "input": str(input_str)[:400],
            }
        )

    def on_tool_end(self, output, **kwargs):
        self.q.put({"type": "tool_end", "output": str(output)[:400]})

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
            result = agent.invoke(
                {"input": req.message}, config={"callbacks": [handler]}
            )
            q.put({"type": "final", "answer": result["output"]})
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