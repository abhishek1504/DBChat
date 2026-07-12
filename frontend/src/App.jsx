import { useRef, useState } from "react";
import { connect, streamChat } from "./api.js";
import ConnectionPanel from "./components/ConnectionPanel.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import ChatInput from "./components/ChatInput.jsx";

export default function App() {
  const [session, setSession] = useState(null); // { session_id, tables, provider, model_name }
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);

  // Token batching: hundreds of state updates per answer (one per token)
  // makes React re-render constantly and the UI feel sluggish. We buffer
  // tokens and flush ~12x per second instead.
  const tokenBuffer = useRef("");
  const flushTimer = useRef(null);

  async function handleConnect(details) {
    const s = await connect(details); // throws on failure; panel shows it
    setSession(s);

    const sample = s.tables.slice(0, 8).join(", ");
    setMessages([
      {
        role: "assistant",
        content:
          `Connected — ${s.tables.length} table${
            s.tables.length === 1 ? "" : "s"
          } visible` +
          (s.tables.length > 8 ? ` (${sample}, …)` : ` (${sample})`) +
          `. What would you like to know?`,
        trace: [],
        charts: [],
      },
    ]);
  }

  function handleDisconnect() {
    setSession(null);
    setMessages([]);
  }

  async function handleSend(text) {
    if (!text.trim() || busy) return;
    setBusy(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text, trace: [], charts: [] },
      { role: "assistant", content: "", trace: [], charts: [], streaming: true },
    ]);

    const updateLast = (fn) =>
      setMessages((prev) => {
        const next = prev.slice();
        next[next.length - 1] = fn(next[next.length - 1]);
        return next;
      });

    const flushTokens = () => {
      const chunk = tokenBuffer.current;
      tokenBuffer.current = "";
      flushTimer.current = null;
      if (chunk) updateLast((m) => ({ ...m, content: m.content + chunk }));
    };

    const resetTokenState = () => {
      if (flushTimer.current) clearTimeout(flushTimer.current);
      flushTimer.current = null;
      tokenBuffer.current = "";
    };

    try {
      for await (const ev of streamChat(session.session_id, text)) {
        if (ev.type === "tool_start") {
          updateLast((m) => ({
            ...m,
            trace: [
              ...m.trace,
              { tool: ev.tool, input: ev.input, status: "running" },
            ],
          }));
        } else if (ev.type === "tool_end") {
          updateLast((m) => {
            const trace = m.trace.slice();
            for (let i = trace.length - 1; i >= 0; i--) {
              if (trace[i].status === "running") {
                trace[i] = { ...trace[i], status: "done", output: ev.output };
                break;
              }
            }
            return { ...m, trace };
          });
        } else if (ev.type === "chart") {
          updateLast((m) => ({ ...m, charts: [...m.charts, ev.spec] }));
        } else if (ev.type === "token") {
          tokenBuffer.current += ev.token;
          if (!flushTimer.current) {
            flushTimer.current = setTimeout(flushTokens, 80);
          }
        } else if (ev.type === "final") {
          // The final answer replaces whatever streamed in, so drop any
          // pending buffer rather than flushing it.
          resetTokenState();
          updateLast((m) => ({ ...m, content: ev.answer, streaming: false }));
        } else if (ev.type === "error") {
          resetTokenState();
          updateLast((m) => ({
            ...m,
            content: `Something went wrong: ${ev.message}`,
            streaming: false,
            error: true,
          }));
        }
      }
    } catch (err) {
      resetTokenState();
      updateLast((m) => ({
        ...m,
        content: `Request failed: ${err.message}`,
        streaming: false,
        error: true,
      }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          DB<span className="brand-accent">Chat</span>
        </div>
        <ConnectionPanel
          session={session}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
        />
      </aside>

      <main className="stage">
        <div className={`status-bar ${session ? "is-connected" : "is-idle"}`}>
          {session ? (
            <>
              <span className="status-dot" />
              connected · {session.tables.length} table
              {session.tables.length === 1 ? "" : "s"} · {session.provider}
              {session.model_name ? ` (${session.model_name})` : ""}
            </>
          ) : (
            <>Not connected — fill in the sidebar to start a session.</>
          )}
        </div>

        {session ? (
          <div className="chat-layout">
            <ChatWindow messages={messages} />
            <ChatInput onSend={handleSend} disabled={busy} />
          </div>
        ) : (
          <div className="stage-empty">
            <p>Connect a database using the sidebar to start asking questions.</p>
          </div>
        )}
      </main>
    </div>
  );
}
