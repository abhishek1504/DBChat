import { useRef, useState } from "react";
import { connect, streamChat } from "./api.js";
import ConnectionPanel from "./components/ConnectionPanel.jsx";
import ChatWindow from "./components/ChatWindow.jsx";
import ChatInput from "./components/ChatInput.jsx";

export default function App() {
  const [session, setSession] = useState(null); // { session_id, tables }
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
      },
    ]);
  }

  async function handleSend(text) {
    if (!text.trim() || busy) return;
    setBusy(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text, trace: [] },
      { role: "assistant", content: "", trace: [], streaming: true },
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
    <div className="app">
      <header className="masthead">
        <span className="brand">
          DB<span className="brand-accent">Chat</span>
        </span>
        {session && (
          <span className="conn-badge" title={session.tables.join(", ")}>
            ● connected · {session.tables.length} tables
          </span>
        )}
      </header>

      {!session ? (
        <ConnectionPanel onConnect={handleConnect} />
      ) : (
        <main className="chat-layout">
          <ChatWindow messages={messages} />
          <ChatInput onSend={handleSend} disabled={busy} />
        </main>
      )}
    </div>
  );
}
