import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChartCard from "./ChartCard.jsx";

function TraceLog({ trace }) {
  if (!trace.length) return null;
  return (
    <div className="trace">
      <div className="trace-title">agent log</div>
      {trace.map((t, i) => (
        <div key={i} className={`trace-row ${t.status}`}>
          <span className="trace-tick">{t.status === "done" ? "✓" : "…"}</span>
          <span className="trace-tool">{t.tool}</span>
          {t.input && <span className="trace-input">{t.input}</span>}
        </div>
      ))}
    </div>
  );
}

function Message({ msg }) {
  const isAssistant = msg.role === "assistant";

  // Markdown parsing is expensive, so we only do it ONCE — after the
  // answer is final. While streaming, show cheap plain text; the final
  // event swaps in the parsed, formatted version.
  const renderMarkdown = isAssistant && !msg.streaming;

  return (
    <div className={`msg ${msg.role} ${msg.error ? "is-error" : ""}`}>
      <div className="msg-who">{msg.role === "user" ? "you" : "dbchat"}</div>
      <div className="msg-body">
        <TraceLog trace={msg.trace} />
        <div className="msg-text">
          {renderMarkdown ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.content}
            </ReactMarkdown>
          ) : (
            msg.content
          )}
          {msg.streaming && !msg.content && (
            <span className="thinking">thinking…</span>
          )}
          {msg.streaming && msg.content && <span className="caret" />}
        </div>
        {msg.charts && msg.charts.length > 0 && (
          <div className="chart-stack">
            {msg.charts.map((spec, i) => (
              <ChartCard key={i} spec={spec} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatWindow({ messages }) {
  const endRef = useRef(null);

  // Scroll instantly, and only when a NEW message appears — not on every
  // token update. Restarting a smooth-scroll animation hundreds of times
  // per answer was a large part of the perceived slowness.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "auto" });
  }, [messages.length]);

  return (
    <div className="chat-window">
      {messages.map((m, i) => (
        <Message key={i} msg={m} />
      ))}
      <div ref={endRef} />
    </div>
  );
}
