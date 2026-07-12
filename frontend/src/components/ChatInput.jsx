import { useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState("");

  function send() {
    if (!text.trim() || disabled) return;
    onSend(text);
    setText("");
  }

  return (
    <div className="input-bar">
      <input
        value={text}
        placeholder="Ask anything about the database"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && send()}
        disabled={disabled}
      />
      <button onClick={send} disabled={disabled || !text.trim()}>
        {disabled ? "Working…" : "Ask"}
      </button>
    </div>
  );
}
