import { useState } from "react";

const FIELDS = [
  { name: "host", label: "Host", placeholder: "localhost", type: "text" },
  { name: "user", label: "User", placeholder: "postgres", type: "text" },
  { name: "password", label: "Password", placeholder: "", type: "password" },
  { name: "database", label: "Database", placeholder: "shopdb", type: "text" },
  { name: "groq_api_key", label: "Groq API key", placeholder: "gsk_…", type: "password" },
];

export default function ConnectionPanel({ onConnect }) {
  const [form, setForm] = useState({
    host: "",
    user: "",
    password: "",
    database: "",
    groq_api_key: "",
  });
  const [error, setError] = useState("");
  const [connecting, setConnecting] = useState(false);

  const complete = Object.values(form).every((v) => v.trim() !== "");

  async function submit() {
    if (!complete || connecting) return;
    setConnecting(true);
    setError("");

    // Trim every field: pasted API keys and hostnames routinely carry
    // stray spaces or a trailing newline, which cause confusing 401s
    // and connection failures much later.
    const cleaned = Object.fromEntries(
      Object.entries(form).map(([key, value]) => [key, value.trim()])
    );

    try {
      await onConnect(cleaned);
    } catch (err) {
      setError(err.message);
    } finally {
      setConnecting(false);
    }
  }

  return (
    <main className="connect-wrap">
      <section className="connect-card">
        <h1>Open a session</h1>
        <p className="connect-sub">
          Point DBChat at a PostgreSQL database, then ask it questions in plain
          English.
        </p>

        {FIELDS.map((f) => (
          <label key={f.name} className="field">
            <span>{f.label}</span>
            <input
              type={f.type}
              placeholder={f.placeholder}
              value={form[f.name]}
              onChange={(e) => setForm({ ...form, [f.name]: e.target.value })}
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </label>
        ))}

        {error && <p className="connect-error">{error}</p>}

        <button
          className="connect-btn"
          onClick={submit}
          disabled={!complete || connecting}
        >
          {connecting ? "Connecting…" : "Connect"}
        </button>
      </section>
    </main>
  );
}
