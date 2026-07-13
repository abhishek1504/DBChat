import { useState } from "react";

const DB_FIELDS = [
  { name: "host", label: "Host", placeholder: "localhost", type: "text" },
  { name: "user", label: "User", placeholder: "postgres", type: "text" },
  { name: "password", label: "Password", placeholder: "", type: "password" },
  { name: "database", label: "Database", placeholder: "shopdb", type: "text" },
];

const PROVIDERS = [
  { value: "groq", label: "Groq", keyLabel: "Groq API key", keyPlaceholder: "gsk_…" },
  { value: "openai", label: "OpenAI", keyLabel: "OpenAI API key", keyPlaceholder: "sk-…" },
  { value: "anthropic", label: "Anthropic", keyLabel: "Anthropic API key", keyPlaceholder: "sk-ant-…" },
  { value: "ollama", label: "Ollama (local)", keyLabel: null, keyPlaceholder: "" },
];

const EMPTY_FORM = {
  host: "",
  user: "",
  password: "",
  database: "",
  provider: "groq",
  api_key: "",
  model_name: "",
  base_url: "",
};

export default function ConnectionPanel({ session, onConnect, onDisconnect }) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState("");
  const [connecting, setConnecting] = useState(false);

  // Once connected, collapse to a compact status card — the full form
  // would eat the sidebar's width for no reason once a session exists.
  if (session) {
    return (
      <div className="conn-compact">
        <div className="conn-compact-row">
          <span className="status-dot" />
          <span>Connected</span>
        </div>
        <dl className="conn-compact-meta">
          <dt>Database</dt>
          <dd>{form.database || "—"}</dd>
          <dt>Tables</dt>
          <dd>{session.tables.length}</dd>
          <dt>Provider</dt>
          <dd>
            {session.provider}
            {session.model_name ? ` · ${session.model_name}` : ""}
          </dd>
        </dl>
        <button className="conn-change-btn" onClick={onDisconnect}>
          Change connection
        </button>
      </div>
    );
  }

  const providerInfo = PROVIDERS.find((p) => p.value === form.provider) ?? PROVIDERS[0];
  const needsApiKey = providerInfo.keyLabel !== null;

  const complete =
    DB_FIELDS.every((f) => form[f.name].trim() !== "") &&
    (!needsApiKey || form.api_key.trim() !== "");

  function setField(name, value) {
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function submit() {
    if (!complete || connecting) return;
    setConnecting(true);
    setError("");

    // Trim every field: pasted API keys and hostnames routinely carry
    // stray spaces or a trailing newline, which cause confusing 401s
    // and connection failures much later.
    const payload = {
      host: form.host.trim(),
      user: form.user.trim(),
      password: form.password.trim(),
      database: form.database.trim(),
      provider: form.provider,
      api_key: needsApiKey ? form.api_key.trim() : "",
      model_name: form.model_name.trim() || undefined,
      base_url: form.provider === "ollama" ? form.base_url.trim() || undefined : undefined,
    };

    try {
      await onConnect(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setConnecting(false);
    }
  }

  return (
    <section className="connect-card">
      <h1>Open a session</h1>
      <p className="connect-sub">
        Point DBChat at a database, then ask it questions in plain English.
      </p>

      {DB_FIELDS.map((f) => (
        <label key={f.name} className="field">
          <span>{f.label}</span>
          <input
            type={f.type}
            placeholder={f.placeholder}
            value={form[f.name]}
            onChange={(e) => setField(f.name, e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </label>
      ))}

      <label className="field">
        <span>Model provider</span>
        <select
          value={form.provider}
          onChange={(e) => setField("provider", e.target.value)}
        >
          {PROVIDERS.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </label>

      {needsApiKey ? (
        <label className="field">
          <span>{providerInfo.keyLabel}</span>
          <input
            type="password"
            placeholder={providerInfo.keyPlaceholder}
            value={form.api_key}
            onChange={(e) => setField("api_key", e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </label>
      ) : (
        <label className="field">
          <span>Ollama base URL (optional)</span>
          <input
            type="text"
            placeholder="http://localhost:11434"
            value={form.base_url}
            onChange={(e) => setField("base_url", e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </label>
      )}

      <label className="field">
        <span>Model (optional)</span>
        <input
          type="text"
          placeholder="uses a sensible default"
          value={form.model_name}
          onChange={(e) => setField("model_name", e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      </label>

      {error && <p className="connect-error">{error}</p>}

      <button
        className="connect-btn"
        onClick={submit}
        disabled={!complete || connecting}
      >
        {connecting ? "Connecting…" : "Connect"}
      </button>
    </section>
  );
}
