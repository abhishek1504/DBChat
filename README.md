# DBChat — Chat with Your Database

A Streamlit app that lets you query a PostgreSQL database in plain English. It uses a LangChain SQL agent powered by a Groq-hosted LLM: ask a question, and the agent inspects your schema, writes and runs the SQL, and answers conversationally — showing its intermediate steps live in the UI.

## How It Works

```
You (English question)
   → Streamlit chat UI
   → LangChain SQL agent (think → act → observe loop)
   → SQLDatabaseToolkit (list tables · describe schema · check query · run query)
   → PostgreSQL (via SQLAlchemy + psycopg2)
   → Answer streamed back to chat
```

The LLM is `openai/gpt-oss-20b` served on Groq, using native tool-calling.

## Prerequisites

- Python 3.10+ recommended (3.9 minimum)
- A running PostgreSQL database you can connect to
- A free Groq API key from [console.groq.com](https://console.groq.com)

## Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/abhishek1504/DBChat.git
cd DBChat

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary streamlit

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Configuration

Fill in the sidebar fields when the app loads:

| Field | Description |
|---|---|
| DB Host | PostgreSQL host, e.g. `localhost` or `host:5432` |
| Postgres User | Database username |
| DB Password | Database password |
| DB Name | Database to connect to |
| Groq API Key | Your key from console.groq.com |

Optionally, set the API key once via a `.env` file instead of typing it each run:

```
GROQ_API_KEY=your_key_here
```

> **Never commit `.env`** — it is listed in `.gitignore` for a reason.

## Usage

Once all fields are filled, a chat input appears. Ask things like:

- "What tables are in this database?"
- "How many orders were placed last month?"
- "Show me the top 5 customers by total spend."

The agent's reasoning steps (tables inspected, SQL executed) appear expandable in the chat via the Streamlit callback handler. Use the **Clear message history** button in the sidebar to reset the conversation.

## Project Structure

```
DBChat/
├── app.py             # The entire app: UI, DB connection, agent setup, chat loop
├── requirements.txt   # Python dependencies
├── .gitignore         # Excludes venv/, .env, caches
└── README.md
```

## Troubleshooting

- **`command not found: streamlit`** — activate the venv first (`source venv/bin/activate`).
- **`No module named 'psycopg2'`** — run `pip install psycopg2-binary` inside the venv.
- **`model_decommissioned` error** — Groq rotates models; check [current models](https://console.groq.com/docs/models) and update `model_name` in `app.py`.
- **Chat input never appears** — all five sidebar fields must be filled; the script stops until they are.

## Tech Stack

[Streamlit](https://streamlit.io) · [LangChain](https://python.langchain.com) · [Groq](https://groq.com) · [SQLAlchemy](https://www.sqlalchemy.org) · PostgreSQL
