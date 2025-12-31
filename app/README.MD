````markdown
# Personal Assistant (Life + DS) — CLI + Tools + SQLite

A local-first personal assistant that:
- Maintains **persistent conversation state** in SQLite
- Supports multiple “agents” via prefixes:
  - `life:` reminders, routines, health/food logging
  - `ds:` data-science tutoring and progress tracking (extendable)
- Uses the OpenAI **Responses API** with **tool calling** (functions) to execute actions and store results locally

> This project is designed to be extended: add new tools (functions), new agents, and RAG memory over your exported ChatGPT history.

---

## Features

### ✅ Conversation persistence
- Conversations and messages are stored in a local SQLite DB.
- Supports:
  - `/new` create new conversation
  - `/list` list conversations
  - `/use <id>` switch conversation
  - Auto-resume latest conversation on startup (if implemented)

### ✅ Tool calling (Ticket C)
- The assistant can call local tools such as:
  - `create_reminder`, `list_reminders`
  - `ds_record_progress` (or similar DS progress tools)
- Tool outputs are saved to SQLite.

### ✅ Multi-intent inputs
You can run multiple agent actions in one line:
```text
life: list reminders, ds: log that I studied pandas groupby score 7 notes struggled with multiindex
````

### ✅ Per-agent memory separation (optional)

If enabled, `life` and `ds` can have separate conversation threads so their histories do not mix.

---

## Project layout

```text
ass/
  app/
    __init__.py
    chat.py
    config.py
    db.py
    schema.sql
    tool_loop.py
    tool_runtime.py
    tool_schemas.py
    tools.py
  data/
    assistant.sqlite3            # created at runtime (ignored by git)
    .gitkeep                     # keeps folder in repo
  scripts/                       # optional: import/export, RAG, etc.
  .env                           # OPENAI_API_KEY (ignored by git)
  .gitignore
  README.md
```

---

## Requirements

* Python 3.12+
* An OpenAI API key with access to the chosen model

---

## Setup

### 1) Create venv and install deps

```bash
cd ~/ass
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install openai python-dotenv
```

### 2) Set your API key

Create a `.env` file in the repo root:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-REPLACE_ME
EOF
```

### 3) Run

```bash
python3 -m app.chat
```

---

## Usage

### CLI commands

* `/new` — start a new conversation
* `/list` — list recent conversations
* `/use <id>` — switch to a conversation id
* `/exit` — quit

### Agent prefixes

#### Life agent

Examples:

```text
life: remind me tomorrow at 9am to take meds
life: list reminders
```

#### Data science agent

Examples:

```text
ds: quiz me on train/test split and leakage
ds: log that I studied pandas joins score 8 notes got confused by suffixes
```

### Multi-intent in one input

```text
life: list reminders, ds: log that I studied linux CLI for data work score 7 notes practiced grep/awk
```

---

## Database

The assistant stores data locally in:

* `data/assistant.sqlite3`

Useful inspection commands:

```bash
sqlite3 data/assistant.sqlite3 ".tables"
sqlite3 data/assistant.sqlite3 "select title,due_at,status from reminders order by due_at desc limit 10;"
sqlite3 data/assistant.sqlite3 "select topic,score,notes,created_at from ds_progress order by created_at desc limit 10;"
```

---

## Configuration

Edit `app/config.py`:

* `MODEL` — the OpenAI model name to use
* `MAX_HISTORY_MESSAGES` — number of recent messages to include as context
* `DB_PATH` — SQLite file path

---

## Extending

### Add a new tool

1. Add a tool schema in `app/tool_schemas.py`
2. Implement the tool in `app/tools.py`
3. Route it in `app/tool_runtime.py`
4. Add the tool name to the agent’s tool allowlist

### Add a new agent

1. Add a new prefix (e.g., `research:`) in `chat.py` parsing
2. Add new tool list and system instructions
3. Optionally add per-agent conversation thread mapping

---

## Safety / notes

* Keep secrets out of the repo (`.env` must be ignored).
* Keep personal data exports (e.g., `conversations.json`) out of git (`data/import/` ignored).
* If you add command execution tools, enforce allowlists + sandbox + timeouts.

---

## Roadmap (suggested)

* Ticket D: reminder worker (reminders that fire)
* RAG: import ChatGPT export, chunk, embed, and add `memory_search`
* DS course engine: `create_course`, `next_lesson`, `grade_submission`
* Web UI: Streamlit or FastAPI + frontend

---

```
::contentReference[oaicite:0]{index=0}
```
