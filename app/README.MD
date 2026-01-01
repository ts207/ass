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
* Packages: `openai`, `python-dotenv`, `numpy`

---

## Setup

### 1) Create venv and install deps

```bash
cd ~/ass
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install openai python-dotenv numpy
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
* `EMBEDDING_MODEL` — embedding model for ChatGPT export memory (default `text-embedding-3-small`)
* `MAX_HISTORY_MESSAGES` — number of recent messages to include as context
* `DB_PATH` — SQLite file path

---

## ChatGPT Export Memory

This imports your ChatGPT `conversations.json` as a graph, builds FTS candidates, and optionally reranks with embeddings. The `memory_search_graph` tool is available to both `life` and `ds` agents.

### Steps

```bash
# Apply schema (safe IF NOT EXISTS, run once)
sqlite3 data/assistant.sqlite3 < app/schema.sql

# Import ChatGPT export (place conversations.json in repo root)
python scripts/import_chatgpt_export.py --export conversations.json --auto-agent

# Backfill embeddings (optional rerank; uses float32 blobs)
python scripts/backfill_embeddings.py --model text-embedding-3-small --batch 128

# Verify embeddings count
sqlite3 data/assistant.sqlite3 "select count(*) from chatgpt_node_embeddings;"
```

### Quick checks

```bash
sqlite3 data/assistant.sqlite3 ".tables"
sqlite3 data/assistant.sqlite3 "select count(*) from chatgpt_conversations;"
sqlite3 data/assistant.sqlite3 "select count(*) from chatgpt_nodes_fts;"
sqlite3 data/assistant.sqlite3 "select node_id,title from chatgpt_nodes_fts where chatgpt_nodes_fts match 'python' limit 3;"

# Lightweight end-to-end self-check (imports if export exists, embeds a tiny batch, runs a search)
python scripts/selfcheck_memory.py --auto-agent
```

The runtime tool builds a context window by walking parents (up) and the `main_child_id` chain (down) so responses stay coherent to the main line of the conversation.

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
# Project Review: Local-First Personal Assistant

## How it Works
- **CLI entrypoint (`app/chat.py`)**
  - Starts OpenAI client, initializes SQLite via `schema.sql`, and migrates legacy tables. Conversations are created per user and per agent to isolate history. A reminder watcher thread polls the DB every `--reminder-interval` seconds and prints due reminders to stdout.
  - User input supports commands (`/new`, `/list`, `/use <id>`, `/exit`, `/selftest`) and prefixed requests such as `life:` or `ds:`. Unprefixed text defaults to `life`. Each prefixed segment is routed to the appropriate agent with its own system prompt and tool allowlist.
  - Messages are truncated to `MAX_HISTORY_MESSAGES` and combined with the system prompt before calling the OpenAI Responses API through the tool loop.
- **Tool loop (`app/tool_loop.py`)**
  - Iteratively calls the Responses API with provided tool schemas. When a tool_call is emitted, arguments are parsed, dispatched to local implementations, and results are sent back as follow-up input until a final text response is produced or the iteration limit is reached.
- **Tool implementations (`app/tools.py`, `app/tool_runtime.py`)**
  - Life agent tools: create reminders and list pending reminders (stored in `reminders` table with UTC normalization).
  - Data Science agent tools: generate a short course, start tracking, fetch next lesson, grade submissions (heuristic score plus next lesson), and log ad-hoc progress entries.
- **Persistence (`app/db.py`, `app/schema.sql`)**
  - SQLite tables store conversations, messages, reminders, per-agent session mapping, DS courses/progress, datasets, and experiments. Migrations normalize agent names and backfill reminder timestamps.

## Observations
- Reminder worker only prints due reminders to stdout; no notification mechanism or snooze/update loop beyond status flip to `done`.
- Course/lesson content is template-based and heuristic; grading uses submission length rather than semantic evaluation.
- System prompts are minimal and do not yet leverage personal profile, calendar context, or preference memories.
- No automated tests cover the CLI command routing or tool loop to prevent regressions.

## Recommendations for a More Personalized Agent
- **Stabilize runtime**: add smoke tests for the CLI happy path (new conversation, reminder creation/listing, DS course creation/next-lesson flow) and log tool-call failures for debuggability.
- **User profile & preferences**: introduce a `profiles` table with timezone, working hours, health goals, dietary constraints, study schedule, and preferred notification channels. Load this into system prompts for both agents.
- **Richer reminder lifecycle**: add recurrence (`rrule` is accepted but unused in scheduling), snooze/skip actions, and optional delivery channels (desktop notification/webhook). Replace polling stdout with an extensible notifier interface.
- **Contextual memory**: implement lightweight RAG over past reminders, DS progress, and imported journal notes so the agent can adapt responses and lesson sequencing.
- **Assessment quality**: for the DS agent, move grading to rubric-driven checks (maybe local evaluation scripts) and track lesson artifacts; add `datasets`/`experiments` CRUD tools to close the loop.

## Suggested Next Roadmap
1. **Runtime guardrails**: add smoke tests and a dev command that runs the tool loop against stubbed tool outputs to catch regressions early.
2. **Profile-aware prompts**: add profile CRUD tool + prompt injection so agents tailor reminders and study plans to the individual.
3. **Notification plumbing**: abstract reminder delivery (stdout, webhook, email); add background scheduler or APScheduler-based worker for accuracy.
4. **Memory & insights**: add a `memory_search` tool over combined reminders/progress and optional document embeddings to personalize planning.
5. **Learning engine upgrades**: support multi-lesson branching, prerequisites, and rubric-based grading; log artifacts and feedback for longitudinal insights.
