# ass — Local‑First Assistant (Life + DS) with SQLite + Tools

Local-first assistant with:
- SQLite-backed conversations + reminders + DS progress
- OpenAI Responses API tool-calling loop
- Optional long-term memory from your ChatGPT export (graph + FTS + embeddings rerank)

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install openai python-dotenv numpy streamlit

cat > .env <<'EOF'
OPENAI_API_KEY=sk-REPLACE_ME
EOF

python -m app.chat
```

## Streamlit UI

```bash
streamlit run ui_streamlit.py
```

## Deploy/Run Options

### systemd (CLI daemon)
1) Edit `systemd/ass.service` (set `USER`, paths, `OPENAI_API_KEY`, and any SMTP/IMAP envs).
2) Install + start:
   ```bash
   sudo cp systemd/ass.service /etc/systemd/system/ass.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now ass.service
   ```
3) Logs: `journalctl -u ass.service -f`

### Docker
Build and run the Streamlit UI:
```bash
docker build -t ass .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-REPLACE ass
```
For CLI inside container:
```bash
docker run --rm -it -e OPENAI_API_KEY=sk-REPLACE ass python -m app.chat
```

Mount `data/` as a volume if you want persistence outside the container.

## First-Time Setup Checklist
- [ ] Create and activate venv: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] Install deps: `pip install -U pip && pip install -r requirements.txt`
- [ ] Add `.env` with `OPENAI_API_KEY=...` (and SMTP/IMAP if using email)
- [ ] Initialize DB: `python - <<'PY'\nfrom pathlib import Path\nfrom app.db import connect, init_db, run_migrations\nschema = Path('app/schema.sql').read_text(encoding='utf-8')\nconn = connect('data/assistant.sqlite3'); init_db(conn, schema); run_migrations(conn); conn.close()\nPY`
- [ ] Set your profile timezone: `general: save my timezone as <IANA>` in CLI/Streamlit
- [ ] Decide permissions: enable only what you need (`/perm net on|off`, `/perm fs on|off`, `/perm shell on|off`, `/perm exec on|off`)
- [ ] Optional: enable systemd service (`systemd/ass.service`) or run Streamlit (`streamlit run ui_streamlit.py`)
## How It Works (Architecture)
- `app/chat.py`: CLI entrypoint; keeps a per-agent conversation thread; reminder watcher.
- `ui_streamlit.py`: Web UI; sidebar controls for agent + permissions.
- `app/tools_*.py`: Tool implementations split by domain; `app/tools.py` re-exports for the model.
- `app/tool_runtime.py`: Validates permissions per tool and logs to the audit table.
- `app/tool_loop.py`: Model ↔ tool loop using OpenAI Responses API.
- `app/db.py` + `app/schema.sql`: Schema/migrations for SQLite.
- Memory: Imported ChatGPT export (FTS + optional embeddings) + user profile memory injected into prompts.

## Starter Commands (CLI)
- Reminders/Tasks: `life: remind me tomorrow at 9am to take meds`; `life: create a task to file taxes due next Friday`
- Calendar: `life: list events between 2026-01-10 and 2026-01-20`
- Health: `health: log_metric {"metric":"sleep_hours","value":7.5}`; `health: add a med schedule for atenolol 50mg at 08:00 and 20:00`
- General planning: `general: plan my week with 3 priorities and create tasks`
- Web (needs `/perm net on`): `general: web_search latest NICE guideline on hypertension`
- Code (needs `/perm shell on` + `/perm fs on`): `code: search_code {"query":"TODO","path":"app"}`

## Operations
- Logs: `journalctl -u ass.service -f` (systemd) or container logs; CLI prints reminders to stdout.
- DB backup: copy `data/assistant.sqlite3`; keep `.env` out of git.
- Permissions: use `/perm` (CLI) or Streamlit sidebar; keep shell/fs/exec off unless needed.
- Audit: `general: audit_log_list {"limit":20}` to review tool calls/errors.

## Agents

Use prefixes (you can combine multiple in one line):

```text
general: plan my week and delegate tasks
life: list reminders
life: create a task to renew passport next month
health: help me build a sleep routine
ds: quiz me on train/test split
code: help debug this traceback
general: web_search latest NICE guideline on hypertension
```

The `general` agent is the coordinator and can delegate tasks to the other agents when appropriate.

Streamlit shows a collapsible “Memory used” panel (sidebar) for the last turn.

## Run It (Step-by-Step)
1) Activate your venv and env vars: `source .venv/bin/activate && export $(cat .env | xargs)`.
2) Start the CLI: `python -m app.chat` (keeps reminder watcher alive). For UI: `streamlit run ui_streamlit.py`.
3) Set safety rails: `/perm` then toggle net/fs/shell/exec as needed.
4) Talk to agents: prefix with `general:` (delegates), `life:`, `health:`, `ds:`, `code:`.
5) Background use: keep the CLI running or install `systemd/ass.service` so reminders fire while logged out. For email/SMS pushes, set SMTP/IMAP envs and have the agent send reminders via those channels.

## What This Project Is
- **Agents**: life (tasks/reminders/calendar/contacts/docs/expenses), health (metrics/meds/appointments/meals/workouts), ds (courses/SQL/runs), code (dev helper), general (coordinator + web/kb).
- **Storage**: Single SQLite DB (`data/assistant.sqlite3`) for conversations, reminders, tasks, contacts, docs, expenses, health logs, DS runs, user profile, permissions, audit log, and imported ChatGPT memory.
- **Safety**: Permission gates (`mode`, `allow_network`, `allow_fs_write`, `allow_shell`, `allow_exec`) + audit log.
- **Interfaces**: CLI (`python -m app.chat`) with reminder watcher; Streamlit UI (`ui_streamlit.py`); Docker/systemd options.

## Permissions (Safety Controls)

Tools are gated by a simple permissions model:
- `mode`: `read` or `write`
- capability flags: `allow_network`, `allow_fs_write`, `allow_shell`, `allow_exec`

CLI quick controls:

```text
/perm                # show current permissions
/perm read|write     # set mode
/perm net on|off     # allow network tools (fetch_url/web_search, etc)
/perm fs on|off      # allow filesystem writes (export_pdf, apply_patch)
/perm shell on|off   # allow shell tools (search_code, run_command, git)
/perm exec on|off    # allow code execution tools (run_python)
```

## ChatGPT Export Memory (Graph + Hybrid Search)

This imports `conversations.json` or `chat.html` as a **lossless conversation graph**, builds an SQLite **FTS5** index, and optionally reranks FTS results using OpenAI **embeddings**.

```bash
# Apply schema (safe; uses IF NOT EXISTS)
sqlite3 data/assistant.sqlite3 < app/schema.sql

# Import export (supports conversations.json or chat.html)
python scripts/import_chatgpt_export.py --export conversations.json --auto-agent
# or:
python scripts/import_chatgpt_export.py --export chat.html --auto-agent

# Backfill embeddings (optional; improves relevance)
python scripts/backfill_embeddings.py --model text-embedding-3-small --batch 128 --limit 0

# Verify
sqlite3 data/assistant.sqlite3 "select count(*) from chatgpt_node_embeddings;"
```

Self-check:

```bash
python scripts/selfcheck_memory.py --auto-agent --query python --search-agent ds
```

Eval harness (iterate on retrieval quality):

```bash
python scripts/eval_memory.py --queries queries.txt --debug
```

## Profile Memory (Stable Facts)

The assistant can store stable facts/preferences/goals in `user_profiles` and inject them into the system prompt every turn.

Example prompt:

```text
life: save my timezone as Asia/Ulaanbaatar
```

## Tuning

Edit `app/config.py`:
- `MODEL` / `MODEL_CONTEXT_TOKENS`
- `REQUEST_BUDGET_FRACTION` (70% by default; lower if you see context errors)
- `MEMORY_INJECT_K`, `MEMORY_INJECT_CANDIDATE_LIMIT`, `MEMORY_INJECT_MAX_TOKENS`
- `PROFILE_INJECT_MAX_TOKENS`
- `MAX_HISTORY_MESSAGES`

Defaults are set for a 128k context model. For smaller contexts (<=32k), consider dropping `REQUEST_BUDGET_FRACTION` to ~0.5 and `MEMORY_INJECT_MAX_TOKENS` to ~800.

If `tiktoken` is installed, token budgets are more accurate; otherwise the app falls back to conservative byte-based estimates.

## Repo Hygiene

Keep exports and secrets out of git:
- `.env`, `data/assistant.sqlite3`, `data/import/`, `conversations.json`, `chat.html`, `*.zip` are ignored by `.gitignore`.

## Outlook / Recommended Updates
- Add connectors: Google/Outlook Calendar + IMAP/SMTP + mobile push to deliver reminders off-device.
- Reliability: background scheduler (APScheduler or systemd timer) to re-check reminders and missed jobs.
- Safety: per-tool allowlists, rate limits, and clearer consent prompts for shell/fs/exec tools.
- Quality: unit tests around tool routing, permissions, and audit logging; add smoke tests for CLI and Streamlit.
- Memory: compress or summarize old threads; add eviction policy and small embedding cache.
