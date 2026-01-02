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

## Agents

Use prefixes (you can combine multiple in one line):

```text
life: list reminders
health: help me build a sleep routine
ds: quiz me on train/test split
code: help debug this traceback
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

## Profile Memory (Stable Facts)

The assistant can store stable facts/preferences/goals in `user_profiles` and inject them into the system prompt every turn.

Example prompt:

```text
life: save my timezone as Asia/Ulaanbaatar
```

## Tuning

Edit `app/config.py`:
- `MODEL` / `MODEL_CONTEXT_TOKENS`
- `MEMORY_INJECT_K`, `MEMORY_INJECT_CANDIDATE_LIMIT`, `MEMORY_INJECT_MAX_TOKENS`
- `PROFILE_INJECT_MAX_TOKENS`

If `tiktoken` is installed, token budgets are more accurate; otherwise the app falls back to conservative byte-based estimates.

## Repo Hygiene

Keep exports and secrets out of git:
- `.env`, `data/assistant.sqlite3`, `data/import/`, `conversations.json`, `chat.html`, `*.zip` are ignored by `.gitignore`.
