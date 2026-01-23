# agents.md — Project Guidance for Codex

This file defines working rules, architecture expectations, and repo hygiene for this project.
Follow these instructions before making changes.

---

## 1) Mission

Keep the local-first assistant runnable and documented:

- CLI (`python -m app.chat`)
- Telegram bot (`python -m app.telegram_bot`)
- Streamlit UI (`streamlit run ui_streamlit.py`)
- Deterministic export analytics (`python scripts/analyze_export.py ...`)

---

## 2) Golden Path (Must Work)

From repo root on Linux/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Initialize the database using the method documented in `README.md`.

Run:

```bash
python -m app.chat
# optional
python -m app.telegram_bot
```

---

## 3) Repo Hygiene (Non-Negotiable)

- Never commit secrets or runtime artifacts:
  - `.env`, `data/`, `*.sqlite3`, `*.db`, logs, caches, `.venv/`, `__pycache__/`
- If you need to remove non-core/WIP content, move it to `attic/` instead of deleting.
- Keep `README.md` accurate when moving/renaming files or changing commands.

---

## 4) Environment Facts

- Runtime is Linux/WSL. Use Linux paths: `/home/<user>/...` or `/mnt/c/...`.
- Telegram cannot upload files to runtime by default; ask users to place files under `data/imports/`.
- Standard paths:
  - inputs: `data/imports/`
  - outputs: `data/exports/<run_id>/`

---

## 5) Agent + Coordinator Expectations

These rules must remain true in code:

- Router runs for every user message and returns JSON only.
- Router never calls tools directly.
- Coordinator executes tools, enforces permissions, and merges specialist output.
- Specialists return JSON only and propose tools; coordinator executes them.
- Tool usage, router decisions, and token usage should be logged to SQLite.

---

## 6) Deterministic Export Analysis Rules

When asked to analyze ChatGPT exports:

- Read inputs from `data/imports/` only.
- Write outputs under `data/exports/<run_id>/` with:
  - `normalized/`
  - `features/`
  - `report/`
  - `report/figures/`
- Analysis must be executed locally via tools (no “run this yourself” replies).

---

## 7) Testing Expectations

- If you touch core logic, run unit tests:
  - `python -m unittest discover tests`
- If you cannot run tests, say so and explain why.

