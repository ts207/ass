import argparse
import threading
from datetime import datetime, timezone
from openai import OpenAI
from pathlib import Path

from .config import DB_PATH, MODEL, MAX_HISTORY_MESSAGES
from .db import (
    connect,
    init_db,
    create_conversation,
    list_conversations,
    add_message,
    get_recent_messages,
    get_latest_conversation_id,
    get_agent_conversation_id,
    set_agent_conversation_id,
    run_migrations,
)
from .tool_loop import run_with_tools
from .tool_schemas import LIFE_TOOLS, DS_TOOLS

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

def split_prefixed_requests(s: str):
    """
    Accepts inputs like:
      "life: list reminders; ds: next lesson"
    Returns list of (agent, text) preserving order.
    If no prefixes, returns [("life", s)].
    """
    import re

    pattern = re.compile(r"\\b(life|ds|code):", re.IGNORECASE)
    matches = list(pattern.finditer(s))
    if not matches:
        return [("life", s.strip())] if s.strip() else []

    pieces = []
    for i, m in enumerate(matches):
        agent = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        chunk = s[start:end].strip(" ;,")
        if chunk:
            mapped = "ds" if agent in ("ds", "code") else "life"
            pieces.append((mapped, chunk.strip()))
    return pieces


def _msg(role: str, text: str):
    # Simple message shape accepted by Responses API for inputs
    return {"role": role, "content": text}


def _parse_args():
    parser = argparse.ArgumentParser(description="CLI personal assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--reminder-interval", type=int, default=30, help="Seconds between reminder checks")
    return parser.parse_args()


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_dt(dt_str: str):
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def check_due_reminders(conn, *, debug: bool = False):
    """Find scheduled reminders that are due, mark them done, and return them."""
    now = _utcnow()
    now_iso = now.isoformat()
    rows = conn.execute(
        "SELECT id, title, due_at, due_at_utc FROM reminders "
        "WHERE status='scheduled' AND (due_at_utc IS NULL OR due_at_utc<=?)",
        (now_iso,),
    ).fetchall()
    due_ids = []
    due_entries = []
    for r in rows:
        dt_str = r["due_at_utc"] or r["due_at"]
        dt = _parse_dt(dt_str)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.astimezone(timezone.utc) <= now:
            due_ids.append(r["id"])
            due_entries.append(dict(r))
    if due_ids:
        conn.execute(
            f"UPDATE reminders SET status='done', fired_at=?, updated_at=? WHERE id IN ({','.join(['?']*len(due_ids))})",
            [_utcnow().isoformat(), _utcnow().isoformat(), *due_ids],
        )
        conn.commit()
        if debug:
            print(f"[debug] marked {len(due_ids)} reminders done")
    return due_entries


def start_reminder_watcher(db_path: Path, *, interval: int, debug: bool = False):
    stop_event = threading.Event()

    def _loop():
        conn = connect(db_path, check_same_thread=False)
        try:
            while not stop_event.is_set():
                due = check_due_reminders(conn, debug=debug)
                for d in due:
                    print(f"\n[reminder] {d['title']} (due {d['due_at']})")
                stop_event.wait(interval)
        finally:
            conn.close()

    thread = threading.Thread(target=_loop, daemon=True, name="reminder-watcher")
    thread.start()
    return stop_event, thread


def main():
    args = _parse_args()
    debug = args.debug

    client = OpenAI()
    user_id = "local_user"

    conn = connect(DB_PATH)
    init_db(conn, SCHEMA_PATH.read_text(encoding="utf-8"))
    run_migrations(conn)

    convo_id = get_latest_conversation_id(conn, user_id)
    if not convo_id:
        convo_id = create_conversation(conn, user_id, title="CLI chat")

    print(f"Conversation: {convo_id}")
    print("Commands: /new, /list, /use <id>, /exit")

    stop_event, watcher_thread = start_reminder_watcher(DB_PATH, interval=args.reminder_interval, debug=debug)

    try:
        while True:
            user_text = input("> ").strip()
            if not user_text:
                continue

            # --- commands ---
            if user_text == "/exit":
                break

            if user_text == "/new":
                convo_id = create_conversation(conn, user_id, title="CLI chat")
                print(f"Conversation: {convo_id}")
                continue

            if user_text == "/list":
                convos = list_conversations(conn, user_id, limit=20)
                for cid, title, updated in convos:
                    print(f"{cid}  {updated}  {title}")
                continue

            if user_text == "/selftest":
                print("Try prompts:")
                print("  life: remind me tomorrow at 9am to take meds")
                print("  ds: create a short course on feature engineering for beginners")
                print("  ds: next lesson for my course_id")
                continue

            if user_text.startswith("/use "):
                convo_id = user_text.split(" ", 1)[1].strip()
                print(f"Conversation: {convo_id}")
                continue

            # --- handle one or more prefixed agent requests ---
            requests = split_prefixed_requests(user_text)
            responses = []

            for agent, text in requests:
                display_agent = "ds" if agent == "ds" else agent

                # each agent keeps its own conversation thread for memory isolation
                agent_convo = get_agent_conversation_id(conn, user_id, agent)
                if not agent_convo:
                    agent_convo = create_conversation(conn, user_id, title=f"{display_agent} thread")
                    set_agent_conversation_id(conn, user_id, agent, agent_convo)

                tools_schema = LIFE_TOOLS if agent == "life" else DS_TOOLS
                system_instructions = (
                    "You are the Life Manager. User timezone: Europe/Stockholm. "
                    "If you schedule reminders, always output due_at as ISO 8601 with timezone offset."
                    if agent == "life"
                    else
                    "You are a Data Science Course Assistant. You can design short courses, serve lessons, grade submissions, and adapt the plan based on progress."
                )

                history = get_recent_messages(conn, agent_convo, MAX_HISTORY_MESSAGES)

                input_items = [_msg("system", system_instructions)]
                input_items += [_msg(m["role"], m["content"]) for m in history]
                input_items += [_msg("user", text)]

                final_text, _ = run_with_tools(
                    client=client,
                    model=MODEL,
                    tools_schema=tools_schema,
                    input_items=input_items,
                    conn=conn,
                    user_id=user_id,
                    debug=debug,
                )

                # Persist each sub-request as its own turn (keeps memory coherent)
                add_message(conn, agent_convo, "user", f"{display_agent}: {text}")
                add_message(conn, agent_convo, "assistant", final_text or "")

                responses.append(f"[{display_agent}] {final_text}")

            print("\n".join(responses))
    finally:
        stop_event.set()
        watcher_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
