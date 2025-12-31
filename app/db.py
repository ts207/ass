import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect(db_path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn: sqlite3.Connection, schema_sql: str) -> None:
    conn.executescript(schema_sql)
    conn.commit()

def create_conversation(conn: sqlite3.Connection, user_id: str, title: Optional[str] = None) -> str:
    convo_id = f"local_{uuid.uuid4().hex}"
    now = _now_iso()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (convo_id, user_id, title, now, now),
    )
    conn.commit()
    return convo_id

def list_conversations(conn: sqlite3.Connection, user_id: str, limit: int = 20) -> List[Tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT id, COALESCE(title,'' ) AS title, updated_at FROM conversations "
        "WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [(r["id"], r["title"], r["updated_at"]) for r in rows]

def touch_conversation(conn: sqlite3.Connection, convo_id: str) -> None:
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now_iso(), convo_id))
    conn.commit()

def add_message(conn: sqlite3.Connection, convo_id: str, role: str, content: str) -> str:
    msg_id = f"msg_{uuid.uuid4().hex}"
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, convo_id, role, content, _now_iso()),
    )
    touch_conversation(conn, convo_id)
    return msg_id

def get_recent_messages(conn: sqlite3.Connection, convo_id: str, max_messages: int) -> List[Dict[str, str]]:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id=? "
        "ORDER BY created_at DESC LIMIT ?",
        (convo_id, max_messages),
    ).fetchall()
    # DB query returns newest-first; reverse for chronological order
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]
def get_latest_conversation_id(conn, user_id: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    return row["id"] if row else None
def get_agent_conversation_id(conn, user_id: str, agent_name: str) -> str | None:
    row = conn.execute(
        "SELECT conversation_id FROM agent_sessions WHERE user_id=? AND agent_name=?",
        (user_id, agent_name),
    ).fetchone()
    return row["conversation_id"] if row else None

def set_agent_conversation_id(conn, user_id: str, agent_name: str, conversation_id: str) -> None:
    now = _now_iso()
    conn.execute(
        "INSERT INTO agent_sessions (user_id, agent_name, conversation_id, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_id, agent_name) DO UPDATE SET conversation_id=excluded.conversation_id, updated_at=excluded.updated_at",
        (user_id, agent_name, conversation_id, now),
    )
    conn.commit()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def run_migrations(conn: sqlite3.Connection) -> None:
    # Add due_at_utc and fired_at columns for reminders
    if _has_column(conn, "reminders", "due_at") and not _has_column(conn, "reminders", "due_at_utc"):
        conn.execute("ALTER TABLE reminders ADD COLUMN due_at_utc TEXT")
    if _has_column(conn, "reminders", "status") and not _has_column(conn, "reminders", "fired_at"):
        conn.execute("ALTER TABLE reminders ADD COLUMN fired_at TEXT")

    # Code progress table (kept for legacy logging)
    exists_code = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='code_progress'"
    ).fetchone()
    if not exists_code:
        conn.execute(
            "CREATE TABLE code_progress ("
            "id TEXT PRIMARY KEY, "
            "user_id TEXT NOT NULL, "
            "topic TEXT NOT NULL, "
            "notes TEXT, "
            "evidence_path TEXT, "
            "created_at TEXT NOT NULL)"
        )

    # DS tables (courses, progress, datasets, experiments, and ds_progress)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS courses ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "course_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS course_progress ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "course_id TEXT NOT NULL, "
        "lesson_key TEXT NOT NULL, "
        "status TEXT NOT NULL, "
        "score REAL, "
        "attempts INTEGER DEFAULT 0, "
        "updated_at TEXT NOT NULL, "
        "UNIQUE(user_id, course_id, lesson_key))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ds_progress ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "topic TEXT NOT NULL, "
        "score REAL, "
        "notes TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS datasets ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "path TEXT NOT NULL, "
        "schema_json TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS experiments ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "goal TEXT NOT NULL, "
        "approach_json TEXT, "
        "metrics_json TEXT, "
        "artifact_path TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_course_progress_user_course "
        "ON course_progress(user_id, course_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ds_progress_user_time "
        "ON ds_progress(user_id, created_at)"
    )

    # Backfill due_at_utc where missing
    rows = conn.execute(
        "SELECT id, due_at FROM reminders WHERE due_at_utc IS NULL"
    ).fetchall()
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["due_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            due_at_utc = dt.astimezone(timezone.utc).isoformat()
            conn.execute("UPDATE reminders SET due_at_utc=? WHERE id=?", (due_at_utc, r["id"]))
        except Exception:
            continue

    # Agent session table migration: normalize legacy agent names to ds and refresh constraint
    try:
        existing_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
        ).fetchone()
        needs_upgrade = existing_sql and ("'cyber'" in existing_sql["sql"] or "'ds'" not in existing_sql["sql"])
        if needs_upgrade:
            conn.execute("ALTER TABLE agent_sessions RENAME TO agent_sessions_old")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "user_id TEXT NOT NULL, "
                "agent_name TEXT NOT NULL CHECK(agent_name IN ('life','ds')), "
                "conversation_id TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, "
                "PRIMARY KEY (user_id, agent_name))"
            )
            conn.execute(
                "INSERT OR REPLACE INTO agent_sessions (user_id, agent_name, conversation_id, updated_at) "
                "SELECT user_id, "
                "CASE WHEN agent_name IN ('cyber','code') THEN 'ds' ELSE agent_name END, "
                "conversation_id, updated_at FROM agent_sessions_old"
            )
            conn.execute("DROP TABLE agent_sessions_old")
    except Exception:
        pass

    conn.commit()
