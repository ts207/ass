import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from app.tools_core import _truncate_json_str

def _fts_query(user_query: str, *, mode: str = "auto") -> str:
    """
    Make a safer FTS5 MATCH query from free text.
    - splits into tokens
    - ANDs tokens by default
    - quotes tokens with special chars
    """
    tokens = [t for t in re.split(r"\s+", user_query.strip()) if t]
    if not tokens:
        return ""
    out = []
    for t in tokens:
        # Quote tokens with non-word chars or FTS operators to avoid syntax errors.
        if re.search(r'[^A-Za-z0-9_]', t) or re.search(r'["\*\:\-\(\)\[\]\{\}\^~]', t):
            t = '"' + t.replace('"', '""') + '"'
        out.append(t)
    if mode == "or":
        joiner = " OR "
    elif mode == "and":
        joiner = " AND "
    else:
        joiner = " OR " if len(out) > 6 else " AND "
    return joiner.join(out)

def chatgpt_fts_candidates(conn, query: str, agent: str, limit: int = 100, *, mode: str = "auto") -> List[Dict[str, Any]]:
    """
    Return candidate nodes from FTS5 ordered by bm25 rank (best first).
    Includes extra fields used for hybrid scoring.
    """
    q = _fts_query(query, mode=mode)
    if not q or limit <= 0:
        return []

    # Oversample to allow dedupe and secondary filtering without starving results.
    raw_limit = min(max(limit * 5, limit), 5000)
    rows = conn.execute(
        """
        SELECT
          chatgpt_nodes_fts.node_id,
          chatgpt_nodes_fts.conversation_id,
          chatgpt_nodes_fts.title,
          bm25(chatgpt_nodes_fts) AS bm25_rank,
          chatgpt_nodes.create_time
        FROM chatgpt_nodes_fts
        JOIN chatgpt_nodes ON chatgpt_nodes.node_id = chatgpt_nodes_fts.node_id
        WHERE chatgpt_nodes_fts MATCH ?
          AND (? = 'general' OR chatgpt_nodes_fts.agent = ? OR chatgpt_nodes_fts.agent = 'general')
        ORDER BY bm25(chatgpt_nodes_fts) ASC
        LIMIT ?
        """,
        (q, agent, agent, raw_limit),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        node_id = r[0]
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(
            {
                "node_id": node_id,
                "conversation_id": r[1],
                "title": r[2] or "",
                "bm25": float(r[3]) if r[3] is not None else 0.0,
                "create_time": int(r[4] or 0),
            }
        )
        if len(out) >= limit:
            break
    return out


def chatgpt_get_embeddings(conn, node_ids: List[str]) -> Dict[str, Any]:
    """
    Return {node_id: np.ndarray} for the provided ids where embeddings exist.
    """
    import numpy as np

    if not node_ids:
        return {}
    placeholders = ",".join(["?"] * len(node_ids))
    rows = conn.execute(
        f"""
        SELECT node_id, dim, vec
        FROM chatgpt_node_embeddings
        WHERE node_id IN ({placeholders})
        """,
        node_ids,
    ).fetchall()

    out: Dict[str, Any] = {}
    for nid, dim, blob in rows:
        if blob is None:
            continue
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim and arr.size >= dim:
            arr = arr[:dim]
        out[nid] = arr
    return out

def chatgpt_get_node(conn, node_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT node_id, conversation_id, parent_id, role, text, create_time, is_message, main_child_id, agent
        FROM chatgpt_nodes
        WHERE node_id = ?
        """,
        (node_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "node_id": row[0],
        "conversation_id": row[1],
        "parent_id": row[2],
        "role": row[3],
        "text": row[4],
        "create_time": row[5],
        "is_message": row[6],
        "main_child_id": row[7],
        "agent": row[8],
    }

def chatgpt_get_conversation_title(conn, conversation_id: str) -> str:
    row = conn.execute(
        "SELECT title FROM chatgpt_conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    return row[0] if row and row[0] else ""

def chatgpt_context_window(
    conn,
    node_id: str,
    up: int = 6,
    down: int = 4,
) -> List[Dict[str, Any]]:
    """
    Faithful context:
    - walk up via parent_id (skip is_message=0)
    - walk down via main_child_id (skip is_message=0)
    Returns messages ordered oldest -> newest.
    """
    center = chatgpt_get_node(conn, node_id)
    if not center:
        return []

    # Walk up (parents)
    up_nodes: List[Dict[str, Any]] = []
    cur = center
    steps = 0
    while steps < up and cur.get("parent_id"):
        cur = chatgpt_get_node(conn, cur["parent_id"])
        if not cur:
            break
        if cur["is_message"] == 1 and (cur.get("text") or "").strip():
            up_nodes.append(cur)
            steps += 1

    up_nodes.reverse()  # oldest -> newest

    # Center (include even if not message? typically message)
    mid_nodes: List[Dict[str, Any]] = []
    if center["is_message"] == 1 and (center.get("text") or "").strip():
        mid_nodes.append(center)

    # Walk down (mainline children)
    down_nodes: List[Dict[str, Any]] = []
    cur = center
    steps = 0
    while steps < down and cur.get("main_child_id"):
        cur = chatgpt_get_node(conn, cur["main_child_id"])
        if not cur:
            break
        if cur["is_message"] == 1 and (cur.get("text") or "").strip():
            down_nodes.append(cur)
            steps += 1

    return up_nodes + mid_nodes + down_nodes

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect(db_path: str | Path, *, check_same_thread: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=timeout,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
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

def get_recent_thread_messages(conn: sqlite3.Connection, convo_id: str, max_messages: int) -> List[Dict[str, str]]:
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE conversation_id=? AND role IN ('user','assistant') "
        "ORDER BY created_at DESC LIMIT ?",
        (convo_id, max_messages),
    ).fetchall()
    rows = list(reversed(rows))
    return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

def get_conversation_message_count(conn: sqlite3.Connection, convo_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id=? AND role IN ('user','assistant')",
        (convo_id,),
    ).fetchone()
    return int(row["cnt"] or 0) if row else 0

def get_messages_after_offset(conn: sqlite3.Connection, convo_id: str, offset: int) -> List[Dict[str, Any]]:
    off = max(0, int(offset or 0))
    rows = conn.execute(
        "SELECT id, role, content, created_at FROM messages "
        "WHERE conversation_id=? AND role IN ('user','assistant') "
        "ORDER BY created_at ASC LIMIT -1 OFFSET ?",
        (convo_id, off),
    ).fetchall()
    return [dict(r) for r in rows]

def get_conversation_summary(conn: sqlite3.Connection, conversation_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT conversation_id, user_id, agent, summary, message_count, last_message_id, updated_at "
        "FROM conversation_summaries WHERE conversation_id=?",
        (conversation_id,),
    ).fetchone()
    return dict(row) if row else None

def upsert_conversation_summary(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    user_id: str,
    agent: str,
    summary: str,
    message_count: int,
    last_message_id: str | None,
) -> None:
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO conversation_summaries (
          conversation_id, user_id, agent, summary, message_count, last_message_id, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id) DO UPDATE SET
          user_id=excluded.user_id,
          agent=excluded.agent,
          summary=excluded.summary,
          message_count=excluded.message_count,
          last_message_id=excluded.last_message_id,
          updated_at=excluded.updated_at
        """,
        (
            conversation_id,
            user_id,
            agent,
            summary or "",
            int(message_count or 0),
            last_message_id,
            now,
        ),
    )
    conn.commit()

def get_last_user_message_id(conn: sqlite3.Connection, convo_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM messages WHERE conversation_id=? AND role='user' "
        "ORDER BY created_at DESC LIMIT 1",
        (convo_id,),
    ).fetchone()
    return row["id"] if row else None

def _truncate_snippet(text: str, max_chars: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

def record_turn_memory_usage(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_id: str,
    agent: str,
    results: Optional[List[Dict[str, Any]]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> None:
    now = _now_iso()
    results = results or []

    score_map: Dict[str, Any] = {}
    if isinstance(debug, dict):
        for row in debug.get("top_scores") or []:
            node_id = row.get("node_id")
            if node_id:
                score_map[node_id] = row.get("score")

    meta_base: Dict[str, Any] = {}
    if isinstance(debug, dict):
        qinfo = debug.get("query")
        if qinfo:
            meta_base["query"] = qinfo
        if "candidates" in debug:
            meta_base["candidates"] = debug.get("candidates")
        if "used_embeddings" in debug:
            meta_base["used_embeddings"] = debug.get("used_embeddings")
        if "mmr_used" in debug:
            meta_base["mmr_used"] = debug.get("mmr_used")
        if "mmr_lambda" in debug:
            meta_base["mmr_lambda"] = debug.get("mmr_lambda")
        if "mmr_pool" in debug:
            meta_base["mmr_pool"] = debug.get("mmr_pool")
        if debug.get("fts_query"):
            meta_base["fts_query"] = debug.get("fts_query")
        if debug.get("fts_mode"):
            meta_base["fts_mode"] = debug.get("fts_mode")
        if debug.get("agent_tag"):
            meta_base["agent_tag"] = debug.get("agent_tag")

    def _meta_json(title: Optional[str] = None) -> Optional[str]:
        if not meta_base and not title:
            return None
        meta = dict(meta_base)
        if title:
            meta["title"] = title
        return json.dumps(meta, ensure_ascii=False)

    conn.execute(
        "DELETE FROM turn_memory_usage WHERE conversation_id=? AND turn_id=?",
        (conversation_id, turn_id),
    )

    if not results:
        conn.execute(
            "INSERT INTO turn_memory_usage "
            "(id, conversation_id, turn_id, agent, node_id, rank, score, snippet, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"tmu_{uuid.uuid4().hex}",
                conversation_id,
                turn_id,
                agent,
                None,
                0,
                None,
                None,
                _meta_json(),
                now,
            ),
        )
        conn.commit()
        return

    for idx, r in enumerate(results, start=1):
        node_id = r.get("node_id")
        title = (r.get("title") or "").strip() or None
        snippet = _truncate_snippet(r.get("context") or "")
        score = score_map.get(node_id)
        conn.execute(
            "INSERT INTO turn_memory_usage "
            "(id, conversation_id, turn_id, agent, node_id, rank, score, snippet, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"tmu_{uuid.uuid4().hex}",
                conversation_id,
                turn_id,
                agent,
                node_id,
                idx,
                score,
                snippet,
                _meta_json(title),
                now,
            ),
        )
    conn.commit()

def get_turn_memory_usage(
    conn: sqlite3.Connection,
    conversation_id: str,
    turn_id: str,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT node_id, rank, score, snippet, meta_json "
        "FROM turn_memory_usage WHERE conversation_id=? AND turn_id=? "
        "ORDER BY rank ASC",
        (conversation_id, turn_id),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        meta: Dict[str, Any] = {}
        raw = r["meta_json"] if hasattr(r, "keys") else r[4]
        if raw:
            try:
                meta = json.loads(raw)
            except Exception:
                meta = {}
        out.append(
            {
                "node_id": r["node_id"] if hasattr(r, "keys") else r[0],
                "rank": r["rank"] if hasattr(r, "keys") else r[1],
                "score": r["score"] if hasattr(r, "keys") else r[2],
                "snippet": r["snippet"] if hasattr(r, "keys") else r[3],
                "meta": meta,
            }
        )
    return out


def record_turn_tool_usage(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_id: str,
    agent: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    now = _now_iso()
    tool_calls = tool_calls or []
    conn.execute(
        "DELETE FROM turn_tool_usage WHERE conversation_id=? AND turn_id=?",
        (conversation_id, turn_id),
    )
    if not tool_calls:
        conn.commit()
        return

    for call in tool_calls:
        tool_name = call.get("tool_name") or call.get("name")
        status = call.get("status") or "ok"
        error = call.get("error")
        duration_ms = call.get("duration_ms")
        input_obj = call.get("input")
        output_obj = call.get("output")
        input_json = json.dumps(input_obj, ensure_ascii=False) if input_obj is not None else None
        output_json = json.dumps(output_obj, ensure_ascii=False) if output_obj is not None else None
        conn.execute(
            "INSERT INTO turn_tool_usage "
            "(id, conversation_id, turn_id, agent, tool_name, input_json, output_json, status, error, duration_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"ttu_{uuid.uuid4().hex}",
                conversation_id,
                turn_id,
                agent,
                tool_name,
                _truncate_json_str(input_json) if input_json else None,
                _truncate_json_str(output_json) if output_json else None,
                status,
                error,
                int(duration_ms) if duration_ms is not None else None,
                now,
            ),
        )
    conn.commit()


def get_turn_tool_usage(
    conn: sqlite3.Connection,
    conversation_id: str,
    turn_id: str,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT tool_name, status, error, duration_ms, input_json, output_json "
        "FROM turn_tool_usage WHERE conversation_id=? AND turn_id=? "
        "ORDER BY created_at ASC",
        (conversation_id, turn_id),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        def _load(raw):
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return raw
        out.append(
            {
                "tool_name": r["tool_name"] if hasattr(r, "keys") else r[0],
                "status": r["status"] if hasattr(r, "keys") else r[1],
                "error": r["error"] if hasattr(r, "keys") else r[2],
                "duration_ms": r["duration_ms"] if hasattr(r, "keys") else r[3],
                "input": _load(r["input_json"] if hasattr(r, "keys") else r[4]),
                "output": _load(r["output_json"] if hasattr(r, "keys") else r[5]),
            }
        )
    return out


def record_turn_token_usage(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_id: str,
    agent: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    tool_calls: Optional[int],
) -> None:
    now = _now_iso()
    conn.execute(
        "DELETE FROM turn_token_usage WHERE conversation_id=? AND turn_id=?",
        (conversation_id, turn_id),
    )
    conn.execute(
        "INSERT INTO turn_token_usage "
        "(id, conversation_id, turn_id, agent, model, prompt_tokens, completion_tokens, total_tokens, tool_calls, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"ttok_{uuid.uuid4().hex}",
            conversation_id,
            turn_id,
            agent,
            model,
            int(prompt_tokens) if prompt_tokens is not None else None,
            int(completion_tokens) if completion_tokens is not None else None,
            int(total_tokens) if total_tokens is not None else None,
            int(tool_calls) if tool_calls is not None else None,
            now,
        ),
    )
    conn.commit()


def record_turn_router_decision(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    turn_id: str,
    agent: str,
    decision: Dict[str, Any],
    raw_output: str | None = None,
) -> None:
    now = _now_iso()
    decision_payload = dict(decision or {})
    if raw_output:
        decision_payload["raw_output"] = raw_output
    proposed = decision_payload.get("proposed_tools")
    proposed_json = json.dumps(proposed, ensure_ascii=False) if proposed is not None else None
    decision_json = json.dumps(decision_payload, ensure_ascii=False)
    conn.execute(
        "DELETE FROM turn_router_decisions WHERE conversation_id=? AND turn_id=?",
        (conversation_id, turn_id),
    )
    conn.execute(
        "INSERT INTO turn_router_decisions "
        "(id, conversation_id, turn_id, agent, need_tools, task_type, confidence, proposed_tools_json, decision_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"trd_{uuid.uuid4().hex}",
            conversation_id,
            turn_id,
            agent,
            1 if decision_payload.get("need_tools") else 0,
            decision_payload.get("task_type"),
            float(decision_payload.get("confidence")) if decision_payload.get("confidence") is not None else None,
            _truncate_json_str(proposed_json) if proposed_json else None,
            _truncate_json_str(decision_json),
            now,
        ),
    )
    conn.commit()


def get_turn_token_usage(
    conn: sqlite3.Connection,
    conversation_id: str,
    turn_id: str,
) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT model, prompt_tokens, completion_tokens, total_tokens, tool_calls "
        "FROM turn_token_usage WHERE conversation_id=? AND turn_id=?",
        (conversation_id, turn_id),
    ).fetchone()
    if not row:
        return None
    return {
        "model": row["model"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "tool_calls": row["tool_calls"],
    }
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
    if _has_column(conn, "reminders", "notes") and not _has_column(conn, "reminders", "channels_json"):
        conn.execute("ALTER TABLE reminders ADD COLUMN channels_json TEXT")

    # Permissions: add filesystem read consent flag
    if _has_column(conn, "user_permissions", "allow_fs_write") and not _has_column(conn, "user_permissions", "allow_fs_read"):
        conn.execute("ALTER TABLE user_permissions ADD COLUMN allow_fs_read INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE user_permissions SET allow_fs_read=allow_fs_write")

    # Router decisions (per turn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_router_decisions ("
        "id TEXT PRIMARY KEY, "
        "conversation_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "need_tools INTEGER, "
        "task_type TEXT, "
        "confidence REAL, "
        "proposed_tools_json TEXT, "
        "decision_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "FOREIGN KEY(conversation_id) REFERENCES conversations(id))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_router_decisions_turn "
        "ON turn_router_decisions(conversation_id, turn_id)"
    )

    # Thread summaries (per conversation)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_summaries ("
        "conversation_id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "summary TEXT NOT NULL, "
        "message_count INTEGER NOT NULL DEFAULT 0, "
        "last_message_id TEXT, "
        "updated_at TEXT NOT NULL, "
        "FOREIGN KEY(conversation_id) REFERENCES conversations(id))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_summaries_user_agent "
        "ON conversation_summaries(user_id, agent)"
    )

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

    # Life manager: calendar, tasks, contacts, docs, email drafts, expenses
    conn.execute(
        "CREATE TABLE IF NOT EXISTS calendar_events ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "start_at TEXT NOT NULL, "
        "start_at_utc TEXT, "
        "end_at TEXT NOT NULL, "
        "end_at_utc TEXT, "
        "location TEXT, "
        "notes TEXT, "
        "status TEXT NOT NULL DEFAULT 'scheduled', "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_user_start "
        "ON calendar_events(user_id, start_at_utc)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "notes TEXT, "
        "priority INTEGER, "
        "due_at TEXT, "
        "due_at_utc TEXT, "
        "rrule TEXT, "
        "status TEXT NOT NULL DEFAULT 'open', "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, "
        "completed_at TEXT)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_user_due "
        "ON tasks(user_id, due_at_utc)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_user_status "
        "ON tasks(user_id, status)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS contacts ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "email TEXT, "
        "phone TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_contacts_user_name "
        "ON contacts(user_id, name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_contacts_user_email "
        "ON contacts(user_id, email)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "title TEXT NOT NULL, "
        "content TEXT NOT NULL, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_user_time "
        "ON documents(user_id, updated_at)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS email_drafts ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "to_json TEXT NOT NULL, "
        "subject TEXT NOT NULL, "
        "body TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'draft', "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_email_drafts_user_time "
        "ON email_drafts(user_id, updated_at)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS expenses ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "amount REAL NOT NULL, "
        "currency TEXT NOT NULL, "
        "category TEXT, "
        "merchant TEXT, "
        "notes TEXT, "
        "occurred_at TEXT, "
        "occurred_at_utc TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_expenses_user_time "
        "ON expenses(user_id, occurred_at_utc)"
    )

    # Health: metrics, medication schedules, appointments, meals, workouts
    conn.execute(
        "CREATE TABLE IF NOT EXISTS health_metrics ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "metric TEXT NOT NULL, "
        "value REAL NOT NULL, "
        "unit TEXT, "
        "recorded_at TEXT, "
        "recorded_at_utc TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_user_metric_time "
        "ON health_metrics(user_id, metric, recorded_at_utc)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS medication_schedules ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "medication TEXT NOT NULL, "
        "dose TEXT, "
        "unit TEXT, "
        "times_json TEXT NOT NULL, "
        "start_date TEXT, "
        "end_date TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS appointments ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "event_id TEXT NOT NULL, "
        "provider TEXT, "
        "reason TEXT, "
        "created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_appointments_user_time "
        "ON appointments(user_id, updated_at)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meals ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "summary TEXT NOT NULL, "
        "calories REAL, "
        "protein_g REAL, "
        "carbs_g REAL, "
        "fat_g REAL, "
        "recorded_at TEXT, "
        "recorded_at_utc TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_meals_user_time "
        "ON meals(user_id, recorded_at_utc)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS workouts ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "workout_type TEXT NOT NULL, "
        "duration_min REAL, "
        "intensity TEXT, "
        "calories REAL, "
        "recorded_at TEXT, "
        "recorded_at_utc TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_workouts_user_time "
        "ON workouts(user_id, recorded_at_utc)"
    )

    # DS: run tracking
    conn.execute(
        "CREATE TABLE IF NOT EXISTS ds_runs ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "params_json TEXT, "
        "metrics_json TEXT, "
        "notes TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ds_runs_user_time "
        "ON ds_runs(user_id, created_at)"
    )

    # Permissions / audit log
    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_permissions ("
        "user_id TEXT PRIMARY KEY, "
        "mode TEXT NOT NULL CHECK(mode IN ('read','write')), "
        "allow_network INTEGER NOT NULL DEFAULT 0, "
        "allow_fs_read INTEGER NOT NULL DEFAULT 0, "
        "allow_fs_write INTEGER NOT NULL DEFAULT 0, "
        "allow_shell INTEGER NOT NULL DEFAULT 0, "
        "allow_exec INTEGER NOT NULL DEFAULT 0, "
        "updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_log ("
        "id TEXT PRIMARY KEY, "
        "user_id TEXT NOT NULL, "
        "tool TEXT NOT NULL, "
        "payload_json TEXT, "
        "result_json TEXT, "
        "status TEXT NOT NULL, "
        "error TEXT, "
        "duration_ms INTEGER, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_user_time "
        "ON audit_log(user_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_tool_time "
        "ON audit_log(tool, created_at)"
    )

    if _has_column(conn, "audit_log", "id") and not _has_column(conn, "audit_log", "duration_ms"):
        conn.execute("ALTER TABLE audit_log ADD COLUMN duration_ms INTEGER")

    # Turn memory usage (for last-turn UI/debug)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_memory_usage ("
        "id TEXT PRIMARY KEY, "
        "conversation_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "node_id TEXT, "
        "rank INTEGER, "
        "score REAL, "
        "snippet TEXT, "
        "meta_json TEXT, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_memory_usage_convo_time "
        "ON turn_memory_usage(conversation_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_memory_usage_turn "
        "ON turn_memory_usage(conversation_id, turn_id)"
    )

    # Turn tool usage (for last-turn UI/debug)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_tool_usage ("
        "id TEXT PRIMARY KEY, "
        "conversation_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "tool_name TEXT, "
        "input_json TEXT, "
        "output_json TEXT, "
        "status TEXT NOT NULL, "
        "error TEXT, "
        "duration_ms INTEGER, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_tool_usage_convo_time "
        "ON turn_tool_usage(conversation_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_tool_usage_turn "
        "ON turn_tool_usage(conversation_id, turn_id)"
    )

    # Turn token usage (for last-turn UI/debug)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_token_usage ("
        "id TEXT PRIMARY KEY, "
        "conversation_id TEXT NOT NULL, "
        "turn_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "model TEXT NOT NULL, "
        "prompt_tokens INTEGER, "
        "completion_tokens INTEGER, "
        "total_tokens INTEGER, "
        "tool_calls INTEGER, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_token_usage_turn "
        "ON turn_token_usage(conversation_id, turn_id)"
    )

    # Tool policies (scoped permissions)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tool_policies ("
        "user_id TEXT NOT NULL, "
        "agent TEXT NOT NULL, "
        "tool_name TEXT NOT NULL, "
        "allow INTEGER NOT NULL, "
        "constraints_json TEXT, "
        "updated_at TEXT NOT NULL, "
        "PRIMARY KEY (user_id, agent, tool_name))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_policies_user_agent "
        "ON tool_policies(user_id, agent)"
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

    # Agent session table migration: refresh constraint and normalize legacy agent names
    try:
        existing_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='agent_sessions'"
        ).fetchone()
        desired = ("life", "ds", "health", "code", "general")
        existing_stmt = ""
        if existing_sql:
            try:
                existing_stmt = existing_sql["sql"] or ""
            except Exception:
                try:
                    existing_stmt = existing_sql[0] or ""
                except Exception:
                    existing_stmt = ""
        needs_upgrade = bool(existing_stmt) and (
            ("'cyber'" in existing_stmt)
            or any(f"'{a}'" not in existing_stmt for a in desired)
        )
        if needs_upgrade:
            conn.execute("ALTER TABLE agent_sessions RENAME TO agent_sessions_old")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "user_id TEXT NOT NULL, "
                "agent_name TEXT NOT NULL CHECK(agent_name IN ('life','ds','health','code','general')), "
                "conversation_id TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, "
                "PRIMARY KEY (user_id, agent_name))"
            )
            conn.execute(
                "INSERT OR REPLACE INTO agent_sessions (user_id, agent_name, conversation_id, updated_at) "
                "SELECT user_id, "
                "CASE "
                "WHEN agent_name = 'cyber' THEN 'code' "
                "WHEN agent_name = 'code' THEN 'code' "
                "WHEN agent_name = 'general' THEN 'general' "
                "WHEN agent_name IN ('life','ds','health') THEN agent_name "
                "ELSE 'life' "
                "END, "
                "conversation_id, updated_at FROM agent_sessions_old"
            )
            conn.execute("DROP TABLE agent_sessions_old")
    except Exception:
        pass

    conn.commit()


def get_user_profile(conn: sqlite3.Connection, user_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT profile_json FROM user_profiles WHERE user_id=?",
        (user_id,),
    ).fetchone()
    if not row:
        return {}
    raw = row["profile_json"] if hasattr(row, "keys") else row[0]
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def upsert_user_profile(conn: sqlite3.Connection, user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    now = _now_iso()
    payload = json.dumps(profile or {}, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO user_profiles (user_id, profile_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          profile_json=excluded.profile_json,
          updated_at=excluded.updated_at
        """,
        (user_id, payload, now, now),
    )
    conn.commit()
    return profile
