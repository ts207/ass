import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.tools_core import (
    DATA_DIR,
    UPLOADS_DIR,
    _id,
    _now_iso,
    _safe_resolve,
    _truncate_json_str,
)
from app.tools_permissions import get_permissions

_SQL_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sql_first_keyword(sql: str) -> str:
    s = (sql or "").lstrip()
    while s.startswith("--"):
        nl = s.find("\n")
        if nl == -1:
            return ""
        s = s[nl + 1 :].lstrip()
    m = re.match(r"(?is)^([a-z_]+)", s)
    return (m.group(1) if m else "").lower()


def query_sql(conn, user_id: str, sql: str, params: List[Any] | None = None, limit: int = 200) -> Dict[str, Any]:
    s = (sql or "").strip()
    if not s:
        raise ValueError("sql is required")
    if s.count(";") > 1 or (";" in s and not s.rstrip().endswith(";")):
        raise ValueError("Only single-statement SQL is supported.")
    kw = _sql_first_keyword(s)
    perms = get_permissions(conn, user_id)
    is_read = kw in ("select", "with")
    if not is_read:
        if perms.get("mode") != "write":
            raise PermissionError("SQL write requires permissions mode='write'.")
        if kw in ("drop", "alter", "attach", "detach", "vacuum", "pragma", "create"):
            raise PermissionError(f"SQL statement '{kw}' is not allowed.")
    cur = conn.execute(s, tuple(params or []))
    if is_read:
        lim = max(1, min(int(limit or 200), 5000))
        rows = cur.fetchmany(lim)
        cols = [d[0] for d in (cur.description or [])]
        out_rows = [dict(zip(cols, r)) for r in rows]
        return {"columns": cols, "rows": out_rows, "returned": len(out_rows)}
    conn.commit()
    return {"rows_affected": int(cur.rowcount or 0)}


def read_table(conn, user_id: str, table: str, limit: int = 200, where: str | None = None) -> Dict[str, Any]:
    if not _SQL_TABLE_RE.match(table or ""):
        raise ValueError("Invalid table name.")
    lim = max(1, min(int(limit or 200), 5000))
    cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = [c["name"] for c in cols_info]
    where_clause = ""
    if where:
        if ";" in where:
            raise ValueError("Invalid where clause.")
        where_clause = f" AND ({where})"
    user_clause = " AND user_id=?" if "user_id" in cols else ""
    sql = f"SELECT * FROM {table} WHERE 1=1{user_clause}{where_clause} LIMIT ?"
    params: list[Any] = []
    if "user_id" in cols:
        params.append(user_id)
    params.append(lim)
    cur = conn.execute(sql, tuple(params))
    rows = cur.fetchall()
    return {"table": table, "rows": [dict(r) for r in rows], "returned": len(rows)}


def write_table(conn, user_id: str, table: str, rows: List[Dict[str, Any]], mode: str = "append") -> Dict[str, Any]:
    if not _SQL_TABLE_RE.match(table or ""):
        raise ValueError("Invalid table name.")
    if table not in ("datasets", "experiments", "ds_runs", "ds_progress"):
        raise PermissionError("write_table is restricted to datasets/experiments/ds_runs/ds_progress.")
    if not rows:
        return {"inserted": 0}
    cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = [c["name"] for c in cols_info]
    if "user_id" in cols:
        cols_no_user = [c for c in cols if c != "user_id"]
    else:
        cols_no_user = cols
    if mode == "replace" and "user_id" in cols:
        conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
    inserted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = {k: row.get(k) for k in cols_no_user if k in row}
        if "user_id" in cols:
            payload["user_id"] = user_id
        if not payload:
            continue
        keys = list(payload.keys())
        placeholders = ",".join(["?"] * len(keys))
        conn.execute(
            f"INSERT INTO {table} ({','.join(keys)}) VALUES ({placeholders})",
            tuple(payload[k] for k in keys),
        )
        inserted += 1
    conn.commit()
    return {"inserted": inserted, "table": table}


def upload_file(source_path: str, dest_name: str | None = None) -> Dict[str, Any]:
    src = Path(source_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise ValueError("source_path must be an existing file")
    name = dest_name or src.name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:120] or "upload.bin"
    dst = (UPLOADS_DIR / f"{_id('up')}_{safe}").resolve()
    shutil.copy2(src, dst)
    return {"stored_path": str(dst), "size_bytes": dst.stat().st_size}


def download_file(path: str) -> Dict[str, Any]:
    p = _safe_resolve(path, root=DATA_DIR)
    if not p.exists() or not p.is_file():
        raise ValueError("File not found under data/")
    return {"path": str(p), "size_bytes": p.stat().st_size}


def run_python(code: str, timeout_sec: int = 20) -> Dict[str, Any]:
    if not code or not isinstance(code, str):
        raise ValueError("code is required")
    timeout = max(1, min(int(timeout_sec or 20), 120))
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    out = _truncate_json_str(out, max_chars=20_000)
    return {"exit_code": proc.returncode, "output": out}


def log_run(conn, user_id: str, name: str, params: Dict[str, Any] | None = None, metrics: Dict[str, Any] | None = None, notes: str | None = None) -> Dict[str, Any]:
    rid = _id("run")
    now = _now_iso()
    conn.execute(
        "INSERT INTO ds_runs (id,user_id,name,params_json,metrics_json,notes,created_at) VALUES (?,?,?,?,?,?,?)",
        (rid, user_id, name, json.dumps(params or {}, ensure_ascii=False), json.dumps(metrics or {}, ensure_ascii=False), notes or "", now),
    )
    conn.commit()
    return {"id": rid, "name": name, "created_at": now}


def get_run(conn, user_id: str, run_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT id,name,params_json,metrics_json,notes,created_at FROM ds_runs WHERE id=? AND user_id=?",
        (run_id, user_id),
    ).fetchone()
    if not row:
        raise ValueError("Run not found for this user.")
    d = dict(row)
    for k in ("params_json", "metrics_json"):
        try:
            d[k.replace("_json", "")] = json.loads(d.get(k) or "{}")
        except Exception:
            d[k.replace("_json", "")] = {}
        d.pop(k, None)
    return d


def ds_create_course(conn, user_id: str, goal: str, level: str, constraints: str | None = None) -> Dict[str, Any]:
    cid = _id("course")
    now = _now_iso()
    lessons: List[Dict[str, Any]] = [
        {
            "key": "lesson-1",
            "title": f"{goal} foundations",
            "content": f"Brief overview of {goal} for {level} learners.",
            "task": "Summarize the key concepts and outline a toy example.",
        },
        {
            "key": "lesson-2",
            "title": f"{goal} hands-on",
            "content": "Work through a small, reproducible exercise. Use synthetic data if none is provided.",
            "task": "Implement the exercise and describe outputs or observations.",
        },
        {
            "key": "lesson-3",
            "title": f"{goal} reflection",
            "content": "Reflect on what worked, what was difficult, and how you would improve the approach.",
            "task": "Write a brief retro and propose a follow-up experiment.",
        },
    ]
    course = {
        "id": cid,
        "goal": goal,
        "level": level,
        "constraints": constraints or "",
        "lessons": lessons,
        "created_at": now,
    }
    conn.execute(
        "INSERT INTO courses (id,user_id,course_json,created_at) VALUES (?,?,?,?)",
        (cid, user_id, json.dumps(course, ensure_ascii=False), now),
    )
    conn.commit()
    return course


def _get_course(conn, user_id: str, course_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT course_json FROM courses WHERE id=? AND user_id=?",
        (course_id, user_id),
    ).fetchone()
    if not row:
        raise ValueError("Course not found for this user.")
    return json.loads(row["course_json"])


def _ensure_progress_rows(conn, user_id: str, course_id: str, lessons: List[Dict[str, Any]]):
    now = _now_iso()
    for lesson in lessons:
        conn.execute(
            "INSERT OR IGNORE INTO course_progress (id,user_id,course_id,lesson_key,status,score,attempts,updated_at) "
            "VALUES (?,?,?,?, 'pending', NULL, 0, ?)",
            (_id("cprog"), user_id, course_id, lesson["key"], now),
        )
    conn.commit()


def _progress_map(conn, user_id: str, course_id: str) -> Dict[str, Dict[str, Any]]:
    rows = conn.execute(
        "SELECT lesson_key,status,score,attempts FROM course_progress WHERE user_id=? AND course_id=?",
        (user_id, course_id),
    ).fetchall()
    return {r["lesson_key"]: dict(r) for r in rows}


def ds_start_course(conn, user_id: str, course_id: str) -> Dict[str, Any]:
    course = _get_course(conn, user_id, course_id)
    _ensure_progress_rows(conn, user_id, course_id, course.get("lessons", []))
    progress = _progress_map(conn, user_id, course_id)
    for lesson in course.get("lessons", []):
        if progress.get(lesson["key"], {}).get("status") != "done":
            conn.execute(
                "UPDATE course_progress SET status='in_progress', updated_at=? WHERE user_id=? AND course_id=? AND lesson_key=?",
                (_now_iso(), user_id, course_id, lesson["key"]),
            )
            conn.commit()
            break
    return {"course_id": course_id, "status": "started"}


def ds_next_lesson(conn, user_id: str, course_id: str) -> Dict[str, Any]:
    course = _get_course(conn, user_id, course_id)
    lessons = course.get("lessons", [])
    if not lessons:
        return {"message": "No lessons found in this course."}

    _ensure_progress_rows(conn, user_id, course_id, lessons)
    progress = _progress_map(conn, user_id, course_id)

    next_lesson = None
    for lesson in lessons:
        status = progress.get(lesson["key"], {}).get("status", "pending")
        if status != "done":
            next_lesson = lesson
            if status != "in_progress":
                conn.execute(
                    "UPDATE course_progress SET status='in_progress', updated_at=? WHERE user_id=? AND course_id=? AND lesson_key=?",
                    (_now_iso(), user_id, course_id, lesson["key"]),
                )
                conn.commit()
                progress = _progress_map(conn, user_id, course_id)
            break

    if not next_lesson:
        return {"message": "Course completed. No remaining lessons."}

    return {
        "course_id": course_id,
        "lesson_key": next_lesson["key"],
        "lesson": next_lesson,
        "progress": progress.get(next_lesson["key"], {}),
    }


def ds_grade_submission(conn, user_id: str, course_id: str, lesson_key: str, submission: str) -> Dict[str, Any]:
    course = _get_course(conn, user_id, course_id)
    lessons = {lesson_item["key"]: lesson_item for lesson_item in course.get("lessons", [])}
    lesson = lessons.get(lesson_key)
    if not lesson:
        raise ValueError("Lesson not found in course.")

    _ensure_progress_rows(conn, user_id, course_id, course.get("lessons", []))
    now = _now_iso()
    submission_length = max(1, len(submission))
    score = round(min(1.0, 0.5 + submission_length / 5000), 2)
    feedback = (
        f"Scored {score:.2f} based on completeness heuristics. "
        "Highlight what you learned and outline one improvement for the next attempt."
    )
    conn.execute(
        "UPDATE course_progress SET status='done', score=?, attempts=attempts+1, updated_at=? "
        "WHERE user_id=? AND course_id=? AND lesson_key=?",
        (score, now, user_id, course_id, lesson_key),
    )
    conn.commit()

    next_payload = ds_next_lesson(conn, user_id, course_id)
    return {
        "lesson_key": lesson_key,
        "score": score,
        "feedback": feedback,
        "next": next_payload,
        "lesson_summary": lesson,
    }


def ds_record_progress(conn, user_id: str, topic: str, score: float | None = None, notes: str | None = None):
    pid = _id("ds")
    now = _now_iso()
    score_val = float(score) if score is not None else None
    conn.execute(
        "INSERT INTO ds_progress (id,user_id,topic,score,notes,created_at) VALUES (?,?,?,?,?,?)",
        (pid, user_id, topic, score_val, notes or "", now),
    )
    conn.commit()
    return {"id": pid, "topic": topic, "score": score_val, "notes": notes}
