import json, uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np
from openai import OpenAI

from app.config import EMBEDDING_MODEL

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _parse_dt(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str)


def create_reminder(conn, user_id: str, title: str, due_at: str, notes: str | None = None, rrule: str | None = None):
    rid = _id("rem")
    now = _now_iso()
    try:
        dt = _parse_dt(due_at)
    except Exception as e:
        raise ValueError(f"Invalid due_at format: {e}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    due_at_utc = dt.astimezone(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO reminders (id,user_id,title,due_at,due_at_utc,rrule,notes,status,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?, 'scheduled', ?,?)",
        (rid, user_id, title, due_at, due_at_utc, rrule, notes or "", now, now),
    )
    conn.commit()
    return {"id": rid, "title": title, "due_at": due_at, "due_at_utc": due_at_utc}


def list_reminders(conn, user_id: str, limit: int = 10):
    rows = conn.execute(
        "SELECT id,title,due_at,rrule,notes,status FROM reminders WHERE user_id=? AND status='scheduled' ORDER BY due_at ASC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return {"reminders": [dict(r) for r in rows]}


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
    # Mark the first pending lesson as in_progress
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
    lessons = {l["key"]: l for l in course.get("lessons", [])}
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

    # Recommend the next lesson if any remain
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

from app.db import (
    chatgpt_fts_candidates,
    chatgpt_context_window,
    chatgpt_get_conversation_title,
    chatgpt_get_embeddings,
    get_user_profile,
    upsert_user_profile,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _embed_query(text: str) -> np.ndarray:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    vec = resp.data[0].embedding
    return np.array(vec, dtype=np.float32)


def memory_search_graph_tool(conn, query: str, agent: str, k: int = 5, candidate_limit: int = 100,
                             context_up: int = 6, context_down: int = 4, use_embeddings: bool = True):
    cands = chatgpt_fts_candidates(conn, query=query, agent=agent, limit=candidate_limit)

    # Deduplicate by node_id while preserving order
    seen = set()
    ordered = []
    for c in cands:
        nid = c["node_id"]
        if nid not in seen:
            seen.add(nid)
            ordered.append(c)

    reranked = ordered
    if use_embeddings and ordered:
        emb_map = chatgpt_get_embeddings(conn, [c["node_id"] for c in ordered])
        if emb_map:
            try:
                query_vec = _embed_query(query)
                scored = []
                no_embeds = []
                for idx, c in enumerate(ordered):
                    emb = emb_map.get(c["node_id"])
                    if emb is None or emb.size == 0:
                        no_embeds.append((idx, c))
                        continue
                    sim = _cosine_similarity(query_vec, emb)
                    scored.append((sim, idx, c))

                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    reranked = [c for _, _, c in scored] + [c for _, c in no_embeds]
            except Exception:
                # Fall back to FTS order if embedding call fails
                pass

    results = []
    for c in reranked[:k]:
        nid = c["node_id"]
        cid = c["conversation_id"]
        title = chatgpt_get_conversation_title(conn, cid)
        ctx = chatgpt_context_window(conn, nid, up=context_up, down=context_down)

        context_text = "\n".join(
            f'{m["role"]}: {m["text"]}' for m in ctx
        )

        results.append({
            "node_id": nid,
            "conversation_id": cid,
            "title": title,
            "context": context_text,
        })

    return {"results": results, "returned": len(results)}


def _deep_merge_profile(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    for key, value in (updates or {}).items():
        if (
            key in out
            and isinstance(out.get(key), dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge_profile(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def set_profile_tool(
    conn,
    *,
    user_id: str,
    profile: Dict[str, Any],
    replace: bool = False,
    remove_keys: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Persist stable, user-approved facts/preferences/goals that should be injected every turn.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object/dict")

    existing = get_user_profile(conn, user_id)
    if replace:
        merged = dict(profile)
    else:
        merged = _deep_merge_profile(existing, profile)

    if remove_keys:
        for k in remove_keys:
            if isinstance(k, str):
                merged.pop(k, None)

    upsert_user_profile(conn, user_id, merged)
    return {"user_id": user_id, "profile": merged}
