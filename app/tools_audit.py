import json
from typing import Any, Dict

from app.tools_core import _id, _now_iso, _truncate_json_str


def audit_log_append(
    conn,
    *,
    user_id: str,
    tool: str,
    payload: Any = None,
    result: Any = None,
    status: str = "ok",
    error: str | None = None,
) -> str:
    aid = _id("audit")
    now = _now_iso()
    payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
    conn.execute(
        "INSERT INTO audit_log (id,user_id,tool,payload_json,result_json,status,error,created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            aid,
            user_id,
            tool,
            _truncate_json_str(payload_json) if payload_json else None,
            _truncate_json_str(result_json) if result_json else None,
            status,
            error,
            now,
        ),
    )
    conn.commit()
    return aid


def log_action(conn, user_id: str, tool: str, payload: Dict[str, Any], result_id: str | None = None) -> Dict[str, Any]:
    aid = audit_log_append(
        conn,
        user_id=user_id,
        tool=tool,
        payload=payload,
        result={"result_id": result_id} if result_id else None,
        status="manual",
        error=None,
    )
    return {"id": aid, "tool": tool, "result_id": result_id}


def audit_log_list(conn, user_id: str, limit: int = 20, tool: str | None = None) -> Dict[str, Any]:
    limit_val = max(1, min(int(limit or 20), 200))
    if tool:
        rows = conn.execute(
            "SELECT id, tool, status, error, created_at FROM audit_log WHERE user_id=? AND tool=? ORDER BY created_at DESC LIMIT ?",
            (user_id, tool, limit_val),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, tool, status, error, created_at FROM audit_log WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit_val),
        ).fetchall()
    return {"entries": [dict(r) for r in rows]}
