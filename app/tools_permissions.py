import json
from typing import Any, Dict, Optional

from app.tools_core import _now_iso

DEFAULT_PERMISSIONS: Dict[str, Any] = {
    "mode": "write",
    "allow_network": False,
    "allow_fs_write": False,
    "allow_shell": False,
    "allow_exec": False,
}


def get_permissions(conn, user_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT mode, allow_network, allow_fs_write, allow_shell, allow_exec FROM user_permissions WHERE user_id=?",
        (user_id,),
    ).fetchone()
    if not row:
        return dict(DEFAULT_PERMISSIONS)
    return {
        "mode": row["mode"] if row["mode"] in ("read", "write") else DEFAULT_PERMISSIONS["mode"],
        "allow_network": bool(row["allow_network"]),
        "allow_fs_write": bool(row["allow_fs_write"]),
        "allow_shell": bool(row["allow_shell"]),
        "allow_exec": bool(row["allow_exec"]),
    }


def permissions_get(conn, user_id: str) -> Dict[str, Any]:
    return {"user_id": user_id, "permissions": get_permissions(conn, user_id)}


def permissions_set(
    conn,
    user_id: str,
    *,
    mode: str,
    allow_network: Optional[bool] = None,
    allow_fs_write: Optional[bool] = None,
    allow_shell: Optional[bool] = None,
    allow_exec: Optional[bool] = None,
) -> Dict[str, Any]:
    if mode not in ("read", "write"):
        raise ValueError("mode must be 'read' or 'write'")
    existing = get_permissions(conn, user_id)
    updated = {
        "mode": mode,
        "allow_network": existing["allow_network"] if allow_network is None else bool(allow_network),
        "allow_fs_write": existing["allow_fs_write"] if allow_fs_write is None else bool(allow_fs_write),
        "allow_shell": existing["allow_shell"] if allow_shell is None else bool(allow_shell),
        "allow_exec": existing["allow_exec"] if allow_exec is None else bool(allow_exec),
    }
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO user_permissions (user_id, mode, allow_network, allow_fs_write, allow_shell, allow_exec, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          mode=excluded.mode,
          allow_network=excluded.allow_network,
          allow_fs_write=excluded.allow_fs_write,
          allow_shell=excluded.allow_shell,
          allow_exec=excluded.allow_exec,
          updated_at=excluded.updated_at
        """,
        (
            user_id,
            updated["mode"],
            1 if updated["allow_network"] else 0,
            1 if updated["allow_fs_write"] else 0,
            1 if updated["allow_shell"] else 0,
            1 if updated["allow_exec"] else 0,
            now,
        ),
    )
    conn.commit()
    return {"user_id": user_id, "permissions": updated}


def tool_policy_set(
    conn,
    *,
    user_id: str,
    agent: str,
    tool_name: str,
    allow: bool,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent_val = (agent or "").strip().lower()
    if agent_val not in ("life", "health", "ds", "code", "general", "any"):
        raise ValueError("agent must be one of: life, health, ds, code, general, any")
    tool = (tool_name or "").strip()
    if not tool:
        raise ValueError("tool_name is required")
    now = _now_iso()
    constraints_json = json.dumps(constraints, ensure_ascii=False) if constraints else None
    conn.execute(
        """
        INSERT INTO tool_policies (user_id, agent, tool_name, allow, constraints_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, agent, tool_name) DO UPDATE SET
          allow=excluded.allow,
          constraints_json=excluded.constraints_json,
          updated_at=excluded.updated_at
        """,
        (user_id, agent_val, tool, 1 if allow else 0, constraints_json, now),
    )
    conn.commit()
    return {
        "user_id": user_id,
        "agent": agent_val,
        "tool_name": tool,
        "allow": bool(allow),
        "constraints": constraints or {},
    }


def tool_policy_list(
    conn,
    *,
    user_id: str,
    agent: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    lim = max(1, min(int(limit or 200), 500))
    params = [user_id]
    where = "user_id=?"
    if agent:
        where += " AND agent=?"
        params.append(agent.strip().lower())
    rows = conn.execute(
        f"SELECT agent, tool_name, allow, constraints_json, updated_at FROM tool_policies WHERE {where} "
        "ORDER BY updated_at DESC LIMIT ?",
        (*params, lim),
    ).fetchall()
    entries = []
    for r in rows:
        constraints = None
        raw = r["constraints_json"]
        if raw:
            try:
                constraints = json.loads(raw)
            except Exception:
                constraints = None
        entries.append(
            {
                "agent": r["agent"],
                "tool_name": r["tool_name"],
                "allow": bool(r["allow"]),
                "constraints": constraints,
                "updated_at": r["updated_at"],
            }
        )
    return {"entries": entries}


def get_tool_policy(conn, *, user_id: str, agent: str, tool_name: str) -> Optional[Dict[str, Any]]:
    agent_val = (agent or "").strip().lower() or "general"
    tool = (tool_name or "").strip()
    if not tool:
        return None
    row = conn.execute(
        "SELECT allow, constraints_json FROM tool_policies WHERE user_id=? AND agent=? AND tool_name=?",
        (user_id, agent_val, tool),
    ).fetchone()
    if not row and agent_val != "any":
        row = conn.execute(
            "SELECT allow, constraints_json FROM tool_policies WHERE user_id=? AND agent='any' AND tool_name=?",
            (user_id, tool),
        ).fetchone()
    if not row:
        return None
    constraints = None
    raw = row["constraints_json"]
    if raw:
        try:
            constraints = json.loads(raw)
        except Exception:
            constraints = None
    return {
        "allow": bool(row["allow"]),
        "constraints": constraints,
    }
