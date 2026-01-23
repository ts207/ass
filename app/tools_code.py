import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

from app.tools_core import REPOS_DIR, WORKSPACE_ROOT, _id, _now_iso, _safe_resolve, _truncate_json_str


def code_record_progress(
    conn,
    user_id: str,
    topic: str,
    notes: str | None = None,
    evidence_path: str | None = None,
) -> Dict[str, Any]:
    pid = _id("code")
    now = _now_iso()
    conn.execute(
        "INSERT INTO code_progress (id,user_id,topic,notes,evidence_path,created_at) VALUES (?,?,?,?,?,?)",
        (pid, user_id, topic, notes or "", evidence_path, now),
    )
    conn.commit()
    return {
        "id": pid,
        "topic": topic,
        "notes": notes or "",
        "evidence_path": evidence_path,
        "created_at": now,
    }


def code_list_progress(conn, user_id: str, limit: int = 10) -> Dict[str, Any]:
    rows = conn.execute(
        "SELECT topic, notes, evidence_path, created_at FROM code_progress WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return {"entries": [dict(r) for r in rows]}


def search_code(query: str, path: str | None = None, limit: int = 50) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = max(1, min(int(limit or 50), 500))
    root = WORKSPACE_ROOT
    if path:
        root = _safe_resolve(path, root=WORKSPACE_ROOT)
    rg = shutil.which("rg")
    results: List[Dict[str, Any]] = []
    if rg:
        cmd = [rg, "--no-heading", "--line-number", "--color", "never", q, str(root)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        for line in (proc.stdout or "").splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            results.append({"path": parts[0], "line": int(parts[1]), "text": parts[2]})
            if len(results) >= lim:
                break
        return {"results": results, "returned": len(results), "exit_code": proc.returncode}

    cmd2 = ["grep", "-R", "-n", q, str(root)]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
    for line in (proc2.stdout or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        results.append({"path": parts[0], "line": int(parts[1]), "text": parts[2]})
        if len(results) >= lim:
            break
    return {"results": results, "returned": len(results), "exit_code": proc2.returncode}


def open_file(path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
    p = _safe_resolve(path, root=WORKSPACE_ROOT)
    if not p.exists() or not p.is_file():
        raise ValueError("File not found.")
    start = max(1, int(start_line or 1))
    end = max(start, min(int(end_line or 200), start + 2000))
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    snippet = "\n".join(lines[start - 1 : end])
    return {"path": str(p), "start_line": start, "end_line": end, "text": snippet}


def list_files(path: str, glob: str | None = None, limit: int = 200) -> Dict[str, Any]:
    if not path or not isinstance(path, str):
        raise ValueError("path is required")
    lim = max(1, min(int(limit or 200), 1000))
    root = _safe_resolve(path, root=WORKSPACE_ROOT)
    if not root.exists():
        raise ValueError("Path not found.")
    entries: List[Dict[str, Any]] = []
    if glob:
        candidates = root.glob(glob)
    else:
        candidates = root.iterdir()
    for p in candidates:
        try:
            stat = p.stat()
        except Exception:
            stat = None
        entries.append(
            {
                "path": str(p),
                "is_dir": p.is_dir(),
                "size": int(stat.st_size) if stat and p.is_file() else None,
            }
        )
        if len(entries) >= lim:
            break
    return {"path": str(root), "glob": glob, "returned": len(entries), "entries": entries}


def apply_patch(patch: str) -> Dict[str, Any]:
    if not patch or not isinstance(patch, str):
        raise ValueError("patch is required")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".diff") as f:
        f.write(patch)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", tmp],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = ((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")).strip()
        return {"applied": proc.returncode == 0, "exit_code": proc.returncode, "output": _truncate_json_str(out, 10_000)}
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def run_command(command: str, cwd: str | None = None, timeout_sec: int = 60) -> Dict[str, Any]:
    if not command or not isinstance(command, str):
        raise ValueError("command is required")
    timeout = max(1, min(int(timeout_sec or 60), 600))
    workdir = WORKSPACE_ROOT
    if cwd:
        workdir = _safe_resolve(cwd, root=WORKSPACE_ROOT)
    proc = subprocess.run(
        command,
        cwd=str(workdir),
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return {"exit_code": proc.returncode, "cwd": str(workdir), "output": _truncate_json_str(out, 20_000)}


def clone_repo(repo_url: str, dest_dir: str | None = None) -> Dict[str, Any]:
    url = (repo_url or "").strip()
    if not url:
        raise ValueError("repo_url is required")
    safe_name = dest_dir or re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("/")[-1]).strip("_")
    safe_name = safe_name[:80] or _id("repo")
    dest = (REPOS_DIR / safe_name).resolve()
    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        raise ValueError(f"git clone failed: {out.strip()}")
    return {"cloned": True, "repo_url": url, "path": str(dest)}
