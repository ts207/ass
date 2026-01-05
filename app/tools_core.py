import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import Request, urlopen

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE_ROOT / "data"
EXPORTS_DIR = DATA_DIR / "exports"
UPLOADS_DIR = DATA_DIR / "uploads"
REPOS_DIR = DATA_DIR / "repos"
for _d in (DATA_DIR, EXPORTS_DIR, UPLOADS_DIR, REPOS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _parse_dt(dt_str: str) -> datetime:
    s = (dt_str or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _to_utc_iso(dt: datetime) -> str:
    return _ensure_tz(dt).astimezone(timezone.utc).isoformat()


def _truncate_json_str(s: str, max_chars: int = 20_000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n[truncated]"


def _safe_resolve(path: str, *, root: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root.resolve())
    except Exception:
        raise ValueError(f"Path is outside allowed root: {root}")
    return p


def _http_get(url: str, *, max_bytes: int = 400_000, timeout: int = 20) -> Tuple[int, str, Dict[str, str], str]:
    req = Request(url, headers={"User-Agent": "ass-local-assistant/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        final_url = resp.geturl()
        headers = {k.lower(): v for k, v in (resp.headers.items() if resp.headers else [])}
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            data = data[:max_bytes]
        charset = "utf-8"
        ct = headers.get("content-type", "")
        m = re.search(r"charset=([A-Za-z0-9_\-]+)", ct)
        if m:
            charset = m.group(1)
        try:
            text = data.decode(charset, errors="ignore")
        except Exception:
            text = data.decode("utf-8", errors="ignore")
    return status, final_url, headers, text
