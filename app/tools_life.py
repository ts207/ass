import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from app.tools_core import (
    EXPORTS_DIR,
    WORKSPACE_ROOT,
    _ensure_tz,
    _http_get,
    _id,
    _now_iso,
    _parse_dt,
    _safe_resolve,
    _to_utc_iso,
)


def create_reminder(
    conn,
    user_id: str,
    title: str,
    due_at: str,
    notes: str | None = None,
    rrule: str | None = None,
    channels: List[str] | None = None,
):
    rid = _id("rem")
    now = _now_iso()
    try:
        dt = _parse_dt(due_at)
    except Exception as e:
        raise ValueError(f"Invalid due_at format: {e}")
    dt = _ensure_tz(dt)
    due_at_utc = _to_utc_iso(dt)
    channels_json = json.dumps(channels or [], ensure_ascii=False)
    conn.execute(
        "INSERT INTO reminders (id,user_id,title,due_at,due_at_utc,rrule,channels_json,notes,status,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?, 'scheduled', ?,?)",
        (rid, user_id, title, due_at, due_at_utc, rrule, channels_json, notes or "", now, now),
    )
    conn.commit()
    return {
        "id": rid,
        "title": title,
        "due_at": due_at,
        "due_at_utc": due_at_utc,
        "channels": channels or [],
    }


def list_reminders(conn, user_id: str, limit: int = 10):
    rows = conn.execute(
        "SELECT id,title,due_at,rrule,channels_json,notes,status FROM reminders WHERE user_id=? AND status='scheduled' ORDER BY due_at ASC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["channels"] = json.loads(d.get("channels_json") or "[]")
        except Exception:
            d["channels"] = []
        d.pop("channels_json", None)
        out.append(d)
    return {"reminders": out}


def create_event(
    conn,
    user_id: str,
    title: str,
    start_at: str,
    end_at: str,
    location: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    eid = _id("evt")
    now = _now_iso()
    start_dt = _ensure_tz(_parse_dt(start_at))
    end_dt = _ensure_tz(_parse_dt(end_at))
    if end_dt <= start_dt:
        raise ValueError("end_at must be after start_at")
    conn.execute(
        "INSERT INTO calendar_events (id,user_id,title,start_at,start_at_utc,end_at,end_at_utc,location,notes,status,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,'scheduled',?,?)",
        (
            eid,
            user_id,
            title,
            start_at,
            _to_utc_iso(start_dt),
            end_at,
            _to_utc_iso(end_dt),
            location or "",
            notes or "",
            now,
            now,
        ),
    )
    conn.commit()
    return {
        "id": eid,
        "title": title,
        "start_at": start_at,
        "start_at_utc": _to_utc_iso(start_dt),
        "end_at": end_at,
        "end_at_utc": _to_utc_iso(end_dt),
        "location": location or "",
        "notes": notes or "",
        "status": "scheduled",
    }


def update_event(
    conn,
    user_id: str,
    event_id: str,
    title: str | None = None,
    start_at: str | None = None,
    end_at: str | None = None,
    location: str | None = None,
    notes: str | None = None,
    status: str | None = None,
) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT id,title,start_at,end_at,location,notes,status FROM calendar_events WHERE id=? AND user_id=?",
        (event_id, user_id),
    ).fetchone()
    if not row:
        raise ValueError("Event not found for this user.")
    cur = dict(row)
    new_title = title if title is not None else cur["title"]
    new_start = start_at if start_at is not None else cur["start_at"]
    new_end = end_at if end_at is not None else cur["end_at"]
    start_dt = _ensure_tz(_parse_dt(new_start))
    end_dt = _ensure_tz(_parse_dt(new_end))
    if end_dt <= start_dt:
        raise ValueError("end_at must be after start_at")
    new_location = location if location is not None else cur.get("location", "")
    new_notes = notes if notes is not None else cur.get("notes", "")
    new_status = status if status is not None else cur.get("status", "scheduled")
    if new_status not in ("scheduled", "canceled"):
        raise ValueError("status must be 'scheduled' or 'canceled'")
    now = _now_iso()
    conn.execute(
        "UPDATE calendar_events SET title=?, start_at=?, start_at_utc=?, end_at=?, end_at_utc=?, location=?, notes=?, status=?, updated_at=? "
        "WHERE id=? AND user_id=?",
        (
            new_title,
            new_start,
            _to_utc_iso(start_dt),
            new_end,
            _to_utc_iso(end_dt),
            new_location,
            new_notes,
            new_status,
            now,
            event_id,
            user_id,
        ),
    )
    conn.commit()
    return {
        "id": event_id,
        "title": new_title,
        "start_at": new_start,
        "start_at_utc": _to_utc_iso(start_dt),
        "end_at": new_end,
        "end_at_utc": _to_utc_iso(end_dt),
        "location": new_location,
        "notes": new_notes,
        "status": new_status,
        "updated_at": now,
    }


def delete_event(conn, user_id: str, event_id: str, hard_delete: bool = False) -> Dict[str, Any]:
    if hard_delete:
        cur = conn.execute(
            "DELETE FROM calendar_events WHERE id=? AND user_id=?",
            (event_id, user_id),
        )
        conn.commit()
        return {"id": event_id, "deleted": bool(cur.rowcount)}
    out = update_event(conn, user_id, event_id, status="canceled")
    return {"id": event_id, "canceled": True, "event": out}


def list_events(
    conn,
    user_id: str,
    start_at: str | None = None,
    end_at: str | None = None,
    limit: int = 20,
    include_canceled: bool = False,
) -> Dict[str, Any]:
    limit_val = max(1, min(int(limit or 20), 200))
    now = datetime.now(timezone.utc)
    if start_at:
        start_dt = _ensure_tz(_parse_dt(start_at)).astimezone(timezone.utc)
    else:
        start_dt = now
    if end_at:
        end_dt = _ensure_tz(_parse_dt(end_at)).astimezone(timezone.utc)
    else:
        end_dt = start_dt + timedelta(days=30)
    if end_dt <= start_dt:
        raise ValueError("end_at must be after start_at")

    where = "user_id=? AND end_at_utc>=? AND start_at_utc<=?"
    params: list[Any] = [user_id, start_dt.isoformat(), end_dt.isoformat()]
    if not include_canceled:
        where += " AND status!='canceled'"
    rows = conn.execute(
        f"SELECT id,title,start_at,end_at,location,notes,status FROM calendar_events WHERE {where} ORDER BY start_at_utc ASC LIMIT ?",
        (*params, limit_val),
    ).fetchall()
    return {"events": [dict(r) for r in rows], "window": {"start_at_utc": start_dt.isoformat(), "end_at_utc": end_dt.isoformat()}}


def free_busy(conn, user_id: str, start_at: str, end_at: str) -> Dict[str, Any]:
    start_dt = _ensure_tz(_parse_dt(start_at)).astimezone(timezone.utc)
    end_dt = _ensure_tz(_parse_dt(end_at)).astimezone(timezone.utc)
    if end_dt <= start_dt:
        raise ValueError("end_at must be after start_at")
    rows = conn.execute(
        """
        SELECT id, title, start_at_utc, end_at_utc
        FROM calendar_events
        WHERE user_id=?
          AND status!='canceled'
          AND end_at_utc>=?
          AND start_at_utc<=?
        ORDER BY start_at_utc ASC
        """,
        (user_id, start_dt.isoformat(), end_dt.isoformat()),
    ).fetchall()
    blocks = []
    for r in rows:
        blocks.append({"event_id": r["id"], "title": r["title"], "start_at_utc": r["start_at_utc"], "end_at_utc": r["end_at_utc"]})
    return {"busy": blocks, "window": {"start_at_utc": start_dt.isoformat(), "end_at_utc": end_dt.isoformat()}}


def create_task(
    conn,
    user_id: str,
    title: str,
    notes: str | None = None,
    priority: int | None = None,
    due_at: str | None = None,
    rrule: str | None = None,
) -> Dict[str, Any]:
    tid = _id("task")
    now = _now_iso()
    due_at_utc = None
    if due_at:
        due_dt = _ensure_tz(_parse_dt(due_at))
        due_at_utc = _to_utc_iso(due_dt)
    pr = int(priority) if priority is not None else None
    if pr is not None and (pr < 1 or pr > 5):
        raise ValueError("priority must be 1..5")
    conn.execute(
        "INSERT INTO tasks (id,user_id,title,notes,priority,due_at,due_at_utc,rrule,status,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?, 'open', ?,?)",
        (tid, user_id, title, notes or "", pr, due_at, due_at_utc, rrule, now, now),
    )
    conn.commit()
    return {"id": tid, "title": title, "notes": notes or "", "priority": pr, "due_at": due_at, "due_at_utc": due_at_utc, "rrule": rrule, "status": "open"}


def complete_task(conn, user_id: str, task_id: str) -> Dict[str, Any]:
    now = _now_iso()
    cur = conn.execute(
        "UPDATE tasks SET status='completed', completed_at=?, updated_at=? WHERE id=? AND user_id=? AND status!='completed'",
        (now, now, task_id, user_id),
    )
    conn.commit()
    return {"id": task_id, "completed": bool(cur.rowcount), "completed_at": now}


def list_tasks(conn, user_id: str, status: str = "open", limit: int = 20) -> Dict[str, Any]:
    limit_val = max(1, min(int(limit or 20), 200))
    st = (status or "open").lower()
    if st not in ("open", "completed", "all"):
        raise ValueError("status must be open|completed|all")
    where = "user_id=?"
    params: list[Any] = [user_id]
    if st != "all":
        where += " AND status=?"
        params.append(st)
    rows = conn.execute(
        f"SELECT id,title,notes,priority,due_at,rrule,status,created_at,completed_at FROM tasks WHERE {where} ORDER BY COALESCE(due_at_utc, created_at) ASC LIMIT ?",
        (*params, limit_val),
    ).fetchall()
    return {"tasks": [dict(r) for r in rows]}


def add_contact(conn, user_id: str, name: str, email: str | None = None, phone: str | None = None, notes: str | None = None) -> Dict[str, Any]:
    cid = _id("contact")
    now = _now_iso()
    conn.execute(
        "INSERT INTO contacts (id,user_id,name,email,phone,notes,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (cid, user_id, name, (email or "").strip(), (phone or "").strip(), notes or "", now, now),
    )
    conn.commit()
    return {"id": cid, "name": name, "email": (email or "").strip(), "phone": (phone or "").strip(), "notes": notes or ""}


def get_contact(conn, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    limit_val = max(1, min(int(limit or 5), 20))
    row = conn.execute(
        "SELECT id,name,email,phone,notes,updated_at FROM contacts WHERE user_id=? AND id=?",
        (user_id, q),
    ).fetchone()
    if row:
        return {"matches": [dict(row)]}
    like = f"%{q}%"
    rows = conn.execute(
        "SELECT id,name,email,phone,notes,updated_at FROM contacts "
        "WHERE user_id=? AND (name LIKE ? OR email LIKE ? OR phone LIKE ?) "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, like, like, like, limit_val),
    ).fetchall()
    return {"matches": [dict(r) for r in rows]}


def convert_timezone(dt: str, to_tz: str, from_tz: str | None = None) -> Dict[str, Any]:
    dt_parsed = _parse_dt(dt)
    if dt_parsed.tzinfo is None:
        if from_tz:
            dt_parsed = dt_parsed.replace(tzinfo=ZoneInfo(from_tz))
        else:
            dt_parsed = dt_parsed.replace(tzinfo=timezone.utc)
    out_dt = dt_parsed.astimezone(ZoneInfo(to_tz))
    return {"input": dt, "from_tz": from_tz or str(dt_parsed.tzinfo), "to_tz": to_tz, "output": out_dt.isoformat()}


def _parse_latlon(s: str) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _geocode_nominatim(q: str) -> Tuple[float, float, str]:
    params = {"format": "json", "q": q, "limit": 1}
    url = "https://nominatim.openstreetmap.org/search?" + urlencode(params)
    status, _, _, text = _http_get(url, max_bytes=200_000)
    if status < 200 or status >= 300:
        raise ValueError(f"Geocoding failed (HTTP {status})")
    try:
        data = json.loads(text or "[]")
    except Exception as e:
        raise ValueError(f"Geocoding parse failed: {e}")
    if not data:
        raise ValueError("Geocoding returned no results.")
    item = data[0]
    return float(item["lat"]), float(item["lon"]), str(item.get("display_name") or q)


def estimate_travel_time(origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
    mode_val = (mode or "driving").lower()
    if mode_val not in ("driving", "walking", "cycling"):
        raise ValueError("mode must be driving|walking|cycling")
    o_ll = _parse_latlon(origin)
    d_ll = _parse_latlon(destination)
    o_name = origin
    d_name = destination
    if o_ll is None:
        lat, lon, disp = _geocode_nominatim(origin)
        o_ll = (lat, lon)
        o_name = disp
    if d_ll is None:
        lat, lon, disp = _geocode_nominatim(destination)
        d_ll = (lat, lon)
        d_name = disp
    (olat, olon) = o_ll
    (dlat, dlon) = d_ll
    url = f"https://router.project-osrm.org/route/v1/{mode_val}/{olon:.6f},{olat:.6f};{dlon:.6f},{dlat:.6f}?overview=false"
    status, _, _, text = _http_get(url, max_bytes=400_000)
    if status < 200 or status >= 300:
        raise ValueError(f"Routing failed (HTTP {status})")
    data = json.loads(text or "{}")
    routes = data.get("routes") or []
    if not routes:
        raise ValueError("No route found.")
    r0 = routes[0]
    dur_s = float(r0.get("duration") or 0.0)
    dist_m = float(r0.get("distance") or 0.0)
    return {
        "origin": {"query": origin, "resolved": o_name, "lat": olat, "lon": olon},
        "destination": {"query": destination, "resolved": d_name, "lat": dlat, "lon": dlon},
        "mode": mode_val,
        "duration_sec": dur_s,
        "distance_m": dist_m,
        "duration_min": round(dur_s / 60.0, 1),
        "distance_km": round(dist_m / 1000.0, 2),
    }


def draft_email(
    conn,
    user_id: str,
    to: List[str],
    subject: str,
    purpose: str = "general",
    context: str | None = None,
    desired_outcome: str | None = None,
    tone: str = "polite",
) -> Dict[str, Any]:
    purpose_val = (purpose or "general").strip()
    if purpose_val not in ("complaint", "booking", "follow_up", "general"):
        purpose_val = "general"
    tone_val = (tone or "polite").strip()
    if tone_val not in ("neutral", "polite", "firm"):
        tone_val = "polite"
    greeting = "Hello,"
    closing = "Best regards,"
    if tone_val == "firm":
        closing = "Sincerely,"
    body_parts = [greeting, ""]
    if context:
        body_parts.append(context.strip())
        body_parts.append("")
    if desired_outcome:
        body_parts.append(f"Requested outcome: {desired_outcome.strip()}")
        body_parts.append("")
    if purpose_val == "complaint":
        body_parts.append("Please let me know how you can resolve this and the timeline.")
    elif purpose_val == "booking":
        body_parts.append("Could you confirm availability and the next steps to book?")
    elif purpose_val == "follow_up":
        body_parts.append("Following up on the above—could you share an update when you can?")
    else:
        body_parts.append("Thank you.")
    body_parts.extend(["", closing, ""])
    body = "\n".join(body_parts).strip() + "\n"

    did = _id("draft")
    now = _now_iso()
    conn.execute(
        "INSERT INTO email_drafts (id,user_id,to_json,subject,body,status,created_at,updated_at) VALUES (?,?,?,?,?,'draft',?,?)",
        (did, user_id, json.dumps(to or [], ensure_ascii=False), subject, body, now, now),
    )
    conn.commit()
    return {"id": did, "to": to or [], "subject": subject, "body": body, "purpose": purpose_val, "tone": tone_val, "status": "draft"}


def send_email(to: List[str], subject: str, body: str, cc: List[str] | None = None, bcc: List[str] | None = None) -> Dict[str, Any]:
    import smtplib
    from email.message import EmailMessage

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM") or user
    if not host or not from_addr:
        raise ValueError("SMTP is not configured (set SMTP_HOST/SMTP_USER/SMTP_PASS/SMTP_FROM).")

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to or [])
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    msg.set_content(body or "")

    recipients = (to or []) + (cc or []) + (bcc or [])
    if not recipients:
        raise ValueError("No recipients provided.")

    use_tls = os.getenv("SMTP_TLS", "true").lower() != "false"
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        if user and password:
            server.login(user, password)
        server.send_message(msg, from_addr=from_addr, to_addrs=recipients)
    return {"sent": True, "to": to, "cc": cc or [], "bcc": bcc or [], "subject": subject}


def search_email(query: str, limit: int = 10) -> Dict[str, Any]:
    import imaplib
    import email

    host = os.getenv("IMAP_HOST")
    user = os.getenv("IMAP_USER")
    password = os.getenv("IMAP_PASS")
    mailbox = os.getenv("IMAP_MAILBOX", "INBOX")
    if not host or not user or not password:
        raise ValueError("IMAP is not configured (set IMAP_HOST/IMAP_USER/IMAP_PASS).")
    limit_val = max(1, min(int(limit or 10), 50))

    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")

    def _search_key(qs: str) -> str:
        if "@" in qs:
            return f'(FROM "{qs}")'
        return f'(TEXT "{qs}")'

    with imaplib.IMAP4_SSL(host) as M:
        M.login(user, password)
        M.select(mailbox)
        typ, data = M.search(None, _search_key(q))
        if typ != "OK":
            return {"matches": []}
        ids = (data[0] or b"").split()
        ids = list(reversed(ids))[:limit_val]
        out = []
        for mid in ids:
            typ2, msg_data = M.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
            if typ2 != "OK" or not msg_data:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            out.append(
                {
                    "id": mid.decode("utf-8", errors="ignore"),
                    "subject": str(msg.get("Subject") or ""),
                    "from": str(msg.get("From") or ""),
                    "date": str(msg.get("Date") or ""),
                }
            )
    return {"matches": out}


def create_doc(conn, user_id: str, title: str, content: str) -> Dict[str, Any]:
    did = _id("doc")
    now = _now_iso()
    conn.execute(
        "INSERT INTO documents (id,user_id,title,content,created_at,updated_at) VALUES (?,?,?,?,?,?)",
        (did, user_id, title, content or "", now, now),
    )
    conn.commit()
    return {"id": did, "title": title, "updated_at": now}


def append_doc(conn, user_id: str, doc_id: str, content: str) -> Dict[str, Any]:
    now = _now_iso()
    cur = conn.execute(
        "UPDATE documents SET content=content || ?, updated_at=? WHERE id=? AND user_id=?",
        ("\n\n" + (content or ""), now, doc_id, user_id),
    )
    conn.commit()
    if not cur.rowcount:
        raise ValueError("Document not found for this user.")
    return {"id": doc_id, "appended": True, "updated_at": now}


def _pdf_escape_text(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", "")
    )


def _write_simple_pdf(text: str, out_path: Path) -> None:
    lines = []
    for raw in (text or "").splitlines():
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        while len(raw) > 110:
            lines.append(raw[:110])
            raw = raw[110:]
        lines.append(raw)

    max_lines_per_page = 48
    pages = [lines[i : i + max_lines_per_page] for i in range(0, max(1, len(lines)), max_lines_per_page)]
    if not pages:
        pages = [[""]]

    objects: Dict[int, bytes] = {}
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    page_nums: List[int] = []
    next_obj = 4
    for page_lines in pages:
        page_num = next_obj
        content_num = next_obj + 1
        next_obj += 2
        page_nums.append(page_num)
        x = 72
        y = 750
        leading = 14
        parts = ["BT", "/F1 12 Tf", f"{x} {y} Td"]
        for idx, line in enumerate(page_lines):
            if idx > 0:
                parts.append(f"0 -{leading} Td")
            parts.append(f"({_pdf_escape_text(line)}) Tj")
        parts.append("ET")
        stream_data = ("\n".join(parts) + "\n").encode("utf-8")
        objects[content_num] = b"<< /Length %d >>\nstream\n%bendstream" % (len(stream_data), stream_data)
        objects[page_num] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> "
            + (b"/Contents %d 0 R >>" % content_num)
        )

    kids = " ".join([f"{n} 0 R" for n in page_nums]).encode("utf-8")
    objects[2] = b"<< /Type /Pages /Kids [ %b ] /Count %d >>" % (kids, len(page_nums))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    max_obj = max(objects.keys())
    out = bytearray()
    out.extend(b"%PDF-1.4\n")
    offsets = [0] * (max_obj + 1)
    for obj_num in range(1, max_obj + 1):
        offsets[obj_num] = len(out)
        body = objects[obj_num]
        out.extend(f"{obj_num} 0 obj\n".encode("utf-8"))
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_offset = len(out)
    out.extend(f"xref\n0 {max_obj + 1}\n".encode("utf-8"))
    out.extend(b"0000000000 65535 f \n")
    for obj_num in range(1, max_obj + 1):
        out.extend(f"{offsets[obj_num]:010d} 00000 n \n".encode("utf-8"))
    out.extend(
        (
            f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))


def export_pdf(conn, user_id: str, doc_id: str, output_path: str | None = None) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT title, content FROM documents WHERE id=? AND user_id=?",
        (doc_id, user_id),
    ).fetchone()
    if not row:
        raise ValueError("Document not found for this user.")
    title = row["title"] or doc_id
    content = row["content"] or ""
    if output_path:
        outp = _safe_resolve(output_path, root=WORKSPACE_ROOT)
    else:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:60] or doc_id
        outp = (EXPORTS_DIR / f"{safe_name}_{doc_id}.pdf").resolve()
    _write_simple_pdf(f"{title}\n\n{content}".strip(), outp)
    return {"doc_id": doc_id, "output_path": str(outp)}


def log_expense(
    conn,
    user_id: str,
    amount: float,
    currency: str = "USD",
    category: str | None = None,
    merchant: str | None = None,
    notes: str | None = None,
    occurred_at: str | None = None,
) -> Dict[str, Any]:
    eid = _id("exp")
    now = _now_iso()
    if occurred_at:
        dt = _ensure_tz(_parse_dt(occurred_at))
        occ_utc = _to_utc_iso(dt)
    else:
        occurred_at = now
        occ_utc = now
    conn.execute(
        "INSERT INTO expenses (id,user_id,amount,currency,category,merchant,notes,occurred_at,occurred_at_utc,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (eid, user_id, float(amount), (currency or "USD").upper(), category or "", merchant or "", notes or "", occurred_at, occ_utc, now),
    )
    conn.commit()
    return {"id": eid, "amount": float(amount), "currency": (currency or "USD").upper(), "category": category or "", "merchant": merchant or "", "occurred_at": occurred_at, "occurred_at_utc": occ_utc}


def budget_status(conn, user_id: str, window: str = "month", currency: str = "USD") -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    win = (window or "month").lower()
    if win == "week":
        start = now - timedelta(days=7)
    else:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    cur = (currency or "USD").upper()
    rows = conn.execute(
        """
        SELECT category, SUM(amount) AS total
        FROM expenses
        WHERE user_id=? AND currency=? AND occurred_at_utc>=?
        GROUP BY category
        ORDER BY total DESC
        """,
        (user_id, cur, start.isoformat()),
    ).fetchall()
    total = sum(float(r["total"] or 0.0) for r in rows)
    return {
        "currency": cur,
        "window": {"start_at_utc": start.isoformat(), "end_at_utc": now.isoformat()},
        "total": total,
        "by_category": [{"category": r["category"] or "", "total": float(r["total"] or 0.0)} for r in rows],
    }
