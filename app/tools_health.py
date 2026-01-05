import csv
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from app.tools_core import DATA_DIR, _ensure_tz, _http_get, _id, _now_iso, _parse_dt, _safe_resolve, _to_utc_iso
from app.tools_life import create_doc, create_event


def log_metric(
    conn,
    user_id: str,
    metric: str,
    value: float,
    unit: str | None = None,
    recorded_at: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    mid = _id("hm")
    now = _now_iso()
    if recorded_at:
        dt = _ensure_tz(_parse_dt(recorded_at))
        rec_utc = _to_utc_iso(dt)
    else:
        recorded_at = now
        rec_utc = now
    conn.execute(
        "INSERT INTO health_metrics (id,user_id,metric,value,unit,recorded_at,recorded_at_utc,notes,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, user_id, metric, float(value), unit or "", recorded_at, rec_utc, notes or "", now),
    )
    conn.commit()
    return {"id": mid, "metric": metric, "value": float(value), "unit": unit or "", "recorded_at": recorded_at, "recorded_at_utc": rec_utc}


def get_metric_trend(conn, user_id: str, metric: str, window: str = "30d", bucket: str = "day") -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    win = (window or "30d").lower()
    days = {"7d": 7, "14d": 14, "30d": 30, "90d": 90}.get(win, 30)
    start = now - timedelta(days=days)
    b = (bucket or "day").lower()
    if b not in ("day", "week"):
        b = "day"
    group_expr = "substr(recorded_at_utc, 1, 10)"

    rows = conn.execute(
        f"""
        SELECT {group_expr} AS day, AVG(value) AS avg_value, MIN(value) AS min_value, MAX(value) AS max_value, COUNT(*) AS n
        FROM health_metrics
        WHERE user_id=? AND metric=? AND recorded_at_utc>=?
        GROUP BY day
        ORDER BY day ASC
        """,
        (user_id, metric, start.isoformat()),
    ).fetchall()
    points = [dict(r) for r in rows]
    if b == "week":
        week_map: Dict[str, Dict[str, Any]] = {}
        for r in points:
            day_str = r.get("day")
            if not day_str:
                continue
            try:
                day_dt = datetime.fromisoformat(day_str)
            except Exception:
                continue
            week_start = (day_dt - timedelta(days=day_dt.weekday())).date().isoformat()
            agg = week_map.get(week_start)
            if not agg:
                agg = {"day": week_start, "avg_value_sum": 0.0, "n": 0, "min_value": None, "max_value": None}
                week_map[week_start] = agg
            n = int(r.get("n") or 0)
            avg_val = float(r.get("avg_value") or 0.0)
            agg["avg_value_sum"] += avg_val * n
            agg["n"] += n
            cur_min = r.get("min_value")
            cur_max = r.get("max_value")
            if cur_min is not None:
                agg["min_value"] = cur_min if agg["min_value"] is None else min(agg["min_value"], cur_min)
            if cur_max is not None:
                agg["max_value"] = cur_max if agg["max_value"] is None else max(agg["max_value"], cur_max)
        points = []
        for key in sorted(week_map.keys()):
            agg = week_map[key]
            n = agg["n"] or 0
            avg_val = (agg["avg_value_sum"] / n) if n else None
            points.append(
                {
                    "day": agg["day"],
                    "avg_value": avg_val,
                    "min_value": agg["min_value"],
                    "max_value": agg["max_value"],
                    "n": n,
                }
            )
    return {"metric": metric, "window": {"start_at_utc": start.isoformat(), "end_at_utc": now.isoformat()}, "points": points}


def med_schedule_add(
    conn,
    user_id: str,
    medication: str,
    times: List[str],
    dose: str | None = None,
    unit: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    if not times:
        raise ValueError("times is required")
    clean_times = []
    for t in times:
        t = (t or "").strip()
        if not re.match(r"^\d{2}:\d{2}$", t):
            raise ValueError("times must be HH:MM (24h)")
        clean_times.append(t)
    sid = _id("med")
    now = _now_iso()
    conn.execute(
        "INSERT INTO medication_schedules (id,user_id,medication,dose,unit,times_json,start_date,end_date,notes,created_at,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (sid, user_id, medication, dose or "", unit or "", json.dumps(clean_times, ensure_ascii=False), start_date, end_date, notes or "", now, now),
    )
    conn.commit()
    return {"id": sid, "medication": medication, "dose": dose or "", "unit": unit or "", "times": clean_times, "start_date": start_date, "end_date": end_date, "notes": notes or ""}


def med_schedule_check(conn, user_id: str, at: str | None = None, window_minutes: int = 60) -> Dict[str, Any]:
    from app.db import get_user_profile

    profile = get_user_profile(conn, user_id)
    tz_name = profile.get("timezone") if isinstance(profile.get("timezone"), str) else None
    if not tz_name:
        tz_name = profile.get("tz") if isinstance(profile.get("tz"), str) else None
    tz = ZoneInfo(tz_name) if tz_name else timezone.utc

    now_utc = datetime.now(timezone.utc)
    at_dt = _ensure_tz(_parse_dt(at)).astimezone(timezone.utc) if at else now_utc
    at_local = at_dt.astimezone(tz)
    local_date = at_local.date().isoformat()

    wmin = max(5, min(int(window_minutes or 60), 720))
    window = timedelta(minutes=wmin)

    rows = conn.execute(
        "SELECT id, medication, dose, unit, times_json, start_date, end_date, notes FROM medication_schedules WHERE user_id=?",
        (user_id,),
    ).fetchall()
    due = []
    for r in rows:
        start_d = r["start_date"] or None
        end_d = r["end_date"] or None
        if start_d and local_date < start_d:
            continue
        if end_d and local_date > end_d:
            continue
        try:
            times = json.loads(r["times_json"] or "[]")
        except Exception:
            times = []
        for tstr in times:
            if not isinstance(tstr, str):
                continue
            hh, mm = tstr.split(":")
            sched_local = datetime(at_local.year, at_local.month, at_local.day, int(hh), int(mm), tzinfo=tz)
            diff = abs(sched_local - at_local)
            if diff <= window:
                due.append(
                    {
                        "schedule_id": r["id"],
                        "medication": r["medication"],
                        "dose": r["dose"],
                        "unit": r["unit"],
                        "scheduled_local": sched_local.isoformat(),
                        "window_minutes": wmin,
                        "notes": r["notes"],
                    }
                )
    return {"at": at_dt.isoformat(), "timezone": str(tz), "due": due}


def med_interaction_check(medications: List[str]) -> Dict[str, Any]:
    db_path = DATA_DIR / "med_interactions.json"
    if not db_path.exists():
        return {
            "configured": False,
            "message": "No vetted interaction DB configured. Add data/med_interactions.json to enable local checks.",
            "medications": medications,
            "interactions": [],
        }
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"configured": False, "error": f"Failed to read med_interactions.json: {e}", "interactions": []}
    meds = [m.strip().lower() for m in (medications or []) if str(m).strip()]
    meds_sorted = sorted(set(meds))
    hits = []
    for i in range(len(meds_sorted)):
        for j in range(i + 1, len(meds_sorted)):
            key = f"{meds_sorted[i]}|{meds_sorted[j]}"
            val = data.get(key) or data.get(f"{meds_sorted[j]}|{meds_sorted[i]}")
            if val:
                hits.append({"pair": [meds_sorted[i], meds_sorted[j]], "info": val})
    return {"configured": True, "medications": medications, "interactions": hits}


def create_appointment(
    conn,
    user_id: str,
    title: str,
    start_at: str,
    end_at: str,
    provider: str | None = None,
    location: str | None = None,
    reason: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    event = create_event(conn, user_id, title, start_at, end_at, location=location, notes=notes)
    appt_id = _id("appt")
    now = _now_iso()
    conn.execute(
        "INSERT INTO appointments (id,user_id,event_id,provider,reason,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
        (appt_id, user_id, event["id"], provider or "", reason or "", now, now),
    )
    conn.commit()
    return {"id": appt_id, "event": event, "provider": provider or "", "reason": reason or ""}


def previsit_checklist(conn, user_id: str, appointment_id: str, focus: str | None = None) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT a.id AS appointment_id, a.provider, a.reason, e.title, e.start_at, e.location, e.notes
        FROM appointments a
        JOIN calendar_events e ON e.id = a.event_id
        WHERE a.user_id=? AND a.id=?
        """,
        (user_id, appointment_id),
    ).fetchone()
    if not row:
        raise ValueError("Appointment not found for this user.")
    title = f"Pre-visit checklist: {row['title']}"
    bullets = [
        "Bring a list of current medications and dosages.",
        "Bring relevant records/lab results if you have them.",
        "Write down your top 3 questions/concerns.",
        "Note symptom timeline (when it started, triggers, what helps).",
        "Bring insurance/ID if applicable.",
    ]
    if focus:
        bullets.insert(0, f"Focus: {focus}")
    content = (
        f"Appointment: {row['title']}\nWhen: {row['start_at']}\nWhere: {row['location']}\nProvider: {row['provider']}\nReason: {row['reason']}\n\n"
        + "\n".join([f"- {b}" for b in bullets])
        + "\n"
    )
    doc = create_doc(conn, user_id, title=title, content=content)
    return {"appointment_id": appointment_id, "doc_id": doc["id"], "title": title}


def log_meal(
    conn,
    user_id: str,
    summary: str,
    calories: float | None = None,
    protein_g: float | None = None,
    carbs_g: float | None = None,
    fat_g: float | None = None,
    recorded_at: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    mid = _id("meal")
    now = _now_iso()
    if recorded_at:
        dt = _ensure_tz(_parse_dt(recorded_at))
        rec_utc = _to_utc_iso(dt)
    else:
        recorded_at = now
        rec_utc = now
    conn.execute(
        "INSERT INTO meals (id,user_id,summary,calories,protein_g,carbs_g,fat_g,recorded_at,recorded_at_utc,notes,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            mid,
            user_id,
            summary,
            float(calories) if calories is not None else None,
            float(protein_g) if protein_g is not None else None,
            float(carbs_g) if carbs_g is not None else None,
            float(fat_g) if fat_g is not None else None,
            recorded_at,
            rec_utc,
            notes or "",
            now,
        ),
    )
    conn.commit()
    return {"id": mid, "summary": summary, "recorded_at": recorded_at, "recorded_at_utc": rec_utc}


def log_workout(
    conn,
    user_id: str,
    workout_type: str,
    duration_min: float | None = None,
    intensity: str | None = None,
    calories: float | None = None,
    recorded_at: str | None = None,
    notes: str | None = None,
) -> Dict[str, Any]:
    wid = _id("workout")
    now = _now_iso()
    if recorded_at:
        dt = _ensure_tz(_parse_dt(recorded_at))
        rec_utc = _to_utc_iso(dt)
    else:
        recorded_at = now
        rec_utc = now
    conn.execute(
        "INSERT INTO workouts (id,user_id,workout_type,duration_min,intensity,calories,recorded_at,recorded_at_utc,notes,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            wid,
            user_id,
            workout_type,
            float(duration_min) if duration_min is not None else None,
            intensity or "",
            float(calories) if calories is not None else None,
            recorded_at,
            rec_utc,
            notes or "",
            now,
        ),
    )
    conn.commit()
    return {"id": wid, "workout_type": workout_type, "recorded_at": recorded_at, "recorded_at_utc": rec_utc}


def import_health_data(conn, user_id: str, path: str, format: str = "csv") -> Dict[str, Any]:
    p = _safe_resolve(path, root=DATA_DIR)
    fmt = (format or "csv").lower()
    if fmt not in ("csv", "json", "apple_health_export"):
        fmt = "csv"
    if not p.exists():
        raise ValueError("File not found.")
    inserted = 0
    if fmt == "csv":
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = (row.get("metric") or "").strip()
                value = row.get("value")
                if not metric or value is None or value == "":
                    continue
                unit = (row.get("unit") or "").strip()
                recorded_at = (row.get("recorded_at") or "").strip() or None
                notes = (row.get("notes") or "").strip() or None
                log_metric(conn, user_id, metric=metric, value=float(value), unit=unit or None, recorded_at=recorded_at, notes=notes)
                inserted += 1
    elif fmt == "json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                metric = str(row.get("metric") or "").strip()
                value = row.get("value")
                if not metric or value is None:
                    continue
                unit = str(row.get("unit") or "").strip()
                recorded_at = row.get("recorded_at")
                notes = row.get("notes")
                log_metric(conn, user_id, metric=metric, value=float(value), unit=unit or None, recorded_at=recorded_at, notes=notes)
                inserted += 1
    else:
        return {"inserted": 0, "message": "apple_health_export import not implemented yet; provide CSV/JSON instead."}
    return {"inserted": inserted, "path": str(p), "format": fmt}


def clinical_guideline_search(query: str, limit: int = 5) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET

    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = max(1, min(int(limit or 5), 10))
    term = f"{q} guideline"
    esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
        {"db": "pubmed", "term": term, "retmode": "xml", "retmax": str(lim)}
    )
    status, _, _, xml1 = _http_get(esearch, max_bytes=400_000)
    if status < 200 or status >= 300:
        raise ValueError(f"PubMed search failed (HTTP {status})")
    root = ET.fromstring(xml1)
    ids = [e.text for e in root.findall(".//IdList/Id") if e.text]
    if not ids:
        return {"query": q, "results": []}
    esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    )
    status2, _, _, xml2 = _http_get(esummary, max_bytes=800_000)
    if status2 < 200 or status2 >= 300:
        raise ValueError(f"PubMed summary failed (HTTP {status2})")
    root2 = ET.fromstring(xml2)
    out = []
    for docsum in root2.findall(".//DocSum"):
        pmid = ""
        title = ""
        source = ""
        pubdate = ""
        for item in docsum.findall("./Item"):
            name = item.attrib.get("Name")
            if name == "Id":
                pmid = item.text or ""
            elif name == "Title":
                title = item.text or ""
            elif name == "Source":
                source = item.text or ""
            elif name == "PubDate":
                pubdate = item.text or ""
        if pmid:
            out.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "source": source,
                    "pubdate": pubdate,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
    return {"query": q, "results": out}


_PHQ9 = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being fidgety or restless",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

_GAD7 = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]


def screening_get_form(name: str) -> Dict[str, Any]:
    n = (name or "").strip().upper()
    if n == "PHQ-9":
        questions = _PHQ9
    elif n == "GAD-7":
        questions = _GAD7
    else:
        raise ValueError("name must be PHQ-9 or GAD-7")
    return {
        "name": n,
        "scale": "0-3 per item (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
        "questions": [{"index": i + 1, "text": q} for i, q in enumerate(questions)],
        "response_options": [0, 1, 2, 3],
    }


def screening_score(name: str, responses: List[int]) -> Dict[str, Any]:
    n = (name or "").strip().upper()
    if n == "PHQ-9":
        expected = 9
        cutoffs = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"), (15, 19, "moderately severe"), (20, 27, "severe")]
    elif n == "GAD-7":
        expected = 7
        cutoffs = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"), (15, 21, "severe")]
    else:
        raise ValueError("name must be PHQ-9 or GAD-7")
    if len(responses or []) != expected:
        raise ValueError(f"Expected {expected} responses.")
    scores = []
    for x in responses:
        xi = int(x)
        if xi < 0 or xi > 3:
            raise ValueError("Each response must be 0..3.")
        scores.append(xi)
    total = sum(scores)
    severity = "unknown"
    for lo, hi, label in cutoffs:
        if lo <= total <= hi:
            severity = label
            break
    return {"name": n, "total": total, "severity": severity}


def escalation_protocol(symptom: str | None = None) -> Dict[str, Any]:
    red_flags = [
        "Severe chest pain, trouble breathing, fainting, or signs of stroke (face drooping, arm weakness, speech difficulty).",
        "Severe allergic reaction (swelling of face/lips/tongue, wheezing, hives with breathing difficulty).",
        "Suicidal thoughts or intent, or thoughts of self-harm.",
        "Severe abdominal pain, uncontrolled bleeding, or severe dehydration.",
        "High fever with stiff neck, confusion, or a new rash that doesn't blanch.",
    ]
    return {"symptom": symptom, "red_flags": red_flags, "advice": "If any red flags apply, seek urgent/emergency care now or call local emergency services."}
