from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from app.tools_core import _ensure_tz, _parse_dt, _to_utc_iso

_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _get_token_path() -> Path:
    env_path = (os.environ.get("GOOGLE_CALENDAR_TOKEN_PATH") or "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "google_calendar_token.json"


def _get_client_secrets_path() -> Path:
    env_path = (os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRETS") or "").strip()
    if not env_path:
        raise ValueError(
            "GOOGLE_CALENDAR_CLIENT_SECRETS is required (path to credentials.json)."
        )
    return Path(env_path).expanduser()


def _load_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = _get_token_path()
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        secrets_path = _get_client_secrets_path()
        if not secrets_path.exists():
            raise ValueError(f"Google client secrets not found: {secrets_path}")
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), _SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _get_service():
    from googleapiclient.discovery import build

    creds = _load_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def google_calendar_create_event(
    *,
    title: str,
    due_at: str,
    notes: str | None = None,
    duration_minutes: int | None = None,
    reminder_minutes: int | None = None,
) -> Dict[str, Any]:
    if not title:
        raise ValueError("title is required")
    if not due_at:
        raise ValueError("due_at is required")

    try:
        start_dt = _ensure_tz(_parse_dt(due_at))
    except Exception as e:
        raise ValueError(f"Invalid due_at format: {e}")

    dur = int(duration_minutes or 30)
    if dur <= 0:
        dur = 30
    end_dt = start_dt + timedelta(minutes=dur)

    event: Dict[str, Any] = {
        "summary": title,
        "description": notes or "",
        "start": {"dateTime": start_dt.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
    }

    if reminder_minutes is not None:
        event["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": int(reminder_minutes)}],
        }

    service = _get_service()
    created = service.events().insert(calendarId="primary", body=event).execute()
    return {
        "event_id": created.get("id"),
        "html_link": created.get("htmlLink"),
        "title": title,
        "due_at": due_at,
        "due_at_utc": _to_utc_iso(start_dt),
        "end_at": end_dt.isoformat(),
        "reminder_minutes": reminder_minutes,
    }


def maybe_sync_reminder_to_google_calendar(
    *,
    title: str,
    due_at: str,
    notes: str | None = None,
    reminder_minutes: int | None = None,
) -> Dict[str, Any] | None:
    flag = (os.environ.get("GOOGLE_CALENDAR_AUTO_SYNC") or "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    try:
        return google_calendar_create_event(
            title=title,
            due_at=due_at,
            notes=notes,
            duration_minutes=30,
            reminder_minutes=reminder_minutes,
        )
    except Exception as e:
        return {"error": str(e)}
