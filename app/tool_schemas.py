# app/tool_schemas.py

memory_search_graph = {
    "type": "function",
    "name": "memory_search_graph",
    "description": "Search imported ChatGPT export memory (graph) and return faithful context windows. Uses FTS candidates and optional embeddings rerank.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "agent": {"type": "string", "enum": ["life", "health", "ds", "code", "general"]},
            "k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            "candidate_limit": {"type": "integer", "default": 250, "minimum": 10, "maximum": 500},
            "context_up": {"type": "integer", "default": 6, "minimum": 0, "maximum": 30},
            "context_down": {"type": "integer", "default": 2, "minimum": 0, "maximum": 30},
            "use_embeddings": {"type": "boolean", "default": True},
        },
        "required": ["query", "agent"],
    },
}

code_record_progress = {
    "type": "function",
    "name": "code_record_progress",
    "description": "Log coding/dev progress for the user (stored locally).",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "notes": {"type": "string"},
            "evidence_path": {"type": "string"},
        },
        "required": ["topic"],
    },
}

code_list_progress = {
    "type": "function",
    "name": "code_list_progress",
    "description": "List recent coding/dev progress entries.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
        },
        "required": [],
    },
}

set_profile = {
    "type": "function",
    "name": "set_profile",
    "description": "Save/update stable user profile facts/preferences/goals locally so they can be injected every turn.",
    "parameters": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "description": "Profile fields to merge (e.g. {\"timezone\":\"Asia/Ulaanbaatar\",\"preferences\":{...}}).",
                "additionalProperties": True,
            },
            "replace": {
                "type": "boolean",
                "default": False,
                "description": "If true, replace the entire stored profile instead of merging.",
            },
            "remove_keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional top-level keys to remove from the stored profile.",
            },
        },
        "required": ["profile"],
    },
}

permissions_get = {
    "type": "function",
    "name": "permissions_get",
    "description": "Return the current permissions/consent gates for this user (read/write mode and enabled capabilities).",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

permissions_set = {
    "type": "function",
    "name": "permissions_set",
    "description": "Update permissions/consent gates for this user (read/write mode plus optional capability flags).",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["read", "write"]},
            "allow_network": {"type": "boolean", "description": "Allow network access for tools like fetch_url/web_search."},
            "allow_fs_write": {"type": "boolean", "description": "Allow writing files to disk (docs export, code apply_patch)."},
            "allow_shell": {"type": "boolean", "description": "Allow running shell commands (run_command, run_tests, git)."},
            "allow_exec": {"type": "boolean", "description": "Allow executing code snippets (run_python/run_r)."},
        },
        "required": ["mode"],
    },
}

tool_policy_set = {
    "type": "function",
    "name": "tool_policy_set",
    "description": "Set allow/deny policy for a specific tool and agent scope, with optional constraints.",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": ["life", "health", "ds", "code", "general", "any"]},
            "tool_name": {"type": "string"},
            "allow": {"type": "boolean"},
            "constraints": {
                "type": "object",
                "description": "Optional constraints (fs_roots, shell_allowlist, network_allowlist).",
                "additionalProperties": True,
            },
        },
        "required": ["agent", "tool_name", "allow"],
    },
}

tool_policy_list = {
    "type": "function",
    "name": "tool_policy_list",
    "description": "List tool policies for the user (optionally filtered by agent).",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": ["life", "health", "ds", "code", "general", "any"]},
            "limit": {"type": "integer", "default": 200, "minimum": 1, "maximum": 500},
        },
        "required": [],
    },
}

log_action = {
    "type": "function",
    "name": "log_action",
    "description": "Write an explicit entry to the audit log (tool, payload, and optional result_id). Tool calls are also logged automatically.",
    "parameters": {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "payload": {"type": "object", "additionalProperties": True},
            "result_id": {"type": "string"},
        },
        "required": ["tool", "payload"],
    },
}

audit_log_list = {
    "type": "function",
    "name": "audit_log_list",
    "description": "List recent audit log entries for the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            "tool": {"type": "string", "description": "Optional tool name filter."},
        },
        "required": [],
    },
}

delegate_agent = {
    "type": "function",
    "name": "delegate_agent",
    "description": "Send a sub-task to a specialized agent (life/health/ds/code) and return its response.",
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": ["life", "health", "ds", "code"]},
            "task": {"type": "string"},
            "include_history": {"type": "boolean", "default": True},
            "history_limit": {"type": "integer", "default": 12, "minimum": 0, "maximum": 50},
        },
        "required": ["agent", "task"],
    },
}

create_reminder = {
    "type": "function",
    "name": "create_reminder",
    "description": "Create a reminder for the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "due_at": {
                "type": "string",
                "description": "ISO 8601 datetime with timezone offset (e.g. 2025-12-31T09:00:00+01:00).",
            },
            "notes": {"type": "string"},
            "rrule": {"type": "string", "description": "Optional recurrence rule (RRULE) string."},
            "channels": {
                "type": "array",
                "items": {"type": "string", "enum": ["push", "email", "sms"]},
                "description": "Optional notification channels (stored; delivery may depend on configuration).",
            },
        },
        "required": ["title", "due_at"],
    },
}

list_reminders = {
    "type": "function",
    "name": "list_reminders",
    "description": "List upcoming scheduled reminders.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
        },
        "required": [],
    },
}

create_event = {
    "type": "function",
    "name": "create_event",
    "description": "Create a calendar event for the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_at": {"type": "string", "description": "ISO 8601 datetime with timezone offset."},
            "end_at": {"type": "string", "description": "ISO 8601 datetime with timezone offset."},
            "location": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["title", "start_at", "end_at"],
    },
}

update_event = {
    "type": "function",
    "name": "update_event",
    "description": "Update an existing calendar event.",
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {"type": "string"},
            "title": {"type": "string"},
            "start_at": {"type": "string"},
            "end_at": {"type": "string"},
            "location": {"type": "string"},
            "notes": {"type": "string"},
            "status": {"type": "string", "enum": ["scheduled", "canceled"]},
        },
        "required": ["event_id"],
    },
}

delete_event = {
    "type": "function",
    "name": "delete_event",
    "description": "Delete (or cancel) a calendar event.",
    "parameters": {
        "type": "object",
        "properties": {"event_id": {"type": "string"}, "hard_delete": {"type": "boolean", "default": False}},
        "required": ["event_id"],
    },
}

list_events = {
    "type": "function",
    "name": "list_events",
    "description": "List calendar events in a time window (defaults to upcoming).",
    "parameters": {
        "type": "object",
        "properties": {
            "start_at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset."},
            "end_at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset."},
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
            "include_canceled": {"type": "boolean", "default": False},
        },
        "required": [],
    },
}

free_busy = {
    "type": "function",
    "name": "free_busy",
    "description": "Return busy time blocks from calendar events for a window.",
    "parameters": {
        "type": "object",
        "properties": {
            "start_at": {"type": "string", "description": "ISO 8601 datetime with timezone offset."},
            "end_at": {"type": "string", "description": "ISO 8601 datetime with timezone offset."},
        },
        "required": ["start_at", "end_at"],
    },
}

create_task = {
    "type": "function",
    "name": "create_task",
    "description": "Create a task with optional due date, priority, and recurrence.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "notes": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1 (highest) to 5 (lowest)."},
            "due_at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset."},
            "rrule": {"type": "string", "description": "Optional recurrence rule (RRULE) string."},
        },
        "required": ["title"],
    },
}

complete_task = {
    "type": "function",
    "name": "complete_task",
    "description": "Mark a task as completed.",
    "parameters": {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
}

list_tasks = {
    "type": "function",
    "name": "list_tasks",
    "description": "List tasks (optionally filtered by status).",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["open", "completed", "all"], "default": "open"},
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
        },
        "required": [],
    },
}

add_contact = {
    "type": "function",
    "name": "add_contact",
    "description": "Add a contact to the local address book.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["name"],
    },
}

get_contact = {
    "type": "function",
    "name": "get_contact",
    "description": "Lookup a contact by id, email, phone, or name (best-effort).",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}},
        "required": ["query"],
    },
}

convert_timezone = {
    "type": "function",
    "name": "convert_timezone",
    "description": "Convert a datetime between timezones.",
    "parameters": {
        "type": "object",
        "properties": {
            "datetime": {"type": "string", "description": "ISO 8601 datetime (with or without offset)."},
            "from_tz": {"type": "string", "description": "IANA timezone name if datetime has no offset."},
            "to_tz": {"type": "string", "description": "IANA timezone name."},
        },
        "required": ["datetime", "to_tz"],
    },
}

estimate_travel_time = {
    "type": "function",
    "name": "estimate_travel_time",
    "description": "Estimate travel time between two locations (best-effort; may require network).",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Address/place or 'lat,lon'."},
            "destination": {"type": "string", "description": "Address/place or 'lat,lon'."},
            "mode": {"type": "string", "enum": ["driving", "walking", "cycling"], "default": "driving"},
        },
        "required": ["origin", "destination"],
    },
}

draft_email = {
    "type": "function",
    "name": "draft_email",
    "description": "Create a structured email draft (stored locally).",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "purpose": {"type": "string", "enum": ["complaint", "booking", "follow_up", "general"], "default": "general"},
            "context": {"type": "string", "description": "Facts to include (order numbers, dates, what happened)."},
            "desired_outcome": {"type": "string", "description": "What you want the recipient to do."},
            "tone": {"type": "string", "enum": ["neutral", "polite", "firm"], "default": "polite"},
        },
        "required": ["to", "subject"],
    },
}

send_email = {
    "type": "function",
    "name": "send_email",
    "description": "Send an email via configured SMTP (requires config).",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "array", "items": {"type": "string"}},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "cc": {"type": "array", "items": {"type": "string"}},
            "bcc": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["to", "subject", "body"],
    },
}

search_email = {
    "type": "function",
    "name": "search_email",
    "description": "Search email via configured IMAP (requires config).",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50}},
        "required": ["query"],
    },
}

create_doc = {
    "type": "function",
    "name": "create_doc",
    "description": "Create a local document (stored in SQLite).",
    "parameters": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "content": {"type": "string"}},
        "required": ["title", "content"],
    },
}

append_doc = {
    "type": "function",
    "name": "append_doc",
    "description": "Append text to an existing local document.",
    "parameters": {
        "type": "object",
        "properties": {"doc_id": {"type": "string"}, "content": {"type": "string"}},
        "required": ["doc_id", "content"],
    },
}

export_pdf = {
    "type": "function",
    "name": "export_pdf",
    "description": "Export a local document to a PDF file (best-effort; requires filesystem write permission).",
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string"},
            "output_path": {"type": "string", "description": "Optional output path; defaults under data/exports/."},
        },
        "required": ["doc_id"],
    },
}

log_expense = {
    "type": "function",
    "name": "log_expense",
    "description": "Log an expense transaction.",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "currency": {"type": "string", "default": "USD"},
            "category": {"type": "string"},
            "merchant": {"type": "string"},
            "notes": {"type": "string"},
            "occurred_at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset."},
        },
        "required": ["amount"],
    },
}

budget_status = {
    "type": "function",
    "name": "budget_status",
    "description": "Summarize spending for a time window (uses logged expenses).",
    "parameters": {
        "type": "object",
        "properties": {
            "window": {"type": "string", "enum": ["week", "month"], "default": "month"},
            "currency": {"type": "string", "default": "USD"},
        },
        "required": [],
    },
}

log_metric = {
    "type": "function",
    "name": "log_metric",
    "description": "Log a health metric (sleep, weight, steps, pain, etc.).",
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "E.g. sleep_hours, weight_kg, steps, pain_0_10."},
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "recorded_at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset."},
            "notes": {"type": "string"},
        },
        "required": ["metric", "value"],
    },
}

get_metric_trend = {
    "type": "function",
    "name": "get_metric_trend",
    "description": "Get a simple trend/summary for a health metric over a window.",
    "parameters": {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "window": {"type": "string", "enum": ["7d", "14d", "30d", "90d"], "default": "30d"},
            "bucket": {"type": "string", "enum": ["day", "week"], "default": "day"},
        },
        "required": ["metric"],
    },
}

med_schedule_add = {
    "type": "function",
    "name": "med_schedule_add",
    "description": "Add a medication schedule (stored locally).",
    "parameters": {
        "type": "object",
        "properties": {
            "medication": {"type": "string"},
            "dose": {"type": "string", "description": "Dose amount as text (e.g. '10', '1 tablet')."},
            "unit": {"type": "string", "description": "Optional unit (e.g. mg)."},
            "times": {"type": "array", "items": {"type": "string"}, "description": "Times of day (HH:MM, local)."},
            "start_date": {"type": "string", "description": "Optional YYYY-MM-DD (local)."},
            "end_date": {"type": "string", "description": "Optional YYYY-MM-DD (local)."},
            "notes": {"type": "string"},
        },
        "required": ["medication", "times"],
    },
}

med_schedule_check = {
    "type": "function",
    "name": "med_schedule_check",
    "description": "Check which scheduled meds are due around a given time (best-effort).",
    "parameters": {
        "type": "object",
        "properties": {
            "at": {"type": "string", "description": "Optional ISO 8601 datetime with timezone offset (defaults to now)."},
            "window_minutes": {"type": "integer", "default": 60, "minimum": 5, "maximum": 720},
        },
        "required": [],
    },
}

med_interaction_check = {
    "type": "function",
    "name": "med_interaction_check",
    "description": "Check interactions between medications (requires a vetted local DB/config).",
    "parameters": {
        "type": "object",
        "properties": {"medications": {"type": "array", "items": {"type": "string"}}},
        "required": ["medications"],
    },
}

create_appointment = {
    "type": "function",
    "name": "create_appointment",
    "description": "Create a health appointment (stored as a calendar event + metadata).",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_at": {"type": "string"},
            "end_at": {"type": "string"},
            "provider": {"type": "string"},
            "location": {"type": "string"},
            "reason": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["title", "start_at", "end_at"],
    },
}

previsit_checklist = {
    "type": "function",
    "name": "previsit_checklist",
    "description": "Generate a pre-visit checklist stub for an appointment (stored as a doc).",
    "parameters": {
        "type": "object",
        "properties": {"appointment_id": {"type": "string"}, "focus": {"type": "string"}},
        "required": ["appointment_id"],
    },
}

log_meal = {
    "type": "function",
    "name": "log_meal",
    "description": "Log a meal entry.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "calories": {"type": "number"},
            "protein_g": {"type": "number"},
            "carbs_g": {"type": "number"},
            "fat_g": {"type": "number"},
            "recorded_at": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["summary"],
    },
}

log_workout = {
    "type": "function",
    "name": "log_workout",
    "description": "Log a workout entry.",
    "parameters": {
        "type": "object",
        "properties": {
            "workout_type": {"type": "string"},
            "duration_min": {"type": "number"},
            "intensity": {"type": "string", "enum": ["low", "moderate", "high"]},
            "calories": {"type": "number"},
            "recorded_at": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["workout_type"],
    },
}

import_health_data = {
    "type": "function",
    "name": "import_health_data",
    "description": "Import health data from a local file (best-effort; format-specific).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "format": {"type": "string", "enum": ["csv", "json", "apple_health_export"], "default": "csv"},
        },
        "required": ["path"],
    },
}

clinical_guideline_search = {
    "type": "function",
    "name": "clinical_guideline_search",
    "description": "Search for clinical guidance (returns cited sources; requires network permission).",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10}},
        "required": ["query"],
    },
}

screening_get_form = {
    "type": "function",
    "name": "screening_get_form",
    "description": "Return a validated questionnaire form (PHQ-9, GAD-7) as structured JSON.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["PHQ-9", "GAD-7"]}},
        "required": ["name"],
    },
}

screening_score = {
    "type": "function",
    "name": "screening_score",
    "description": "Score a completed questionnaire (PHQ-9, GAD-7) from responses.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["PHQ-9", "GAD-7"]},
            "responses": {"type": "array", "items": {"type": "integer"}, "description": "Per-question 0-3 scores in order."},
        },
        "required": ["name", "responses"],
    },
}

escalation_protocol = {
    "type": "function",
    "name": "escalation_protocol",
    "description": "Return red-flag triage guidance (when to seek urgent care).",
    "parameters": {
        "type": "object",
        "properties": {"symptom": {"type": "string"}},
        "required": [],
    },
}

query_sql = {
    "type": "function",
    "name": "query_sql",
    "description": "Run a SQL query against the local SQLite DB (SELECT-only unless write mode is enabled).",
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {"type": "string"},
            "params": {"type": "array", "items": {}},
            "limit": {"type": "integer", "default": 200, "minimum": 1, "maximum": 5000},
        },
        "required": ["sql"],
    },
}

read_table = {
    "type": "function",
    "name": "read_table",
    "description": "Read rows from a local SQLite table (safe wrapper around SELECT).",
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "limit": {"type": "integer", "default": 200, "minimum": 1, "maximum": 5000},
            "where": {"type": "string", "description": "Optional SQL WHERE clause without 'WHERE'."},
        },
        "required": ["table"],
    },
}

write_table = {
    "type": "function",
    "name": "write_table",
    "description": "Insert rows into a local SQLite table (write mode required).",
    "parameters": {
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "rows": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
            "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"},
        },
        "required": ["table", "rows"],
    },
}

upload_file = {
    "type": "function",
    "name": "upload_file",
    "description": "Copy a local file into the assistant's data area (write mode required).",
    "parameters": {
        "type": "object",
        "properties": {"source_path": {"type": "string"}, "dest_name": {"type": "string"}},
        "required": ["source_path"],
    },
}

download_file = {
    "type": "function",
    "name": "download_file",
    "description": "Return metadata for a file path under the assistant data area (read-only; no binary transfer).",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}

run_python = {
    "type": "function",
    "name": "run_python",
    "description": "Execute a short Python snippet and return stdout/stderr (exec permission required).",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "timeout_sec": {"type": "integer", "default": 20, "minimum": 1, "maximum": 120}},
        "required": ["code"],
    },
}

log_run = {
    "type": "function",
    "name": "log_run",
    "description": "Log an experiment/run with params + metrics (stored locally).",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True},
            "metrics": {"type": "object", "additionalProperties": True},
            "notes": {"type": "string"},
        },
        "required": ["name"],
    },
}

get_run = {
    "type": "function",
    "name": "get_run",
    "description": "Get a logged run by id.",
    "parameters": {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"]},
}

search_code = {
    "type": "function",
    "name": "search_code",
    "description": "Search code with ripgrep (requires shell permission).",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500}},
        "required": ["query"],
    },
}

open_file = {
    "type": "function",
    "name": "open_file",
    "description": "Read a file from disk (best-effort; restricted to workspace).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "default": 1, "minimum": 1},
            "end_line": {"type": "integer", "default": 200, "minimum": 1, "maximum": 2000},
        },
        "required": ["path"],
    },
}

apply_patch_tool = {
    "type": "function",
    "name": "apply_patch",
    "description": "Apply a unified diff patch using git apply (filesystem + shell permission required).",
    "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]},
}

run_command = {
    "type": "function",
    "name": "run_command",
    "description": "Run a shell command (shell permission required).",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout_sec": {"type": "integer", "default": 60, "minimum": 1, "maximum": 600},
        },
        "required": ["command"],
    },
}

clone_repo = {
    "type": "function",
    "name": "clone_repo",
    "description": "Clone a git repo into the assistant data area (shell + filesystem permission required).",
    "parameters": {
        "type": "object",
        "properties": {"repo_url": {"type": "string"}, "dest_dir": {"type": "string"}},
        "required": ["repo_url"],
    },
}

fetch_url = {
    "type": "function",
    "name": "fetch_url",
    "description": "Fetch a URL and return the response body (network permission required).",
    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
}

extract_text = {
    "type": "function",
    "name": "extract_text",
    "description": "Extract readable text from HTML content (best-effort).",
    "parameters": {"type": "object", "properties": {"html": {"type": "string"}}, "required": ["html"]},
}

web_search = {
    "type": "function",
    "name": "web_search",
    "description": "Search the web (network permission required; backend may be limited).",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10}}, "required": ["query"]},
}

kb_search = {
    "type": "function",
    "name": "kb_search",
    "description": "Search local documents stored by the assistant.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}}, "required": ["query"]},
}

kb_get_doc = {
    "type": "function",
    "name": "kb_get_doc",
    "description": "Get a local document by id.",
    "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
}

LIFE_TOOLS = [
    permissions_get,
    permissions_set,
    audit_log_list,
    log_action,
    create_event,
    update_event,
    delete_event,
    list_events,
    free_busy,
    create_task,
    complete_task,
    list_tasks,
    create_reminder,
    list_reminders,
    add_contact,
    get_contact,
    convert_timezone,
    estimate_travel_time,
    draft_email,
    send_email,
    search_email,
    create_doc,
    append_doc,
    export_pdf,
    log_expense,
    budget_status,
    memory_search_graph,
    set_profile,
]

HEALTH_TOOLS = [
    permissions_get,
    permissions_set,
    audit_log_list,
    log_action,
    create_appointment,
    previsit_checklist,
    log_metric,
    get_metric_trend,
    med_schedule_add,
    med_schedule_check,
    med_interaction_check,
    log_meal,
    log_workout,
    import_health_data,
    clinical_guideline_search,
    screening_get_form,
    screening_score,
    escalation_protocol,
    create_reminder,
    list_reminders,
    create_event,
    update_event,
    delete_event,
    list_events,
    free_busy,
    memory_search_graph,
    set_profile,
]

DS_TOOLS = [
    {
        "type": "function",
        "name": "ds_create_course",
        "description": "Design a short data science course given the goal and level; returns a JSON plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "level": {"type": "string", "description": "Starting level such as beginner/intermediate/advanced."},
                "constraints": {"type": "string", "description": "Any constraints or preferences such as time or tools."},
            },
            "required": ["goal", "level"],
        },
    },
    {
        "type": "function",
        "name": "ds_start_course",
        "description": "Initialize course progress tracking for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
            },
            "required": ["course_id"],
        },
    },
    {
        "type": "function",
        "name": "ds_next_lesson",
        "description": "Return the next lesson payload for the user to work on.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
            },
            "required": ["course_id"],
        },
    },
    {
        "type": "function",
        "name": "ds_grade_submission",
        "description": "Grade a submission for a given lesson and return score/feedback/next steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_id": {"type": "string"},
                "lesson_key": {"type": "string"},
                "submission": {"type": "string"},
            },
            "required": ["course_id", "lesson_key", "submission"],
        },
    },
    {
        "type": "function",
        "name": "ds_record_progress",
        "description": "Log data science learning progress for the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "score": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
    query_sql,
    read_table,
    write_table,
    upload_file,
    download_file,
    run_python,
    log_run,
    get_run,
    permissions_get,
    permissions_set,
    audit_log_list,
    log_action,
    memory_search_graph,
    set_profile,
]

CODE_TOOLS = [
    permissions_get,
    permissions_set,
    audit_log_list,
    log_action,
    search_code,
    open_file,
    apply_patch_tool,
    run_command,
    clone_repo,
    create_reminder,
    list_reminders,
    code_record_progress,
    code_list_progress,
    memory_search_graph,
    set_profile,
]

GENERAL_TOOLS = [
    permissions_get,
    permissions_set,
    tool_policy_set,
    tool_policy_list,
    audit_log_list,
    log_action,
    delegate_agent,
    web_search,
    fetch_url,
    extract_text,
    kb_search,
    kb_get_doc,
    memory_search_graph,
    set_profile,
]
