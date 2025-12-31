# app/tool_schemas.py

LIFE_TOOLS = [
    {
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
            },
            "required": ["title", "due_at"],
        },
    },
    {
        "type": "function",
        "name": "list_reminders",
        "description": "List upcoming scheduled reminders.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": [],
        },
    },
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
]
