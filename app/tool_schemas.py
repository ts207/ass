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
            "candidate_limit": {"type": "integer", "default": 100, "minimum": 10, "maximum": 500},
            "context_up": {"type": "integer", "default": 6, "minimum": 0, "maximum": 30},
            "context_down": {"type": "integer", "default": 4, "minimum": 0, "maximum": 30},
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
    memory_search_graph,
    set_profile,
]

HEALTH_TOOLS = [
    LIFE_TOOLS[0],
    LIFE_TOOLS[1],
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
    memory_search_graph,
    set_profile,
]

CODE_TOOLS = [
    LIFE_TOOLS[0],
    LIFE_TOOLS[1],
    code_record_progress,
    code_list_progress,
    memory_search_graph,
    set_profile,
]
