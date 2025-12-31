# app/tool_runtime.py

import json
from typing import Any, Dict

from . import tools as t

class ToolError(Exception):
    pass

def call_tool(name: str, args: Dict[str, Any], *, conn, user_id: str) -> str:
    """
    Must return a STRING. JSON string is fine.
    The model will interpret it. :contentReference[oaicite:2]{index=2}
    """
    try:
        if name == "create_reminder":
            out = t.create_reminder(
                conn,
                user_id=user_id,
                title=args["title"],
                due_at=args["due_at"],
                notes=args.get("notes"),
                rrule=args.get("rrule"),
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "list_reminders":
            out = t.list_reminders(conn, user_id=user_id, limit=int(args.get("limit", 10)))
            return json.dumps(out, ensure_ascii=False)

        if name == "ds_create_course":
            out = t.ds_create_course(
                conn,
                user_id=user_id,
                goal=args["goal"],
                level=args["level"],
                constraints=args.get("constraints"),
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "ds_start_course":
            out = t.ds_start_course(conn, user_id=user_id, course_id=args["course_id"])
            return json.dumps(out, ensure_ascii=False)

        if name == "ds_next_lesson":
            out = t.ds_next_lesson(conn, user_id=user_id, course_id=args["course_id"])
            return json.dumps(out, ensure_ascii=False)

        if name == "ds_grade_submission":
            out = t.ds_grade_submission(
                conn,
                user_id=user_id,
                course_id=args["course_id"],
                lesson_key=args["lesson_key"],
                submission=args["submission"],
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "ds_record_progress":
            out = t.ds_record_progress(
                conn,
                user_id=user_id,
                topic=args["topic"],
                score=args.get("score"),
                notes=args.get("notes"),
            )
            return json.dumps(out, ensure_ascii=False)

        raise ToolError(f"Unknown tool: {name}")

    except Exception as e:
        # Return an error string; the model can recover.
        return json.dumps({"error": str(e), "tool": name}, ensure_ascii=False)
