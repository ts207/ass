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

        if name == "code_record_progress":
            out = t.code_record_progress(
                conn,
                user_id=user_id,
                topic=args["topic"],
                notes=args.get("notes"),
                evidence_path=args.get("evidence_path"),
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "code_list_progress":
            out = t.code_list_progress(
                conn,
                user_id=user_id,
                limit=int(args.get("limit", 10)),
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "memory_search_graph":
            use_embeddings_val = args.get("use_embeddings", True)
            if isinstance(use_embeddings_val, str):
                use_embeddings_flag = use_embeddings_val.lower() != "false"
            else:
                use_embeddings_flag = bool(use_embeddings_val)
            out = t.memory_search_graph_tool(
                conn,
                query=args["query"],
                agent=args["agent"],
                k=int(args.get("k", 5)),
                candidate_limit=int(args.get("candidate_limit", 100)),
                context_up=int(args.get("context_up", 6)),
                context_down=int(args.get("context_down", 4)),
                use_embeddings=use_embeddings_flag,
            )
            return json.dumps(out, ensure_ascii=False)

        if name == "set_profile":
            profile = args.get("profile")
            if profile is None and isinstance(args.get("updates"), dict):
                profile = args.get("updates")
            replace_flag = bool(args.get("replace", False))
            remove_keys = args.get("remove_keys")
            if remove_keys is not None and not isinstance(remove_keys, list):
                remove_keys = None
            out = t.set_profile_tool(
                conn,
                user_id=user_id,
                profile=profile or {},
                replace=replace_flag,
                remove_keys=remove_keys,
            )
            return json.dumps(out, ensure_ascii=False)

        raise ToolError(f"Unknown tool: {name}")

    except Exception as e:
        # Return an error string; the model can recover.
        return json.dumps({"error": str(e), "tool": name}, ensure_ascii=False)
