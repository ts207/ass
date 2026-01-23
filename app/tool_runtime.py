# app/tool_runtime.py

import json
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from . import tools as t
from .tool_registry import get_tool_meta
from .tools_permissions import get_tool_policy

class ToolError(Exception):
    pass


def _normalize_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _path_within_roots(path: str, roots: list[str]) -> bool:
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return False
    for root in roots:
        try:
            r = Path(root).expanduser().resolve()
        except Exception:
            continue
        try:
            p.relative_to(r)
            return True
        except Exception:
            continue
    return False


def _enforce_constraints(name: str, args: Dict[str, Any], constraints: Dict[str, Any]) -> None:
    if not constraints:
        return

    fs_roots = constraints.get("fs_roots") or constraints.get("fs_allowlist")
    if isinstance(fs_roots, list) and fs_roots:
        for key in ("path", "output_path", "dest_dir", "source_path", "cwd"):
            raw = args.get(key)
            if not raw or not isinstance(raw, str):
                continue
            if not _path_within_roots(raw, fs_roots):
                raise ToolError(f"Permission denied: path '{raw}' is outside allowed roots.")

    shell_allow = constraints.get("shell_allowlist") or constraints.get("shell_allow")
    shell_deny = constraints.get("shell_denylist") or constraints.get("shell_deny")
    cmd = args.get("command") if isinstance(args.get("command"), str) else ""
    if cmd and (shell_allow or shell_deny):
        first = cmd.strip().split()[0] if cmd.strip() else ""
        if shell_allow and isinstance(shell_allow, list) and first and first not in shell_allow:
            raise ToolError("Permission denied: shell command not in allowlist.")
        if shell_deny and isinstance(shell_deny, list) and first in shell_deny:
            raise ToolError("Permission denied: shell command is denied by policy.")

    net_allow = constraints.get("network_allowlist") or constraints.get("network_allow")
    net_deny = constraints.get("network_denylist") or constraints.get("network_deny")
    if net_allow or net_deny:
        url = ""
        if isinstance(args.get("url"), str):
            url = args["url"]
        elif isinstance(args.get("repo_url"), str):
            url = args["repo_url"]
        domain = _normalize_domain(url) or ("duckduckgo.com" if name == "web_search" else "")
        if net_allow and isinstance(net_allow, list) and domain and domain not in net_allow:
            raise ToolError("Permission denied: network domain not in allowlist.")
        if net_deny and isinstance(net_deny, list) and domain and domain in net_deny:
            raise ToolError("Permission denied: network domain denied by policy.")

def call_tool(name: str, args: Dict[str, Any], *, conn, user_id: str, agent: str | None = None) -> str:
    """
    Must return a STRING. JSON string is fine.
    The model will interpret it. :contentReference[oaicite:2]{index=2}
    """
    try:
        start_time = time.perf_counter()
        perms = t.get_permissions(conn, user_id)
        agent_val = (agent or "general").strip().lower()

        meta = get_tool_meta(name)
        if not meta:
            raise ToolError(f"Unknown tool: {name}")

        caps = set(meta.get("capabilities") or [])
        if "write" in caps and perms.get("mode") != "write":
            raise ToolError("Permission denied: this tool requires permissions_set(mode='write').")
        if "network" in caps and not perms.get("allow_network"):
            raise ToolError("Permission denied: this tool requires permissions_set(..., allow_network=true).")
        if "fs_read" in caps and not perms.get("allow_fs_read"):
            raise ToolError("Permission denied: this tool requires permissions_set(..., allow_fs_read=true).")
        if "fs_write" in caps and not perms.get("allow_fs_write"):
            raise ToolError("Permission denied: this tool requires permissions_set(..., allow_fs_write=true).")
        if "shell" in caps and not perms.get("allow_shell"):
            raise ToolError("Permission denied: this tool requires permissions_set(..., allow_shell=true).")
        if "exec" in caps and not perms.get("allow_exec"):
            raise ToolError("Permission denied: this tool requires permissions_set(..., allow_exec=true).")
        if name == "memory_search_graph" and args.get("use_embeddings", True) and not perms.get("allow_network"):
            raise ToolError("Permission denied: embeddings require permissions_set(..., allow_network=true).")

        policy = get_tool_policy(conn, user_id=user_id, agent=agent_val, tool_name=name)
        if policy:
            if not policy.get("allow", True):
                raise ToolError("Permission denied: tool disabled by policy.")
            constraints = policy.get("constraints") or {}
            if isinstance(constraints, dict) and constraints:
                _enforce_constraints(name, args, constraints)

        out: Any = None

        if name == "permissions_get":
            out = t.permissions_get(conn, user_id=user_id)
        elif name == "permissions_set":
            out = t.permissions_set(
                conn,
                user_id=user_id,
                mode=args["mode"],
                allow_network=args.get("allow_network"),
                allow_fs_read=args.get("allow_fs_read"),
                allow_fs_write=args.get("allow_fs_write"),
                allow_shell=args.get("allow_shell"),
                allow_exec=args.get("allow_exec"),
            )
        elif name == "tool_policy_set":
            out = t.tool_policy_set(
                conn,
                user_id=user_id,
                agent=args["agent"],
                tool_name=args["tool_name"],
                allow=bool(args["allow"]),
                constraints=args.get("constraints"),
            )
        elif name == "tool_policy_list":
            out = t.tool_policy_list(
                conn,
                user_id=user_id,
                agent=args.get("agent"),
                limit=int(args.get("limit", 200)),
            )
        elif name == "log_action":
            out = t.log_action(
                conn,
                user_id=user_id,
                tool=args["tool"],
                payload=args.get("payload") or {},
                result_id=args.get("result_id"),
            )
        elif name == "audit_log_list":
            out = t.audit_log_list(
                conn,
                user_id=user_id,
                limit=int(args.get("limit", 20)),
                tool=args.get("tool"),
            )
        elif name == "delegate_agent":
            out = t.delegate_agent(
                conn,
                user_id=user_id,
                agent=args["agent"],
                task=args["task"],
                call_tool_fn=call_tool,
                include_history=bool(args.get("include_history", True)),
                history_limit=int(args.get("history_limit", 12)),
            )

        if name == "create_reminder":
            out = t.create_reminder(
                conn,
                user_id=user_id,
                title=args["title"],
                due_at=args["due_at"],
                notes=args.get("notes"),
                rrule=args.get("rrule"),
                channels=args.get("channels"),
            )
        elif name == "list_reminders":
            out = t.list_reminders(conn, user_id=user_id, limit=int(args.get("limit", 10)))
        elif name == "create_event":
            out = t.create_event(
                conn,
                user_id=user_id,
                title=args["title"],
                start_at=args["start_at"],
                end_at=args["end_at"],
                location=args.get("location"),
                notes=args.get("notes"),
            )
        elif name == "update_event":
            out = t.update_event(
                conn,
                user_id=user_id,
                event_id=args["event_id"],
                title=args.get("title"),
                start_at=args.get("start_at"),
                end_at=args.get("end_at"),
                location=args.get("location"),
                notes=args.get("notes"),
                status=args.get("status"),
            )
        elif name == "delete_event":
            out = t.delete_event(
                conn,
                user_id=user_id,
                event_id=args["event_id"],
                hard_delete=bool(args.get("hard_delete", False)),
            )
        elif name == "list_events":
            out = t.list_events(
                conn,
                user_id=user_id,
                start_at=args.get("start_at"),
                end_at=args.get("end_at"),
                limit=int(args.get("limit", 20)),
                include_canceled=bool(args.get("include_canceled", False)),
            )
        elif name == "free_busy":
            out = t.free_busy(conn, user_id=user_id, start_at=args["start_at"], end_at=args["end_at"])
        elif name == "create_task":
            out = t.create_task(
                conn,
                user_id=user_id,
                title=args["title"],
                notes=args.get("notes"),
                priority=args.get("priority"),
                due_at=args.get("due_at"),
                rrule=args.get("rrule"),
            )
        elif name == "complete_task":
            out = t.complete_task(conn, user_id=user_id, task_id=args["task_id"])
        elif name == "list_tasks":
            out = t.list_tasks(conn, user_id=user_id, status=str(args.get("status", "open")), limit=int(args.get("limit", 20)))
        elif name == "add_contact":
            out = t.add_contact(
                conn,
                user_id=user_id,
                name=args["name"],
                email=args.get("email"),
                phone=args.get("phone"),
                notes=args.get("notes"),
            )
        elif name == "get_contact":
            out = t.get_contact(conn, user_id=user_id, query=args["query"], limit=int(args.get("limit", 5)))
        elif name == "convert_timezone":
            out = t.convert_timezone(args["datetime"], to_tz=args["to_tz"], from_tz=args.get("from_tz"))
        elif name == "estimate_travel_time":
            out = t.estimate_travel_time(args["origin"], args["destination"], mode=args.get("mode", "driving"))
        elif name == "draft_email":
            out = t.draft_email(
                conn,
                user_id=user_id,
                to=args.get("to") or [],
                subject=args["subject"],
                purpose=args.get("purpose", "general"),
                context=args.get("context"),
                desired_outcome=args.get("desired_outcome"),
                tone=args.get("tone", "polite"),
            )
        elif name == "send_email":
            out = t.send_email(
                to=args.get("to") or [],
                subject=args["subject"],
                body=args["body"],
                cc=args.get("cc"),
                bcc=args.get("bcc"),
            )
        elif name == "search_email":
            out = t.search_email(query=args["query"], limit=int(args.get("limit", 10)))
        elif name == "create_doc":
            out = t.create_doc(conn, user_id=user_id, title=args["title"], content=args["content"])
        elif name == "append_doc":
            out = t.append_doc(conn, user_id=user_id, doc_id=args["doc_id"], content=args["content"])
        elif name == "export_pdf":
            out = t.export_pdf(conn, user_id=user_id, doc_id=args["doc_id"], output_path=args.get("output_path"))
        elif name == "log_expense":
            out = t.log_expense(
                conn,
                user_id=user_id,
                amount=float(args["amount"]),
                currency=args.get("currency", "USD"),
                category=args.get("category"),
                merchant=args.get("merchant"),
                notes=args.get("notes"),
                occurred_at=args.get("occurred_at"),
            )
        elif name == "budget_status":
            out = t.budget_status(conn, user_id=user_id, window=args.get("window", "month"), currency=args.get("currency", "USD"))

        elif name == "log_metric":
            out = t.log_metric(
                conn,
                user_id=user_id,
                metric=args["metric"],
                value=float(args["value"]),
                unit=args.get("unit"),
                recorded_at=args.get("recorded_at"),
                notes=args.get("notes"),
            )
        elif name == "get_metric_trend":
            out = t.get_metric_trend(conn, user_id=user_id, metric=args["metric"], window=args.get("window", "30d"), bucket=args.get("bucket", "day"))
        elif name == "med_schedule_add":
            out = t.med_schedule_add(
                conn,
                user_id=user_id,
                medication=args["medication"],
                dose=args.get("dose"),
                unit=args.get("unit"),
                times=args.get("times") or [],
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
                notes=args.get("notes"),
            )
        elif name == "med_schedule_check":
            out = t.med_schedule_check(conn, user_id=user_id, at=args.get("at"), window_minutes=int(args.get("window_minutes", 60)))
        elif name == "med_interaction_check":
            out = t.med_interaction_check(args.get("medications") or [])
        elif name == "create_appointment":
            out = t.create_appointment(
                conn,
                user_id=user_id,
                title=args["title"],
                start_at=args["start_at"],
                end_at=args["end_at"],
                provider=args.get("provider"),
                location=args.get("location"),
                reason=args.get("reason"),
                notes=args.get("notes"),
            )
        elif name == "previsit_checklist":
            out = t.previsit_checklist(conn, user_id=user_id, appointment_id=args["appointment_id"], focus=args.get("focus"))
        elif name == "log_meal":
            out = t.log_meal(
                conn,
                user_id=user_id,
                summary=args["summary"],
                calories=args.get("calories"),
                protein_g=args.get("protein_g"),
                carbs_g=args.get("carbs_g"),
                fat_g=args.get("fat_g"),
                recorded_at=args.get("recorded_at"),
                notes=args.get("notes"),
            )
        elif name == "log_workout":
            out = t.log_workout(
                conn,
                user_id=user_id,
                workout_type=args["workout_type"],
                duration_min=args.get("duration_min"),
                intensity=args.get("intensity"),
                calories=args.get("calories"),
                recorded_at=args.get("recorded_at"),
                notes=args.get("notes"),
            )
        elif name == "import_health_data":
            out = t.import_health_data(conn, user_id=user_id, path=args["path"], format=args.get("format", "csv"))
        elif name == "clinical_guideline_search":
            out = t.clinical_guideline_search(query=args["query"], limit=int(args.get("limit", 5)))
        elif name == "screening_get_form":
            out = t.screening_get_form(args["name"])
        elif name == "screening_score":
            out = t.screening_score(args["name"], args.get("responses") or [])
        elif name == "escalation_protocol":
            out = t.escalation_protocol(args.get("symptom"))

        elif name == "query_sql":
            out = t.query_sql(conn, user_id=user_id, sql=args["sql"], params=args.get("params"), limit=int(args.get("limit", 200)))
        elif name == "read_table":
            out = t.read_table(conn, user_id=user_id, table=args["table"], limit=int(args.get("limit", 200)), where=args.get("where"))
        elif name == "write_table":
            out = t.write_table(conn, user_id=user_id, table=args["table"], rows=args.get("rows") or [], mode=args.get("mode", "append"))
        elif name == "upload_file":
            out = t.upload_file(source_path=args["source_path"], dest_name=args.get("dest_name"))
        elif name == "download_file":
            out = t.download_file(path=args["path"])
        elif name == "run_python":
            out = t.run_python(code=args["code"], timeout_sec=int(args.get("timeout_sec", 20)))
        elif name == "log_run":
            out = t.log_run(conn, user_id=user_id, name=args["name"], params=args.get("params"), metrics=args.get("metrics"), notes=args.get("notes"))
        elif name == "get_run":
            out = t.get_run(conn, user_id=user_id, run_id=args["run_id"])

        elif name == "search_code":
            out = t.search_code(query=args["query"], path=args.get("path"), limit=int(args.get("limit", 50)))
        elif name == "open_file":
            out = t.open_file(path=args["path"], start_line=int(args.get("start_line", 1)), end_line=int(args.get("end_line", 200)))
        elif name == "list_files":
            out = t.list_files(
                path=args["path"],
                glob=args.get("glob"),
                limit=int(args.get("limit", 200)),
            )
        elif name == "apply_patch":
            out = t.apply_patch(args["patch"])
        elif name == "run_command":
            out = t.run_command(command=args["command"], cwd=args.get("cwd"), timeout_sec=int(args.get("timeout_sec", 60)))
        elif name == "clone_repo":
            out = t.clone_repo(repo_url=args["repo_url"], dest_dir=args.get("dest_dir"))

        elif name == "fetch_url":
            out = t.fetch_url(args["url"])
        elif name == "extract_text":
            out = t.extract_text(args["html"])
        elif name == "web_search":
            out = t.web_search(args["query"], limit=int(args.get("limit", 5)))
        elif name == "kb_search":
            out = t.kb_search(conn, user_id=user_id, query=args["query"], limit=int(args.get("limit", 5)))
        elif name == "kb_get_doc":
            out = t.kb_get_doc(conn, user_id=user_id, doc_id=args["doc_id"])

        if name == "ds_create_course":
            out = t.ds_create_course(
                conn,
                user_id=user_id,
                goal=args["goal"],
                level=args["level"],
                constraints=args.get("constraints"),
            )
        elif name == "ds_start_course":
            out = t.ds_start_course(conn, user_id=user_id, course_id=args["course_id"])
        elif name == "ds_next_lesson":
            out = t.ds_next_lesson(conn, user_id=user_id, course_id=args["course_id"])
        elif name == "ds_grade_submission":
            out = t.ds_grade_submission(
                conn,
                user_id=user_id,
                course_id=args["course_id"],
                lesson_key=args["lesson_key"],
                submission=args["submission"],
            )
        elif name == "ds_record_progress":
            out = t.ds_record_progress(
                conn,
                user_id=user_id,
                topic=args["topic"],
                score=args.get("score"),
                notes=args.get("notes"),
            )
        elif name == "code_record_progress":
            out = t.code_record_progress(
                conn,
                user_id=user_id,
                topic=args["topic"],
                notes=args.get("notes"),
                evidence_path=args.get("evidence_path"),
            )
        elif name == "code_list_progress":
            out = t.code_list_progress(
                conn,
                user_id=user_id,
                limit=int(args.get("limit", 10)),
            )
        elif name == "memory_search_graph":
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
                candidate_limit=int(args.get("candidate_limit", 250)),
                context_up=int(args.get("context_up", 6)),
                context_down=int(args.get("context_down", 2)),
                use_embeddings=use_embeddings_flag,
            )
        elif name == "set_profile":
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
        elif out is None:
            raise ToolError(f"Unknown tool: {name}")

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        try:
            t.audit_log_append(
                conn,
                user_id=user_id,
                tool=name,
                payload=args,
                result=out,
                status="ok",
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        return json.dumps(out, ensure_ascii=False)

    except Exception as e:
        # Return an error string; the model can recover.
        err = str(e)
        try:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            t.audit_log_append(
                conn,
                user_id=user_id,
                tool=name,
                payload=args,
                result={"error": err},
                status="error",
                error=err,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        return json.dumps({"error": err, "tool": name}, ensure_ascii=False)
