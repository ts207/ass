from __future__ import annotations

from typing import Any, Dict, List, Set

from app.tool_schemas import LIFE_TOOLS, HEALTH_TOOLS, DS_TOOLS, CODE_TOOLS, GENERAL_TOOLS


WRITE_TOOLS: Set[str] = {
    # life
    "create_event",
    "update_event",
    "delete_event",
    "create_task",
    "complete_task",
    "add_contact",
    "create_doc",
    "append_doc",
    "draft_email",
    "log_expense",
    "send_email",
    # health
    "create_appointment",
    "previsit_checklist",
    "log_metric",
    "med_schedule_add",
    "log_meal",
    "log_workout",
    "import_health_data",
    # ds
    "write_table",
    "upload_file",
    "run_python",
    "log_run",
    # profile / permissions
    "set_profile",
    "tool_policy_set",
    # filesystem / shell
    "export_pdf",
    "apply_patch",
    "run_command",
    "clone_repo",
    # existing
    "create_reminder",
    "ds_create_course",
    "ds_start_course",
    "ds_next_lesson",
    "ds_grade_submission",
    "ds_record_progress",
    "code_record_progress",
}

NETWORK_TOOLS: Set[str] = {
    "estimate_travel_time",
    "send_email",
    "search_email",
    "fetch_url",
    "web_search",
    "clinical_guideline_search",
    "clone_repo",
}

FS_WRITE_TOOLS: Set[str] = {
    "export_pdf",
    "upload_file",
    "apply_patch",
    "clone_repo",
}

SHELL_TOOLS: Set[str] = {
    "search_code",
    "run_command",
    "apply_patch",
    "clone_repo",
}

EXEC_TOOLS: Set[str] = {"run_python"}


def _schema_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for schema in (LIFE_TOOLS + HEALTH_TOOLS + DS_TOOLS + CODE_TOOLS + GENERAL_TOOLS):
        if isinstance(schema, dict) and schema.get("name"):
            out[schema["name"]] = schema
    return out


_SCHEMAS = _schema_map()

_AGENT_TOOL_LISTS: Dict[str, List[str]] = {
    "life": [s.get("name") for s in LIFE_TOOLS if isinstance(s, dict)],
    "health": [s.get("name") for s in HEALTH_TOOLS if isinstance(s, dict)],
    "ds": [s.get("name") for s in DS_TOOLS if isinstance(s, dict)],
    "code": [s.get("name") for s in CODE_TOOLS if isinstance(s, dict)],
    "general": [s.get("name") for s in GENERAL_TOOLS if isinstance(s, dict)],
}


def _agent_scopes(tool_name: str) -> List[str]:
    scopes = []
    for agent, tools in _AGENT_TOOL_LISTS.items():
        if tool_name in tools:
            scopes.append(agent)
    return scopes or ["general"]


def _capabilities_for(tool_name: str) -> List[str]:
    caps: Set[str] = set()
    if tool_name in WRITE_TOOLS:
        caps.add("write")
    if tool_name in NETWORK_TOOLS:
        caps.add("network")
    if tool_name in FS_WRITE_TOOLS:
        caps.add("fs_write")
    if tool_name in SHELL_TOOLS:
        caps.add("shell")
    if tool_name in EXEC_TOOLS:
        caps.add("exec")
    return sorted(caps)


def _side_effects(capabilities: List[str]) -> str:
    if not capabilities:
        return "none"
    if "shell" in capabilities:
        return "shell"
    if "exec" in capabilities:
        return "exec"
    if "network" in capabilities:
        return "network"
    if "fs_write" in capabilities:
        return "writes_fs"
    return "writes_db"


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
for name, schema in _SCHEMAS.items():
    caps = _capabilities_for(name)
    scopes = _agent_scopes(name)
    TOOL_REGISTRY[name] = {
        "name": name,
        "agent_scope": scopes[0] if len(scopes) == 1 else "any",
        "agent_scopes": scopes,
        "capabilities": caps,
        "side_effects": _side_effects(caps),
        "idempotent": "write" not in caps,
        "input_schema": schema.get("parameters"),
        "output_schema": None,
    }


def get_tool_meta(name: str) -> Dict[str, Any] | None:
    return TOOL_REGISTRY.get(name)


def get_tool_capabilities(name: str) -> List[str]:
    meta = get_tool_meta(name) or {}
    caps = meta.get("capabilities") or []
    return list(caps)


def get_tool_scopes(name: str) -> List[str]:
    meta = get_tool_meta(name) or {}
    scopes = meta.get("agent_scopes") or []
    return list(scopes)
