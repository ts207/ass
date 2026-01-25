import json
import re
import time
import uuid
from typing import Any, Dict, List, Tuple
from openai import OpenAI

from .tool_registry import get_tool_meta
from .tools_permissions import get_permissions


def _item_to_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


def _extract_text_from_output_items(output_items: List[Dict[str, Any]]) -> str:
    # Response items typically include {"type":"message","content":[{"type":"output_text","text":"..."}]}
    parts: List[str] = []
    for it in output_items:
        if it.get("type") != "message":
            continue
        for c in it.get("content", []) or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") in ("output_text", "text") and "text" in c:
                parts.append(c["text"])
            elif "text" in c and isinstance(c["text"], str):
                parts.append(c["text"])
    return "\n".join([p for p in parts if p.strip()]).strip()


def _parse_tool_args(raw_args: Any, *, debug: bool = False) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args) if raw_args else {}
    except Exception:
        if debug:
            print("[debug] failed to parse tool arguments; using empty dict")
        return {}


def _extract_final_text(response, output_items: List[Dict[str, Any]]) -> str:
    text = (getattr(response, "output_text", "") or "").strip()
    if text:
        return text
    return _extract_text_from_output_items(output_items)


def _extract_usage(response: Any) -> Dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _merge_usage(total: Dict[str, int], usage: Dict[str, int | None]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(key)
        if val is None:
            continue
        total[key] = total.get(key, 0) + int(val)


def _safe_json_loads(raw: str) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def run_with_tools(
    *,
    client: OpenAI,
    model: str,
    tools_schema: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    conn,
    user_id: str,
    agent: str | None = None,
    max_iters: int = 8,
    debug: bool = False,
    call_tool_fn=None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    previous_response_id = None
    follow_up_input: List[Dict[str, Any]] = input_items
    tool_events: List[Dict[str, Any]] = []
    usage_totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_call_count = 0

    for i in range(max_iters):
        response = client.responses.create(
            model=model,
            tools=tools_schema,
            input=follow_up_input,
            previous_response_id=previous_response_id,
            truncation="auto",
        )
        output_items = [_item_to_dict(x) for x in (response.output or [])]
        _merge_usage(usage_totals, _extract_usage(response))

        if debug:
            types = [x.get("type") for x in output_items]
            print(f"[debug] iter={i} resp_id={response.id} output_types={types}")
            if getattr(response, "output_text", None) is not None:
                print(f"[debug] output_text_repr={repr(response.output_text)}")

        tool_calls = [x for x in output_items if x.get("type") == "function_call"]
        if not tool_calls:
            return (
                _extract_final_text(response, output_items),
                output_items,
                tool_events,
                {"model": model, "tool_calls": tool_call_count, **usage_totals},
            )

        tool_outputs: List[Dict[str, Any]] = []
        for tc in tool_calls:
            name = tc.get("name")
            call_id = tc.get("call_id")
            args = _parse_tool_args(tc.get("arguments", "{}"), debug=debug)

            if call_tool_fn is None:
                from .tool_runtime import call_tool as _call_tool
                call_tool_fn = _call_tool
            tool_call_count += 1
            start = time.perf_counter()
            status = "ok"
            error = None
            try:
                try:
                    result = call_tool_fn(name, args, conn=conn, user_id=user_id, agent=agent)
                except TypeError:
                    result = call_tool_fn(name, args, conn=conn, user_id=user_id)
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                status = "error"
                error = str(e)
                result = json.dumps({"error": error, "tool": name}, ensure_ascii=False)
            duration_ms = int((time.perf_counter() - start) * 1000)
            tool_events.append(
                {
                    "tool_name": name,
                    "input": args,
                    "output": _safe_json_loads(result),
                    "status": status,
                    "error": error,
                    "duration_ms": duration_ms,
                }
            )

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            })

        if debug:
            names = [tc.get("name") for tc in tool_calls]
            print(f"[debug] executed_tools={names}")

        previous_response_id = response.id
        follow_up_input = tool_outputs

    return (
        "Error: tool loop exceeded max iterations.",
        [],
        tool_events,
        {"model": model, "tool_calls": tool_call_count, **usage_totals},
    )


_ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier. Return JSON only, no prose. "
    "Schema: {\"primary_agent\":\"general|life|health|ds|code\",\"need_tools\":true|false,"
    "\"proposed_tools\":[\"tool_name\"],\"task_type\":\"execute|analyze|plan|debug|write\",\"confidence\":0.0}. "
    "Rules: choose the best primary_agent, set need_tools when tools are required, "
    "proposed_tools may be empty if unsure, confidence is 0..1."
)

_VALID_AGENTS = {"general", "life", "health", "ds", "code"}
_VALID_TASK_TYPES = {"execute", "analyze", "plan", "debug", "write"}


def _extract_json_obj(raw: Any) -> Dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return None
    return None


def _normalize_router_decision(raw: Any) -> Dict[str, Any]:
    data = _extract_json_obj(raw) or {}
    primary = str(data.get("primary_agent") or "general").strip().lower()
    if primary not in _VALID_AGENTS:
        primary = "general"
    task_type = str(data.get("task_type") or "analyze").strip().lower()
    if task_type not in _VALID_TASK_TYPES:
        task_type = "analyze"
    proposed = data.get("proposed_tools") or []
    if not isinstance(proposed, list):
        proposed = []
    proposed_tools = [str(x).strip() for x in proposed if str(x).strip()]
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    need_tools = bool(data.get("need_tools", False))
    return {
        "primary_agent": primary,
        "need_tools": need_tools,
        "proposed_tools": proposed_tools,
        "task_type": task_type,
        "confidence": conf,
    }


def run_router(
    *,
    client: OpenAI,
    model: str,
    user_text: str,
    debug: bool = False,
) -> Tuple[Dict[str, Any], str, Dict[str, int | None]]:
    input_items = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_text or ""},
    ]
    response = client.responses.create(
        model=model,
        input=input_items,
        truncation="auto",
    )
    output_items = [_item_to_dict(x) for x in (response.output or [])]
    raw = _extract_final_text(response, output_items)
    if debug:
        print(f"[debug] router_raw={raw!r}")
    decision = _normalize_router_decision(raw)
    usage = _extract_usage(response)
    return decision, raw, usage


def _has_fenced_code(text: str) -> bool:
    return "```" in (text or "")


def _mentions_run_yourself(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in ("run this", "run the following", "run it", "execute this", "copy/paste"))


def _normalize_specialist_payload(raw: Any, task_id: str) -> Dict[str, Any]:
    data = _extract_json_obj(raw) or {}
    status = str(data.get("status") or "done").strip().lower()
    if status not in ("done", "blocked", "needs_more_info"):
        status = "done"
    artifacts = data.get("artifacts") or []
    if not isinstance(artifacts, list):
        artifacts = []
    artifacts = [str(a).strip() for a in artifacts if str(a).strip()]
    result = str(data.get("result") or "").strip()
    proposed = data.get("proposed_tool_calls") or []
    if not isinstance(proposed, list):
        proposed = []
    proposed_calls = []
    for item in proposed:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        reason = str(item.get("reason") or "").strip()
        if tool:
            proposed_calls.append({"tool": tool, "args": args, "reason": reason})
    return {
        "task_id": str(data.get("task_id") or task_id),
        "status": status,
        "artifacts": artifacts,
        "result": result,
        "proposed_tool_calls": proposed_calls,
    }


def _permission_error(caps: List[str], perms: Dict[str, Any]) -> str | None:
    if "write" in caps and perms.get("mode") != "write":
        return "write"
    if "network" in caps and not perms.get("allow_network"):
        return "net"
    if "fs_read" in caps and not perms.get("allow_fs_read"):
        return "fs"
    if "fs_write" in caps and not perms.get("allow_fs_write"):
        return "fsw"
    if "shell" in caps and not perms.get("allow_shell"):
        return "shell"
    if "exec" in caps and not perms.get("allow_exec"):
        return "exec"
    return None


def run_with_coordinator(
    *,
    client: OpenAI,
    model: str,
    tools_schema: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    conn,
    user_id: str,
    agent: str | None = None,
    max_turns: int = 5,
    max_tool_calls: int = 5,
    tool_required: bool = False,
    task_type: str | None = None,
    debug: bool = False,
    call_tool_fn=None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    tool_names = [t.get("name") for t in tools_schema if isinstance(t, dict)]
    allowed_tools = {n for n in tool_names if n}
    task_id = f"T-{uuid.uuid4().hex[:8]}"
    agent_val = (agent or "general").strip().lower() or "general"
    task = {
        "task_id": task_id,
        "agent": agent_val,
        "input": (input_items[-1].get("content") if input_items else "") or "",
        "acceptance_criteria": [
            "Address the user request clearly and directly.",
            "If tool_required=true, use tools to obtain needed data before finalizing.",
        ],
        "budget": {"max_tool_calls": int(max_tool_calls), "max_turns": int(max_turns)},
        "task_type": task_type or "analyze",
        "tool_required": bool(tool_required),
    }

    base_system = ""
    if input_items and input_items[0].get("role") == "system":
        base_system = str(input_items[0].get("content") or "")
    tool_list = ", ".join([n for n in tool_names if n]) or "(none)"
    specialist_system = (
        base_system
        + "\n\nCoordinator protocol: Return strict JSON only. "
        + "Schema: {\"task_id\":\"T-...\",\"status\":\"done|blocked|needs_more_info\","
        + "\"artifacts\":[\"path\"],\"result\":\"...\",\"proposed_tool_calls\":[{\"tool\":\"name\",\"args\":{},\"reason\":\"...\"}]}. "
        + "Specialists do not execute tools; propose tool calls and wait. "
        + "If blocked by permissions, set status=\"blocked\" and result=\"BLOCKED: need /perm <flag> on\". "
        + "Never include fenced code blocks or extra text.\n"
        + f"Task object:\n{json.dumps(task, ensure_ascii=False)}\n"
        + f"Available tools: {tool_list}\n"
        + f"Constraints: tool_required={bool(tool_required)}, allow_exec={bool(get_permissions(conn, user_id).get('allow_exec'))}."
    )

    base_messages = list(input_items)
    if base_messages and base_messages[0].get("role") == "system":
        base_messages[0] = {"role": "system", "content": specialist_system}
    else:
        base_messages.insert(0, {"role": "system", "content": specialist_system})

    tool_events: List[Dict[str, Any]] = []
    usage_totals: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_call_count = 0
    feedback = None
    tool_results = None
    last_feedback = None

    perms = get_permissions(conn, user_id)

    for turn in range(max_turns):
        messages = list(base_messages)
        if feedback:
            messages.append({"role": "system", "content": f"COORDINATOR_FEEDBACK: {feedback}"})
        if tool_results is not None:
            messages.append({"role": "system", "content": "TOOL_RESULTS_JSON: " + json.dumps(tool_results, ensure_ascii=False)})
            messages.append({"role": "user", "content": "Use the tool results and return JSON only."})

        response = client.responses.create(
            model=model,
            input=messages,
            truncation="auto",
        )
        output_items = [_item_to_dict(x) for x in (response.output or [])]
        _merge_usage(usage_totals, _extract_usage(response))
        raw = _extract_final_text(response, output_items)
        if debug:
            print(f"[debug] specialist_raw={raw!r}")

        if not isinstance(raw, str) or not raw.strip():
            feedback = "Empty response. Return strict JSON only."
            last_feedback = feedback
            tool_results = None
            continue

        parsed = _extract_json_obj(raw)
        if parsed is None:
            feedback = "Invalid JSON. Return strict JSON only."
            last_feedback = feedback
            tool_results = None
            continue

        payload = _normalize_specialist_payload(raw, task_id)
        result_text = payload.get("result") or ""
        proposed_calls = payload.get("proposed_tool_calls") or []

        if payload.get("status") == "blocked":
            return result_text, tool_events, {"model": model, "tool_calls": tool_call_count, **usage_totals}

        if proposed_calls:
            if tool_call_count + len(proposed_calls) > max_tool_calls:
                feedback = "Tool budget exceeded. Propose fewer tool calls."
                tool_results = None
                continue

            round_results: List[Dict[str, Any]] = []
            any_executed = False
            blocked_flag = None

            for tc in proposed_calls:
                name = tc.get("tool")
                args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                if name not in allowed_tools:
                    msg = "Tool not allowed for this agent."
                    round_results.append({"tool": name, "status": "error", "error": msg})
                    tool_events.append(
                        {
                            "tool_name": name,
                            "input": args,
                            "output": {"error": msg},
                            "status": "error",
                            "error": msg,
                            "duration_ms": 0,
                        }
                    )
                    continue
                meta = get_tool_meta(name or "")
                if not meta:
                    round_results.append({"tool": name, "status": "error", "error": "Unknown tool"})
                    tool_events.append(
                        {
                            "tool_name": name,
                            "input": args,
                            "output": {"error": "Unknown tool"},
                            "status": "error",
                            "error": "Unknown tool",
                            "duration_ms": 0,
                        }
                    )
                    continue

                required = (meta.get("input_schema") or {}).get("required") or []
                missing = [r for r in required if r not in args]
                if missing:
                    msg = f"Missing required args: {', '.join(missing)}"
                    round_results.append({"tool": name, "status": "error", "error": msg})
                    tool_events.append(
                        {
                            "tool_name": name,
                            "input": args,
                            "output": {"error": msg},
                            "status": "error",
                            "error": msg,
                            "duration_ms": 0,
                        }
                    )
                    continue

                perm_err = _permission_error(meta.get("capabilities") or [], perms)
                if perm_err:
                    blocked_flag = perm_err
                    msg = f"Permission denied: requires /perm {perm_err} on"
                    round_results.append({"tool": name, "status": "error", "error": msg})
                    tool_events.append(
                        {
                            "tool_name": name,
                            "input": args,
                            "output": {"error": msg},
                            "status": "error",
                            "error": msg,
                            "duration_ms": 0,
                        }
                    )
                    continue

                if call_tool_fn is None:
                    from .tool_runtime import call_tool as _call_tool
                    call_tool_fn = _call_tool
                start = time.perf_counter()
                status = "ok"
                error = None
                try:
                    try:
                        result = call_tool_fn(name, args, conn=conn, user_id=user_id, agent=agent_val)
                    except TypeError:
                        result = call_tool_fn(name, args, conn=conn, user_id=user_id)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    status = "error"
                    error = str(e)
                    result = json.dumps({"error": error, "tool": name}, ensure_ascii=False)
                duration_ms = int((time.perf_counter() - start) * 1000)
                tool_events.append(
                    {
                        "tool_name": name,
                        "input": args,
                        "output": _safe_json_loads(result),
                        "status": status,
                        "error": error,
                        "duration_ms": duration_ms,
                    }
                )
                round_results.append(
                    {
                        "tool": name,
                        "status": status,
                        "error": error,
                        "output": _safe_json_loads(result),
                    }
                )
                any_executed = True
                tool_call_count += 1

            if blocked_flag and not any_executed:
                feedback = f"Permission blocked: respond with status=blocked and result='BLOCKED: need /perm {blocked_flag} on'."
                tool_results = round_results
                continue

            tool_results = round_results
            feedback = None
            continue

        if tool_required and tool_call_count == 0:
            feedback = "Tool-required step: propose tool calls before completing."
            last_feedback = feedback
            tool_results = None
            continue

        if not proposed_calls and not result_text.strip() and not tool_required:
            feedback = "Missing result. Return a non-empty result field in the JSON."
            last_feedback = feedback
            tool_results = None
            continue

        if agent_val in ("ds", "code") and tool_required:
            if _has_fenced_code(result_text) and (tool_call_count == 0 or _mentions_run_yourself(result_text)):
                feedback = "Tool-required step: do not return fenced code blocks; propose tool calls instead."
                tool_results = None
                continue

        artifacts = payload.get("artifacts") or []
        if artifacts:
            result_text = result_text + "\n\nArtifacts:\n" + "\n".join(artifacts)
        return result_text, tool_events, {"model": model, "tool_calls": tool_call_count, **usage_totals}

    tail = f" {last_feedback}" if last_feedback else ""
    return (
        f"Error: coordinator exceeded max turns.{tail}",
        tool_events,
        {"model": model, "tool_calls": tool_call_count, **usage_totals},
    )
