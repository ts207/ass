import json
import time
from typing import Any, Dict, List, Tuple
from openai import OpenAI


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
