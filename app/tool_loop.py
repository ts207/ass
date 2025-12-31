import json
from typing import Any, Dict, List, Tuple
from openai import OpenAI

from .tool_runtime import call_tool


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


def run_with_tools(
    *,
    client: OpenAI,
    model: str,
    tools_schema: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    conn,
    user_id: str,
    max_iters: int = 8,
    debug: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    previous_response_id = None
    follow_up_input: List[Dict[str, Any]] = input_items

    for i in range(max_iters):
        response = client.responses.create(
            model=model,
            tools=tools_schema,
            input=follow_up_input,
            previous_response_id=previous_response_id,
        )
        output_items = [_item_to_dict(x) for x in (response.output or [])]

        if debug:
            types = [x.get("type") for x in output_items]
            print(f"[debug] iter={i} resp_id={response.id} output_types={types}")
            if getattr(response, "output_text", None) is not None:
                print(f"[debug] output_text_repr={repr(response.output_text)}")

        tool_calls = [x for x in output_items if x.get("type") == "function_call"]
        if not tool_calls:
            return _extract_final_text(response, output_items), output_items

        tool_outputs: List[Dict[str, Any]] = []
        for tc in tool_calls:
            name = tc.get("name")
            call_id = tc.get("call_id")
            args = _parse_tool_args(tc.get("arguments", "{}"), debug=debug)

            result = call_tool(name, args, conn=conn, user_id=user_id)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

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

    return "Error: tool loop exceeded max iterations.", []
