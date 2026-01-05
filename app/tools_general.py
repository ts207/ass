import html as html_lib
import json
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from app.config import MODEL, PROFILE_INJECT_MAX_TOKENS
from app.token_utils import try_get_encoding, truncate_to_tokens
from app.tools_core import _http_get, _truncate_json_str
from app.tools_memory import _get_openai_client
from app.tool_loop import run_with_tools

from app.db import (
    add_message,
    create_conversation,
    get_agent_conversation_id,
    get_recent_messages,
    get_user_profile,
    set_agent_conversation_id,
)


def fetch_url(url: str) -> Dict[str, Any]:
    u = (url or "").strip()
    if not u:
        raise ValueError("url is required")
    status, final_url, headers, text = _http_get(u, max_bytes=400_000)
    return {"status": status, "url": final_url, "content_type": headers.get("content-type", ""), "text": _truncate_json_str(text, 200_000)}


def extract_text(html: str) -> Dict[str, Any]:
    s = html or ""
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return {"text": _truncate_json_str(s, 200_000)}


def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = max(1, min(int(limit or 5), 10))
    url = "https://duckduckgo.com/html/?" + urlencode({"q": q})
    status, _, _, html_text = _http_get(url, max_bytes=800_000)
    if status < 200 or status >= 300:
        raise ValueError(f"Search failed (HTTP {status})")
    results = []
    for m in re.finditer(r'(?is)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text):
        href = html_lib.unescape(m.group(1))
        title = html_lib.unescape(re.sub(r"(?is)<.*?>", "", m.group(2))).strip()
        if "duckduckgo.com/l/?" in href and "uddg=" in href:
            try:
                qs = parse_qs(urlparse(href).query)
                if "uddg" in qs and qs["uddg"]:
                    href = unquote(qs["uddg"][0])
            except Exception:
                pass
        if title and href:
            results.append({"title": title, "url": href})
        if len(results) >= lim:
            break
    return {"query": q, "results": results}


def kb_search(conn, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    lim = max(1, min(int(limit or 5), 20))
    like = f"%{q}%"
    rows = conn.execute(
        "SELECT id, title, updated_at, substr(content, 1, 400) AS snippet FROM documents "
        "WHERE user_id=? AND (title LIKE ? OR content LIKE ?) ORDER BY updated_at DESC LIMIT ?",
        (user_id, like, like, lim),
    ).fetchall()
    return {"results": [dict(r) for r in rows]}


def kb_get_doc(conn, user_id: str, doc_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT id, title, content, updated_at FROM documents WHERE user_id=? AND id=?",
        (user_id, doc_id),
    ).fetchone()
    if not row:
        raise ValueError("Document not found for this user.")
    return dict(row)


def _agent_tools_schema(agent: str):
    from app.tool_schemas import LIFE_TOOLS, HEALTH_TOOLS, DS_TOOLS, CODE_TOOLS

    if agent == "life":
        return LIFE_TOOLS
    if agent == "health":
        return HEALTH_TOOLS
    if agent == "code":
        return CODE_TOOLS
    return DS_TOOLS


def _agent_system_instructions(agent: str, tz: str) -> str:
    if agent == "life":
        return (
            f"You are the Life Manager. User timezone: {tz}. "
            "You can manage calendar events, tasks, reminders, contacts, documents, and expenses using tools. "
            "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
            "If a tool errors due to permissions, explain what must be enabled. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
            "\n\nMemory policy: If the user references past context or asks for personalization/continuation, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
            "Life query format: <goal/symptom> <routine> <constraint> <timeframe> (use discriminative nouns)."
        )
    if agent == "health":
        return (
            f"You are the Health assistant. User timezone: {tz}. "
            "You are not a doctor; give general, evidence-based guidance and encourage professional help for urgent symptoms. "
            "You can log metrics, med schedules, appointments, meals, workouts, and screening forms using tools. "
            "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
            "If a tool errors due to permissions, explain what must be enabled. "
            "If the user explicitly asks to save/update stable facts (timezone, conditions, meds, preferences, goals), call set_profile."
            "\n\nMemory policy: If the user references past symptoms, treatments, labs, routines, or asks to continue a plan, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
            "Health query format: <symptom/goal> <med/supplement/routine> <constraint> <timeframe>."
        )
    if agent == "code":
        return (
            "You are the Coding assistant. Help with debugging, architecture, and implementation details. "
            "When useful, log progress with code_record_progress and review history with code_list_progress. "
            "If a tool errors due to permissions, explain what must be enabled. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
            "\n\nMemory policy: If the user references prior debugging, ongoing work, or asks to continue, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
            "Code query format: <language/tool> <error/problem> <file/module> <goal>."
        )
    return (
        "You are a Data Science Course Assistant. You can design short courses, serve lessons, grade submissions, and adapt the plan based on progress. "
        "You can query the local SQLite DB, upload files, and log experiment runs using tools. "
        "If a tool errors due to permissions, explain what must be enabled. "
        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
        "\n\nMemory policy: If the user references past work, progress, or asks to continue, call memory_search_graph before answering. "
        "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
        "DS query format: <topic> <error/problem> <library/tool> <outcome/goal>."
    )


def delegate_agent(
    conn,
    *,
    user_id: str,
    agent: str,
    task: str,
    call_tool_fn,
    include_history: bool = True,
    history_limit: int = 12,
) -> Dict[str, Any]:
    agent_val = (agent or "").strip().lower()
    if agent_val not in ("life", "health", "ds", "code"):
        raise ValueError("agent must be one of: life, health, ds, code")
    text = (task or "").strip()
    if not text:
        raise ValueError("task is required")
    if call_tool_fn is None:
        raise ValueError("call_tool_fn is required for delegation")

    profile = get_user_profile(conn, user_id)
    tz = ""
    if isinstance(profile.get("timezone"), str):
        tz = profile["timezone"].strip()
    elif isinstance(profile.get("tz"), str):
        tz = profile["tz"].strip()
    if not tz:
        tz = "Asia/Ulaanbaatar"

    system_instructions = _agent_system_instructions(agent_val, tz)

    if profile:
        enc = try_get_encoding()
        profile_blob = json.dumps(profile, ensure_ascii=False, indent=2)
        profile_blob = truncate_to_tokens(profile_blob, PROFILE_INJECT_MAX_TOKENS, enc)
        if profile_blob:
            system_instructions = (
                system_instructions
                + "\n\nUser profile (stable facts, user-provided). Use as ground truth; do not invent missing fields:\n"
                + profile_blob
            )

    agent_convo = get_agent_conversation_id(conn, user_id, agent_val)
    if not agent_convo:
        agent_convo = create_conversation(conn, user_id, title=f"{agent_val} thread")
        set_agent_conversation_id(conn, user_id, agent_val, agent_convo)

    history_msgs: List[Dict[str, Any]] = []
    lim = max(0, min(int(history_limit or 0), 50))
    if include_history and lim > 0:
        history_msgs = get_recent_messages(conn, agent_convo, lim)

    input_items = [{"role": "system", "content": system_instructions}]
    input_items.extend(history_msgs)
    input_items.append({"role": "user", "content": text})

    client = _get_openai_client()
    tools_schema = _agent_tools_schema(agent_val)

    try:
        final_text, _ = run_with_tools(
            client=client,
            model=MODEL,
            tools_schema=tools_schema,
            input_items=input_items,
            conn=conn,
            user_id=user_id,
            debug=False,
            call_tool_fn=call_tool_fn,
        )
    except Exception as e:
        if "context_length_exceeded" in str(e) and history_msgs:
            final_text, _ = run_with_tools(
                client=client,
                model=MODEL,
                tools_schema=tools_schema,
                input_items=[{"role": "system", "content": system_instructions}, {"role": "user", "content": text}],
                conn=conn,
                user_id=user_id,
                debug=False,
                call_tool_fn=call_tool_fn,
            )
        else:
            raise

    add_message(conn, agent_convo, "user", f"{agent_val}: {text}")
    add_message(conn, agent_convo, "assistant", final_text or "")

    return {"agent": agent_val, "task": text, "response": final_text, "conversation_id": agent_convo}
