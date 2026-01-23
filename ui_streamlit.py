import hashlib
import json
from pathlib import Path
import re

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app.config import (
    DB_PATH,
    MODEL,
    MAX_HISTORY_MESSAGES,
    MODEL_CONTEXT_TOKENS,
    REQUEST_BUDGET_FRACTION,
    MEMORY_INJECT_MAX_TOKENS,
    MEMORY_INJECT_K,
    MEMORY_INJECT_CANDIDATE_LIMIT,
    PROFILE_INJECT_MAX_TOKENS,
)
from app.db import (
    connect,
    init_db,
    run_migrations,
    get_recent_messages,
    add_message,
    create_conversation,
    get_user_profile,
    get_agent_conversation_id,
    set_agent_conversation_id,
    get_last_user_message_id,
    get_turn_memory_usage,
    record_turn_memory_usage,
    get_turn_tool_usage,
    record_turn_tool_usage,
    get_turn_token_usage,
    record_turn_token_usage,
    record_turn_router_decision,
)
from app.tool_loop import run_with_coordinator, run_router
from app.tool_runtime import call_tool
from app.tool_schemas import LIFE_TOOLS, HEALTH_TOOLS, DS_TOOLS, CODE_TOOLS, GENERAL_TOOLS
from app.token_utils import try_get_encoding, count_message_tokens, token_len, truncate_to_tokens

load_dotenv()

SCHEMA_PATH = Path("app/schema.sql")
ENCODING = try_get_encoding()
REQUEST_BUDGET = int(MODEL_CONTEXT_TOKENS * REQUEST_BUDGET_FRACTION)


def _count_tokens(messages):
    return count_message_tokens(messages, ENCODING)


def _build_context_token_aware(system_message: str, history_messages, new_user_text: str, budget_tokens: int):
    base = [{"role": "system", "content": system_message}]
    acc_tokens = _count_tokens(base)
    user_tokens = token_len("user", ENCODING) + token_len(new_user_text, ENCODING)
    acc_tokens += user_tokens

    selected = []
    for m in reversed(history_messages):
        t = token_len(m["role"], ENCODING) + token_len(m["content"], ENCODING)
        if acc_tokens + t > budget_tokens:
            break
        selected.append({"role": m["role"], "content": m["content"]})
        acc_tokens += t
    selected.reverse()
    return base + selected + [{"role": "user", "content": new_user_text}]


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    return truncate_to_tokens(text, max_tokens, ENCODING)


def _format_memory_results(results, *, max_chars: int = 6000) -> str:
    parts = []
    seen_nodes = set()
    seen_ctx = set()
    for r in results or []:
        title = (r.get("title") or "").strip()
        context = (r.get("context") or "").strip()
        if not context:
            continue
        node_id = str(r.get("node_id") or "").strip()
        if node_id and node_id in seen_nodes:
            continue
        norm = re.sub(r"\s+", " ", context.lower()).strip()
        digest = hashlib.sha1(norm[:800].encode("utf-8")).hexdigest()
        if digest in seen_ctx:
            continue
        seen_ctx.add(digest)
        if node_id:
            seen_nodes.add(node_id)
        header_bits = [f"Title: {title}" if title else "Title: (untitled)"]
        if node_id:
            header_bits.append(f"Node: {node_id}")
        convo_id = str(r.get("conversation_id") or "").strip()
        if convo_id:
            header_bits.append(f"Conversation: {convo_id}")
        header = " | ".join(header_bits)
        parts.append(f"{header}\n{context}")
    blob = "\n\n---\n\n".join(parts).strip()
    if len(blob) > max_chars:
        blob = blob[:max_chars].rstrip() + "\n\n[truncated]"
    return blob


def _call_with_context_retry(
    *,
    client,
    model,
    tools_schema,
    input_items,
    conn,
    user_id,
    agent: str,
    tool_required: bool,
    task_type: str | None,
):
    try:
        final_text, tool_events, usage_stats = run_with_coordinator(
            client=client,
            model=model,
            tools_schema=tools_schema,
            input_items=input_items,
            conn=conn,
            user_id=user_id,
            agent=agent,
            call_tool_fn=call_tool,
            tool_required=tool_required,
            task_type=task_type,
            debug=False,
        )
        return final_text, tool_events, usage_stats
    except Exception as e:
        if "context_length_exceeded" not in str(e):
            raise
        reduced_budget = max(4096, int(REQUEST_BUDGET * 0.6))
        history = get_recent_messages(conn, st.session_state.convo_id, 12)
        base_system = ""
        if input_items and input_items[0].get("role") == "system":
            base_system = str(input_items[0].get("content") or "")
        input_items = _build_context_token_aware(
            system_message=(
                base_system + "\n\nNote: context was trimmed due to token limits."
                if base_system
                else "Note: context was trimmed due to token limits."
            ),
            history_messages=history,
            new_user_text=input_items[-1]["content"],
            budget_tokens=reduced_budget,
        )
        final_text, tool_events, usage_stats = run_with_coordinator(
            client=client,
            model=model,
            tools_schema=tools_schema,
            input_items=input_items,
            conn=conn,
            user_id=user_id,
            agent=agent,
            call_tool_fn=call_tool,
            tool_required=tool_required,
            task_type=task_type,
            debug=False,
        )
        return final_text, tool_events, usage_stats


_MEMORY_TRIGGERS = (
    "last time",
    "previous",
    "previously",
    "earlier",
    "before",
    "remember",
    "what did i",
    "what did we",
    "as we discussed",
    "as i said",
    "you said",
    "continue",
    "pick up",
    "where was i",
    "my plan",
    "my schedule",
    "my progress",
    "my goals",
    "my preferences",
)


def _should_retrieve_memory(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    return any(k in t for k in _MEMORY_TRIGGERS)


st.set_page_config(page_title="Assistant", layout="wide")
st.title("Assistant UI")


@st.cache_resource
def get_client():
    return OpenAI()


def require_client():
    try:
        return get_client()
    except Exception as e:
        st.error(f"OpenAI client init failed: {e}\n\nSet `OPENAI_API_KEY` (e.g. in a `.env` file) and reload.")
        st.stop()


@st.cache_resource
def get_conn():
    conn = connect(DB_PATH)
    init_db(conn, SCHEMA_PATH.read_text(encoding="utf-8"))
    run_migrations(conn)
    return conn


conn = get_conn()
user_id = "local_user"

# --- session state ---
if "agent" not in st.session_state:
    st.session_state.agent = "general"

if "convo_id" not in st.session_state:
    agent_convo = get_agent_conversation_id(conn, user_id, st.session_state.agent)
    if not agent_convo:
        agent_convo = create_conversation(conn, user_id, title=f"{st.session_state.agent} thread")
        set_agent_conversation_id(conn, user_id, st.session_state.agent, agent_convo)
    st.session_state.convo_id = agent_convo

# --- sidebar ---
with st.sidebar:
    st.header("Controls")
    selected_agent = st.selectbox("Agent", ["general", "life", "health", "ds", "code"], index=0)
    if selected_agent != st.session_state.agent:
        st.session_state.agent = selected_agent

    agent_convo = get_agent_conversation_id(conn, user_id, st.session_state.agent)
    if not agent_convo:
        agent_convo = create_conversation(conn, user_id, title=f"{st.session_state.agent} thread")
        set_agent_conversation_id(conn, user_id, st.session_state.agent, agent_convo)
    if st.session_state.convo_id != agent_convo:
        st.session_state.convo_id = agent_convo
        st.rerun()

    if st.button("New conversation"):
        new_id = create_conversation(conn, user_id, title=f"{st.session_state.agent} thread")
        set_agent_conversation_id(conn, user_id, st.session_state.agent, new_id)
        st.session_state.convo_id = new_id
        st.rerun()

    st.caption(f"Conversation: {st.session_state.convo_id}")

    st.divider()
    st.subheader("Permissions")
    from app.tools import permissions_get, permissions_set

    current_perms = permissions_get(conn, user_id=user_id)["permissions"]
    allow_network = bool(current_perms.get("allow_network"))
    perm_mode = st.selectbox(
        "Mode",
        ["read", "write"],
        index=1 if current_perms.get("mode") == "write" else 0,
        key="perm_mode",
    )
    perm_allow_network = st.checkbox(
        "Allow network tools",
        value=bool(current_perms.get("allow_network")),
        key="perm_allow_network",
    )
    perm_allow_fs_read = st.checkbox(
        "Allow filesystem reads",
        value=bool(current_perms.get("allow_fs_read")),
        key="perm_allow_fs_read",
    )
    perm_allow_fs_write = st.checkbox(
        "Allow filesystem writes",
        value=bool(current_perms.get("allow_fs_write")),
        key="perm_allow_fs_write",
    )
    perm_allow_shell = st.checkbox(
        "Allow shell commands",
        value=bool(current_perms.get("allow_shell")),
        key="perm_allow_shell",
    )
    perm_allow_exec = st.checkbox(
        "Allow code execution",
        value=bool(current_perms.get("allow_exec")),
        key="perm_allow_exec",
    )
    if st.button("Save permissions"):
        permissions_set(
            conn,
            user_id=user_id,
            mode=perm_mode,
            allow_network=perm_allow_network,
            allow_fs_read=perm_allow_fs_read,
            allow_fs_write=perm_allow_fs_write,
            allow_shell=perm_allow_shell,
            allow_exec=perm_allow_exec,
        )
        st.rerun()

    last_turn_id = get_last_user_message_id(conn, st.session_state.convo_id)
    mem_usage = []
    tool_usage = []
    token_usage = None
    if last_turn_id:
        debug_conn = connect(DB_PATH)
        try:
            mem_usage = get_turn_memory_usage(debug_conn, st.session_state.convo_id, last_turn_id)
            tool_usage = get_turn_tool_usage(debug_conn, st.session_state.convo_id, last_turn_id)
            token_usage = get_turn_token_usage(debug_conn, st.session_state.convo_id, last_turn_id)
        finally:
            debug_conn.close()

    with st.expander("Memory used", expanded=False):
        mem_items = [row for row in mem_usage if row.get("node_id")]
        if not mem_usage:
            st.caption("No memory retrieval on the last turn.")
        elif not mem_items:
            st.caption("No memory snippets used on the last turn.")
        else:
            for row in mem_items[:3]:
                meta = row.get("meta") or {}
                title = meta.get("title") or "(untitled)"
                st.write(f"- `{row.get('node_id')}` — {title}")
                ctx = (row.get("snippet") or "").strip()
                if ctx:
                    st.caption(ctx[:300] + ("..." if len(ctx) > 300 else ""))

    with st.expander("Memory debug (last turn)", expanded=False):
        meta = {}
        for row in mem_usage:
            if row.get("meta"):
                meta = row.get("meta") or {}
                break
        if not meta:
            st.caption("No memory debug recorded for the last turn.")
        else:
            st.json(meta)

    with st.expander("Tools used (last turn)", expanded=False):
        if not tool_usage:
            st.caption("No tools recorded for the last turn.")
        else:
            for row in tool_usage[:6]:
                header = f"{row.get('tool_name')} ({row.get('status')})"
                st.markdown(f"**{header}**")
                if row.get("error"):
                    st.caption(f"Error: {row.get('error')}")
                input_preview = row.get("input")
                if input_preview is not None:
                    text = json.dumps(input_preview, ensure_ascii=False, indent=2)
                    if len(text) > 2000:
                        text = text[:2000].rstrip() + "\n[truncated]"
                    st.code(text, language="json")
                output_preview = row.get("output")
                if output_preview is not None:
                    text = json.dumps(output_preview, ensure_ascii=False, indent=2)
                    if len(text) > 2000:
                        text = text[:2000].rstrip() + "\n[truncated]"
                    st.code(text, language="json")

    with st.expander("Token/cost (last turn)", expanded=False):
        if not token_usage:
            st.caption("No token usage recorded for the last turn.")
        else:
            st.write(f"Model: `{token_usage.get('model')}`")
            if token_usage.get("prompt_tokens") is not None:
                st.write(f"Prompt tokens: `{token_usage.get('prompt_tokens')}`")
            if token_usage.get("completion_tokens") is not None:
                st.write(f"Completion tokens: `{token_usage.get('completion_tokens')}`")
            if token_usage.get("total_tokens") is not None:
                st.write(f"Total tokens: `{token_usage.get('total_tokens')}`")
            if token_usage.get("tool_calls") is not None:
                st.write(f"Tool calls: `{token_usage.get('tool_calls')}`")
            st.caption("Cost estimate not configured.")

# --- load and display history ---
history = get_recent_messages(conn, st.session_state.convo_id, MAX_HISTORY_MESSAGES)

for m in history:
    role = (m.get("role") or "assistant").strip().lower()
    role = role if role in ("user", "assistant") else "assistant"
    content = m.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# --- chat input ---
prompt = st.chat_input("Type a message…")
if prompt:
    client = require_client()
    agent = st.session_state.agent
    if agent == "life":
        tools_schema = LIFE_TOOLS
    elif agent == "health":
        tools_schema = HEALTH_TOOLS
    elif agent == "code":
        tools_schema = CODE_TOOLS
    elif agent == "general":
        tools_schema = GENERAL_TOOLS
    else:
        tools_schema = DS_TOOLS

    profile = get_user_profile(conn, user_id)
    tz = ""
    if isinstance(profile.get("timezone"), str):
        tz = profile["timezone"].strip()
    elif isinstance(profile.get("tz"), str):
        tz = profile["tz"].strip()
    if not tz:
        tz = "Asia/Ulaanbaatar"
    if agent == "life":
        system_instructions = (
            f"You are the Life Manager. User timezone: {tz}. "
            "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile. "
            "If a tool errors due to permissions, ask the user to toggle Permissions in the sidebar."
            "\n\nMemory policy: If the user references past context or asks for personalization/continuation, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
            "Life query format: <goal/symptom> <routine> <constraint> <timeframe>."
        )
    elif agent == "health":
        system_instructions = (
            f"You are the Health assistant. User timezone: {tz}. "
            "You are not a doctor; give general, evidence-based guidance and encourage professional help for urgent symptoms. "
            "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
            "If the user explicitly asks to save/update stable facts (timezone, conditions, meds, preferences, goals), call set_profile. "
            "If a tool errors due to permissions, ask the user to toggle Permissions in the sidebar."
            "\n\nMemory policy: If the user references past symptoms, treatments, labs, routines, or asks to continue a plan, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
            "Health query format: <symptom/goal> <med/supplement/routine> <constraint> <timeframe>."
        )
    elif agent == "code":
        system_instructions = (
            "You are the Coding assistant. Help with debugging, architecture, and implementation details. "
            "When useful, log progress with code_record_progress and review history with code_list_progress. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile. "
            "If a tool errors due to permissions, ask the user to toggle Permissions in the sidebar."
            "\n\nMemory policy: If the user references prior debugging, ongoing work, or asks to continue, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
            "Code query format: <language/tool> <error/problem> <file/module> <goal>."
        )
    elif agent == "general":
        system_instructions = (
            "You are the Coordinator (General) agent. Route specialized tasks to life/health/ds/code using delegate_agent. "
            "Use your own tools (web_search, fetch_url, extract_text, kb_search) for research and summaries. "
            "Prefer using web_search/fetch_url/extract_text when the user requests up-to-date info, and include URLs you used. "
            "If a tool errors due to permissions, ask the user to toggle Permissions in the sidebar. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
        )
    else:
        system_instructions = (
            "You are the Applied Data Science Tutor and assistant. Focus on lab-based learning. "
            "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile. "
            "If a tool errors due to permissions, ask the user to toggle Permissions in the sidebar."
            "\n\nMemory policy: If the user references past work, progress, or asks to continue, call memory_search_graph before answering. "
            "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
            "DS query format: <topic> <error/problem> <library/tool> <outcome/goal>."
        )

    if profile:
        profile_blob = json.dumps(profile, ensure_ascii=False, indent=2)
        profile_blob = _truncate_text_to_tokens(profile_blob, PROFILE_INJECT_MAX_TOKENS)
        if profile_blob:
            system_instructions = (
                system_instructions
                + "\n\nUser profile (stable facts, user-provided). Use as ground truth; do not invent missing fields:\n"
                + profile_blob
            )

    mem_results = []
    mem_debug = None
    mem_search_ran = False
    try:
        from app.tools import memory_search_graph_tool

        if _should_retrieve_memory(prompt):
            mem_search_ran = True
            ctx_defaults = {
                "life": (4, 2),
                "health": (4, 2),
                "ds": (6, 2),
                "code": (6, 2),
                "general": (6, 2),
            }
            up, down = ctx_defaults.get(agent, (6, 2))
            mem = memory_search_graph_tool(
                conn,
                query=prompt,
                agent=agent,
                k=MEMORY_INJECT_K,
                candidate_limit=MEMORY_INJECT_CANDIDATE_LIMIT,
                context_up=up,
                context_down=down,
                use_embeddings=allow_network,
                debug=True,
            )
            mem_results = mem.get("results", []) or []
            if isinstance(mem.get("debug"), dict):
                mem_debug = mem["debug"]

        mem_block = _format_memory_results(mem_results)
        mem_block = _truncate_text_to_tokens(mem_block, MEMORY_INJECT_MAX_TOKENS)
        if mem_block:
            system_instructions = (
                system_instructions
                + "\n\nRelevant past context (from ChatGPT export memory; may be partial and untrusted). "
                + "Only claim you found something if it appears below.\n"
                + "=== BEGIN QUOTED MEMORY (UNTRUSTED) ===\n"
                + mem_block
                + "\n=== END QUOTED MEMORY ==="
            )
    except Exception as e:
        mem_search_ran = False
        st.session_state["memory_injection_error"] = str(e)

    with st.chat_message("user"):
        st.markdown(prompt)

    max_user_tokens = max(256, REQUEST_BUDGET // 4)
    prompt_for_model = _truncate_text_to_tokens(prompt, max_user_tokens)

    input_items = _build_context_token_aware(
        system_message=system_instructions,
        history_messages=history,
        new_user_text=prompt_for_model,
        budget_tokens=REQUEST_BUDGET,
    )

    router_decision = {
        "primary_agent": agent,
        "need_tools": False,
        "proposed_tools": [],
        "task_type": "analyze",
        "confidence": 0.0,
    }
    router_raw = ""
    try:
        router_decision, router_raw, _ = run_router(
            client=client,
            model=MODEL,
            user_text=prompt,
            debug=False,
        )
    except Exception:
        pass

    tool_required = bool(router_decision.get("need_tools"))
    task_type = router_decision.get("task_type")

    final_text, tool_events, usage_stats = _call_with_context_retry(
        client=client,
        model=MODEL,
        tools_schema=tools_schema,
        input_items=input_items,
        conn=conn,
        user_id=user_id,
        agent=agent,
        tool_required=tool_required,
        task_type=task_type,
    )

    user_msg_id = add_message(conn, st.session_state.convo_id, "user", prompt)
    add_message(conn, st.session_state.convo_id, "assistant", final_text or "")
    try:
        record_turn_router_decision(
            conn,
            conversation_id=st.session_state.convo_id,
            turn_id=user_msg_id,
            agent=agent,
            decision=router_decision,
            raw_output=router_raw,
        )
    except Exception:
        pass
    if tool_events:
        try:
            record_turn_tool_usage(
                conn,
                conversation_id=st.session_state.convo_id,
                turn_id=user_msg_id,
                agent=agent,
                tool_calls=tool_events,
            )
        except Exception:
            pass
    try:
        record_turn_token_usage(
            conn,
            conversation_id=st.session_state.convo_id,
            turn_id=user_msg_id,
            agent=agent,
            model=MODEL,
            prompt_tokens=usage_stats.get("prompt_tokens"),
            completion_tokens=usage_stats.get("completion_tokens"),
            total_tokens=usage_stats.get("total_tokens"),
            tool_calls=usage_stats.get("tool_calls"),
        )
    except Exception:
        pass
    if mem_search_ran:
        try:
            record_turn_memory_usage(
                conn,
                conversation_id=st.session_state.convo_id,
                turn_id=user_msg_id,
                agent=agent,
                results=mem_results,
                debug=mem_debug,
            )
        except Exception:
            pass

    with st.chat_message("assistant"):
        st.markdown(final_text or "")

    st.rerun()
