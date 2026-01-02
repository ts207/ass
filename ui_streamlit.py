import streamlit as st
from openai import OpenAI
import json
from pathlib import Path
import re

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
from app.db import connect, init_db, run_migrations, get_recent_messages, add_message, create_conversation, get_user_profile
from app.tool_loop import run_with_tools
from app.tool_schemas import LIFE_TOOLS, DS_TOOLS
from app.token_utils import try_get_encoding, count_message_tokens, token_len, truncate_to_tokens

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
    for r in results or []:
        title = (r.get("title") or "").strip()
        context = (r.get("context") or "").strip()
        if not context:
            continue
        header = f"Title: {title}" if title else "Title: (untitled)"
        parts.append(f"{header}\n{context}")
    blob = "\n\n---\n\n".join(parts).strip()
    if len(blob) > max_chars:
        blob = blob[:max_chars].rstrip() + "\n\n[truncated]"
    return blob


def _call_with_context_retry(*, client, model, tools_schema, input_items, conn, user_id):
    try:
        return run_with_tools(
            client=client,
            model=model,
            tools_schema=tools_schema,
            input_items=input_items,
            conn=conn,
            user_id=user_id,
            debug=False,
        )
    except Exception as e:
        if "context_length_exceeded" not in str(e):
            raise
        reduced_budget = max(4096, int(REQUEST_BUDGET * 0.6))
        system_msg = input_items[0]["content"] + "\n\nNote: context was trimmed aggressively due to context window limits."
        # Keep only last few messages + user prompt.
        history_msgs = [m for m in input_items[1:-1] if isinstance(m, dict) and "role" in m and "content" in m]
        user_msg = input_items[-1]["content"]
        trimmed_items = _build_context_token_aware(
            system_message=system_msg,
            history_messages=history_msgs[-10:],
            new_user_text=_truncate_text_to_tokens(user_msg, max(256, reduced_budget // 4)),
            budget_tokens=reduced_budget,
        )
        return run_with_tools(
            client=client,
            model=model,
            tools_schema=tools_schema,
            input_items=trimmed_items,
            conn=conn,
            user_id=user_id,
            debug=False,
        )


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so", "to", "of", "in", "on", "at", "for", "from",
    "with", "without", "is", "are", "was", "were", "be", "been", "being", "i", "you", "we", "they", "he", "she",
    "it", "this", "that", "these", "those", "my", "your", "our", "their", "me", "him", "her", "them", "as",
}


def _query_variants(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    variants = [raw]

    if ENCODING is not None:
        toks = ENCODING.encode(raw)
        if len(toks) > 80:
            variants.append(ENCODING.decode(toks[:80]))
    else:
        if len(raw) > 400:
            variants.append(raw[:400])

    words = re.findall(r"[A-Za-z0-9_']{3,}", raw.lower())
    seen = set()
    kept = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if w in seen:
            continue
        seen.add(w)
        kept.append(w)
        if len(kept) >= 10:
            break
    if kept:
        variants.append(" ".join(kept))

    out = []
    for v in variants:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out

st.set_page_config(page_title="Assistant", layout="wide")
st.title("Assistant UI")

# --- init singletons ---
@st.cache_resource
def get_client():
    return OpenAI()

@st.cache_resource
def get_conn():
    conn = connect(DB_PATH)
    init_db(conn, SCHEMA_PATH.read_text(encoding="utf-8"))
    run_migrations(conn)
    return conn

client = get_client()
conn = get_conn()
user_id = "local_user"

# --- session state ---
if "convo_id" not in st.session_state:
    st.session_state.convo_id = create_conversation(conn, user_id, title="Streamlit chat")

if "agent" not in st.session_state:
    st.session_state.agent = "life"

# --- sidebar ---
with st.sidebar:
    st.header("Controls")
    st.session_state.agent = st.selectbox("Agent", ["life", "ds"], index=0)  # change to ["life","ds"] if needed
    if st.button("New conversation"):
        st.session_state.convo_id = create_conversation(conn, user_id, title="Streamlit chat")
        st.rerun()

    st.caption(f"Conversation: {st.session_state.convo_id}")

# --- load and display history ---
history = get_recent_messages(conn, st.session_state.convo_id, MAX_HISTORY_MESSAGES)

for m in history:
    role = m.get("role", "assistant")
    content = m.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# --- chat input ---
prompt = st.chat_input("Type a message…")
if prompt:
    agent = st.session_state.agent
    tools_schema = LIFE_TOOLS if agent == "life" else DS_TOOLS

    profile = get_user_profile(conn, user_id)
    tz = ""
    if isinstance(profile.get("timezone"), str):
        tz = profile["timezone"].strip()
    elif isinstance(profile.get("tz"), str):
        tz = profile["tz"].strip()
    if not tz:
        tz = "Asia/Ulaanbaatar"
    system_instructions = (
        f"You are the Life Manager. User timezone: {tz}. "
        "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
        if agent == "life"
        else
        "You are the Applied Data Science Tutor and assistant. Focus on data data analysis lab-based learning. "
        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
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

    # Auto-retrieve from imported ChatGPT export memory and inject as system context.
    try:
        from app.tools import memory_search_graph_tool

        mem_results = []
        seen_nodes = set()
        variants = _query_variants(prompt)
        agent_tags = [agent] if agent in ("life", "ds") else ["general"]

        for agent_tag in agent_tags:
            for q in variants[:2]:
                mem = memory_search_graph_tool(
                    conn,
                    query=q,
                    agent=agent_tag,
                    k=MEMORY_INJECT_K,
                    candidate_limit=MEMORY_INJECT_CANDIDATE_LIMIT,
                    context_up=6,
                    context_down=4,
                    use_embeddings=True,
                )
                for r in mem.get("results", []) or []:
                    nid = r.get("node_id")
                    if nid and nid not in seen_nodes:
                        seen_nodes.add(nid)
                        mem_results.append(r)
                if len(mem_results) >= MEMORY_INJECT_K:
                    break
            if len(mem_results) >= MEMORY_INJECT_K:
                break

        mem_block = _format_memory_results(mem_results)
        mem_block = _truncate_text_to_tokens(mem_block, MEMORY_INJECT_MAX_TOKENS)
        if mem_block:
            system_instructions = (
                system_instructions
                + "\n\nRelevant past context (from ChatGPT export memory; may be partial). "
                + "Only claim you found something if it appears below:\n"
                + mem_block
            )
    except Exception as e:
        st.session_state["memory_injection_error"] = str(e)

    # show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # build input items (system + db history + new user msg)
    max_user_tokens = max(256, REQUEST_BUDGET // 4)
    prompt_for_model = _truncate_text_to_tokens(prompt, max_user_tokens)

    input_items = _build_context_token_aware(
        system_message=system_instructions,
        history_messages=history,
        new_user_text=prompt_for_model,
        budget_tokens=REQUEST_BUDGET,
    )

    final_text, _ = _call_with_context_retry(
        client=client,
        model=MODEL,
        tools_schema=tools_schema,
        input_items=input_items,
        conn=conn,
        user_id=user_id,
    )

    # persist
    add_message(conn, st.session_state.convo_id, "user", prompt)
    add_message(conn, st.session_state.convo_id, "assistant", final_text or "")

    # show assistant
    with st.chat_message("assistant"):
        st.markdown(final_text or "")

    st.rerun()
