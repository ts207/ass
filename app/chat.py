import argparse
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .config import (
    DB_PATH,
    MODEL,
    MODEL_CONTEXT_TOKENS,
    REQUEST_BUDGET_FRACTION,
    MEMORY_INJECT_MAX_TOKENS,
    MEMORY_INJECT_K,
    MEMORY_INJECT_CANDIDATE_LIMIT,
    PROFILE_INJECT_MAX_TOKENS,
)
from .db import (
    connect,
    init_db,
    create_conversation,
    list_conversations,
    add_message,
    get_latest_conversation_id,
    get_agent_conversation_id,
    set_agent_conversation_id,
    run_migrations,
    get_user_profile,
    record_turn_memory_usage,
    record_turn_tool_usage,
    record_turn_token_usage,
    record_turn_router_decision,
)
from .tool_loop import run_with_coordinator, run_router
from .tools_permissions import get_permissions
from .tool_runtime import call_tool
from .tool_schemas import LIFE_TOOLS, HEALTH_TOOLS, DS_TOOLS, CODE_TOOLS, GENERAL_TOOLS
from .token_utils import try_get_encoding, count_message_tokens, token_len, truncate_to_tokens

load_dotenv()

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

def split_prefixed_requests(s: str, *, default_agent: str = "life"):
    """
    Accepts inputs like:
      "life: list reminders; ds: next lesson"
    Returns list of (agent, text) preserving order.
    If no prefixes, routes to default_agent (defaults to "life").
    """
    import re

    default = (default_agent or "").strip().lower() or "life"
    if default not in ("life", "health", "ds", "code", "general"):
        default = "life"

    pattern = re.compile(r"\b(life|health|ds|code|general):", re.IGNORECASE)
    matches = list(pattern.finditer(s))
    if not matches:
        text = s.strip()
        return [(default, text)] if text else []

    pieces = []
    for i, m in enumerate(matches):
        agent = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        chunk = s[start:end].strip(" ;,")
        if chunk:
            pieces.append((agent, chunk.strip()))
    return pieces


def _msg(role: str, text: str):
    # Simple message shape accepted by Responses API for inputs
    return {"role": role, "content": text}


def _get_encoding():
    return try_get_encoding()


def _count_tokens(messages, enc) -> int:
    return count_message_tokens(messages, enc)


def _fetch_all_messages(conn, convo_id: str):
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
        (convo_id,),
    ).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]


def _summarize_messages(client: OpenAI, model: str, messages):
    formatted = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt = (
        "Summarize the following conversation into concise bullet points capturing key facts, decisions, "
        "and follow-ups. Keep it short and information-dense."
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": formatted},
        ],
    )
    return (resp.output_text or "").strip()


def _count_tokens_for_context(messages, system_message: str, enc) -> int:
    base = [{"role": "system", "content": system_message}]
    return _count_tokens(base + messages, enc)


def _ensure_budget_with_summary(conn, client, model, convo_id: str, system_message: str, budget_tokens: int, enc, user_id: str):
    messages = _fetch_all_messages(conn, convo_id)
    while True:
        total = _count_tokens_for_context(messages, system_message, enc)
        if total <= budget_tokens:
            return messages
        if len(messages) <= 4:
            return messages
        chunk = []
        chunk_tokens = 0
        target = max(512, budget_tokens // 5)
        for m in messages:
            t = token_len(m["role"], enc) + token_len(m["content"], enc)
            chunk.append(m)
            chunk_tokens += t
            if chunk_tokens >= target:
                break
        summary = _summarize_messages(client, model, chunk)
        if not summary:
            return messages
        ids = [m["id"] for m in chunk]
        conn.execute(
            f"DELETE FROM messages WHERE id IN ({','.join(['?']*len(ids))})",
            ids,
        )
        add_message(conn, convo_id, "system", f"Conversation summary: {summary}")
        messages = _fetch_all_messages(conn, convo_id)


def _build_context_token_aware(system_message: str, history_messages, new_user_text: str, budget_tokens: int, enc):
    base = [{"role": "system", "content": system_message}]
    acc_tokens = _count_tokens(base, enc)
    user_tokens = token_len("user", enc) + token_len(new_user_text, enc)
    acc_tokens += user_tokens

    selected = []
    for m in reversed(history_messages):
        t = token_len(m["role"], enc) + token_len(m["content"], enc)
        if acc_tokens + t > budget_tokens:
            break
        selected.append({"role": m["role"], "content": m["content"]})
        acc_tokens += t
    selected.reverse()
    return base + selected + [{"role": "user", "content": new_user_text}]


def _truncate_text_to_tokens(text: str, max_tokens: int, enc) -> str:
    return truncate_to_tokens(text, max_tokens, enc)


def _format_memory_results(results: list[dict], *, max_chars: int = 6000) -> str:
    parts: list[str] = []
    seen_nodes: set[str] = set()
    seen_ctx: set[str] = set()
    for r in results:
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


def _parse_args():
    parser = argparse.ArgumentParser(description="CLI personal assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--reminder-interval", type=int, default=30, help="Seconds between reminder checks")
    return parser.parse_args()


def _utcnow():
    return datetime.now(timezone.utc)


def _parse_dt(dt_str: str):
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def check_due_reminders(conn, *, debug: bool = False):
    """Find scheduled reminders that are due, mark them done, and return them."""
    now = _utcnow()
    now_iso = now.isoformat()
    rows = conn.execute(
        "SELECT id, user_id, title, due_at, due_at_utc FROM reminders "
        "WHERE status='scheduled' AND (due_at_utc IS NULL OR due_at_utc<=?)",
        (now_iso,),
    ).fetchall()
    due_ids = []
    due_entries = []
    for r in rows:
        dt_str = r["due_at_utc"] or r["due_at"]
        dt = _parse_dt(dt_str)
        if dt is None:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt.astimezone(timezone.utc) <= now:
            due_ids.append(r["id"])
            due_entries.append(dict(r))
    if due_ids:
        conn.execute(
            f"UPDATE reminders SET status='done', fired_at=?, updated_at=? WHERE id IN ({','.join(['?']*len(due_ids))})",
            [_utcnow().isoformat(), _utcnow().isoformat(), *due_ids],
        )
        conn.commit()
        if debug:
            print(f"[debug] marked {len(due_ids)} reminders done")
    return due_entries


def start_reminder_watcher(db_path: Path, *, interval: int, debug: bool = False):
    stop_event = threading.Event()

    def _loop():
        conn = connect(db_path, check_same_thread=False)
        try:
            while not stop_event.is_set():
                due = check_due_reminders(conn, debug=debug)
                for d in due:
                    print(f"\n[reminder] {d['title']} (due {d['due_at']})")
                stop_event.wait(interval)
        finally:
            conn.close()

    thread = threading.Thread(target=_loop, daemon=True, name="reminder-watcher")
    thread.start()
    return stop_event, thread


def main():
    args = _parse_args()
    debug = args.debug

    try:
        client = OpenAI()
    except Exception as e:
        raise SystemExit(
            "OpenAI client init failed. Set OPENAI_API_KEY (e.g. in a .env file) and retry."
        ) from e
    enc = _get_encoding()
    REQUEST_BUDGET = int(MODEL_CONTEXT_TOKENS * REQUEST_BUDGET_FRACTION)
    user_id = "local_user"

    conn = connect(DB_PATH)
    init_db(conn, SCHEMA_PATH.read_text(encoding="utf-8"))
    run_migrations(conn)

    convo_id = get_latest_conversation_id(conn, user_id)
    if not convo_id:
        convo_id = create_conversation(conn, user_id, title="CLI chat")

    print(f"Conversation: {convo_id}")
    print("Commands: /new, /list, /use <id>, /perm, /exit")

    stop_event, watcher_thread = start_reminder_watcher(DB_PATH, interval=args.reminder_interval, debug=debug)

    try:
        while True:
            user_text = input("> ").strip()
            if not user_text:
                continue

            # --- commands ---
            if user_text == "/exit":
                break

            if user_text == "/new":
                convo_id = create_conversation(conn, user_id, title="CLI chat")
                print(f"Conversation: {convo_id}")
                continue

            if user_text == "/list":
                convos = list_conversations(conn, user_id, limit=20)
                for cid, title, updated in convos:
                    print(f"{cid}  {updated}  {title}")
                continue

            if user_text == "/selftest":
                print("Try prompts:")
                print("  life: remind me tomorrow at 9am to take meds")
                print("  life: create a task to file taxes due next Friday")
                print("  ds: create a short course on feature engineering for beginners")
                print("  ds: next lesson for my course_id")
                print("  general: web_search best time to visit japan")
                continue

            if user_text.startswith("/use "):
                convo_id = user_text.split(" ", 1)[1].strip()
                print(f"Conversation: {convo_id}")
                continue

            if user_text.startswith("/perm"):
                from app.tools import permissions_get, permissions_set

                parts = user_text.split()
                if len(parts) == 1:
                    print(json.dumps(permissions_get(conn, user_id=user_id), ensure_ascii=False, indent=2))
                    continue

                existing = permissions_get(conn, user_id=user_id)["permissions"]
                mode = existing.get("mode", "read")

                def _usage() -> None:
                    print(
                        "Usage: /perm [read|write] OR /perm {net|fs|fsw|shell|exec} {on|off} OR /perm fs {read|write} {on|off} OR /perm"
                    )

                args = parts[1:]
                if len(args) == 1 and "=" in args[0]:
                    left, right = args[0].split("=", 1)
                    args = [left, right]

                if len(args) >= 1 and args[0] in ("read", "write"):
                    mode = args[0]
                    updated = permissions_set(conn, user_id=user_id, mode=mode)
                    print(json.dumps(updated, ensure_ascii=False, indent=2))
                    continue

                if len(args) == 2 and args[0] in ("read", "write") and args[1] in ("on", "off"):
                    mode = args[0]
                    updated = permissions_set(conn, user_id=user_id, mode=mode)
                    print(json.dumps(updated, ensure_ascii=False, indent=2))
                    continue

                # /perm fs read on|off  (and /perm fs=read)
                if len(args) == 2 and args[0].lower() in ("fs", "files", "file", "filesystem") and args[1] in ("read", "write"):
                    args = [args[0], args[1], "on"]
                if len(args) == 3 and args[0].lower() in ("fs", "files", "file", "filesystem") and args[2] in ("on", "off"):
                    scope = args[1].lower()
                    val = args[2] == "on"
                    kwargs = {"mode": mode}
                    if scope == "read":
                        kwargs["allow_fs_read"] = val
                    elif scope == "write":
                        kwargs["allow_fs_write"] = val
                    else:
                        _usage()
                        continue
                    updated = permissions_set(conn, user_id=user_id, **kwargs)
                    print(json.dumps(updated, ensure_ascii=False, indent=2))
                    continue

                if len(args) == 2 and args[1] in ("on", "off"):
                    flag = args[0].lower()
                    val = args[1] == "on"
                    kwargs = {"mode": mode}
                    if flag in ("net", "network"):
                        kwargs["allow_network"] = val
                    elif flag in ("fs", "files", "file", "filesystem"):
                        kwargs["allow_fs_read"] = val
                    elif flag in ("fsw", "fs_write", "fswrite", "filesystem_write", "files_write"):
                        kwargs["allow_fs_write"] = val
                    elif flag in ("shell",):
                        kwargs["allow_shell"] = val
                    elif flag in ("exec", "python"):
                        kwargs["allow_exec"] = val
                    else:
                        _usage()
                        continue
                    updated = permissions_set(conn, user_id=user_id, **kwargs)
                    print(json.dumps(updated, ensure_ascii=False, indent=2))
                    continue

                _usage()
                continue

            # --- handle one or more prefixed agent requests ---
            requests = split_prefixed_requests(user_text)
            responses = []

            for agent, text in requests:
                display_agent = "ds" if agent == "ds" else agent

                # each agent keeps its own conversation thread for memory isolation
                agent_convo = get_agent_conversation_id(conn, user_id, agent)
                if not agent_convo:
                    agent_convo = create_conversation(conn, user_id, title=f"{display_agent} thread")
                    set_agent_conversation_id(conn, user_id, agent, agent_convo)

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
                        "You can manage calendar events, tasks, reminders, contacts, documents, and expenses using tools. "
                        "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
                        "If a tool errors due to permissions, ask the user to enable it via /perm (e.g. /perm net on, /perm fs on, /perm fsw on, /perm shell on). "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                        "\n\nMemory policy: If the user references past context or asks for personalization/continuation, call memory_search_graph before answering. "
                        "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
                        "Life query format: <goal/symptom> <routine> <constraint> <timeframe> (use discriminative nouns). "
                        "After retrieval: briefly summarize what you found and ground your answer in it; if nothing relevant, say so and suggest what to log."
                    )
                elif agent == "health":
                    system_instructions = (
                        f"You are the Health assistant. User timezone: {tz}. "
                        "You are not a doctor; give general, evidence-based guidance and encourage professional help for urgent symptoms. "
                        "You can log metrics, meds schedules, appointments, meals, workouts, and screening forms using tools. "
                        "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
                        "If a tool errors due to permissions, ask the user to enable it via /perm (e.g. /perm net on). "
                        "If the user explicitly asks to save/update stable facts (timezone, conditions, meds, preferences, goals), call set_profile."
                        "\n\nMemory policy: If the user references past symptoms, treatments, labs, routines, or asks to continue a plan, call memory_search_graph before answering. "
                        "Use k=5 candidate_limit=250 context_up=4 context_down=2. "
                        "Health query format: <symptom/goal> <med/supplement/routine> <constraint> <timeframe>. "
                        "After retrieval: summarize what you found and ground your advice; if nothing relevant, say so and ask clarifying questions."
                    )
                elif agent == "code":
                    system_instructions = (
                        "You are the Coding assistant. Help with debugging, architecture, and implementation details. "
                        "When useful, log progress with code_record_progress and review history with code_list_progress. "
                        "If explicitly enabled in permissions, you can search/read files and run safe repo commands via tools. "
                        "If a tool errors due to permissions, ask the user to enable it via /perm (e.g. /perm shell on, /perm fs on, /perm fsw on). "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                        "\n\nMemory policy: If the user references prior debugging, ongoing work, or asks to continue, call memory_search_graph before answering. "
                        "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
                        "Code query format: <language/tool> <error/problem> <file/module> <goal>. "
                        "After retrieval: cite what you found and then propose the next concrete steps."
                    )
                elif agent == "general":
                    system_instructions = (
                        "You are the Coordinator (General) agent. Route specialized tasks to life/health/ds/code using delegate_agent. "
                        "Use your own tools (web_search, fetch_url, extract_text, kb_search) for research and summaries. "
                        "Prefer using web_search/fetch_url/extract_text when the user requests up-to-date info, and include URLs you used. "
                        "If a tool errors due to permissions, ask the user to enable it via /perm net on (and /perm write if needed). "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                    )
                else:
                    system_instructions = (
                        "You are a Data Science Course Assistant. You can design short courses, serve lessons, grade submissions, and adapt the plan based on progress. "
                        "You can also query the local SQLite DB, list local files, and log experiment runs using tools. "
                        "If files are needed, ask the user to place them in data/imports/ and use list_files to confirm paths. "
                        "If a tool errors due to permissions, ask the user to enable it via /perm (e.g. /perm fs on for list_files, /perm exec on for run_python). "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                        "\n\nMemory policy: If the user references past work, progress, or asks to continue, call memory_search_graph before answering. "
                        "Use k=5 candidate_limit=250 context_up=6 context_down=2. "
                        "DS query format: <topic> <error/problem> <library/tool> <outcome/goal> (avoid filler). "
                        "After retrieval: summarize what you found and ground your answer; if memory is weak, say so and suggest what to log next."
                    )

                if profile:
                    profile_blob = json.dumps(profile, ensure_ascii=False, indent=2)
                    profile_blob = _truncate_text_to_tokens(profile_blob, PROFILE_INJECT_MAX_TOKENS, enc)
                    if profile_blob:
                        system_instructions = (
                            system_instructions
                            + "\n\nUser profile (stable facts, user-provided). Use as ground truth; do not invent missing fields:\n"
                            + profile_blob
                        )

                # Auto-retrieve from imported ChatGPT export memory and inject as system context.
                # This makes memory available even when the model doesn't decide to call the tool.
                mem_results: list[dict] = []
                mem_debug = None
                mem_search_ran = False
                try:
                    from app.tools import memory_search_graph_tool

                    allow_network = False
                    try:
                        allow_network = bool(get_permissions(conn, user_id).get("allow_network"))
                    except Exception:
                        allow_network = False
                    if _should_retrieve_memory(text):
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
                            query=text,
                            agent=agent,
                            k=MEMORY_INJECT_K,
                            candidate_limit=MEMORY_INJECT_CANDIDATE_LIMIT,
                            context_up=up,
                            context_down=down,
                            use_embeddings=allow_network,
                            debug=debug,
                        )
                        mem_results = mem.get("results", []) or []
                        if isinstance(mem.get("debug"), dict):
                            mem_debug = mem["debug"]
                        if debug and isinstance(mem.get("debug"), dict):
                            dbg = mem["debug"]
                            qd = dbg.get("query", {})
                            print(
                                f"[debug] memory: cleaned_query={qd.get('cleaned')} keywords={qd.get('keywords')} "
                                f"candidates={dbg.get('candidates')} used_embeddings={dbg.get('used_embeddings')}"
                            )

                    mem_block = _format_memory_results(mem_results)
                    mem_block = _truncate_text_to_tokens(mem_block, MEMORY_INJECT_MAX_TOKENS, enc)
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
                    mem_results = []
                    mem_debug = None
                    if debug:
                        print(f"[debug] memory injection failed: {e}")

                history = _ensure_budget_with_summary(
                    conn,
                    client,
                    MODEL,
                    agent_convo,
                    system_instructions,
                    REQUEST_BUDGET,
                    enc,
                    user_id,
                )

                # Ensure a single huge user paste doesn't blow the budget by itself.
                # Leave headroom for system + tool reasoning.
                max_user_tokens = max(256, REQUEST_BUDGET // 4)
                text_for_model = _truncate_text_to_tokens(text, max_user_tokens, enc)

                input_items = _build_context_token_aware(
                    system_message=system_instructions,
                    history_messages=history,
                    new_user_text=text_for_model,
                    budget_tokens=REQUEST_BUDGET,
                    enc=enc,
                )
                if debug:
                    print(f"[debug] approx request tokens: {_count_tokens(input_items, enc)}")

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
                        user_text=text,
                        debug=debug,
                    )
                except Exception as e:
                    if debug:
                        print(f"[debug] router failed: {e}")

                tool_required = bool(router_decision.get("need_tools"))
                task_type = router_decision.get("task_type")

                tool_events = []
                usage_stats = {}
                try:
                    final_text, tool_events, usage_stats = run_with_coordinator(
                        client=client,
                        model=MODEL,
                        tools_schema=tools_schema,
                        input_items=input_items,
                        conn=conn,
                        user_id=user_id,
                        agent=agent,
                        tool_required=tool_required,
                        task_type=task_type,
                        debug=debug,
                        call_tool_fn=call_tool,
                    )
                except Exception as e:
                    # If we still overflow the model's context window, retry with a smaller budget and no memory injection.
                    if "context_length_exceeded" in str(e):
                        if debug:
                            print("[debug] context_length_exceeded: retrying with reduced budget")
                        reduced_budget = max(4096, int(REQUEST_BUDGET * 0.6))
                        input_items = _build_context_token_aware(
                            system_message=(
                                system_instructions
                                + "\n\nNote: context was trimmed aggressively due to context window limits."
                            ),
                            history_messages=history[-10:],
                            new_user_text=_truncate_text_to_tokens(text, max(256, reduced_budget // 4), enc),
                            budget_tokens=reduced_budget,
                            enc=enc,
                        )
                        final_text, tool_events, usage_stats = run_with_coordinator(
                            client=client,
                            model=MODEL,
                            tools_schema=tools_schema,
                            input_items=input_items,
                            conn=conn,
                            user_id=user_id,
                            agent=agent,
                            tool_required=tool_required,
                            task_type=task_type,
                            debug=debug,
                            call_tool_fn=call_tool,
                        )
                    else:
                        raise

                # Persist each sub-request as its own turn (keeps memory coherent)
                user_msg_id = add_message(conn, agent_convo, "user", f"{display_agent}: {text}")
                add_message(conn, agent_convo, "assistant", final_text or "")
                try:
                    record_turn_router_decision(
                        conn,
                        conversation_id=agent_convo,
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
                            conversation_id=agent_convo,
                            turn_id=user_msg_id,
                            agent=agent,
                            tool_calls=tool_events,
                        )
                    except Exception:
                        pass
                try:
                    record_turn_token_usage(
                        conn,
                        conversation_id=agent_convo,
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
                            conversation_id=agent_convo,
                            turn_id=user_msg_id,
                            agent=agent,
                            results=mem_results,
                            debug=mem_debug,
                        )
                    except Exception:
                        pass

                responses.append(f"[{display_agent}] {final_text}")

            print("\n".join(responses))
    finally:
        stop_event.set()
        watcher_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
