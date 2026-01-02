import argparse
import json
import threading
from datetime import datetime, timezone
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import re
load_dotenv()

from .config import (
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
from .db import (
    connect,
    init_db,
    create_conversation,
    list_conversations,
    add_message,
    get_recent_messages,
    get_latest_conversation_id,
    get_agent_conversation_id,
    set_agent_conversation_id,
    run_migrations,
    get_user_profile,
)
from .tool_loop import run_with_tools
from .tool_schemas import LIFE_TOOLS, HEALTH_TOOLS, DS_TOOLS, CODE_TOOLS
from .token_utils import try_get_encoding, count_message_tokens, token_len, truncate_to_tokens

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

def split_prefixed_requests(s: str):
    """
    Accepts inputs like:
      "life: list reminders; ds: next lesson"
    Returns list of (agent, text) preserving order.
    If no prefixes, returns [("life", s)].
    """
    import re

    pattern = re.compile(r"\b(life|health|ds|code):", re.IGNORECASE)
    matches = list(pattern.finditer(s))
    if not matches:
        return [("life", s.strip())] if s.strip() else []

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
    for r in results:
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


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so", "to", "of", "in", "on", "at", "for", "from",
    "with", "without", "is", "are", "was", "were", "be", "been", "being", "i", "you", "we", "they", "he", "she",
    "it", "this", "that", "these", "those", "my", "your", "our", "their", "me", "him", "her", "them", "as",
}


def _query_variants(text: str, enc) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    variants: list[str] = [raw]

    # Shortened variant (first ~80 tokens) to avoid over-specificity.
    if enc is not None:
        toks = enc.encode(raw)
        if len(toks) > 80:
            variants.append(enc.decode(toks[:80]))
    else:
        if len(raw) > 400:
            variants.append(raw[:400])

    # Keyword variant for better FTS recall.
    words = re.findall(r"[A-Za-z0-9_']{3,}", raw.lower())
    seen: set[str] = set()
    kept: list[str] = []
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

    # Deduplicate while preserving order.
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in out:
            out.append(v)
    return out


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
        "SELECT id, title, due_at, due_at_utc FROM reminders "
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

    client = OpenAI()
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
    print("Commands: /new, /list, /use <id>, /exit")

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
                print("  ds: create a short course on feature engineering for beginners")
                print("  ds: next lesson for my course_id")
                continue

            if user_text.startswith("/use "):
                convo_id = user_text.split(" ", 1)[1].strip()
                print(f"Conversation: {convo_id}")
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
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                    )
                elif agent == "health":
                    system_instructions = (
                        f"You are the Health assistant. User timezone: {tz}. "
                        "You are not a doctor; give general, evidence-based guidance and encourage professional help for urgent symptoms. "
                        "If you schedule reminders, always output due_at as ISO 8601 with timezone offset. "
                        "If the user explicitly asks to save/update stable facts (timezone, conditions, meds, preferences, goals), call set_profile."
                    )
                elif agent == "code":
                    system_instructions = (
                        "You are the Coding assistant. Help with debugging, architecture, and implementation details. "
                        "When useful, log progress with code_record_progress and review history with code_list_progress. "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
                    )
                else:
                    system_instructions = (
                        "You are a Data Science Course Assistant. You can design short courses, serve lessons, grade submissions, and adapt the plan based on progress. "
                        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile."
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
                try:
                    from app.tools import memory_search_graph_tool

                    mem_results: list[dict] = []
                    seen_nodes: set[str] = set()
                    variants = _query_variants(text, enc)
                    agent_tags = [agent] if agent in ("life", "health", "ds", "code") else ["general"]

                    # Keep this bounded; don't spam the API.
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
                    mem_block = _truncate_text_to_tokens(mem_block, MEMORY_INJECT_MAX_TOKENS, enc)
                    if mem_block:
                        system_instructions = (
                            system_instructions
                            + "\n\nRelevant past context (from ChatGPT export memory; may be partial). "
                            + "Only claim you found something if it appears below:\n"
                            + mem_block
                        )
                except Exception as e:
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

                try:
                    final_text, _ = run_with_tools(
                        client=client,
                        model=MODEL,
                        tools_schema=tools_schema,
                        input_items=input_items,
                        conn=conn,
                        user_id=user_id,
                        debug=debug,
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
                        final_text, _ = run_with_tools(
                            client=client,
                            model=MODEL,
                            tools_schema=tools_schema,
                            input_items=input_items,
                            conn=conn,
                            user_id=user_id,
                            debug=debug,
                        )
                    else:
                        raise

                # Persist each sub-request as its own turn (keeps memory coherent)
                add_message(conn, agent_convo, "user", f"{display_agent}: {text}")
                add_message(conn, agent_convo, "assistant", final_text or "")

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
