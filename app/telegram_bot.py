"""
Lightweight Telegram bridge for the assistant.
- Uses long polling (no public webhook needed).
- Routes messages to the General agent, which can delegate via tools.
- Sends due reminders to Telegram if it knows the chat_id.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import (
    DB_PATH,
    MODEL,
    MODEL_CONTEXT_TOKENS,
    REQUEST_BUDGET_FRACTION,
    MEMORY_INJECT_MAX_TOKENS,
    MEMORY_INJECT_K,
    MEMORY_INJECT_CANDIDATE_LIMIT,
    PROFILE_INJECT_MAX_TOKENS,
)
from app.chat import (
    _build_context_token_aware,
    _ensure_budget_with_summary,
    _format_memory_results,
    _get_encoding,
    _should_retrieve_memory,
    _truncate_text_to_tokens,
    check_due_reminders,
    split_prefixed_requests,
)
from app.db import (
    add_message,
    connect,
    create_conversation,
    get_agent_conversation_id,
    get_user_profile,
    init_db,
    run_migrations,
    set_agent_conversation_id,
)
from app.tool_loop import run_with_tools
from app.tool_runtime import call_tool
from app.tool_schemas import GENERAL_TOOLS
from app.tools import memory_search_graph_tool
from app.tools_general import delegate_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
REQUEST_BUDGET = int(MODEL_CONTEXT_TOKENS * REQUEST_BUDGET_FRACTION)
ENCODING = _get_encoding()
OPENAI_CLIENT = OpenAI()
_DB_READY = False


def _init_db_if_needed():
    global _DB_READY
    if _DB_READY:
        return
    conn = connect(DB_PATH)
    try:
        init_db(conn, SCHEMA_PATH.read_text(encoding="utf-8"))
        run_migrations(conn)
        _DB_READY = True
    finally:
        conn.close()


def _parse_allowed_ids(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return {p for p in parts if p}


def _parse_default_chat_id(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid TELEGRAM_DEFAULT_CHAT_ID value %r; ignoring and treating as unset.",
            raw,
        )
    return None


def _chunk_message(text: str, *, limit: int = 3800) -> List[str]:
    if len(text) <= limit:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + limit])
        start += limit
    return out


def _general_system_instructions(profile: Dict[str, str]) -> str:
    tz = ""
    if isinstance(profile.get("timezone"), str):
        tz = profile["timezone"].strip()
    elif isinstance(profile.get("tz"), str):
        tz = profile["tz"].strip()
    if not tz:
        tz = "Asia/Ulaanbaatar"

    system_instructions = (
        "You are the Coordinator (General) agent. You are chatting with the user via Telegram; keep replies concise and actionable. "
        "Route specialized tasks to life/health/ds/code using delegate_agent. "
        "Use your own tools (web_search, fetch_url, extract_text, kb_search) for research and summaries. "
        "If a tool errors due to permissions, explain how to enable with permissions_set (mode, allow_network, allow_fs_write, allow_shell, allow_exec). "
        "If the user explicitly asks to save/update stable facts (timezone, preferences, goals), call set_profile. "
        f"User timezone: {tz}."
    )

    if profile:
        try:
            import json

            profile_blob = json.dumps(profile, ensure_ascii=False, indent=2)
            profile_blob = _truncate_text_to_tokens(profile_blob, PROFILE_INJECT_MAX_TOKENS, ENCODING)
            if profile_blob:
                system_instructions += (
                    "\n\nUser profile (stable facts, user-provided). Use as ground truth; do not invent missing fields:\n"
                    + profile_blob
                )
        except Exception:
            pass

    return system_instructions


def _run_general_agent(conn, *, user_id: str, text: str, debug: bool = False) -> str:
    agent = "general"
    agent_convo = get_agent_conversation_id(conn, user_id, agent)
    if not agent_convo:
        agent_convo = create_conversation(conn, user_id, title=f"{agent} thread")
        set_agent_conversation_id(conn, user_id, agent, agent_convo)

    profile = get_user_profile(conn, user_id)
    system_instructions = _general_system_instructions(profile or {})

    try:
        mem_results: List[Dict[str, str]] = []
        if _should_retrieve_memory(text):
            mem = memory_search_graph_tool(
                conn,
                query=text,
                agent=agent,
                k=MEMORY_INJECT_K,
                candidate_limit=MEMORY_INJECT_CANDIDATE_LIMIT,
                context_up=6,
                context_down=2,
                use_embeddings=True,
                debug=debug,
            )
            mem_results = mem.get("results", []) or []
            if debug and isinstance(mem.get("debug"), dict):
                logger.info("memory debug: %s", mem["debug"])

        mem_block = _format_memory_results(mem_results)
        mem_block = _truncate_text_to_tokens(mem_block, MEMORY_INJECT_MAX_TOKENS, ENCODING)
        if mem_block:
            system_instructions = (
                system_instructions
                + "\n\nRelevant past context (from ChatGPT export memory; may be partial). "
                + "Only claim you found something if it appears below:\n"
                + mem_block
            )
    except Exception as e:
        if debug:
            logger.warning("memory injection failed: %s", e)

    history = _ensure_budget_with_summary(
        conn,
        OPENAI_CLIENT,
        MODEL,
        agent_convo,
        system_instructions,
        REQUEST_BUDGET,
        ENCODING,
        user_id,
    )

    max_user_tokens = max(256, REQUEST_BUDGET // 4)
    text_for_model = _truncate_text_to_tokens(text, max_user_tokens, ENCODING)

    input_items = _build_context_token_aware(
        system_message=system_instructions,
        history_messages=history,
        new_user_text=text_for_model,
        budget_tokens=REQUEST_BUDGET,
        enc=ENCODING,
    )

    try:
        final_text, _ = run_with_tools(
            client=OPENAI_CLIENT,
            model=MODEL,
            tools_schema=GENERAL_TOOLS,
            input_items=input_items,
            conn=conn,
            user_id=user_id,
            debug=debug,
            call_tool_fn=call_tool,
        )
    except Exception as e:
        if "context_length_exceeded" in str(e):
            reduced_budget = max(4096, int(REQUEST_BUDGET * 0.6))
            input_items = _build_context_token_aware(
                system_message=system_instructions + "\n\nNote: context was trimmed due to token limits.",
                history_messages=history[-10:],
                new_user_text=_truncate_text_to_tokens(text, max(256, reduced_budget // 4), ENCODING),
                budget_tokens=reduced_budget,
                enc=ENCODING,
            )
            final_text, _ = run_with_tools(
                client=OPENAI_CLIENT,
                model=MODEL,
                tools_schema=GENERAL_TOOLS,
                input_items=input_items,
                conn=conn,
                user_id=user_id,
                debug=debug,
                call_tool_fn=call_tool,
            )
        else:
            raise

    add_message(conn, agent_convo, "user", f"{agent}: {text}")
    add_message(conn, agent_convo, "assistant", final_text or "")

    return final_text


def handle_user_text(user_id: str, text: str, *, debug: bool = False) -> str:
    _init_db_if_needed()
    conn = connect(DB_PATH, check_same_thread=False)
    try:
        requests = split_prefixed_requests(text)
        responses: List[str] = []
        for agent, chunk in requests:
            chunk = chunk.strip()
            if not chunk:
                continue
            if agent == "general":
                resp = _run_general_agent(conn, user_id=user_id, text=chunk, debug=debug)
            else:
                res = delegate_agent(
                    conn,
                    user_id=user_id,
                    agent=agent,
                    task=chunk,
                    call_tool_fn=call_tool,
                    include_history=True,
                    history_limit=12,
                )
                resp = res.get("response") if isinstance(res, dict) else str(res)
            responses.append(f"[{agent}] {resp}")

        return "\n".join(responses) if responses else "(no response)"
    finally:
        conn.close()


async def _send_reply(update: Update, text: str):
    if not update.effective_chat:
        return
    for chunk in _chunk_message(text or "(empty)"):
        try:
            await update.effective_chat.send_message(chunk, disable_web_page_preview=True)
        except Exception as e:
            logger.error("failed to send chunk: %s", e)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_chat or not update.effective_user:
        return

    allowed: Set[str] = context.application.bot_data.get("allowed_ids", set())
    user_id = str(update.effective_user.id)
    if allowed and user_id not in allowed:
        await _send_reply(update, "Access denied for this bot. Ask the owner to whitelist your user id.")
        return

    context.application.bot_data.setdefault("user_chat_ids", {})[user_id] = update.effective_chat.id

    text = update.effective_message.text or update.effective_message.caption or ""
    if not text.strip():
        await _send_reply(update, "Send text to chat with the assistant.")
        return

    loop = asyncio.get_running_loop()
    try:
        reply = await loop.run_in_executor(None, handle_user_text, user_id, text, False)
    except Exception as e:
        logger.exception("handle_user_text failed")
        reply = f"Error: {e}"

    await _send_reply(update, reply)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user and update.effective_chat:
        context.application.bot_data.setdefault("user_chat_ids", {})[str(update.effective_user.id)] = update.effective_chat.id
    await _send_reply(
        update,
        "Hi! I am your assistant on Telegram.\n\n"
        "Just send messages like:\n"
        "general: plan my week\n"
        "life: remind me at 9am to take meds\n"
        "code: help me debug this traceback\n",
    )


async def reminder_job(context: ContextTypes.DEFAULT_TYPE):
    _init_db_if_needed()
    conn = connect(DB_PATH, check_same_thread=False)
    try:
        due = check_due_reminders(conn, debug=False)
    finally:
        conn.close()

    if not due:
        return

    chat_map: Dict[str, int] = context.application.bot_data.get("user_chat_ids", {}) or {}
    default_chat = context.application.bot_data.get("default_chat_id")
    for d in due:
        chat_id = chat_map.get(str(d.get("user_id"))) or default_chat
        if not chat_id:
            continue
        title = d.get("title") or "Reminder"
        due_at = d.get("due_at") or ""
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"⏰ Reminder: {title} (due {due_at})")
        except Exception as e:
            logger.error("failed to send reminder to %s: %s", chat_id, e)


def main():
    _init_db_if_needed()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in your environment or .env file.")

    allowed_ids = _parse_allowed_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS"))
    default_chat_val = _parse_default_chat_id(os.getenv("TELEGRAM_DEFAULT_CHAT_ID"))

    app = Application.builder().token(token).build()
    app.bot_data["allowed_ids"] = allowed_ids
    app.bot_data["default_chat_id"] = default_chat_val
    app.bot_data["user_chat_ids"] = {}

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.job_queue.run_repeating(reminder_job, interval=30, first=10)

    logger.info("Starting Telegram bot (long polling). Allowed IDs: %s", allowed_ids or "any")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
