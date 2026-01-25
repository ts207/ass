from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from app.db import (
    get_conversation_message_count,
    get_conversation_summary,
    get_messages_after_offset,
    upsert_conversation_summary,
)

SUMMARY_MIN_NEW_MESSAGES = 30
SUMMARY_CHUNK_SIZE = 20

_SUMMARY_FORCE_TRIGGERS = (
    "summarize this thread",
    "summarize the thread",
    "summarize this conversation",
    "summary of this thread",
    "thread summary",
)


def should_force_thread_summary(text: str) -> bool:
    t = (text or "").lower()
    return any(trigger in t for trigger in _SUMMARY_FORCE_TRIGGERS)


def build_thread_context(
    system_instructions: str,
    *,
    profile_blob: Optional[str] = None,
    summary_text: Optional[str] = None,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    user_text: str,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = [{"role": "system", "content": system_instructions}]
    if profile_blob:
        items.append(
            {
                "role": "system",
                "content": (
                    "User profile (stable facts, user-provided). "
                    "Use as ground truth; do not invent missing fields:\n"
                    + profile_blob
                ),
            }
        )
    if summary_text:
        items.append({"role": "system", "content": "Thread summary:\n" + summary_text})
    for msg in recent_messages or []:
        role = msg.get("role") or "assistant"
        content = msg.get("content") or ""
        items.append({"role": role, "content": content})
    items.append({"role": "user", "content": user_text})
    return items


def _chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    step = max(1, int(size or 1))
    for i in range(0, len(items), step):
        yield items[i : i + step]


def _format_messages(messages: List[Dict[str, Any]]) -> str:
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])


def _summarize_update(
    client: OpenAI,
    model: str,
    *,
    old_summary: str,
    new_messages: List[Dict[str, Any]],
) -> str:
    system_prompt = (
        "You maintain a compact durable summary of a conversation thread.\n"
        "Update the summary using:\n"
        "1) The existing summary (may be empty)\n"
        "2) The new messages since the last update\n\n"
        "Rules:\n"
        "- Output 1-2 paragraphs, ~120-250 words.\n"
        "- Preserve decisions, commitments, constraints, preferences, and current plan.\n"
        "- Include open questions / next steps if they matter for future turns.\n"
        "- Do not include verbose dialogue or timestamps unless critical.\n"
        "- Use plain text."
    )
    user_content = (
        "EXISTING SUMMARY:\n"
        + (old_summary or "(none)")
        + "\n\nNEW MESSAGES:\n"
        + _format_messages(new_messages)
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return (resp.output_text or "").strip()


def update_thread_summary(
    conn,
    client: OpenAI,
    model: str,
    *,
    conversation_id: str,
    user_id: str,
    agent: str,
    force: bool = False,
    min_new_messages: int = SUMMARY_MIN_NEW_MESSAGES,
    chunk_size: int = SUMMARY_CHUNK_SIZE,
) -> Optional[str]:
    row = get_conversation_summary(conn, conversation_id)
    prev_summary = row["summary"] if row else ""
    prev_count = int(row["message_count"]) if row else 0
    total_count = get_conversation_message_count(conn, conversation_id)
    if total_count <= 0:
        return None
    if not force and (total_count - prev_count) < int(min_new_messages or 0):
        return None
    new_messages = get_messages_after_offset(conn, conversation_id, prev_count)
    if not new_messages:
        return None

    summary = prev_summary
    for chunk in _chunked(new_messages, chunk_size):
        updated = _summarize_update(client, model, old_summary=summary, new_messages=chunk)
        if updated:
            summary = updated
        else:
            break

    last_message_id = new_messages[-1]["id"]
    upsert_conversation_summary(
        conn,
        conversation_id=conversation_id,
        user_id=user_id,
        agent=agent,
        summary=summary or prev_summary or "",
        message_count=total_count,
        last_message_id=last_message_id,
    )
    return summary
