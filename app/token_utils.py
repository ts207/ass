from __future__ import annotations

from typing import Any, Dict, Iterable


def try_get_encoding():
    """
    Best-effort tokenizer for approximate budgeting.
    - If `tiktoken` is installed, uses `cl100k_base`.
    - Otherwise falls back to UTF-8 byte length (safe but conservative).
    """
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def token_len(text: str, enc=None) -> int:
    if text is None:
        text = ""
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.encode("utf-8"))


def count_message_tokens(messages: Iterable[Dict[str, Any]], enc=None) -> int:
    total = 0
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        total += token_len(role, enc) + token_len(content, enc)
    return total


def truncate_to_tokens(text: str, max_tokens: int, enc=None, *, suffix: str = "\n\n[truncated]") -> str:
    if text is None:
        text = ""
    if max_tokens <= 0:
        return ""

    if enc is not None:
        try:
            toks = enc.encode(text)
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens]) + suffix
        except Exception:
            pass

    data = text.encode("utf-8")
    if len(data) <= max_tokens:
        return text
    trimmed = data[:max_tokens].decode("utf-8", errors="ignore")
    return trimmed + suffix
