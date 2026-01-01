#!/usr/bin/env python3
"""
Backfill embeddings for ChatGPT export memory:
- Picks message nodes present in chatgpt_nodes_fts but missing in chatgpt_node_embeddings
- Calls OpenAI embeddings in batches and stores float32 blobs
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from openai import OpenAI
import tiktoken

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "assistant.sqlite3"
DEFAULT_MODEL = "text-embedding-3-small"
MAX_TOKENS_PER_REQUEST = 8000
MAX_TOKENS_PER_TEXT = 8000
ENCODING = tiktoken.get_encoding("cl100k_base")


def ensure_tables(conn: sqlite3.Connection) -> None:
    required = {"chatgpt_nodes", "chatgpt_nodes_fts", "chatgpt_node_embeddings"}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view','virtual table')"
    ).fetchall()
    existing = {r[0] for r in rows}
    missing = required - existing
    if missing:
        raise RuntimeError(f"Missing tables (apply schema.sql first): {sorted(missing)}")


def fetch_pending(conn: sqlite3.Connection, limit: int | None) -> List[Dict[str, Any]]:
    sql = """
    SELECT n.node_id, f.text
    FROM chatgpt_nodes n
    JOIN chatgpt_nodes_fts f ON f.node_id = n.node_id
    LEFT JOIN chatgpt_node_embeddings e ON e.node_id = n.node_id
    WHERE n.is_message = 1
      AND e.node_id IS NULL
      AND TRIM(COALESCE(f.text,'')) != ''
    ORDER BY n.create_time DESC
    """
    params: Sequence[Any] = ()
    if limit and limit > 0:
        sql += " LIMIT ?"
        params = (limit,)
    rows = conn.execute(sql, params).fetchall()
    return [{"node_id": r[0], "text": r[1]} for r in rows]


def embed_batch(client: OpenAI, texts: List[str], model: str, retries: int = 5, base_delay: float = 0.5) -> List[List[float]]:
    delay = base_delay
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == retries - 1:
                raise
        time.sleep(delay)
        delay *= 2
    return []

def _token_len(text: str) -> int:
    return len(ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    toks = ENCODING.encode(text)
    if len(toks) <= max_tokens:
        return text
    return ENCODING.decode(toks[:max_tokens]) + "\n\n[truncated]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of nodes to process (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds to pause between batches")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        ensure_tables(conn)
        limit = None if args.limit is None or args.limit <= 0 else args.limit
        pending = fetch_pending(conn, limit)
        total = len(pending)
        if total == 0:
            print("No pending embeddings.")
            return

        client = OpenAI()
        processed = 0

        while pending:
            batch = []
            batch_tokens = 0
            while pending and len(batch) < args.batch:
                item = pending[0]
                text = (item.get("text") or "").strip()
                if not text:
                    pending.pop(0)
                    continue
                text = _truncate_to_tokens(text, MAX_TOKENS_PER_TEXT)
                tlen = _token_len(text)
                if batch and batch_tokens + tlen > MAX_TOKENS_PER_REQUEST:
                    break
                batch.append({"node_id": item["node_id"], "text": text})
                batch_tokens += tlen
                pending.pop(0)

            if not batch:
                # If a single item is too large, truncate harder and continue.
                item = pending.pop(0)
                text = _truncate_to_tokens((item.get("text") or ""), MAX_TOKENS_PER_TEXT)
                batch = [{"node_id": item["node_id"], "text": text}]

            texts = [b["text"] for b in batch]
            ids = [b["node_id"] for b in batch]

            vectors = embed_batch(client, texts, model=args.model, retries=5, base_delay=0.5)
            if len(vectors) != len(ids):
                raise RuntimeError("Embedding response size mismatch.")

            to_insert = []
            for nid, vec in zip(ids, vectors):
                arr = np.array(vec, dtype=np.float32)
                dim = int(arr.size)
                to_insert.append((nid, dim, arr.tobytes()))

            conn.executemany(
                """
                INSERT OR REPLACE INTO chatgpt_node_embeddings (node_id, dim, vec)
                VALUES (?, ?, ?)
                """,
                to_insert,
            )
            conn.commit()

            processed += len(ids)
            remaining = len(pending)
            print(f"Processed {processed}/{total} (remaining {remaining})")
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
