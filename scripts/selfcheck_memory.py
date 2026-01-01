#!/usr/bin/env python3
"""
Lightweight self-check for ChatGPT export memory:
- Optionally imports a conversations.json
- Backfills a small batch of embeddings
- Runs a sample FTS query and a memory_search_graph_tool call
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backfill_embeddings
import import_chatgpt_export
from app.config import EMBEDDING_MODEL
from app.tools import memory_search_graph_tool


def _insert_embeddings(conn: sqlite3.Connection, rows, model: str, batch_size: int) -> int:
    if not rows:
        return 0
    client = OpenAI()
    batch = rows[:batch_size]
    texts = [r["text"] for r in batch if r.get("text")]
    ids = [r["node_id"] for r in batch if r.get("text")]
    if not ids:
        return 0

    vectors = backfill_embeddings.embed_batch(client, texts, model=model)
    to_insert = []
    for nid, vec in zip(ids, vectors):
        arr = np.array(vec, dtype=np.float32)
        to_insert.append((nid, int(arr.size), arr.tobytes()))

    conn.executemany(
        "INSERT OR REPLACE INTO chatgpt_node_embeddings (node_id, dim, vec) VALUES (?, ?, ?)",
        to_insert,
    )
    conn.commit()
    return len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data/assistant.sqlite3"))
    ap.add_argument("--export", type=Path, default=Path("conversations.json"))
    ap.add_argument("--model", type=str, default=EMBEDDING_MODEL)
    ap.add_argument("--batch", type=int, default=8, help="Mini batch size for the embedding spot-check")
    ap.add_argument("--limit", type=int, default=10, help="Max nodes to embed during the check")
    ap.add_argument("--agent", type=str, choices=["life", "ds", "general"], default=None,
                    help="Force import agent tag (optional)")
    ap.add_argument("--auto-agent", action="store_true", help="Infer agent tags during import")
    ap.add_argument("--query", type=str, default="python", help="Sample search query")
    ap.add_argument("--search-agent", type=str, default="ds", choices=["life", "ds", "general"])
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        print("1) Importing conversations.json (if present)")
        try:
            import_chatgpt_export.ensure_db(conn)
            if args.export.exists():
                import_chatgpt_export.import_export(conn, args.export, args.agent, args.auto_agent)
            else:
                print(f"   skip: {args.export} not found")
        except Exception as e:
            print(f"   import failed: {e}")

        print("2) Backfilling a small embedding batch")
        try:
            backfill_embeddings.ensure_tables(conn)
            pending = backfill_embeddings.fetch_pending(conn, args.limit)
            inserted = _insert_embeddings(conn, pending, args.model, args.batch)
            print(f"   embedded {inserted} nodes (of {len(pending)} pending)")
        except Exception as e:
            print(f"   embedding step failed: {e}")

        print("3) Counts check")
        try:
            row = conn.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM chatgpt_conversations),
                  (SELECT COUNT(*) FROM chatgpt_nodes),
                  (SELECT COUNT(*) FROM chatgpt_nodes_fts),
                  (SELECT COUNT(*) FROM chatgpt_node_embeddings)
                """
            ).fetchone()
            print(f"   conversations={row[0]} nodes={row[1]} fts={row[2]} embeddings={row[3]}")
        except Exception as e:
            print(f"   count query failed: {e}")

        print("4) Sample memory search")
        try:
            res = memory_search_graph_tool(
                conn,
                query=args.query,
                agent=args.search_agent,
                k=2,
                candidate_limit=50,
                context_up=4,
                context_down=2,
                use_embeddings=False,
            )
            print(f"   returned={res.get('returned')} first_context=")
            if res.get("results"):
                print(res["results"][0]["context"][:400])
        except Exception as e:
            print(f"   memory search failed: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
