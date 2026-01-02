#!/usr/bin/env python3
"""
Quick evaluation harness for memory retrieval quality.

Usage:
  python scripts/eval_memory.py --queries queries.txt

queries.txt can be:
  - one query per line, optionally prefixed with "life:", "health:", "ds:", "code:"
  - blank lines / lines starting with "#" are ignored

Or JSON:
  - ["query1", "query2", ...]
  - [{"agent":"ds","query":"..."}, ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.tools import memory_search_graph_tool  # noqa: E402


def _default_context(agent: str) -> tuple[int, int]:
    if agent in ("life", "health"):
        return 4, 2
    if agent in ("ds", "code"):
        return 6, 2
    return 6, 2


def _load_queries(path: Optional[Path], default_agent: str) -> List[Dict[str, str]]:
    if not path:
        return [
            {"agent": "ds", "query": "pandas multiindex merge suffixes"},
            {"agent": "life", "query": "sleep routine schedule"},
            {"agent": "health", "query": "halitosis breath routine"},
            {"agent": "code", "query": "sqlite fts5 bm25 query"},
        ]

    if not path.exists():
        raise SystemExit(f"Queries file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        out: List[Dict[str, str]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    out.append({"agent": default_agent, "query": item})
                elif isinstance(item, dict):
                    q = item.get("query") or item.get("text") or ""
                    a = (item.get("agent") or default_agent).strip()
                    if q:
                        out.append({"agent": a, "query": str(q)})
        return out

    out: List[Dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(life|health|ds|code|general):\s*(.+)$", line, re.IGNORECASE)
        if m:
            out.append({"agent": m.group(1).lower(), "query": m.group(2).strip()})
        else:
            out.append({"agent": default_agent, "query": line})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("data/assistant.sqlite3"))
    ap.add_argument("--queries", type=Path, default=None, help="Path to queries.txt or queries.json")
    ap.add_argument("--agent", type=str, default="ds", choices=["life", "health", "ds", "code", "general"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--candidate-limit", type=int, default=250)
    ap.add_argument("--context-up", type=int, default=0, help="0 = use per-agent default")
    ap.add_argument("--context-down", type=int, default=0, help="0 = use per-agent default")
    ap.add_argument("--no-embeddings", action="store_true", help="Disable embeddings rerank")
    ap.add_argument("--debug", action="store_true", help="Print scoring/debug info")
    args = ap.parse_args()

    items = _load_queries(args.queries, args.agent)
    if not items:
        print("No queries.")
        return

    conn = sqlite3.connect(str(args.db))
    try:
        for item in items:
            agent = item["agent"]
            query = item["query"]
            up, down = _default_context(agent)
            if args.context_up > 0:
                up = args.context_up
            if args.context_down > 0:
                down = args.context_down

            res: Dict[str, Any] = memory_search_graph_tool(
                conn,
                query=query,
                agent=agent,
                k=args.k,
                candidate_limit=args.candidate_limit,
                context_up=up,
                context_down=down,
                use_embeddings=not args.no_embeddings,
                debug=args.debug,
            )

            print(f"\n=== [{agent}] {query}")
            if args.debug and isinstance(res.get("debug"), dict):
                dbg = res["debug"]
                qd = dbg.get("query", {})
                print(f"cleaned={qd.get('cleaned')} keywords={qd.get('keywords')} candidates={dbg.get('candidates')} used_embeddings={dbg.get('used_embeddings')}")
                top_scores = dbg.get("top_scores") or []
                for row in top_scores[:5]:
                    print(f"score={row.get('score'):.3f} fts={row.get('fts'):.3f} rec={row.get('recency'):.3f} cos={row.get('cosine')}  {row.get('title')}")

            results = res.get("results") or []
            for i, r in enumerate(results, start=1):
                title = r.get("title") or "(untitled)"
                node_id = r.get("node_id")
                ctx = (r.get("context") or "").strip()
                ctx = ctx[:500] + ("…" if len(ctx) > 500 else "")
                print(f"\n{i}. {title}  ({node_id})\n{ctx}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

