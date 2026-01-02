#!/usr/bin/env python3
"""
Import ChatGPT export conversations.json into SQLite:
- chatgpt_conversations
- chatgpt_nodes (graph, lossless)
- chatgpt_nodes_fts (FTS index for retrieval)

Assumes your schema.sql already created these tables.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "assistant.sqlite3"
DEFAULT_EXPORT = REPO_ROOT / "conversations.json"
DEFAULT_HTML = REPO_ROOT / "chat.html"


# --------- Text extraction (handles common export variants) ---------

def extract_text(message: Dict[str, Any]) -> str:
    """Best-effort extraction of text from a ChatGPT export 'message' object."""
    if not message:
        return ""

    content = message.get("content") or {}
    ctype = content.get("content_type")

    # Typical: {"content_type":"text","parts":[...]}
    parts = content.get("parts")
    if isinstance(parts, list) and parts:
        # parts can include non-strings; stringify safely
        return "\n".join(str(p) for p in parts if p is not None).strip()

    # Some exports: {"text": "..."} or similar
    if isinstance(content.get("text"), str):
        return content["text"].strip()

    # Fallback: try common nested keys
    if ctype == "multimodal_text" and isinstance(content.get("parts"), list):
        return "\n".join(str(p) for p in content["parts"] if p is not None).strip()

    return ""


def normalize_conversations(data: Any) -> List[Dict[str, Any]]:
    """ChatGPT exports sometimes are a list; sometimes wrapped in a dict."""
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if isinstance(data, dict):
        for key in ("conversations", "items", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return [c for c in v if isinstance(c, dict)]
        # Sometimes a single conversation dict is provided
        if "mapping" in data and isinstance(data["mapping"], dict):
            return [data]
    return []


def _load_export_data(export_path: Path) -> Any:
    """
    Load either conversations.json or chat.html (jsonData in <script>).
    """
    if export_path.suffix.lower() in (".html", ".htm"):
        text = export_path.read_text(encoding="utf-8")
        marker = "var jsonData = "
        start = text.find(marker)
        if start == -1:
            raise RuntimeError("Could not find jsonData in HTML export.")
        start += len(marker)
        # Find the end of the JSON array/object by bracket matching.
        i = start
        n = len(text)
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] not in "[{":
            raise RuntimeError("Invalid jsonData start in HTML export.")

        depth = 0
        in_str = False
        escape = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    json_str = text[i : j + 1].strip()
                    return json.loads(json_str)
        raise RuntimeError("Could not parse jsonData from HTML export.")
    return json.loads(export_path.read_text(encoding="utf-8"))


# --------- Agent tagging (optional) ---------

def auto_agent(text: str) -> str:
    t = text.lower()
    life_keys = (
        "routine", "reminder", "sleep", "diet", "health", "halitosis", "breath",
        "exercise", "anxiety", "stress", "doctor", "symptom"
    )
    ds_keys = (
        "python", "statistics", "machine learning", "ml", "data science",
        "model", "regression", "classification", "pandas", "numpy", "sklearn"
    )
    if any(k in t for k in life_keys):
        return "life"
    if any(k in t for k in ds_keys):
        return "ds"
    return "general"


# --------- Mainline computation (graph faithful) ---------

def compute_subtree_scores(
    mapping: Dict[str, Any],
    node_is_message: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, Optional[str]]]:
    """
    For each node, compute:
    - score = number of message-nodes in its subtree (including itself if message)
    - main_child_id = child with highest score (tie-break by child create_time, then id)
    """
    memo_score: Dict[str, int] = {}
    memo_main: Dict[str, Optional[str]] = {}

    def node_create_time(nid: str) -> int:
        m = (mapping.get(nid) or {}).get("message") or {}
        return int(m.get("create_time") or 0)

    def dfs(nid: str) -> int:
        if nid in memo_score:
            return memo_score[nid]

        node = mapping.get(nid) or {}
        children = node.get("children") or []
        children = [c for c in children if isinstance(c, str)]

        best_child = None
        best_child_score = -1
        best_child_ct = -1

        total = node_is_message.get(nid, 0)
        for c in children:
            s = dfs(c)
            total += s

            ct = node_create_time(c)
            if (s > best_child_score) or (s == best_child_score and ct > best_child_ct) or (
                s == best_child_score and ct == best_child_ct and str(c) > str(best_child)
            ):
                best_child = c
                best_child_score = s
                best_child_ct = ct

        memo_score[nid] = total
        memo_main[nid] = best_child
        return total

    # Ensure all nodes are visited
    for nid in mapping.keys():
        dfs(nid)

    return memo_score, memo_main


# --------- DB helpers ---------

def ensure_db(conn: sqlite3.Connection) -> None:
    # Fail fast if schema not applied
    required = {
        "chatgpt_conversations",
        "chatgpt_nodes",
        "chatgpt_nodes_fts",
        "chatgpt_node_embeddings",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view','virtual table')"
    ).fetchall()
    existing = {r[0] for r in rows}
    missing = required - existing
    if missing:
        raise RuntimeError(f"Missing tables (apply schema.sql first): {sorted(missing)}")


def import_export(
    conn: sqlite3.Connection,
    export_path: Path,
    force_agent: Optional[str],
    use_auto_agent: bool,
) -> None:
    data = _load_export_data(export_path)
    conversations = normalize_conversations(data)

    if not conversations:
        raise RuntimeError("No conversations found in export (unexpected JSON structure).")

    cur = conn.cursor()

    conv_count = 0
    node_count = 0
    msg_node_count = 0
    fts_count = 0
    inserted_fts: set[str] = set()

    # One transaction for speed
    cur.execute("BEGIN")
    # FTS5 tables don't enforce uniqueness on node_id; repeated imports will accumulate duplicates.
    # Rebuild the FTS index each run to keep retrieval accurate and fast.
    cur.execute("DELETE FROM chatgpt_nodes_fts")

    for conv in conversations:
        conv_id = conv.get("id")
        mapping = conv.get("mapping")
        if not isinstance(conv_id, str) or not isinstance(mapping, dict):
            continue

        title = conv.get("title")
        create_time = conv.get("create_time")
        update_time = conv.get("update_time")

        cur.execute(
            """
            INSERT INTO chatgpt_conversations (id, title, create_time, update_time)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              create_time=excluded.create_time,
              update_time=excluded.update_time
            """,
            (conv_id, title, create_time, update_time),
        )
        conv_count += 1

        # Precompute is_message and extracted text
        node_is_message: Dict[str, int] = {}
        node_text: Dict[str, str] = {}
        node_role: Dict[str, str] = {}
        node_ct: Dict[str, int] = {}

        for nid, node in mapping.items():
            if not isinstance(nid, str) or not isinstance(node, dict):
                continue

            message = node.get("message")
            if not isinstance(message, dict):
                node_is_message[nid] = 0
                node_text[nid] = ""
                node_role[nid] = ""
                node_ct[nid] = 0
                continue

            role = ((message.get("author") or {}).get("role") or "").strip()
            text = extract_text(message)
            ct = int(message.get("create_time") or 0)

            is_msg = 1 if (text.strip() != "" and role != "") else 0
            node_is_message[nid] = is_msg
            node_text[nid] = text
            node_role[nid] = role
            node_ct[nid] = ct

        # Compute mainline child pointers (faithful traversal)
        _, main_child = compute_subtree_scores(mapping, node_is_message)

        # Insert nodes + FTS
        for nid, node in mapping.items():
            if not isinstance(nid, str) or not isinstance(node, dict):
                continue

            parent_id = node.get("parent")
            if parent_id is not None and not isinstance(parent_id, str):
                parent_id = None

            role = node_role.get(nid, "")
            text = node_text.get(nid, "")
            ct = node_ct.get(nid, 0)
            is_msg = node_is_message.get(nid, 0)
            mc = main_child.get(nid)

            if force_agent:
                agent = force_agent
            elif use_auto_agent:
                agent = auto_agent(text)
            else:
                agent = "general"

            cur.execute(
                """
                INSERT INTO chatgpt_nodes
                  (node_id, conversation_id, parent_id, role, text, create_time, is_message, main_child_id, agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                  conversation_id=excluded.conversation_id,
                  parent_id=excluded.parent_id,
                  role=excluded.role,
                  text=excluded.text,
                  create_time=excluded.create_time,
                  is_message=excluded.is_message,
                  main_child_id=excluded.main_child_id,
                  agent=excluded.agent
                """,
                (nid, conv_id, parent_id, role, text, ct, is_msg, mc, agent),
            )
            node_count += 1
            if is_msg:
                msg_node_count += 1
                # Index only real message nodes (reduces noise)
                # FTS5 doesn't enforce uniqueness on node_id; handle rare collisions defensively.
                if nid in inserted_fts:
                    cur.execute("DELETE FROM chatgpt_nodes_fts WHERE node_id = ?", (nid,))
                cur.execute(
                    """
                    INSERT INTO chatgpt_nodes_fts
                      (node_id, conversation_id, title, agent, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (nid, conv_id, title, agent, text),
                )
                inserted_fts.add(nid)
                fts_count += 1

    cur.execute("COMMIT")

    # Print final counts from DB (more reliable across re-imports and node_id collisions).
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM chatgpt_conversations),
          (SELECT COUNT(*) FROM chatgpt_nodes),
          (SELECT COUNT(*) FROM chatgpt_nodes WHERE is_message=1),
          (SELECT COUNT(*) FROM chatgpt_nodes_fts)
        """
    ).fetchone()
    print("IMPORT COMPLETE")
    print(f"  conversations: {row[0]}")
    print(f"  nodes total:   {row[1]}")
    print(f"  msg nodes:     {row[2]}")
    print(f"  fts rows:      {row[3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT,
                    help="Path to conversations.json or chat.html (HTML export with jsonData).")
    ap.add_argument("--agent", type=str, choices=["life", "ds", "general"], default=None,
                    help="Force all imported nodes to one agent tag")
    ap.add_argument("--auto-agent", "--autoagent", dest="auto_agent", action="store_true",
                    help="Heuristically tag each node as life/ds/general based on text")
    args = ap.parse_args()

    if not args.export.exists():
        if args.export == DEFAULT_EXPORT and DEFAULT_HTML.exists():
            args.export = DEFAULT_HTML
            print(f"[info] conversations.json not found; using HTML export at {args.export}")
        else:
            raise SystemExit(f"Export not found: {args.export} (place conversations.json or chat.html in repo root or pass --export)")

    conn = sqlite3.connect(args.db)
    try:
        ensure_db(conn)
        import_export(conn, args.export, args.agent, args.auto_agent)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
