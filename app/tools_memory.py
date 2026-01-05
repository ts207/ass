import hashlib
import math
import re
import time
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from app.config import EMBEDDING_MODEL
from app.db import (
    chatgpt_context_window,
    chatgpt_fts_candidates,
    chatgpt_get_conversation_title,
    chatgpt_get_embeddings,
)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


_openai_client: OpenAI | None = None
_embed_cache: dict[str, np.ndarray] = {}
_EMBED_CACHE_MAX = 64


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _embed_query(text: str) -> np.ndarray:
    key = f"{EMBEDDING_MODEL}:{text}"
    cached = _embed_cache.get(key)
    if cached is not None:
        return cached

    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    vec = resp.data[0].embedding
    arr = np.array(vec, dtype=np.float32)
    _embed_cache[key] = arr
    if len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.pop(next(iter(_embed_cache)))
    return arr


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "so", "to", "of", "in", "on", "at", "for", "from",
    "with", "without", "is", "are", "was", "were", "be", "been", "being", "i", "you", "we", "they", "he", "she",
    "it", "this", "that", "these", "those", "my", "your", "our", "their", "me", "him", "her", "them", "as",
}


def _rewrite_retrieval_query(query: str, *, max_terms: int = 6) -> Dict[str, Any]:
    raw = (query or "").strip()
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
        if len(kept) >= max_terms:
            break
    cleaned = " ".join(kept) if kept else raw
    return {"raw": raw, "cleaned": cleaned, "keywords": kept}


def _recency_boost(create_time: int, *, half_life_days: float = 180.0) -> float:
    if not create_time or create_time <= 0:
        return 0.0
    age_days = max(0.0, (time.time() - float(create_time)) / 86400.0)
    return float(math.exp(-age_days / half_life_days))


def _fts_score_from_bm25(bm25_val: float) -> float:
    raw = max(0.0, -float(bm25_val))
    return float(raw / (1.0 + raw))


def memory_search_graph_tool(
    conn,
    query: str,
    agent: str,
    k: int = 5,
    candidate_limit: int = 250,
    context_up: int = 6,
    context_down: int = 2,
    use_embeddings: bool = True,
    debug: bool = False,
):
    agent_tag = agent
    if agent_tag == "health":
        agent_tag = "life"
    elif agent_tag == "code":
        agent_tag = "ds"

    qinfo = _rewrite_retrieval_query(query)
    cleaned_query = qinfo["cleaned"] or qinfo["raw"]

    cands = chatgpt_fts_candidates(conn, query=cleaned_query, agent=agent_tag, limit=candidate_limit, mode="and")
    used_fts_query = cleaned_query
    used_fts_mode = "and"
    if not cands and qinfo.get("keywords"):
        cands = chatgpt_fts_candidates(conn, query=cleaned_query, agent=agent_tag, limit=candidate_limit, mode="or")
        used_fts_mode = "or"
    if not cands and qinfo.get("keywords"):
        kws = qinfo["keywords"]
        for n in (3, 2, 1):
            if len(kws) >= n:
                q2 = " ".join(kws[:n]).strip()
                if q2 and q2 != used_fts_query:
                    cands = chatgpt_fts_candidates(conn, query=q2, agent=agent_tag, limit=candidate_limit, mode="and")
                    used_fts_query = q2
                    used_fts_mode = "and"
                    if cands:
                        break
        if not cands and qinfo.get("raw") and qinfo["raw"] != used_fts_query:
            cands = chatgpt_fts_candidates(conn, query=qinfo["raw"], agent=agent_tag, limit=candidate_limit, mode="or")
            used_fts_query = qinfo["raw"]
            used_fts_mode = "or"

    reranked = cands
    debug_scores: list[Dict[str, Any]] = []
    try:
        query_vec = _embed_query(query) if use_embeddings else None
        emb_map = chatgpt_get_embeddings(conn, [c["node_id"] for c in cands]) if use_embeddings else {}
        scored_with_emb: List[tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
        scored_no_emb: List[tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []

        for idx, c in enumerate(cands):
            bm25_val = float(c.get("bm25") or 0.0)
            fts_s = _fts_score_from_bm25(bm25_val)
            title_boost = 0.0
            if qinfo.get("keywords") and c.get("title"):
                title_l = str(c.get("title") or "").lower()
                if title_l:
                    hits = sum(1 for kw in qinfo["keywords"] if kw in title_l)
                    if hits > 0:
                        title_boost = hits / max(1, len(qinfo["keywords"]))
                        fts_s = min(1.0, fts_s + 0.15 * title_boost)
            rec_s = _recency_boost(int(c.get("create_time") or 0))

            emb = emb_map.get(c["node_id"]) if emb_map else None
            cos = 0.0
            has_emb = emb is not None and emb.size > 0 and query_vec is not None
            if has_emb:
                cos = max(0.0, _cosine_similarity(query_vec, emb))

            if has_emb:
                score = 0.65 * cos + 0.25 * fts_s + 0.10 * rec_s
                scored_with_emb.append(
                    (
                        score,
                        idx,
                        c,
                        {
                            "cosine": cos,
                            "fts": fts_s,
                            "recency": rec_s,
                            "bm25": bm25_val,
                            "title_boost": title_boost,
                        },
                    )
                )
            else:
                score = 0.25 * fts_s + 0.10 * rec_s
                scored_no_emb.append(
                    (
                        score,
                        idx,
                        c,
                        {
                            "cosine": None,
                            "fts": fts_s,
                            "recency": rec_s,
                            "bm25": bm25_val,
                            "title_boost": title_boost,
                        },
                    )
                )

        if scored_with_emb:
            scored_with_emb.sort(key=lambda x: x[0], reverse=True)
            scored_no_emb.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            reranked = [c for _, _, c, _ in scored_with_emb] + [c for _, _, c, _ in scored_no_emb]
            if debug:
                debug_scores = [
                    {
                        "node_id": c["node_id"],
                        "conversation_id": c.get("conversation_id"),
                        "title": c.get("title", ""),
                        "score": float(score),
                        **parts,
                    }
                    for score, _, c, parts in (scored_with_emb[:10] + scored_no_emb[:5])
                ]
        else:
            scored_no_emb.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            reranked = [c for _, _, c, _ in scored_no_emb]
            if debug:
                debug_scores = [
                    {
                        "node_id": c["node_id"],
                        "conversation_id": c.get("conversation_id"),
                        "title": c.get("title", ""),
                        "score": float(score),
                        **parts,
                    }
                    for score, _, c, parts in scored_no_emb[:10]
                ]
    except Exception:
        reranked = cands

    results: list[Dict[str, Any]] = []
    seen_ctx: set[str] = set()
    for c in reranked:
        if len(results) >= k:
            break
        nid = c["node_id"]
        cid = c["conversation_id"]
        title = (c.get("title") or "").strip() or chatgpt_get_conversation_title(conn, cid)
        ctx = chatgpt_context_window(conn, nid, up=context_up, down=context_down)
        context_text = "\n".join(f'{m["role"]}: {m["text"]}' for m in ctx).strip()
        if not context_text:
            continue
        norm = re.sub(r"\s+", " ", context_text.lower()).strip()
        digest = hashlib.sha1(norm[:800].encode("utf-8")).hexdigest()
        if digest in seen_ctx:
            continue
        seen_ctx.add(digest)
        results.append(
            {
                "node_id": nid,
                "conversation_id": cid,
                "title": title,
                "context": context_text,
            }
        )

    out: Dict[str, Any] = {"results": results, "returned": len(results)}
    if debug:
        out["debug"] = {
            "query": qinfo,
            "fts_query": used_fts_query,
            "fts_mode": used_fts_mode,
            "agent": agent,
            "agent_tag": agent_tag,
            "candidate_limit": candidate_limit,
            "candidates": len(cands),
            "used_embeddings": bool(use_embeddings and emb_map),
            "top_scores": debug_scores,
        }
    return out
