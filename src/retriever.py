"""Retrieval: optional query decomposition + hybrid (dense+BM25) + rerank + merge.

Full path (all flags on):

    decompose question -> for each sub-question:
        dense vector search (top_n)  ⊕  BM25 search (top_n)  --RRF-->  candidate pool
    pool (de-duped across sub-questions)
    -> cross-encoder rerank against the ORIGINAL question
    -> relevance gate (drop < rerank_threshold)        [abstention mechanism]
    -> small-to-big: merge matched children up to their parents (top_k)

Each piece is a config flag, so before/after is a flag flip:
- HYBRID (#3) — add BM25 + RRF fusion
- DECOMPOSE (#4) — split multi-hop questions, pool sub-question hits
- CONTEXTUAL_HEADERS (#2) — index-time; lives in chunking.py
- RERANK (Break #1), SMALL_TO_BIG, ABSTAIN — as before
"""
from __future__ import annotations

import logging

from config import CONFIG
from src.embeddings import embed_query
from src import vectorstore
from src import lexical
from src import rerank as rerank_mod
from src.decompose import decompose

logger = logging.getLogger(__name__)


def _rrf(lists: list[list[dict]], k: int) -> list[dict]:
    """Reciprocal Rank Fusion over ranked candidate lists, keyed by chunk_id.

    The first list's dicts win on collision (we pass dense first, so the fused
    candidate keeps the cosine `similarity` when available).
    """
    score: dict[str, float] = {}
    store: dict[str, dict] = {}
    for lst in lists:
        for rank, c in enumerate(lst):
            cid = c["chunk_id"]
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
            store.setdefault(cid, c)
    ranked = sorted(store.values(), key=lambda c: score[c["chunk_id"]], reverse=True)
    return ranked


def _candidates_for(query: str) -> list[dict]:
    """Dense (and optionally BM25-fused) candidate children for one query."""
    dense = vectorstore.query(embed_query(query), top_n=CONFIG.top_n)
    if not CONFIG.hybrid:
        return dense
    lex = lexical.search(query, top_n=CONFIG.top_n)
    return _rrf([dense, lex], CONFIG.rrf_k)[: CONFIG.top_n]


def _merge_to_parents(ranked: list[dict]) -> list[dict]:
    """Map ranked child hits up to their (de-duplicated) parent chunks, top_k."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in ranked:
        pid = c["metadata"].get("parent_id")
        if pid is None:  # flat index — pass the hit through
            out.append(c)
        elif pid not in seen:
            seen.add(pid)
            parent = vectorstore.get_parent(pid)
            meta = (
                {k: parent[k] for k in ("arxiv_id", "title", "page", "section")}
                if parent else c["metadata"]
            )
            out.append({
                "chunk_id": pid,
                "text": parent["text"] if parent else c["text"],
                "metadata": meta,
                "similarity": c.get("similarity"),
                "rerank_score": c.get("rerank_score"),
                "matched_child": c["text"],
            })
        if len(out) >= CONFIG.top_k:
            break
    return out


def retrieve(question: str) -> list[dict]:
    """Return the final list of context dicts for a question."""
    queries = decompose(question) if CONFIG.decompose else (question,)
    if len(queries) > 1:
        logger.info("decomposed into %d sub-questions", len(queries))

    # Pool candidate children across (sub-)queries, de-duplicated by chunk_id.
    pool: dict[str, dict] = {}
    for q in queries:
        for c in _candidates_for(q):
            pool.setdefault(c["chunk_id"], c)
    candidates = list(pool.values())

    # In pure-dense mode the cheap cosine pre-filter still applies; in hybrid mode
    # RRF has no single similarity scale, so the cross-encoder gate does the work.
    if not CONFIG.hybrid:
        candidates = [c for c in candidates if c["similarity"] >= CONFIG.sim_threshold]
    if not candidates:
        return []

    if CONFIG.rerank:
        # Rerank the whole pool against the ORIGINAL question, then gate.
        reranked = rerank_mod.rerank(question, candidates, top_k=len(candidates))
        ranked = [c for c in reranked if c["rerank_score"] >= CONFIG.rerank_threshold]
        logger.info(
            "retrieve: pool=%d reranked, %d above gate %.1f",
            len(reranked), len(ranked), CONFIG.rerank_threshold,
        )
    else:
        ranked = sorted(candidates, key=lambda c: c.get("similarity") or 0.0, reverse=True)

    if not ranked:
        return []
    if CONFIG.small_to_big:
        return _merge_to_parents(ranked)
    return ranked[: CONFIG.top_k]
