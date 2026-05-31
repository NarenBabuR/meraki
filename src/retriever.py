"""Two-stage retrieval, with optional small-to-big auto-merging.

    embed query -> vector search top_n (wide) -> similarity threshold filter
                -> [optional] cross-encoder rerank + relevance gate
                -> [small-to-big] merge matched children up to their parents
                -> top_k

The top_n / top_k split is the knob the Break/Fix story rides on:
- RERANK=false (Break #1): skip reranking, just take top_k by cosine. Cheaper but
  lower precision.
- The cross-encoder relevance gate is what lets the pipeline abstain on
  out-of-domain queries (nothing clears the bar -> return []).

Small-to-big (CONFIG.small_to_big): we search/rank short *children* for precise
matching, then return the larger *parent* chunk each matched child belongs to
(de-duplicated), so the LLM gets enough surrounding context to answer and cite.
"""
from __future__ import annotations

import logging

from config import CONFIG
from src.embeddings import embed_query
from src import vectorstore
from src import rerank as rerank_mod

logger = logging.getLogger(__name__)


def _merge_to_parents(ranked: list[dict]) -> list[dict]:
    """Map ranked child hits up to their (de-duplicated) parent chunks, top_k."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in ranked:
        pid = c["metadata"].get("parent_id")
        if pid is None:  # index built in flat mode — pass the hit through
            out.append(c)
        elif pid not in seen:
            seen.add(pid)
            parent = vectorstore.get_parent(pid)
            meta = (
                {k: parent[k] for k in ("arxiv_id", "title", "page", "section")}
                if parent
                else c["metadata"]
            )
            out.append(
                {
                    "chunk_id": pid,
                    "text": parent["text"] if parent else c["text"],
                    "metadata": meta,
                    "similarity": c["similarity"],
                    "rerank_score": c.get("rerank_score"),
                    "matched_child": c["text"],
                }
            )
        if len(out) >= CONFIG.top_k:
            break
    return out


def retrieve(question: str) -> list[dict]:
    """Return the final list of context dicts for a question.

    Each dict: {chunk_id, text, metadata, similarity, [rerank_score], [matched_child]}.
    """
    q_emb = embed_query(question)
    candidates = vectorstore.query(q_emb, top_n=CONFIG.top_n)

    # Coarse cosine pre-filter.
    kept = [c for c in candidates if c["similarity"] >= CONFIG.sim_threshold]
    logger.info(
        "retrieve: %d candidates, %d above threshold %.2f",
        len(candidates), len(kept), CONFIG.sim_threshold,
    )
    if not kept:
        return []

    if CONFIG.rerank:
        # Rank all kept candidates, then drop those the cross-encoder judges
        # irrelevant. On out-of-domain queries everything falls below the gate,
        # so nothing survives and the generator abstains.
        reranked = rerank_mod.rerank(question, kept, top_k=len(kept))
        ranked = [c for c in reranked if c["rerank_score"] >= CONFIG.rerank_threshold]
        logger.info(
            "retrieve: %d reranked, %d above rerank_threshold %.1f",
            len(reranked), len(ranked), CONFIG.rerank_threshold,
        )
    else:
        ranked = kept  # cosine order; no gate (mirrors Break #1)

    if not ranked:
        return []

    if CONFIG.small_to_big:
        return _merge_to_parents(ranked)
    return ranked[: CONFIG.top_k]
