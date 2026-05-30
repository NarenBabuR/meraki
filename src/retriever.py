"""Two-stage retrieval — ported shape from file-processor's retrieve_relevant_content.

    embed query -> vector search top_n (wide) -> similarity threshold filter
                -> [optional] cross-encoder rerank -> top_k (narrow)

The top_n / top_k split is the knob the Break/Fix story rides on:
- RERANK=false (Break #1): skip reranking, just take the top_k by cosine. Cheaper
  but lower precision — a strong bi-encoder candidate isn't always the best answer.
- The SIM_THRESHOLD filter is what lets the abstention FIX work: on an
  out-of-domain query nothing clears the bar, so we return [] and the generator
  abstains instead of hallucinating.
"""
from __future__ import annotations

import logging

from config import CONFIG
from src.embeddings import embed_query
from src import vectorstore
from src import rerank as rerank_mod

logger = logging.getLogger(__name__)


def retrieve(question: str) -> list[dict]:
    """Return the final list of context dicts for a question.

    Each dict: {chunk_id, text, metadata, similarity, [rerank_score]}.
    """
    q_emb = embed_query(question)
    candidates = vectorstore.query(q_emb, top_n=CONFIG.top_n)

    # Similarity-threshold gate (drives abstention on out-of-domain queries).
    kept = [c for c in candidates if c["similarity"] >= CONFIG.sim_threshold]
    logger.info(
        "retrieve: %d candidates, %d above threshold %.2f",
        len(candidates), len(kept), CONFIG.sim_threshold,
    )
    if not kept:
        return []

    if not CONFIG.rerank:
        return kept[: CONFIG.top_k]

    reranked = rerank_mod.rerank(question, kept, top_k=CONFIG.top_k)
    # Cross-encoder relevance gate: drop chunks the reranker judges irrelevant.
    # On out-of-domain queries every chunk falls below the threshold, so we
    # return nothing -> the generator abstains instead of hallucinating.
    relevant = [c for c in reranked if c["rerank_score"] >= CONFIG.rerank_threshold]
    logger.info(
        "retrieve: %d reranked, %d above rerank_threshold %.1f",
        len(reranked), len(relevant), CONFIG.rerank_threshold,
    )
    return relevant
