"""Cross-encoder reranking (local).

To keep the single-key constraint, reranking uses a local sentence-transformers
CrossEncoder: given (query, candidates), it scores each query/passage pair jointly
and returns the best `top_k`.

A bi-encoder (the embedding model) scores query and passage independently; a
cross-encoder reads them together and is markedly better at precision — which is
exactly what Break #1 demonstrates by turning this off.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import CONFIG

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    logger.info("Loading reranker %s", CONFIG.rerank_model)
    return CrossEncoder(CONFIG.rerank_model)


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with the cross-encoder; return top_k.

    Each candidate dict gets a `rerank_score` added. Input candidates carry the
    `text` field (from vectorstore.query).
    """
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = _model().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
