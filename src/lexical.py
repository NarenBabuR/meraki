"""Sparse BM25 retrieval over the same chunks Chroma holds (for hybrid search).

Dense embeddings match on meaning but blur exact tokens — model names ("RoPE",
"ColBERT"), metrics, and numbers ("175 billion"). BM25 nails those. We build the
BM25 index once from the documents already stored in Chroma (no second copy of
the corpus), cache it process-wide, and fuse its ranking with the dense ranking
via Reciprocal Rank Fusion in the retriever.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src import vectorstore

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index():
    """Build BM25 over all documents in the Chroma collection (cached)."""
    col = vectorstore.get_collection()
    data = col.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    corpus = [_tokenize(d) for d in docs]
    logger.info("Built BM25 index over %d documents", len(docs))
    return BM25Okapi(corpus), ids, docs, metas


def search(query: str, top_n: int) -> list[dict]:
    """Return up to top_n BM25 hits as {chunk_id, text, metadata, bm25_score}."""
    bm25, ids, docs, metas = _index()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:top_n]:
        if scores[i] <= 0:
            break
        out.append(
            {
                "chunk_id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "bm25_score": float(scores[i]),
            }
        )
    return out
