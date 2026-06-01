"""Local embeddings via sentence-transformers (BGE).

BGE retrieval models are trained with an instruction prefix on *queries only*
(the query/document asymmetry common to instruction-tuned retrievers).
`embed_documents` never prefixes; `embed_query` prefixes iff
CONFIG.use_query_instruction. Dropping the prefix (Break #2) creates a
representation mismatch between queries and documents and degrades recall.

The model is loaded lazily and cached process-wide so the ~130MB download /
load happens once.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import CONFIG

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    logger.info("Loading embedding model %s", CONFIG.embed_model)
    return SentenceTransformer(CONFIG.embed_model)


def embedding_dim() -> int:
    return _model().get_sentence_embedding_dimension()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks. Normalized for cosine similarity."""
    vecs = _model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=len(texts) > 64,
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a user query, applying the BGE instruction prefix when enabled."""
    text = query
    if CONFIG.use_query_instruction:
        text = CONFIG.query_instruction + query
    vec = _model().encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


class BGEEmbeddings:
    """LangChain Embeddings-compatible wrapper around the local BGE model.

    Required by langchain_chroma.Chroma, which expects an object with
    embed_documents / embed_query methods matching the LangChain Embeddings ABC.
    We inherit the interface without importing langchain_core here so this module
    stays usable even if the langchain stack is not installed.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_query(text)
