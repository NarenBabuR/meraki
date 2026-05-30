"""ChromaDB persistent vector store.

Embedded + file-backed (no server, no docker) so a grader can run everything
with a single `pip install`. Cosine space to match the normalized BGE vectors.
We pass embeddings explicitly (Chroma never calls an embedding fn itself), which
keeps the index-time and query-time embedding code in one place (embeddings.py).
"""
from __future__ import annotations

import logging

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from src.chunking import Chunk
from src.embeddings import embed_documents, embedding_dim

logger = logging.getLogger(__name__)


def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(reset: bool = False):
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[Chunk], batch_size: int = 256) -> int:
    """Embed and upsert chunks into a fresh collection. Returns count indexed."""
    col = get_collection(reset=True)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embed_documents([c.text for c in batch])
        col.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        logger.info("Indexed %d/%d", min(start + batch_size, len(chunks)), len(chunks))
    return col.count()


def query(query_embedding: list[float], top_n: int) -> list[dict]:
    """Return up to top_n candidates as
    {chunk_id, text, metadata, similarity} dicts, sorted by similarity desc.

    Chroma returns cosine *distance* (1 - cosine_similarity); we convert back.
    """
    col = get_collection()
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            {
                "chunk_id": cid,
                "text": doc,
                "metadata": meta,
                "similarity": 1.0 - dist,
            }
        )
    return out


def assert_dim_matches() -> None:
    """Guard against a stale index built with a different embedding model."""
    col = get_collection()
    if col.count() == 0:
        return
    peek = col.peek(limit=1)
    embs = peek.get("embeddings")
    if embs is not None and len(embs) > 0:
        stored = len(embs[0])
        current = embedding_dim()
        if stored != current:
            raise RuntimeError(
                f"Index dim {stored} != model dim {current}. "
                f"Wipe {CHROMA_DIR} and re-run scripts/build_index.py."
            )
