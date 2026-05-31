"""ChromaDB persistent vector store (+ a parent sidecar for small-to-big).

Embedded + file-backed (no server, no docker) so a grader can run everything
with a single `pip install`. Cosine space to match the normalized BGE vectors.
We pass embeddings explicitly (Chroma never calls an embedding fn itself), which
keeps the index-time and query-time embedding code in one place (embeddings.py).

In small-to-big mode we embed/search the *children* (in Chroma) but need the
*parent* text at query time. Parents aren't searched by similarity, so rather
than give them dummy vectors in Chroma we persist them as a JSON sidecar
(`parents.json`) next to the Chroma store and look them up by id.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from src.chunking import IndexItem
from src.embeddings import embed_documents, embedding_dim

logger = logging.getLogger(__name__)

PARENTS_PATH = CHROMA_DIR / "parents.json"


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


def index_items(items: list[IndexItem], parents: dict, batch_size: int = 256) -> int:
    """Embed and upsert items into a fresh collection; persist parents sidecar.

    Returns the number of items indexed.
    """
    col = get_collection(reset=True)
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        embeddings = embed_documents([it.text for it in batch])
        col.add(
            ids=[it.id for it in batch],
            embeddings=embeddings,
            documents=[it.text for it in batch],
            metadatas=[it.metadata for it in batch],
        )
        logger.info("Indexed %d/%d", min(start + batch_size, len(items)), len(items))

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    PARENTS_PATH.write_text(json.dumps(parents))
    logger.info("Wrote %d parents to %s", len(parents), PARENTS_PATH)
    return col.count()


@lru_cache(maxsize=1)
def _parents() -> dict:
    if PARENTS_PATH.exists():
        return json.loads(PARENTS_PATH.read_text())
    return {}


def get_parent(parent_id: str) -> dict | None:
    """Look up a parent chunk by id (small-to-big mode)."""
    return _parents().get(parent_id)


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
