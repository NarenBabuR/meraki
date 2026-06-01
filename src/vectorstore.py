"""ChromaDB vector store via LangChain's ParentDocumentRetriever.

Index time (build_index.py):
  ParentDocumentRetriever.add_documents(parent_docs)
    ↳ child_splitter splits each parent into ~256-char children
    ↳ children stored in langchain_chroma.Chroma (embeddings via BGEEmbeddings)
    ↳ parents stored in LocalFileStore (persistent, keyed by UUID)

Query time (retriever.py):
  query()      — dense similarity search on child vectors (raw Chroma call)
  get_parent() — fetch parent Document from LocalFileStore by UUID
  get_collection() — raw chromadb collection for BM25 index building

The 'parent_id' metadata key on each child is the UUID that maps it to its
parent in the LocalFileStore, matching the retriever.py merge logic exactly.
"""
from __future__ import annotations

import logging
import shutil
from functools import lru_cache

from langchain_chroma import Chroma
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # suppress ChromaDB telemetry noise

from config import CHROMA_DIR, COLLECTION_NAME, CONFIG
from src.embeddings import BGEEmbeddings
from src.chunking import ContextualChildSplitter, SEPARATORS

logger = logging.getLogger(__name__)

PARENT_STORE_DIR = CHROMA_DIR / "parent_store"

# --------------------------------------------------------------------------- #
# Internal helpers — lazy, cached
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _embeddings() -> BGEEmbeddings:
    return BGEEmbeddings()


def _lc_vectorstore(reset: bool = False) -> Chroma:
    """LangChain Chroma wrapper (cosine space, persistent)."""
    if reset:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _byte_store() -> LocalFileStore:
    PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return LocalFileStore(str(PARENT_STORE_DIR))


def _docstore():
    """Document-typed key-value store backed by LocalFileStore (persistent)."""
    return create_kv_docstore(_byte_store())


def _child_splitter() -> ContextualChildSplitter:
    return ContextualChildSplitter(
        chunk_size=CONFIG.child_chunk_size,
        chunk_overlap=int(CONFIG.child_chunk_size * CONFIG.chunk_overlap_ratio),
        separators=SEPARATORS,
        length_function=len,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_collection():
    """Raw chromadb collection — used by lexical.py for BM25 index building."""
    return _lc_vectorstore()._collection


def index_documents(docs: list[Document], parent_batch_size: int = 150) -> int:
    """Index parent documents using ParentDocumentRetriever.

    LangChain handles child splitting, embedding, and storage of both children
    (in Chroma) and parents (in LocalFileStore). Returns the number of child
    chunks indexed.

    We batch the add_documents call (150 parents at a time) so the resulting
    child chunks never exceed Chroma's maximum batch size of 5461. The same
    retriever instance (and its ContextualChildSplitter with its dedup seen-set)
    is reused across batches so cross-parent deduplication still works.
    """
    # Reset existing stores.
    _lc_vectorstore(reset=True)
    if PARENT_STORE_DIR.exists():
        shutil.rmtree(PARENT_STORE_DIR)

    retriever = ParentDocumentRetriever(
        vectorstore=_lc_vectorstore(),
        docstore=_docstore(),
        child_splitter=_child_splitter(),
        id_key="parent_id",   # matches retriever.py's c["metadata"]["parent_id"]
    )
    for start in range(0, len(docs), parent_batch_size):
        batch = docs[start: start + parent_batch_size]
        retriever.add_documents(batch)
        logger.info("Indexed parents %d/%d", min(start + parent_batch_size, len(docs)), len(docs))

    count = retriever.vectorstore._collection.count()
    logger.info("Indexed %d child chunks, parents stored in %s", count, PARENT_STORE_DIR)
    return count


def query(query_embedding: list[float], top_n: int) -> list[dict]:
    """Dense similarity search on child vectors.

    Returns up to top_n candidates as {chunk_id, text, metadata, similarity},
    sorted by similarity descending. Chroma returns cosine distance (1 - sim);
    we convert back to similarity.
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
        out.append({
            "chunk_id": cid,
            "text": doc,
            "metadata": meta,
            "similarity": 1.0 - dist,
        })
    return out


def get_parent(parent_id: str) -> dict | None:
    """Fetch a parent Document from LocalFileStore by its UUID.

    Returns a plain dict {text, arxiv_id, title, page, section} matching the
    shape that retriever.py's _merge_to_parents expects.
    """
    docs = _docstore().mget([parent_id])
    if not docs or docs[0] is None:
        return None
    doc: Document = docs[0]
    return {"text": doc.page_content, **doc.metadata}


def index_items(items, parents: dict, batch_size: int = 256) -> int:
    """Embed and upsert flat IndexItems into a fresh collection (flat mode only)."""
    from src.embeddings import embed_documents
    col = _lc_vectorstore(reset=True)._collection
    for start in range(0, len(items), batch_size):
        batch = items[start: start + batch_size]
        embeddings = embed_documents([it.text for it in batch])
        col.add(
            ids=[it.id for it in batch],
            embeddings=embeddings,
            documents=[it.text for it in batch],
            metadatas=[it.metadata for it in batch],
        )
        logger.info("Indexed %d/%d", min(start + batch_size, len(items)), len(items))
    return col.count()


def assert_dim_matches() -> None:
    """Guard against a stale index built with a different embedding model."""
    col = get_collection()
    if col.count() == 0:
        return
    peek = col.peek(limit=1)
    embs = peek.get("embeddings")
    if embs is not None and len(embs) > 0:
        from src.embeddings import embedding_dim
        stored = len(embs[0])
        current = embedding_dim()
        if stored != current:
            raise RuntimeError(
                f"Index dim {stored} != model dim {current}. "
                f"Wipe {CHROMA_DIR} and re-run scripts/build_index.py."
            )
