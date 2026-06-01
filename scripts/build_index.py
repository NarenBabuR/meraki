"""Build the vector index: download arXiv PDFs -> extract -> chunk -> embed -> persist.

Run once before querying or evaluating:

    python scripts/build_index.py

Re-run after changing CHUNK_SIZE / EMBED_MODEL (those invalidate the index).

Small-to-big mode (default):
  pages_to_documents()  — converts pages to parent Document objects with section metadata
  vectorstore.index_documents()  — ParentDocumentRetriever splits into children,
                                    embeds via BGE, stores in Chroma + LocalFileStore

Flat mode (SMALL_TO_BIG=false):
  chunk_pages()          — per-page recursive chunking with dedupe
  vectorstore.index_items() — embed and upsert into Chroma (legacy flat path)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, describe  # noqa: E402
from src.ingest import download_papers, ingest_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")


def main() -> None:
    log.info("Config: %s", describe())
    papers = download_papers()
    log.info("%d papers available", len(papers))
    pages = ingest_all(papers)
    log.info("%d total pages of text", len(pages))

    if CONFIG.small_to_big:
        from src.chunking import pages_to_documents
        from src.vectorstore import index_documents
        docs = pages_to_documents(pages)
        count = index_documents(docs)
        log.info("DONE — indexed %d child chunks (small-to-big) into Chroma", count)
    else:
        from src.chunking import chunk_pages, IndexItem
        from src.vectorstore import index_items
        items = chunk_pages(pages)
        count = index_items(items, {})
        log.info("DONE — indexed %d flat chunks into Chroma", count)


if __name__ == "__main__":
    main()
