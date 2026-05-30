"""Build the vector index: download arXiv PDFs -> extract -> chunk -> embed -> persist.

Run once before querying or evaluating:

    python scripts/build_index.py

Re-run after changing CHUNK_SIZE / EMBED_MODEL (those invalidate the index).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import describe  # noqa: E402
from src.ingest import download_papers, ingest_all  # noqa: E402
from src.chunking import chunk_pages  # noqa: E402
from src.vectorstore import index_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")


def main() -> None:
    log.info("Config: %s", describe())
    papers = download_papers()
    log.info("%d papers available", len(papers))
    pages = ingest_all(papers)
    log.info("%d total pages of text", len(pages))
    chunks = chunk_pages(pages)
    count = index_chunks(chunks)
    log.info("DONE — indexed %d chunks into Chroma", count)


if __name__ == "__main__":
    main()
