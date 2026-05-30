"""Chunking — ported pattern from file-processor/src/embeddings.py.

Uses LangChain's RecursiveCharacterTextSplitter (chunk_size / overlap from
config) and de-duplicates identical chunks. The dedupe matters for arXiv PDFs:
running headers, footers, and page numbers repeat across pages and would
otherwise flood the index with near-useless near-duplicates.

We drop the markdown header splitter the original used — arXiv PDFs aren't
markdown, so it adds nothing here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG
from src.ingest import Page

logger = logging.getLogger(__name__)

# Paragraph/line-oriented separators (the non-markdown subset of the original).
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str       # f"{arxiv_id}:{page}:{ordinal}"
    arxiv_id: str
    title: str
    page: int
    text: str

    @property
    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "page": self.page,
        }


def _splitter() -> RecursiveCharacterTextSplitter:
    overlap = int(CONFIG.chunk_size * CONFIG.chunk_overlap_ratio)
    return RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
        add_start_index=False,
    )


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    splitter = _splitter()
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for page in pages:
        for ordinal, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < 40:           # skip stray fragments
                continue
            key = piece.lower()
            if key in seen:               # dedupe repeated headers/footers
                continue
            seen.add(key)
            cid = f"{page.arxiv_id}:{page.page}:{ordinal}"
            chunks.append(
                Chunk(cid, page.arxiv_id, page.title, page.page, piece)
            )
    logger.info("Produced %d unique chunks from %d pages", len(chunks), len(pages))
    return chunks
