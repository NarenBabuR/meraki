"""Chunking — flat or small-to-big (parent/child), selected by CONFIG.small_to_big.

Flat mode: one recursive splitter, ~1024-char chunks, dedupe. Each chunk is both
what we embed and what we show.

Small-to-big mode (default): we produce LangChain Document objects representing
parent chunks (~1024 chars), each annotated with the detected paper section.
ParentDocumentRetriever (vectorstore.py) then splits each parent into children
(~256 chars) via ContextualChildSplitter, stores children in Chroma, and stores
parents in a LocalFileStore. At query time it maps child hits back to parents.

This is the LangChain ParentDocumentRetriever pattern — vectorstore.py wires
the retriever; chunking.py's job is producing the right Document objects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CONFIG
from src.ingest import Page
from src.sectioning import (
    group_by_paper,
    join_pages,
    detect_sections,
    section_at,
    page_at,
)

logger = logging.getLogger(__name__)

# Paragraph/line-oriented separators.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _splitter(size: int, add_start_index: bool = False) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * CONFIG.chunk_overlap_ratio),
        separators=SEPARATORS,
        length_function=len,
        add_start_index=add_start_index,
    )


# --------------------------------------------------------------------------- #
# Flat mode (unchanged)
# --------------------------------------------------------------------------- #
@dataclass
class IndexItem:
    """One unit to embed + store in the vector DB (flat mode only)."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_pages(pages: list[Page]) -> list[IndexItem]:
    """Per-page recursive chunking with dedupe (flat mode)."""
    splitter = _splitter(CONFIG.chunk_size)
    items: list[IndexItem] = []
    seen: set[str] = set()
    for page in pages:
        for ordinal, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < 40:
                continue
            key = piece.lower()
            if key in seen:
                continue
            seen.add(key)
            cid = f"{page.arxiv_id}:{page.page}:{ordinal}"
            items.append(
                IndexItem(
                    id=cid,
                    text=piece,
                    metadata={
                        "arxiv_id": page.arxiv_id,
                        "title": page.title,
                        "page": page.page,
                    },
                )
            )
    return items


# --------------------------------------------------------------------------- #
# Small-to-big mode — LangChain Document objects
# --------------------------------------------------------------------------- #
class ContextualChildSplitter(RecursiveCharacterTextSplitter):
    """Child splitter for ParentDocumentRetriever.

    Splits parent text into ~256-char children. If CONTEXTUAL_HEADERS is on,
    prepends '<title> — <section>' to each child before embedding so the vector
    captures document provenance (lightweight contextual retrieval).

    The seen set deduplicates across all parents within one add_documents call,
    matching the behaviour of the old _build_hierarchical function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen: set[str] = set()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            title = doc.metadata.get("title", "")
            section = doc.metadata.get("section", "")
            for chunk in self.split_text(doc.page_content):
                chunk = chunk.strip()
                if len(chunk) < 40:
                    continue
                key = chunk.lower()
                if key in self._seen:
                    continue
                self._seen.add(key)
                if CONFIG.contextual_headers and title:
                    embed_text = f"{title} — {section}\n\n{chunk}"
                else:
                    embed_text = chunk
                result.append(Document(
                    page_content=embed_text,
                    metadata={**doc.metadata},
                ))
        return result


def pages_to_documents(pages: list[Page]) -> list[Document]:
    """Convert extracted pages to parent Document objects for ParentDocumentRetriever.

    Each Document is one parent chunk (~1024 chars) with section and page metadata.
    ParentDocumentRetriever will split these into children via ContextualChildSplitter.
    """
    parent_splitter = _splitter(CONFIG.chunk_size, add_start_index=True)
    docs: list[Document] = []

    for arxiv_id, doc_pages in group_by_paper(pages):
        title = doc_pages[0].title
        full, bounds = join_pages(doc_pages)
        sections = detect_sections(full)

        for pdoc in parent_splitter.create_documents([full]):
            ptext = pdoc.page_content.strip()
            if len(ptext) < 60:
                continue
            start = pdoc.metadata.get("start_index", 0)
            docs.append(Document(
                page_content=ptext,
                metadata={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "page": page_at(start, bounds),
                    "section": section_at(start, sections),
                },
            ))

    logger.info("pages_to_documents: %d parent documents from %d pages", len(docs), len(pages))
    return docs
