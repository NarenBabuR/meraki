"""Chunking — flat or small-to-big (parent/child), selected by CONFIG.small_to_big.

Flat mode (ported from file-processor/src/embeddings.py): one recursive splitter,
~1024-char chunks, dedupe. Each chunk is both what we embed and what we show.

Small-to-big mode (default): split each paper into ~1024-char **parents**, then
split each parent into ~256-char **children**. We embed the children (precise
matching against short, focused text) but return the **parent** to the LLM (enough
surrounding context to answer). Every chunk also carries its detected paper
`section` (see sectioning.py). This is the LangChain ParentDocumentRetriever /
"auto-merging" idea, implemented without the framework.

Both modes emit a uniform `IndexItem` list (what gets embedded + stored) plus a
`parents` dict (empty in flat mode). The de-dupe trick from the original matters
either way: arXiv PDFs repeat headers/footers across pages.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

# Paragraph/line-oriented separators (the non-markdown subset of the original).
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class IndexItem:
    """One unit to embed + store in the vector DB."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _splitter(size: int, add_start_index: bool = False) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * CONFIG.chunk_overlap_ratio),
        separators=SEPARATORS,
        length_function=len,
        add_start_index=add_start_index,
    )


# --------------------------------------------------------------------------- #
# Flat mode
# --------------------------------------------------------------------------- #
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
# Small-to-big mode
# --------------------------------------------------------------------------- #
def _build_hierarchical(pages: list[Page]) -> tuple[list[IndexItem], dict]:
    parent_splitter = _splitter(CONFIG.chunk_size, add_start_index=True)
    child_splitter = _splitter(CONFIG.child_chunk_size)

    children: list[IndexItem] = []
    parents: dict[str, dict] = {}
    seen: set[str] = set()

    for arxiv_id, doc_pages in group_by_paper(pages):
        title = doc_pages[0].title
        full, bounds = join_pages(doc_pages)
        sections = detect_sections(full)

        for pi, pdoc in enumerate(parent_splitter.create_documents([full])):
            ptext = pdoc.page_content.strip()
            if len(ptext) < 60:
                continue
            start = pdoc.metadata.get("start_index", 0)
            page = page_at(start, bounds)
            section = section_at(start, sections)
            pid = f"{arxiv_id}:p{pi}"
            parents[pid] = {
                "text": ptext,
                "arxiv_id": arxiv_id,
                "title": title,
                "page": page,
                "section": section,
            }
            for ci, ctext in enumerate(child_splitter.split_text(ptext)):
                ctext = ctext.strip()
                if len(ctext) < 40:
                    continue
                key = ctext.lower()
                if key in seen:
                    continue
                seen.add(key)
                # Contextual header: prepend paper title + section so the
                # embedding (and BM25) capture provenance, not just local text.
                if CONFIG.contextual_headers:
                    embed_text = f"{title} — {section}\n\n{ctext}"
                else:
                    embed_text = ctext
                children.append(
                    IndexItem(
                        id=f"{pid}:c{ci}",
                        text=embed_text,
                        metadata={
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "page": page,
                            "section": section,
                            "parent_id": pid,
                        },
                    )
                )
    logger.info(
        "Hierarchical: %d parents, %d children from %d pages",
        len(parents), len(children), len(pages),
    )
    return children, parents


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def build_chunks(pages: list[Page]) -> tuple[list[IndexItem], dict]:
    """Return (items_to_embed, parents). `parents` is empty in flat mode."""
    if CONFIG.small_to_big:
        return _build_hierarchical(pages)
    items = chunk_pages(pages)
    logger.info("Flat: %d chunks from %d pages", len(items), len(pages))
    return items, {}
