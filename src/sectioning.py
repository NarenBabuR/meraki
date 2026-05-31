"""Section detection over a paper's full text.

arXiv papers have a regular skeleton (Abstract, 1 Introduction, 3 Method,
Conclusion, ...). We reconstruct the full document from its per-page text, find
section headings with a heuristic regex, and provide helpers to map any character
offset back to its page number and enclosing section title.

This is best-effort: PDF text extraction is imperfect, so some headings are
missed and some regions fall back to a generic label. Section is used as
*metadata* (better citations + a light retrieval signal), so rough detection is
acceptable — the parent/child structure (chunking.py) is the load-bearing part.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from src.ingest import Page

# Numbered headings like "3 Method", "3.1 Architecture" (capital after the number
# required, so body sentences starting with a digit don't match).
_NUM = r"\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z][A-Za-z0-9 ,:&\-]{2,50}"
# Common named sections (case-insensitive, whole line).
_NAMED = (
    r"(?i:abstract|introduction|related work|background|preliminaries|"
    r"methods?|methodology|approach|architecture|model|datasets?|"
    r"experiments?|experimental setup|results?|evaluation|analysis|"
    r"ablations?|discussion|conclusions?|limitations?|"
    r"acknowledg\w*|appendix)"
)
SECTION_PAT = re.compile(r"(?m)^\s{0,4}(" + _NUM + r"|" + _NAMED + r")\s*$")


def group_by_paper(pages: list[Page]):
    """Group a flat list of pages by arxiv_id, preserving order."""
    g: "OrderedDict[str, list[Page]]" = OrderedDict()
    for p in pages:
        g.setdefault(p.arxiv_id, []).append(p)
    return g.items()


def join_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate one paper's pages into full text.

    Returns (full_text, bounds) where bounds is [(start, end, page_number), ...]
    so an offset into full_text can be mapped back to its page.
    """
    texts, bounds, cur = [], [], 0
    for p in pages:
        texts.append(p.text)
        bounds.append((cur, cur + len(p.text), p.page))
        cur += len(p.text) + 2  # "\n\n" join separator
    return "\n\n".join(texts), bounds


def page_at(idx: int, bounds: list[tuple[int, int, int]]) -> int:
    pg = bounds[0][2] if bounds else 1
    for start, _end, page in bounds:
        if idx >= start:
            pg = page
        else:
            break
    return pg


def detect_sections(full_text: str) -> list[tuple[int, str]]:
    """Return [(offset, section_title), ...] sorted by offset."""
    secs = [(m.start(), m.group(1).strip()) for m in SECTION_PAT.finditer(full_text)]
    if not secs or secs[0][0] > 0:
        secs.insert(0, (0, "Preamble"))
    return secs


def section_at(idx: int, sections: list[tuple[int, str]]) -> str:
    title = sections[0][1] if sections else "Body"
    for offset, name in sections:
        if idx >= offset:
            title = name
        else:
            break
    # Normalize whitespace in the heading for clean metadata.
    return re.sub(r"\s+", " ", title)[:60]
