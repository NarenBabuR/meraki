"""LLM query decomposition for multi-hop questions.

Multi-hop is the measured weak spot (precision ~0.46): retrieval finds one hop
but not the other. We ask a cheap model to split a question into the minimal set
of standalone sub-questions; the retriever then searches for each and pools the
results (reranking the pool against the *original* question). Single-fact lookups
are returned unchanged, so the cost is a no-op there.

Off by default (CONFIG.decompose) — it adds one LLM call per query.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

import anthropic

from config import CONFIG

logger = logging.getLogger(__name__)

_SYSTEM = """You decompose a research question into the minimal set of standalone sub-questions needed to answer it via document retrieval.

Rules:
- If the question asks for ONE fact, return just the original question.
- If it requires connecting multiple facts (multi-hop), return 2-3 self-contained sub-questions, each answerable on its own.
- Output ONLY a JSON array of strings. No prose."""


def _extract_json_array(text: str) -> list[str] | None:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return [s for s in arr if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        return None


@lru_cache(maxsize=256)
def decompose(question: str) -> tuple[str, ...]:
    """Return (original, *sub_questions). Falls back to (original,) on any error."""
    try:
        client = anthropic.Anthropic(max_retries=2)
        resp = client.messages.create(
            model=CONFIG.decompose_model,
            max_tokens=300,
            temperature=0,
            system=_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        subs = _extract_json_array(text) or []
    except Exception as e:  # never let decomposition break retrieval
        logger.warning("decompose failed (%s); using original question", e)
        subs = []

    # Always include the original; cap the number of extra sub-questions.
    extras = [s for s in subs if s.strip().lower() != question.strip().lower()]
    queries = [question] + extras[: CONFIG.max_subquestions]
    return tuple(queries)
