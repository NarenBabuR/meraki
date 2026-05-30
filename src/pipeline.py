"""The RAG pipeline: ties retrieval and generation together.

`answer()` returns the answer plus the retrieved contexts and timing, which is
exactly the tuple Ragas needs — so evaluation runs the *live* pipeline rather
than a parallel eval-only code path. What you measure is what you ship.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.retriever import retrieve
from src.generate import generate_answer


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[dict] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def context_texts(self) -> list[str]:
        return [c["text"] for c in self.contexts]

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms


def answer(question: str) -> RAGResult:
    t0 = time.perf_counter()
    contexts = retrieve(question)
    t1 = time.perf_counter()
    text = generate_answer(question, contexts)
    t2 = time.perf_counter()
    return RAGResult(
        question=question,
        answer=text,
        contexts=contexts,
        retrieval_ms=(t1 - t0) * 1000,
        generation_ms=(t2 - t1) * 1000,
    )
