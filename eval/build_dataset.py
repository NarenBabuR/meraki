"""Turn the hand-curated gold Q&A into a Ragas EvaluationDataset.

For each gold question we run the *live* pipeline to collect the model's
`response` and the `retrieved_contexts`, then pair them with the gold
`reference` answer. The `category` tag (in_domain / out_of_domain / multi_hop)
is carried alongside so we can slice metrics by subset — crucial for the
abstention FIX, whose effect is concentrated on the out_of_domain rows.
"""
from __future__ import annotations

import json
import logging

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

from config import GOLD_QA_PATH
from src.pipeline import answer

logger = logging.getLogger(__name__)


def load_gold() -> list[dict]:
    with open(GOLD_QA_PATH) as f:
        return json.load(f)


def build(gold: list[dict] | None = None) -> tuple[EvaluationDataset, list[dict]]:
    """Run the pipeline over gold questions.

    Returns (EvaluationDataset, rows) where `rows` carries category + the raw
    pipeline output (answer, contexts, latency) for per-subset analysis.
    """
    gold = gold or load_gold()

    # Warm up the local models (embedding + cross-encoder) once so per-query
    # latency reflects steady-state cost, not the one-time model load.
    from src.retriever import retrieve
    retrieve("warmup")

    samples: list[SingleTurnSample] = []
    rows: list[dict] = []
    for i, item in enumerate(gold, start=1):
        q = item["question"]
        logger.info("[%d/%d] %s", i, len(gold), q)
        res = answer(q)
        samples.append(
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=res.context_texts or ["(no context retrieved)"],
                response=res.answer,
                reference=item["reference"],
            )
        )
        rows.append(
            {
                "question": q,
                "category": item.get("category", "in_domain"),
                "reference": item["reference"],
                "answer": res.answer,
                "n_contexts": len(res.contexts),
                "retrieval_ms": round(res.retrieval_ms, 1),
                "generation_ms": round(res.generation_ms, 1),
            }
        )
    return EvaluationDataset(samples=samples), rows
