"""Ragas configuration: Claude as the judge LLM, local BGE as the embeddings.

The single most common Ragas footgun: if a metric has no LLM/embeddings attached
it silently falls back to OpenAI and errors on a missing OPENAI_API_KEY. So we
build the wrappers once and attach them to *every* metric explicitly.

Metrics and what each one catches in this pipeline:
- LLMContextPrecisionWithReference : are the retrieved chunks relevant & ranked
  well?  (moves when reranking is on/off — Break #1)
- LLMContextRecall                 : did retrieval surface the chunks needed to
  support the reference answer?     (moves with query/doc embedding match — Break #2)
- Faithfulness                     : is the answer grounded in the contexts, i.e.
  no hallucination?                 (moves with the abstention guardrail — the FIX)
- ResponseRelevancy                : does the answer actually address the question?
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from config import CONFIG


def _judge():
    return LangchainLLMWrapper(
        ChatAnthropic(model=CONFIG.judge_model, temperature=0, max_tokens=1024)
    )


def _judge_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=CONFIG.embed_model)
    )


def build_metrics():
    """Metric objects with the Claude judge + local embeddings attached to each."""
    llm = _judge()
    emb = _judge_embeddings()
    return [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
    ]


def run_ragas(dataset: EvaluationDataset):
    """Run Ragas over the dataset; returns a ragas EvaluationResult."""
    metrics = build_metrics()
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_judge(),
        embeddings=_judge_embeddings(),
        run_config=RunConfig(max_workers=CONFIG.eval_max_workers),
        show_progress=True,
    )
