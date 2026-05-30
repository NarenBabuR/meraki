"""Single source of truth for the RAG pipeline.

Every tunable lives here and is overridable by an environment variable, so the
Build / Break / Fix experiments are config flips rather than code edits:

    RERANK=false python scripts/run_eval.py --tag no_rerank

`snapshot()` captures the active config into eval result files so every score is
traceable to the settings that produced it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
GOLD_QA_PATH = EVAL_DIR / "gold_qa.json"

COLLECTION_NAME = "arxiv_ml"


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Config:
    # --- Chunking (changing these requires a re-index) ---
    chunk_size: int = _int("CHUNK_SIZE", 1024)
    chunk_overlap_ratio: float = _float("CHUNK_OVERLAP_RATIO", 0.2)

    # --- Embeddings ---
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    # Break #2 toggle: BGE expects an instruction prefix on *queries only*.
    # Dropping it creates a query/document representation mismatch -> recall drops.
    use_query_instruction: bool = _bool("USE_QUERY_INSTRUCTION", True)
    query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # --- Retrieval ---
    top_n: int = _int("TOP_N", 25)          # wide candidate pool from vector search
    top_k: int = _int("TOP_K", 5)           # final chunks handed to the LLM
    # Coarse cosine pre-filter. BGE has a high similarity baseline (~0.5 even for
    # unrelated text), so this only drops truly orthogonal candidates.
    sim_threshold: float = _float("SIM_THRESHOLD", 0.30)
    # Break #1 toggle: disable the cross-encoder reranker.
    rerank: bool = _bool("RERANK", True)
    rerank_model: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # Cross-encoder relevance gate. Measured separation on this corpus:
    # in-domain top scores ~+5..+8 (worst legit case ~-0.4), out-of-domain ~-7..-10.
    # A chunk scoring below this is treated as irrelevant; if none survive, the
    # retriever returns nothing and (with ABSTAIN) the pipeline declines to answer.
    rerank_threshold: float = _float("RERANK_THRESHOLD", -3.0)

    # --- Generation / safety ---
    # FIX toggle: the abstention guardrail. False -> the model answers even when
    # retrieval found nothing relevant (hallucination on out-of-domain queries).
    abstain: bool = _bool("ABSTAIN", True)
    gen_model: str = os.getenv("GEN_MODEL", "claude-sonnet-4-6")
    gen_max_tokens: int = _int("GEN_MAX_TOKENS", 1024)

    # --- Eval ---
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
    eval_max_workers: int = _int("EVAL_MAX_WORKERS", 4)

    def snapshot(self) -> dict:
        """Config as a plain dict, for embedding in eval result files."""
        return asdict(self)


CONFIG = Config()


def describe() -> str:
    """Human-readable one-liner of the toggles that matter for experiments."""
    c = CONFIG
    return (
        f"chunk_size={c.chunk_size} top_n={c.top_n} top_k={c.top_k} "
        f"rerank={c.rerank} query_instruction={c.use_query_instruction} "
        f"abstain={c.abstain} gen={c.gen_model}"
    )
