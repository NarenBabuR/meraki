# arXiv ML RAG — Build, Break, Fix

A small, runnable Retrieval-Augmented Generation pipeline over a corpus of ~20
machine-learning papers from arXiv. It ships with a query UI, a Ragas-based
evaluation harness, and a reproducible **Build → Break → Fix** story where each
failure mode is a single config flag.

> Meraki Labs work-trial, PS1. The companion write-up (decisions, failure
> analysis, production notes) is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

---

## What it does

```
arXiv PDFs ─► extract ─► section-tag ─► parent (1024) → child (256) chunks
                                          │ embed children (local BGE)
                                          ▼
                                       ChromaDB  +  parents sidecar
query ─► embed ─► vector search (top_n) ─► threshold ─► cross-encoder rerank
      ─► relevance gate ─► merge children → parents (top_k) ─► Claude (grounded, abstains)
      ─► answer + cited sources (with section)
```

**Chunking is small-to-big** (default): embed small ~256-char *children* for
precise matching, but return the ~1024-char *parent* to the LLM for context. Each
chunk is tagged with its paper section. Set `SMALL_TO_BIG=false` for flat 1024
chunking. (See the write-up for the measured flat-vs-small-to-big tradeoff.)

- **Generation + eval judge:** Claude (Anthropic API) — the only thing that
  needs a key.
- **Embeddings + reranking:** local `sentence-transformers` models (free,
  offline, reproducible). No AWS / OpenAI / vector-DB server required.
- **Vector store:** embedded persistent ChromaDB (a folder on disk).

## Quick start

Requires Python 3.11+ and an Anthropic API key.

```bash
# 1. Install (a clean virtualenv is strongly recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your key
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
python scripts/test_key.py            # confirms the key works (one cheap call)

# 3. Build the index (downloads ~20 PDFs, embeds them; a few minutes, runs once)
python scripts/build_index.py

# 4a. Chat UI
streamlit run app.py

# 4b. ...or one-off from Python
python -c "from src.pipeline import answer; print(answer('What is scaled dot-product attention?').answer)"
```

First run downloads two small models from Hugging Face (~210 MB total, cached
afterwards): `BAAI/bge-small-en-v1.5` (embeddings) and
`cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker).

## Evaluation (Build / Break / Fix)

Quality is measured with [Ragas](https://docs.ragas.io): **context precision**,
**context recall**, **faithfulness**, and **answer relevancy**. The judge is
Claude; the gold set is 26 hand-curated Q&A
([`data/eval/gold_qa.json`](data/eval/gold_qa.json)) spanning in-domain,
multi-hop, and out-of-domain (unanswerable) questions.

Every failure mode is a config flip, so before/after numbers are reproducible:

```bash
python scripts/run_eval.py --tag baseline
RERANK=false                python scripts/run_eval.py --tag no_rerank             # Break #1
USE_QUERY_INSTRUCTION=false python scripts/run_eval.py --tag no_query_instruction  # Break #2
ABSTAIN=false               python scripts/run_eval.py --tag no_abstain            # remove the FIX

# Side-by-side (overall + out-of-domain subset)
python scripts/run_eval.py --compare baseline no_abstain
```

Abstention is measured separately with a deterministic (no-judge) metric — the
right tool for "does it refuse the unanswerable?" (the write-up explains why Ragas
faithfulness is not):

```bash
ABSTAIN=true  python scripts/abstention_check.py --tag fix_on
ABSTAIN=false python scripts/abstention_check.py --tag fix_off
```

Results are written to `data/eval/results/<tag>.json` (committed as evidence) and
printed as a table. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for the analysis and
the headline numbers.

## Configuration

All knobs live in [`config.py`](config.py) and are environment-overridable; see
[`.env.example`](.env.example). The ones that matter:

| Var | Default | Effect |
|---|---|---|
| `SMALL_TO_BIG` | `true` | parent/child retrieval + section metadata (re-index after changing) |
| `CHUNK_SIZE` | 1024 | parent / flat chunk length (re-index) |
| `CHILD_CHUNK_SIZE` | 256 | child chunk length in small-to-big (re-index) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model (re-index after changing) |
| `CONTEXTUAL_HEADERS` | `true` | prepend title+section to each child before embedding (re-index) |
| `HYBRID` | `true` | fuse dense + BM25 with Reciprocal Rank Fusion |
| `DECOMPOSE` | `false` | LLM splits multi-hop questions into sub-queries (adds a call; big multi-hop gain) |
| `USE_QUERY_INSTRUCTION` | `true` | BGE query-prefix asymmetry — **Break #2** |
| `TOP_N` / `TOP_K` | 25 / 5 | candidate pool / final chunks to the LLM |
| `RERANK` | `true` | cross-encoder reranking — **Break #1** |
| `RERANK_THRESHOLD` | -3.0 | cross-encoder relevance gate (drives abstention) |
| `ABSTAIN` | `true` | refuse to answer when nothing relevant — **the FIX** |
| `GEN_MODEL` | `claude-sonnet-4-6` | generation model |
| `JUDGE_MODEL` | `claude-sonnet-4-6` | Ragas judge (use a Haiku id for cheaper eval) |

## Layout

```
config.py            # single source of truth for all toggles
src/
  ingest.py          # arXiv download + pypdf text extraction
  sectioning.py      # detect paper sections; offset -> page/section mapping
  chunking.py        # flat OR small-to-big (parent/child) chunking + dedupe + contextual headers
  embeddings.py      # local BGE; query/doc instruction asymmetry
  vectorstore.py     # ChromaDB persistent (cosine) + parents sidecar
  lexical.py         # BM25 sparse index (for hybrid retrieval)
  decompose.py       # LLM query decomposition (multi-hop)
  rerank.py          # local cross-encoder reranker
  retriever.py       # decompose -> dense⊕BM25 (RRF) -> rerank -> gate -> merge-to-parents
  generate.py        # Claude wrapper: retry/fallback + abstention prompt
  pipeline.py        # ties retrieval + generation; returns answer + contexts
scripts/
  build_index.py     # build the vector index
  run_eval.py        # run Ragas for a config, write results/<tag>.json
  abstention_check.py# deterministic abstention/hallucination metric (no judge)
  test_key.py        # validate ANTHROPIC_API_KEY
eval/
  ragas_eval.py      # Ragas wired to Claude judge + local embeddings
  build_dataset.py   # run live pipeline over gold Q&A -> EvaluationDataset
app.py               # Streamlit chat UI with inline retrieved chunks
data/eval/gold_qa.json   # 26 hand-curated questions
data/eval/results/       # committed eval outputs
```

## Notes & limitations

- The corpus is pinned by arXiv ID for reproducibility; edit
  `DEFAULT_ARXIV_IDS` in `src/ingest.py` to change it.
- `RERANK_THRESHOLD` is a heuristic tuned on this corpus' score distribution, not
  a learned cutoff — see the write-up's "what I'd do next".
- This is a demo, not a service: no auth, no persistence beyond the index, single
  node. The write-up's production section covers what changes at 100k+ users.
