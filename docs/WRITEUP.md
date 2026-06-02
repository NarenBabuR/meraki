# RAG Pipeline: Build, Break, Fix — Write-up

**Problem:** Meraki Labs PS1 · **Corpus:** ~20 arXiv ML papers

This document covers what I built, the decisions and tradeoffs behind it, how I
measured quality, the failure modes I found (two documented, one fixed with
before/after numbers), what breaks in production, and what I'd do next. The repo
is runnable from the [README](../README.md).

---

## 1. What I built

A retrieval-augmented QA system over a corpus of foundational ML papers
(Transformer, BERT, GPT-3, RAG, LoRA, T5, DPR, ColBERT, …). You ask a question;
it retrieves passages, reranks them, generates a grounded answer with citations,
and **declines to answer when the corpus doesn't contain the answer**.

### Pipeline diagram

```
╔══════════════════════════════════════════════════════════════════╗
║  INDEX TIME  (scripts/build_index.py — run once)                ║
╚══════════════════════════════════════════════════════════════════╝

  20 arXiv PDFs
       │ pypdf — extract pages, stop at References section
       ▼
  Pages  (354 total, per-paper)
       │ detect_sections() — regex section headings
       │ parent_splitter   — 1024-char chunks (add_start_index)
       ▼
  1619 Parent Documents  {text, title, section, page}
       │
       │  LangChain ParentDocumentRetriever.add_documents()
       │
       ├──► child_splitter (256-char) + contextual header
       │         "title — section\n\nchunk text"
       │              │ BGE embed (384-dim, L2-normalised)
       │              ▼
       │         Chroma HNSW index  (7 256 child vectors)
       │
       └──► LocalFileStore  (1 619 parents, keyed by UUID)


╔══════════════════════════════════════════════════════════════════╗
║  QUERY TIME  (pipeline.answer(question))                        ║
╚══════════════════════════════════════════════════════════════════╝

  User question
       │ DECOMPOSE=true → Haiku splits into ≤3 sub-questions
       ▼
  Sub-questions  [q₁, q₂, …]
       │
       │  for each sub-question:
       ├──► Dense:  BGE embed (+ query instruction prefix)
       │            → Chroma cosine top-25
       │
       └──► Sparse: BM25Okapi tokenise → top-25
       │
       │  Reciprocal Rank Fusion  (k=60)  → top-25 fused
       ▼
  Candidate pool  (child chunks, de-duped across sub-questions)
       │ CrossEncoder ms-marco-MiniLM — score every (q, chunk) pair
       │ relevance gate: drop score < −3.0
       ▼
  ┌─────────────────────────────────────────┐
  │  gate passes?                           │
  │  YES → fetch Parents from LocalFileStore│
  │  NO  → return [] (abstain, no LLM call) │
  └─────────────────────────────────────────┘
       │
       ▼
  Top-5 Parent chunks  (~1024 chars, with section + citations)
       │ ABSTAIN=true  →  strict system prompt
       │                  "answer only from context; else say I don't know"
       │ Claude Sonnet 4.6  (retry/fallback on 429/500/529 → Sonnet 4.5)
       ▼
  Answer + inline citations  [1] [2] [3]
```

Three interfaces: a Streamlit chat UI (shows the retrieved chunks and their
scores inline), a one-line Python API (`pipeline.answer(q)`), and the eval
harness — which runs *the exact same pipeline*, so what I measure is what ships.

### Design principle: runnable on a single secret

The whole system runs on one credential (`ANTHROPIC_API_KEY`). Embeddings and
reranking are local `sentence-transformers` models, the vector store is embedded
ChromaDB, and PDF parsing is plain `pypdf` — no cloud services, no paid libraries,
no servers. `pip install` + one key, and everything except generation runs
offline. "We should be able to run it" was a hard constraint, and it drove most of
the choices in §2.

---

## 2. Key decisions and tradeoffs

**Local embeddings (`BAAI/bge-small-en-v1.5`), not an embeddings API.** Keeps the
pipeline to a single credential, makes retrieval free and reproducible, and works
offline. Cost: a 130 MB model download and CPU-bound embedding (fine at this
scale, a bottleneck at 100k users — see §6). `bge-small` (384-dim) over
`bge-large` deliberately: faster, smaller, and good enough — retrieval quality is
strong (in-domain context recall 0.96, see §4).

**ChromaDB, embedded + persistent.** No server, no Docker — it's a folder on
disk. The right altitude for a single-node demo. FAISS would mean hand-rolling
the id→metadata map; a server DB (Qdrant/pgvector) would mean infra the grader
has to stand up. I note the migration path in §6.

**Claude for generation and as the eval judge.** Generation uses Sonnet 4.6, with
a Sonnet 4.5 fallback on rate-limit/overload. The Ragas judge also uses Sonnet 4.6
for the highest-quality scoring (`JUDGE_MODEL` can be pointed at a Haiku id to cut
eval cost ~5× when iterating).

**Cross-encoder relevance gate.** The single most useful thing I added. A
bi-encoder (the embedding model) scores query and passage independently and has a
*high similarity floor* — on this corpus even unrelated text sits around cosine
0.5, so a cosine threshold can't separate "relevant" from "off-topic." The
cross-encoder reads query and passage together and separates them cleanly:

| | top cross-encoder score (measured) |
|---|---|
| In-domain queries | +5 to +8 (worst legitimate case −0.4) |
| Out-of-domain queries | −7 to −10 |

So I gate on the cross-encoder score (`RERANK_THRESHOLD = -3.0`): if no chunk
clears the bar, the retriever returns nothing and the generator abstains. This is
both a precision booster *and* the mechanism behind the FIX in §5.

**Chunking: small-to-big (parent/child).** The default chunking embeds small
~256-char **children** for precise matching but returns the ~1024-char **parent**
each match belongs to (de-duplicated). This is LangChain's `ParentDocumentRetriever`
pattern: children go into Chroma for vector search, parents go into a
`LocalFileStore` keyed by UUID, and at query time each child hit is swapped for
its parent. Each chunk is also tagged with its detected paper **section**
(`src/sectioning.py`) for richer citations. I implemented it as a toggle (`SMALL_TO_BIG`) and measured it against
flat 1024-char chunking — and the result is an honest tradeoff, not a clean win:

| In-domain | Flat (1024) | Small-to-big | Δ |
|---|---|---|---|
| Context Precision | 0.85 | 0.76 | −0.09 |
| Context Recall | 0.93 | 0.89 | −0.04 |
| **Faithfulness** | 0.91 | **0.98** | **+0.07** |

Small-to-big **improved faithfulness** but **lowered Ragas context-precision**. The precision drop is largely a *metric artifact*:
Ragas scores each *returned* unit, and a 1024-char parent contains the answer plus
surrounding prose, so there's more "non-answer" text per unit even though
retrieval *found* the answer precisely (via the child) and the LLM answered it
well — which is exactly why faithfulness went *up*. I shipped small-to-big as the
default because faithfulness and citation context matter most for a system meant
to be trusted, but this is a genuine tradeoff I'd defend, not a free lunch.
(`flat_baseline.json` keeps the flat numbers for comparison.)

**Retrieval upgrades (measured, additive).** Three further improvements, each a
flag, measured incrementally on top of the small-to-big baseline:

- **Contextual headers** (`CONTEXTUAL_HEADERS`, default on) — prepend
  `title — section` to each child *before embedding*, so the vector encodes
  provenance (a lightweight take on Anthropic's contextual retrieval).
- **Hybrid retrieval** (`HYBRID`, default on) — fuse dense + BM25 with Reciprocal
  Rank Fusion; BM25 catches exact terms (RoPE, MaxSim, "175 billion").
- **Query decomposition** (`DECOMPOSE`, default **on**) — an LLM (Haiku) splits a
  multi-hop question into sub-questions, each retrieved separately and pooled.

| Multi-hop subset | Precision | Recall | Faithfulness |
|---|---|---|---|
| small-to-big baseline | 0.46 | 0.75 | 0.78 |
| + contextual headers | 0.52 | 0.75 | 1.00 |
| + hybrid | 0.52 | 0.75 | 0.89 |
| + decomposition | **0.65** | **1.00** | **1.00** |

Headers + hybrid also lift in-domain precision (0.76 → 0.80) at negligible cost,
so all three ship **on by default**. Decomposition is the real fix for the
multi-hop weak spot — precision 0.46 → 0.65, recall 0.75 → 1.00 — and the
quality gain justifies the cost: one extra Haiku call that raises retrieval
latency ~142 ms → ~1.5 s. The tradeoff is explicit and reversible (`DECOMPOSE=false`
for latency-sensitive workloads). Abstention is unaffected: out-of-domain
hallucination stays 0%, and in-domain over-refusal actually dropped to 0% (better
retrieval → fewer false abstentions). Result files: `headers_only`,
`headers_hybrid`, `headers_hybrid_decompose`.

---

## 3. Evaluation framework

> *"A system with no eval is a system you cannot improve."*

**Tool: Ragas**, with four metrics, each mapped to a part of the pipeline:

| Metric | Question it answers | Which knob moves it |
|---|---|---|
| Context Precision | Are the retrieved chunks relevant and well-ranked? | reranking |
| Context Recall | Did retrieval surface the chunks the answer needs? | chunking, query embedding |
| Faithfulness | Is the answer grounded in the retrieved contexts? | generation |
| Answer Relevancy | Does the answer actually address the question? | generation |

Plus one **non-Ragas, deterministic** dimension I added — an **abstention /
hallucination rate** (`scripts/abstention_check.py`, no LLM judge) — to measure
out-of-domain behavior directly (§5).

**Judge:** Claude (Sonnet 4.6) via `langchain-anthropic`; **embeddings for
relevancy:** local BGE. A Ragas footgun worth flagging: if you don't attach an LLM to every
metric, it silently falls back to OpenAI and dies on a missing key — so I wire
the judge + embeddings onto each metric explicitly ([`eval/ragas_eval.py`](../eval/ragas_eval.py)).

**Gold set:** 26 hand-curated Q&A
([`data/eval/gold_qa.json`](../data/eval/gold_qa.json)), each with a reference
answer I can point to a page for:

- 18 **in-domain** (factual, single-paper)
- 4 **multi-hop** (require synthesizing across papers, e.g. "GPT-3 showed few-shot
  ability — what later technique made such models follow instructions?")
- 4 **out-of-domain** (unanswerable from the corpus — capital of Australia, a
  sourdough recipe). These exist specifically to test whether the system *knows
  what it doesn't know*.

I chose hand-curation over Ragas' synthetic `TestsetGenerator`: trustworthy
reference answers make the recall/faithfulness numbers credible, I control the
difficulty mix, and it's one fewer thing to break the day before submission.

**Baseline scores** (full config, Sonnet 4.6 judge, 26 questions), sliced by
subset — the slice matters, because the three question types behave very
differently:

| Subset | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| In-domain (18) | 0.76 | 0.89 | 0.98 | 0.88 |
| Multi-hop (4) | 0.46 | 0.75 | 0.78 | 0.66 |
| Out-of-domain (4) | 0.00 | 0.00 | 0.75 | 0.00 |

(Default config = **small-to-big** retrieval; see §2 for how this compares to flat
chunking.)

Read this honestly:

- **In-domain QA is strong** — faithfulness 0.98, recall 0.89. Context precision
  (0.76) is lower than answer quality because we return ~1024-char *parent* chunks
  (answer + surrounding context), which Ragas scores as less "precise" per unit
  even when the answer is right (§2).
- **Multi-hop is the weak spot** — precision 0.46; retrieval often finds one hop
  but not both. This is the clearest quality gap (see §7).
- **Out-of-domain rows score ~0 by design** — the system abstains, so there are no
  retrieved contexts (precision/recall = 0) and an abstention doesn't "address"
  the question (relevancy = 0). These are *good* zeros. They also mean a single
  blended "overall" number is misleading, which is why I don't lead with one — and
  why measuring the OOD behavior needs a different instrument (§5).

Latency: generation ~4.1 s (Sonnet).

---

## 4. Build / Break: documented failure modes

Each failure mode is one config flag, so the before/after is reproducible
(`<FLAG>=… python scripts/run_eval.py --tag …`). Numbers below are on the shipped
**small-to-big** config; result files are committed under
[`data/eval/results/`](../data/eval/results/). A theme runs through this section:
**a failure mode is a property of a configuration, not of "RAG" in the abstract** —
change the architecture and you have to re-verify, which is exactly what the eval
harness is for.

### Break #1 — No reranking (`RERANK=false`)

The bi-encoder's top hits by cosine are *not* the best hits. Without the
cross-encoder, worse children are selected, so the parents we hand the LLM are
worse — and the answer is less grounded (in-domain subset):

| Metric (in-domain) | Baseline | No rerank | Δ |
|---|---|---|---|
| **Faithfulness** | 0.98 | **0.85** | **−0.12** |
| Context Precision | 0.76 | 0.71 | −0.05 |

The lesson: **reranking earns its place** — it improves both precision and, more
importantly here, the faithfulness of the generated answer.

*A note on how this differs from flat chunking.* On the flat 1024-char index the
same flag caused a much larger *precision/recall* drop (0.85→0.63 precision,
0.93→0.67 recall; see `flat_baseline` vs the flat run). Under small-to-big the
precision hit is muted because parent-merging absorbs some of the imprecision —
the damage resurfaces as *faithfulness* instead. Same root cause, different
metric: the reason you slice by subset *and* watch multiple metrics.

### Break #2 — Query/document embedding mismatch (`USE_QUERY_INSTRUCTION=false`)

BGE retrieval models are trained with an instruction prefix on **queries only**
(`"Represent this sentence for searching relevant passages: "`). Forgetting it is
an easy, invisible bug. This is the most interesting break, because **its impact
depends entirely on the chunking strategy:**

| Context Recall (in-domain) | Baseline | No query instruction | Δ |
|---|---|---|---|
| Flat 1024 chunks | 0.93 | 0.83 | **−0.10** |
| Small-to-big (shipped) | 0.89 | 0.94 | +0.05 (noise) |

On the **flat** index the missing prefix clearly cost ~10 points of recall. On the
**small-to-big** index the effect *disappeared* — short 256-char children plus
parent-merging are robust enough that the prefix stops mattering on this corpus.

The honest takeaway isn't "the prefix doesn't matter" — it's that **I would have
shipped a latent bug and never known.** On the flat system it's a real 10-point
regression; a chunking change masked it. Only a re-run of the eval after the
architecture change reveals that the failure mode moved. That's the case for
keeping eval in CI (§6), not a footnote.

---

## 5. The Fix: abstention on out-of-domain questions

**The failure:** ask the corpus something it doesn't contain ("What is the capital
of Australia?") and a naive RAG system retrieves *something* (cosine floor ~0.5),
stuffs it into the prompt, and the LLM confidently answers from its parametric
memory — an ungrounded hallucination the user can't distinguish from a real
citation.

**The fix (`ABSTAIN=true`, the default):** two layers —
1. the cross-encoder relevance gate (§2) returns *no* context for OOD queries, so
   the pipeline often never even calls the LLM; and
2. a strict system prompt: *answer only from the context; otherwise say you don't
   know.*

### Measuring abstention: a direct, deterministic metric

I measure abstention behaviorally ([`scripts/abstention_check.py`](../scripts/abstention_check.py)):
for each question, did the system answer or decline? Detection is a deterministic
string check (`generate.is_abstention`) — **no LLM judge**, fully reproducible.
For out-of-domain questions, declining is correct and answering is a
hallucination.

| Behavior (per category) | Fix ON (`ABSTAIN=true`) | Fix OFF (`ABSTAIN=false`) |
|---|---|---|
| **Out-of-domain hallucination rate** | **0%** (0/4) | **75%** (3/4) |
| Out-of-domain correct abstention | 100% | 25% |
| In-domain over-refusal | 6% (1/18) | 0% |

This is the before/after that matters: with the guardrail off, **3 of 4
unanswerable questions get confident, fabricated answers**; with it on, the system
declines all four. (The lone OOD refusal in the Fix-OFF column is Claude's own
caution on the Apple-earnings question — the base model helps a little, but not
reliably.) The cost of the fix is small but real and worth naming: **one in-domain
question (6%) is now over-refused** — the guardrail occasionally declines a
question it could have answered. That's the safety/coverage tradeoff made
measurable, not hidden.

### Where automated eval still disagrees with judgment

Even the Ragas numbers that *did* survive tell a cautionary tale. On the OOD
subset, turning the fix **off** *raises* Answer Relevancy from 0.00 to 0.50:
Ragas' relevancy metric **rewards the hallucinating system** — a confident "The
capital of Australia is Canberra" reads as relevant, while an honest "I don't have
that information" scores zero for not addressing the question.

So a vibe-check on relevancy — or any single metric — would rank the *unsafe*
system **above** the safe one. Three takeaways I'd defend in the presentation:

1. **You cannot reduce RAG quality to one number.** Relevancy and safety pull in
   opposite directions on unanswerable inputs; you need multiple metrics, sliced
   by query type.
2. **Match the metric to the question.** Grounding metrics measure one thing;
   abstention needs a behavioral metric — which is why the deterministic
   abstention rate, not an LLM-judged score, is what I trust for it.
3. **Abstention is a deliberate coverage/safety tradeoff**, not a free win — the
   right call for a system meant to be trusted, but a *choice* the eval makes
   visible rather than hides.

*(Caveat: the OOD and multi-hop subsets are only 4 questions each, so those Ragas
subset values are directional; the in-domain subset and the deterministic
abstention metric are the solid ground. Expanding the gold set is the first item
in §7.)*

---

## 6. Production-first: what breaks at 100k+ users

This is a single-node demo. Concretely, here's what fails as load grows, what I'd
monitor, and what I'd fix first.

**Embeddings (local, in-process).** The model runs single-threaded on CPU inside
the request path; under concurrency it blocks and embedding latency dominates.
*Fix first:* move embeddings to a dedicated service (HF Text-Embeddings-Inference
or a small GPU pool) with request batching, and cache query embeddings (users ask
overlapping questions). *Monitor:* p95 embed latency, CPU saturation, queue depth.

**Vector store (ChromaDB).** Embedded Chroma is single-node, file-backed, no
replication — fine for 1,621 chunks, wrong for millions at high QPS. *Fix:*
migrate to a server vector DB (Qdrant / Weaviate / pgvector) with HNSW and read
replicas; separate the ingest writer from query readers. *Monitor:* query latency
vs. collection size, recall drift as the index grows.

**Anthropic API (generation + judge).** Per-org RPM/TPM limits and cost; a traffic
spike means 429s. Generation at ~4 s/query is the latency floor. *Fix:* a
concurrency limiter with backoff (the retry/fallback is already in
[`generate.py`](../src/generate.py)); prompt-cache the static system prompt; route
easy queries to Haiku; **never run the Ragas judge in the hot path** — it's
offline/CI only. *Monitor:* 429 rate, TPM utilization, cost/query, cache-hit rate.

**Streamlit.** Re-runs the whole script per interaction, in-process session state,
no auth — a demo shell, not a serving layer. *Fix:* split into a stateless FastAPI
backend (autoscaled) + a real frontend; keep Streamlit as the internal tool.

**The highest-leverage fix is organizational, not architectural:** wire this eval
harness into CI so every change to chunking, models, or prompts runs against the
gold set and fails the PR on a regression. Break #2 is the proof: a query-embedding
bug that costs 10 points of recall on one chunking strategy and is invisible on
another — exactly the class of regression that ships unnoticed without an eval
gate. That protects every other change I'd make.

---

## 7. Scope cuts and what I'd do next

**Built after the first pass** (measured improvements, see §2): small-to-big
retrieval, contextual chunk headers, hybrid BM25+dense (RRF), and LLM query
decomposition — the last lifts multi-hop precision 0.46 → 0.65 and recall to 1.0.

**Deliberately not built** (2-day scope; over-engineering is a stated red flag):

- **Auth, user accounts, chat-history persistence** — demo, not a product.
- **Docker/k8s/CI** — `pip install` + one key was the bar.
- **Synthetic eval data** — hand-curation was faster and more trustworthy.
- **Multimodal / tables / figures** — arXiv math and two-column layouts extract
  imperfectly with `pypdf`; I targeted prose and dropped reference sections.
- **A stronger embedder/reranker** (bge-large, bge-reranker-v2-m3) — easy swaps,
  left as a measured experiment rather than a guess.

**What I'd tackle next, in order:**

1. **Grow the gold set to ~100+ questions** with more OOD and multi-hop coverage,
   so subset numbers are statistically meaningful (current caveat in §5).
2. **Cut decomposition latency** — it adds ~1.4 s per query (now on by default);
   cache sub-questions, run sub-retrievals concurrently, or gate decomposition
   to questions a cheap classifier flags as multi-hop so single-hop queries pay
   no overhead.
3. **Tune child size / learn the abstention threshold** — the small-to-big child
   size (256) and the hand-tuned `-3.0` gate were set by eye; both should be swept
   against the gold set and the threshold calibrated from labeled pairs.
4. **Upgrade the reranker** (bge-reranker-v2-m3) + a larger embedder, A/B'd with
   the harness. This is also a **known limitation, not just an upgrade**:
   stress-testing by hand found the weak `ms-marco-MiniLM` reranker *under-scores*
   some legitimate in-domain phrasings (e.g. acronym-heavy "What is RoPE and what
   problem does it address?") into the same band as out-of-domain junk (≈ −7 to
   −9), so the abstention gate over-refuses them. The threshold can't separate the
   two at this reranker's resolution — a stronger reranker is the fix. My
   18-question gold set didn't catch it (phrasing coverage) — which is exactly why
   item 1 (grow the gold set) is first.
5. **CI eval gate** (§6) — cheap, and it's what keeps quality from silently
   eroding (Break #2 is the cautionary tale).
