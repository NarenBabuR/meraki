# RAG Pipeline: Build, Break, Fix — Write-up

**Author:** Naren · **Problem:** Meraki Labs PS1 · **Corpus:** ~20 arXiv ML papers

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

```
PDFs ─ extract ─ chunk ─ embed (local BGE) ─► ChromaDB (1,621 chunks)
query ─ embed ─ vector search (top 25) ─ cosine filter
      ─ cross-encoder rerank ─ relevance gate ─ top 5
      ─ Claude (grounded prompt, abstains if nothing relevant) ─► answer + sources
```

Three interfaces: a Streamlit chat UI (shows the retrieved chunks and their
scores inline), a one-line Python API (`pipeline.answer(q)`), and the eval
harness — which runs *the exact same pipeline*, so what I measure is what ships.

### Reuse vs. rebuild

I was given an existing production repo (`file-processor`) that already does
document extraction, chunking, embedding, retrieval, and reranking. It is a
capable system, but it's built on AWS Bedrock (Cohere embeddings + rerank),
Redis-as-vector-store, SQS workers, S3, and paid PDF libraries (PyMuPDF Pro,
Aspose). Handing a grader something that needs five cloud credentials to boot
fails the "we should be able to run it" bar.

So I **ported the patterns, not the infrastructure**:

| Pattern (from `file-processor`) | What I kept | What I changed |
|---|---|---|
| Markdown-aware recursive chunking + dedupe | Recursive splitter, dedupe of repeated headers/footers | Dropped the markdown-header splitter (arXiv PDFs aren't markdown) |
| `search_query` / `search_document` embedding asymmetry | The asymmetry, as BGE's query-instruction prefix | Cohere Bedrock → local `bge-small` |
| Two-stage `top_n → rerank → top_k` retrieval | The wide-then-narrow structure | Added an explicit cross-encoder relevance gate |
| Cohere rerank v3.5 (Bedrock) | The rerank step | Local `cross-encoder/ms-marco-MiniLM` (no AWS) |
| Claude wrapper with retry/fallback | Retry on 429/500/529, Sonnet→Sonnet-4.5 fallback | Stripped the usage-tracker and 4-tier logic |

Result: one secret (`ANTHROPIC_API_KEY`), `pip install`, runs offline for
everything except generation.

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
hallucination rate** (`scripts/abstention_check.py`, no LLM judge). §5 explains why
Ragas was the wrong tool for that one and this is the right one.

**Judge:** Claude (Sonnet 4.6) via `langchain-anthropic`; **embeddings for
relevancy:** local BGE. A Ragas footgun worth flagging: if you don't attach an LLM to every
metric, it silently falls back to OpenAI and dies on a missing key — so I wire
the judge + embeddings onto each metric explicitly ([`eval/ragas_eval.py`](../eval/ragas_eval.py)).

I also ran the full suite under a Haiku judge while iterating; the headline
ranking of configs was stable, but one metric (faithfulness on out-of-domain
questions) swung wildly between judges — which turned out to be a finding in
itself, see §5.

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
| In-domain (18) | 0.85 | 0.93 | 0.91 | 0.92 |
| Multi-hop (4) | 0.57 | 0.75 | 0.77 | 0.65 |
| Out-of-domain (4) | 0.00 | 0.00 | 0.50 | 0.00 |

Read this honestly:

- **In-domain QA is strong** — precision 0.85, recall 0.93, faithfulness 0.91.
- **Multi-hop is the weak spot** — precision drops to 0.57; retrieval often finds
  one hop but not both. This is the clearest quality gap (see §7).
- **Out-of-domain rows score ~0 by design** — the system abstains, so there are no
  retrieved contexts (precision/recall = 0) and an abstention doesn't "address"
  the question (relevancy = 0). These are *good* zeros. They also mean a single
  blended "overall" number is misleading, which is why I don't lead with one — and
  why measuring the OOD behavior needs a different instrument (§5).

Latency: retrieval ~327 ms mean (p95 729 ms), generation ~3.7 s (Sonnet).

---

## 4. Build / Break: two documented failure modes

Each failure mode is one config flag, so the before/after is reproducible
(`<FLAG>=… python scripts/run_eval.py --tag …`). Result files are committed under
[`data/eval/results/`](../data/eval/results/).

### Break #1 — No reranking (`RERANK=false`)

The bi-encoder's top-5 by cosine is *not* the best top-5. Measured on the
**in-domain** subset (where retrieval quality is the whole story):

| Metric (in-domain) | Baseline | No rerank | Δ |
|---|---|---|---|
| Context Precision | 0.85 | **0.63** | **−0.22** |
| Context Recall | 0.93 | 0.67 | **−0.26** |
| Answer Relevancy | 0.92 | 0.82 | −0.10 |

Context precision falls 0.85 → 0.63 and recall 0.93 → 0.67. The lesson: with a
high-floor bi-encoder, **reranking isn't a nice-to-have — it's what makes
retrieval precise.**

*Why in-domain, not overall?* At the overall level the precision drop looks small
(0.67 → 0.62) — but that's a measurement artifact, and a useful one to understand.
Turning off rerank also removes the relevance gate, so out-of-domain queries now
retrieve (irrelevant) chunks instead of abstaining; Ragas then scores those OOD
contexts non-zero, which *inflates* the overall average and masks the real
in-domain damage. Slicing by subset is what makes the true effect legible — a
recurring theme in this eval.

### Break #2 — Query/document embedding mismatch (`USE_QUERY_INSTRUCTION=false`)

BGE retrieval models are trained with an instruction prefix on **queries only**
(`"Represent this sentence for searching relevant passages: "`). Forgetting it —
an easy, invisible bug — embeds queries and documents in subtly mismatched ways:

| Metric (in-domain) | Baseline | No query instruction | Δ |
|---|---|---|---|
| Context Recall | 0.93 | 0.83 | **−0.10** |
| Context Precision | 0.85 | 0.80 | −0.05 |

Honest finding: the degradation is **real but graceful** — recall drops ~10
points, precision ~5 — not catastrophic, because `bge-small` still retrieves
reasonably without the prefix. I'm documenting rather than fixing it because (a)
it's already correct in the baseline, and (b) it's exactly the kind of silent
single-digit regression that an eval harness catches and a vibe-check never would.
At 100k queries/day, 10 points of recall is a lot of unanswered questions.

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

### First attempt to measure it — and why Ragas faithfulness was the wrong tool

My instinct was to measure this with Ragas faithfulness on the OOD subset. That
turned out to be a trap, and the trap is instructive. Under a **Haiku** judge,
turning the fix off dropped OOD faithfulness 0.50 → 0.18 — a clean story. Under
the **Sonnet** judge, the same comparison was 0.50 → 0.50 — *no signal at all.*

The reason: faithfulness scores whether the answer's claims are supported by the
**retrieved context**. But a correctly-gated OOD query has *no* retrieved context
(an empty placeholder), so "faithfulness to nothing" is degenerate — and different
judges resolve that degenerate case differently. Faithfulness is the right metric
for "did the answer stick to the documents," but it is the **wrong instrument for
measuring abstention**, and it's judge-dependent exactly where I needed it to be
solid. That's a real lesson about not trusting a metric outside the regime it was
designed for.

### The right tool: a direct, deterministic abstention metric

So I measure the behavior directly ([`scripts/abstention_check.py`](../scripts/abstention_check.py)):
for each question, did the system answer or decline? Detection is a deterministic
string check (`generate.is_abstention`) — **no LLM judge**, fully reproducible.
For out-of-domain questions, declining is correct and answering is a
hallucination.

| Behavior (per category) | Fix ON (`ABSTAIN=true`) | Fix OFF (`ABSTAIN=false`) |
|---|---|---|
| **Out-of-domain hallucination rate** | **0%** (0/4) | **75%** (3/4) |
| Out-of-domain correct abstention | 100% | 25% |
| In-domain over-refusal | 0% (answers all 18) | 0% |

This is the before/after that matters: with the guardrail off, **3 of 4
unanswerable questions get confident, fabricated answers**; with it on, the system
declines all four — and crucially does **not** over-refuse, answering every
in-domain question. (The lone OOD refusal in the Fix-OFF column is Claude's own
caution on the Apple-earnings question — a reminder the base model helps a little,
but not reliably.)

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
2. **Match the metric to the question.** Faithfulness measures grounding;
   abstention needs a behavioral metric. Using the former for the latter gave me a
   number that flipped with the judge.
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
gold set and fails the PR on a regression. Break #2 — a silent 5-point recall drop
— is exactly the class of bug that ships unnoticed without it. That protects every
other change I'd make.

---

## 7. Scope cuts and what I'd do next

**Deliberately not built** (2-day scope; over-engineering is a stated red flag):

- **Auth, user accounts, chat-history persistence** — demo, not a product.
- **Docker/k8s/CI** — `pip install` + one key was the bar.
- **Hybrid (BM25 + dense) retrieval** — dense + cross-encoder is sufficient here
  and avoids a second index. The clearest *next* upgrade.
- **Multi-hop / agentic query decomposition** — would directly attack the weakest
  measured subset (multi-hop recall 0.50), but it's a rabbit hole I scoped out.
- **Synthetic eval data** — hand-curation was faster and more trustworthy.
- **Multimodal / tables / figures** — arXiv math and two-column layouts extract
  imperfectly with `pypdf`; I targeted prose and dropped reference sections.

**What I'd tackle next, in order:**

1. **Grow the gold set to ~100+ questions** with more OOD and multi-hop coverage,
   so subset numbers are statistically meaningful (current caveat in §5).
2. **Multi-hop retrieval** — query decomposition / sub-question retrieval to lift
   the 0.50 multi-hop recall, the clearest quality gap.
3. **Learn the abstention threshold** instead of the hand-tuned `-3.0` — calibrate
   it from labeled relevant/irrelevant pairs rather than eyeballing the
   distribution.
4. **Hybrid retrieval** for exact-term queries (model names, metrics, numbers)
   where dense embeddings underperform sparse matching.
5. **CI eval gate** (§6) — cheap, and it's what keeps quality from silently
   eroding.
