# Code Walkthrough — how the whole thing works

A guided tour of every file, the two workflows, and how a question becomes an
answer. Read this top-to-bottom and you'll understand the entire system. It
pairs with [WRITEUP.md](WRITEUP.md) (the *why* / decisions) — this doc is the
*how* / mechanics.

---

## 0. The 30-second mental model

There are **two workflows**:

1. **Index-time** (run once, `scripts/build_index.py`): download papers → tag
   sections → split into big *parent* chunks, then small *child* chunks → embed
   the children → store children in a local vector DB and parents in a sidecar.
2. **Query-time** (every question, `pipeline.answer()`): turn the question into a
   vector → find similar children → re-score them with a smarter model → keep the
   best → **expand each match to its parent chunk** → hand those to Claude → get a
   grounded answer (or an honest "I don't know").

> This is **small-to-big retrieval**: match on small, focused children (precise),
> but give the LLM the larger parent (enough context). It's the default;
> `SMALL_TO_BIG=false` falls back to flat single-size chunks.

Everything else (config, eval, the app) supports those two paths.

```
INDEX-TIME (once)
  arXiv PDFs ─► extract ─► tag sections ─► parent (1024) → child (256) ─► embed children
                                                                              │
                                                          ChromaDB (children) + parents.json
QUERY-TIME (per question)                                                     ▼
  question ─► embed ─► vector search (25) ─► cosine filter ─► rerank ─► gate
                                          ─► merge children→parents (top 5)
                                                                              │
                                                                              ▼
                                                          Claude (grounded, can abstain)
                                                                              ▼
                                                          answer + cited sources (w/ section)
```

The repo is small on purpose (~12 source files). No frameworks, no abstraction
layers — just plain functions and one dataclass per concept.

---

## 1. `config.py` — the control panel

**Everything tunable lives here, and every setting can be overridden by an
environment variable.** This is the single most important file to understand,
because the entire "Build / Break / Fix" story is just flipping values defined
here.

How it works:

- It's a frozen `@dataclass` called `Config`. Each field reads from an env var
  with a default, via tiny helpers (`_bool`, `_int`, `_float`). Example:
  ```python
  rerank: bool = _bool("RERANK", True)          # RERANK=false flips it
  chunk_size: int = _int("CHUNK_SIZE", 1024)
  ```
- `load_dotenv()` runs at import, so your `.env` (with `ANTHROPIC_API_KEY`) is
  loaded automatically.
- `CONFIG = Config()` is a module-level singleton everything imports.
- It defines the **paths** (`PDF_DIR`, `CHROMA_DIR`, `GOLD_QA_PATH`, …) so no file
  location is hard-coded elsewhere.
- `snapshot()` returns the config as a dict — this gets saved into every eval
  result file, so you can always tell which settings produced which scores.
- `describe()` prints a one-line summary of the toggles.

The settings that matter:

| Field | Default | What it controls |
|---|---|---|
| `chunk_size` / `chunk_overlap_ratio` | 1024 / 0.2 | how text is split (changing → re-index) |
| `embed_model` | `BAAI/bge-small-en-v1.5` | the local embedding model |
| `use_query_instruction` | `True` | BGE query prefix — **Break #2** |
| `top_n` / `top_k` | 25 / 5 | wide candidate pool / final chunks |
| `sim_threshold` | 0.30 | coarse cosine filter |
| `rerank` | `True` | cross-encoder reranking — **Break #1** |
| `rerank_threshold` | -3.0 | relevance gate (drives abstention) |
| `abstain` | `True` | refuse when nothing relevant — **the FIX** |
| `gen_model` | `claude-sonnet-4-6` | generation model |
| `judge_model` | `claude-sonnet-4-6` | the Ragas eval judge |

> **Key idea:** because config is read from the environment, you run an experiment
> like `RERANK=false python scripts/run_eval.py --tag no_rerank` — no code edits.

---

## 2. Index-time: building the searchable corpus

### 2.1 `src/ingest.py` — get the papers, extract the text

**`DEFAULT_ARXIV_IDS`** — a hard-coded list of ~20 foundational ML paper IDs
(Transformer, BERT, GPT-3, RAG, LoRA, …). Pinned by ID (not a search query) so the
corpus is identical for anyone who runs it.

**`Page`** (dataclass) — one page of one paper: `{arxiv_id, title, page, text}`.

**`download_papers()`**
- Uses the `arxiv` library to fetch each paper's metadata (title, pdf URL).
- Downloads the PDF with `urllib`, **using `certifi`'s certificate bundle** —
  this works around the macOS/python.org SSL issue where downloads fail with
  "certificate verify failed."
- Skips files already on disk, sleeps 3s between downloads (polite to arXiv).
- Returns `[{arxiv_id, title, path}, …]`.

**`extract_pages(pdf_path, …)`**
- Opens the PDF with `pypdf`, reads text page by page.
- `_clean()` normalizes whitespace.
- **Stops at the References/Bibliography section** — citation lists are retrieval
  noise, so we cut the index off there.
- Returns a list of `Page` objects (skips empty pages).

**`ingest_all(papers)`** — runs `extract_pages` over every paper, returns all
`Page`s flattened into one list.

> Why `pypdf` and not the fancy PyMuPDF Pro the original `file-processor` used?
> arXiv papers are born-digital, so plain `pypdf` extracts them fine — and it's
> free and dependency-light. Math/two-column layouts extract imperfectly; we
> compensate by chunking carefully and de-duplicating (next files).

### 2.2a `src/sectioning.py` — find the section each piece of text belongs to

Before chunking we reconstruct each paper's full text and locate its sections, so
every chunk can be tagged with where it came from (better citations + a light
retrieval signal).

- **`group_by_paper(pages)`** — groups the flat page list back into per-paper
  lists (order preserved).
- **`join_pages(pages)`** — concatenates one paper's pages into `full_text`, and
  returns `bounds` (`[(start, end, page_no), …]`) so any character offset can be
  mapped back to its page.
- **`detect_sections(full_text)`** — a regex (`SECTION_PAT`) finds headings:
  numbered ones like "3 Method" / "3.1 Architecture" (capital required after the
  number, so body sentences don't match) and named ones (Abstract, Introduction,
  Results, Conclusion, …). Returns `[(offset, title), …]`.
- **`page_at(idx, bounds)` / `section_at(idx, sections)`** — map an offset to its
  page number / enclosing section title.

> Best-effort by design: PDF extraction is messy, so some headings are missed
> (you'll see artifacts like "§1 I NTRODUCTION"). Section is *metadata*, not the
> load-bearing part — so rough detection is fine.

### 2.2b `src/chunking.py` — split into chunks (small-to-big or flat)

A whole page is too coarse to retrieve precisely; a sentence is too small to
answer from. Small-to-big resolves this by keeping **two** sizes.

**`IndexItem`** (dataclass) — the uniform unit both modes emit: `{id, text,
metadata}`. `metadata` carries `arxiv_id, title, page, section` and (in
small-to-big) a `parent_id` linking the child to its parent.

**`build_chunks(pages)`** — the dispatcher. Returns `(items_to_embed, parents)`:
- **Small-to-big** (`_build_hierarchical`, the default):
  1. For each paper, reconstruct `full_text` + detect sections.
  2. Split into **parent** chunks (~1024 chars) using the recursive splitter with
     `add_start_index=True`, so each parent knows its offset → page + section.
  3. Split each parent into **child** chunks (~256 chars).
  4. Emit one `IndexItem` per **child** (this is what gets embedded), each tagged
     with its `parent_id`, page, and section. Build a `parents` dict
     `{parent_id: {text, arxiv_id, title, page, section}}`.
  - The children are what we search; the parents are what we later show the LLM.
- **Flat** (`chunk_pages`, `SMALL_TO_BIG=false`): per-page ~1024-char chunks, no
  parents. Each chunk is both embedded and shown.

Both modes use the recursive splitter (`SEPARATORS = ["\n\n", "\n", ". ", " ",
""]` — split on paragraph breaks first, then lines, sentences, spaces) and
**de-duplicate** (a `seen` set) to drop the running headers/footers arXiv PDFs
repeat on every page. Fragments under 40 chars are skipped.

> Ported from `file-processor/src/embeddings.py` (recursive splitter + dedupe),
> extended with the parent/child structure and section tagging; the original's
> markdown-header splitter is dropped (irrelevant for PDFs).

### 2.3 `src/embeddings.py` — turn text into vectors (locally)

**`_model()`** — loads the `sentence-transformers` model named in
`CONFIG.embed_model` (BGE-small). Wrapped in `@lru_cache(maxsize=1)` so the
~130 MB model loads **once per process** and is reused.

**`embed_documents(texts)`** — embeds corpus chunks. `normalize_embeddings=True`
so vectors are unit-length (required for cosine similarity to behave). Returns a
list of float lists.

**`embed_query(query)`** — embeds a **question**. The important bit:
```python
if CONFIG.use_query_instruction:
    text = CONFIG.query_instruction + query   # "Represent this sentence for searching relevant passages: " + query
```
BGE retrieval models are trained with an instruction prefix on **queries only**
(documents get no prefix). Matching that asymmetry is what makes retrieval work
well; dropping it is **Break #2**. This mirrors the `search_query` vs
`search_document` distinction in the original Cohere-based code.

### 2.4 `src/vectorstore.py` — store and search vectors (ChromaDB)

ChromaDB runs **embedded** (no server) and **persistent** (saved to
`data/chroma/`). It's just a folder on disk.

**`get_collection(reset=False)`** — opens (or creates) the collection, configured
with `{"hnsw:space": "cosine"}` so it uses cosine distance to match our
normalized BGE vectors. `reset=True` deletes and recreates it (used when
rebuilding the index).

**`index_items(items, parents)`** — the write path:
- Embeds the items (the **children** in small-to-big) in batches of 256 and
  `col.add(ids, embeddings, documents, metadatas)`.
- Writes the `parents` dict to a JSON **sidecar** (`data/chroma/parents.json`).
  Parents aren't searched by similarity, so rather than give them dummy vectors in
  Chroma we just store them by id and look them up at query time.
- Returns the count stored.

**`get_parent(parent_id)`** — loads `parents.json` (cached) and returns one
parent. Used by the retriever to expand a matched child into its parent.

**`query(query_embedding, top_n)`** — the read path:
- `col.query(...)` returns the `top_n` nearest items (children).
- Chroma returns cosine **distance** (`1 − similarity`); we convert it back to a
  **similarity** (higher = more similar) and return
  `[{chunk_id, text, metadata, similarity}, …]` sorted best-first. The `metadata`
  carries `parent_id` and `section`.

**`assert_dim_matches()`** — a safety check: if the stored vectors' dimension
doesn't match the current embedding model, it errors with "wipe and re-index."
Catches the classic "I changed `EMBED_MODEL` but forgot to rebuild" bug.

### 2.5 `scripts/build_index.py` — the orchestrator

Wires the above together in order:
```
download_papers() → ingest_all() → build_chunks() → index_items()
```
Run it once: `python scripts/build_index.py`. On this corpus (small-to-big) it
produces **~7,250 children + ~1,620 parents** in a ~38 MB Chroma store.

---

## 3. Query-time: answering a question

### 3.1 `src/rerank.py` — re-score candidates with a smarter model

**`_model()`** — loads a `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`), cached once.

**`rerank(query, candidates, top_k)`**
- Builds `(query, chunk_text)` pairs and scores each with the cross-encoder.
- Adds a `rerank_score` to each candidate, sorts descending, returns the best
  `top_k`.

> **Why two models?** The embedding model (a *bi-encoder*) encodes the query and
> each chunk *separately* — fast, but it has a high "similarity floor" (even
> unrelated text scores ~0.5). The cross-encoder reads the query and chunk
> *together*, so it judges relevance far more sharply. Measured on this corpus:
> relevant chunks score +5 to +8, off-topic chunks score −7 to −10. That gap is
> what powers both precision and the abstention gate. It replaces the Cohere
> reranker from `file-processor` (which needed AWS) with a free local model.

### 3.1a `src/lexical.py` & `src/decompose.py` — hybrid + multi-hop helpers

- **`lexical.py`** builds a **BM25** index over the same documents Chroma holds
  (cached), and `search(query, top_n)` returns sparse keyword matches. Used for
  hybrid retrieval — BM25 catches exact tokens dense embeddings blur (model names,
  numbers). Toggle: `HYBRID`.
- **`decompose.py`** asks a cheap LLM (Haiku) to split a multi-hop question into
  standalone sub-questions (single-fact questions pass through unchanged). Toggle:
  `DECOMPOSE` (off by default — it adds an LLM call).

### 3.2 `src/retriever.py` — the full retrieval pipeline

`retrieve(question)` ties it together. Step by step:

```python
queries = decompose(question) if CONFIG.decompose else (question,)  # 1. multi-hop split
pool = {}                                                           # 2. per (sub-)query:
for q in queries:                                                   #    dense ⊕ BM25 (RRF)
    for c in _candidates_for(q): pool.setdefault(c["chunk_id"], c)  #    pooled, de-duped
candidates = list(pool.values())
if not CONFIG.hybrid:                                               # 3. cosine pre-filter
    candidates = [c for c in candidates if c["similarity"] >= sim_threshold]
if CONFIG.rerank:                                                   # 4. rerank vs ORIGINAL q
    reranked = rerank(question, candidates, top_k=len(candidates))
    ranked = [c for c in reranked if c["rerank_score"] >= rerank_threshold]  # + GATE
else:
    ranked = sorted(candidates, key=lambda c: c.get("similarity") or 0, reverse=True)
if not ranked: return []
if CONFIG.small_to_big: return _merge_to_parents(ranked)            # 5. children → parents
return ranked[:top_k]
```

`_candidates_for(q)` does the dense search and, when `HYBRID` is on, fuses it with
BM25 via **Reciprocal Rank Fusion** (`_rrf`, k=60). The two-stage shape
(**wide search → narrow rerank**) is the pattern from `file-processor`. Parts to
highlight:

- **The gate** (step 4): on an out-of-domain question every child scores below
  `rerank_threshold` (−3.0), so `ranked` is empty → `retrieve` returns `[]` → the
  generator abstains. This is the safety mechanism.
- **`_merge_to_parents`** (step 5, small-to-big): walks the ranked children
  best-first, and for each one fetches its parent via `vectorstore.get_parent` and
  adds it to the output — **skipping parents already added**, so multiple children
  from the same parent collapse to one context. Stops at `top_k` parents. Each
  returned context carries the parent text, the parent's metadata (incl.
  `section`), and the matched child's text (`matched_child`) for display. So we
  *search* precise children but *return* coherent parents.

### 3.3 `src/generate.py` — produce the answer with Claude

**Constants**
- `RETRYABLE = {429, 500, 529}` — HTTP statuses worth retrying.
- `FALLBACK_MODEL = "claude-sonnet-4-5"` — used if the primary model keeps
  failing.
- `NO_CONTEXT_ANSWER` — the canned "I don't have enough information…" reply.
- `_ABSTENTION_MARKERS` + `is_abstention(answer)` — a deterministic check for
  whether an answer is a refusal. Used by the abstention eval metric (no LLM judge
  needed).

**Two system prompts**
- `_SYSTEM_ABSTAIN` (used when `ABSTAIN=true`): "Answer ONLY from the context; if
  it's not there, say exactly '<NO_CONTEXT_ANSWER>'; cite passages by number."
- `_SYSTEM_NO_ABSTAIN` (when `ABSTAIN=false`): "Answer as best you can" — the
  permissive prompt that lets the model hallucinate. This is the FIX toggle.

**`generate_answer(question, contexts)`**
1. If `abstain` is on **and** `contexts` is empty → return `NO_CONTEXT_ANSWER`
   immediately, **without calling Claude** (saves a call, guarantees no
   hallucination on gated OOD queries).
2. Pick the system prompt based on `abstain`.
3. Format the contexts as numbered, cited blocks (`_format_context`).
4. Call Claude (`_call`) with `temperature=0`. On a retryable error, back off and
   retry up to 3×; if the primary model still fails, fall back to Sonnet 4.5.

> Simplified from `file-processor/src/utils/ClaudeAnthropic.py` — same
> retry/fallback idea, none of the usage-tracking or 4-tier complexity.

### 3.4 `src/pipeline.py` — tie it together

**`RAGResult`** (dataclass) — `{question, answer, contexts, retrieval_ms,
generation_ms}` plus helpers `context_texts` and `total_ms`.

**`answer(question)`**
```python
contexts = retrieve(question)            # timed
text = generate_answer(question, contexts)  # timed
return RAGResult(question, text, contexts, retrieval_ms, generation_ms)
```
That's it. It returns **both the answer and the contexts**, which is exactly what
the evaluation needs — so eval runs the *same* pipeline users hit. "What you
measure is what you ship."

### 3.5 `app.py` — the Streamlit chat UI

- Sidebar shows the active config (so during a demo you can see which toggles are
  set).
- Chat input → calls `pipeline.answer()` → renders the answer.
- An expander shows **every retrieved chunk** with its title, page, cosine score,
  and rerank score — so you can *see* what the model was given.
- If no chunks survived the gate, it shows an "abstaining" notice.

Run: `streamlit run app.py`.

---

## 4. Evaluation — measuring quality

### 4.1 `data/eval/gold_qa.json` — the ground truth

26 hand-written questions, each `{question, reference, category, source}`. Three
categories:
- `in_domain` (18) — factual, answerable from one paper.
- `multi_hop` (4) — need info from more than one paper.
- `out_of_domain` (4) — **not** answerable from the corpus (capital of Australia,
  etc.). These test whether the system abstains.

### 4.2 `eval/ragas_eval.py` — wire Ragas to Claude + local embeddings

- `_judge()` wraps Claude (`judge_model`) as the Ragas LLM judge.
- `_judge_embeddings()` wraps local BGE for the one metric that needs embeddings.
- `build_metrics()` returns the four metrics **with the judge/embeddings attached
  to each one** — critical, because a metric with no LLM silently falls back to
  OpenAI and crashes on a missing key.
- `run_ragas(dataset)` calls `evaluate(...)` with a `RunConfig(max_workers=4)` to
  respect rate limits.

The four metrics: **context precision** (are retrieved chunks relevant/ranked
well?), **context recall** (did we find the needed chunks?), **faithfulness** (is
the answer grounded in the contexts?), **answer relevancy** (does it address the
question?).

### 4.3 `eval/build_dataset.py` — run the pipeline over the gold set

- `load_gold()` reads the JSON.
- `build()` first does a `retrieve("warmup")` to load the local models (so the
  first question's latency isn't inflated), then for each gold question runs
  `pipeline.answer()` and packages the result into a Ragas `SingleTurnSample`
  (`user_input`, `retrieved_contexts`, `response`, `reference`).
- It also returns plain `rows` carrying the `category` and latencies, so we can
  slice scores by question type.

### 4.4 `scripts/run_eval.py` — the eval driver

- `run(tag)`: builds the dataset → runs Ragas → converts to a DataFrame → computes
  **overall means, per-category means, and latency** → writes everything (plus the
  config snapshot and per-row scores) to `data/eval/results/<tag>.json` → prints a
  summary.
- `compare(a, b)`: prints two result files side by side (overall + the OOD subset)
  — used for the before/after story.
- Usage: `python scripts/run_eval.py --tag baseline` or
  `python scripts/run_eval.py --compare baseline no_abstain`.

### 4.5 `scripts/abstention_check.py` — the deterministic safety metric

Ragas faithfulness turned out to be the *wrong* tool for measuring abstention (it
scores answers against the retrieved context, but a gated OOD query has no
context — so the number is degenerate and judge-dependent; see WRITEUP §5).

So this script measures the behavior **directly and without a judge**: run the
pipeline over the gold set, use `is_abstention()` to check whether each answer was
a refusal, and report the abstention rate per category. For out-of-domain
questions, *answering* = a hallucination. Run it under each setting:
```
ABSTAIN=true  python scripts/abstention_check.py --tag fix_on
ABSTAIN=false python scripts/abstention_check.py --tag fix_off
```
Result: out-of-domain hallucination rate **0% (fix on) vs 75% (fix off)**.

---

## 5. The Build / Break / Fix story, as code

All three are config flags read by the modules above. Here's the exact path each
one takes:

| Experiment | Env flag | Read in | Effect in the pipeline |
|---|---|---|---|
| **Break #1** | `RERANK=false` | `config.rerank` | `retriever.py` skips the cross-encoder, returns top-k by raw cosine → precision drops |
| **Break #2** | `USE_QUERY_INSTRUCTION=false` | `config.use_query_instruction` | `embeddings.embed_query` drops the BGE prefix → query/doc mismatch → recall drops |
| **The FIX** | `ABSTAIN=false` | `config.abstain` | `generate.py` uses the permissive prompt and answers even with no context → hallucination on OOD |

Because each is one flag, every before/after number is reproducible by re-running
`run_eval.py` (or `abstention_check.py`) with the flag set.

---

## 6. The supporting scripts

- **`scripts/build_index.py`** — builds the vector index (§2.5). Run once.
- **`scripts/test_key.py`** — makes one cheap Claude call to confirm your
  `ANTHROPIC_API_KEY` works; prints a clear pass/fail without revealing the key.
- **`scripts/run_eval.py`** — runs Ragas (§4.4).
- **`scripts/abstention_check.py`** — the abstention metric (§4.5).

---

## 7. Data structures cheat-sheet

| Type | Defined in | Shape |
|---|---|---|
| `Page` | `ingest.py` | `{arxiv_id, title, page, text}` |
| `IndexItem` | `chunking.py` | `{id, text, metadata}` (metadata: arxiv_id, title, page, section, parent_id) |
| parent | `parents.json` | `{parent_id: {text, arxiv_id, title, page, section}}` |
| candidate dict | `vectorstore.query` | `{chunk_id, text, metadata, similarity, (rerank_score)}` |
| context (small-to-big) | `retriever._merge_to_parents` | candidate + `{matched_child}`, `text` = parent text |
| `RAGResult` | `pipeline.py` | `{question, answer, contexts, retrieval_ms, generation_ms}` |

The flow (small-to-big): `Page → parent → child IndexItem → (embedded) → Chroma
row → candidate child → merged up to its parent → context passed to Claude`.

---

## 8. End-to-end trace of one question

Asking *"What are the two pre-training tasks used to train BERT?"*:

1. `pipeline.answer()` calls `retriever.retrieve()`.
2. `embed_query()` prepends the BGE instruction and embeds it → 384-dim vector.
3. `vectorstore.query()` returns the 25 nearest **children** (BERT-paper children
   score ~0.78–0.86 cosine).
4. Cosine filter keeps those above 0.30 (all of them here).
5. `rerank()` re-scores all kept children with the cross-encoder; the real BERT
   children rise to the top with high positive scores.
6. The gate drops children below −3.0; the survivors remain (ranked).
7. `_merge_to_parents()` walks them best-first and expands each to its **parent**
   (~1024 chars), de-duping, until 5 parents — so the LLM gets full context, not
   256-char snippets.
8. `generate_answer()` formats those 5 parents as numbered citations (with
   section), sends them to Claude with the strict prompt.
9. Claude returns: *"...Masked Language Model (MLM) and Next Sentence Prediction
   (NSP) [4]..."* with citations.
10. `RAGResult` bundles the answer + the 5 parent contexts + timings; the app shows
    the answer, each parent, and the matched child snippet.

For *"What is the capital of Australia?"*: steps 1–5 happen, but at step 6 every
child scores ~−8 (below the gate), so `ranked` is empty and `retrieve()` returns
`[]`; `generate_answer` sees empty contexts + abstain on → returns the canned "I
don't have enough information…" **without calling Claude.**

---

## 9. How to run everything (quick reference)

```bash
# one-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
python scripts/test_key.py
python scripts/build_index.py        # builds data/chroma/

# use it
streamlit run app.py
python -c "from src.pipeline import answer; print(answer('What is LoRA?').answer)"

# evaluate it
python scripts/run_eval.py --tag baseline
RERANK=false python scripts/run_eval.py --tag no_rerank
python scripts/run_eval.py --compare baseline no_rerank
ABSTAIN=true  python scripts/abstention_check.py --tag fix_on
ABSTAIN=false python scripts/abstention_check.py --tag fix_off
```

---

## 10. Where to look when you want to change X

| Want to… | Edit / run |
|---|---|
| Change the corpus | `DEFAULT_ARXIV_IDS` in `src/ingest.py`, then rebuild |
| Switch to flat chunking | `SMALL_TO_BIG=false`, then rebuild the index |
| Tune parent / child size | `CHUNK_SIZE` / `CHILD_CHUNK_SIZE` env, then rebuild |
| Swap embedding model | `EMBED_MODEL` env, then rebuild (dimension changes) |
| Improve section detection | `SECTION_PAT` in `src/sectioning.py`, then rebuild |
| Change how many chunks reach the LLM | `TOP_K` env |
| Make abstention stricter/looser | `RERANK_THRESHOLD` env |
| Use a cheaper eval judge | `JUDGE_MODEL=claude-haiku-4-5-20251001` |
| Add eval questions | append to `data/eval/gold_qa.json` |
| Change the answer style | the system prompts in `src/generate.py` |
```
