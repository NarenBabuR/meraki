"""Answer generation with Claude.

Ported (and heavily simplified) from file-processor/src/utils/ClaudeAnthropic.py:
- retry with backoff on transient errors (429 / 500 / 529)
- one model-tier fallback (configured gen_model -> Sonnet 4.5) when the primary
  is rate-limited or overloaded

The ABSTAIN toggle is the FIX in the Build/Break/Fix story. When on, the system
prompt instructs the model to answer strictly from the provided context and to
say it doesn't know otherwise. The retriever also short-circuits to a canned
"no relevant context" answer when nothing clears the similarity threshold, so we
never even spend a Claude call hallucinating on out-of-domain queries.
"""
from __future__ import annotations

import logging
import time

import anthropic

from config import CONFIG

logger = logging.getLogger(__name__)

RETRYABLE = {429, 500, 529}
FALLBACK_MODEL = "claude-sonnet-4-5"

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided documents to answer that."
)

# Phrases that signal the model declined to answer. Used by the abstention
# metric (eval/abstention.py) to detect refusals deterministically, without a
# judge. Kept deliberately simple and inspectable.
_ABSTENTION_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "cannot be answered",
    "can't answer",
    "cannot answer",
    "don't have information",
    "no relevant information",
    "not contained in the",
    "isn't in the provided",
    "is not in the provided",
    "i don't know",
)


def is_abstention(answer: str) -> bool:
    """True if the answer is a refusal/abstention rather than a substantive reply."""
    a = answer.strip().lower()
    return any(m in a for m in _ABSTENTION_MARKERS)

_SYSTEM_ABSTAIN = """You are a precise research assistant answering questions about a corpus of machine-learning papers.

Rules:
- Answer ONLY using the numbered context passages provided. Do not use prior knowledge.
- If the context does not contain the answer, reply exactly: "I don't have enough information in the provided documents to answer that."
- Cite the passages you used by their number, e.g. [1], [3].
- Be concise and factual."""

_SYSTEM_NO_ABSTAIN = """You are a helpful research assistant answering questions about machine-learning papers. Use the provided context passages, and answer the question as best you can. Cite passages by number where relevant, e.g. [1], [3]."""


def _client() -> anthropic.Anthropic:
    # API key picked up from ANTHROPIC_API_KEY (loaded via config -> dotenv).
    return anthropic.Anthropic(max_retries=0)  # we handle retries ourselves


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        cite = f"{meta.get('title', '?')} (arXiv:{meta.get('arxiv_id', '?')}, p.{meta.get('page', '?')})"
        blocks.append(f"[{i}] {cite}\n{c['text']}")
    return "\n\n".join(blocks)


def _call(client: anthropic.Anthropic, model: str, system: str, user: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=CONFIG.gen_max_tokens,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate an answer from retrieved contexts, with retry + fallback."""
    if CONFIG.abstain and not contexts:
        # Retriever found nothing above threshold — don't even call the model.
        return NO_CONTEXT_ANSWER

    system = _SYSTEM_ABSTAIN if CONFIG.abstain else _SYSTEM_NO_ABSTAIN
    user = (
        f"Context passages:\n\n{_format_context(contexts)}\n\n"
        f"Question: {question}"
    )
    client = _client()
    models = [CONFIG.gen_model]
    if FALLBACK_MODEL != CONFIG.gen_model:
        models.append(FALLBACK_MODEL)

    last_err: Exception | None = None
    for model in models:
        for attempt in range(3):
            try:
                return _call(client, model, system, user)
            except anthropic.APIStatusError as e:
                last_err = e
                if e.status_code in RETRYABLE:
                    wait = 2 ** attempt
                    logger.warning(
                        "Claude %s returned %s; retry in %ss",
                        model, e.status_code, wait,
                    )
                    time.sleep(wait)
                    continue
                raise
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        logger.warning("Falling back from %s after repeated failures", model)
    raise RuntimeError(f"Generation failed after retries: {last_err}")
