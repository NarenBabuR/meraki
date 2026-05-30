"""Streamlit chat interface for the RAG pipeline.

Shows the answer plus every retrieved chunk inline (title, page, cosine +
rerank scores) so you can *see* what the model was given — which makes the
failure modes legible during a demo. Active config is shown in the sidebar.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config import CONFIG
from src.pipeline import answer

st.set_page_config(page_title="arXiv ML RAG", page_icon="📄", layout="wide")
st.title("📄 arXiv ML RAG — Build, Break, Fix")

with st.sidebar:
    st.header("Active config")
    st.caption("Set via env vars before launching (see .env.example).")
    st.write(
        {
            "gen_model": CONFIG.gen_model,
            "embed_model": CONFIG.embed_model,
            "top_n / top_k": f"{CONFIG.top_n} / {CONFIG.top_k}",
            "rerank": CONFIG.rerank,
            "query_instruction": CONFIG.use_query_instruction,
            "sim_threshold": CONFIG.sim_threshold,
            "abstain": CONFIG.abstain,
        }
    )
    st.markdown(
        "**Toggles for the demo**\n\n"
        "- `RERANK=false` — Break #1 (precision)\n"
        "- `USE_QUERY_INSTRUCTION=false` — Break #2 (recall)\n"
        "- `ABSTAIN=false` — removes the FIX (hallucination on OOD)"
    )

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

prompt = st.chat_input("Ask about transformers, BERT, RAG, LoRA, …")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating…"):
            res = answer(prompt)
        st.markdown(res.answer)
        st.caption(
            f"retrieval {res.retrieval_ms:.0f} ms · generation {res.generation_ms:.0f} ms · "
            f"{len(res.contexts)} chunks"
        )
        if res.contexts:
            with st.expander(f"🔎 {len(res.contexts)} retrieved chunks"):
                for i, c in enumerate(res.contexts, start=1):
                    m = c["metadata"]
                    score = f"cosine={c['similarity']:.3f}"
                    if "rerank_score" in c:
                        score += f" · rerank={c['rerank_score']:.3f}"
                    st.markdown(
                        f"**[{i}] {m['title']}** "
                        f"(arXiv:{m['arxiv_id']}, p.{m['page']}) — {score}"
                    )
                    st.text(c["text"][:600] + ("…" if len(c["text"]) > 600 else ""))
                    st.divider()
        else:
            st.info("No chunks cleared the similarity threshold — abstaining.")

    st.session_state.history.append({"role": "assistant", "content": res.answer})
