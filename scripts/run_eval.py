"""Run the Ragas evaluation for the current config and write a result file.

The current config (RERANK, USE_QUERY_INSTRUCTION, ABSTAIN, ...) is read from the
environment, so each Build/Break/Fix experiment is one command:

    python scripts/run_eval.py --tag baseline
    RERANK=false              python scripts/run_eval.py --tag no_rerank
    USE_QUERY_INSTRUCTION=false python scripts/run_eval.py --tag no_query_instruction
    ABSTAIN=false             python scripts/run_eval.py --tag no_abstain

Results (overall + per-category means + the config snapshot + per-row scores) are
written to data/eval/results/<tag>.json and printed as a table.

Compare two runs:
    python scripts/run_eval.py --compare baseline no_abstain
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config import CONFIG, RESULTS_DIR, describe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_eval")

METRIC_COLS = [
    "llm_context_precision_with_reference",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]


def _means(df: pd.DataFrame) -> dict:
    return {m: round(float(df[m].mean()), 4) for m in METRIC_COLS if m in df.columns}


def run(tag: str) -> dict:
    # Imports here so --compare doesn't pay the model-load cost.
    from eval.build_dataset import build
    from eval.ragas_eval import run_ragas

    log.info("Eval tag=%s | %s", tag, describe())
    dataset, rows = build()
    result = run_ragas(dataset)
    df = result.to_pandas()

    # Attach category from rows (same order as samples) for per-subset slicing.
    df["category"] = [r["category"] for r in rows]
    df["question"] = [r["question"] for r in rows]
    df["retrieval_ms"] = [r["retrieval_ms"] for r in rows]
    df["generation_ms"] = [r["generation_ms"] for r in rows]

    overall = _means(df)
    by_category = {
        cat: _means(sub) for cat, sub in df.groupby("category")
    }
    latency = {
        "retrieval_ms_mean": round(float(df["retrieval_ms"].mean()), 1),
        "generation_ms_mean": round(float(df["generation_ms"].mean()), 1),
        "retrieval_ms_p95": round(float(df["retrieval_ms"].quantile(0.95)), 1),
    }

    payload = {
        "tag": tag,
        "config": CONFIG.snapshot(),
        "n_questions": len(df),
        "overall": overall,
        "by_category": by_category,
        "latency": latency,
        "per_row": df[
            ["question", "category", *[c for c in METRIC_COLS if c in df.columns]]
        ].round(4).to_dict(orient="records"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote %s", out)

    _print_summary(payload)
    return payload


def _print_summary(p: dict) -> None:
    print(f"\n=== {p['tag']}  (n={p['n_questions']}) ===")
    print("OVERALL:")
    for k, v in p["overall"].items():
        print(f"  {k:42s} {v:.4f}")
    print("BY CATEGORY:")
    for cat, m in p["by_category"].items():
        cells = "  ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in m.items())
        print(f"  {cat:16s} {cells}")
    print("LATENCY:", p["latency"])


def compare(tag_a: str, tag_b: str) -> None:
    a = json.loads((RESULTS_DIR / f"{tag_a}.json").read_text())
    b = json.loads((RESULTS_DIR / f"{tag_b}.json").read_text())
    print(f"\n=== {tag_a}  vs  {tag_b} (overall) ===")
    keys = sorted(set(a["overall"]) | set(b["overall"]))
    print(f"{'metric':42s} {tag_a:>12s} {tag_b:>12s} {'delta':>10s}")
    for k in keys:
        va, vb = a["overall"].get(k), b["overall"].get(k)
        if va is None or vb is None:
            continue
        print(f"{k:42s} {va:12.4f} {vb:12.4f} {vb - va:+10.4f}")
    # Out-of-domain subset (where the abstention FIX shows up most).
    if "out_of_domain" in a["by_category"] and "out_of_domain" in b["by_category"]:
        print(f"\n=== {tag_a} vs {tag_b}  (out_of_domain subset) ===")
        ca, cb = a["by_category"]["out_of_domain"], b["by_category"]["out_of_domain"]
        for k in sorted(set(ca) | set(cb)):
            va, vb = ca.get(k), cb.get(k)
            if va is None or vb is None:
                continue
            print(f"{k:42s} {va:12.4f} {vb:12.4f} {vb - va:+10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="label for this run (writes results/<tag>.json)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two existing result files")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.tag:
        run(args.tag)
    else:
        ap.error("provide --tag <name> or --compare <a> <b>")


if __name__ == "__main__":
    main()
