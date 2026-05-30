"""Direct abstention / hallucination metric — the right tool for the FIX.

Ragas faithfulness is the wrong instrument for measuring abstention: on a
correctly-gated out-of-domain query the retrieved context is empty, so
"faithfulness to the context" is degenerate and judge-dependent (Haiku and
Sonnet disagree on it). What we actually care about is behavioral and
deterministic: *did the system decline to answer a question it cannot support?*

This script runs the live pipeline over the gold set and reports, per category,
the fraction of questions the system abstained on (detected by
`generate.is_abstention`, no LLM judge involved). The desired behavior:

    out_of_domain  -> abstention rate ~1.0   (refuse the unanswerable)
    in_domain      -> abstention rate ~0.0   (don't over-refuse real questions)

Run it under each setting of the FIX toggle:

    ABSTAIN=true  python scripts/abstention_check.py --tag fix_on
    ABSTAIN=false python scripts/abstention_check.py --tag fix_off
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, RESULTS_DIR, GOLD_QA_PATH  # noqa: E402
from src.generate import is_abstention  # noqa: E402
from src.pipeline import answer  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="label, e.g. fix_on / fix_off")
    args = ap.parse_args()

    gold = json.loads(Path(GOLD_QA_PATH).read_text())
    by_cat: dict[str, list[dict]] = defaultdict(list)

    for item in gold:
        res = answer(item["question"])
        abstained = is_abstention(res.answer)
        by_cat[item.get("category", "in_domain")].append(
            {
                "question": item["question"],
                "abstained": abstained,
                "n_contexts": len(res.contexts),
                "answer": res.answer[:200],
            }
        )

    summary = {}
    for cat, rows in by_cat.items():
        rate = sum(r["abstained"] for r in rows) / len(rows)
        summary[cat] = {"abstention_rate": round(rate, 4), "n": len(rows)}

    # For out-of-domain, abstaining is correct; not abstaining is a hallucination.
    ood = summary.get("out_of_domain", {})
    hallucination_rate = round(1.0 - ood.get("abstention_rate", 0.0), 4) if ood else None

    payload = {
        "tag": args.tag,
        "abstain_enabled": CONFIG.abstain,
        "by_category": summary,
        "out_of_domain_hallucination_rate": hallucination_rate,
        "detail": by_cat,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"abstention_{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== abstention [{args.tag}] (ABSTAIN={CONFIG.abstain}) ===")
    for cat, s in summary.items():
        print(f"  {cat:16s} abstention_rate={s['abstention_rate']:.2f}  (n={s['n']})")
    if hallucination_rate is not None:
        print(f"  >> out-of-domain hallucination rate: {hallucination_rate:.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
