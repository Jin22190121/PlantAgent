"""Run the procedure-RAG golden test set and report Recall@k / top-1 accuracy.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --doc-filter      # use per-case doc_type filter
    python scripts/eval_rag.py --tier A          # restrict to step chunks only

Reads `data/eval/golden_set.json`.
Targets (per docs/DESIGN.md): Recall@3 ≥ 90 %, top-1 ≥ 70 %.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npp_agent.mcp_servers.procedure import ProcedureMCP


def evaluate(use_doc_filter: bool, tier: str | None, k: int = 3, top_n: int = 5):
    cases = json.loads(Path("data/eval/golden_set.json").read_text())["cases"]
    proc = ProcedureMCP()

    hits_top1 = 0
    hits_topk = 0
    per_doc = {"GOP": [0, 0, 0], "EOP": [0, 0, 0], "AOP": [0, 0, 0]}
    misses = []

    for case in cases:
        relevant = set(case["relevant_ids"])
        dt = case["doc_type"] if use_doc_filter else None
        result = proc.call_tool("search_procedure", {
            "situation": case["query"],
            "n_results": top_n,
            "doc_type": dt,
            "tier": tier,
        })
        retrieved = [p["id"] for p in result["procedures"]]
        top1 = retrieved[0] if retrieved else None
        topk = set(retrieved[:k])

        is_top1 = top1 in relevant
        is_topk = bool(relevant & topk)

        if is_top1: hits_top1 += 1
        if is_topk: hits_topk += 1
        per_doc[case["doc_type"]][0] += 1
        if is_top1: per_doc[case["doc_type"]][1] += 1
        if is_topk: per_doc[case["doc_type"]][2] += 1

        mark = "✓" if is_topk else "✗"
        print(f"{mark} [{case['id']}] {case['query'][:50]:50s} "
              f"→ top1={top1} (relevant={list(relevant)})")
        if not is_topk:
            misses.append(case["id"])

    n = len(cases)
    print()
    print(f"Recall@1 = {hits_top1}/{n} = {hits_top1/n:.1%}")
    print(f"Recall@{k} = {hits_topk}/{n} = {hits_topk/n:.1%}")
    print()
    print("Per-document breakdown:")
    for d, (total, t1, tk) in per_doc.items():
        if total:
            print(f"  {d}: top1 {t1}/{total} = {t1/total:.0%}, "
                  f"recall@{k} {tk}/{total} = {tk/total:.0%}")
    if misses:
        print(f"\nMisses ({len(misses)}): {misses}")

    # Hit targets per design
    target_top1 = 0.70
    target_recall = 0.90
    pass_t1 = hits_top1 / n >= target_top1
    pass_rk = hits_topk / n >= target_recall
    print()
    print(f"PoC target  Recall@{k} ≥ {target_recall:.0%} → {'✅ PASS' if pass_rk else '❌ FAIL'}")
    print(f"PoC target  top-1 ≥ {target_top1:.0%}        → {'✅ PASS' if pass_t1 else '❌ FAIL'}")
    return 0 if (pass_rk and pass_t1) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-filter", action="store_true",
                    help="apply per-case doc_type filter")
    ap.add_argument("--tier", default=None, help="A | B | C")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()
    sys.exit(evaluate(args.doc_filter, args.tier, args.k, args.top_n))
