"""옵션 C 데모: LLM 호출 없이 인덱스·검색 동작만 확인.

골드셋의 질문들을 retriever에 던지고 상위 청크의 페이지·절·미리보기를 출력.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings, resolve_path
from src.retrieval.retriever import count, search


def main() -> int:
    cfg = load_settings()
    n = count()
    print(f"[i] ChromaDB 청크 수: {n}")
    if n == 0:
        print("[!] 인덱싱된 청크가 없습니다. 먼저 build_index를 실행하세요.")
        return 1

    goldset_path = resolve_path(cfg["paths"]["goldset"])
    items = []
    with open(goldset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # 시연용으로 첫 5개만
    sample = items[:5]
    top_k = cfg["retrieval"]["top_k"]
    print(f"[i] 골드셋 총 {len(items)}문항 중 처음 {len(sample)}개로 검색 시연 (top-k={top_k})\n")

    out_lines = ["# 검색 시연 결과 (LLM 호출 없음)\n"]
    out_lines.append(f"- 청크 수: **{n}**")
    out_lines.append(f"- 임베딩: `{cfg['embedding']['model_name']}`")
    out_lines.append(f"- top-k: {top_k}\n")

    for it in sample:
        q = it["question"]
        cat = it["category"]
        print(f"❓ [{it['qid']}|{cat}] {q}")
        results = search(q, top_k=top_k)
        out_lines.append(f"## {it['qid']} ({cat})")
        out_lines.append(f"**질문**: {q}\n")
        out_lines.append(f"**정답 키워드**: {', '.join(it.get('gold_keywords', []))}\n")
        out_lines.append("| 순위 | 페이지 | 절 | 점수 | 미리보기 |")
        out_lines.append("|---|---|---|---|---|")
        for i, r in enumerate(results, start=1):
            preview = r.text.replace("\n", " ").replace("|", "/")[:80]
            print(f"   {i}. p.{r.page} §{r.section} {r.section_title} (score={r.score:.3f})")
            print(f"      {preview}...")
            out_lines.append(
                f"| {i} | {r.page} | {r.section} {r.section_title} | {r.score:.3f} | {preview}... |"
            )
        # 키워드 적중 확인
        all_text = " ".join(r.text for r in results)
        hits = [k for k in it.get("gold_keywords", []) if k in all_text]
        miss = [k for k in it.get("gold_keywords", []) if k not in all_text]
        verdict = f"검색 청크 합집합에서 키워드 적중 {len(hits)}/{len(it.get('gold_keywords', []))}"
        print(f"   ➤ {verdict} (적중: {hits}, 누락: {miss})\n")
        out_lines.append(f"\n> 키워드 적중: **{len(hits)}/{len(it.get('gold_keywords', []))}** "
                         f"(적중: {hits}, 누락: {miss})\n")

    report_path = resolve_path(cfg["paths"]["report_dir"]) / "retrieval_demo.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[OK] 검색 시연 리포트: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
