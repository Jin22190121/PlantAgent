"""평가 러너: 골드셋을 양 시스템에 동일 입력으로 실행하고 메트릭을 집계."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import time
from datetime import datetime
from pathlib import Path

from src.agents.factory import get_agent, SYSTEMS
from src.config import load_settings, resolve_path
from src.evaluation.metrics import (
    exact_match,
    keyword_hit_rate,
    retrieval_pr,
    token_prf1,
)


def load_goldset(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def evaluate_one(item: dict, response_dict: dict, agent_response) -> dict:
    pred = agent_response.answer
    gold_answer = item.get("gold_answer", "")
    gold_keywords = item.get("gold_keywords", [])
    gold_pages = item.get("gold_pages", [])

    prf1 = token_prf1(pred, gold_answer)
    ret = retrieval_pr(agent_response.cited_pages, gold_pages)
    return {
        **response_dict,
        "qid": item.get("qid"),
        "category": item.get("category", "uncategorized"),
        "gold_answer": gold_answer,
        "gold_keywords": gold_keywords,
        "gold_pages": gold_pages,
        "token_precision": prf1["precision"],
        "token_recall": prf1["recall"],
        "token_f1": prf1["f1"],
        "exact_match": exact_match(pred, gold_answer),
        "keyword_hit_rate": keyword_hit_rate(pred, gold_keywords),
        "retrieval_recall@k": ret["recall@k"],
        "retrieval_precision@k": ret["precision@k"],
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    avg = lambda key: stats.fmean(r[key] for r in rows)
    p95 = lambda key: sorted(r[key] for r in rows)[int(0.95 * (len(rows) - 1))]
    return {
        "n": len(rows),
        "token_f1_mean": avg("token_f1"),
        "token_precision_mean": avg("token_precision"),
        "token_recall_mean": avg("token_recall"),
        "exact_match_mean": avg("exact_match"),
        "keyword_hit_rate_mean": avg("keyword_hit_rate"),
        "retrieval_recall@k_mean": avg("retrieval_recall@k"),
        "retrieval_precision@k_mean": avg("retrieval_precision@k"),
        "latency_ms_mean": avg("latency_ms"),
        "latency_ms_p95": p95("latency_ms"),
        "prompt_tokens_mean": avg("prompt_tokens"),
        "completion_tokens_mean": avg("completion_tokens"),
        "cost_usd_total": sum(r["cost_usd"] for r in rows),
    }


def per_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: aggregate(rs) for c, rs in cats.items()}


def run(
    systems: list[str],
    output_prefix: str | None = None,
    limit: int | None = None,
) -> dict:
    cfg = load_settings()
    goldset_path = resolve_path(cfg["paths"]["goldset"])
    report_dir = resolve_path(cfg["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    items = load_goldset(goldset_path)
    if not items:
        raise RuntimeError(f"골드셋이 비어있습니다: {goldset_path}")
    if limit is not None and limit > 0:
        items = items[:limit]
        print(f"[i] --limit {limit} 적용 → 처음 {len(items)}문항만 평가")
    print(f"[i] 골드셋: {len(items)}문항, 평가 대상: {systems}")

    timestamp = output_prefix or datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows: dict[str, list[dict]] = {}
    rpd_aborted = False
    for sys_key in systems:
        agent = get_agent(sys_key)
        sys_cfg_key = "system_a" if sys_key.upper() == "A" else "system_b"
        pacing = float(cfg.get(sys_cfg_key, {}).get("request_pacing_sec", 0) or 0)
        if pacing > 0:
            est_min = pacing * len(items) / 60.0
            print(
                f"  [{sys_key}] 요청 간 페이싱 {pacing:.0f}s 적용 "
                f"(예상 최소 {est_min:.1f}분)"
            )
        rows: list[dict] = []
        for i, item in enumerate(items, start=1):
            q = item["question"]
            print(f"  [{sys_key}] ({i}/{len(items)}) {q[:40]}...")
            try:
                resp = agent.ask(q)
                row = evaluate_one(item, resp.to_dict(), resp)
                rows.append(row)
            except Exception as e:
                print(f"    ! 실패: {e}")
                # 일일 한도(RPD) 초과면 남은 문항은 모두 실패가 자명하므로 중단
                if "RPD_EXHAUSTED" in str(e):
                    rpd_aborted = True
                rows.append({
                    "qid": item.get("qid"),
                    "category": item.get("category", "uncategorized"),
                    "system": sys_key,
                    "question": q,
                    "answer": f"[ERROR] {e}",
                    "retrieved_ids": [],
                    "retrieved_pages": [],
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "gold_answer": item.get("gold_answer", ""),
                    "gold_keywords": item.get("gold_keywords", []),
                    "gold_pages": item.get("gold_pages", []),
                    "token_precision": 0.0,
                    "token_recall": 0.0,
                    "token_f1": 0.0,
                    "exact_match": 0.0,
                    "keyword_hit_rate": 0.0,
                    "retrieval_recall@k": 0.0,
                    "retrieval_precision@k": 0.0,
                })

            # 부분 저장: 매 문항 직후 누적 결과를 디스크에 덮어쓰기
            all_rows[sys_key] = rows
            _save_partial(all_rows, cfg, timestamp, report_dir)

            if rpd_aborted:
                print(
                    f"  [{sys_key}] 일일 한도 초과로 평가 중단. "
                    f"처리된 {len(rows)}/{len(items)} 문항까지 저장됨."
                )
                break

            # 다음 요청 전 RPM 한도 회피를 위한 페이싱 (마지막 문항 뒤는 생략)
            if pacing > 0 and i < len(items):
                time.sleep(pacing)

        if rpd_aborted:
            break

    result = {
        "timestamp": timestamp,
        "config": {
            "embedding": cfg["embedding"]["model_name"],
            "chunk_size": cfg["ingestion"]["chunk_size"],
            "top_k": cfg["retrieval"]["top_k"],
            "system_a": cfg["system_a"],
            "system_b": cfg["system_b"],
        },
        "summary": {s: aggregate(rows) for s, rows in all_rows.items()},
        "per_category": {s: per_category(rows) for s, rows in all_rows.items()},
        "rows": all_rows,
    }

    json_path = report_dir / f"report_{timestamp}.json"
    csv_path = report_dir / f"report_{timestamp}.csv"
    md_path = report_dir / f"report_{timestamp}.md"
    _write_json(result, json_path)
    _write_csv(all_rows, csv_path)
    _write_markdown(result, md_path)
    print(f"\n[OK] 리포트 저장:\n  JSON: {json_path}\n  CSV:  {csv_path}\n  MD:   {md_path}")
    return result


def _save_partial(
    all_rows: dict[str, list[dict]], cfg: dict, timestamp: str, report_dir: Path
) -> None:
    """문항 처리 직후 누적 결과를 디스크에 부분 저장 (중단 시 보존용)."""
    if not any(all_rows.values()):
        return
    partial = {
        "timestamp": timestamp,
        "partial": True,
        "config": {
            "embedding": cfg["embedding"]["model_name"],
            "chunk_size": cfg["ingestion"]["chunk_size"],
            "top_k": cfg["retrieval"]["top_k"],
            "system_a": cfg["system_a"],
            "system_b": cfg["system_b"],
        },
        "summary": {s: aggregate(rows) for s, rows in all_rows.items() if rows},
        "per_category": {s: per_category(rows) for s, rows in all_rows.items() if rows},
        "rows": all_rows,
    }
    _write_json(partial, report_dir / f"report_{timestamp}.partial.json")


def _write_json(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _write_csv(all_rows: dict[str, list[dict]], path: Path) -> None:
    fieldnames = [
        "system", "qid", "category", "question", "answer",
        "gold_answer", "gold_pages", "retrieved_pages",
        "token_f1", "token_precision", "token_recall",
        "exact_match", "keyword_hit_rate",
        "retrieval_recall@k", "retrieval_precision@k",
        "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for sys_key, rows in all_rows.items():
            for r in rows:
                row = dict(r)
                row["gold_pages"] = ",".join(map(str, r.get("gold_pages", [])))
                row["retrieved_pages"] = ",".join(map(str, r.get("retrieved_pages", [])))
                w.writerow(row)


def _write_markdown(result: dict, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# FSAR Agent 평가 리포트 — {result['timestamp']}\n")
    lines.append("## 통제 변수")
    cfg = result["config"]
    lines.append(f"- 임베딩: `{cfg['embedding']}`")
    lines.append(f"- 청크 크기: {cfg['chunk_size']}, top-k: {cfg['top_k']}")
    lines.append(f"- System A: **{cfg['system_a']['name']}** (`{cfg['system_a']['model']}`)")
    lines.append(f"- System B: **{cfg['system_b']['name']}** (`{cfg['system_b']['model']}`)\n")

    lines.append("## 전체 요약")
    lines.append("| 메트릭 | System A | System B |")
    lines.append("|---|---|---|")
    sumry = result["summary"]
    a = sumry.get("A", {})
    b = sumry.get("B", {})
    if a or b:
        keys = [
            ("token_f1_mean", "Token F1"),
            ("token_precision_mean", "Token Precision"),
            ("token_recall_mean", "Token Recall"),
            ("exact_match_mean", "Exact Match"),
            ("keyword_hit_rate_mean", "Keyword 적중률"),
            ("retrieval_recall@k_mean", "Retrieval Recall@k"),
            ("retrieval_precision@k_mean", "Retrieval Precision@k"),
            ("latency_ms_mean", "평균 지연(ms)"),
            ("latency_ms_p95", "p95 지연(ms)"),
            ("prompt_tokens_mean", "평균 입력 토큰"),
            ("completion_tokens_mean", "평균 출력 토큰"),
            ("cost_usd_total", "총 비용 ($)"),
        ]
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
        for k, label in keys:
            av = a.get(k, "—") if a else "—"
            bv = b.get(k, "—") if b else "—"
            lines.append(f"| {label} | {_fmt(av)} | {_fmt(bv)} |")
    lines.append("")

    lines.append("## 카테고리별 Token F1")
    cats = set()
    for s in result["per_category"].values():
        cats.update(s.keys())
    lines.append("| 카테고리 | System A | System B |")
    lines.append("|---|---|---|")
    for c in sorted(cats):
        a_cat = result["per_category"].get("A", {}).get(c, {})
        b_cat = result["per_category"].get("B", {}).get(c, {})
        a_v = a_cat.get("token_f1_mean", None)
        b_v = b_cat.get("token_f1_mean", None)
        a_s = f"{a_v:.4f}" if a_v is not None else "—"
        b_s = f"{b_v:.4f}" if b_v is not None else "—"
        lines.append(f"| {c} | {a_s} | {b_s} |")
    lines.append("")

    # 양쪽 시스템 모두 결과가 있으면 문항별 비교 + 자동 코멘트 추가
    rows = result.get("rows", {})
    has_both = bool(rows.get("A")) and bool(rows.get("B"))
    if has_both:
        lines.extend(_render_side_by_side(rows["A"], rows["B"]))
        lines.extend(_render_winner_analysis(rows["A"], rows["B"]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _render_side_by_side(a_rows: list[dict], b_rows: list[dict]) -> list[str]:
    """문항 ID 기준으로 A·B 응답을 나란히 표시."""
    out = ["## 문항별 시스템 비교 (System A vs System B)\n"]
    a_by_qid = {r["qid"]: r for r in a_rows}
    b_by_qid = {r["qid"]: r for r in b_rows}
    qids = sorted(set(a_by_qid) | set(b_by_qid))

    out.append("| QID | Cat | Token F1 (A/B) | Latency ms (A/B) | Tokens out (A/B) |")
    out.append("|---|---|---|---|---|")
    for qid in qids:
        ra = a_by_qid.get(qid, {})
        rb = b_by_qid.get(qid, {})
        cat = ra.get("category") or rb.get("category", "")
        out.append(
            f"| {qid} | {cat} | "
            f"{ra.get('token_f1', 0):.3f} / {rb.get('token_f1', 0):.3f} | "
            f"{ra.get('latency_ms', 0):.0f} / {rb.get('latency_ms', 0):.0f} | "
            f"{ra.get('completion_tokens', 0)} / {rb.get('completion_tokens', 0)} |"
        )
    out.append("")

    out.append("### 문항별 답변 전문")
    for qid in qids:
        ra = a_by_qid.get(qid, {})
        rb = b_by_qid.get(qid, {})
        q = ra.get("question") or rb.get("question", "")
        gold = ra.get("gold_answer") or rb.get("gold_answer", "")
        out.append(f"\n#### {qid} — {q}\n")
        out.append(f"**정답:** {gold}\n")
        out.append(f"**System A (Gemini):** {ra.get('answer', '—')}\n")
        out.append(f"**System B (EXAONE):** {rb.get('answer', '—')}\n")
    out.append("")
    return out


def _render_winner_analysis(a_rows: list[dict], b_rows: list[dict]) -> list[str]:
    """Token F1·지연·키워드 적중률 기준 자동 코멘트."""
    out = ["## 자동 비교 분석\n"]
    a_by_qid = {r["qid"]: r for r in a_rows}
    b_by_qid = {r["qid"]: r for r in b_rows}
    common = sorted(set(a_by_qid) & set(b_by_qid))
    if not common:
        out.append("_공통 평가 문항이 없습니다._\n")
        return out

    a_wins = b_wins = ties = 0
    for qid in common:
        af = a_by_qid[qid]["token_f1"]
        bf = b_by_qid[qid]["token_f1"]
        if abs(af - bf) < 1e-9:
            ties += 1
        elif af > bf:
            a_wins += 1
        else:
            b_wins += 1

    n = len(common)
    out.append(f"- 공통 평가 문항: **{n}개**")
    out.append(f"- Token F1 승부: **A {a_wins}승 / B {b_wins}승 / 무 {ties}**")

    def _avg(rows, key):
        vals = [r.get(key, 0) for r in rows]
        return sum(vals) / len(vals) if vals else 0

    a_lat = _avg(a_rows, "latency_ms")
    b_lat = _avg(b_rows, "latency_ms")
    a_kw = _avg(a_rows, "keyword_hit_rate")
    b_kw = _avg(b_rows, "keyword_hit_rate")
    out.append(f"- 평균 지연: **A {a_lat:.0f}ms / B {b_lat:.0f}ms**")
    out.append(f"- 평균 키워드 적중률: **A {a_kw:.3f} / B {b_kw:.3f}**")

    verdicts = []
    if a_wins > b_wins:
        verdicts.append(f"답변 품질(F1)은 System A가 우세 ({a_wins} vs {b_wins})")
    elif b_wins > a_wins:
        verdicts.append(f"답변 품질(F1)은 System B가 우세 ({b_wins} vs {a_wins})")
    else:
        verdicts.append("답변 품질(F1)은 양 시스템 비등")
    if a_lat < b_lat * 0.7:
        verdicts.append(f"응답 속도는 A가 압도적 우세 ({a_lat:.0f} vs {b_lat:.0f} ms)")
    elif b_lat < a_lat * 0.7:
        verdicts.append(f"응답 속도는 B가 압도적 우세 ({b_lat:.0f} vs {a_lat:.0f} ms)")
    out.append("\n**요약 판정**: " + "; ".join(verdicts) + ".\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(SYSTEMS),
        choices=list(SYSTEMS),
        help="평가할 시스템 (기본: A B)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="골드셋에서 처음 N개 문항만 평가 (무료 티어 일일 한도 절약용)",
    )
    args = parser.parse_args()
    run(args.systems, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
