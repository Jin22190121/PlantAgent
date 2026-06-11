"""사람이 직접 채점한 평가 기록을 JSONL로 누적하고 집계·내보내기.

데이터 모델
-----------
한 응답에 대한 인간 채점 단위로 한 줄. 동일 질문에 A·B 답변이 있으면 2줄 생성.

JSONL 스키마:
{
  "timestamp": "ISO-8601",
  "question": "질문 본문",
  "category": "사실검색|위치지정|요약|비교|기타",
  "system": "A" | "B",
  "system_model": "gemini-2.5-flash-lite" 등,
  "answer": "AI 답변 전문",
  "retrieved_pages": [8, 16, ...],
  "latency_ms": 1234.5,
  "prompt_tokens": 2800,
  "completion_tokens": 70,
  "accuracy": "correct" | "partial" | "wrong" | "na",
  "rating": 1~5,
  "note": "자유 코멘트"
}
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from src.config import load_settings, resolve_path

ACCURACY_LEVELS = ["correct", "partial", "wrong", "na"]
ACCURACY_KO = {
    "correct": "정답",
    "partial": "부분정답",
    "wrong": "오답",
    "na": "평가불가",
}

LOG_FILENAME = "human_eval.jsonl"


@dataclass
class HumanEvalRecord:
    timestamp: str
    question: str
    category: str
    system: str
    system_model: str
    answer: str
    retrieved_pages: list[int]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    accuracy: str
    rating: int
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def log_path() -> Path:
    cfg = load_settings()
    return resolve_path(cfg["paths"]["report_dir"]) / LOG_FILENAME


def append_record(record: HumanEvalRecord) -> Path:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_records() -> list[dict]:
    path = log_path()
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc_counts = {a: 0 for a in ACCURACY_LEVELS}
    ratings: list[int] = []
    latencies: list[float] = []
    for r in rows:
        a = r.get("accuracy", "na")
        if a not in acc_counts:
            a = "na"
        acc_counts[a] += 1
        ratings.append(int(r.get("rating", 0) or 0))
        latencies.append(float(r.get("latency_ms", 0) or 0))
    return {
        "n": n,
        "accuracy_counts": acc_counts,
        "accuracy_rate": acc_counts["correct"] / n,
        "partial_rate": acc_counts["partial"] / n,
        "wrong_rate": acc_counts["wrong"] / n,
        "rating_mean": (sum(ratings) / len(ratings)) if ratings else 0.0,
        "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }


def aggregate_stats(records: list[dict]) -> dict:
    if not records:
        return {"n_total": 0, "by_system": {}, "by_category": {}}

    by_system = {}
    for sys_key in ("A", "B"):
        rs = [r for r in records if r.get("system") == sys_key]
        if rs:
            by_system[sys_key] = _agg(rs)

    cats = sorted({r.get("category", "기타") for r in records})
    by_category = {c: _agg([r for r in records if r.get("category") == c]) for c in cats}

    return {
        "n_total": len(records),
        "by_system": by_system,
        "by_category": by_category,
    }


def export_csv(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    fields = [
        "timestamp", "category", "system", "system_model",
        "question", "answer", "retrieved_pages",
        "latency_ms", "prompt_tokens", "completion_tokens",
        "accuracy", "rating", "note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            if isinstance(row.get("retrieved_pages"), list):
                row["retrieved_pages"] = ",".join(map(str, row["retrieved_pages"]))
            w.writerow(row)
    return path


def export_markdown(records: list[dict], stats: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# 인간 평가 리포트 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"누적 평가 수: **{stats.get('n_total', 0)}**\n")

    by_system = stats.get("by_system", {})
    if by_system:
        lines.append("## 시스템별 정확률·평점\n")
        lines.append("| 시스템 | n | 정답률 | 부분정답률 | 오답률 | 평균 평점 | 평균 지연(ms) |")
        lines.append("|---|---|---|---|---|---|---|")
        for sys_key, s in by_system.items():
            lines.append(
                f"| {sys_key} | {s['n']} | {s['accuracy_rate']:.2%} | "
                f"{s['partial_rate']:.2%} | {s['wrong_rate']:.2%} | "
                f"{s['rating_mean']:.2f} | {s['latency_ms_mean']:.0f} |"
            )
        lines.append("")

    by_cat = stats.get("by_category", {})
    if by_cat:
        lines.append("## 카테고리별 정답률\n")
        lines.append("| 카테고리 | n | 정답률 | 평균 평점 |")
        lines.append("|---|---|---|---|")
        for cat, s in by_cat.items():
            lines.append(
                f"| {cat} | {s['n']} | {s['accuracy_rate']:.2%} | {s['rating_mean']:.2f} |"
            )
        lines.append("")

    lines.append("## 문항별 상세\n")
    for i, r in enumerate(records, start=1):
        acc_ko = ACCURACY_KO.get(r.get("accuracy", "na"), r.get("accuracy", "na"))
        lines.append(f"### {i}. [{r.get('category', '기타')}] {r.get('question', '')}\n")
        lines.append(
            f"- **System {r.get('system', '?')}** "
            f"(`{r.get('system_model', '?')}`)"
        )
        lines.append(f"- 정확성: **{acc_ko}**, 평점: {r.get('rating', 0)}/5")
        lines.append(
            f"- 지연: {r.get('latency_ms', 0):.0f}ms, "
            f"출력 토큰: {r.get('completion_tokens', 0)}"
        )
        lines.append(f"- 인용 페이지: {r.get('retrieved_pages', [])}")
        lines.append(f"\n**답변**: {r.get('answer', '')}\n")
        if r.get("note"):
            lines.append(f"**코멘트**: {r['note']}\n")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
