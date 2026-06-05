"""두 개(이상)의 평가 리포트 JSON을 하나로 합쳐 비교 리포트 생성.

예: System A를 먼저 실행하고 (report_TS_A.json), 시간 두고 System B 실행
(report_TS_B.json) 후 두 결과를 합쳐 비교 리포트를 만들 때.

사용:
    python scripts/merge_reports.py data/eval/report_TS_A.json data/eval/report_TS_B.json
    → data/eval/report_merged_<timestamp>.{json,csv,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_settings, resolve_path
from src.evaluation.runner import (
    aggregate,
    per_category,
    _write_csv,
    _write_json,
    _write_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path, help="합칠 report_*.json 파일들")
    parser.add_argument("--prefix", default=None, help="출력 prefix (기본: 현재 시각)")
    args = parser.parse_args()

    cfg = load_settings()
    report_dir = resolve_path(cfg["paths"]["report_dir"])

    merged_rows: dict[str, list[dict]] = {}
    sources = []
    for p in args.reports:
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        sources.append(str(p))
        for sys_key, rows in r.get("rows", {}).items():
            if not rows:
                continue
            # 동일 sys_key가 여러 리포트에 있으면 가장 마지막 것이 이김
            merged_rows[sys_key] = rows

    if not merged_rows:
        print("[!] 합칠 행이 없습니다.")
        return 1

    ts = args.prefix or f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = {
        "timestamp": ts,
        "merged_from": sources,
        "config": {
            "embedding": cfg["embedding"]["model_name"],
            "chunk_size": cfg["ingestion"]["chunk_size"],
            "top_k": cfg["retrieval"]["top_k"],
            "system_a": cfg["system_a"],
            "system_b": cfg["system_b"],
        },
        "summary": {s: aggregate(rows) for s, rows in merged_rows.items()},
        "per_category": {s: per_category(rows) for s, rows in merged_rows.items()},
        "rows": merged_rows,
    }

    json_path = report_dir / f"report_{ts}.json"
    csv_path = report_dir / f"report_{ts}.csv"
    md_path = report_dir / f"report_{ts}.md"
    _write_json(result, json_path)
    _write_csv(merged_rows, csv_path)
    _write_markdown(result, md_path)
    print(f"[OK] 병합 리포트:\n  JSON: {json_path}\n  CSV:  {csv_path}\n  MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
