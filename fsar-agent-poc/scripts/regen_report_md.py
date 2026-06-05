"""기존 report_*.json을 다시 읽어 같은 위치에 report_*.md를 재생성.

사용 예:
    python scripts/regen_report_md.py data/eval/report_20260605_020853.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.runner import _write_markdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="report_*.json 또는 *.partial.json")
    args = parser.parse_args()
    with open(args.json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    md_path = args.json_path.with_suffix(".regen.md")
    _write_markdown(result, md_path)
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
