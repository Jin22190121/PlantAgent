#!/usr/bin/env bash
# 골드셋 기반 정량 평가 실행 → data/eval/report_*.{json,csv,md}
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluation.runner "$@"
