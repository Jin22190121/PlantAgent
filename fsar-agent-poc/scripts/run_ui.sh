#!/usr/bin/env bash
# Streamlit 비교 대시보드 실행
set -euo pipefail
cd "$(dirname "$0")/.."
# 프로젝트 루트를 PYTHONPATH에 추가 (streamlit이 sys.path를 자동 설정 안 함)
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$(pwd)"
streamlit run src/ui/app.py --server.port "${STREAMLIT_PORT:-8501}"
