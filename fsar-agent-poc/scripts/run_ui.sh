#!/usr/bin/env bash
# Streamlit 비교 대시보드 실행
set -euo pipefail
cd "$(dirname "$0")/.."
streamlit run src/ui/app.py --server.port "${STREAMLIT_PORT:-8501}"
