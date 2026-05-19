#!/usr/bin/env bash
# data/raw/*.pdf 를 청크로 만들어 ChromaDB에 인덱싱
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.ingestion.build_index "$@"
