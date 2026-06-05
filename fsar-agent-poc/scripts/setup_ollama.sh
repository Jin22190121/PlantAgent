#!/usr/bin/env bash
# System B(EXAONE)를 위한 Ollama 설치 + 모델 다운로드 + 서버 백그라운드 시작
set -euo pipefail

MODEL="${OLLAMA_MODEL:-exaone3.5:7.8b}"

# 1) Ollama 설치 (이미 있으면 스킵)
if ! command -v ollama >/dev/null 2>&1; then
    echo "[i] Ollama 설치 중..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[i] Ollama 이미 설치됨: $(ollama --version 2>&1 | head -1)"
fi

# 2) ollama serve 백그라운드 실행 (Codespace에는 systemd 없음)
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "[i] ollama serve 백그라운드 실행..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 3) 서버 readiness 확인
for i in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
        echo "[i] Ollama 서버 응답 OK"
        break
    fi
    echo "    서버 대기... ($i/10)"
    sleep 2
done

# 4) 모델 다운로드
echo "[i] 모델 pull: $MODEL (약 5GB, 처음 1회만 시간 소요)"
ollama pull "$MODEL"

# 5) 검증
echo ""
echo "[i] 사용 가능한 모델:"
ollama list

echo ""
echo "[OK] Ollama + $MODEL 준비 완료. 평가 실행:"
echo "     python -m src.evaluation.runner --systems B --limit 6"
