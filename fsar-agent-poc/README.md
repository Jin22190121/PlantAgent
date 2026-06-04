# FSAR Agent PoC — Gemini vs EXAONE 한글 원전 문서 비교

신고리 3,4호기 **최종안전성분석보고서(FSAR) 1장** 한글 PDF를 학습하여 질의·요약·검색에 응답하는 AI Agent의 **이중 LLM 성능 비교 PoC**.

## 비교 대상

| 시스템 | 모델 | 실행 방식 | 라이선스 |
|---|---|---|---|
| **System A** | `gemini-1.5-flash` (Google) | 무료 티어 API | Google API ToS |
| **System B** | `exaone3.5:7.8b` (LG AI Research) | 로컬 Ollama | EXAONE AI Model License (PoC OK) |

두 시스템은 **임베딩·청킹·검색·프롬프트·온도가 모두 동일**하며 오직 생성 LLM만 다르다 (`config/settings.yaml` 단일 소스).

## 빠른 시작

```bash
# 1) 의존성 설치
cd fsar-agent-poc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1-1) 스캔 PDF용 Tesseract 한국어 OCR (대상 PDF가 텍스트 레이어 없는 스캔본일 때)
# Ubuntu / Debian / Codespaces:
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
# macOS:
# brew install tesseract tesseract-lang

# 2) Ollama 및 모델 (System B)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull exaone3.5:7.8b

# 3) 환경변수 (System A)
cp .env.example .env
# .env 편집해서 GOOGLE_API_KEY 입력

# 4) PDF 배치
# Google Drive에서 신고리 3,4호기 FSAR 1장 PDF 다운로드 →
#   data/raw/SK34_FSAR_Ch1.pdf 로 저장
#   https://drive.google.com/file/d/1lfC8Af_YgbcwOXoxl0aYc2AKQdiHcHHG/view

# 5) 인덱싱
bash scripts/run_ingest.sh

# 6) 정성 비교 UI
bash scripts/run_ui.sh
# → http://localhost:8501

# 7) 정량 평가
bash scripts/run_eval.sh
# → data/eval/report_YYYYMMDD_HHMMSS.md
```

## 평가 방법론

- **검색 품질**: Recall@5, Precision@5 (정답 페이지 적중)
- **답변 품질**: KoNLPy Mecab 형태소 기반 token-level Precision/Recall/F1, Exact Match, 키워드 적중률
- **운영 효율**: 평균/p95 응답 지연(ms), 평균 토큰(입력+출력), 비용($/쿼리)

골드셋(`data/eval/goldset.jsonl`)은 카테고리별 20~30개 Q&A로 구성:
사실 검색 / 위치 지정 / 요약 / 비교

## 디렉토리

```
fsar-agent-poc/
├── config/           통제변수·프롬프트 (단일 소스)
├── data/
│   ├── raw/          PDF 배치 위치 (gitignored)
│   ├── processed/    추출 텍스트·청크
│   └── eval/         골드셋·리포트
├── vectorstore/      ChromaDB 영속화 (gitignored)
├── src/
│   ├── ingestion/    PDF → 청크 → ChromaDB
│   ├── embeddings/   BGE-M3 (공통)
│   ├── retrieval/    top-k=5 검색 (공통)
│   ├── agents/       base / gemini / ollama
│   ├── evaluation/   metrics · runner · report
│   └── ui/           Streamlit 대시보드
└── scripts/          run_ingest.sh, run_ui.sh, run_eval.sh
```

## 라이선스 주의

- EXAONE-3.5는 **비상업 연구·평가** 목적에 한해 사용 가능 (PoC 범위 내).
- 상용 전환 시 Qwen2.5-7B(Apache 2.0)로 교체 (`config/settings.yaml`의 `system_b.model`만 변경).
