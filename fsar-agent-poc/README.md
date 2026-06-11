# FSAR Agent PoC — Gemini vs EXAONE 한글 원전 문서 비교

신고리 3,4호기 **최종안전성분석보고서(FSAR) 1장** 한글 스캔 PDF를 학습하여 질의·요약·검색에 응답하는 AI Agent의 **이중 LLM 성능 비교 PoC**.

> 본 프로젝트의 평가 결과 보고서는 [`data/eval/PoC_Report_20260605.md`](data/eval/PoC_Report_20260605.md)를 참조하세요.

---

## 1. 한눈에 보기

| 구분 | 내용 |
|---|---|
| 대상 문서 | 신고리 3,4호기 FSAR 1장 (한수원 공개, 약 45 MB / 101쪽 / 스캔 이미지 PDF) |
| System A | Google **Gemini 2.5 Flash Lite** (무료 API) |
| System B | LG **EXAONE-3.5** (Ollama 로컬, 기본 2.4B / 권장 7.8B) |
| 공통 인프라 | Tesseract 한국어 OCR · BGE-M3 임베딩 · ChromaDB · 동일 프롬프트 |
| 평가 메트릭 | Token P/R/F1, Keyword 적중률, Retrieval Recall@k, 지연, 토큰, 비용 |
| UI | Streamlit 비교 대시보드 |

두 시스템은 **임베딩·청킹·검색·프롬프트·온도가 모두 동일**하며 오직 생성 LLM만 다르도록 격리 설계되었다 (`config/settings.yaml` 단일 소스).

## 2. 빠른 시작

```bash
# 0) 작업 폴더로 이동 + 가상환경
cd fsar-agent-poc
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 0-1) 스캔 PDF용 Tesseract 한국어 OCR
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-kor
# macOS: brew install tesseract tesseract-lang

# 1) System A — Gemini API Key 설정
cp .env.example .env
# .env 편집해서 GOOGLE_API_KEY=AIzaSy... 입력
# (Codespaces Secret으로도 가능)

# 2) System B — Ollama + EXAONE 모델 (선택적)
bash scripts/setup_ollama.sh
# 기본: exaone3.5:2.4b. 7.8B로 바꾸려면:
#   OLLAMA_MODEL=exaone3.5:7.8b bash scripts/setup_ollama.sh

# 3) PDF 배치 (Google Drive 등에서 수동 다운로드)
cp ~/Downloads/신고리3,4호기FSAR1장_vol.1.pdf data/raw/
# 다운로드 링크: https://drive.google.com/file/d/1lfC8Af_YgbcwOXoxl0aYc2AKQdiHcHHG/view

# 4) 인덱싱 (OCR → 청크 → ChromaDB, 약 10~15분)
bash scripts/run_ingest.sh

# 5-A) 검색만 검증 (LLM 호출 없음, API Key 불필요)
python scripts/demo_retrieval.py

# 5-B) System A 평가 (6문항, 약 1분)
bash scripts/run_eval.sh --systems A --limit 6

# 5-C) System B 평가 (CPU 환경에서 6문항 약 16분, GPU면 1~2분)
bash scripts/run_eval.sh --systems B --limit 6

# 5-D) 두 결과 병합 → 비교 리포트
python scripts/merge_reports.py \
    $(ls -t data/eval/report_*.json | grep -v merged | grep -v partial | head -2)

# 6) Streamlit 대시보드 (정성 비교 + 정량 평가 + 인간 채점 UI)
bash scripts/run_ui.sh
# → http://localhost:8501
# 3개 페이지:
#   - 질의 비교: 즉시 A·B 답변 비교
#   - 정량 평가: 골드셋 기반 자동 메트릭
#   - 인간 평가(대화형): 사용자가 직접 채점, 누적 통계·CSV/MD 내보내기
```

## 3. 디렉토리 구조

```
fsar-agent-poc/
├── README.md                            본 문서
├── requirements.txt                     Python 의존성
├── .env.example                         API Key 템플릿
├── .gitignore                           PDF/벡터스토어/리포트 제외
│
├── config/                              ⭐ 통제 변수 단일 소스
│   ├── settings.yaml                    모든 통제·독립 변수
│   └── prompts/
│       ├── qa_ko.txt                    Q&A 공통 프롬프트
│       └── summary_ko.txt               요약 공통 프롬프트
│
├── data/
│   ├── raw/                             PDF 배치 위치 (gitignore)
│   ├── processed/                       OCR된 페이지 JSONL (gitignore)
│   └── eval/
│       ├── README.md                    골드셋 스키마·보정 가이드
│       ├── goldset.jsonl                Q&A 정답셋 (검증 6 + 드래프트 6)
│       ├── PoC_Report_20260605.md       PoC 평가 결과 보고서
│       └── report_*.{json,csv,md}       자동 생성 리포트 (gitignore)
│
├── vectorstore/                         ChromaDB 영속화 (gitignore)
│
├── src/                                 ⭐ 핵심 소스
│   ├── config.py                        settings.yaml/env 로더
│   ├── ingestion/                       PDF → 청크 파이프라인
│   ├── embeddings/                      BGE-M3 공통 임베더
│   ├── retrieval/                       ChromaDB 검색
│   ├── agents/                          Gemini / EXAONE 에이전트
│   ├── evaluation/                      메트릭 · 러너 · 리포트
│   └── ui/                              Streamlit 대시보드
│
├── scripts/                             실행 스크립트
│   ├── run_ingest.sh                    인덱싱
│   ├── run_eval.sh                      평가
│   ├── run_ui.sh                        대시보드
│   ├── setup_ollama.sh                  Ollama 설치·모델 다운로드
│   ├── demo_retrieval.py                LLM 없이 검색만 시연
│   ├── merge_reports.py                 A/B 리포트 병합
│   ├── list_gemini_models.py            사용 가능한 Gemini 모델 확인
│   └── regen_report_md.py               JSON → MD 재생성
│
└── tests/
    └── test_metrics.py                  메트릭 단위 검증
```

## 4. 파일별 상세 설명

### 4.1 프로젝트 메타데이터

| 파일 | 내용 |
|---|---|
| `README.md` | 본 문서 — 빠른 시작, 디렉토리, 파일별 설명, 워크플로 |
| `requirements.txt` | Python 의존성. PyMuPDF·KoNLPy·pytesseract·sentence-transformers·chromadb·langchain·google-generativeai·ollama·streamlit·plotly 등 |
| `.env.example` | `GOOGLE_API_KEY`, `OLLAMA_HOST` 템플릿. 사용자가 `.env`로 복사 후 키 입력 |
| `.gitignore` | `.venv/`, `*.pyc`, `.env`, `data/raw/*.pdf`, `data/processed/*.jsonl`, `vectorstore/`, `data/eval/report_*` 제외 |

### 4.2 설정 (`config/`)

| 파일 | 내용 |
|---|---|
| `config/settings.yaml` | **통제·독립 변수 단일 소스**. 경로·청킹·임베딩·검색·생성 파라미터·System A/B 모델 정의·평가 토크나이저 등. 두 시스템 비교의 공정성 핵심 |
| `config/prompts/qa_ko.txt` | Q&A 공통 프롬프트. 컨텍스트만 근거로 답변, 출처 `[출처: p.N §X.Y]` 형식 인용 강제, 정보 부족 시 "자료에서 확인할 수 없습니다" 응답 규칙 |
| `config/prompts/summary_ko.txt` | 요약 전용 프롬프트. 5문장 이내 요약 + 출처 인용 |

### 4.3 데이터 (`data/`)

| 파일 | 내용 |
|---|---|
| `data/raw/` | 사용자가 PDF를 수동 배치하는 폴더 (gitignore, `.gitkeep`만 추적) |
| `data/processed/` | OCR 추출 결과(`*.pages.jsonl`) 저장 폴더 (gitignore) |
| `data/eval/README.md` | 골드셋 스키마 정의 및 정답 보정 절차 가이드 |
| `data/eval/goldset.jsonl` | Q&A 정답셋 (Q01~Q06: 페이지 8 기반 검증, Q07~Q12: 드래프트). 스키마: `{qid, category, question, gold_answer, gold_keywords, gold_pages}` |
| `data/eval/PoC_Report_20260605.md` | **본 PoC 평가 결과 보고서**. 목적·개요·실험조건·실험결과·결과분석(A vs B) 구성 |

### 4.4 인제스천 (`src/ingestion/`)

| 파일 | 내용 |
|---|---|
| `src/ingestion/pdf_loader.py` | PyMuPDF로 PDF 페이지별 텍스트 추출. **빈 텍스트 페이지는 Tesseract 한국어 OCR로 자동 폴백** (300 DPI 렌더링 → kor+eng 인식). `PageRecord` 데이터클래스 (`page`, `section`, `section_title`, `text`, `source_file`, `via_ocr`) yield. 절 번호 정규식 자동 감지. 진행률 로그 출력 |
| `src/ingestion/chunker.py` | LangChain `RecursiveCharacterTextSplitter`로 청킹. 한국어 separators (`\n\n`, `\n`, `。`, `. `, ` `). `Chunk` 데이터클래스 (id·text·메타데이터). 청크 크기/오버랩은 `settings.yaml`에서 통제 |
| `src/ingestion/build_index.py` | **인덱싱 CLI**. `data/raw/*.pdf`를 모두 처리 → 페이지 JSONL 저장 → 청킹 → ChromaDB 적재. `--reset` 옵션으로 컬렉션 초기화 가능. `scripts/run_ingest.sh`가 호출 |

### 4.5 임베딩·검색 (`src/embeddings/`, `src/retrieval/`)

| 파일 | 내용 |
|---|---|
| `src/embeddings/embedder.py` | BGE-M3 임베더(`SentenceTransformer`). `lru_cache`로 모델 단일 로드. **HF Hub 429 회피를 위해 임포트 전 `HF_HUB_OFFLINE=1` 자동 설정** (캐시된 모델만 사용, `HF_TOKEN`이 있으면 해제). `embed(texts)` 단일 진입점 |
| `src/retrieval/retriever.py` | ChromaDB cosine top-k 검색. `RetrievedChunk` 데이터클래스 (text·page·section·score·`format_for_prompt()`). `_get_collection`은 영속화 디렉토리로 `PersistentClient` 생성. `add_chunks`/`search`/`count`/`reset_collection`/`format_context` 함수 제공 |

### 4.6 에이전트 (`src/agents/`)

| 파일 | 내용 |
|---|---|
| `src/agents/base_agent.py` | **공통 RAG 추상 클래스** `FSARAgent`. 흐름: 질문→검색(공통)→프롬프트 조립(공통)→`_generate`(시스템별)→`AgentResponse` 반환. `AgentResponse` 데이터클래스 (answer·retrieved·latency_ms·prompt_tokens·completion_tokens·cost_usd). 비용 계산은 `settings.yaml`의 단가 적용 |
| `src/agents/gemini_agent.py` | **System A**. `google-generativeai` SDK로 Gemini 호출. **429 자동 재시도** (에러 메시지의 `retry_delay` 정규식 파싱 + 지수 백오프 + 안전 마진 1초). **RPD(일일 한도) 초과는 즉시 중단** (`PerDay`/`RequestsPerDay` 문자열 감지, `RPD_EXHAUSTED` 예외) |
| `src/agents/ollama_agent.py` | **System B**. `ollama` 파이썬 클라이언트로 로컬 추론. `OLLAMA_HOST` 환경변수 사용. `seed`·`temperature`·`num_predict` 옵션 동일 적용 |
| `src/agents/factory.py` | 에이전트 팩토리. `get_agent('A')` → `GeminiAgent`, `get_agent('B')` → `OllamaAgent`. UI·평가 러너의 공통 진입점 |

### 4.7 평가 (`src/evaluation/`)

| 파일 | 내용 |
|---|---|
| `src/evaluation/metrics.py` | **메트릭 정의**. KoNLPy Okt(또는 Mecab) 형태소 기반 `token_prf1` (Precision·Recall·F1), `exact_match`, `keyword_hit_rate` (substring/morph_exact 모드), `retrieval_pr` (Recall@k·Precision@k). 형태소 분석기 미설치 시 정규식 fallback |
| `src/evaluation/runner.py` | **평가 러너**. 골드셋 로드 → 시스템별 순회 → 문항당 `agent.ask` → 메트릭 계산 → JSON/CSV/MD 리포트 생성. 주요 기능: ①**RPM 페이싱** (시스템별 `request_pacing_sec`), ②**부분 저장** (`report_<TS>.partial.json`을 매 문항 후 덮어쓰기), ③**RPD 감지 시 즉시 중단**, ④**`--limit N`** 인자 (앞 N개만 평가), ⑤**문항별 비교 표 + 자동 코멘트**(승부 카운트·평균 지연·키워드 적중률·요약 판정) |
| `src/evaluation/human_eval.py` | **인간 채점 로그 관리**. `HumanEvalRecord` 데이터클래스, `append_record/load_records/aggregate_stats/export_csv/export_markdown`. 저장: `data/eval/human_eval.jsonl` (소스, git 추적). 다차원 채점(정확성 3단계 + 평점 1~5 + 코멘트) 및 시스템·카테고리별 집계 |

### 4.8 UI (`src/ui/`)

| 파일 | 내용 |
|---|---|
| `src/ui/app.py` | **Streamlit 비교 대시보드**. 3개 페이지: ①**질의 비교** (동일 질문을 A·B에 동시 입력 → 좌우 답변·지연·토큰·검색 청크 미리보기 병기), ②**정량 평가** (`runner.run()` 호출하고 결과 차트로 시각화, 기존 `report_*.json` 로드 가능), ③**인간 평가(대화형)** — 사용자가 PDF로 정답을 알고 두 시스템 답변을 직접 채점. 채팅 스레드, 다차원 채점(정확성/평점/코멘트), 누적 통계 사이드바, 검색 청크 미리보기, CSV/MD 내보내기 |

### 4.9 공통 모듈 (`src/`)

| 파일 | 내용 |
|---|---|
| `src/config.py` | `settings.yaml` + `.env` 로더. `load_settings()`(LRU 캐시), `resolve_path(rel)`(프로젝트 루트 기준 절대 경로), `load_prompt(name)`, `google_api_key()`(미설정 시 RuntimeError), `ollama_host()` |
| `src/__init__.py`, `src/*/__init__.py` | 패키지 마커 (빈 파일) |

### 4.10 실행 스크립트 (`scripts/`)

| 파일 | 내용 |
|---|---|
| `scripts/run_ingest.sh` | `python -m src.ingestion.build_index "$@"` 래퍼. 인자 전달 (`--reset` 등) |
| `scripts/run_eval.sh` | `python -m src.evaluation.runner "$@"` 래퍼. `--systems`/`--limit` 전달 |
| `scripts/run_ui.sh` | `streamlit run src/ui/app.py`. `STREAMLIT_PORT` 환경변수로 포트 변경 가능 (기본 8501) |
| `scripts/setup_ollama.sh` | **Ollama 자동 설치·실행·모델 pull**. Codespace에 systemd 없으므로 `nohup ollama serve` 백그라운드 실행. 서버 readiness 폴링 후 `ollama pull $OLLAMA_MODEL` (기본 `exaone3.5:7.8b`, 환경변수로 변경 가능) |
| `scripts/demo_retrieval.py` | **LLM 호출 없이 검색만 시연**. 골드셋 첫 5문항으로 검색 → 각 청크의 page·section·score·미리보기 출력 + 키워드 적중 확인. `data/eval/retrieval_demo.md` 생성. **API Key 불필요** |
| `scripts/merge_reports.py` | 별도 시점에 실행된 A/B 리포트 JSON 두 개(이상) 병합 → `report_merged_<TS>.{json,csv,md}` 생성. 동일 시스템 키가 중복되면 마지막 것이 이김 |
| `scripts/list_gemini_models.py` | 본인 `GOOGLE_API_KEY`로 호출 가능한 Gemini 모델 목록 조회. 404 'model not found' 디버깅용 |
| `scripts/regen_report_md.py` | 기존 `report_*.json`을 다시 MD로 변환 (재평가 없이 리포트 형식만 갱신할 때) |

### 4.11 테스트 (`tests/`)

| 파일 | 내용 |
|---|---|
| `tests/test_metrics.py` | 메트릭 단위 검증. `exact_match`, `keyword_hit_rate`, `token_prf1`, `retrieval_pr` 6개 케이스. KoNLPy 미설치 환경에서도 fallback으로 통과 |

## 5. 통제·독립 변수 (공정 비교 설계)

`config/settings.yaml`이 단일 진실 소스.

**통제 (Controlled — 두 시스템 동일)**:
- 임베딩: `BAAI/bge-m3`
- 청크: 800자 / overlap 100자
- 검색: ChromaDB cosine, top-k=5
- 프롬프트: `config/prompts/qa_ko.txt`
- 생성: temperature=0.2, max_tokens=1024, seed=42
- 컨텍스트 포맷: `[출처: p.{page} §{section} {title}]\n{본문}`

**독립 (Independent — 시스템별 다름)**:
- LLM: `gemini-2.5-flash-lite` vs `exaone3.5:2.4b` (또는 7.8b)
- 페이싱: 5초 (A, RPM 15 회피) vs 0초 (B, 로컬)

## 6. 워크플로 다이어그램

```
                    PDF (45MB 스캔)
                          │
                  PyMuPDF 텍스트 추출
                          │
              [빈 페이지?] ──예──→ Tesseract OCR (kor+eng, 300 DPI)
                          │
                  PageRecord JSONL
                          │
              RecursiveCharacterTextSplitter (800/100)
                          │
                Chunk + 메타데이터
                          │
                   BGE-M3 임베딩
                          │
                ChromaDB PersistentClient
                          │
       ┌──────────────────┴──────────────────┐
       │                                     │
   (질의 입력)                            (평가 골드셋)
       │                                     │
   임베딩→top-k=5                         각 문항 ×
   공통 프롬프트                       {System A, System B}
       │                                     │
   ┌───┴───┐                            메트릭 계산
   │       │                                 │
  A답변   B답변                       JSON·CSV·MD 리포트
   │       │                                 │
  Streamlit                          merge_reports.py
   비교 UI                                   │
                                    비교 보고서 (MD)
```

## 7. 무료 티어 제약 메모

| 모델 | RPM | RPD | 메모 |
|---|---|---|---|
| gemini-2.5-flash-lite | 15 | **1,000** | ★ 권장 (`settings.yaml` 기본) |
| gemini-2.0-flash | 15 | 200 | 대안 |
| gemini-2.5-flash | 5 | **20** | 빠르게 한도 도달 |

EXAONE은 로컬이라 무제한이지만, **7.8B는 GPU 필수, 2.4B는 CPU 가능** (Codespace 4-core에서 문항당 2~4분).

## 8. 라이선스·법적 주의

- 본 PoC 코드는 사용자가 정한 라이선스로 배포 (별도 명시 없을 시 저장소 소유자 권리)
- EXAONE-3.5는 **EXAONE AI Model License**로 비상업 연구·평가 한정 (PoC OK, 상용은 Qwen2.5 등 Apache 2.0 대안 권장)
- Tesseract OCR: Apache 2.0
- BGE-M3: MIT
- 신고리 3,4호기 FSAR은 한수원 정보 공개용 문서 — **재배포 권리는 별도**이므로 본 저장소는 PDF 자체를 git에 두지 않음 (`.gitignore` 처리)

## 9. 참고 산출물

- **PoC 평가 보고서**: [`data/eval/PoC_Report_20260605.md`](data/eval/PoC_Report_20260605.md)
- **골드셋 가이드**: [`data/eval/README.md`](data/eval/README.md)
- **Streamlit 대시보드**: `bash scripts/run_ui.sh` 후 http://localhost:8501

---

*저장소: `jin22190121/PlantAgent` · 브랜치: `claude/nuclear-safety-ai-agent-hbDZu`*
