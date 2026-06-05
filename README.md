# PlantAgent — 원자력발전소 운영절차서 기반 AI Agent PoC

> Westinghouse 4-loop PWR 운영절차서 3종 (**GOP / EOP / AOP**)을 학습하고 가상 시뮬레이터에서
> 실시간 운전을 지원하는 **LangGraph + HITL 안전 게이트** 기반 PoC 시스템.
>
> 운전원이 한 번 채팅 입력하면, AI Agent가 절차서를 인용하며 매 단계를 안내하고,
> 모든 실 조작은 **Human-in-the-Loop 승인 게이트**를 통과해야만 시뮬레이터에 반영됩니다.

---

## 목차

1. [핵심 기능](#핵심-기능)
2. [빠른 시작](#빠른-시작)
3. [절차서 / 시나리오 소개](#절차서--시나리오-소개)
4. [전체 파일 구조와 역할](#전체-파일-구조와-역할)
5. [아키텍처 다이어그램](#아키텍처-다이어그램)
6. [환경 변수](#환경-변수)
7. [Git 영구 보존 정책](#git-영구-보존-정책)

---

## 핵심 기능

| 영역 | 내용 |
|---|---|
| **운영절차서 학습** | GOP·EOP·AOP 3종 PDF를 ChromaDB에 3-tier 청크(Step / Body / CAUTION)로 인덱싱. 한국어 키워드 강화 + TF-IDF 어휘 검색을 RRF로 융합. **Recall@3 = 90.9 %, Top-1 = 77.3 %** (PoC 합격 기준 통과) |
| **가상 시뮬레이터** | Python 단순 ODE 모델. 1초 = 시뮬 60초 가속. `PlantState` 데이터클래스 + 22개 운전원 도구 (PZR 히터·RCP 1~4·SI·MFW·MSIV·atm dump·CR 철수 등) |
| **시나리오 3종** | `scenario_gop_heatup` (콜드셧다운 → 핫셧다운), `scenario_eop_reactor_trip_si` (원자로 트립+SI), `scenario_aop_cr_inaccessible` (주제어실 접근 불능) |
| **LangGraph 8 노드** | assess_state → retrieve_procedure → plan_action → **approval_gate** → execute_action → verify_outcome → log_step → respond |
| **HITL 안전 계층** | R/A/E tier 분류 + HMAC 승인 토큰 + `interrupt()` 기반 운전원 승인 모달. E-tier 도구는 토큰 없이는 라우터에서 차단 |
| **단계 진행 추적** | `Annotated[list[str], _append_unique]` reducer로 `completed_step_ids` 영속화. `_next_unmarked_step()`이 다음 step을 결정론적으로 선택 |
| **다중 조작 게이트** | `MULTI_ACTION_GATES["GOP-A-6"]` — 4기 RCP 모두 기동될 때까지 한 step으로 유지 |
| **도구 args 자동 보정** | `_fix_tool_args()` — LLM이 `pump_id` 누락 시 plant 상태에서 다음 OFF인 펌프 자동 선택 |
| **LLM 모델 로테이션** | Gemini 무료 6개 모델 자동 체인 (2.5-flash → 2.5-flash-lite → 2.0-flash → 2.0-flash-lite → 1.5-flash → 1.5-flash-8b) + Groq fallback |
| **실시간 UI** | 3-pane 디자인. 좌: 플랜트 상태 + Chart.js 트렌드. 중: step-grouped 채팅. 우: 절차 트래커 자동 체크. HITL 승인 모달 |
| **관측성** | `LANGFUSE_*` 환경변수 설정 시 LangChain 콜백 자동 부착 |
| **데모용 시각화** | `static/demo.html` (50초 GOP 데모), `static/gop_demo.html` (Phase 1-4 통합 데모), `docs/architecture.svg` (시스템 구조도) |

---

## 빠른 시작

### 1. 사전 준비
```bash
git clone https://github.com/Jin22190121/PlantAgent.git
cd PlantAgent
git checkout claude/nuclear-plant-ai-agent-MYwtm
```

### 2. 환경변수 설정 (`.env`)
```bash
cat > .env <<EOF
# 필수 — Gemini API 키 (https://aistudio.google.com/apikey)
GOOGLE_API_KEY=AIza...

# 권장 — Groq fallback (https://console.groq.com/keys)
# Gemini 한도 도달 시 자동 우회
GROQ_API_KEY=gsk_...

# 선택 — LangFuse 관측성 (https://cloud.langfuse.com)
# LANGFUSE_PUBLIC_KEY=pk-...
# LANGFUSE_SECRET_KEY=sk-...
EOF
```

### 3. 의존성 설치 + 서버 기동
```bash
./scripts/run.sh deps      # pip install -r requirements.txt
./scripts/run.sh           # 백그라운드 기동 (terminal 닫혀도 살아있음)
./scripts/run.sh logs      # 로그 확인
```

### 4. 브라우저 접속

| URL | 용도 |
|---|---|
| `http://localhost:8000/` | **라이브 운전 화면** (실시간 시뮬레이터 + AI Agent) |
| `http://localhost:8000/demo` | 자동 재생 데모 (50초) |
| `http://localhost:8000/gop-demo` | Phase 1-4 GOP 통합 데모 (72초) |
| `http://localhost:8000/health` | 시스템 상태 JSON |
| `http://localhost:8000/docs` | FastAPI 자동 문서 |

### 5. 사용 흐름
1. 상단 드롭다운에서 시나리오 선택 (GOP/EOP/AOP)
2. 채팅에 `"기동을 시작합니다"` 한 번 입력
3. AI가 첫 step 안내 → 운전원이 `"확인했습니다"` 또는 **➡ 다음 단계** 버튼 클릭
4. E-tier 행동 시 **HITL 승인 모달** 자동 팝업 → 운전원 [✓ 승인]
5. 시뮬레이터 상태 변화 → 절차 트래커 자동 체크 → 다음 step
6. MODE 4 진입 시 AI가 절차 완료 선언

### 6. RAG 품질 검증
```bash
python3 scripts/eval_rag.py --doc-filter --tier A
# → Recall@3 = 90.9 %, Top-1 = 77.3 %
```

---

## 절차서 / 시나리오 소개

| 약어 | 한국어 | 출처 | PoC 시나리오 |
|---|---|---|---|
| **GOP** | 일반/정상운전절차서 (General Operating Procedure) | Westinghouse Tech Sys Manual §19.0 App19-1 §A (NRC ML11223A342) | **콜드셧다운 → 핫셧다운 기동** (MODE 5 → 4) — 9개 step |
| **EOP** | 비상운전절차서 (Emergency Operating Procedure) | Ginna Station E-0 (NRC ML17263A589) | **원자로 트립 + 안전주입** — 16개 immediate-action steps |
| **AOP** | 비정상운전절차서 (Abnormal Operating Procedure) | Point Beach AOP-10 (NRC ML030730736) | **주제어실 접근 불능** — 양 호기 정지 + 현장 운전 (17 step) |

---

## 전체 파일 구조와 역할

> 각 파일이 어떤 코드를 담고 있는지 한눈에 파악할 수 있도록 정리했습니다.

```
PlantAgent/                                    # 저장소 루트
│
├── README.md                                  # ★ 이 파일
├── requirements.txt                           # Python 의존성
├── .gitignore                                 # data/checkpoints, *.sqlite, .env 제외
│
├── web_server.py              ★ 405줄 — FastAPI 메인 (실행 진입점)
├── main.py                       170줄 — Legacy CLI 단발 실행
├── model_list.py                       — Gemini 모델 목록 조회 유틸
│
├── docs/                                      # 설계 문서 + 다이어그램
│   ├── DESIGN.md              ★ 558줄 — 7-Task 설계문서 (확장된 설계계획)
│   ├── architecture.svg          14KB — 6-Layer 시스템 구조도 (브라우저로 열림)
│   ├── flow.svg                   9KB — 운전원 1회 요청 처리 시퀀스 다이어그램
│   └── architecture.html             — SVG 뷰어 + PNG 저장 안내
│
├── npp_agent/                ★ 코어 패키지 (총 ~2,200줄)
│   ├── __init__.py
│   │
│   ├── sim/                                   # 가상 NPP 시뮬레이터 (Task 3)
│   │   ├── __init__.py
│   │   ├── state.py            183줄 — PlantState dataclass + 시나리오 3종 프리셋
│   │   │                                · sim_time_s · mode · T_RCS_avg_F · P_RCS_psig
│   │   │                                · PZR_level_pct · PZR_temp_F · RCP_running[4]
│   │   │                                · RHR_pump_on · PZR_heater_on · si_signal
│   │   │                                · reactor_tripped · turbine_tripped · MSIV_open
│   │   │                                · control_room_evacuated 등 30+ 변수
│   │   ├── dynamics.py         ~190줄 — ODE 우변 + 알람 자동 평가
│   │   │                                · saturation_pressure_psig() 보간 함수
│   │   │                                · _decay_heat_mw() 시간함수
│   │   │                                · step(s, dt) 통합 (RCS·PZR·SG·CNMT)
│   │   │                                · evaluate_alarms() 자동 setpoint 위반 감지
│   │   └── engine.py           ~140줄 — asyncio 백그라운드 루프 + 22개 도구 메서드
│   │                                    · set_pzr_heater · start_rcp · stop_rcp
│   │                                    · actuate_si · manual_reactor_trip 등
│   │                                    · reset(scenario_id) — 시나리오 스왑
│   │
│   ├── ingest/                                # PDF 학습 파이프라인 (Task 1)
│   │   ├── __init__.py
│   │   ├── pdf_loader.py        66줄 — zlib 기반 PDF 텍스트 추출 (외부 의존 無)
│   │   ├── chunking.py         113줄 — Tier B/C 청크 (Body sliding + CAUTION 분리)
│   │   │                                · slide_body() · extract_cautions()
│   │   │                                · collect_aux_chunks()
│   │   ├── lexical.py           92줄 — TF-IDF 한국어 토큰 인덱스 + RRF 융합
│   │   │                                · TfIdfIndex · rrf_fuse()
│   │   └── procedures_data.py  345줄 — GOP 9 + EOP 16 + AOP 17 step 큐레이션
│   │                                    · ALL_PROCEDURES · DOC_META
│   │                                    · ACTION_TO_STEP_ID · SETPOINT_INDEX
│   │
│   ├── mcp_servers/                           # MCP 도구 서버
│   │   ├── __init__.py
│   │   ├── simulator.py         90줄 — 시뮬레이터 도구 22개 (R 3 + E 19)
│   │   ├── procedure.py        348줄 — ChromaDB 3-tier RAG + 9개 검색 도구
│   │   │                                · search_procedure (doc_type, tier 필터)
│   │   │                                · get_step / get_next_step
│   │   │                                · search_by_mode_transition
│   │   │                                · search_by_setpoint_violation
│   │   │                                · list_documents / list_steps_in_doc
│   │   │                                · 한국어 키워드 강화 (_KW_EXPAND)
│   │   │                                · 벡터 + 어휘 하이브리드 (RRF)
│   │   ├── training.py          57줄 — 행동 로그 (log_action / get_session_summary)
│   │   └── router.py            53줄 — Tier 가드 적용 도구 디스패치
│   │                                    · source=agent/operator_ui/graph_internal
│   │
│   ├── safety/                                # HITL 안전 계층 (Task 4)
│   │   ├── __init__.py
│   │   └── tier_guard.py       145줄 — R/A/E 분류 + HMAC 승인 토큰
│   │                                    · TIER_MAP (35개 도구 분류)
│   │                                    · issue_token() · verify_token()
│   │                                    · 10분 만료 + thread_id 바인딩
│   │
│   ├── graph/                                 # LangGraph 오케스트레이션 (Task 5)
│   │   ├── __init__.py
│   │   ├── state.py             67줄 — AgentState TypedDict + reducer
│   │   │                                · final_messages: Annotated[..., add]
│   │   │                                · completed_step_ids: Annotated[..., _append_unique]
│   │   │                                · current_step_id: Annotated[..., _last_wins]
│   │   ├── nodes.py            ~570줄 — 8개 노드 + 헬퍼
│   │   │                                · assess_state · retrieve_procedure
│   │   │                                · plan_action (system prompt + LLM 호출)
│   │   │                                · approval_gate (interrupt for E-tier)
│   │   │                                · execute_action · verify_outcome
│   │   │                                · log_step · respond
│   │   │                                · _next_unmarked_step() · _fix_tool_args()
│   │   │                                · MULTI_ACTION_GATES · _CONFIRM_RE
│   │   │                                · End-of-procedure short-circuit
│   │   └── build.py            105줄 — StateGraph 컴파일 + AsyncSqliteSaver + LangFuse
│   │
│   ├── llm/                                   # LLM 추상화 (Task 7)
│   │   ├── __init__.py
│   │   └── factory.py          120줄 — Gemini 6개 모델 자동 로테이션 + Groq
│   │                                    · DEFAULT_GEMINI_MODELS (6 모델)
│   │                                    · _make_gemini(model_name)
│   │                                    · _build_chain() — .with_fallbacks()
│   │                                    · llm_status() (헬스 진단)
│   │
│   └── observability/                         # 관측성 (Task 5)
│       ├── __init__.py
│       └── langfuse_setup.py    72줄 — LangFuse 콜백 (env-gated)
│                                        · langfuse_callbacks() — [] 또는 [handler]
│                                        · langfuse_status() — 상태 진단
│
├── static/                                    # UI (Task 6)
│   ├── index.html          ★ ~1180줄 — 라이브 운영 UI
│   │                                    · 3-pane 레이아웃 + Chart.js 트렌드
│   │                                    · HITL 승인 모달
│   │                                    · 절차 인용 클릭 시 원문 모달
│   │                                    · step-section 헤더 + step-block 그룹화
│   │                                    · SSE 이벤트 핸들러 (8 종)
│   ├── demo.html               705줄 — Phase 0 자동 재생 (50초, 시뮬레이터만)
│   └── gop_demo.html           766줄 — Phase 1-4 통합 GOP 데모 (72초)
│                                        · LangGraph 노드 트레이스 + HITL 모달
│                                        · 9 step 전체 시뮬
│
├── scripts/                                   # 운영 스크립트
│   ├── run.sh                  100줄 — nohup 백그라운드 launcher (Codespaces 친화)
│   │                                    · start / stop / restart / logs / status / deps
│   └── eval_rag.py              91줄 — RAG 골든셋 측정 (Recall@k / Top-1)
│
└── data/                                      # 데이터 (런타임 생성 포함)
    ├── procedures/                            # PDF 추출 텍스트
    │   ├── GOP-Westinghouse-Sec19.txt   54 KB — Westinghouse §19.0 본문
    │   ├── EOP-E-0_Ginna.txt            37 KB — Ginna E-0 본문
    │   └── AOP-10_PointBeach.txt        16 KB — Point Beach AOP-10 본문
    ├── eval/
    │   └── golden_set.json                   # 22 한국어 질의 골든셋
    ├── index/                  *.gitignored — ChromaDB persistent (런타임 생성)
    └── checkpoints/            *.gitignored — AsyncSqliteSaver (런타임 생성)
        └── agent.sqlite
```

| 합계 | 약 5,000 줄 (코드 + UI + 데이터 큐레이션) |
|---|---|

---

## 아키텍처 다이어그램

```
┌──────────────────────────────────────────────────────────────────┐
│ ① 사용자       👤 운전원 + 🖥 웹 브라우저 (static/index.html)     │
├──────────────────────────────────────────────────────────────────┤
│ ② 웹 서버      ⚡ FastAPI (REST + SSE)                            │
│                · /plant/stream · /chat/stream · /sim/action      │
│                · /chat/approve · /health                          │
├──────────────────────────────────────────────────────────────────┤
│ ③ AI 오케스트레이션  🧠 LangGraph 8 노드 (Async SqliteSaver 영속화)│
│                assess → retrieve → plan → ⚠ approval_gate         │
│                → execute → verify → log → respond                 │
├──────────────────────────────────────────────────────────────────┤
│ ④ 도구 + 안전  🛡 Tier Guard + 🔧 MCP Router + 🤖 LLM Factory     │
│                · R/A/E HMAC 승인 토큰                              │
│                · Gemini 6 모델 자동 로테이션 + Groq               │
│                · 📊 LangFuse (env-gated)                          │
├──────────────────────────────────────────────────────────────────┤
│ ⑤ 핵심 서비스  🏭 Simulator + 📚 RAG + 📝 Training Log            │
│                · PlantState + ODE + 22 도구                       │
│                · 3-tier 청크 + 하이브리드 검색                     │
├──────────────────────────────────────────────────────────────────┤
│ ⑥ 데이터       📕 PDF × 3 · 🗄 ChromaDB · 🗄 SqliteSaver · 🌐 LLM  │
└──────────────────────────────────────────────────────────────────┘
```

자세한 다이어그램은 [docs/architecture.svg](docs/architecture.svg) (브라우저로 직접 열림) 또는
[docs/architecture.html](docs/architecture.html) 뷰어 참조.

---

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `GOOGLE_API_KEY` | ✅ | — | Gemini API 키. 미설정 시 LLM 비활성 |
| `GROQ_API_KEY` | 권장 | — | Gemini 한도 도달 시 자동 fallback |
| `GEMINI_MODELS` | 선택 | (6 모델 기본) | 콤마 구분 사용자 지정 체인 |
| `GEMINI_MODEL` | 선택 (legacy) | — | 단일 모델 강제 |
| `MODEL_PROVIDER` | 선택 | `gemini` | `gemini` 또는 `groq` |
| `LANGFUSE_PUBLIC_KEY` | 선택 | — | LangFuse 트레이스 활성화 |
| `LANGFUSE_SECRET_KEY` | 선택 | — | (같은 짝) |
| `LANGFUSE_HOST` | 선택 | cloud | 자체호스팅 URL |
| `NPP_APPROVAL_SECRET` | 선택 | ephemeral | HMAC 토큰 시드 |
| `PORT` | 선택 | 8000 | `run.sh`가 사용하는 포트 |

---

## Git 영구 보존 정책

### 현재 보존 상태
- **브랜치**: `claude/nuclear-plant-ai-agent-MYwtm` (GitHub 원격 저장소에 영구 보존)
- **태그 시작점**: 본 README 작성 시점 — `poc-v1.0` 태그로 마킹
- **저장소 URL**: `https://github.com/Jin22190121/PlantAgent`

### 영구 보존 권장 추가 조치
1. **GitHub Release 생성** — 저장소 → Releases → "Draft a new release" → 태그 `poc-v1.0` 선택
2. **Archive 다운로드** — GitHub Release는 자동으로 `.tar.gz` / `.zip` 아카이브 보존
3. **PR을 통해 `main` 브랜치 병합** — 장기적으로 `main`에 머지하면 안전
4. **README + DESIGN 문서를 정기적으로 백업**해 두면 코드 손실 시에도 재구축 가능

### 복원 방법
```bash
# Release archive에서 복원
wget https://github.com/Jin22190121/PlantAgent/archive/refs/tags/poc-v1.0.tar.gz
tar -xzf poc-v1.0.tar.gz
cd PlantAgent-poc-v1.0
./scripts/run.sh deps && ./scripts/run.sh
```

---

## PoC 합격 지표

| 기준 | 목표 | 실측 |
|---|---|---|
| RAG Recall@3 | ≥ 90 % | **90.9 %** ✅ |
| RAG Top-1 정확도 | ≥ 70 % | **77.3 %** ✅ |
| 시뮬레이터 시나리오 | 3 종 | GOP/EOP/AOP 모두 작동 ✅ |
| HITL 안전 게이트 | 토큰 검증 | E-tier 자동 차단 ✅ |
| 절차 인용 가시성 | step ID 클릭 → 원문 | UI 모달 ✅ |
| LLM 한도 회복력 | 자동 fallback | 6 Gemini 모델 로테이션 + Groq ✅ |

---

## 라이선스 / 사용 데이터 출처

- **GOP 절차서**: USNRC HRTD Westinghouse Technology Systems Manual (공개)
- **EOP 절차서**: Rochester G&E Ginna E-0 (NRC ADAMS 공개)
- **AOP 절차서**: Point Beach AOP-10 (NRC ADAMS 공개)
- **코드**: MIT
