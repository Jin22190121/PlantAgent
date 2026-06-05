# 정상운전 절차서 기반 NPP AI Agent PoC 설계 계획

## Context

기존 `PlantAgent` 저장소는 비상운전(LOCA, SBO 등) 시나리오를 가정하고 만들어졌으나(`procedure.py`의 E-0/E-1/ECA-0.0, `simulator.py`의 LOCA 상태값), 사용자는 **정상운전 절차서**(Westinghouse Technology Systems Manual Section 19.0 — Plant Operations, NRC ML11223A342) 기반의 실시간 운전 지원 AI Agent를 개발하고자 한다.

본 계획은 다음 세 가지를 다룬다.
1. PDF(정상운전 절차서) 학습 방법
2. PoC 대상 절차 사례 — **Appendix 19-1 A절 (MODE 5 → MODE 4)**
3. PoC를 위한 가상 NPP 시뮬레이션 — **Python 내장 ODE 모델**

PDF 분석 결과 본 문서는 다음 구조이다.
- 19.1 Introduction
- 19.2 Plant Heatup (Initial Conditions, Operations)
- 19.3 Reactor Startup to Minimum Load
- 19.4 Power Operations
- 19.5 Plant Shutdown
- **Appendix 19-1: Plant Startup from Cold Shutdown**
  - A. MODE 5 → MODE 4 (20 step) ← **PoC 대상**
  - B. MODE 4 → MODE 3 (8 step)
  - C. MODE 3 → MODE 1 (29 step)
- Checklist 1, Checklist 2

기존 인프라(ChromaDB RAG, MCP-style 서버, FastAPI 웹서버)는 그대로 재활용하되, **LLM·오케스트레이션 계층은 LangGraph + LangFuse + Claude(또는 오픈소스) 조합으로 재설계**하고, **모든 조작 행동은 Human-in-the-Loop(HITL)로 운전원 승인을 받는 구조**로 전환한다.

---

## Task 1. 정상운전 절차서 PDF 학습 방법 설계

### 1-1. 파싱 및 구조화 파이프라인 (오프라인 1회 수행)

스크립트: 신규 `npp_agent/ingest/pdf_loader.py`

1. **텍스트 추출**: `pypdf` 또는 `pdfplumber` 사용 (`requirements.txt`에 추가). PoC에서는 빠른 개발을 위해 `pypdf`를 우선.
2. **계층 분리**: 정규식과 페이지 기준으로 절(section)·부록(appendix)·step 단위 분해.
   - Level 0: 본문 vs 부록
   - Level 1: 절 번호 (19.1 ~ 19.5) / 부록 절 (A, B, C)
   - Level 2: 부록 step 번호 (1 ~ 20)
3. **CAUTION/NOTE 분리 색인**: 절차서에서 `CAUTION:`, `NOTE:`로 시작하는 블록을 별도 객체로 분리하여 step과 연결.
4. **메타데이터 추출** (Gemini 2.5 Flash로 LLM 후처리):
   - `mode_transition`: e.g. `"5→4"`
   - `parameters`: `["RCS_pressure", "PZR_level", "T_avg"]`
   - `setpoints`: `{"heatup_rate_max": "100°F/hr", "PZR_level_target": "25%"}`
   - `actions`: 동사 추출 (`verify`, `energize`, `withdraw`, `transfer` ...)

### 1-2. 청크 전략 (3-tier)

| Tier | 단위 | 예시 |
|---|---|---|
| **A. Step 청크** | 부록 step 1개 | "Step A-4: Energize pressurizer heaters and begin pressurizer heatup" |
| **B. 설명 청크** | 본문 200~500 token 슬라이딩 | "19.2.2 Operations 도입부" |
| **C. CAUTION 청크** | CAUTION/NOTE 단위 | "Do not exceed a heatup rate of 100°F/hr" |

각 청크는 동일 ChromaDB collection 내에 `tier` 메타데이터로 구분.

### 1-3. 듀얼 인덱스

- **Vector Index** (ChromaDB, 기존 사용): 의미 검색용
- **Setpoint 룩업 테이블** (in-memory dict 또는 SQLite): `parameter → [관련 step IDs]`로 즉시 조회. 시뮬레이터에서 setpoint 위반 시 곧장 관련 절차 step을 가져오기 위함.

### 1-4. 검색 인터페이스 확장 (`procedure.py` 교체)

기존 `search_procedure(situation)` 외에 다음 도구 추가:
- `get_step(step_id)` — 특정 step 정확 조회 (e.g. `"App19-1-A-5"`)
- `search_by_mode_transition(from_mode, to_mode)` — MODE 전환 컨텍스트 검색
- `search_by_setpoint_violation(parameter, value)` — 시뮬레이터 알람 → 절차 매핑
- `get_cautions(step_id)` — 해당 step의 CAUTION 모두 반환
- `get_next_step(current_step_id)` — 순차 진행

### 1-5. 검증 방법

- 골든셋: PDF에서 핵심 step 5~10개를 골라 `"PZR 수위 25%로 낮춰야 하는 시점은?"` 류의 자연어 질의 → 정답 step ID로 검색되는지 확인.
- Recall@3 ≥ 90%, top-1 정확도 ≥ 70%를 PoC 성공 기준으로 설정.

---

## Task 2. PoC 대상 절차 — Appendix 19-1 A절 (MODE 5 → MODE 4)

**선정 근거**:
- 명확한 시작/종료 조건: 초기 T_avg 150~160°F, P_RCS 320~400 psig (solid), PZR 25% 목표 → 종료 시 T_avg 200°F 도달 + MODE 4 진입
- 단 20개 step으로 PoC 범위에 적합
- 핵심 변수가 적음 (T_RCS, P_RCS, T_PZR, L_PZR, RCP 운전 수)
- 명확한 setpoint와 CAUTION:
  - 가열률 ≤ 100°F/hr
  - PZR-spray 온도차 ≤ 320°F
  - PZR 수위 80% 미만 → 충전펌프 기동
  - 200°F 도달 전 RCP 5분 운전 후 시료
- 향후 확장 경로: B절(MODE 4→3), C절(MODE 3→1)로 동일 아키텍처에서 점진적 추가 가능

**핵심 step 시나리오 (요약)**:

| Step | 행동 | 자동 검증 가능 setpoint |
|---|---|---|
| 1 | 운전감독 허가 확인 | (수동 입력) |
| 2 | SG 수위를 33±5% NR로 설정 | SG_level 28~38% |
| 3 | RCP 봉수 주입 확인 | seal_inj_flow > 0 |
| 4 | PZR 히터 기동 | heater_on == True |
| 5 | PZR steam bubble 형성 (a/b/c 세부 단계) | 320~400 psig 유지, 25% 목표 |
| 6 | RCP 기동, 5분 후 시료 | RCP 4기 운전, 5분 경과 |
| 7 | T_RCS < 160°F 유지 (RHR 열교환기 유량 조절) | 가열률 모니터 |
| 8 | RHR 펌프 정지 | RHR_pump == OFF |
| 9 | T_RCS 200°F까지 자연 가열 | T_RCS → 200°F |
| 10~11 | 화학 시방, PZR level 자동 모드 | (정보성) |
| 12 | 정지뱅크 인출 + 정지여유도 확인 | SDM 확인 |
| 13 | MODE 4 Tech Spec 확인 (Checklist 1) | (체크리스트) |
| 14~16 | VCT 수소 블랭킷, MSIV 개방, PZR 가열 지속 | |
| 17~20 | 복수 정화, 응축 정렬, 진공 형성, 터빈 가열 | (상태 정보) |

**PoC MVP 범위**: Step 1 ~ Step 9까지를 우선 자동 안내 (1차 시스템 위주)

---

## Task 3. 가상 NPP 시뮬레이션 설계

### 3-1. 아키텍처

기존 `npp_agent/mcp_servers/simulator.py`를 **교체**하고 `npp_agent/sim/` 하위에 다음 추가:
- `npp_agent/sim/state.py` — 상태 변수 정의 (`@dataclass`)
- `npp_agent/sim/dynamics.py` — ODE 우변 함수
- `npp_agent/sim/engine.py` — 시간 진행 루프 (Euler integration, `asyncio.Task`로 백그라운드 실행)
- `npp_agent/sim/operator.py` — 운전원 조작 명령 큐
- `npp_agent/mcp_servers/simulator.py` — 위 엔진을 MCP 도구로 노출

### 3-2. 상태 변수 (MODE 5→4 PoC 충분 범위)

```python
@dataclass
class PlantState:
    # 시간
    sim_time_s: float = 0.0
    mode: int = 5
    # 1차 계통
    T_RCS_avg_F: float = 155.0          # 콜드셧다운 초기값
    P_RCS_psig: float = 360.0           # solid
    PZR_level_pct: float = 90.0         # solid 초기
    PZR_temp_F: float = 155.0
    # 펌프/밸브
    RCP_running: list[bool] = [False]*4
    RHR_pump_on: bool = True
    PZR_heater_on: bool = False
    PZR_spray_open: bool = False
    charging_flow_gpm: float = 0.0
    letdown_flow_gpm: float = 0.0
    HCV_128_open: bool = True           # RHR→CVCS letdown
    PCV_131_setpoint_psig: float = 360.0
    # 2차 계통 (단순)
    SG_level_NR_pct: float = 100.0      # wet layup 시작
    MSIV_open: bool = False
    # 핵계측 (MODE 5→4 PoC에서는 사용 약함)
    K_eff: float = 0.97
    # 알람/이벤트
    alarms: list[str] = []
    heatup_rate_F_per_hr: float = 0.0
```

### 3-3. ODE 모델 (단순화)

PoC에서는 정밀도보다 절차 검증이 목적이므로 다음 단순 모델 채택:

```
# 1차 계통 열수지
dT_RCS/dt = (Q_RCP_heat + Q_decay - Q_letdown_cooling - Q_RHR) / (m_RCS * Cp)
  - Q_RCP_heat = N_RCP_running × ~5 MW (RCP 기계 손실)
  - Q_decay = decay heat (시간 함수, MODE 5에서 ~0.5%P0)
  - Q_RHR = RHR_pump_on × heat_exchanger_capacity

# RCS 압력 (solid 일 때)
dP_RCS/dt = K_solid * (charging_flow - letdown_flow)

# Steam bubble 형성 후
P_RCS = P_sat(T_PZR)  if PZR_steam_bubble else solid_balance

# PZR 수위
dL_PZR/dt = (charging_flow - letdown_flow) / pzr_volume + thermal_expansion_term

# PZR 온도
dT_PZR/dt = (Q_heater - Q_spray - Q_loss) / (m_PZR * Cp)
  - Q_heater = PZR_heater_on × ~1.4 MW
  - Q_spray = PZR_spray_open × ~spray_flow * Cp * (T_PZR - T_spray)

# 가열률 모니터 (CAUTION: ≤ 100°F/hr)
heatup_rate = (T_RCS_avg_F - T_RCS_avg_F_60min_ago)  # 60분 슬라이딩 윈도우
```

수치적분: 단순 명시적 Euler, dt = 1 sim-second. NumPy 미사용 가능.

### 3-4. 운전원 조작 인터페이스 (MCP 도구)

| 도구 | 설명 |
|---|---|
| `get_plant_state()` | 현재 상태 스냅샷 |
| `get_alarm_list()` | 활성 알람 |
| `set_pzr_heater(on: bool)` | PZR 히터 on/off |
| `set_pzr_spray(open: bool)` | PZR 살수밸브 |
| `start_rcp(pump_id: 1~4)` | RCP 기동 |
| `stop_rcp(pump_id)` | RCP 정지 |
| `set_charging_flow(gpm)` | 충전 유량 |
| `set_letdown_flow(gpm)` | 방출 유량 |
| `set_rhr_pump(on: bool)` | RHR 펌프 |
| `open_msiv()` | MSIV 개방 |
| `set_sg_level_target(pct)` | SG 수위 목표 |
| `advance_time(seconds, speed=1.0)` | 시뮬레이션 시간 진행 (가속) |
| `reset(scenario="cold_shutdown_initial")` | 시나리오 리셋 |

### 3-5. 자동 알람·이벤트 규칙

- `heatup_rate > 100°F/hr` → CAUTION 알람 → AI Agent가 "Step 7 - 가열률 초과: RHR 열교환기 유량 증가 필요" 자동 안내
- `T_PZR - T_spray > 320°F` → CAUTION 알람
- `PZR_level < 80% AND charging_pump == OFF` → 자동 충전펌프 기동 안내
- `T_RCS ≥ 200°F` → "MODE 4 진입 가능, Checklist 1 확인" 이벤트
- `seal_inj_flow == 0 AND any RCP running` → 즉시 위험 알람

### 3-6. 시뮬레이션 실행 모드

- **수동 step**: `advance_time(60)` 호출로 1분 단위 진행
- **자동 진행**: `web_server.py`의 `asyncio.Task`로 1초 = 시뮬 1분 가속(60×). UI에서 일시정지/재개 가능.

### 3-7. UI/UX (기존 `static/index.html` 확장)

- 좌측: 실시간 상태 패널 (T_RCS, P_RCS, PZR 수위, MODE)
- 중앙: 채팅창 (AI Agent와 대화)
- 우측: 절차 진행 트래커 (현재 step 강조 표시, 완료 step 체크)
- 하단: 알람 띠

---

## Task 4. Human-in-the-Loop (HITL) 구조

원자력 운전절차는 **오류 시 안전·법적 책임이 큼** → AI가 단독으로 조작을 수행해서는 안 된다. 모든 조작성 도구는 운전원의 **명시적 승인**을 거쳐야 한다.

### 4-1. 행동 분류 (3-tier)

| Tier | 분류 | 예시 | 승인 방식 |
|---|---|---|---|
| **R (Read)** | 읽기 전용 | `get_plant_state`, `search_procedure` | 자동 실행 |
| **A (Advise)** | 안내/추천 | "Step 4: PZR 히터 기동 권고" | 운전원 표시만, 승인 불필요 |
| **E (Execute)** | 실 조작 | `set_pzr_heater(on=True)` | **운전원 승인 필수** |

### 4-2. LangGraph `interrupt()` 기반 승인 플로우

```
[Plan Node] → AI가 "다음 step은 PZR 히터 기동" 판단
   ↓
[Approval Gate Node] → interrupt({step_id, action, reason, expected_state})
   ↓ (운전원이 UI에서 승인/거부/수정)
[Execute Node] → Command(resume=...) → 시뮬레이터 도구 호출
   ↓
[Verify Node] → 도구 실행 후 상태 변화 확인
   ↓
[Log Node] → TrainingMCP로 행동 기록
```

`langgraph.checkpoint.sqlite.SqliteSaver`로 그래프 상태를 영속화 → 운전원 승인 대기 중 세션 끊겨도 복구 가능.

### 4-3. UI 승인 컴포넌트 사양

승인 모달 표시 항목:
- 인용된 절차 step ID + 원문 텍스트
- 제안 행동 (도구명 + 인자)
- 현재 상태 vs 예상 결과 상태 비교
- CAUTION/NOTE 자동 표시
- 버튼: `승인 / 거부 / 수정 / 절차 다시 검색`

### 4-4. 거부·수정 처리

- 거부: AI에게 거부 사유와 함께 재계획 요청 → `Command(resume={"approved": False, "reason": "..."})` 
- 수정: 운전원이 인자 변경 (예: 충전유량 60→75 gpm) → AI는 이를 새 컨텍스트로 인식

### 4-5. 안전 가드 (HITL 우회 금지)

- 시뮬레이터 도구는 LangGraph 노드 외부에서는 호출되지 않도록 라우터에서 강제
- E-tier 도구는 직접 `router.call()` 시 거부, 반드시 `approval_gate` 통과한 토큰을 동반해야 함

---

## Task 5. LangGraph + LangFuse 도입

### 5-1. LangGraph (필수 채택)

**선정 근거**:
- 절차서 진행은 본질적으로 상태 머신 → LangGraph 그래프 모델과 1:1 매핑
- `interrupt()` / `Command(resume=)` 으로 HITL 네이티브 지원 (Task 4)
- Checkpointer로 장기 세션 복구
- 도구 호출 흐름이 LangSmith/LangFuse와 자동 연동

**그래프 노드 설계**:
```
START
  → assess_state (시뮬레이터 상태 조회 + 알람 체크)
  → retrieve_procedure (현재 mode + 진행 step 기반 RAG 검색)
  → plan_action (Claude/오픈소스 LLM이 다음 step 결정)
  → approval_gate (interrupt → 운전원 승인)
  → execute_action (도구 호출)
  → verify_outcome (예상 상태 도달 확인)
  → log_step (TrainingMCP)
  → check_completion (MODE 4 도달? → END / 계속)
  → assess_state  (루프)
```

상태(`AgentState` TypedDict): `current_step_id`, `plant_state`, `proposed_action`, `approval_status`, `procedure_history`, `alarm_buffer`.

### 5-2. 관측·평가 도구 — **LangFuse 권장 (LangSmith 비권장)**

| 항목 | LangSmith | LangFuse |
|---|---|---|
| 라이선스 | 상용 (LangChain Inc.) | 오픈소스 (MIT) + 호스팅 옵션 |
| 자체호스팅 | 유료 Enterprise | **무료 self-host (Docker Compose)** |
| 데이터 주권 | 외부 클라우드 | **온프레미스 가능** |
| LangGraph 통합 | 네이티브 | `langfuse-langgraph` 핸들러 또는 OTel |
| 비용 (PoC) | 트레이스/월 한도 | 0원 (자체 운영) |

**원자력 도메인은 데이터 주권이 중요** → 외부 SaaS에 운전 로그를 보내지 않는 LangFuse 자체호스팅을 권장.

LangFuse에서 추적할 항목:
- 그래프 실행 트레이스 (노드별 입출력)
- 운전원 승인/거부 이벤트
- 도구 호출 (특히 E-tier)
- 절차 step 인용 정확도
- 응답 지연시간, 토큰/원가

### 5-3. 평가 데이터셋 (LangFuse Datasets)

PoC 합격 판정용 시나리오 셋:
- `golden/cold_to_hot_normal.json` — 정상 진행 시나리오 (20 step 완주)
- `golden/heatup_rate_violation.json` — 가열률 초과 알람 시 회복 시나리오
- `golden/pzr_level_low.json` — PZR 수위 80% 미만 → 충전펌프 안내
- `golden/wrong_mode_request.json` — 운전원이 잘못된 절차 요청 시 거부 응답

각 시나리오는 LangFuse에 등록되어 회귀 테스트용으로 자동 실행.

---

## Task 6. 웹 대화창 UI/UX 설계

### 6-1. 화면 레이아웃 (3-Pane)

```
┌────────────────────────────────────────────────────────────────┐
│  Header: 시나리오명 / 시뮬레이션 시각 / MODE 표시 / Pause·Reset │
├──────────┬───────────────────────────────────┬───────────────┤
│          │                                   │                │
│  Left    │           Center                  │     Right      │
│  Panel   │           Chat Panel              │     Panel      │
│          │                                   │                │
│  Plant   │   - AI 안내 메시지 (절차 인용)    │  Procedure     │
│  Mimic   │   - HITL 승인 모달               │  Tracker       │
│  +       │   - 운전원 입력창                 │  (체크리스트)  │
│  Trends  │                                   │                │
│          │                                   │  Alarm List    │
├──────────┴───────────────────────────────────┴───────────────┤
│  Footer: 현재 행동 / LangFuse trace 링크 / 헬프              │
└────────────────────────────────────────────────────────────────┘
```

### 6-2. Left Panel — 플랜트 상태

- **계기 다이얼**: T_RCS (analog 게이지), P_RCS, PZR 수위 (수직 막대)
- **추세 그래프**: 최근 60분 T_RCS, P_RCS, PZR 수위 (Chart.js / Recharts)
- **상태 표시등**: RCP 1~4, RHR 펌프, MSIV, PZR 히터, PZR 살수밸브 (녹/적색 LED 형식)
- **MODE 표시**: 큼지막한 배지 (5/4/3/2/1)
- **Tech Spec 위반 표시**: 가열률, 압력-온도 한도

### 6-3. Center Panel — 채팅 + HITL

- **메시지 카드 디자인**:
  - AI 메시지: 좌측 정렬, 절차 인용은 노란색 배경 박스 (step ID 클릭 시 원문 모달)
  - 운전원 메시지: 우측 정렬
  - CAUTION: 빨간 테두리 + 경고 아이콘
  - NOTE: 파란 배경 + 정보 아이콘
- **승인 모달 (HITL)**:
  ```
  ┌──────────────────────────────────────┐
  │ ⚠ 조작 승인 요청                      │
  │ ─────────────────────────────────── │
  │ 절차 인용: App19-1-A Step 4          │
  │ "Energize pressurizer heaters and    │
  │  begin pressurizer heatup"           │
  │                                      │
  │ 제안 행동:                           │
  │   set_pzr_heater(on=True)           │
  │                                      │
  │ 현재 상태:    PZR 히터 OFF, 155°F   │
  │ 예상 결과:    PZR 히터 ON, 가열 시작│
  │                                      │
  │ ⚠ CAUTION: 가열률 ≤ 100°F/hr 유지  │
  │                                      │
  │ [✓ 승인] [✗ 거부] [✎ 수정] [↻ 재검색]│
  └──────────────────────────────────────┘
  ```
- **빠른 명령 버튼**: "현재 상태 분석", "다음 step", "이 step 건너뛰기 사유", "절차서 검색"
- **타임라인 타임스탬프**: 모든 메시지에 시뮬레이션 시각 + 실제 시각

### 6-4. Right Panel — 절차 진행 트래커

- 현재 절차(Appendix 19-1 A) 20개 step 모두 나열
- 상태별 색상: ⏳완료 / ▶ 진행중 / ⏸ 대기 / ✗실패 / —미시작
- 각 step 클릭 시 우측 패널이 원문 + CAUTION 표시
- 진행률 바: "8/20 (40%)"
- 알람 리스트: 우선순위별 정렬, 클릭 시 관련 절차 자동 검색

### 6-5. 핵심 UX 원칙

1. **인용 가시성**: AI 응답에 절차 step ID가 노란 배지로 강조, 원문 호버/클릭으로 즉시 확인
2. **승인 플로우 단축**: E-tier 도구는 한 번의 클릭으로 승인 가능, 단 CAUTION 항목은 별도 체크박스 강제
3. **상황 인식**: 좌측 패널 변화는 채팅 메시지 발생 시 잠깐 하이라이트
4. **실수 방지**: 거부 시 사유 입력 강제 (LangFuse 학습 데이터)
5. **신뢰성 표시**: 각 응답 하단에 "검색된 step 3개 중 1순위 / 신뢰도 92%" 형태로 표기

### 6-6. 기술 스택

- **프론트엔드**: 기존 단일 `static/index.html` → 다음 옵션 중 택1
  - **(권장) HTMX + Alpine.js**: 의존성 최소, 단일 파일에 가까운 구조 유지, FastAPI SSE와 자연스럽게 결합
  - React + Vite: 복잡한 상태 관리가 필요할 경우
- **차트**: Chart.js (CDN, 빌드 불필요)
- **WebSocket/SSE**: 시뮬레이터 상태 실시간 푸시 (1Hz)
- **승인 모달**: 표준 HTML `<dialog>` 요소

---

## Task 7. LLM 모델 선택 (무료 모델 한정)

### 7-1. Gemini 2.5 Flash 한계 검토

기존 `main.py`는 `gemini-2.5-flash` 사용. 한계:
- 무료 tier: 분당 10회·일 250회(2025년 기준 정책 변경 반영) — PoC 시연 중 도구 호출 다수 발생 시 한도 도달 가능
- 절차 인용 시 영문/한글 혼재 환각 사례 존재
- LangChain 통합은 `langchain-google-genai`로 안정적, LangGraph 도구 호출 OK

### 7-2. 무료 후보 비교 (전부 비용 0원 또는 자체호스팅)

| 모델 | 접근 방식 | 무료 한도 | 도구 호출 | 한국어 | PoC 적합도 |
|---|---|---|---|---|---|
| **Gemini 2.5 Flash** | Google AI Studio API | 10 RPM / 250 RPD | 양호 | 양호 | ★★★★ |
| **Gemini 2.5 Pro** | Google AI Studio API | 5 RPM / 100 RPD | 우수 | 우수 | ★★★★ (저빈도 시) |
| **Llama 4 Scout 17B** (Groq) | Groq Cloud 무료 tier | ~30 RPM, 일 한도 넉넉 | 양호 (function calling) | 보통 | ★★★★ |
| **Llama 4 Maverick** (Cerebras) | Cerebras Cloud 무료 tier | 매우 빠름 (2000 t/s+) | 양호 | 보통 | ★★★ |
| **Qwen3-32B-Instruct** (Ollama 자체호스팅) | 로컬 GPU (24GB+) | 무제한 | 양호 | 우수 | ★★★★★ (GPU 보유 시) |
| **Qwen3-14B-Instruct** (Ollama) | 로컬 GPU (16GB+) | 무제한 | 양호 | 양호 | ★★★★ |
| **Gemma 3 27B** (Ollama) | 로컬 GPU (24GB+) | 무제한 | 보통 | 보통 | ★★★ |
| **DeepSeek-V3** (HF Inference / OpenRouter free) | 외부 API 무료 변종 | 분당 한도 있음 | 우수 | 양호 | ★★★ |
| **Mistral Small 3** (Mistral La Plateforme) | 무료 tier | 1 RPS | 양호 | 보통 | ★★★ |

### 7-3. 권장안 (전부 무료)

**1차 PoC (외부 API 사용 가능 환경)** — **Gemini 2.5 Flash + Groq Llama 4 Scout 이중화**
- 이유:
  - 기존 Gemini 코드 자산 재활용 (마이그레이션 비용 최소)
  - Gemini 한도 도달 시 Groq Llama 4로 자동 fallback (무료 tier 합산 시 PoC 데모 충분)
  - 둘 다 LangChain 네이티브 지원: `langchain-google-genai`, `langchain-groq`
  - LangGraph의 `with_fallbacks()` 또는 `RunnableWithFallbacks`로 한도 초과 자동 전환
- 비용: 0원

**2차 PoC (자체호스팅 가능 환경)** — **Qwen3-32B + Ollama / vLLM**
- 이유:
  - 원자력 도메인은 데이터 주권 중요 → 외부 API 송신 금지 환경 대응
  - 24GB VRAM(RTX 3090 / 4090 / A5000) 1장으로 4bit 양자화 구동 가능, 즉 별도 비용 없이 일반 워크스테이션 1대로 PoC 시연 가능
  - 한국어 성능 우수, 도구 호출 (Hermes 포맷) 안정
  - GPU 미보유 시 Qwen3-14B(16GB) 또는 Qwen3-8B(8GB)로 다운스케일
- LangChain 통합: Ollama → `langchain-ollama`, vLLM → `langchain-openai`(OpenAI 호환 엔드포인트)

**Fallback / 시연용 백업** — **Cerebras Llama 4** (속도가 매우 빨라 시연 효과 큼)

### 7-4. 모델 추상화 계층

`npp_agent/llm/factory.py` 신설 — 모든 옵션이 무료:
```python
def get_chat_model(provider: str = None):
    provider = provider or os.environ.get("MODEL_PROVIDER", "gemini")
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    elif provider == "groq":
        return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    elif provider == "ollama":
        return ChatOllama(model="qwen3:32b", base_url=os.environ["OLLAMA_URL"])
    elif provider == "vllm":
        return ChatOpenAI(base_url=os.environ["VLLM_URL"], api_key="local",
                          model="Qwen/Qwen3-32B-Instruct")
    elif provider == "cerebras":
        return ChatCerebras(model="llama-4-maverick")
    raise ValueError(provider)

def get_chat_model_with_fallback():
    primary = get_chat_model("gemini")
    backup = get_chat_model("groq")
    return primary.with_fallbacks([backup])
```
LangGraph 노드는 모델 객체를 주입받아 사용 → 모델 교체 시 그래프 변경 불요.

### 7-5. 임베딩 (RAG) — 전부 오픈소스 무료

기존 ChromaDB 기본 임베더(Sentence-Transformers `all-MiniLM-L6-v2`, 영어 위주) 대신:
- **권장: BGE-M3** (`BAAI/bge-m3`) — 다국어, 한·영 혼재 절차에 최적, 1024 dim, 로컬 추론
- 대안: `multilingual-e5-large` — 비슷한 성능, 라이선스 동일
- 모두 `sentence-transformers`로 로컬 구동, GPU 없어도 CPU 동작 가능 (PoC 인덱싱 1회분은 수 분)

### 7-6. 모델 한도 대비 운영 전략

- **요청 캐싱**: `langchain.cache.SQLiteCache`로 동일 프롬프트 캐시 (절차 검색 결과 등 반복 호출 방지)
- **프롬프트 압축**: 시스템 프롬프트에 절차 전체를 포함하지 않고 RAG 결과 top-k만 주입
- **비동기 배칭**: LangGraph 노드 중 독립 도구는 `asyncio.gather`로 병렬화하여 응답 지연 최소화
- **모니터링**: LangFuse에서 모델별 호출 횟수·한도 도달 빈도 대시보드 구성, 임계 도달 시 자동 fallback

---

## 핵심 파일 변경/신설 목록

### 수정
- `requirements.txt` — `pypdf`, `numpy`, `langgraph`, `langchain-google-genai`, `langchain-groq`, `langchain-ollama`, `langchain-openai`(vLLM/Cerebras용), `langchain-community`, `langfuse`, `sentence-transformers`(BGE-M3) 추가. 기존 `google-generativeai`는 유지(fallback 1차 모델).
- `npp_agent/mcp_servers/procedure.py` — **교체**. PDF 로드 인덱스 + setpoint 룩업 추가
- `npp_agent/mcp_servers/simulator.py` — **교체**. MODE 5 시작 상태 + 신규 조작 도구 + E-tier 가드
- `npp_agent/mcp_servers/router.py` — Tier 분류(R/A/E), HITL 승인 토큰 검증 추가
- `main.py`, `web_server.py` — Gemini 직접 호출 코드 제거 → LangGraph 실행 진입점으로 교체. SSE 스트림에 그래프 노드 이벤트 + interrupt 이벤트 추가.
- `static/index.html` — 3-pane 레이아웃, HITL 승인 모달, 절차 트래커, 추세 차트 추가

### 신설
- `npp_agent/ingest/pdf_loader.py` — PDF 파싱·메타데이터 추출·ChromaDB 적재
- `npp_agent/ingest/__init__.py`
- `npp_agent/sim/state.py`, `dynamics.py`, `engine.py`, `__init__.py` — ODE 시뮬레이터
- `npp_agent/llm/factory.py` — 모델 추상화 (anthropic/vllm/google 스위칭)
- `npp_agent/llm/__init__.py`
- `npp_agent/graph/state.py` — LangGraph `AgentState` TypedDict
- `npp_agent/graph/nodes.py` — 노드 구현 (assess_state / retrieve_procedure / plan_action / approval_gate / execute_action / verify_outcome / log_step / check_completion)
- `npp_agent/graph/build.py` — `StateGraph` 조립, SqliteSaver 연결, LangFuse 콜백 등록
- `npp_agent/graph/__init__.py`
- `npp_agent/observability/langfuse_setup.py` — LangFuse 클라이언트 초기화
- `npp_agent/safety/tier_guard.py` — R/A/E 분류 + 승인 토큰 검증
- `data/procedures/Section19.0.pdf` — PDF 원본 사본
- `data/index/` — ChromaDB persistent 디렉토리
- `data/checkpoints/` — LangGraph SqliteSaver 디렉토리
- `data/eval/` — LangFuse 평가 데이터셋 (golden cases)
- `docker-compose.langfuse.yml` — LangFuse 자체호스팅 설정 (PoC용)
- `static/components/` — UI 부분 (chart.js, hitl_modal, procedure_tracker, mimic_panel) — 단일 HTML 분할 시
- `.env.example` — `MODEL_PROVIDER`, `ANTHROPIC_API_KEY`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY` 등

### 재사용 (변경 없음)
- `npp_agent/mcp_servers/training.py` — LangGraph `log_step` 노드에서 호출만 변경

---

## 검증 시나리오 (End-to-End)

1. `python -m npp_agent.ingest.pdf_loader` 실행 → ChromaDB에 절차 인덱싱 (1회)
2. `uvicorn web_server:app` 실행
3. 브라우저에서 시뮬레이터 리셋 → MODE 5 콜드셧다운 초기 상태 표시
4. 운전원: "기동 시작" 입력
5. AI Agent 기대 동작:
   - `get_plant_state` 호출 → MODE 5 확인
   - `search_by_mode_transition(5, 4)` 호출 → Appendix A step 1 검색
   - "Step 1: 운전감독 허가 받으셨습니까?" 안내
6. 운전원: "예" → AI는 step 2~3 안내 → SG 수위 설정 / RCP seal 확인
7. AI가 step 4 안내: "PZR 히터 기동" → 운전원이 `set_pzr_heater(on=True)` 도구 호출 트리거
8. 시뮬레이터 시간 진행 → T_PZR 상승 → AI가 step 5 진행 안내
9. AI가 step 6 안내: RCP 1~4 순차 기동
10. RCS 자연 가열 (50°F/hr) → 200°F 근접 시 AI가 Checklist 1 항목 확인 → MODE 4 전환 선언

**합격 기준**:
- 모든 안내 메시지에 절차 step ID 인용 포함
- CAUTION 위반 시 즉시 알람 + 관련 절차 인용
- 시뮬레이터 콜드셧다운 → MODE 4 도달까지 AI 안내만으로 완주 가능

---

## 단계별 구현 순서 제안

1. **인프라 셋업**: LangFuse Docker Compose 기동, Anthropic API 키 / vLLM 엔드포인트 준비, `.env.example` 작성
2. **Task 1 (PDF 학습)**: `pdf_loader.py` + 새 `procedure.py` → 검색 골든셋 검증 (Recall@3 ≥ 90%)
3. **Task 3 (시뮬레이터)**: `sim/` 모듈 + 새 `simulator.py` → MODE 5 시작 상태에서 자연 가열 단위 테스트
4. **Task 7 (LLM 추상화)**: `llm/factory.py` + Claude Sonnet 4.6 1차 적용
5. **Task 5 (LangGraph)**: `graph/` 모듈 + 노드별 단위 테스트 + LangFuse 트레이스 확인
6. **Task 4 (HITL)**: `tier_guard.py` + `approval_gate` 노드 + interrupt/resume 동작 검증
7. **Task 6 (UI)**: 3-pane 레이아웃 + 승인 모달 + 절차 트래커 + 추세 차트
8. **End-to-End 시나리오**: 콜드셧다운 → MODE 4 도달까지 운전원이 AI 안내만으로 완주
9. **회귀 테스트**: LangFuse Datasets 4종 시나리오 자동 실행, 통과율 측정
10. (옵션) MODE 4→3, 3→1 절차 확장 / Qwen3 자체호스팅 모델 swap 검증
