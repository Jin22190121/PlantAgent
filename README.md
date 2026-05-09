# PlantAgent — 정상운전 절차서 기반 NPP AI Agent (PoC)

Westinghouse 4-loop PWR의 **Appendix 19-1 §A (MODE 5 → MODE 4) 기동**
시나리오를 대상으로, 실시간 시뮬레이터 + 절차서 RAG + LLM 운전 안내를
통합한 PoC 시스템.

## 구성

```
npp_agent/
├── sim/                       # 실시간 시뮬레이션 엔진
│   ├── state.py               #   PlantState dataclass
│   ├── dynamics.py            #   ODE step + 알람 판정
│   └── engine.py              #   asyncio 백그라운드 루프
├── ingest/
│   ├── pdf_loader.py          # PDF 텍스트 추출 (zlib 기반, 외부 의존 없음)
│   └── procedures_data.py     # Appendix 19-1 §A 20개 step 정의
└── mcp_servers/
    ├── simulator.py           # 시뮬레이터 도구 (state/action) 노출
    ├── procedure.py           # ChromaDB 기반 절차 RAG 검색
    ├── training.py            # 행동 로그
    └── router.py              # 도구 → 서버 라우팅
web_server.py                  # FastAPI: SSE state, action, chat
static/
├── index.html                 # 실시간 운전 화면 (백엔드 연동)
└── demo.html                  # 자동 재생 데모 (50s, 백엔드 불필요)
```

## 실행

```bash
pip install -r requirements.txt

# (선택) AI 채팅 사용 시 환경변수 설정
export GOOGLE_API_KEY=...

uvicorn web_server:app --host 0.0.0.0 --port 8000
```

브라우저에서:
- **`http://localhost:8000/`** — 실시간 시뮬레이터 + AI Agent
- **`http://localhost:8000/demo`** — 자동 재생 영상용 데모 (백엔드 불필요)

## 주요 기능

### 시뮬레이터 (백그라운드 ODE 루프)
- 1 real sec = 60 sim sec 가속 (조정 가능)
- 1차 계통 열수지: RCP 운전열 + 붕괴열 - RHR 냉각
- PZR 가열/살수, steam bubble 형성 자동 판정
- 가열률 (1시간 슬라이딩 윈도우) 자동 모니터
- 알람: 가열률 100°F/hr 초과, ΔT 320°F 초과, RCP NPSH 부족 등

### AI Agent (Gemini 2.5 Flash)
- 절차서 RAG 검색 (`search_procedure`, `search_by_mode_transition`, ...)
- 시뮬레이터 직접 조작 (`set_pzr_heater`, `start_rcp`, ...)
- 모든 안내에 절차 step ID 인용

### UI
- 좌: 실시간 플랜트 상태 (SSE) - 클릭으로 펌프/밸브 직접 조작
- 중: AI 채팅 + 도구 호출 트레이스
- 우: 절차 진행 트래커 (상태로부터 자동 step 진행 표시)
- 하: 알람 띠

## PDF 학습

```bash
python -m npp_agent.ingest.pdf_loader path/to/Section19.0.pdf
```

추출된 텍스트는 `procedures_data.py`의 step 정의와 비교/보완 가능.

## 테스트

```bash
# 시뮬레이터만 단독 실행 (LLM 불필요)
python -c "
import asyncio
from npp_agent.sim import SimulationEngine

async def main():
    e = SimulationEngine(sim_speed=600)
    await e.start()
    e.set_pzr_heater(True)
    for i in range(10):
        await asyncio.sleep(1)
        s = e.get_state()
        print(f't={s[\"sim_time_s\"]:.0f}s T_RCS={s[\"T_RCS_avg_F\"]:.1f}°F '
              f'T_PZR={s[\"PZR_temp_F\"]:.1f}°F bubble={s[\"steam_bubble_formed\"]}')
    await e.stop()

asyncio.run(main())
"
```

## 라이선스

MIT
