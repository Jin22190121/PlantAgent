"""FastAPI web server for the NPP AI Agent PoC.

Endpoints:
  GET  /                  → static UI (static/index.html)
  GET  /demo              → static/demo.html (canned animated demo)
  GET  /plant/state       → current simulator snapshot (REST)
  GET  /plant/stream      → live state SSE (every ~1 s)
  POST /sim/action        → direct operator action (HITL approve dispatch)
  POST /sim/control       → engine control (pause/resume/reset/speed)
  POST /chat/stream       → AI agent chat (SSE)
  DELETE /chat/{sid}      → reset chat session
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from npp_agent.sim import SimulationEngine
from npp_agent.mcp_servers.simulator import SimulatorMCP
from npp_agent.mcp_servers.procedure import ProcedureMCP
from npp_agent.mcp_servers.training  import TrainingMCP
from npp_agent.mcp_servers.router    import MCPRouter


# ── Engine + MCP routing (shared singleton) ─────────────────
engine = SimulationEngine(sim_speed=60.0)
simulator = SimulatorMCP(engine=engine)
procedure = ProcedureMCP()
training  = TrainingMCP()
router    = MCPRouter([simulator, procedure, training])

# ── LLM (optional — only if GOOGLE_API_KEY is set) ─────────
HAS_LLM = False
model = None
try:
    if os.environ.get("GOOGLE_API_KEY"):
        import google.generativeai as genai
        from google.generativeai.types import FunctionDeclaration, Tool

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        mcp_tools = Tool(function_declarations=[
            FunctionDeclaration(
                name="get_plant_state",
                description="현재 플랜트 전체 파라미터(온도/압력/수위/MODE/펌프 등) 반환",
                parameters={"type": "object", "properties": {}}),
            FunctionDeclaration(
                name="get_alarm_list",
                description="현재 활성 알람 목록 반환",
                parameters={"type": "object", "properties": {}}),
            FunctionDeclaration(
                name="search_procedure",
                description="상황 설명으로 정상운전 절차서 RAG 검색",
                parameters={"type": "object", "properties": {
                    "situation": {"type": "string"}}, "required": ["situation"]}),
            FunctionDeclaration(
                name="get_step",
                description="절차 step ID(예: App19-1-A-4)로 단일 절차 반환",
                parameters={"type": "object", "properties": {
                    "step_id": {"type": "string"}}, "required": ["step_id"]}),
            FunctionDeclaration(
                name="get_next_step",
                description="현재 step의 다음 step 반환",
                parameters={"type": "object", "properties": {
                    "step_id": {"type": "string"}}, "required": ["step_id"]}),
            FunctionDeclaration(
                name="search_by_mode_transition",
                description="MODE 전환(from→to)에 해당하는 절차들",
                parameters={"type": "object", "properties": {
                    "from_mode": {"type": "integer"},
                    "to_mode":   {"type": "integer"}},
                    "required": ["from_mode", "to_mode"]}),
            FunctionDeclaration(
                name="get_cautions",
                description="특정 step의 CAUTION/NOTE 반환",
                parameters={"type": "object", "properties": {
                    "step_id": {"type": "string"}}, "required": ["step_id"]}),
            # Operator action tools (HITL: AI proposes, server applies)
            FunctionDeclaration(
                name="set_pzr_heater", description="가압기 히터 on/off",
                parameters={"type": "object", "properties": {
                    "on": {"type": "boolean"}}, "required": ["on"]}),
            FunctionDeclaration(
                name="set_pzr_spray", description="가압기 살수 밸브 open/close",
                parameters={"type": "object", "properties": {
                    "open": {"type": "boolean"}}, "required": ["open"]}),
            FunctionDeclaration(
                name="set_rhr_pump", description="RHR 펌프 on/off",
                parameters={"type": "object", "properties": {
                    "on": {"type": "boolean"}}, "required": ["on"]}),
            FunctionDeclaration(
                name="start_rcp", description="RCP(1~4) 기동",
                parameters={"type": "object", "properties": {
                    "pump_id": {"type": "integer"}}, "required": ["pump_id"]}),
            FunctionDeclaration(
                name="stop_rcp", description="RCP(1~4) 정지",
                parameters={"type": "object", "properties": {
                    "pump_id": {"type": "integer"}}, "required": ["pump_id"]}),
            FunctionDeclaration(
                name="set_charging_flow", description="충전 유량(gpm) 설정",
                parameters={"type": "object", "properties": {
                    "gpm": {"type": "number"}}, "required": ["gpm"]}),
            FunctionDeclaration(
                name="set_letdown_flow", description="방출 유량(gpm) 설정",
                parameters={"type": "object", "properties": {
                    "gpm": {"type": "number"}}, "required": ["gpm"]}),
            FunctionDeclaration(
                name="set_sg_level_target", description="SG 협역 수위 목표(%) 설정",
                parameters={"type": "object", "properties": {
                    "pct": {"type": "number"}}, "required": ["pct"]}),
            FunctionDeclaration(
                name="open_msiv", description="MSIV 개방",
                parameters={"type": "object", "properties": {}}),
            FunctionDeclaration(
                name="advance_time", description="시뮬레이션 시간 진행(초)",
                parameters={"type": "object", "properties": {
                    "seconds": {"type": "number"}}, "required": ["seconds"]}),
        ])
        HAS_LLM = True
except Exception as exc:  # pragma: no cover
    print(f"[warn] LLM init failed: {exc}")
    HAS_LLM = False


SYSTEM_PROMPT = """당신은 한국 운전원을 보조하는 정상운전 절차서 AI Agent입니다.
대상 시나리오: Westinghouse 4-loop PWR, Appendix 19-1 §A (MODE 5 → MODE 4 기동).

규칙:
1) 모든 안내에 절차 step ID(예: App19-1-A-4)를 인용하세요.
2) 작업 시작 시 get_plant_state로 상태를 확인하고, search_by_mode_transition 또는
   search_procedure로 해당 절차를 찾으세요.
3) 운전원 조작이 필요하면 적절한 도구(set_pzr_heater, start_rcp, set_rhr_pump 등)를
   호출하세요. 호출 전 CAUTION을 반드시 인용하세요.
4) 절차서에 없는 내용은 생성하지 마세요. 모르면 모른다고 답하세요.
5) 응답은 한국어로 간결하게."""


# ── chat sessions ───────────────────────────────
chat_sessions: Dict[str, Any] = {}


# ── FastAPI app ─────────────────────────────────
app = FastAPI(title="NPP AI Agent PoC")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def _start_sim():
    await engine.start()


@app.on_event("shutdown")
async def _stop_sim():
    await engine.stop()


# ── plain pages ─────────────────────────────────
@app.get("/")
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/demo")
async def demo_page():
    with open("static/demo.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── plant state REST + SSE ──────────────────────
@app.get("/plant/state")
async def plant_state():
    return engine.get_state()


@app.get("/plant/stream")
async def plant_stream():
    async def gen() -> AsyncIterator[bytes]:
        while True:
            payload = engine.get_state()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── direct operator actions (UI buttons) ────────
class ActionReq(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@app.post("/sim/action")
async def sim_action(req: ActionReq):
    """Apply an operator action directly (used by UI buttons / HITL approval)."""
    result = router.call(req.tool, req.args)
    return JSONResponse(result)


class ControlReq(BaseModel):
    op: str  # pause | resume | reset | speed
    value: Any = None


@app.post("/sim/control")
async def sim_control(req: ControlReq):
    if req.op == "pause":
        engine.pause()
        return {"ok": True, "paused": True}
    if req.op == "resume":
        engine.resume()
        return {"ok": True, "paused": False}
    if req.op == "reset":
        engine.reset()
        return {"ok": True, "scenario": "cold_shutdown_initial"}
    if req.op == "speed":
        return engine.set_sim_speed(float(req.value or 60))
    raise HTTPException(400, f"unknown op: {req.op}")


# ── AI chat ──────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


def _summarize_result(tool_name: str, result: dict) -> str:
    if not isinstance(result, dict):
        return "OK"
    if "error" in result:
        return f"⚠ {result['error']}"
    if tool_name == "get_plant_state":
        return (f"MODE {result.get('mode')} · T_RCS {result.get('T_RCS_avg_F'):.1f}°F · "
                f"P_RCS {result.get('P_RCS_psig'):.0f} psig · "
                f"PZR {result.get('PZR_level_pct'):.1f}%")
    if tool_name == "get_alarm_list":
        return f"활성 알람 {result.get('alarm_count', 0)}건"
    if tool_name in ("search_procedure", "search_by_mode_transition"):
        ids = [p["id"] for p in result.get("procedures", [])]
        return "검색됨: " + ", ".join(ids[:3]) if ids else "결과 없음"
    if tool_name == "get_step":
        return f"{result.get('id', '?')} — {result.get('title', '')}"
    return result.get("status") or "완료"


async def run_agent_stream(session_id: str, question: str) -> AsyncIterator[str]:
    if not HAS_LLM:
        msg = ("LLM 사용 불가 — GOOGLE_API_KEY 환경변수를 설정하세요. "
               "시뮬레이터와 UI는 정상 동작합니다.")
        yield f"data: {json.dumps({'type': 'error', 'text': msg}, ensure_ascii=False)}\n\n"
        return

    import google.generativeai as genai

    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[
            {"role": "user",  "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["네, 이해했습니다. 정상운전 절차서에 따라 안내합니다."]},
        ])
    chat = chat_sessions[session_id]

    def evt(t: str, **payload):
        return f"data: {json.dumps({'type': t, **payload}, ensure_ascii=False)}\n\n"

    def send(message, tools=None, max_retry=3):
        for k in range(max_retry):
            try:
                return chat.send_message(message, tools=tools) if tools else chat.send_message(message)
            except Exception as ex:
                if "429" in str(ex) and k < max_retry - 1:
                    time.sleep((k + 1) * 10)
                    continue
                raise

    yield evt("status", text="🤖 분석 시작...")
    try:
        response = send(question, tools=mcp_tools)
        for _ in range(8):
            fcs, texts = [], []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fcs.append(part.function_call)
                elif hasattr(part, "text") and part.text.strip():
                    texts.append(part.text.strip())

            if not fcs:
                yield evt("answer", text="\n".join(texts) or "(응답 없음)")
                return

            tool_parts = []
            for fc in fcs:
                name = fc.name
                args = dict(fc.args)
                yield evt("tool_start", tool=name, args=str(args)[:120])
                result = router.call(name, args)
                yield evt("tool_done", tool=name,
                          summary=_summarize_result(name, result))
                tool_parts.append(genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": json.dumps(result, ensure_ascii=False)})))
            yield evt("status", text="📊 결과 분석 중...")
            response = send(tool_parts, tools=mcp_tools)
        yield evt("error", text="최대 도구 호출 단계 도달")
    except Exception as ex:
        yield evt("error", text=f"오류: {ex}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(run_agent_stream(req.session_id, req.message),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"status": "session reset"}


@app.get("/health")
async def health():
    return {"ok": True, "llm": HAS_LLM,
            "sim_running": engine._task is not None and not engine._task.done()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)
