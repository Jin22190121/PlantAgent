import os
import json
import time
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from npp_agent.mcp_servers.simulator import SimulatorMCP
from npp_agent.mcp_servers.procedure import ProcedureMCP
from npp_agent.mcp_servers.training  import TrainingMCP
from npp_agent.mcp_servers.router    import MCPRouter

# ── 초기화 ───────────────────────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

simulator = SimulatorMCP()
procedure = ProcedureMCP()
training  = TrainingMCP()
router    = MCPRouter([simulator, procedure, training])

# ── 도구 정의 ────────────────────────────────────
mcp_tools = Tool(function_declarations=[
    FunctionDeclaration(
        name="get_plant_state",
        description="현재 플랜트 전체 파라미터 상태를 반환합니다.",
        parameters={"type": "object", "properties": {}}
    ),
    FunctionDeclaration(
        name="get_alarm_list",
        description="현재 활성화된 알람 목록을 반환합니다.",
        parameters={"type": "object", "properties": {}}
    ),
    FunctionDeclaration(
        name="search_procedure",
        description="사고 상황 설명으로 관련 절차서를 검색합니다.",
        parameters={
            "type": "object",
            "properties": {
                "situation": {
                    "type": "string",
                    "description": "현재 상황 설명"
                }
            },
            "required": ["situation"]
        }
    ),
    FunctionDeclaration(
        name="get_caution_notes",
        description="특정 절차서의 주의사항을 반환합니다.",
        parameters={
            "type": "object",
            "properties": {
                "procedure_id": {
                    "type": "string",
                    "description": "절차서 ID"
                }
            },
            "required": ["procedure_id"]
        }
    ),
    FunctionDeclaration(
        name="log_action",
        description="훈련생 행동을 기록합니다.",
        parameters={
            "type": "object",
            "properties": {
                "session_id":    {"type": "string"},
                "action":        {"type": "string"},
                "procedure_ref": {"type": "string"}
            },
            "required": ["session_id", "action"]
        }
    ),
])

# ── 세션 저장소 ──────────────────────────────────
# session_id → chat 객체
chat_sessions = {}

# ── FastAPI 앱 ───────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    trainee_name: str = "훈련생"


# ── Agent 실행 (SSE 스트리밍) ────────────────────
async def run_agent_stream(session_id: str, question: str):
    """
    Agent 실행 결과를 SSE로 실시간 스트리밍
    도구 호출 현황 + 최종 답변을 순차 전송
    """

    # 세션 없으면 새로 생성
    if session_id not in chat_sessions:
        system_prompt = """당신은 원자력발전소 교육훈련 보조 AI입니다.
1. 사고 대응이나 훈련 상황 시에는 get_plant_state와 search_procedure를 사용하여 상태를 확인하고 절차서를 인용하여 답변하세요.
2. 원자력 발전소의 용어(예: ECCS, RCS 등), 원리, 일반 지식에 대한 질문에는 전문적이고 상세하게 직접 답변하세요.
3. 구체적인 조치 사항이 필요한 경우에는 반드시 관련 절차서 번호를 인용하세요."""

        chat_sessions[session_id] = model.start_chat(history=[
            {"role": "user",  "parts": [system_prompt]},
            {"role": "model", "parts": ["네, 이해했습니다."]}
        ])

    chat = chat_sessions[session_id]

    def send_event(event_type: str, data: dict):
        """SSE 이벤트 포맷으로 변환"""
        return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"

    def send_with_retry(message, tools=None, max_retry=3):
        for attempt in range(max_retry):
            try:
                if tools:
                    response = chat.send_message(message, tools=tools)
                else:
                    response = chat.send_message(message)
                yield response # 실제 응답 객체를 yield
                return # 응답을 yield한 후 제너레이터 종료
            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 15
                    yield send_event("status", {
                        "text": f"⏳ API 한도 도달 — {wait}초 대기 중..."
                    })
                    time.sleep(wait)
                else:
                    raise e
        raise Exception("최대 재시도 횟수 초과")

    try:
        # 시작 신호
        yield send_event("status", {"text": "🤖 AI 분석 시작..."})

        response = None
        for chunk in send_with_retry(question, tools=mcp_tools):
            if isinstance(chunk, str):
                yield chunk
            else:
                response = chunk

        if response is None:
            yield send_event("error", {"text": "응답을 받지 못했습니다."})
            return

        step = 1
        while step <= 6:
            fc_parts   = []
            text_parts = []

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") \
                        and part.function_call.name:
                    fc_parts.append(part.function_call)
                elif hasattr(part, "text") and part.text.strip():
                    text_parts.append(part.text.strip())

            # 최종 답변
            if not fc_parts:
                final = "\n".join(text_parts)
                yield send_event("answer", {"text": final})
                break

            # 도구 호출
            tool_response_parts = []
            for fc in fc_parts:
                tool_name = fc.name
                tool_args = dict(fc.args)

                # 도구 호출 시작 알림
                yield send_event("tool_start", {
                    "tool": tool_name,
                    "args": str(tool_args)[:80]
                })

                result = router.call(tool_name, tool_args)

                # 도구 완료 알림
                yield send_event("tool_done", {
                    "tool": tool_name,
                    "summary": _summarize_result(tool_name, result)
                })

                tool_response_parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": json.dumps(
                                result, ensure_ascii=False
                            )}
                        )
                    )
                )

            yield send_event("status", {"text": "📊 결과 분석 중..."})
            time.sleep(2)

            response = None
            for chunk in send_with_retry(
                tool_response_parts, tools=mcp_tools
            ):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    response = chunk

            step += 1

    except Exception as e:
        yield send_event("error", {"text": f"오류 발생: {str(e)}"})


def _summarize_result(tool_name: str, result: dict) -> str:
    """도구 결과 요약 (UI 표시용)"""
    if tool_name == "get_plant_state":
        rcs = result.get("rcs", {})
        safety = result.get("safety", {})
        return (
            f"RCS 압력: {rcs.get('pressure_psia')} psia | "
            f"SI: {safety.get('si_signal')} | "
            f"ECCS: {safety.get('eccs_status')}"
        )
    elif tool_name == "search_procedure":
        procs = result.get("procedures", [])
        ids = [p["id"] for p in procs]
        return f"검색된 절차서: {', '.join(ids)}"
    elif tool_name == "get_alarm_list":
        count = result.get("alarm_count", 0)
        return f"활성 알람 {count}개"
    elif tool_name == "log_action":
        return result.get("status", "기록 완료")
    return "완료"


# ── API 엔드포인트 ───────────────────────────────
@app.get("/")
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 스트리밍 채팅 엔드포인트"""
    return StreamingResponse(
        run_agent_stream(req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    """세션 초기화 (새 대화 시작)"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "세션 초기화 완료"}


@app.get("/plant/status")
async def get_plant_status():
    """현재 플랜트 상태 조회"""
    result = simulator.call_tool("get_plant_state", {})
    return result


if __name__ == "__main__":
    import uvicorn
    # reload=True를 사용하려면 app 객체 대신 "파일명:app" 문자열을 전달해야 합니다.
    uvicorn.run(
        "web_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )