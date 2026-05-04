import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
import time
import json

from npp_agent.mcp_servers.simulator import SimulatorMCP
from npp_agent.mcp_servers.procedure import ProcedureMCP
from npp_agent.mcp_servers.training  import TrainingMCP
from npp_agent.mcp_servers.router    import MCPRouter
from google.generativeai.types import FunctionDeclaration, Tool

# ── LLM 초기화 ────────────────────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ── MCP 서버 및 라우터 초기화 ─────────────────────
simulator = SimulatorMCP()
procedure = ProcedureMCP()
training  = TrainingMCP()
router    = MCPRouter([simulator, procedure, training])

# ── 도구 정의 ─────────────────────────────────────
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
                "session_id": {"type": "string"},
                "action":     {"type": "string"},
                "procedure_ref": {"type": "string"}
            },
            "required": ["session_id", "action"]
        }
    ),
])


# ── Agent 실행 함수 ───────────────────────────────
def run_agent(question: str, chat):

    def send_with_retry(message, tools=None, max_retry=3):
        for attempt in range(max_retry):
            try:
                if tools:
                    return chat.send_message(message, tools=tools)
                return chat.send_message(message)
            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 15
                    print(f"  ⏳ {wait}초 대기 후 재시도...")
                    time.sleep(wait)
                else:
                    raise e
        raise Exception("최대 재시도 횟수 초과")

    response = send_with_retry(question, tools=mcp_tools)

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

        if not fc_parts:
            return "\n".join(text_parts) or "응답 없음"

        tool_response_parts = []
        for fc in fc_parts:
            result = router.call(fc.name, dict(fc.args))
            tool_response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": json.dumps(
                            result, ensure_ascii=False
                        )}
                    )
                )
            )

        time.sleep(2)
        response = send_with_retry(
            tool_response_parts, tools=mcp_tools
        )
        step += 1

    return "최대 단계 도달"


# ── 메인 실행 ─────────────────────────────────────
def main():
    print("=" * 55)
    print("🏭 NPP AI Agent 시작")
    print("=" * 55)

    system_prompt = """당신은 원자력발전소 교육훈련 보조 AI입니다.
1. get_plant_state 로 플랜트 상태를 먼저 확인하세요.
2. search_procedure 로 관련 절차서를 검색하세요.
3. 반드시 절차서 번호를 인용하여 답변하세요."""

    chat = model.start_chat(history=[
        {"role": "user",  "parts": [system_prompt]},
        {"role": "model", "parts": ["네, 이해했습니다."]}
    ])

    print("\n🚨 [상황 발생]")
    print("RCS 압력 저하, SI 신호 발생, 원자로 자동 정지\n")

    answer = run_agent(
        "현재 상황을 확인하고 즉시 조치사항을 알려주세요.",
        chat
    )

    print("\n" + "=" * 55)
    print("📋 AI 운전 가이드")
    print("=" * 55)
    print(answer)


if __name__ == "__main__":
    main()