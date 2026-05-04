import datetime


class TrainingMCP:
    """
    훈련 기록 및 평가 MCP 서버
    훈련생 행동 로그, 점수, 세션 관리
    """
    SERVER_NAME = "training-mcp"

    def __init__(self):
        self.sessions = {}

    def list_tools(self):
        return [
            {
                "name": "log_action",
                "description": "훈련생 행동 기록"
            },
            {
                "name": "get_session_summary",
                "description": "훈련 세션 요약 반환"
            }
        ]

    def call_tool(self, tool_name: str, args: dict):
        if tool_name == "log_action":
            return self._log_action(
                args.get("session_id", ""),
                args.get("action", ""),
                args.get("procedure_ref", "")
            )
        elif tool_name == "get_session_summary":
            return self._get_session_summary(
                args.get("session_id", "")
            )
        return {"error": f"알 수 없는 도구: {tool_name}"}

    def _log_action(self, session_id, action, procedure_ref):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "procedure_ref": procedure_ref
        })
        return {
            "status": "기록 완료",
            "total_actions": len(self.sessions[session_id])
        }

    def _get_session_summary(self, session_id):
        actions = self.sessions.get(session_id, [])
        return {
            "session_id": session_id,
            "total_actions": len(actions),
            "actions": actions
        }  