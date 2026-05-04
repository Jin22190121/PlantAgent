import json
import datetime


class SimulatorMCP:
    """
    훈련용 시뮬레이터 연동 MCP 서버
    실제 환경: OPC-UA로 시뮬레이터에서 데이터 수신
    현재: 샘플 데이터 반환 (LOCA 시나리오)
    """
    SERVER_NAME = "simulator-mcp"

    def list_tools(self):
        return [
            {
                "name": "get_plant_state",
                "description": "현재 플랜트 전체 파라미터 반환"
            },
            {
                "name": "get_alarm_list",
                "description": "현재 활성 알람 목록 반환"
            }
        ]

    def call_tool(self, tool_name: str, args: dict):
        if tool_name == "get_plant_state":
            return self._get_plant_state()
        elif tool_name == "get_alarm_list":
            return self._get_alarm_list()
        return {"error": f"알 수 없는 도구: {tool_name}"}

    def _get_plant_state(self):
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "reactor": {
                "power_percent": 0.0,
                "coolant_temp_in_C": 288.1,
                "coolant_temp_out_C": 312.4,
                "flow_percent": 71.2
            },
            "rcs": {
                "pressure_psia": 1580.2,
                "pzr_level_percent": 28.4,
                "pressure_alarm": True
            },
            "safety": {
                "si_signal": True,
                "hi_signal": True,
                "reactor_trip": True,
                "eccs_status": "RUNNING"
            },
            "electrical": {
                "offsite_power": True,
                "edg1_status": "STANDBY"
            }
        }

    def _get_alarm_list(self):
        return {
            "alarm_count": 3,
            "alarms": [
                {
                    "id": "ALM-001",
                    "name": "RCS 압력 저하",
                    "priority": 1,
                    "time": "10:23:45"
                },
                {
                    "id": "ALM-002",
                    "name": "SI 신호 발생",
                    "priority": 1,
                    "time": "10:23:46"
                },
                {
                    "id": "ALM-003",
                    "name": "원자로 자동 정지",
                    "priority": 2,
                    "time": "10:23:47"
                }
            ]
        }