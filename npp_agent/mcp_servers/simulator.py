"""Simulator MCP — wraps the live SimulationEngine.

Tools exposed to the LLM:
  • get_plant_state — current state snapshot
  • get_alarm_list  — active alarms
  • set_pzr_heater, set_pzr_spray, set_rhr_pump
  • start_rcp, stop_rcp
  • set_charging_flow, set_letdown_flow
  • set_sg_level_target, open_msiv
  • advance_time (for batch progression in non-async contexts)
"""
from typing import Any, Dict

from npp_agent.sim import SimulationEngine
from npp_agent.sim.dynamics import step as _phys_step, evaluate_alarms


class SimulatorMCP:
    SERVER_NAME = "simulator-mcp"

    def __init__(self, engine: SimulationEngine | None = None):
        # Engine is shared with the FastAPI app; create a default if none given.
        self.engine = engine or SimulationEngine()

    # ── tool catalog ─────────────────────────────
    def list_tools(self):
        return [
            {"name": "get_plant_state",   "description": "현재 플랜트 전체 파라미터 반환"},
            {"name": "get_alarm_list",    "description": "현재 활성 알람 목록 반환"},
            {"name": "set_pzr_heater",    "description": "가압기 히터 on/off"},
            {"name": "set_pzr_spray",     "description": "가압기 살수 밸브 open/close"},
            {"name": "set_rhr_pump",      "description": "RHR 펌프 on/off"},
            {"name": "start_rcp",         "description": "RCP(1~4) 기동"},
            {"name": "stop_rcp",          "description": "RCP(1~4) 정지"},
            {"name": "set_charging_flow", "description": "충전 유량(gpm) 설정"},
            {"name": "set_letdown_flow",  "description": "방출 유량(gpm) 설정"},
            {"name": "set_sg_level_target","description":"SG 협역 수위 목표(%) 설정"},
            {"name": "open_msiv",         "description": "MSIV 개방"},
            {"name": "advance_time",      "description": "(시뮬 단위 초) 시간 진행"},
            {"name": "reset_simulator",   "description": "시뮬레이터 초기 상태 리셋"},
        ]

    # ── dispatcher ───────────────────────────────
    def call_tool(self, tool_name: str, args: Dict[str, Any]):
        e = self.engine
        try:
            if tool_name == "get_plant_state":
                return e.get_state()
            if tool_name == "get_alarm_list":
                snap = e.get_state()
                return {"alarm_count": len(snap["alarms"]),
                        "alarms": snap["alarms"]}
            if tool_name == "set_pzr_heater":
                return e.set_pzr_heater(bool(args.get("on", False)))
            if tool_name == "set_pzr_spray":
                return e.set_pzr_spray(bool(args.get("open", False)))
            if tool_name == "set_rhr_pump":
                return e.set_rhr_pump(bool(args.get("on", False)))
            if tool_name == "start_rcp":
                return e.start_rcp(int(args.get("pump_id", 0)))
            if tool_name == "stop_rcp":
                return e.stop_rcp(int(args.get("pump_id", 0)))
            if tool_name == "set_charging_flow":
                return e.set_charging_flow(float(args.get("gpm", 0)))
            if tool_name == "set_letdown_flow":
                return e.set_letdown_flow(float(args.get("gpm", 0)))
            if tool_name == "set_sg_level_target":
                return e.set_sg_level_target(float(args.get("pct", 33)))
            if tool_name == "open_msiv":
                return e.open_msiv()
            if tool_name == "advance_time":
                seconds = float(args.get("seconds", 60))
                # Synchronous advance for tools that need to "skip ahead"
                _phys_step(e.state, seconds)
                evaluate_alarms(e.state)
                return {"ok": True, "advanced_seconds": seconds,
                        "sim_time_s": e.state.sim_time_s}
            if tool_name == "reset_simulator":
                return e.reset(args.get("scenario", "cold_shutdown_initial"))
            return {"error": f"알 수 없는 도구: {tool_name}"}
        except Exception as ex:  # pragma: no cover (defensive)
            return {"error": f"{type(ex).__name__}: {ex}"}
