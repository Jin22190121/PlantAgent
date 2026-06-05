"""Simulator MCP — wraps the live SimulationEngine.

Tools cover GOP heatup, EOP reactor-trip/SI, and AOP control-room evac.
"""
from typing import Any, Dict

from npp_agent.sim import SimulationEngine
from npp_agent.sim.dynamics import step as _phys_step, evaluate_alarms


class SimulatorMCP:
    SERVER_NAME = "simulator-mcp"

    def __init__(self, engine: SimulationEngine | None = None):
        self.engine = engine or SimulationEngine()

    def list_tools(self):
        return [
            # READ
            {"name": "get_plant_state",   "description": "현재 플랜트 전체 파라미터"},
            {"name": "get_alarm_list",    "description": "현재 활성 알람"},
            {"name": "list_scenarios",    "description": "사용 가능한 시나리오 목록"},
            # GOP-style
            {"name": "set_pzr_heater",    "description": "가압기 히터 on/off (E)"},
            {"name": "set_pzr_spray",     "description": "가압기 살수 밸브 (E)"},
            {"name": "set_rhr_pump",      "description": "RHR 펌프 on/off (E)"},
            {"name": "start_rcp",         "description": "RCP(1~4) 기동 (E)"},
            {"name": "stop_rcp",          "description": "RCP(1~4) 정지 (E)"},
            {"name": "set_charging_flow", "description": "충전 유량(gpm) (E)"},
            {"name": "set_letdown_flow",  "description": "방출 유량(gpm) (E)"},
            {"name": "set_sg_level_target","description":"SG 협역 수위 목표(%) (E)"},
            {"name": "open_msiv",         "description": "MSIV 개방 (E)"},
            {"name": "close_msiv",        "description": "MSIV 차단 (E)"},
            {"name": "advance_time",      "description": "시뮬 시간 진행(초) (E)"},
            {"name": "reset_simulator",   "description": "시나리오 초기 상태로 리셋 (E)"},
            # EOP-specific
            {"name": "manual_reactor_trip","description":"수동 원자로 트립 (E)"},
            {"name": "trip_turbine",       "description":"터빈 트립 (E)"},
            {"name": "actuate_si",         "description":"안전주입 SI 수동 작동 (E)"},
            {"name": "trip_mfw",           "description":"주급수펌프 정지 (E)"},
            {"name": "start_afw",          "description":"보조급수 펌프 기동(MDAFW/TDAFW) (E)"},
            {"name": "start_tdafw",        "description":"터빈구동 보조급수 (E)"},
            # AOP-specific
            {"name": "align_charging_to_rwst","description":"충전펌프 흡입을 RWST로 정렬 (E)"},
            {"name": "set_atm_dump",          "description":"대기증기방출 setpoint(psig) (E)"},
            {"name": "evacuate_control_room", "description":"주제어실 철수 선언 (E)"},
        ]

    def call_tool(self, tool_name: str, args: Dict[str, Any]):
        e = self.engine
        try:
            t = tool_name
            if t == "get_plant_state":  return e.get_state()
            if t == "get_alarm_list":
                snap = e.get_state()
                return {"alarm_count": len(snap["alarms"]), "alarms": snap["alarms"]}
            if t == "list_scenarios":   return e.list_scenarios()
            if t == "set_pzr_heater":   return e.set_pzr_heater(bool(args.get("on", False)))
            if t == "set_pzr_spray":    return e.set_pzr_spray(bool(args.get("open", False)))
            if t == "set_rhr_pump":     return e.set_rhr_pump(bool(args.get("on", False)))
            if t == "start_rcp":        return e.start_rcp(int(args.get("pump_id", 0)))
            if t == "stop_rcp":         return e.stop_rcp(int(args.get("pump_id", 0)))
            if t == "set_charging_flow":return e.set_charging_flow(float(args.get("gpm", 0)))
            if t == "set_letdown_flow": return e.set_letdown_flow(float(args.get("gpm", 0)))
            if t == "set_sg_level_target": return e.set_sg_level_target(float(args.get("pct", 33)))
            if t == "open_msiv":        return e.open_msiv()
            if t == "close_msiv":       return e.close_msiv()
            if t == "manual_reactor_trip": return e.manual_reactor_trip()
            if t == "trip_turbine":     return e.trip_turbine()
            if t == "actuate_si":       return e.actuate_si()
            if t == "trip_mfw":         return e.trip_mfw()
            if t == "start_afw":
                return e.start_afw(mdafw=bool(args.get("mdafw", True)),
                                   tdafw=bool(args.get("tdafw", True)))
            if t == "start_tdafw":      return e.start_tdafw()
            if t == "align_charging_to_rwst": return e.align_charging_to_rwst()
            if t == "set_atm_dump":
                return e.set_atm_dump(float(args.get("psig", 1005)))
            if t == "evacuate_control_room": return e.evacuate_control_room()
            if t == "advance_time":
                seconds = float(args.get("seconds", 60))
                _phys_step(e.state, seconds)
                evaluate_alarms(e.state)
                return {"ok": True, "advanced_seconds": seconds,
                        "sim_time_s": e.state.sim_time_s}
            if t == "reset_simulator":
                return e.reset(args.get("scenario", "scenario_gop_heatup"))
            return {"error": f"알 수 없는 도구: {tool_name}"}
        except Exception as ex:  # pragma: no cover
            return {"error": f"{type(ex).__name__}: {ex}"}
