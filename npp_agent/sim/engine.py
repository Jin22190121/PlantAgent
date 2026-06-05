"""Async simulation engine with multi-scenario reset.

`reset(scenario_id)` swaps in a preset initial condition matching the
selected procedure document.
"""
import asyncio
import time
from typing import Optional

from .state import PlantState, SCENARIOS
from .dynamics import step, evaluate_alarms


class SimulationEngine:
    def __init__(self, sim_speed: float = 60.0):
        self.state = PlantState()
        self.sim_speed = sim_speed
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._paused = False

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    async def _run(self):
        last = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.2)
                now = time.monotonic()
                dt_real = now - last
                last = now
                if self._paused: continue
                dt_sim = dt_real * self.sim_speed
                async with self._lock:
                    step(self.state, dt_sim)
                    evaluate_alarms(self.state)
        except asyncio.CancelledError:
            return

    def get_state(self) -> dict:
        return self.state.snapshot()

    # ── operator actions (mutate state) ──
    def set_pzr_heater(self, on: bool) -> dict:
        self.state.PZR_heater_on = bool(on)
        return {"ok": True, "PZR_heater_on": self.state.PZR_heater_on}

    def set_pzr_spray(self, open: bool) -> dict:
        self.state.PZR_spray_open = bool(open)
        return {"ok": True, "PZR_spray_open": self.state.PZR_spray_open}

    def start_rcp(self, pump_id: int) -> dict:
        if not (1 <= pump_id <= 4):
            return {"ok": False, "error": "pump_id must be 1..4"}
        if self.state.P_RCS_psig < 320:
            return {"ok": False,
                    "error": f"P_RCS {self.state.P_RCS_psig:.1f} psig < 320 psig (RCP NPSH 부족)"}
        if self.state.seal_inj_flow_gpm <= 0:
            return {"ok": False, "error": "Seal injection 상실 - RCP 기동 금지"}
        self.state.RCP_running[pump_id - 1] = True
        return {"ok": True, "RCP_running": list(self.state.RCP_running)}

    def stop_rcp(self, pump_id: int) -> dict:
        if not (1 <= pump_id <= 4):
            return {"ok": False, "error": "pump_id must be 1..4"}
        self.state.RCP_running[pump_id - 1] = False
        return {"ok": True, "RCP_running": list(self.state.RCP_running)}

    def set_charging_flow(self, gpm: float) -> dict:
        gpm = max(0.0, float(gpm))
        self.state.charging_flow_gpm = gpm
        return {"ok": True, "charging_flow_gpm": gpm}

    def set_letdown_flow(self, gpm: float) -> dict:
        gpm = max(0.0, float(gpm))
        if gpm > 120.0:
            return {"ok": False, "error": "letdown 최대 120 gpm 초과"}
        self.state.letdown_flow_gpm = gpm
        return {"ok": True, "letdown_flow_gpm": gpm}

    def set_rhr_pump(self, on: bool) -> dict:
        self.state.RHR_pump_on = bool(on)
        return {"ok": True, "RHR_pump_on": self.state.RHR_pump_on}

    def open_msiv(self) -> dict:
        self.state.MSIV_open = True
        return {"ok": True, "MSIV_open": True}

    def close_msiv(self) -> dict:
        self.state.MSIV_open = False
        return {"ok": True, "MSIV_open": False}

    def set_sg_level_target(self, pct: float) -> dict:
        pct = max(0.0, min(100.0, float(pct)))
        self.state.SG_level_target_pct = pct
        return {"ok": True, "SG_level_target_pct": pct}

    def set_sim_speed(self, speed: float) -> dict:
        self.sim_speed = max(1.0, min(600.0, float(speed)))
        return {"ok": True, "sim_speed": self.sim_speed}

    # ── EOP/AOP-specific actions ──
    def manual_reactor_trip(self) -> dict:
        self.state.reactor_tripped = True
        self.state.turbine_tripped = True
        return {"ok": True, "reactor_tripped": True, "turbine_tripped": True}

    def trip_turbine(self) -> dict:
        self.state.turbine_tripped = True
        self.state.mfw_pump_running = False
        return {"ok": True, "turbine_tripped": True}

    def actuate_si(self) -> dict:
        self.state.si_signal = True
        self.state.si_pumps_running = True
        self.state.rhr_pumps_safety_running = True
        self.state.ci_signal = True
        self.state.cvi_signal = True
        self.state.si_sequenced = True
        return {"ok": True, "si_signal": True, "si_pumps_running": True,
                "ci_signal": True}

    def trip_mfw(self) -> dict:
        self.state.mfw_pump_running = False
        return {"ok": True, "mfw_pump_running": False}

    def start_afw(self, mdafw: bool = True, tdafw: bool = True) -> dict:
        self.state.afw_mdafw_running = bool(mdafw)
        self.state.afw_tdafw_running = bool(tdafw)
        return {"ok": True,
                "afw_mdafw_running": self.state.afw_mdafw_running,
                "afw_tdafw_running": self.state.afw_tdafw_running}

    def start_tdafw(self) -> dict:
        return self.start_afw(mdafw=False, tdafw=True)

    def align_charging_to_rwst(self) -> dict:
        self.state.charging_suction_rwst = True
        self.state.charging_flow_gpm = 60.0
        return {"ok": True, "charging_suction_rwst": True}

    def set_atm_dump(self, psig: float) -> dict:
        psig = max(800.0, min(1200.0, float(psig)))
        self.state.atm_steam_dump_setpoint_psig = psig
        return {"ok": True, "atm_steam_dump_setpoint_psig": psig}

    def evacuate_control_room(self) -> dict:
        self.state.control_room_evacuated = True
        return {"ok": True, "control_room_evacuated": True,
                "note": "현장 운전으로 전환됨"}

    # ── reset ──
    def reset(self, scenario: str = "scenario_gop_heatup") -> dict:
        if scenario not in SCENARIOS:
            scenario = "scenario_gop_heatup"
        preset = SCENARIOS[scenario]["init"]
        new_state = PlantState()
        for k, v in preset.items():
            if hasattr(new_state, k):
                setattr(new_state, k, list(v) if isinstance(v, list) else v)
        self.state = new_state
        return {"ok": True, "scenario": scenario,
                "label": SCENARIOS[scenario]["label"],
                "doc_type": SCENARIOS[scenario]["doc_type"]}

    def list_scenarios(self) -> dict:
        return {"scenarios": [
            {"id": k, "label": v["label"], "doc_type": v["doc_type"],
             "description": v["description"]}
            for k, v in SCENARIOS.items()
        ]}
