"""Async simulation engine.

Runs a background asyncio task that integrates the plant dynamics in
real-time-with-acceleration. Provides thread-safe operator action methods
and a snapshot API.
"""
import asyncio
import time
from typing import Optional

from .state import PlantState
from .dynamics import step, evaluate_alarms


class SimulationEngine:
    def __init__(self, sim_speed: float = 60.0):
        self.state = PlantState()
        # 1 real second = `sim_speed` simulated seconds
        self.sim_speed = sim_speed
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._paused = False

    # ── lifecycle ────────────────────────────────
    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
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
                if self._paused:
                    continue
                dt_sim = dt_real * self.sim_speed
                async with self._lock:
                    step(self.state, dt_sim)
                    evaluate_alarms(self.state)
        except asyncio.CancelledError:
            return

    # ── snapshot ─────────────────────────────────
    def get_state(self) -> dict:
        return self.state.snapshot()

    # ── operator actions (synchronous: only mutate state) ─
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

    def set_sg_level_target(self, pct: float) -> dict:
        pct = max(0.0, min(100.0, float(pct)))
        self.state.SG_level_target_pct = pct
        return {"ok": True, "SG_level_target_pct": pct}

    def set_sim_speed(self, speed: float) -> dict:
        self.sim_speed = max(1.0, min(600.0, float(speed)))
        return {"ok": True, "sim_speed": self.sim_speed}

    def reset(self, scenario: str = "cold_shutdown_initial") -> dict:
        self.state = PlantState()
        return {"ok": True, "scenario": scenario}
