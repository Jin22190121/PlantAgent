"""Plant state for MODE 5→4 PoC.

Variables and initial values follow the Appendix 19-1 §I (Initial Conditions)
of the Westinghouse Technology Systems Manual Section 19.0.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple


MODE_NAMES = {
    5: "COLD SHUTDOWN",
    4: "HOT SHUTDOWN",
    3: "HOT STANDBY",
    2: "STARTUP",
    1: "POWER OPERATION",
}


@dataclass
class PlantState:
    # Time
    sim_time_s: float = 0.0
    mode: int = 5

    # Reactor coolant system (1차 계통)
    T_RCS_avg_F: float = 155.0          # MODE 5 initial: 150~160°F
    P_RCS_psig: float = 360.0           # solid plant: 320~400 psig
    PZR_level_pct: float = 90.0         # solid initial
    PZR_temp_F: float = 155.0
    seal_inj_flow_gpm: float = 8.0      # supplied from CVCS
    steam_bubble_formed: bool = False

    # Pumps & valves
    RCP_running: List[bool] = field(default_factory=lambda: [False] * 4)
    RHR_pump_on: bool = True
    PZR_heater_on: bool = False
    PZR_spray_open: bool = False
    charging_flow_gpm: float = 0.0
    letdown_flow_gpm: float = 0.0
    HCV_128_open: bool = True
    PCV_131_setpoint_psig: float = 360.0

    # Secondary
    SG_level_NR_pct: float = 100.0      # wet layup at start
    SG_level_target_pct: float = 100.0
    MSIV_open: bool = False

    # Reactor neutronics (subcritical for 5→4)
    K_eff: float = 0.97

    # Derived / monitoring
    heatup_rate_F_per_hr: float = 0.0
    alarms: List[Dict] = field(default_factory=list)

    # Internal: rolling window for heatup rate (sim_time_s, T_RCS_avg_F)
    _T_history: List[Tuple[float, float]] = field(default_factory=list)

    def snapshot(self) -> dict:
        d = asdict(self)
        d.pop("_T_history", None)
        d["mode_name"] = MODE_NAMES.get(self.mode, "UNKNOWN")
        d["RCP_count"] = sum(self.RCP_running)
        return d
