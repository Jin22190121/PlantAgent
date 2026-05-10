"""Plant state for GOP/EOP/AOP scenarios.

Variables cover three scenarios:
  • GOP — heatup from cold shutdown (MODE 5→4)
  • EOP — reactor trip + safety injection (from MODE 1)
  • AOP — control room inaccessibility (from MODE 1, manual remote ops)
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
    # ── Time & MODE ─────────────────────────────
    sim_time_s: float = 0.0
    mode: int = 5
    scenario_id: str = "scenario_gop_heatup"
    active_doc_type: str = "GOP"   # "GOP"|"EOP"|"AOP"

    # ── Reactor / neutronics ────────────────────
    reactor_power_pct: float = 0.0
    K_eff: float = 0.97
    reactor_tripped: bool = False
    rods_bottom: bool = True
    neutron_flux_pct: float = 0.0

    # ── Reactor coolant system (1°) ─────────────
    T_RCS_avg_F: float = 155.0
    P_RCS_psig: float = 360.0
    PZR_level_pct: float = 90.0
    PZR_temp_F: float = 155.0
    seal_inj_flow_gpm: float = 8.0
    steam_bubble_formed: bool = False

    # Pumps & valves (1°)
    RCP_running: List[bool] = field(default_factory=lambda: [False] * 4)
    RHR_pump_on: bool = True
    PZR_heater_on: bool = False
    PZR_spray_open: bool = False
    charging_flow_gpm: float = 0.0
    letdown_flow_gpm: float = 0.0
    charging_suction_rwst: bool = False
    HCV_128_open: bool = True
    PCV_131_setpoint_psig: float = 360.0

    # ── Secondary (2°) ──────────────────────────
    SG_level_NR_pct: float = 100.0
    SG_level_target_pct: float = 100.0
    MSIV_open: bool = False
    turbine_tripped: bool = True
    P_steam_psig: float = 0.0
    atm_steam_dump_setpoint_psig: float = 1092.0

    # Feedwater
    mfw_pump_running: bool = False
    afw_mdafw_running: bool = False
    afw_tdafw_running: bool = False
    afw_flow_total_gpm: float = 0.0

    # ── Containment ─────────────────────────────
    P_cnmt_psig: float = 0.0
    cnmt_recirc_fans_on: bool = False
    ci_signal: bool = False
    cvi_signal: bool = False

    # ── Safeguards / SI ─────────────────────────
    si_signal: bool = False
    si_pumps_running: bool = False
    rhr_pumps_safety_running: bool = False
    si_flow_gpm: float = 0.0
    si_sequenced: bool = False

    # ── Power / electrical ──────────────────────
    bus_voltage_V: float = 4160.0
    sw_pumps_running: int = 2
    ccw_pump_running: bool = True

    # ── AOP-specific ────────────────────────────
    control_room_evacuated: bool = False

    # ── Derived / monitoring ────────────────────
    heatup_rate_F_per_hr: float = 0.0
    alarms: List[Dict] = field(default_factory=list)

    # ── Internal: rolling T history ─────────────
    _T_history: List[Tuple[float, float]] = field(default_factory=list)

    # ── snapshot (for SSE / LLM context) ────────
    def snapshot(self) -> dict:
        d = asdict(self)
        d.pop("_T_history", None)
        d["mode_name"] = MODE_NAMES.get(self.mode, "UNKNOWN")
        d["RCP_count"] = sum(self.RCP_running)
        return d


# ─── Scenario presets ───────────────────────────
SCENARIOS: Dict[str, Dict] = {
    "scenario_gop_heatup": {
        "label": "GOP — 콜드셧다운 → 핫셧다운 기동",
        "doc_type": "GOP",
        "description": ("Westinghouse §19.0 App19-1 §A. 핵연료 교체 후 재기동 시작 "
                        "상태. RHR로 붕괴열 제거 중."),
        "init": {
            "mode": 5, "scenario_id": "scenario_gop_heatup",
            "active_doc_type": "GOP",
            "reactor_power_pct": 0.0, "K_eff": 0.97,
            "reactor_tripped": False, "rods_bottom": True,
            "T_RCS_avg_F": 155.0, "P_RCS_psig": 360.0,
            "PZR_level_pct": 90.0, "PZR_temp_F": 155.0,
            "RCP_running": [False] * 4, "RHR_pump_on": True,
            "PZR_heater_on": False, "PZR_spray_open": False,
            "SG_level_NR_pct": 100.0, "SG_level_target_pct": 100.0,
            "MSIV_open": False, "turbine_tripped": True,
            "mfw_pump_running": False,
            "afw_mdafw_running": False, "afw_tdafw_running": False,
            "P_cnmt_psig": 0.0, "si_signal": False,
            "P_steam_psig": 0.0,
        },
    },

    "scenario_eop_reactor_trip_si": {
        "label": "EOP — 원자로 트립 + 안전주입 (LOCA)",
        "doc_type": "EOP",
        "description": ("Ginna E-0. 100%P 정상 운전 중 RCS 누설 발생 → "
                        "PZR 압력 1750 psig 미만 → SI 자동 작동 + 원자로 트립."),
        "init": {
            "mode": 1, "scenario_id": "scenario_eop_reactor_trip_si",
            "active_doc_type": "EOP",
            "reactor_power_pct": 100.0, "K_eff": 1.0,
            "reactor_tripped": False, "rods_bottom": False,
            "neutron_flux_pct": 100.0,
            "T_RCS_avg_F": 575.0, "P_RCS_psig": 1640.0,  # below SI setpoint
            "PZR_level_pct": 22.0, "PZR_temp_F": 575.0,
            "RCP_running": [True] * 4, "RHR_pump_on": False,
            "PZR_heater_on": True, "PZR_spray_open": False,
            "SG_level_NR_pct": 50.0, "MSIV_open": True,
            "turbine_tripped": False,
            "P_steam_psig": 1005.0,
            "mfw_pump_running": True,
            "afw_mdafw_running": False, "afw_tdafw_running": False,
            "P_cnmt_psig": 0.5,
            # SI not yet actuated — agent must do it
            "si_signal": False, "si_pumps_running": False,
            "steam_bubble_formed": True,
        },
    },

    "scenario_aop_cr_inaccessible": {
        "label": "AOP — 주제어실 접근 불능",
        "doc_type": "AOP",
        "description": ("Point Beach AOP-10. 100%P 정상 운전 중 주제어실 화재로 "
                        "철수 결정. 양 호기 정지 + 현장 운전 전환."),
        "init": {
            "mode": 1, "scenario_id": "scenario_aop_cr_inaccessible",
            "active_doc_type": "AOP",
            "reactor_power_pct": 100.0, "K_eff": 1.0,
            "reactor_tripped": False, "rods_bottom": False,
            "neutron_flux_pct": 100.0,
            "T_RCS_avg_F": 575.0, "P_RCS_psig": 2235.0,
            "PZR_level_pct": 60.0, "PZR_temp_F": 653.0,
            "RCP_running": [True] * 4, "RHR_pump_on": False,
            "PZR_heater_on": True, "PZR_spray_open": False,
            "SG_level_NR_pct": 50.0, "SG_level_target_pct": 50.0,
            "MSIV_open": True, "turbine_tripped": False,
            "P_steam_psig": 1092.0, "atm_steam_dump_setpoint_psig": 1092.0,
            "mfw_pump_running": True,
            "afw_mdafw_running": False, "afw_tdafw_running": False,
            "P_cnmt_psig": 0.0, "si_signal": False,
            "control_room_evacuated": False,
            "steam_bubble_formed": True,
        },
    },
}
