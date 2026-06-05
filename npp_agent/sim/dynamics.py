"""Simplified ODE dynamics for GOP/EOP/AOP scenarios.

Tuning targets:
  • GOP — 4 RCPs, no RHR → ~50°F/hr heatup
  • EOP — reactor trip drops power exponentially; SI raises P_RCS
  • AOP — manual reactor trip + steam dump @1005 psig stabilizes plant
"""
from .state import PlantState


# Tuning constants
K_RCS_T = 1.0 / 1370.0
K_PZR_T = 1.0 / 18.0
K_SOLID_P = 0.6
PZR_VOL_GAL = 1800.0
HEATER_MW = 1.4
RCP_HEAT_MW_EACH = 4.5
RHR_REMOVAL_MW = 22.0


def saturation_pressure_psig(T_F: float) -> float:
    if T_F < 212:   return 0.0
    if T_F < 300:   return (T_F - 212) * 1.6
    if T_F < 400:   return 140 + (T_F - 300) * 2.5
    if T_F < 500:   return 390 + (T_F - 400) * 6.0
    if T_F < 600:   return 990 + (T_F - 500) * 12.0
    return 2200.0


def _decay_heat_mw(s: PlantState) -> float:
    """Approximate decay heat after a trip.
    Power drops fast (~6% at 1s, ~3% at 1min, ~1% at 1hr after shutdown).
    Pre-trip: full reactor power (rough).
    """
    if s.mode == 5:
        return 1.0  # cold shutdown decay heat residual
    if s.reactor_tripped:
        # crude: 6% * (t/10)^-0.2, capped
        t = max(1.0, s.sim_time_s)
        decay = 0.07 * t ** -0.2
        return max(0.5, min(0.07, decay) * 1700.0)  # 1700 MWth ~ Ginna
    # Fission heat
    return 1700.0 * (s.reactor_power_pct / 100.0)


def step(s: PlantState, dt: float) -> None:
    """Advance state by `dt` simulated seconds (in-place)."""
    # ── Reactor: handle trip dynamics ─────────────
    if s.reactor_tripped and s.reactor_power_pct > 0.5:
        # exp decay toward decay heat fraction (~6%) in ~5 s, then slowly fade
        s.reactor_power_pct = max(0.0, s.reactor_power_pct - dt * 20.0)
        s.neutron_flux_pct = s.reactor_power_pct
        if s.reactor_power_pct < 5.0:
            s.rods_bottom = True

    # ── 1차 계통 열수지 ───────────────────────────
    Q_rcp = sum(s.RCP_running) * RCP_HEAT_MW_EACH
    Q_decay = _decay_heat_mw(s)

    # RHR: thermostat at 155°F when on (cold shutdown context)
    Q_rhr = 0.0
    if s.RHR_pump_on:
        deficit = s.T_RCS_avg_F - 155.0
        Q_rhr = max(0.0, min(RHR_REMOVAL_MW, 0.6 * deficit + 1.0))

    # SG heat removal proportional to MFW/AFW flow + steam dump
    Q_sg_removed = 0.0
    if s.MSIV_open and s.mfw_pump_running:
        Q_sg_removed = Q_decay + 1700.0 * (s.reactor_power_pct / 100.0) * 0.99
    elif s.afw_tdafw_running or s.afw_mdafw_running:
        Q_sg_removed = max(Q_decay - 1.0, 0.0)
        # AFW heat removal scaling
        Q_sg_removed += s.afw_flow_total_gpm * 0.03

    Q_letdown = 0.003 * s.letdown_flow_gpm
    Q_si_inj = 0.0
    if s.si_pumps_running and s.P_RCS_psig < 1400:
        Q_si_inj = 5.0  # cold injection cooling

    dT_rcs = K_RCS_T * (Q_rcp + Q_decay - Q_rhr - Q_letdown
                        - Q_sg_removed - Q_si_inj)
    s.T_RCS_avg_F = max(50.0, s.T_RCS_avg_F + dT_rcs * dt)

    # ── 가압기 ────────────────────────────────────
    Q_h = HEATER_MW if s.PZR_heater_on else 0.0
    target_pzr_T = 432.0 if not s.steam_bubble_formed else (s.PZR_temp_F if s.mode <= 4 else 653.0)
    overshoot = max(0.0, s.PZR_temp_F - target_pzr_T)
    Q_h *= max(0.0, 1.0 - overshoot / 5.0)
    Q_spray = 0.0
    if s.PZR_spray_open and s.PZR_temp_F > s.T_RCS_avg_F:
        Q_spray = 0.02 * (s.PZR_temp_F - s.T_RCS_avg_F)
    Q_loss = 0.05
    dT_pzr = K_PZR_T * (Q_h - Q_spray - Q_loss)
    if s.mode >= 3:
        # in hot states, PZR follows RCS more directly
        dT_pzr += 0.01 * (s.T_RCS_avg_F - s.PZR_temp_F)
    s.PZR_temp_F = max(s.T_RCS_avg_F, s.PZR_temp_F + dT_pzr * dt)

    # ── PZR level + flows ────────────────────────
    net_flow = s.charging_flow_gpm - s.letdown_flow_gpm
    if s.si_pumps_running and s.P_RCS_psig < 1500:
        net_flow += 200.0  # SI inflow
    s.PZR_level_pct += (net_flow / PZR_VOL_GAL) * (dt / 60.0) * 100.0
    s.PZR_level_pct = max(0.0, min(100.0, s.PZR_level_pct))

    # AFW total flow
    s.afw_flow_total_gpm = (
        (250.0 if s.afw_tdafw_running else 0.0)
        + (180.0 if s.afw_mdafw_running else 0.0)
    )

    # Steam-bubble formation
    P_sat = saturation_pressure_psig(s.PZR_temp_F)
    if (not s.steam_bubble_formed
            and s.PZR_temp_F >= 425.0
            and s.PZR_level_pct < 60.0):
        s.steam_bubble_formed = True
        s.alarms.append({"level": "info",
                         "text": "Steam bubble formed in pressurizer",
                         "t": s.sim_time_s})

    # ── RCS 압력 ─────────────────────────────────
    if s.steam_bubble_formed:
        target_p = max(320.0, P_sat)
        # SI raises pressure if injecting at low P
        if s.si_pumps_running and s.P_RCS_psig < 1400:
            target_p = max(target_p, 1300.0)
        s.P_RCS_psig += (target_p - s.P_RCS_psig) * min(1.0, dt / 30.0)
    else:
        target_p = s.PCV_131_setpoint_psig
        s.P_RCS_psig += (target_p - s.P_RCS_psig) * min(1.0, dt / 60.0)
        s.P_RCS_psig += K_SOLID_P * net_flow * (dt / 60.0)
        s.P_RCS_psig = max(280.0, min(450.0, s.P_RCS_psig))

    # ── Steam pressure ───────────────────────────
    if s.MSIV_open:
        # follow Tavg saturation roughly when at power
        target_steam = max(0.0, s.atm_steam_dump_setpoint_psig if s.turbine_tripped
                           else 1092.0)
        s.P_steam_psig += (target_steam - s.P_steam_psig) * min(1.0, dt / 60.0)
    else:
        # MSIV closed: pressure rises from decay heat unless ARV/dump
        s.P_steam_psig += 0.5 * dt
        s.P_steam_psig = min(1200.0, s.P_steam_psig)

    # ── SG level: lag toward target via MFW/AFW ──
    diff = s.SG_level_target_pct - s.SG_level_NR_pct
    rate = 0.02 if (s.mfw_pump_running or s.afw_flow_total_gpm > 50) else 0.005
    s.SG_level_NR_pct += rate * diff * dt
    s.SG_level_NR_pct = max(0.0, min(100.0, s.SG_level_NR_pct))

    # ── time + heatup rate ──────────────────────
    s.sim_time_s += dt
    s._T_history.append((s.sim_time_s, s.T_RCS_avg_F))
    s._T_history = [(t, T) for (t, T) in s._T_history
                    if s.sim_time_s - t <= 3600.0]
    if len(s._T_history) >= 2:
        t0, T0 = s._T_history[0]
        if (s.sim_time_s - t0) > 60:
            s.heatup_rate_F_per_hr = (s.T_RCS_avg_F - T0) / ((s.sim_time_s - t0) / 3600.0)

    # ── Auto MODE transitions ───────────────────
    if s.mode == 5 and s.T_RCS_avg_F >= 200.0:
        s.mode = 4
        s.alarms.append({"level": "success",
                         "text": "MODE 5 → MODE 4 (HOT SHUTDOWN) 진입",
                         "t": s.sim_time_s})


def evaluate_alarms(s: PlantState) -> None:
    existing = {a["text"] for a in s.alarms}

    def add(level: str, text: str):
        if text not in existing:
            s.alarms.append({"level": level, "text": text, "t": s.sim_time_s})
            existing.add(text)

    if s.mode == 5 and s.heatup_rate_F_per_hr > 100.0:
        add("danger", f"가열률 {s.heatup_rate_F_per_hr:.0f}°F/hr > 100°F/hr 제한")
    if any(s.RCP_running) and s.seal_inj_flow_gpm <= 0.0:
        add("danger", "RCP 운전 중 seal injection 상실")
    if (s.PZR_spray_open and s.PZR_temp_F - s.T_RCS_avg_F > 320.0):
        add("danger", "ΔT(PZR-spray) 320°F 제한 초과")

    # EOP-specific: SI auto setpoints
    if not s.si_signal:
        if (s.P_RCS_psig < 1750.0 and s.mode <= 2):
            add("danger", f"PZR 압력 {s.P_RCS_psig:.0f} psig < 1750 psig — SI 작동 신호")
        if (s.P_steam_psig < 514.0 and s.mode <= 2):
            add("danger", f"증기관 압력 {s.P_steam_psig:.0f} psig < 514 psig — SI 신호")
        if s.P_cnmt_psig > 4.0:
            add("danger", f"격납건물 압력 {s.P_cnmt_psig:.1f} psig > 4 — SI/CI 신호")

    if s.mode >= 3 and s.SG_level_NR_pct < 5.0:
        add("danger", "SG 협역 수위 < 5% — 보조급수 필수")

    if len(s.alarms) > 16:
        s.alarms[:] = s.alarms[-16:]
