"""Simplified ODE dynamics tuned to match the procedure's expected behaviour.

The model is NOT a high-fidelity thermal-hydraulic simulator. Constants are
chosen so that:
  • PZR heater alone raises PZR_temp from 155°F to 428°F in ~25 sim min
  • 4 RCPs running with no RHR raises T_RCS at ~50°F/hr (matches §19.2.2)
  • Steam bubble forms when PZR reaches saturation (~428°F at 320 psig)
  • Solid-plant pressure responds linearly to charging - letdown
"""
from .state import PlantState


# Tuning constants (chosen for procedure-realistic timing, not physical accuracy)
# Target: 4 RCPs + no RHR → ~50°F/hr heatup (per §19.2.2)
K_RCS_T = 1.0 / 1370.0     # °F per (MW·sec) for RCS heat balance
K_PZR_T = 1.0 / 18.0       # °F per (MW·sec) for PZR (much smaller mass)
K_SOLID_P = 0.6            # psig per (gpm·sec)  for solid plant
PZR_VOL_GAL = 1800.0       # nominal pressurizer volume
HEATER_MW = 1.4
RCP_HEAT_MW_EACH = 4.5
DECAY_HEAT_MW = 1.0        # ~few hours after shutdown
RHR_REMOVAL_MW = 22.0      # when running (sized to dominate 0-RCP case)


def saturation_pressure_psig(T_F: float) -> float:
    """Crude saturation curve.  Good enough between 150°F and 600°F."""
    if T_F < 212:
        return 0.0
    if T_F < 300:
        return (T_F - 212) * 1.6
    if T_F < 400:
        return 140 + (T_F - 300) * 2.5
    if T_F < 500:
        return 390 + (T_F - 400) * 6.0
    if T_F < 600:
        return 990 + (T_F - 500) * 12.0
    return 2200.0


def step(s: PlantState, dt: float) -> None:
    """Advance state by `dt` simulated seconds (in-place)."""
    # ── 1차 계통 열수지 ───────────────────────────
    Q_rcp = sum(s.RCP_running) * RCP_HEAT_MW_EACH
    Q_decay = DECAY_HEAT_MW
    # RHR acts as a thermostat targeting ~155°F (cold shutdown).
    # Effective removal scales with (T_RCS - 155) and is capped to prevent
    # over-cooling below the target.
    Q_rhr = 0.0
    if s.RHR_pump_on:
        deficit = s.T_RCS_avg_F - 155.0
        # Pull heat with proportional gain; just enough to balance Q_rcp+Q_decay
        # at steady state when T = 155.
        Q_rhr = max(0.0, min(RHR_REMOVAL_MW, 0.6 * deficit + Q_decay))
    Q_letdown = 0.003 * s.letdown_flow_gpm
    dT_rcs = K_RCS_T * (Q_rcp + Q_decay - Q_rhr - Q_letdown)
    s.T_RCS_avg_F = max(50.0, s.T_RCS_avg_F + dT_rcs * dt)

    # ── 가압기 온도 ───────────────────────────────
    Q_h = HEATER_MW if s.PZR_heater_on else 0.0
    # PZR heater self-throttles toward a target setpoint. Pre-bubble target
    # is 428°F (saturation @ 320 psig).  After bubble, target is held stable.
    target_pzr_T = 432.0 if not s.steam_bubble_formed else 435.0
    overshoot = max(0.0, s.PZR_temp_F - target_pzr_T)
    Q_h *= max(0.0, 1.0 - overshoot / 5.0)
    Q_spray = 0.0
    if s.PZR_spray_open and s.PZR_temp_F > s.T_RCS_avg_F:
        Q_spray = 0.02 * (s.PZR_temp_F - s.T_RCS_avg_F)
    Q_loss = 0.05
    dT_pzr = K_PZR_T * (Q_h - Q_spray - Q_loss)
    s.PZR_temp_F = max(s.T_RCS_avg_F, s.PZR_temp_F + dT_pzr * dt)

    # ── 가압기 수위 ──────────────────────────────
    net_flow = s.charging_flow_gpm - s.letdown_flow_gpm    # gpm
    s.PZR_level_pct += (net_flow / PZR_VOL_GAL) * (dt / 60.0) * 100.0
    s.PZR_level_pct = max(0.0, min(100.0, s.PZR_level_pct))

    # ── Steam bubble 형성 판정 ────────────────────
    P_sat = saturation_pressure_psig(s.PZR_temp_F)
    if (not s.steam_bubble_formed
            and s.PZR_temp_F >= 425.0
            and s.PZR_level_pct < 60.0):
        s.steam_bubble_formed = True
        s.alarms.append({
            "level": "info",
            "text": "Steam bubble formed in pressurizer",
            "t": s.sim_time_s,
        })

    # ── RCS 압력 ─────────────────────────────────
    if s.steam_bubble_formed:
        # Track saturation for current PZR temperature
        target_p = max(320.0, P_sat)
        s.P_RCS_psig += (target_p - s.P_RCS_psig) * min(1.0, dt / 30.0)
    else:
        # Solid plant: PCV-131 holds pressure near setpoint; net flow nudges it
        target_p = s.PCV_131_setpoint_psig
        s.P_RCS_psig += (target_p - s.P_RCS_psig) * min(1.0, dt / 60.0)
        s.P_RCS_psig += K_SOLID_P * net_flow * (dt / 60.0)
        s.P_RCS_psig = max(280.0, min(450.0, s.P_RCS_psig))

    # ── 증기발생기 수위 (1차 라그) ─────────────────
    diff = s.SG_level_target_pct - s.SG_level_NR_pct
    s.SG_level_NR_pct += 0.02 * diff * dt
    s.SG_level_NR_pct = max(0.0, min(100.0, s.SG_level_NR_pct))

    # ── 시간 진행 + 가열률(1시간 슬라이딩 윈도우) ────
    s.sim_time_s += dt
    s._T_history.append((s.sim_time_s, s.T_RCS_avg_F))
    s._T_history = [(t, T) for (t, T) in s._T_history
                    if s.sim_time_s - t <= 3600.0]
    if len(s._T_history) >= 2:
        t0, T0 = s._T_history[0]
        window_hr = (s.sim_time_s - t0) / 3600.0
        if window_hr > 0.01:
            s.heatup_rate_F_per_hr = (s.T_RCS_avg_F - T0) / window_hr

    # ── 자동 MODE 전환 ────────────────────────────
    if s.mode == 5 and s.T_RCS_avg_F >= 200.0:
        s.mode = 4
        s.alarms.append({
            "level": "success",
            "text": "MODE 5 → MODE 4 (HOT SHUTDOWN) 진입",
            "t": s.sim_time_s,
        })


def evaluate_alarms(s: PlantState) -> None:
    """Check setpoint violations and append alarms (deduplicated by text)."""
    existing = {a["text"] for a in s.alarms}

    def add(level: str, text: str):
        if text not in existing:
            s.alarms.append({"level": level, "text": text, "t": s.sim_time_s})
            existing.add(text)

    if s.heatup_rate_F_per_hr > 100.0:
        add("danger", f"가열률 {s.heatup_rate_F_per_hr:.0f}°F/hr > 100°F/hr 제한")
    if any(s.RCP_running) and s.seal_inj_flow_gpm <= 0.0:
        add("danger", "RCP 운전 중 RCP seal injection 상실")
    if s.PZR_level_pct < 80.0 and not s.steam_bubble_formed and s.charging_flow_gpm == 0:
        add("caution", "PZR 수위 80% 미만 - 충전펌프 기동 필요")
    if (s.PZR_spray_open and s.PZR_temp_F - s.T_RCS_avg_F > 320.0):
        add("danger", "ΔT(PZR-spray) 320°F 제한 초과")
    # Trim old alarms to last 8 active
    if len(s.alarms) > 12:
        s.alarms[:] = s.alarms[-12:]
