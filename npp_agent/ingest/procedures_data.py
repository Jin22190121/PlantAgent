"""Curated procedure step database — EOP, AOP, GOP.

Three real procedure documents are indexed:
  • GOP — Westinghouse Section 19.0 Appendix 19-1 §A (MODE 5→4)
  • EOP — Ginna Station E-0  Reactor Trip or Safety Injection (21 immediate steps)
  • AOP — Point Beach AOP-10 Control Room Inaccessibility (17 main steps)

Every step has the same minimum schema:
  id            stable identifier ("GOP-A-4", "EOP-E0-3", "AOP-10-5")
  doc_type      "GOP" | "EOP" | "AOP"
  doc_title     human-readable doc title
  source        ADAMS accession + station
  step_no       int
  title         short title
  text          full body (used for embedding)
  parameters    list of related plant-state keys
  setpoints     dict
  cautions      list of CAUTION/NOTE strings
  expected_action  one of: see ACTION_TO_STEP_ID below
"""

# ─── GOP · Westinghouse Sec.19.0 Appendix 19-1 §A · MODE 5 → MODE 4 ──────────
GOP_APP19_1_A = [
    {"id": "GOP-A-1", "doc_type": "GOP", "step_no": 1,
     "title": "운전감독 기동 허가",
     "text": "Step 1. Permission received from Operations Supervisor for startup.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "GOP-A-2", "doc_type": "GOP", "step_no": 2,
     "title": "SG 수위 33±5% NR 설정",
     "text": "Step 2. Begin establishing steam generator water levels to 33 ±5% narrow-range indication.",
     "parameters": ["SG_level_NR_pct"],
     "setpoints": {"SG_level_target_pct": 33, "SG_level_band_pct": 5},
     "cautions": [], "expected_action": "set_sg_level_target"},
    {"id": "GOP-A-3", "doc_type": "GOP", "step_no": 3,
     "title": "RCP 봉수 주입 확인",
     "text": "Step 3. Verify or establish RCP seal injection flow.",
     "parameters": ["seal_inj_flow_gpm"],
     "setpoints": {"seal_inj_flow_min_gpm": 1.0},
     "cautions": [], "expected_action": "operator_check"},
    {"id": "GOP-A-4", "doc_type": "GOP", "step_no": 4,
     "title": "PZR 히터 기동",
     "text": "Step 4. Energize pressurizer heaters and begin pressurizer heatup.",
     "parameters": ["PZR_heater_on", "PZR_temp_F"],
     "setpoints": {"heatup_rate_max_F_per_hr": 100,
                   "delta_T_PZR_spray_max_F": 320},
     "cautions": ["CAUTION: Do not exceed a heatup rate of 100°F/hr in the pressurizer or RCS.",
                  "CAUTION: Do not exceed 320°F ΔT between pressurizer and spray temperature."],
     "expected_action": "set_pzr_heater"},
    {"id": "GOP-A-5", "doc_type": "GOP", "step_no": 5,
     "title": "PZR steam bubble 형성",
     "text": ("Step 5. Establish a pressurizer steam bubble: (a) raise PZR temperature; "
              "(b) adjust charging and letdown to maintain pressure 320–400 psig "
              "while reducing PZR level; (c) as PZR temp approaches 428°F (sat for 320 psig), "
              "reduce PZR level toward 25%."),
     "parameters": ["PZR_temp_F", "PZR_level_pct", "P_RCS_psig"],
     "setpoints": {"P_RCS_solid_min_psig": 320, "P_RCS_solid_max_psig": 400,
                   "PZR_sat_temp_F": 428, "PZR_level_target_pct": 25},
     "cautions": [], "expected_action": "monitor_steam_bubble"},
    {"id": "GOP-A-6", "doc_type": "GOP", "step_no": 6,
     "title": "RCP 1~4 순차 기동",
     "text": "Step 6. Start the reactor coolant pumps. After running for 5 minutes, sample RCS.",
     "parameters": ["RCP_running"],
     "setpoints": {"RCP_NPSH_min_psig": 320},
     "cautions": [], "expected_action": "start_rcp"},
    {"id": "GOP-A-7", "doc_type": "GOP", "step_no": 7,
     "title": "T_RCS < 160°F 유지 + 가열률 모니터",
     "text": "Step 7. Maintain RCS temperature below 160°F by adjusting RHR HX flow.",
     "parameters": ["T_RCS_avg_F", "heatup_rate_F_per_hr"],
     "setpoints": {"T_RCS_step7_max_F": 160, "heatup_rate_max_F_per_hr": 100},
     "cautions": ["NOTE: 160°F limit prevents excessive ΔT between seal injection water in the intermediate leg and the rest of the RCS."],
     "expected_action": "operator_check"},
    {"id": "GOP-A-8", "doc_type": "GOP", "step_no": 8,
     "title": "RHR 펌프 정지",
     "text": "Step 8. Stop RHR pumps.",
     "parameters": ["RHR_pump_on"], "setpoints": {}, "cautions": [],
     "expected_action": "set_rhr_pump_off"},
    {"id": "GOP-A-9", "doc_type": "GOP", "step_no": 9,
     "title": "T_RCS 200°F까지 자연 가열",
     "text": "Step 9. Allow RCS temperature to increase to 200°F.",
     "parameters": ["T_RCS_avg_F"],
     "setpoints": {"T_RCS_mode4_F": 200},
     "cautions": [], "expected_action": "wait_temp"},
]


# ─── EOP · Ginna Station E-0 · Reactor Trip or Safety Injection ─────────────
EOP_E0_GINNA = [
    {"id": "EOP-E0-1", "doc_type": "EOP", "step_no": 1,
     "title": "원자로 트립 확인 / 수동 트립",
     "text": ("Step 1 (IMMEDIATE). Verify reactor trip: at least one train of reactor "
              "trip breakers OPEN, neutron flux DECREASING, MRPI shows all rods on bottom. "
              "If not tripped, manually trip reactor."),
     "parameters": ["reactor_tripped", "neutron_flux_pct", "rods_inserted"],
     "setpoints": {"power_after_trip_max_pct": 5},
     "cautions": ["IF reactor will not trip OR power range NIS > 5%, GO TO FR-S.1 ATWS."],
     "expected_action": "manual_reactor_trip"},
    {"id": "EOP-E0-2", "doc_type": "EOP", "step_no": 2,
     "title": "터빈 정지 밸브 차단 확인",
     "text": "Step 2 (IMMEDIATE). Verify Turbine Stop Valves CLOSED. If turbine trip not verified, close both MSIVs.",
     "parameters": ["turbine_tripped", "MSIV_open"],
     "setpoints": {}, "cautions": [],
     "expected_action": "trip_turbine"},
    {"id": "EOP-E0-3", "doc_type": "EOP", "step_no": 3,
     "title": "AC 비상모선 전압 확인 ≥ 420V",
     "text": "Step 3 (IMMEDIATE). Verify both trains of AC emergency busses energized to at least 420 VOLTS.",
     "parameters": ["bus_voltage_V"],
     "setpoints": {"bus_voltage_min_V": 420},
     "cautions": ["IF power can NOT be restored to at least one train, GO TO ECA-0.0 LOSS OF ALL AC POWER."],
     "expected_action": "operator_check"},
    {"id": "EOP-E0-4", "doc_type": "EOP", "step_no": 4,
     "title": "안전주입 SI 신호 확인 / 수동 작동",
     "text": ("Step 4 (IMMEDIATE). Check any SI annunciator LIT. If any of the following met, "
              "manually actuate SI and CI: PZR pressure < 1750 psig, steamline pressure < 514 psig, "
              "CNMT pressure > 4 psig, SI sequencing started, or operator determines SI required."),
     "parameters": ["si_signal", "P_RCS_psig", "P_steam_psig", "P_cnmt_psig"],
     "setpoints": {"PZR_si_setpoint_psig": 1750,
                   "steamline_si_setpoint_psig": 514,
                   "CNMT_si_setpoint_psig": 4},
     "cautions": ["IF SI is NOT required, GO TO ES-0.1 REACTOR TRIP RESPONSE."],
     "expected_action": "actuate_si"},
    {"id": "EOP-E0-5", "doc_type": "EOP", "step_no": 5,
     "title": "SI / RHR 펌프 운전 확인",
     "text": "Step 5 (IMMEDIATE). Verify SI pumps and both RHR pumps are RUNNING. Manually start if not.",
     "parameters": ["si_pumps_running", "rhr_pumps_running"],
     "setpoints": {}, "cautions": [],
     "expected_action": "verify_si_pumps"},
    {"id": "EOP-E0-6", "doc_type": "EOP", "step_no": 6,
     "title": "격납건물 RECIRC fan 운전",
     "text": "Step 6 (IMMEDIATE). Verify all CNMT RECIRC fans RUNNING and charcoal filter dampers green status lights EXTINGUISHED.",
     "parameters": ["cnmt_recirc_fans"],
     "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "EOP-E0-7", "doc_type": "EOP", "step_no": 7,
     "title": "격납건물 살수(CNMT spray) 미요구 확인",
     "text": "Step 7 (IMMEDIATE). Verify CNMT spray not required: A-27 alarm extinguished, CNMT pressure < 28 psig.",
     "parameters": ["cnmt_spray_required", "P_cnmt_psig"],
     "setpoints": {"CNMT_spray_setpoint_psig": 28},
     "cautions": [], "expected_action": "operator_check"},
    {"id": "EOP-E0-8", "doc_type": "EOP", "step_no": 8,
     "title": "주증기관 격리 필요 여부 점검",
     "text": ("Step 8 (IMMEDIATE). Check if main steamlines should be isolated: any MSIV open, "
              "CNMT pressure < 18 psig, low Tavg(545°F) AND high steam flow, OR high-high steam flow."),
     "parameters": ["MSIV_open", "T_RCS_avg_F", "steam_flow"],
     "setpoints": {"low_Tavg_setpoint_F": 545,
                   "CNMT_steamline_iso_setpoint_psig": 18},
     "cautions": [], "expected_action": "close_msiv"},
    {"id": "EOP-E0-9", "doc_type": "EOP", "step_no": 9,
     "title": "주급수(MFW) 격리 확인",
     "text": "Step 9 (IMMEDIATE). MFW pumps TRIPPED, MFW flow control valves CLOSED, S/G blowdown and sample valves CLOSED.",
     "parameters": ["mfw_pump_running", "mfw_fcv_open"],
     "setpoints": {}, "cautions": [],
     "expected_action": "trip_mfw"},
    {"id": "EOP-E0-10", "doc_type": "EOP", "step_no": 10,
     "title": "보조급수(AFW) 펌프 운전",
     "text": "Step 10 (IMMEDIATE). Verify MDAFW pumps RUNNING and TDAFW pump RUNNING IF NECESSARY.",
     "parameters": ["afw_pumps_running"],
     "setpoints": {}, "cautions": [],
     "expected_action": "start_afw"},
    {"id": "EOP-E0-11", "doc_type": "EOP", "step_no": 11,
     "title": "기기냉각수(SW) 펌프 ≥ 2기 운전",
     "text": "Step 11 (IMMEDIATE). Verify at least two SW pumps running.",
     "parameters": ["sw_pumps_running"],
     "setpoints": {"sw_pump_min_count": 2}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "EOP-E0-12", "doc_type": "EOP", "step_no": 12,
     "title": "격납건물 격리(CI/CVI) 확인",
     "text": "Step 12 (IMMEDIATE). Verify CI and CVI annunciators LIT, all valves bright (closed).",
     "parameters": ["ci_signal", "cvi_signal"],
     "setpoints": {}, "cautions": [], "expected_action": "operator_check"},
    {"id": "EOP-E0-13", "doc_type": "EOP", "step_no": 13,
     "title": "CCW 시스템 상태 확인",
     "text": "Step 13. Check CCW System Status: at least one CCW pump running. CCW from excess letdown CLOSED.",
     "parameters": ["ccw_pump_running"],
     "setpoints": {}, "cautions": ["CAUTION: RCP TRIP CRITERIA on FOLDOUT must be monitored periodically."],
     "expected_action": "operator_check"},
    {"id": "EOP-E0-14", "doc_type": "EOP", "step_no": 14,
     "title": "SI / RHR 펌프 유량 확인",
     "text": "Step 14. Verify SI flow indicators show flow if RCS pressure < 1400 psig. RHR flow if < 140 psig.",
     "parameters": ["si_flow_gpm", "rhr_flow_gpm", "P_RCS_psig"],
     "setpoints": {"si_inj_pressure_psig": 1400, "rhr_inj_pressure_psig": 140},
     "cautions": [], "expected_action": "operator_check"},
    {"id": "EOP-E0-15", "doc_type": "EOP", "step_no": 15,
     "title": "총 AFW 유량 > 200 gpm 확인",
     "text": "Step 15. Verify total AFW flow GREATER THAN 200 GPM. If S/G NR level > 5% in any S/G, control flow.",
     "parameters": ["afw_flow_total_gpm", "SG_level_NR_pct"],
     "setpoints": {"afw_total_min_gpm": 200, "sg_nr_inadequate_pct": 5},
     "cautions": ["IF AFW flow > 200 gpm cannot be established, GO TO FR-H.1 LOSS OF SECONDARY HEAT SINK."],
     "expected_action": "operator_check"},
    {"id": "EOP-E0-21", "doc_type": "EOP", "step_no": 21,
     "title": "RCS Tavg 547°F 안정",
     "text": ("Step 21. Check RCS Tavg STABLE AT OR TRENDING TO 547°F. "
              "If <547°F decreasing: stop dumping steam, close reheater valves, control AFW. "
              "If >547°F increasing: dump steam to condenser or use S/G ARVs."),
     "parameters": ["T_RCS_avg_F"],
     "setpoints": {"T_RCS_no_load_F": 547},
     "cautions": [], "expected_action": "operator_check"},
]


# ─── AOP · Point Beach AOP-10 · Control Room Inaccessibility ─────────────────
AOP_10_POINTBEACH = [
    {"id": "AOP-10-1", "doc_type": "AOP", "step_no": 1,
     "title": "양 호기 수동 원자로 트립",
     "text": "Step 1. Initiate Manual Reactor Trip for Both Units. Unit 1 Reactor TRIPPED, Unit 2 Reactor TRIPPED.",
     "parameters": ["reactor_tripped"], "setpoints": {}, "cautions": [],
     "expected_action": "manual_reactor_trip"},
    {"id": "AOP-10-2", "doc_type": "AOP", "step_no": 2,
     "title": "양 호기 터빈 트립 확인",
     "text": "Step 2. Ensure Both Units Turbine TRIPPED.",
     "parameters": ["turbine_tripped"], "setpoints": {}, "cautions": [],
     "expected_action": "trip_turbine"},
    {"id": "AOP-10-3", "doc_type": "AOP", "step_no": 3,
     "title": "MSIV 차단",
     "text": "Step 3. Shut Main Steam Isolation Valves: 1MS-2018, 1MS-2017, 2MS-2018, 2MS-2017.",
     "parameters": ["MSIV_open"], "setpoints": {}, "cautions": [],
     "expected_action": "close_msiv"},
    {"id": "AOP-10-4", "doc_type": "AOP", "step_no": 4,
     "title": "대기증기방출 1005 psig 조정",
     "text": "Step 4. Adjust Atmospheric Steam Dump Controllers (1HC-468, 1HC-478, 2HC-468, 2HC-478) to 1005 PSIG.",
     "parameters": ["atm_steam_dump_setpoint_psig"],
     "setpoints": {"atm_steam_dump_setpoint_psig": 1005},
     "cautions": [], "expected_action": "set_atm_dump"},
    {"id": "AOP-10-5", "doc_type": "AOP", "step_no": 5,
     "title": "충전펌프 흡입 RWST로 정렬",
     "text": ("Step 5. Align Charging Pump Suctions To RWST: open RWST→charging suction MOV "
              "(1CV-112B, 2CV-112B), shut VCT outlet→charging suction MOV (1CV-112C, 2CV-112C)."),
     "parameters": ["charging_suction_rwst"],
     "setpoints": {}, "cautions": [],
     "expected_action": "align_charging_to_rwst"},
    {"id": "AOP-10-6", "doc_type": "AOP", "step_no": 6,
     "title": "터빈 구동 보조급수펌프 기동",
     "text": "Step 6. Start Turbine-Driven AFW Pumps (1P-29, 2P-29).",
     "parameters": ["tdafw_running"], "setpoints": {}, "cautions": [],
     "expected_action": "start_tdafw"},
    {"id": "AOP-10-7", "doc_type": "AOP", "step_no": 7,
     "title": "MDAFW 토출 밸브 - MANUAL PULLOUT/CLOSE",
     "text": "Step 7. Place Motor-Driven AFW Discharge Valves (AF-4021 for SG-1B, AF-4022 for SG-2A) in MANUAL PULLOUT and CLOSE.",
     "parameters": ["mdafw_disch_valve"],
     "setpoints": {},
     "cautions": ["CAUTION: Placing Main Feed Pump control switches in pull-out will defeat auto start of the Motor Driven AFW pumps."],
     "expected_action": "operator_check"},
    {"id": "AOP-10-8", "doc_type": "AOP", "step_no": 8,
     "title": "주급수펌프 정지 + 제어 AUTO",
     "text": "Step 8. Stop Main Feedwater Pumps (1P-28A, 1P-28B, 2P-28A, 2P-28B) And Place Control Switches In AUTO.",
     "parameters": ["mfw_pump_running"], "setpoints": {}, "cautions": [],
     "expected_action": "trip_mfw"},
    {"id": "AOP-10-9", "doc_type": "AOP", "step_no": 9,
     "title": "히터드레인탱크 펌프 정지",
     "text": "Step 9. Stop Heater Drain Tank Pumps (1P-27A/B/C, 2P-27A/B/C).",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-10", "doc_type": "AOP", "step_no": 10,
     "title": "복수펌프 호기당 1기만 운전",
     "text": "Step 10. Ensure Only One Condensate Pump Running Per Unit (P-25A or P-25B).",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-11", "doc_type": "AOP", "step_no": 11,
     "title": "주제어실 철수",
     "text": "Step 11. Evacuate Control Room and obtain copies of this procedure from the Work Control Center.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "evacuate_control_room"},
    {"id": "AOP-10-12", "doc_type": "AOP", "step_no": 12,
     "title": "CAS에 통보",
     "text": "Step 12. Notify CAS Of Control Room Evacuation.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-13", "doc_type": "AOP", "step_no": 13,
     "title": "면허 운전원 4명 분산 배치",
     "text": ("Step 13. Dispatch Four Licensed Operators to Perform Local Actions: "
              "Unit-1 AFW Pump Operator (Att. A), Unit-2 AFW Pump Operator (Att. B), "
              "Unit-1 Charging Pump Operator (Att. C), Unit-2 Charging Pump Operator (Att. D)."),
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-14", "doc_type": "AOP", "step_no": 14,
     "title": "추가 운전원 분산 배치",
     "text": "Step 14. Dispatch Operators (Turbine Hall Operator [Att. E], PAB Operator [Att. F]).",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-15", "doc_type": "AOP", "step_no": 15,
     "title": "STA TSC 보고 + 비상계획 실행",
     "text": "Step 15. Direct STA To Report To TSC And Implement Emergency Plan.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-16", "doc_type": "AOP", "step_no": 16,
     "title": "현장에서 운전기기 모니터",
     "text": "Step 16. Locally Monitor Operating Equipment Until Control Room Can Be Re-Entered.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
    {"id": "AOP-10-17", "doc_type": "AOP", "step_no": 17,
     "title": "주제어실 재진입 가능 시 EOP-0 진입",
     "text": "Step 17. Check Control Room HABITABLE. WHEN re-entry possible, GO TO EOP-0 REACTOR TRIP OR SI.",
     "parameters": [], "setpoints": {}, "cautions": [],
     "expected_action": "operator_check"},
]


# ─── master tables ────────────────────────────────────────────
ALL_PROCEDURES = GOP_APP19_1_A + EOP_E0_GINNA + AOP_10_POINTBEACH

DOC_META = {
    "GOP": {
        "title": "Westinghouse Section 19.0 Appendix 19-1 §A",
        "long_title": "Plant Heatup: COLD SHUTDOWN → HOT SHUTDOWN (MODE 5 → MODE 4)",
        "source": "USNRC HRTD ML11223A342",
        "kind": "정상운전절차서 (General Operating Procedure)",
        "scenario_id": "scenario_gop_heatup",
    },
    "EOP": {
        "title": "Ginna Station EOP E-0",
        "long_title": "Reactor Trip or Safety Injection",
        "source": "Rochester G&E ML17263A589",
        "kind": "비상운전절차서 (Emergency Operating Procedure)",
        "scenario_id": "scenario_eop_reactor_trip_si",
    },
    "AOP": {
        "title": "Point Beach AOP-10",
        "long_title": "Control Room Inaccessibility",
        "source": "Point Beach NPP ML030730736",
        "kind": "비정상운전절차서 (Abnormal Operating Procedure)",
        "scenario_id": "scenario_aop_cr_inaccessible",
    },
}

# expected_action → (step_id) lookup, for ML→tool mapping
ACTION_TO_STEP_ID = {}
for s in ALL_PROCEDURES:
    a = s.get("expected_action", "operator_check")
    if a not in ("operator_check", "checklist") and a not in ACTION_TO_STEP_ID:
        ACTION_TO_STEP_ID[a] = s["id"]


# ─── setpoint inverse index ───────────────────────────────────
# Map plant-state key → list of step_ids that mention it.
def build_setpoint_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for s in ALL_PROCEDURES:
        for k in s.get("parameters", []):
            idx.setdefault(k, []).append(s["id"])
        for k in s.get("setpoints", {}).keys():
            idx.setdefault(k, []).append(s["id"])
    return idx


SETPOINT_INDEX = build_setpoint_index()
