"""Hand-curated Appendix 19-1 §A procedure steps (MODE 5 → MODE 4).

Each step is the unit of indexing for the RAG layer. Wording is condensed
from the Westinghouse Technology Systems Manual Section 19.0.

Schema:
  id            "App19-1-A-{n}"  unique step identifier
  step_no       integer
  title         short title
  text          full step body (used for embedding)
  mode_from     5
  mode_to       4
  parameters    list of plant-state keys this step touches/monitors
  setpoints     dict of named setpoint values
  cautions      list of CAUTION / NOTE strings tied to this step
  expected_action  one of "operator_check", "set_pzr_heater", ...
                   Used by the agent to map procedure → operator tool.
"""

APPENDIX_19_1_A = [
    {
        "id": "App19-1-A-1",
        "step_no": 1,
        "title": "운전감독 기동 허가",
        "text": ("Step 1. Permission received from Operations Supervisor for "
                 "startup. (운전감독으로부터 기동 허가를 받는다.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-2",
        "step_no": 2,
        "title": "SG 수위 33±5% NR 설정",
        "text": ("Step 2. Begin establishing steam generator water levels to "
                 "33 ±5% narrow-range indication. (SG 수위를 wet layup 100% → "
                 "33±5% 협역으로 조정 시작.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["SG_level_NR_pct"],
        "setpoints": {"SG_level_target_pct": 33, "SG_level_band_pct": 5},
        "cautions": [],
        "expected_action": "set_sg_level_target",
    },
    {
        "id": "App19-1-A-3",
        "step_no": 3,
        "title": "RCP 봉수 주입 확인",
        "text": ("Step 3. Verify or establish RCP seal injection flow. "
                 "(RCP 봉수 주입 유량을 확인하거나 확립.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["seal_inj_flow_gpm"],
        "setpoints": {"seal_inj_flow_min_gpm": 1.0},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-4",
        "step_no": 4,
        "title": "PZR 히터 기동",
        "text": ("Step 4. Energize pressurizer heaters and begin pressurizer "
                 "heatup. (가압기 히터 기동 후 가압기 가열 시작.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["PZR_heater_on", "PZR_temp_F"],
        "setpoints": {"heatup_rate_max_F_per_hr": 100,
                      "delta_T_PZR_spray_max_F": 320},
        "cautions": [
            "CAUTION: Do not exceed a heatup rate of 100°F/hr in the "
            "pressurizer or in the RCS.",
            "CAUTION: Do not exceed 320°F ΔT between pressurizer and spray "
            "temperature.",
        ],
        "expected_action": "set_pzr_heater",
    },
    {
        "id": "App19-1-A-5",
        "step_no": 5,
        "title": "PZR steam bubble 형성",
        "text": ("Step 5. Establish a pressurizer steam bubble: (a) raise "
                 "pressurizer temperature using heaters; (b) adjust charging "
                 "and letdown to maintain pressurizer pressure ~320–400 psig "
                 "while reducing pressurizer level; (c) as pressurizer "
                 "temperature approaches 428°F (saturation for 320 psig), "
                 "reduce pressurizer level toward 25%."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["PZR_temp_F", "PZR_level_pct", "P_RCS_psig",
                       "charging_flow_gpm", "letdown_flow_gpm"],
        "setpoints": {
            "P_RCS_solid_min_psig": 320,
            "P_RCS_solid_max_psig": 400,
            "PZR_sat_temp_F": 428,
            "PZR_level_target_pct": 25,
        },
        "cautions": [],
        "expected_action": "monitor_steam_bubble",
    },
    {
        "id": "App19-1-A-6",
        "step_no": 6,
        "title": "RCP 1~4 순차 기동",
        "text": ("Step 6. Start the reactor coolant pumps. After running the "
                 "pumps for 5 minutes, sample the RCS for chemistry "
                 "specifications. Partially open pressurizer spray valves for "
                 "coolant mixing."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["RCP_running"],
        "setpoints": {"RCP_NPSH_min_psig": 320,
                      "RCP_runtime_before_sample_min": 5},
        "cautions": [],
        "expected_action": "start_rcp",
    },
    {
        "id": "App19-1-A-7",
        "step_no": 7,
        "title": "T_RCS < 160°F 유지 + 가열률 모니터",
        "text": ("Step 7. Maintain the RCS temperature below 160°F by "
                 "adjusting flow through the RHR heat exchangers. NOTE: The "
                 "160°F limit is based on cold-water-addition accident "
                 "considerations."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["T_RCS_avg_F", "heatup_rate_F_per_hr"],
        "setpoints": {"T_RCS_step7_max_F": 160,
                      "heatup_rate_max_F_per_hr": 100},
        "cautions": [
            "NOTE: 160°F limit prevents excessive ΔT between seal injection "
            "water in the intermediate leg and the rest of the RCS."
        ],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-8",
        "step_no": 8,
        "title": "RHR 펌프 정지",
        "text": "Step 8. Stop RHR pumps.",
        "mode_from": 5, "mode_to": 4,
        "parameters": ["RHR_pump_on"],
        "setpoints": {},
        "cautions": [],
        "expected_action": "set_rhr_pump_off",
    },
    {
        "id": "App19-1-A-9",
        "step_no": 9,
        "title": "T_RCS 200°F까지 자연 가열",
        "text": ("Step 9. Allow RCS temperature to increase to 200°F. (RCP "
                 "기계 손실열 + 붕괴열로 자연 가열, ~50°F/hr 예상.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["T_RCS_avg_F"],
        "setpoints": {"T_RCS_mode4_F": 200},
        "cautions": [],
        "expected_action": "wait_temp",
    },
    {
        "id": "App19-1-A-10",
        "step_no": 10,
        "title": "1차 계통 화학 시방 확인",
        "text": ("Step 10. When RCS temperature reaches 200°F, ascertain that "
                 "primary system water chemistry is within specifications."),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-11",
        "step_no": 11,
        "title": "PZR 수위 자동 모드",
        "text": ("Step 11. When pressurizer level is at the no-load operating "
                 "level (25%), place the pressurizer level control system in "
                 "automatic."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["PZR_level_pct"],
        "setpoints": {"PZR_level_no_load_pct": 25},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-12",
        "step_no": 12,
        "title": "정지뱅크 인출 + SDM 확인",
        "text": ("Step 12. Verify shutdown rods are withdrawn and that "
                 "sufficient SHUTDOWN MARGIN is available."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["K_eff"],
        "setpoints": {"SDM_min_pct_dk_k": 1.0},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-13",
        "step_no": 13,
        "title": "MODE 4 Tech Spec 확인 (Checklist 1)",
        "text": ("Step 13. Ensure applicable pre-startup Technical "
                 "Specification requirements are met. (See Checklist No. 1.)"),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "checklist",
    },
    {
        "id": "App19-1-A-14",
        "step_no": 14,
        "title": "VCT 수소 블랭킷",
        "text": ("Step 14. At 200°F RCS temperature, establish a hydrogen "
                 "blanket in the volume control tank."),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-15",
        "step_no": 15,
        "title": "MSIV 개방 + 주증기관 가열",
        "text": "Step 15. Open main steam isolation valves and warm main steam lines.",
        "mode_from": 5, "mode_to": 4,
        "parameters": ["MSIV_open"],
        "setpoints": {},
        "cautions": [],
        "expected_action": "open_msiv",
    },
    {
        "id": "App19-1-A-16",
        "step_no": 16,
        "title": "PZR 가열 지속 + 저압 letdown",
        "text": ("Step 16. Continue pressurizer heatup to maintain desired "
                 "pressure. Use low-pressure letdown control valve to "
                 "maintain letdown flow. Once a steam bubble is established "
                 "in the pressurizer, RCS pressure will be controlled by "
                 "heater and spray operation."),
        "mode_from": 5, "mode_to": 4,
        "parameters": ["PZR_heater_on", "letdown_flow_gpm"],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-17",
        "step_no": 17,
        "title": "복수 정화 진행",
        "text": "Step 17. Condensate cleanup is in progress.",
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-18",
        "step_no": 18,
        "title": "복수/급수 at-power 정렬",
        "text": ("Step 18. When condensate chemistry is within specifications, "
                 "align the condensate and feedwater systems to the normal "
                 "at-power configuration."),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-19",
        "step_no": 19,
        "title": "복수기 진공 확인",
        "text": "Step 19. Verify condenser vacuum.",
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {},
        "cautions": [],
        "expected_action": "operator_check",
    },
    {
        "id": "App19-1-A-20",
        "step_no": 20,
        "title": "주터빈 가열",
        "text": ("Step 20. Warm the main turbine. CAUTION: Prior to reaching "
                 "350°F in the RCS, verify control rod drive mechanism "
                 "cooling fans are operating, and terminate residual heat "
                 "removal letdown to the CVCS."),
        "mode_from": 5, "mode_to": 4,
        "parameters": [],
        "setpoints": {"T_RCS_pre_mode3_F": 350},
        "cautions": [
            "CAUTION: Verify CRDM cooling fans are operating before 350°F.",
            "CAUTION: Terminate RHR letdown to CVCS before 350°F.",
        ],
        "expected_action": "operator_check",
    },
]


# Quick action → procedure mapping (used by simulator for context-aware help)
ACTION_TO_STEP_ID = {
    s["expected_action"]: s["id"]
    for s in APPENDIX_19_1_A
    if s["expected_action"] not in ("operator_check", "checklist")
}
