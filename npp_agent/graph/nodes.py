"""LangGraph nodes for the NPP AI Agent.

Pipeline (per operator turn):
  assess_state → retrieve_procedure → plan_action → approval_gate →
  execute_action → verify_outcome → log_step → respond

`approval_gate` calls `interrupt(...)` when the proposed action is E-tier.
The graph pauses and surfaces the payload to the caller; resuming with
`Command(resume={"approved": True/False, ...})` continues execution.
"""
from __future__ import annotations
import json
import re
from typing import Any

from langgraph.types import interrupt

from npp_agent.safety import classify, requires_approval, issue_token, Tier
from .state import AgentState


# ── helpers ───────────────────────────────────────
def _safe_json_extract(text: str) -> dict | None:
    """Extract first valid JSON object from possibly fenced LLM output."""
    if not text:
        return None
    text = text.strip()
    # try to strip fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # try to locate object braces
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


PLANNER_SYSTEM = """당신은 한국 운전원을 보조하는 NPP AI Agent입니다.
3개 절차서를 학습했습니다:
  • GOP (Westinghouse §19.0 App19-1 §A) — 콜드셧다운 → 핫셧다운 기동
  • EOP (Ginna E-0) — 원자로 트립 / 안전주입
  • AOP (Point Beach AOP-10) — 주제어실 접근 불능

당신의 역할: 활성 시나리오와 현재 플랜트 상태를 보고, 해당 doc_type의
절차서를 인용하여 다음 행동을 한 번에 하나만 결정합니다.

다음 JSON 객체 하나만 출력하세요. 다른 텍스트 금지.

{
  "kind": "advise" | "execute" | "respond",
  "tool": null | "set_pzr_heater" | "set_rhr_pump" | "start_rcp" | "stop_rcp" |
                 "set_charging_flow" | "set_letdown_flow" | "set_sg_level_target" |
                 "open_msiv" | "close_msiv" | "manual_reactor_trip" | "trip_turbine" |
                 "actuate_si" | "trip_mfw" | "start_afw" | "start_tdafw" |
                 "align_charging_to_rwst" | "set_atm_dump" | "evacuate_control_room" |
                 "advance_time",
  "args": {},
  "rationale": "<왜 이 행동인지>",
  "cited_step_id": "<예: GOP-A-4 / EOP-E0-1 / AOP-10-3>",
  "expected_outcome": "<예상 상태 변화>",
  "cautions": ["..."],
  "message_to_operator": "<운전원 안내 한국어 — step ID 인용 포함>"
}

규칙:
- 반드시 활성 시나리오의 doc_type에 해당하는 절차 step을 인용하세요.
- IMMEDIATE ACTION step은 즉시 실행, 비-immediate는 advise/respond로 안내.
- 안전: 가열률 ≤ 100°F/hr, RCP 기동 전 P_RCS ≥ 320 psig, SI 자동 setpoint(PZR<1750psig).
- 모르면 kind="respond"로 정보 요청.
"""


# ── 1. assess_state ───────────────────────────────
def make_assess_state(router):
    def node(state: AgentState) -> dict:
        plant = router.call_internal("get_plant_state", {})
        alarms = router.call_internal("get_alarm_list", {})
        return {
            "plant_state": plant,
            "alarms": alarms.get("alarms", []) if isinstance(alarms, dict) else [],
        }
    return node


# ── 2. retrieve_procedure ─────────────────────────
def make_retrieve_procedure(router):
    def node(state: AgentState) -> dict:
        plant = state.get("plant_state", {}) or {}
        active_doc = plant.get("active_doc_type", "GOP")

        # Anchor: full step list of the active doc (in order)
        in_doc = router.call_internal("list_steps_in_doc",
                                      {"doc_type": active_doc})
        anchor = in_doc.get("steps", []) if isinstance(in_doc, dict) else []

        # Semantic search constrained to active doc
        sit = state.get("operator_input", "") or ""
        ranked = []
        if sit:
            sem = router.call_internal("search_procedure",
                                        {"situation": sit, "n_results": 4,
                                         "doc_type": active_doc})
            ranked = sem.get("procedures", []) if isinstance(sem, dict) else []

        # Setpoint-violation lookup from active alarms
        violations = []
        for a in state.get("alarms", []) or []:
            text = (a.get("text") or "")
            if "psig" in text or "°F/hr" in text or "수위" in text:
                # naive trigger: just include relevant steps for active doc
                pass

        seen = set()
        merged = []
        # Order: semantic top-k, then anchor list (so plan_action sees both)
        for src in (ranked, anchor):
            for p in src:
                pid = p.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                merged.append(p)
        return {"retrieved_procedures": merged[:10]}
    return node


# ── 3. plan_action ────────────────────────────────
def make_plan_action(llm):
    def node(state: AgentState) -> dict:
        plant = state.get("plant_state", {})
        alarms = state.get("alarms", [])
        procs = state.get("retrieved_procedures", [])
        op_in = state.get("operator_input", "")

        # Build context message
        context = {
            "scenario": {
                "scenario_id":      plant.get("scenario_id"),
                "active_doc_type":  plant.get("active_doc_type"),
                "mode":             plant.get("mode"),
                "mode_name":        plant.get("mode_name"),
            },
            "plant": {
                "T_RCS_F":          round(plant.get("T_RCS_avg_F", 0), 1),
                "P_RCS_psig":       round(plant.get("P_RCS_psig", 0), 0),
                "PZR_level_pct":    round(plant.get("PZR_level_pct", 0), 1),
                "PZR_temp_F":       round(plant.get("PZR_temp_F", 0), 1),
                "P_steam_psig":     round(plant.get("P_steam_psig", 0), 0),
                "P_cnmt_psig":      round(plant.get("P_cnmt_psig", 0), 1),
                "SG_level_NR_pct":  round(plant.get("SG_level_NR_pct", 0), 1),
                "RCP_running":      plant.get("RCP_running"),
                "RHR_pump_on":      plant.get("RHR_pump_on"),
                "PZR_heater_on":    plant.get("PZR_heater_on"),
                "MSIV_open":        plant.get("MSIV_open"),
                "reactor_tripped":  plant.get("reactor_tripped"),
                "turbine_tripped":  plant.get("turbine_tripped"),
                "si_signal":        plant.get("si_signal"),
                "si_pumps_running": plant.get("si_pumps_running"),
                "mfw_pump_running": plant.get("mfw_pump_running"),
                "afw_running": (plant.get("afw_mdafw_running") or
                                plant.get("afw_tdafw_running")),
                "control_room_evacuated": plant.get("control_room_evacuated"),
                "heatup_rate_F_per_hr":   round(plant.get("heatup_rate_F_per_hr", 0), 1),
                "steam_bubble_formed":    plant.get("steam_bubble_formed"),
            },
            "alarms": [a.get("text") for a in alarms[-5:]],
            "candidate_steps": [
                {"id": p["id"],
                 "doc_type": p.get("doc_type"),
                 "title": p.get("title"),
                 "expected_action": p.get("expected_action"),
                 "text": (p.get("text") or "")[:280]}
                for p in procs[:8]
            ],
            "history_titles": [h.get("title") for h in state.get("procedure_history", [])][-5:],
        }

        msg = (
            f"[운전원 발화]\n{op_in}\n\n"
            f"[현재 컨텍스트 (JSON)]\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            "위 컨텍스트만으로 다음 행동을 정하고, 지정된 JSON 객체 하나만 출력하세요."
        )

        try:
            response = llm.invoke([
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": msg},
            ])
            text = getattr(response, "content", str(response)) or ""
        except Exception as e:
            err = str(e).lower()
            hint = ""
            if "503" in err or "unavailable" in err or "overload" in err \
                    or "429" in err or "quota" in err or "resourceexhausted" in err:
                hint = ("\n💡 체인의 모든 Gemini 모델이 동시에 한도/과부하입니다.\n"
                        "  • 1~5분 후 자동 회복 (각 모델 RPM 한도 1분마다 리셋)\n"
                        "  • 체인 순서 사용자 지정: export GEMINI_MODELS=\"gemini-2.0-flash-lite,gemini-1.5-flash-8b,...\"\n"
                        "  • 일일 한도(RPD) 도달이면 자정(태평양 시간) 후 회복")
            elif "api_key" in err or "credential" in err or "auth" in err:
                hint = ("\n💡 API 키 인식 실패. .env 파일에 GOOGLE_API_KEY 설정 후 재기동.")
            return {
                "proposed_action": {
                    "tool": "", "args": {},
                    "rationale": f"LLM 호출 실패: {e}",
                    "cited_step_id": "",
                    "expected_outcome": "",
                    "cautions": [],
                },
                "approval_status": "n/a",
                "final_messages": [f"⚠ AI 응답 오류: {e}{hint}"],
                "done": True,
            }

        parsed = _safe_json_extract(text) or {}
        kind = parsed.get("kind", "respond")
        message = parsed.get("message_to_operator") or text

        if kind == "execute" and parsed.get("tool"):
            return {
                "proposed_action": {
                    "tool": parsed["tool"],
                    "args": parsed.get("args", {}) or {},
                    "rationale": parsed.get("rationale", ""),
                    "cited_step_id": parsed.get("cited_step_id", ""),
                    "expected_outcome": parsed.get("expected_outcome", ""),
                    "cautions": parsed.get("cautions", []) or [],
                },
                "approval_status": "pending",
                "final_messages": [message],
                "done": False,
            }

        # advise or respond — no execution this turn
        return {
            "proposed_action": {},
            "approval_status": "n/a",
            "final_messages": [message],
            "done": True,
        }
    return node


# ── 4. approval_gate (HITL interrupt) ─────────────
def approval_gate(state: AgentState) -> dict:
    pa = state.get("proposed_action") or {}
    tool = pa.get("tool")
    if not tool:
        return {"approval_status": "n/a"}
    if not requires_approval(tool):
        # R/A tier: auto-approve
        return {"approval_status": "approved"}

    # E-tier: surface to operator and pause
    decision = interrupt({
        "kind": "approval_request",
        "tool": tool,
        "args": pa.get("args", {}),
        "rationale": pa.get("rationale", ""),
        "cited_step_id": pa.get("cited_step_id", ""),
        "expected_outcome": pa.get("expected_outcome", ""),
        "cautions": pa.get("cautions", []),
        "tier": "E",
    })

    if not isinstance(decision, dict):
        return {"approval_status": "rejected", "approval_reason": "invalid resume"}

    if decision.get("approved"):
        # Optionally apply argument modifications
        if "modified_args" in decision and isinstance(decision["modified_args"], dict):
            new_pa = dict(pa)
            new_pa["args"] = decision["modified_args"]
            return {
                "proposed_action": new_pa,
                "approval_status": "modified",
                "approval_reason": decision.get("reason", "operator-modified"),
            }
        return {"approval_status": "approved"}

    return {
        "approval_status": "rejected",
        "approval_reason": decision.get("reason", "operator-rejected"),
    }


# ── 5. execute_action ─────────────────────────────
def make_execute_action(router):
    def node(state: AgentState) -> dict:
        if state.get("approval_status") not in ("approved", "modified"):
            return {"last_tool_result": {"skipped": True}}
        pa = state["proposed_action"]
        tool = pa["tool"]
        args = pa.get("args", {}) or {}
        # Issue token (proves this came through the gate)
        token = issue_token(tool, args, thread_id="graph")
        result = router.call(tool, args, source="agent", approval_token=token, thread_id="graph")
        return {"last_tool_result": result}
    return node


# ── 6. verify_outcome ─────────────────────────────
def make_verify_outcome(router):
    def node(state: AgentState) -> dict:
        if state.get("approval_status") != "approved" and state.get("approval_status") != "modified":
            return {}
        result = state.get("last_tool_result", {})
        # Pull fresh state for verification
        plant = router.call_internal("get_plant_state", {})
        ok = isinstance(result, dict) and result.get("ok") is True
        msgs = []
        if ok:
            tool = state["proposed_action"]["tool"]
            msgs.append(f"✓ {tool} 실행 완료. 현재 MODE {plant.get('mode')}, "
                        f"T_RCS {plant.get('T_RCS_avg_F', 0):.1f}°F, "
                        f"P_RCS {plant.get('P_RCS_psig', 0):.0f} psig.")
        else:
            err = result.get("error", "unknown")
            msgs.append(f"⚠ 도구 실행 실패: {err}")
        return {"plant_state": plant, "final_messages": msgs}
    return node


# ── 7. log_step ───────────────────────────────────
def make_log_step(router, thread_id_resolver):
    def node(state: AgentState) -> dict:
        pa = state.get("proposed_action") or {}
        tool = pa.get("tool")
        if not tool:
            return {}
        sess = thread_id_resolver(state)
        action_str = f"{tool}({json.dumps(pa.get('args', {}), ensure_ascii=False)})"
        try:
            router.call_internal("log_action", {
                "session_id": sess,
                "action": action_str,
                "procedure_ref": pa.get("cited_step_id", ""),
            })
        except Exception:
            pass
        return {
            "procedure_history": [{
                "step_id": pa.get("cited_step_id"),
                "title": pa.get("rationale", "")[:80],
                "tool": tool,
                "status": state.get("approval_status"),
            }],
        }
    return node


# ── 8. respond ────────────────────────────────────
def respond(state: AgentState) -> dict:
    msgs = state.get("final_messages", []) or []
    pa = state.get("proposed_action") or {}
    if state.get("approval_status") == "rejected":
        msgs.append(f"❌ 운전원이 거부함: {state.get('approval_reason', '')}")
    elif state.get("approval_status") == "modified":
        msgs.append(f"✎ 운전원이 인자 수정 후 승인: {pa.get('args')}")
    final = "\n".join(m for m in msgs if m)
    return {"final_answer": final or "(응답 없음)", "done": True}
