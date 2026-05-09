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


PLANNER_SYSTEM = """당신은 한국 운전원을 보조하는 정상운전(Westinghouse PWR, Section 19.0 Appendix 19-1 §A) AI Agent입니다.

당신의 역할: 운전원의 발화와 현재 플랜트 상태를 보고, 다음에 할 행동을 한 번에 하나만 결정합니다.

다음 JSON 객체 하나만 출력하세요. 다른 텍스트 금지.

{
  "kind": "advise" | "execute" | "respond",
  "tool": null | "set_pzr_heater" | "set_pzr_spray" | "set_rhr_pump" | "start_rcp" | "stop_rcp" | "set_charging_flow" | "set_letdown_flow" | "set_sg_level_target" | "open_msiv" | "advance_time",
  "args": {},
  "rationale": "<왜 이 행동인지>",
  "cited_step_id": "App19-1-A-N",
  "expected_outcome": "<예상 상태 변화>",
  "cautions": ["..."],
  "message_to_operator": "<운전원에게 보일 한국어 메시지 — 절차 step ID 포함>"
}

규칙:
- 반드시 절차 step을 인용하세요. 인용된 step의 텍스트만 근거로 삼으세요.
- 실 조작이 필요하면 kind="execute", 안내·확인만이면 "advise", 절차 종료/대기면 "respond".
- 안전: 가열률 ≤ 100°F/hr, RCP 기동 전 P_RCS ≥ 320 psig, ΔT(PZR-spray) ≤ 320°F.
- 모르거나 확신이 없으면 kind="respond"로 운전원에게 정보 요청.
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
        mode = plant.get("mode", 5)
        # Always grab the mode-transition list as anchor
        by_mode = router.call_internal(
            "search_by_mode_transition",
            {"from_mode": mode, "to_mode": max(1, mode - 1)},
        )
        ranked = []
        sit = state.get("operator_input", "") or ""
        if sit:
            sem = router.call_internal(
                "search_procedure", {"situation": sit, "n_results": 4}
            )
            ranked = sem.get("procedures", []) if isinstance(sem, dict) else []
        # Merge keeping order: semantic top-k then mode anchor
        seen = set()
        merged = []
        for src in (ranked, by_mode.get("procedures", []) if isinstance(by_mode, dict) else []):
            for p in src:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                merged.append(p)
        return {"retrieved_procedures": merged[:8]}
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
            "plant": {
                "mode": plant.get("mode"),
                "T_RCS_F": round(plant.get("T_RCS_avg_F", 0), 1),
                "P_RCS_psig": round(plant.get("P_RCS_psig", 0), 0),
                "PZR_level_pct": round(plant.get("PZR_level_pct", 0), 1),
                "PZR_temp_F": round(plant.get("PZR_temp_F", 0), 1),
                "RCP_running": plant.get("RCP_running"),
                "RHR_pump_on": plant.get("RHR_pump_on"),
                "PZR_heater_on": plant.get("PZR_heater_on"),
                "heatup_rate_F_per_hr": round(plant.get("heatup_rate_F_per_hr", 0), 1),
                "steam_bubble_formed": plant.get("steam_bubble_formed"),
            },
            "alarms": [a.get("text") for a in alarms[-5:]],
            "candidate_steps": [
                {"id": p["id"], "title": p.get("title"),
                 "expected_action": p.get("expected_action"),
                 "text": (p.get("text") or "")[:300]}
                for p in procs[:6]
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
            return {
                "proposed_action": {
                    "tool": "", "args": {},
                    "rationale": f"LLM 호출 실패: {e}",
                    "cited_step_id": "",
                    "expected_outcome": "",
                    "cautions": [],
                },
                "approval_status": "n/a",
                "final_messages": [f"⚠ AI 응답 오류: {e}"],
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
