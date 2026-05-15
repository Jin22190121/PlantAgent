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

당신의 역할: 운전원이 다음 단계를 요청할 때마다 **다음 한 개의 step만** 안내합니다.

⚠ 절대 규칙:
1. 입력 컨텍스트의 `step_progress.completed_step_ids` 목록은 이미 끝난 step들입니다.
   이 목록에 있는 step은 절대 다시 cited_step_id로 쓰지 마세요.
2. 입력 컨텍스트의 `next_step_to_advise.id`가 제공되면 그것을 cited_step_id로 사용하세요.
3. `previous_was_just_acknowledged=true`이면 `completed_step_ids`에 previous_step_id를
   포함시켜 진행 상태를 갱신하세요.
4. 이전에 완료된 step으로 되돌아가지 마세요. 항상 새로운 step만 안내하세요.
5. `next_step_to_advise`가 null이면 모든 step이 완료된 것이므로 kind="respond"로
   "절차 완료" 선언.

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
  "cited_step_id": "<반드시 next_step_to_advise.id 와 동일>",
  "completed_step_ids": ["<직전 완료된 step IDs — previous_step_id 포함 필수>"],
  "expected_outcome": "<예상 상태 변화>",
  "cautions": ["..."],
  "message_to_operator": "<운전원 안내. 다음 패턴 권장:\\n
    [직전 단계 완료] ✓ <prev_id> 완료.\\n
    [현재 단계] <cur_id> — <본문 요약>\\n
    [확인 사항] • ... • ...\\n
    [제안 행동] <tool>(<args>) — 승인이 필요합니다.\\n
    [질문] '이 행동을 실행해도 될까요?'>"
}

행동 결정 규칙:
- next_step의 expected_action이 'operator_check' / 'checklist'이면 kind="advise"
  (도구 호출 X, 운전원이 직접 확인 후 '확인했습니다'라고 말하면 완료 처리됨).
- next_step의 expected_action이 'set_pzr_heater', 'start_rcp' 등 도구 이름이면
  kind="execute"로 해당 도구를 호출 (HITL 승인 모달이 뜸).
- 안전: 가열률 ≤ 100°F/hr, RCP 기동 전 P_RCS ≥ 320 psig, SI 자동 setpoint(PZR<1750psig).
- 모르거나 확실치 않으면 kind="respond"로 운전원에게 질문.

도구 args 사용법 (반드시 정확하게 채울 것):
  start_rcp           → {"pump_id": 1 | 2 | 3 | 4}   (현재 plant.RCP_running에서 false인 가장 낮은 번호 선택)
  stop_rcp            → {"pump_id": 1 | 2 | 3 | 4}
  set_pzr_heater      → {"on": true | false}
  set_pzr_spray       → {"open": true | false}
  set_rhr_pump        → {"on": true | false}
  set_sg_level_target → {"pct": <0-100>}        예: 33
  set_charging_flow   → {"gpm": <number>}
  set_letdown_flow    → {"gpm": <number>}       (최대 120 gpm)
  set_atm_dump        → {"psig": <number>}      예: 1005
  open_msiv / close_msiv / manual_reactor_trip / trip_turbine / actuate_si /
  trip_mfw / start_tdafw / align_charging_to_rwst / evacuate_control_room → {}
  start_afw           → {"mdafw": true|false, "tdafw": true|false}
  advance_time        → {"seconds": <number>}

다중 조작 step (예: GOP-A-6 RCP 1~4 순차 기동):
- 한 번에 RCP 한 기만 제안하세요. plant.RCP_running에서 다음 OFF인 인덱스를 선택.
- 한 펌프가 성공하면 다음 turn에서 동일 cited_step_id로 다음 펌프 제안.
- 4기 모두 running 상태가 되면 자동으로 GOP-A-6가 completed로 마킹됩니다.
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
# Operator confirmation phrases — detects intent to advance one step.
# Matches "확인했습니다", "확인 완료", "다음", "다음 단계 진행", "예", "네",
# bare "next", "done", "continue", "ok", "proceed", etc.
_CONFIRM_RE = re.compile(
    r"(확인.*완료|확인했|확인\s*완료|다음(?:\s*단계|\s*절차|으?로)?|진행해|"
    r"\bnext\b|\bdone\b|\bcontinue\b|\bproceed\b|\bok\b|^\s*예\b|^\s*네\b)",
    re.IGNORECASE,
)


def _next_unmarked_step(doc_type: str, completed: list[str]) -> dict | None:
    """Return the lowest-numbered step in `doc_type` not yet in `completed`."""
    try:
        from npp_agent.ingest.procedures_data import ALL_PROCEDURES
    except Exception:
        return None
    done = set(completed or [])
    candidates = [s for s in ALL_PROCEDURES if s.get("doc_type") == doc_type]
    candidates.sort(key=lambda s: s.get("step_no", 0))
    for s in candidates:
        if s.get("id") not in done:
            return s
    return None


def _fix_tool_args(tool: str, args: dict, plant: dict) -> dict:
    """Auto-fill or coerce common tool args based on plant state.
    Prevents LLM-induced runtime errors like 'pump_id must be 1..4'."""
    args = dict(args or {})
    rcps = plant.get("RCP_running") or [False, False, False, False]

    if tool == "start_rcp":
        pid = args.get("pump_id")
        valid = isinstance(pid, int) and 1 <= pid <= 4 and not rcps[pid - 1]
        if not valid:
            # Pick the lowest-index RCP not yet running
            for i, on in enumerate(rcps):
                if not on:
                    args["pump_id"] = i + 1
                    break
    elif tool == "stop_rcp":
        pid = args.get("pump_id")
        valid = isinstance(pid, int) and 1 <= pid <= 4 and rcps[pid - 1]
        if not valid:
            # Pick the highest-index RCP currently running
            for i in range(3, -1, -1):
                if rcps[i]:
                    args["pump_id"] = i + 1
                    break
    elif tool == "set_pzr_heater" and "on" not in args:
        args["on"] = True
    elif tool == "set_pzr_spray" and "open" not in args:
        args["open"] = True
    elif tool == "set_rhr_pump" and "on" not in args:
        args["on"] = False
    elif tool == "set_sg_level_target" and "pct" not in args:
        args["pct"] = 33
    elif tool == "set_atm_dump" and "psig" not in args:
        args["psig"] = 1005
    elif tool == "set_charging_flow" and "gpm" not in args:
        args["gpm"] = 30
    elif tool == "set_letdown_flow" and "gpm" not in args:
        args["gpm"] = 30
    return args


# Multi-action steps — completion criteria check the plant state.
# Until the gate returns True, the cited step stays in current_step_id and
# the next operator/AI turn re-proposes (e.g. next RCP).
MULTI_ACTION_GATES = {
    # All 4 RCPs running
    "GOP-A-6": lambda plant: sum(plant.get("RCP_running") or []) == 4,
}


def _step_completion_gate_passed(step_id: str, plant: dict) -> bool:
    gate = MULTI_ACTION_GATES.get(step_id)
    return True if gate is None else bool(gate(plant))



def make_plan_action(llm):
    def node(state: AgentState) -> dict:
        plant = state.get("plant_state", {})
        alarms = state.get("alarms", [])
        procs = state.get("retrieved_procedures", [])
        op_in = state.get("operator_input", "")
        completed = list(state.get("completed_step_ids") or [])
        current_id = state.get("current_step_id") or ""

        # ── 1) Operator confirmed previous step → mark it done ────────
        auto_completed: list[str] = []
        if current_id and current_id not in completed \
                and _CONFIRM_RE.search(op_in or ""):
            auto_completed.append(current_id)
            completed.append(current_id)

        # ── 2) Pick the next step deterministically ───────────────────
        active_doc = plant.get("active_doc_type", "GOP")
        next_step = _next_unmarked_step(active_doc, completed)

        # Build context message
        context = {
            "scenario": {
                "scenario_id":      plant.get("scenario_id"),
                "active_doc_type":  active_doc,
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
            "step_progress": {
                "completed_step_ids": completed,
                "completed_count":    len(completed),
                "previous_step_id":   current_id,
                "previous_was_just_acknowledged": bool(auto_completed),
            },
            "next_step_to_advise": (
                {
                    "id":               next_step.get("id"),
                    "step_no":          next_step.get("step_no"),
                    "title":            next_step.get("title"),
                    "text":             (next_step.get("text") or "")[:400],
                    "expected_action":  next_step.get("expected_action"),
                    "parameters":       next_step.get("parameters", []),
                    "setpoints":        next_step.get("setpoints", {}),
                    "cautions":         next_step.get("cautions", []),
                } if next_step else None
            ),
            "candidate_steps": [
                {"id": p["id"],
                 "doc_type": p.get("doc_type"),
                 "title": p.get("title"),
                 "expected_action": p.get("expected_action"),
                 "text": (p.get("text") or "")[:280]}
                for p in procs[:6]
            ],
            "history_titles": [h.get("title") for h in state.get("procedure_history", [])][-5:],
        }

        msg = (
            f"[운전원 발화]\n{op_in}\n\n"
            f"[현재 컨텍스트 (JSON)]\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            "위 컨텍스트만으로 다음 행동을 정하고, 지정된 JSON 객체 하나만 출력하세요.\n"
            "★ 중요: `step_progress.completed_step_ids`에 있는 step은 절대 다시 안내하지 마세요.\n"
            "★ `next_step_to_advise`가 제공되면 그 step을 cited_step_id로 사용하세요.\n"
            "★ `previous_was_just_acknowledged=true`이면 `completed_step_ids`에 previous_step_id를 추가하세요."
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
        cited = parsed.get("cited_step_id") or (
            next_step.get("id") if next_step else "")
        # Combine: auto-detected from operator confirmation + explicit from LLM
        llm_completed = parsed.get("completed_step_ids") or []
        # If the cited step has a multi-action completion gate AND the gate
        # is not yet satisfied, do NOT add cited to completed_step_ids even
        # if LLM put it there.
        new_completed = []
        gated = set()
        for sid in {*auto_completed, *llm_completed}:
            if sid and _step_completion_gate_passed(sid, plant):
                new_completed.append(sid)
            else:
                gated.add(sid)
        if gated:
            # Append a hint message so operator sees why a step isn't ticked
            message += (f"\n[안내] {', '.join(gated)} 는 모든 하위 조작이 "
                        f"완료될 때까지 진행 상태를 유지합니다.")

        # Common state update — persisted by the SqliteSaver checkpointer
        common = {
            "current_step_id": cited,
            "completed_step_ids": new_completed,
        }

        if kind == "execute" and parsed.get("tool"):
            tool_name = parsed["tool"]
            # ★ Auto-fill / coerce args so LLM errors don't crash tools
            fixed_args = _fix_tool_args(tool_name,
                                         parsed.get("args", {}) or {},
                                         plant)
            return {
                **common,
                "proposed_action": {
                    "tool": tool_name,
                    "args": fixed_args,
                    "rationale": parsed.get("rationale", ""),
                    "cited_step_id": cited,
                    "expected_outcome": parsed.get("expected_outcome", ""),
                    "cautions": parsed.get("cautions", []) or [],
                    "completed_step_ids": new_completed,
                },
                "approval_status": "pending",
                "final_messages": [message],
                "done": False,
            }

        # advise or respond — no execution this turn
        return {
            **common,
            "proposed_action": {"completed_step_ids": new_completed},
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
        plant = router.call_internal("get_plant_state", {})
        ok = isinstance(result, dict) and result.get("ok") is True
        msgs = []
        pa = state.get("proposed_action") or {}
        tool = pa.get("tool", "")
        cited = pa.get("cited_step_id", "")
        out: dict = {"plant_state": plant}
        if ok:
            msgs.append(f"✓ {tool} 실행 완료. 현재 MODE {plant.get('mode')}, "
                        f"T_RCS {plant.get('T_RCS_avg_F', 0):.1f}°F, "
                        f"P_RCS {plant.get('P_RCS_psig', 0):.0f} psig.")
            # Tool succeeded — mark cited step done only if its multi-action
            # gate is satisfied (e.g. GOP-A-6 needs all 4 RCPs running).
            if cited and _step_completion_gate_passed(cited, plant):
                out["completed_step_ids"] = [cited]
            elif cited:
                rcps = plant.get("RCP_running") or []
                count = sum(1 for x in rcps if x)
                if cited == "GOP-A-6":
                    msgs.append(f"   ↪ RCP {count}/4 가동 — 나머지 펌프 기동 후 단계 완료.")
        else:
            err = result.get("error", "unknown")
            msgs.append(f"⚠ 도구 실행 실패: {err}")
        out["final_messages"] = msgs
        return out
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
