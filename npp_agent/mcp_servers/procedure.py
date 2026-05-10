"""Procedure MCP — multi-document RAG over GOP, EOP, AOP.

Three procedure documents are loaded into a single ChromaDB collection
with `doc_type` metadata for filtering:

  GOP — Westinghouse §19.0 App19-1 §A (MODE 5→4 heatup)        9 steps
  EOP — Ginna E-0 (Reactor Trip / Safety Injection)            16 steps
  AOP — Point Beach AOP-10 (Control Room Inaccessibility)      17 steps

Search tools support both unconstrained semantic search and per-doc-type
filtering. New `search_by_setpoint_violation` maps a parameter name
straight to the steps that monitor it (via SETPOINT_INDEX).
"""
from typing import Any, Dict, List, Optional

import chromadb

from npp_agent.ingest.procedures_data import (
    ALL_PROCEDURES, DOC_META, ACTION_TO_STEP_ID, SETPOINT_INDEX,
)
from npp_agent.ingest.chunking import collect_aux_chunks
from npp_agent.ingest.lexical import TfIdfIndex, rrf_fuse


COLLECTION_NAME = "procedures_v5_hybrid"


# Korean expansion keywords per English domain term — helps the default
# English-leaning embedder (MiniLM) match Korean queries.
_KW_EXPAND = {
    # generic operating-procedure terms
    "manual_reactor_trip":   "원자로 수동 트립 reactor trip",
    "trip_turbine":          "터빈 트립 turbine trip stop valves",
    "actuate_si":            "안전주입 SI 작동 safety injection actuation",
    "trip_mfw":              "주급수 격리 펌프 정지 main feedwater isolation MFW",
    "start_afw":             "보조급수 펌프 기동 auxiliary feedwater AFW MDAFW TDAFW",
    "start_tdafw":           "터빈 구동 보조급수 펌프 turbine-driven AFW TDAFW",
    "close_msiv":            "주증기 격리밸브 차단 main steam isolation MSIV",
    "open_msiv":             "주증기 격리밸브 개방 MSIV open",
    "set_atm_dump":          "대기증기방출 setpoint atmospheric steam dump",
    "align_charging_to_rwst":"충전펌프 흡입 RWST 정렬 charging pump suction",
    "evacuate_control_room": "주제어실 철수 통보 control room evacuation CAS notify",
    "set_pzr_heater":        "가압기 히터 기동 pressurizer heater energize",
    "set_pzr_spray":         "가압기 살수 밸브 pressurizer spray",
    "set_rhr_pump_off":      "RHR 펌프 정지 residual heat removal stop",
    "start_rcp":             "RCP 기동 reactor coolant pump",
    "set_sg_level_target":   "증기발생기 SG 수위 steam generator level",
    "monitor_steam_bubble":  "증기방울 형성 steam bubble pressurizer",
    "wait_temp":             "RCS 온도 도달 자연 가열 RCS temperature",
}


def _korean_kw(step: dict) -> str:
    """Return a small bag of Korean+English domain keywords for embedding."""
    bits = []
    if step.get("expected_action") in _KW_EXPAND:
        bits.append(_KW_EXPAND[step["expected_action"]])
    # Each parameter spelled with a friendly Korean alias for matching
    for p in step.get("parameters", []):
        alias = {
            "T_RCS_avg_F":         "RCS 평균 온도",
            "P_RCS_psig":          "RCS 압력",
            "PZR_level_pct":       "가압기 수위",
            "PZR_temp_F":          "가압기 온도",
            "PZR_heater_on":       "가압기 히터",
            "RHR_pump_on":         "잔열제거 펌프 RHR",
            "RCP_running":         "원자로 냉각재 펌프 RCP",
            "MSIV_open":           "주증기 격리밸브 MSIV",
            "SG_level_NR_pct":     "증기발생기 협역 수위",
            "afw_pumps_running":   "보조급수 펌프 AFW",
            "afw_flow_total_gpm":  "보조급수 총 유량",
            "mfw_pump_running":    "주급수 펌프 MFW",
            "si_signal":           "안전주입 신호 SI",
            "si_pumps_running":    "안전주입 펌프 SI 운전",
            "reactor_tripped":     "원자로 트립",
            "turbine_tripped":     "터빈 트립",
            "ci_signal":           "격납건물 격리 CI",
            "cvi_signal":          "격납건물 환기 격리 CVI",
            "P_steam_psig":        "증기 압력 steamline",
            "P_cnmt_psig":         "격납건물 압력 CNMT",
            "heatup_rate_F_per_hr":"가열률 heatup rate",
            "seal_inj_flow_gpm":   "RCP 봉수 주입 유량",
            "control_room_evacuated":"주제어실 철수",
            "charging_suction_rwst":"충전펌프 흡입 RWST",
        }.get(p)
        if alias:
            bits.append(alias)
    return " · ".join(bits)


class ProcedureMCP:
    SERVER_NAME = "procedure-mcp"

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.steps_by_id = {s["id"]: s for s in ALL_PROCEDURES}
        self.lexical = TfIdfIndex()  # parallel keyword index
        self._docs_by_id: dict[str, str] = {}  # for B/C tier doc text recall
        self._load()

    # ── ingestion (3-tier) ───────────────────────
    def _load(self):
        if self.collection.count() > 0:
            return
        # Tier A — curated step chunks
        # Embed title 3× and Korean keywords for better Korean→English matching
        # with the default English-leaning embedder.
        a_ids = [s["id"] for s in ALL_PROCEDURES]
        a_docs = [
            (f"한국어 제목: {s['title']}. {s['title']}. {s['title']}.\n"
             f"[{s['id']}] [{s['doc_type']}] {s['title']}\n"
             f"{_korean_kw(s)}\n"
             f"{s['text']}\n"
             f"PARAMETERS: {', '.join(s['parameters']) or 'n/a'}\n"
             f"SETPOINTS: {s['setpoints']}\n"
             f"CAUTIONS: {' | '.join(s['cautions']) or 'none'}")
            for s in ALL_PROCEDURES
        ]
        a_metas = [
            {"tier": "A", "doc_type": s["doc_type"],
             "step_no": s["step_no"], "title": s["title"],
             "expected_action": s["expected_action"]}
            for s in ALL_PROCEDURES
        ]
        self.collection.add(documents=a_docs, ids=a_ids, metadatas=a_metas)
        # mirror to lexical
        self.lexical.add(a_docs, a_ids, a_metas)
        for i, d in zip(a_ids, a_docs):
            self._docs_by_id[i] = d

        # Tier B + C — body sliding windows + CAUTION blocks from raw PDF text
        try:
            aux = collect_aux_chunks()
        except Exception as e:
            print(f"[procedure] aux chunk load skipped: {e}")
            aux = []
        if aux:
            self.collection.add(
                documents=[c["document"] for c in aux],
                ids=[c["id"] for c in aux],
                metadatas=[c["metadata"] for c in aux],
            )
            self.lexical.add([c["document"] for c in aux],
                             [c["id"] for c in aux],
                             [c["metadata"] for c in aux])
            for c in aux:
                self._docs_by_id[c["id"]] = c["document"]

        counts = {"A": 0, "B": 0, "C": 0}
        for s in ALL_PROCEDURES:
            counts["A"] += 1
        for c in aux:
            counts[c["metadata"]["tier"]] += 1
        print(f"[procedure] 3-tier loaded — A(step)={counts['A']} "
              f"B(body)={counts['B']} C(caution)={counts['C']} "
              f"(total {self.collection.count()})  + lexical index")

    # ── tool catalog ─────────────────────────────
    def list_tools(self):
        return [
            {"name": "search_procedure",
             "description": "상황 설명으로 절차서 RAG 검색 (top-k). "
                            "doc_type='EOP'|'AOP'|'GOP', tier='A'(step)|'B'(body)|'C'(caution) 필터 지원"},
            {"name": "get_step",
             "description": "step ID로 단일 절차 조회 (예: EOP-E0-1, AOP-10-3, GOP-A-4)"},
            {"name": "get_next_step",
             "description": "현재 step의 다음 step 반환"},
            {"name": "search_by_mode_transition",
             "description": "(GOP 전용) MODE 전환에 해당하는 절차들"},
            {"name": "get_cautions",
             "description": "특정 step의 CAUTION/NOTE 반환"},
            {"name": "get_step_by_action",
             "description": "expected_action으로 매핑된 step 반환"},
            {"name": "search_by_setpoint_violation",
             "description": "파라미터명(예: P_RCS_psig)으로 관련 절차 step 즉시 조회"},
            {"name": "list_documents",
             "description": "로드된 절차서 문서 목록 + 시나리오 ID"},
            {"name": "list_steps_in_doc",
             "description": "특정 doc_type의 모든 step (순서대로)"},
        ]

    # ── dispatcher ───────────────────────────────
    def call_tool(self, tool_name: str, args: Dict[str, Any]):
        if tool_name == "search_procedure":
            return self._search(args.get("situation", ""),
                                int(args.get("n_results", 4)),
                                args.get("doc_type"),
                                args.get("tier"))
        if tool_name == "get_step":
            return self._get(args.get("step_id", ""))
        if tool_name == "get_next_step":
            return self._next(args.get("step_id", ""))
        if tool_name == "search_by_mode_transition":
            return self._by_mode(int(args.get("from_mode", 5)),
                                 int(args.get("to_mode", 4)))
        if tool_name == "get_cautions":
            return self._cautions(args.get("step_id", ""))
        if tool_name == "get_step_by_action":
            return self._by_action(args.get("action", ""))
        if tool_name == "search_by_setpoint_violation":
            return self._by_setpoint(args.get("parameter", ""),
                                     args.get("value"))
        if tool_name == "list_documents":
            return {"documents": [
                {"doc_type": k, **v} for k, v in DOC_META.items()
            ]}
        if tool_name == "list_steps_in_doc":
            return self._list_in_doc(args.get("doc_type", ""))
        return {"error": f"알 수 없는 도구: {tool_name}"}

    # ── implementations ──────────────────────────
    def _search(self, situation: str, n: int, doc_type: Optional[str],
                tier: Optional[str] = None):
        n = max(1, min(n, max(20, len(ALL_PROCEDURES))))
        if not situation:
            return {"query": "", "found": 0, "procedures": []}

        # 1) vector search (chroma)
        kwargs = {"query_texts": [situation],
                  "n_results": min(n * 4, 20)}
        clauses = []
        if doc_type in ("EOP", "AOP", "GOP"):
            clauses.append({"doc_type": doc_type})
        if tier in ("A", "B", "C"):
            clauses.append({"tier": tier})
        if len(clauses) == 1:
            kwargs["where"] = clauses[0]
        elif len(clauses) > 1:
            kwargs["where"] = {"$and": clauses}
        vres = self.collection.query(**kwargs)
        v_ids = list(vres["ids"][0])
        v_metas = {sid: m for sid, m in zip(v_ids, vres["metadatas"][0])}

        # 2) lexical search (TF-IDF over Korean+English text)
        def filt(m):
            if doc_type in ("EOP", "AOP", "GOP") and m.get("doc_type") != doc_type:
                return False
            if tier in ("A", "B", "C") and m.get("tier") != tier:
                return False
            return True
        lex_hits = self.lexical.search(situation, n=min(n * 4, 20), filter_fn=filt)
        l_ids = [sid for sid, _, _ in lex_hits]
        l_metas = {sid: m for sid, m, _ in lex_hits}

        # 3) RRF fusion — vector + lexical
        fused = rrf_fuse([v_ids, l_ids])
        out: List[Dict] = []
        seen = set()
        for sid, score in fused:
            if sid in seen:
                continue
            seen.add(sid)
            meta = v_metas.get(sid) or l_metas.get(sid) or {}
            tier_v = meta.get("tier", "A")
            step = self.steps_by_id.get(sid, {})
            entry = {
                "id": sid,
                "tier": tier_v,
                "doc_type": meta.get("doc_type"),
                "title": meta.get("title"),
                "expected_action": meta.get("expected_action") or "",
                "rrf_score": round(score, 4),
            }
            if tier_v == "A":
                entry.update({
                    "step_no": meta.get("step_no"),
                    "text": step.get("text"),
                    "setpoints": step.get("setpoints", {}),
                    "cautions": step.get("cautions", []),
                })
            else:
                entry.update({"step_no": -1,
                              "text": (self._docs_by_id.get(sid) or "")[:600],
                              "setpoints": {}, "cautions": []})
            out.append(entry)
            if len(out) >= n:
                break
        return {"query": situation,
                "doc_type_filter": doc_type,
                "tier_filter": tier,
                "found": len(out), "procedures": out,
                "fusion": "vector+lexical (RRF)"}

    def _get(self, step_id: str):
        s = self.steps_by_id.get(step_id)
        return s if s else {"error": f"step not found: {step_id}"}

    def _next(self, step_id: str):
        s = self.steps_by_id.get(step_id)
        if not s:
            return {"error": f"step not found: {step_id}"}
        # Find next step in same doc with step_no = current+1
        for c in ALL_PROCEDURES:
            if c["doc_type"] == s["doc_type"] and c["step_no"] == s["step_no"] + 1:
                return c
        return {"end_of_section": True, "current": step_id}

    def _by_mode(self, from_mode: int, to_mode: int):
        # Currently only GOP has mode transitions modeled
        matches = [s for s in ALL_PROCEDURES
                   if s["doc_type"] == "GOP"]
        return {"from_mode": from_mode, "to_mode": to_mode,
                "found": len(matches),
                "procedures": [{"id": s["id"], "step_no": s["step_no"],
                                "title": s["title"]} for s in matches]}

    def _cautions(self, step_id: str):
        s = self.steps_by_id.get(step_id)
        if not s:
            return {"error": f"step not found: {step_id}"}
        return {"step_id": step_id, "cautions": s["cautions"]}

    def _by_action(self, action: str):
        sid = ACTION_TO_STEP_ID.get(action)
        if not sid:
            return {"error": f"no procedure mapped to action: {action}"}
        return self.steps_by_id[sid]

    def _by_setpoint(self, parameter: str, value):
        ids = SETPOINT_INDEX.get(parameter, [])
        steps = [self.steps_by_id[i] for i in ids if i in self.steps_by_id]
        return {
            "parameter": parameter,
            "value": value,
            "found": len(steps),
            "procedures": [{
                "id": s["id"], "doc_type": s["doc_type"],
                "step_no": s["step_no"], "title": s["title"],
                "setpoints": s["setpoints"],
            } for s in steps],
        }

    def _list_in_doc(self, doc_type: str):
        if doc_type not in ("EOP", "AOP", "GOP"):
            return {"error": "doc_type must be one of EOP/AOP/GOP"}
        steps = [s for s in ALL_PROCEDURES if s["doc_type"] == doc_type]
        return {
            "doc_type": doc_type,
            "doc_meta": DOC_META[doc_type],
            "step_count": len(steps),
            "steps": [{"id": s["id"], "step_no": s["step_no"],
                       "title": s["title"],
                       "expected_action": s["expected_action"]}
                      for s in steps],
        }
