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


COLLECTION_NAME = "procedures_v2_eop_aop_gop"


class ProcedureMCP:
    SERVER_NAME = "procedure-mcp"

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.steps_by_id = {s["id"]: s for s in ALL_PROCEDURES}
        self._load()

    # ── ingestion ────────────────────────────────
    def _load(self):
        if self.collection.count() > 0:
            return
        ids = [s["id"] for s in ALL_PROCEDURES]
        docs = [
            f"[{s['id']}] [{s['doc_type']}] {s['title']}\n{s['text']}\n"
            f"PARAMETERS: {', '.join(s['parameters']) or 'n/a'}\n"
            f"SETPOINTS: {s['setpoints']}\n"
            f"CAUTIONS: {' | '.join(s['cautions']) or 'none'}"
            for s in ALL_PROCEDURES
        ]
        metas = [
            {
                "doc_type":    s["doc_type"],
                "step_no":     s["step_no"],
                "title":       s["title"],
                "expected_action": s["expected_action"],
            }
            for s in ALL_PROCEDURES
        ]
        self.collection.add(documents=docs, ids=ids, metadatas=metas)
        counts = {"GOP": 0, "EOP": 0, "AOP": 0}
        for s in ALL_PROCEDURES:
            counts[s["doc_type"]] += 1
        print(f"[procedure] loaded {self.collection.count()} steps "
              f"(GOP={counts['GOP']} EOP={counts['EOP']} AOP={counts['AOP']})")

    # ── tool catalog ─────────────────────────────
    def list_tools(self):
        return [
            {"name": "search_procedure",
             "description": "상황 설명으로 절차서 RAG 검색 (top-k). "
                            "doc_type='EOP'|'AOP'|'GOP' 으로 필터 가능"},
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
                                args.get("doc_type"))
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
    def _search(self, situation: str, n: int, doc_type: Optional[str]):
        n = max(1, min(n, len(ALL_PROCEDURES)))
        if not situation:
            return {"query": "", "found": 0, "procedures": []}
        kwargs = {"query_texts": [situation], "n_results": n}
        if doc_type in ("EOP", "AOP", "GOP"):
            kwargs["where"] = {"doc_type": doc_type}
        res = self.collection.query(**kwargs)
        out: List[Dict] = []
        for i, doc in enumerate(res["documents"][0]):
            sid = res["ids"][0][i]
            meta = res["metadatas"][0][i]
            step = self.steps_by_id.get(sid, {})
            out.append({
                "id": sid,
                "doc_type": meta.get("doc_type"),
                "step_no": meta.get("step_no"),
                "title": meta.get("title"),
                "expected_action": meta.get("expected_action"),
                "text": step.get("text"),
                "setpoints": step.get("setpoints", {}),
                "cautions": step.get("cautions", []),
            })
        return {"query": situation,
                "doc_type_filter": doc_type,
                "found": len(out), "procedures": out}

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
