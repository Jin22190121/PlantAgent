"""Procedure MCP — RAG search over Appendix 19-1 §A normal-ops steps.

Replaces the original emergency-procedure (E-0/E-1/ECA-0.0) collection.
Uses ChromaDB for semantic search and a setpoint lookup for fast direct
addressing.
"""
from typing import Any, Dict, List, Optional

import chromadb

from npp_agent.ingest.procedures_data import APPENDIX_19_1_A, ACTION_TO_STEP_ID


COLLECTION_NAME = "normal_procedures_app19_1_a"


class ProcedureMCP:
    SERVER_NAME = "procedure-mcp"

    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )
        self.steps_by_id = {s["id"]: s for s in APPENDIX_19_1_A}
        self._load()

    # ── ingestion ────────────────────────────────
    def _load(self):
        if self.collection.count() > 0:
            return
        ids = [s["id"] for s in APPENDIX_19_1_A]
        docs = [
            f"[{s['id']}] {s['title']}\n{s['text']}\n"
            f"PARAMETERS: {', '.join(s['parameters']) or 'n/a'}\n"
            f"SETPOINTS: {s['setpoints']}\n"
            f"CAUTIONS: {' | '.join(s['cautions']) or 'none'}"
            for s in APPENDIX_19_1_A
        ]
        metas = [
            {
                "step_no": s["step_no"],
                "mode_from": s["mode_from"],
                "mode_to": s["mode_to"],
                "title": s["title"],
                "expected_action": s["expected_action"],
            }
            for s in APPENDIX_19_1_A
        ]
        self.collection.add(documents=docs, ids=ids, metadatas=metas)
        print(f"[procedure] loaded {self.collection.count()} normal-ops steps")

    # ── tool catalog ─────────────────────────────
    def list_tools(self):
        return [
            {"name": "search_procedure",
             "description": "상황 설명으로 절차서 RAG 검색 (top-k)"},
            {"name": "get_step",
             "description": "step ID로 단일 절차 step 조회"},
            {"name": "get_next_step",
             "description": "현재 step 다음 step 반환"},
            {"name": "search_by_mode_transition",
             "description": "MODE 전환(from→to)에 해당하는 절차들 반환"},
            {"name": "get_cautions",
             "description": "특정 step의 CAUTION/NOTE 반환"},
            {"name": "get_step_by_action",
             "description": "expected_action(예: set_pzr_heater)에 매핑된 step 반환"},
        ]

    # ── dispatcher ───────────────────────────────
    def call_tool(self, tool_name: str, args: Dict[str, Any]):
        if tool_name == "search_procedure":
            return self._search(args.get("situation", ""),
                                int(args.get("n_results", 3)))
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
        return {"error": f"알 수 없는 도구: {tool_name}"}

    # ── implementations ──────────────────────────
    def _search(self, situation: str, n: int):
        n = max(1, min(n, len(APPENDIX_19_1_A)))
        if not situation:
            return {"query": "", "found": 0, "procedures": []}
        res = self.collection.query(query_texts=[situation], n_results=n)
        out: List[Dict] = []
        for i, doc in enumerate(res["documents"][0]):
            sid = res["ids"][0][i]
            meta = res["metadatas"][0][i]
            step = self.steps_by_id.get(sid, {})
            out.append({
                "id": sid,
                "step_no": meta.get("step_no"),
                "title": meta.get("title"),
                "expected_action": meta.get("expected_action"),
                "text": step.get("text"),
                "setpoints": step.get("setpoints", {}),
            })
        return {"query": situation, "found": len(out), "procedures": out}

    def _get(self, step_id: str):
        s = self.steps_by_id.get(step_id)
        if not s:
            return {"error": f"step not found: {step_id}"}
        return s

    def _next(self, step_id: str):
        s = self.steps_by_id.get(step_id)
        if not s:
            return {"error": f"step not found: {step_id}"}
        nxt = f"App19-1-A-{s['step_no'] + 1}"
        return self.steps_by_id.get(nxt, {"end_of_section": True})

    def _by_mode(self, from_mode: int, to_mode: int):
        matches = [s for s in APPENDIX_19_1_A
                   if s["mode_from"] == from_mode and s["mode_to"] == to_mode]
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
