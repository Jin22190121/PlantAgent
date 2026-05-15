"""FastAPI web server for the NPP AI Agent PoC (Phase 1: LangGraph + HITL).

Endpoints:
  GET  /                  → static UI
  GET  /demo              → static/demo.html
  GET  /plant/state       → current simulator snapshot (REST)
  GET  /plant/stream      → live state SSE (every ~1 s)
  POST /sim/action        → direct operator action (UI button — bypasses
                            HITL gate as it is itself an operator decision)
  POST /sim/control       → engine control (pause/resume/reset/speed)
  POST /chat/stream       → AI agent turn via LangGraph (SSE; pauses on
                            interrupt for HITL approval)
  POST /chat/approve      → resume an interrupted graph with approve/reject
  DELETE /chat/{sid}      → reset chat checkpoint
  GET  /health            → status
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
# Search parent directories for .env (handy when cwd is not the repo root)
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)
    print(f"[env] loaded {_dotenv_path}")
else:
    load_dotenv(override=False)
    print("[env] no .env found (relying on process environment)")

# Boot-time diagnostic
print(f"[env] GOOGLE_API_KEY {'set' if os.environ.get('GOOGLE_API_KEY') else 'MISSING'} "
      f"· GROQ_API_KEY {'set' if os.environ.get('GROQ_API_KEY') else 'MISSING'}")

from npp_agent.sim import SimulationEngine
from npp_agent.mcp_servers.simulator import SimulatorMCP
from npp_agent.mcp_servers.procedure import ProcedureMCP
from npp_agent.mcp_servers.training  import TrainingMCP
from npp_agent.mcp_servers.router    import MCPRouter
from npp_agent.llm import llm_status
from npp_agent.observability import langfuse_status

# ── Engine + MCP routing (shared singleton) ─────────────────
engine = SimulationEngine(sim_speed=60.0)
simulator = SimulatorMCP(engine=engine)
procedure = ProcedureMCP()
training  = TrainingMCP()
router    = MCPRouter([simulator, procedure, training])

# ── LangGraph executor (lazy: built on first chat) ──────────
_graph = None
_graph_error: Optional[str] = None


def _get_graph():
    global _graph, _graph_error
    if _graph is not None:
        return _graph
    try:
        from npp_agent.graph import build_graph
        _graph = build_graph(router)
        return _graph
    except Exception as e:
        _graph_error = f"{type(e).__name__}: {e}"
        return None


# ── pending interrupt state per session ─────────────────────
# Maps session_id (= LangGraph thread_id) → True when waiting for approval
_awaiting_approval: dict[str, dict] = {}

# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="NPP AI Agent PoC — Phase 1 (LangGraph + HITL)")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def _start_sim():
    await engine.start()


@app.on_event("shutdown")
async def _stop_sim():
    await engine.stop()


# ── pages ───────────────────────────────────────────────────
@app.get("/")
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/demo")
async def demo_page():
    with open("static/demo.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/gop-demo")
async def gop_demo_page():
    with open("static/gop_demo.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── plant state ────────────────────────────────────────────
@app.get("/plant/state")
async def plant_state():
    return engine.get_state()


@app.get("/plant/stream")
async def plant_stream():
    async def gen() -> AsyncIterator[bytes]:
        try:
            while True:
                payload = engine.get_state()
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            # Client disconnected — clean exit, do not propagate as ERROR
            return
        except Exception as e:  # pragma: no cover
            yield f"event: error\ndata: {e}\n\n".encode()
            return
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# Silence noisy favicon 404 (browsers always probe this URL)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse({}, status_code=204)


# ── sim control ────────────────────────────────────────────
class ActionReq(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@app.post("/sim/action")
async def sim_action(req: ActionReq):
    # Direct UI action — operator press is itself the approval, bypass tier.
    result = router.call(req.tool, req.args, source="operator_ui")
    return JSONResponse(result)


class ControlReq(BaseModel):
    op: str
    value: Any = None


@app.post("/sim/control")
async def sim_control(req: ControlReq):
    if req.op == "pause":
        engine.pause(); return {"ok": True, "paused": True}
    if req.op == "resume":
        engine.resume(); return {"ok": True, "paused": False}
    if req.op == "reset":
        return engine.reset(req.value or "scenario_gop_heatup")
    if req.op == "reset_scenario":
        return engine.reset(req.value or "scenario_gop_heatup")
    if req.op == "speed":
        return engine.set_sim_speed(float(req.value or 60))
    raise HTTPException(400, f"unknown op: {req.op}")


# ── chat (LangGraph) ────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str


def _evt(t: str, **payload) -> str:
    return f"data: {json.dumps({'type': t, **payload}, ensure_ascii=False)}\n\n"


async def _stream_graph(graph, inputs, config) -> AsyncIterator[str]:
    """Run graph.astream and translate node events to SSE strings."""
    interrupted = False
    session_id = config["configurable"]["thread_id"]
    try:
        async for chunk in graph.astream(inputs, config):
            # chunk is dict: {node_name: state_update} or {"__interrupt__": (...)}
            if "__interrupt__" in chunk:
                ints = chunk["__interrupt__"]
                if ints:
                    payload = ints[0].value if hasattr(ints[0], "value") else ints[0]
                    _awaiting_approval[session_id] = payload
                    yield _evt("approval_request", **payload)
                    interrupted = True
                break
            for node_name, update in chunk.items():
                yield _evt("graph_node", node=node_name, update=_safe(update))
                # Surface explicitly-acknowledged completed steps as a structured
                # SSE event so the UI marks the tracker without text heuristics.
                if isinstance(update, dict):
                    # Top-level reducer-accumulated completed step IDs
                    top_cs = update.get("completed_step_ids")
                    if top_cs:
                        yield _evt("step_done", step_ids=list(top_cs))
                    # Legacy: nested in proposed_action (still emitted by plan_action)
                    pa = update.get("proposed_action") or {}
                    cs = pa.get("completed_step_ids") if isinstance(pa, dict) else None
                    if cs:
                        yield _evt("step_done", step_ids=list(cs))
                    # Active step indicator — UI highlights the row in the tracker
                    cur = update.get("current_step_id")
                    if cur:
                        yield _evt("step_active", step_id=cur)
                # Only emit the consolidated final_answer (from respond node).
                # Intermediate final_messages from earlier nodes are reduced
                # into final_answer via the `add` reducer in AgentState, so
                # emitting them here would cause UI duplication.
                if isinstance(update, dict) and update.get("final_answer"):
                    yield _evt("answer", text=update["final_answer"])
        if not interrupted:
            _awaiting_approval.pop(session_id, None)
    except asyncio.CancelledError:
        # Operator browser closed mid-stream — clean exit.
        return
    except Exception as e:
        yield _evt("error", text=f"graph error: {e}")


def _safe(obj):
    """Drop noisy/large fields from node updates before sending to UI."""
    if not isinstance(obj, dict):
        return str(obj)[:200]
    out = {}
    for k, v in obj.items():
        if k in ("plant_state", "alarms", "retrieved_procedures",
                 "last_tool_result", "procedure_history"):
            if isinstance(v, list):
                out[k] = f"<{len(v)} items>"
            elif isinstance(v, dict):
                out[k] = f"<dict {len(v)} keys>"
            else:
                out[k] = str(v)[:60]
        else:
            try:
                out[k] = v if isinstance(v, (str, int, float, bool)) else json.loads(json.dumps(v, default=str))
            except Exception:
                out[k] = str(v)[:200]
    return out


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Run a graph turn. If an interrupt fires, SSE emits an
    `approval_request` event and the graph remains paused awaiting
    POST /chat/approve."""
    graph = _get_graph()
    if graph is None:
        async def err():
            yield _evt("error",
                       text=f"AI 그래프 초기화 실패: {_graph_error}. "
                            f"GOOGLE_API_KEY 설정 후 서버 재시작 필요.")
        return StreamingResponse(err(), media_type="text/event-stream")

    config = {"configurable": {"thread_id": req.session_id}}

    # If session has a stale pending approval, clear it on a fresh chat call
    _awaiting_approval.pop(req.session_id, None)

    inputs = {
        "operator_input": req.message,
        "_session_id": req.session_id,
    }

    async def gen():
        yield _evt("status", text="🤖 AI 분석 시작 (LangGraph)...")
        async for s in _stream_graph(graph, inputs, config):
            yield s

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class ApprovalReq(BaseModel):
    session_id: str
    approved: bool
    reason: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None


@app.post("/chat/approve")
async def chat_approve(req: ApprovalReq):
    """Resume an interrupted graph with operator's decision."""
    graph = _get_graph()
    if graph is None:
        raise HTTPException(503, "graph not initialized")
    if req.session_id not in _awaiting_approval:
        raise HTTPException(409, "no pending approval for this session")

    from langgraph.types import Command
    config = {"configurable": {"thread_id": req.session_id}}
    payload = {"approved": req.approved, "reason": req.reason or ""}
    if req.modified_args is not None:
        payload["modified_args"] = req.modified_args

    async def gen():
        yield _evt("status", text=("✓ 승인 — 도구 실행 중..." if req.approved
                                   else "✗ 거부 — 재계획 중..."))
        async for s in _stream_graph(graph, Command(resume=payload), config):
            yield s

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    _awaiting_approval.pop(session_id, None)
    # Note: SqliteSaver checkpoints persist; clearing them requires deleting
    # the underlying thread state. For PoC we just reset awaiting state.
    return {"status": "session reset"}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "llm": llm_status(),
        "langfuse": langfuse_status(),
        "graph_ready": _graph is not None,
        "graph_error": _graph_error,
        "sim_running": engine._task is not None and not engine._task.done(),
        "pending_approvals": list(_awaiting_approval.keys()),
        "procedures": {"docs": ["GOP", "EOP", "AOP"],
                       "scenarios": list(__import__("npp_agent.sim.state",
                                                    fromlist=["SCENARIOS"]).SCENARIOS.keys())},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)
