"""LangGraph state-machine builder.

Compiles AgentState graph with:
  • SqliteSaver checkpointer (data/checkpoints/agent.sqlite) for HITL session
    resilience across interrupts/disconnects.
  • interrupt() native HITL via approval_gate node.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable

from langgraph.graph import StateGraph, START, END

from npp_agent.llm import get_chat_model, get_chat_model_with_fallback
from npp_agent.observability import langfuse_callbacks, langfuse_status
from .state import AgentState
from .nodes import (
    make_assess_state,
    make_retrieve_procedure,
    make_plan_action,
    approval_gate,
    make_execute_action,
    make_verify_outcome,
    make_log_step,
    respond,
)


def _make_checkpointer():
    """Async-compatible checkpointer.

    The web server uses `graph.astream(...)` which requires an async
    checkpointer. Order of preference:
      1) AsyncSqliteSaver (persistent across restarts; needs aiosqlite)
      2) MemorySaver (in-process only — interrupts still work within session,
         lost on restart)

    The sync SqliteSaver is NOT compatible with astream and is skipped.
    """
    Path("data/checkpoints").mkdir(parents=True, exist_ok=True)
    db_path = "data/checkpoints/agent.sqlite"
    # 1) Try AsyncSqliteSaver (requires aiosqlite)
    try:
        import aiosqlite  # noqa: F401  — ensure backend present
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import asyncio
        async def _open():
            conn = await aiosqlite.connect(db_path)
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            return saver
        try:
            asyncio.get_running_loop()
            # We're inside an event loop already (rare during module import);
            # fall through to MemorySaver to avoid nested-loop issues.
            raise RuntimeError("running loop detected — defer to MemorySaver")
        except RuntimeError:
            saver = asyncio.run(_open())
            print(f"[graph] AsyncSqliteSaver ready at {db_path}")
            return saver
    except Exception as e:
        print(f"[graph] AsyncSqliteSaver unavailable ({e}); using MemorySaver")
    # 2) Fallback — works fully async, no persistence
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def _route_after_plan(state: AgentState) -> str:
    """If planner produced an actionable proposal → approval_gate. Else end."""
    pa = state.get("proposed_action") or {}
    if state.get("done") and not pa.get("tool"):
        return "respond"
    if pa.get("tool"):
        return "approval_gate"
    return "respond"


def _route_after_approval(state: AgentState) -> str:
    if state.get("approval_status") in ("approved", "modified"):
        return "execute_action"
    return "respond"


def build_graph(router, thread_id_resolver: Callable[[dict], str] | None = None):
    """Compile the agent graph. Returns the executable graph."""
    # Attach LangFuse callback if env-configured
    callbacks = langfuse_callbacks()
    # Auto-fallback when secondary providers' env keys are present
    try:
        llm = get_chat_model_with_fallback()
    except Exception:
        llm = get_chat_model()
    if callbacks:
        try:
            llm = llm.with_config({"callbacks": callbacks})
        except Exception:
            pass
    status = langfuse_status()
    print(f"[langfuse] {'enabled' if status['enabled'] else 'disabled'}"
          f" — {status.get('error') or status.get('host')}")
    if thread_id_resolver is None:
        thread_id_resolver = lambda s: s.get("_session_id", "default")

    g = StateGraph(AgentState)
    g.add_node("assess_state",       make_assess_state(router))
    g.add_node("retrieve_procedure", make_retrieve_procedure(router))
    g.add_node("plan_action",        make_plan_action(llm))
    g.add_node("approval_gate",      approval_gate)
    g.add_node("execute_action",     make_execute_action(router))
    g.add_node("verify_outcome",     make_verify_outcome(router))
    g.add_node("log_step",           make_log_step(router, thread_id_resolver))
    g.add_node("respond",            respond)

    g.add_edge(START,                "assess_state")
    g.add_edge("assess_state",       "retrieve_procedure")
    g.add_edge("retrieve_procedure", "plan_action")
    g.add_conditional_edges("plan_action",  _route_after_plan,
                             {"approval_gate": "approval_gate",
                              "respond": "respond"})
    g.add_conditional_edges("approval_gate", _route_after_approval,
                             {"execute_action": "execute_action",
                              "respond": "respond"})
    g.add_edge("execute_action",     "verify_outcome")
    g.add_edge("verify_outcome",     "log_step")
    g.add_edge("log_step",           "respond")
    g.add_edge("respond",            END)

    checkpointer = _make_checkpointer()
    return g.compile(checkpointer=checkpointer)
