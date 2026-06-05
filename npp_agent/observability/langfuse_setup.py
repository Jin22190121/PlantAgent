"""LangFuse observability — env-gated.

If LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set, return a CallbackHandler
to attach to LangChain/LangGraph invocations. Otherwise: noop. This way the
agent runs identically offline, and operators can opt in by exporting the
two env vars (and optionally LANGFUSE_HOST for self-hosted instances).
"""
from __future__ import annotations
import os

_HANDLER = None
_INIT_TRIED = False
_INIT_ERROR: str | None = None


def _try_init():
    """Lazily create a LangFuse handler if env is configured + lib installed."""
    global _HANDLER, _INIT_TRIED, _INIT_ERROR
    if _INIT_TRIED:
        return _HANDLER
    _INIT_TRIED = True
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        _INIT_ERROR = "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set"
        return None
    try:
        from langfuse.callback import CallbackHandler
    except Exception as e:
        _INIT_ERROR = f"langfuse not installed ({e}); pip install langfuse"
        return None
    try:
        host = os.environ.get("LANGFUSE_HOST")  # None → cloud, else self-host URL
        kwargs = {"public_key": pub, "secret_key": sec}
        if host:
            kwargs["host"] = host
        _HANDLER = CallbackHandler(**kwargs)
        return _HANDLER
    except Exception as e:  # pragma: no cover
        _INIT_ERROR = f"CallbackHandler init failed: {e}"
        return None


def langfuse_callbacks() -> list:
    """Return [] when disabled — safe to drop into config={'callbacks': [...]}"""
    h = _try_init()
    return [h] if h else []


def langfuse_status() -> dict:
    h = _try_init()
    return {
        "enabled": h is not None,
        "host": os.environ.get("LANGFUSE_HOST") or "cloud (default)",
        "error": _INIT_ERROR if h is None else None,
    }


def langfuse_event(name: str, **payload):
    """Emit an arbitrary trace event (best-effort — silent on failure)."""
    h = _try_init()
    if not h:
        return
    try:
        # API differs by version; use the most stable signature
        client = getattr(h, "langfuse", None) or h
        if hasattr(client, "trace"):
            t = client.trace(name=name, metadata=payload)
            if t and hasattr(t, "end"):
                t.end()
    except Exception:
        pass
