"""LLM model factory — Gemini multi-model rotation.

PoC strategy:
  Each Google AI Studio free-tier model has its OWN quota pool:
    gemini-2.5-flash      — 10 RPM / 250 RPD (top quality)
    gemini-2.5-flash-lite — ~15 RPM / 1000 RPD
    gemini-2.0-flash      — 10 RPM / 1500 RPD
    gemini-2.0-flash-lite — ~30 RPM / 1500 RPD
    gemini-1.5-flash      — separate quota pool (legacy)
    gemini-1.5-flash-8b   — smallest, fastest, high RPM

  We chain them with `.with_fallbacks(...)` so when the primary 429/503s,
  the request automatically re-routes to the next model in the list.
  Each per-model retry is kept SHORT (2 attempts) so a fully-saturated
  model falls through to the next within a few seconds.

  GEMINI_MODELS env can override the rotation order (comma-separated).
  Optional Groq is appended at the very end if GROQ_API_KEY is set.
"""
from __future__ import annotations
import os
from typing import Optional


# Default rotation — top-quality → broadest-quota
DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def _wrap_retry(model, attempts: int = 2):
    """Short LangChain retry — exhausts fast so chain falls through quickly."""
    try:
        return model.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=attempts,
        )
    except Exception:
        return model


def _make_gemini(model_name: str):
    """Create one Gemini chat model wrapped with short retry."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    return _wrap_retry(ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_retries=1,   # SDK-side; LC-side retry handles transient
    ), attempts=2)


def _make_groq():
    """Optional non-Google last-resort fallback."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise RuntimeError("langchain-groq 미설치") from e
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
    model_name = os.environ.get("GROQ_MODEL",
                                 "meta-llama/llama-4-scout-17b-16e-instruct")
    return _wrap_retry(ChatGroq(model=model_name, api_key=api_key,
                                 temperature=0.2), attempts=2)


def _gemini_model_list() -> list[str]:
    """Resolve rotation order — env override > default list."""
    csv = os.environ.get("GEMINI_MODELS")
    if csv:
        names = [m.strip() for m in csv.split(",") if m.strip()]
        if names:
            return names
    # Single-model override (legacy) — wrap into singleton list
    legacy = os.environ.get("GEMINI_MODEL")
    if legacy:
        return [legacy]
    return DEFAULT_GEMINI_MODELS


def _build_chain():
    """Build the primary + fallbacks chain.
    Order: each Gemini model in rotation, then (optional) Groq."""
    names = _gemini_model_list()
    primary = None
    fallbacks = []
    chain_log = []

    for name in names:
        try:
            m = _make_gemini(name)
            chain_log.append(name)
            if primary is None:
                primary = m
            else:
                fallbacks.append(m)
        except Exception as e:
            chain_log.append(f"{name} (skip: {type(e).__name__})")

    # Optional Groq last fallback
    try:
        fallbacks.append(_make_groq())
        chain_log.append("groq")
    except Exception:
        pass

    if primary is None:
        raise RuntimeError("No usable Gemini model — set GOOGLE_API_KEY.")

    if fallbacks:
        try:
            return primary.with_fallbacks(fallbacks), chain_log
        except Exception:
            return primary, chain_log
    return primary, chain_log


# Lazy singleton — built once on first request to save startup time
_cached_chain = None
_cached_log: list[str] = []


def get_chat_model(provider: Optional[str] = None):
    """Backward-compat entry point — returns the rotating chain regardless
    of `provider` (the rotation handles fallback internally)."""
    global _cached_chain, _cached_log
    if _cached_chain is None:
        _cached_chain, _cached_log = _build_chain()
        print(f"[llm] rotation chain: {' → '.join(_cached_log)}")
    return _cached_chain


def get_chat_model_with_fallback():
    """Alias — chain is already configured with rotation+fallback."""
    return get_chat_model()


def llm_status() -> dict:
    """Diagnostic info for /health endpoint."""
    available = []
    errors = {}
    for name in _gemini_model_list():
        try:
            _make_gemini(name)
            available.append(name)
        except Exception as e:
            errors[name] = str(e)
    try:
        _make_groq()
        available.append("groq")
    except Exception as e:
        errors["groq"] = str(e)
    return {
        "rotation": _gemini_model_list(),
        "available": available,
        "any_available": bool(available),
        "errors": errors,
        "fallback_active": len(available) > 1,
        "chain": _cached_log or None,
    }
