"""LLM model factory.

Centralized model instantiation with provider switching, allowing the
LangGraph nodes to remain provider-agnostic.

Currently supported providers (PoC scope):
  • gemini  — Google AI Studio (default; uses GOOGLE_API_KEY)
  • groq    — Groq Cloud free tier (LLAMA_4 fallback)

Resilience:
  • Each model is wrapped in retry-on-429/quota with exponential backoff
  • get_chat_model_with_fallback() chains primary→Groq automatically when
    GROQ_API_KEY is set
"""
from __future__ import annotations
import os
from typing import Optional


def _wrap_retry(model):
    """Enable LangChain's built-in retry for transient API errors.
    Tuned for Gemini 503 UNAVAILABLE / 429 quota / network blips.
    Falls back to bare model if the LC version doesn't support .with_retry()."""
    try:
        return model.with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=6,   # 1·2·4·8·16·32s with jitter ≈ ~60s total
        )
    except Exception:
        return model


def _gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return _wrap_retry(ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_retries=4,   # SDK-level retry on top of LangChain's
    ))


def _groq():
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise RuntimeError("langchain-groq 미설치: pip install langchain-groq") from e
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 환경변수가 설정되지 않았습니다.")
    model_name = os.environ.get("GROQ_MODEL",
                                 "meta-llama/llama-4-scout-17b-16e-instruct")
    return _wrap_retry(ChatGroq(model=model_name, api_key=api_key, temperature=0.2))


_PROVIDERS = {"gemini": _gemini, "groq": _groq}


def get_chat_model(provider: Optional[str] = None):
    name = (provider or os.environ.get("MODEL_PROVIDER") or "gemini").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()


def get_chat_model_with_fallback():
    """Primary model with optional fallback. When GROQ_API_KEY is set and
    primary is gemini (or vice versa), the secondary is chained via
    .with_fallbacks() so transient 429/quota on primary auto-routes."""
    primary_name = (os.environ.get("MODEL_PROVIDER") or "gemini").lower()
    primary = get_chat_model(primary_name)
    fallbacks = []
    for name in ("groq", "gemini"):
        if name == primary_name:
            continue
        try:
            fallbacks.append(_PROVIDERS[name]())
        except Exception:
            continue
    if fallbacks:
        try:
            return primary.with_fallbacks(fallbacks)
        except Exception:
            return primary
    return primary


def llm_status() -> dict:
    """Diagnostic info for /health endpoint."""
    available = []
    errors = {}
    for name, fn in _PROVIDERS.items():
        try:
            fn()
            available.append(name)
        except Exception as e:
            errors[name] = str(e)
    return {
        "primary": (os.environ.get("MODEL_PROVIDER") or "gemini").lower(),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "groq_model": os.environ.get("GROQ_MODEL",
                                      "meta-llama/llama-4-scout-17b-16e-instruct"),
        "available": available,
        "any_available": bool(available),
        "errors": errors,
        "fallback_active": len(available) > 1,
    }
