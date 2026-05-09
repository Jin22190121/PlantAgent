"""LLM model factory.

Centralized model instantiation with provider switching, allowing the
LangGraph nodes to remain provider-agnostic.

Currently supported providers (PoC scope):
  • gemini  — Google AI Studio (default; uses GOOGLE_API_KEY)
  • groq    — Groq Cloud free tier (LLAMA_4 fallback)

Selection priority:
  1) explicit `provider` arg
  2) MODEL_PROVIDER env var
  3) default 'gemini'
"""
from __future__ import annotations
import os
from typing import Optional


def _gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
    )


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
    return ChatGroq(model=model_name, api_key=api_key, temperature=0.2)


_PROVIDERS = {"gemini": _gemini, "groq": _groq}


def get_chat_model(provider: Optional[str] = None):
    name = (provider or os.environ.get("MODEL_PROVIDER") or "gemini").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[name]()


def get_chat_model_with_fallback():
    """Primary model with optional fallback (only adds fallback if available)."""
    primary = get_chat_model()
    fallbacks = []
    primary_name = (os.environ.get("MODEL_PROVIDER") or "gemini").lower()
    for name in ["groq"]:
        if name == primary_name:
            continue
        try:
            fallbacks.append(_PROVIDERS[name]())
        except Exception:
            continue
    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary


def llm_status() -> dict:
    """Diagnostic info for /health endpoint."""
    available = []
    for name, fn in _PROVIDERS.items():
        try:
            fn()
            available.append(name)
        except Exception:
            pass
    return {
        "primary": (os.environ.get("MODEL_PROVIDER") or "gemini").lower(),
        "available": available,
        "any_available": bool(available),
    }
