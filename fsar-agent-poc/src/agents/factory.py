"""에이전트 팩토리 — UI/평가 공통 진입점."""
from __future__ import annotations

from src.agents.base_agent import FSARAgent

SYSTEMS = ("A", "B")


def get_agent(system: str, prompt_name: str = "qa_ko.txt") -> FSARAgent:
    s = system.upper()
    if s == "A":
        from src.agents.gemini_agent import GeminiAgent
        return GeminiAgent(prompt_name=prompt_name)
    if s == "B":
        from src.agents.ollama_agent import OllamaAgent
        return OllamaAgent(prompt_name=prompt_name)
    raise ValueError(f"unknown system: {system!r} (expected 'A' or 'B')")
