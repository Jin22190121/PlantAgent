"""RAG 에이전트 공통 추상 클래스 — 두 시스템은 _generate만 다르다."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.config import load_prompt, load_settings
from src.retrieval.retriever import (
    RetrievedChunk,
    format_context,
    search,
)


@dataclass
class AgentResponse:
    system: str
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def cited_pages(self) -> list[int]:
        return sorted({c.page for c in self.retrieved})

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "question": self.question,
            "answer": self.answer,
            "retrieved_ids": [c.id for c in self.retrieved],
            "retrieved_pages": [c.page for c in self.retrieved],
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class FSARAgent(ABC):
    """공통 RAG 흐름:
       질문 → 검색(공통) → 프롬프트 조립(공통) → LLM 생성(시스템별)."""

    SYSTEM_KEY = "system_a"  # override in subclass

    def __init__(self, prompt_name: str = "qa_ko.txt"):
        cfg = load_settings()
        self.cfg = cfg
        self.sys_cfg = cfg[self.SYSTEM_KEY]
        self.gen_cfg = cfg["generation"]
        self.prompt_template = load_prompt(prompt_name)

    def ask(self, question: str, top_k: int | None = None) -> AgentResponse:
        retrieved = search(question, top_k=top_k)
        context = format_context(retrieved)
        prompt = self.prompt_template.format(context=context, question=question)

        t0 = time.perf_counter()
        answer, prompt_tokens, completion_tokens, raw = self._generate(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cost = self._compute_cost(prompt_tokens, completion_tokens)
        return AgentResponse(
            system=self.sys_cfg["name"],
            question=question,
            answer=answer,
            retrieved=retrieved,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw=raw,
        )

    def _compute_cost(self, p_tok: int, c_tok: int) -> float:
        in_rate = self.sys_cfg.get("input_cost_per_1k_tokens", 0.0)
        out_rate = self.sys_cfg.get("output_cost_per_1k_tokens", 0.0)
        return (p_tok / 1000.0) * in_rate + (c_tok / 1000.0) * out_rate

    @abstractmethod
    def _generate(self, prompt: str) -> tuple[str, int, int, dict[str, Any]]:
        """반환: (answer_text, prompt_tokens, completion_tokens, raw_metadata)."""
