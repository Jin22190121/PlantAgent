"""System A: Google Gemini (gemini-2.5-flash 등, 무료 티어).

무료 티어 RPM 한도(5/min)에 걸리는 429를 자동 재시도(retry_delay 파싱)한다.
"""
from __future__ import annotations

import re
import time
from typing import Any

import google.generativeai as genai

from src.agents.base_agent import FSARAgent
from src.config import google_api_key

_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{[^}]*seconds:\s*(\d+)", re.S)


class GeminiAgent(FSARAgent):
    SYSTEM_KEY = "system_a"

    def __init__(self, prompt_name: str = "qa_ko.txt"):
        super().__init__(prompt_name=prompt_name)
        genai.configure(api_key=google_api_key())
        self._model = genai.GenerativeModel(
            self.sys_cfg["model"],
            generation_config={
                "temperature": self.gen_cfg["temperature"],
                "max_output_tokens": self.gen_cfg["max_tokens"],
            },
        )
        self._max_retries = int(self.sys_cfg.get("max_retries", 5))

    def _generate(self, prompt: str) -> tuple[str, int, int, dict[str, Any]]:
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._model.generate_content(prompt)
                text = (resp.text or "").strip() if hasattr(resp, "text") else ""
                if not text:
                    try:
                        text = resp.candidates[0].content.parts[0].text.strip()
                    except Exception:
                        text = ""

                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    p_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                    c_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
                else:
                    p_tok, c_tok = self._estimate_tokens(prompt, text)

                return text, p_tok, c_tok, {"model": self.sys_cfg["model"]}
            except Exception as e:
                last_err = e
                msg = str(e)
                is_rate = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
                if not is_rate:
                    raise

                # 일일 한도(RPD) 초과는 재시도해도 무의미하므로 즉시 중단
                if "PerDay" in msg or "RequestsPerDay" in msg:
                    print(
                        "  [429-RPD] 일일 한도(RPD) 초과 — 재시도 의미 없음. "
                        "내일까지 대기하거나 settings.yaml의 model을 다른 모델로 교체하세요."
                    )
                    raise RuntimeError("RPD_EXHAUSTED: " + msg) from e

                wait = self._parse_retry_delay(msg) or min(60, 2 ** attempt * 5)
                wait += 1  # 안전 마진
                print(
                    f"  [429] attempt {attempt}/{self._max_retries} — {wait}s 대기 후 재시도"
                )
                time.sleep(wait)
        raise last_err if last_err else RuntimeError("Gemini 호출 실패 — 알 수 없는 원인")

    @staticmethod
    def _parse_retry_delay(msg: str) -> int | None:
        m = _RETRY_DELAY_RE.search(msg)
        return int(m.group(1)) if m else None

    @staticmethod
    def _estimate_tokens(prompt: str, completion: str) -> tuple[int, int]:
        return max(1, len(prompt) // 4), max(1, len(completion) // 4)
