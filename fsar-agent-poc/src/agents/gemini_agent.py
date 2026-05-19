"""System A: Google Gemini (gemini-1.5-flash, 무료 티어)."""
from __future__ import annotations

from typing import Any

import google.generativeai as genai

from src.agents.base_agent import FSARAgent
from src.config import google_api_key


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

    def _generate(self, prompt: str) -> tuple[str, int, int, dict[str, Any]]:
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

    @staticmethod
    def _estimate_tokens(prompt: str, completion: str) -> tuple[int, int]:
        # 4 chars ≈ 1 token (fallback)
        return max(1, len(prompt) // 4), max(1, len(completion) // 4)
