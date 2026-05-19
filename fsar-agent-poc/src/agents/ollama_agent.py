"""System B: EXAONE-3.5-7.8B via Ollama 로컬 실행."""
from __future__ import annotations

from typing import Any

import ollama

from src.agents.base_agent import FSARAgent
from src.config import ollama_host


class OllamaAgent(FSARAgent):
    SYSTEM_KEY = "system_b"

    def __init__(self, prompt_name: str = "qa_ko.txt"):
        super().__init__(prompt_name=prompt_name)
        self._client = ollama.Client(host=ollama_host())
        self._model_name = self.sys_cfg["model"]

    def _generate(self, prompt: str) -> tuple[str, int, int, dict[str, Any]]:
        resp = self._client.generate(
            model=self._model_name,
            prompt=prompt,
            options={
                "temperature": self.gen_cfg["temperature"],
                "num_predict": self.gen_cfg["max_tokens"],
                "seed": self.gen_cfg.get("seed", 42),
            },
            stream=False,
        )
        text = (resp.get("response") or "").strip()
        p_tok = int(resp.get("prompt_eval_count") or 0)
        c_tok = int(resp.get("eval_count") or 0)
        return text, p_tok, c_tok, {"model": self._model_name}
