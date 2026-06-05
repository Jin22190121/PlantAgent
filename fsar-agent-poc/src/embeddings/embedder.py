"""공통 임베딩 — 두 시스템이 동일하게 사용 (settings.yaml의 embedding 섹션).

BGE-M3는 처음 한 번 다운로드된 후 로컬 캐시(~/.cache/huggingface)에 영구 저장됨.
sentence-transformers가 매번 HF Hub에 업데이트 확인을 위한 HEAD 요청을 보내는데,
무료 비인증 호출은 분당 한도가 있어 429를 자주 발생시킨다.
PoC는 캐시된 모델로 충분하므로 오프라인 모드를 강제한다.
HF_TOKEN 환경변수가 있으면 그 토큰으로 인증되어 한도가 풀린다.
"""
from __future__ import annotations

import os

# sentence_transformers를 import하기 전에 환경변수를 설정해야 함
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from functools import lru_cache  # noqa: E402
from typing import Sequence  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import load_settings  # noqa: E402


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    cfg = load_settings()["embedding"]
    model = SentenceTransformer(cfg["model_name"], device=cfg.get("device", "cpu"))
    return model


def embed(texts: Sequence[str]) -> list[list[float]]:
    cfg = load_settings()["embedding"]
    model = get_embedder()
    vecs = model.encode(
        list(texts),
        normalize_embeddings=cfg.get("normalize", True),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.tolist()
