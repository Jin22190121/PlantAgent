"""공통 임베딩 — 두 시스템이 동일하게 사용 (settings.yaml의 embedding 섹션)."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import load_settings


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
