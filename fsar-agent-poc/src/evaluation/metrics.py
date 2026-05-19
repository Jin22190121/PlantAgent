"""한국어 형태소 기반 답변 메트릭 + 검색 메트릭."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from src.config import load_settings


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


@lru_cache(maxsize=1)
def _tokenizer():
    name = load_settings()["evaluation"]["tokenizer"].lower()
    try:
        from konlpy.tag import Mecab, Okt
    except ImportError as e:
        raise RuntimeError("KoNLPy가 설치되어 있지 않습니다. requirements.txt 확인.") from e
    if name == "mecab":
        return Mecab()
    return Okt()


def tokenize_ko(text: str) -> list[str]:
    text = _normalize(text)
    if not text:
        return []
    try:
        return [m for m in _tokenizer().morphs(text) if m.strip()]
    except Exception:
        # Mecab 미설치 등의 환경에서 fallback: 공백/문장부호 분리
        return [t for t in re.split(r"[^\w가-힣]+", text) if t]


# ── 답변 메트릭 ────────────────────────────────────────

def token_prf1(pred: str, gold: str) -> dict[str, float]:
    pred_toks = tokenize_ko(pred)
    gold_toks = tokenize_ko(gold)
    if not pred_toks and not gold_toks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_toks or not gold_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def keyword_hit_rate(pred: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    cfg = load_settings()["evaluation"]
    mode = cfg.get("keyword_match_mode", "substring")
    norm_pred = _normalize(pred)
    if mode == "morph_exact":
        toks = set(tokenize_ko(pred))
        hits = sum(1 for k in keywords if k in toks)
    else:
        hits = sum(1 for k in keywords if _normalize(k) in norm_pred)
    return hits / len(keywords)


# ── 검색 메트릭 ────────────────────────────────────────

def retrieval_pr(retrieved_pages: list[int], gold_pages: list[int]) -> dict[str, float]:
    if not retrieved_pages and not gold_pages:
        return {"recall@k": 1.0, "precision@k": 1.0}
    if not gold_pages:
        return {"recall@k": 1.0, "precision@k": 0.0}
    retrieved_set = set(retrieved_pages)
    gold_set = set(gold_pages)
    inter = retrieved_set & gold_set
    recall = len(inter) / len(gold_set) if gold_set else 0.0
    precision = len(inter) / len(retrieved_set) if retrieved_set else 0.0
    return {"recall@k": recall, "precision@k": precision}
