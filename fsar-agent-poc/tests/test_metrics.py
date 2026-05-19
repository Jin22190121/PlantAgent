"""평가 메트릭 단위 검증 — KoNLPy 없이 fallback tokenizer로도 동작 확인."""
from __future__ import annotations

import sys
from pathlib import Path

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (  # noqa: E402
    exact_match,
    keyword_hit_rate,
    retrieval_pr,
    token_prf1,
)


def test_exact_match():
    assert exact_match("APR1400 노형", "APR1400 노형") == 1.0
    assert exact_match("APR1400 노형", "APR1400 노형 차이") == 0.0


def test_keyword_hit_rate_substring():
    pred = "신고리 3,4호기는 APR1400 가압경수로이며 출력은 1400 MWe이다."
    assert keyword_hit_rate(pred, ["APR1400", "1400", "MWe"]) == 1.0
    assert keyword_hit_rate(pred, ["APR1400", "OPR1000"]) == 0.5


def test_token_prf1_identical():
    prf = token_prf1("신고리 3,4호기 정격출력 1400 MWe", "신고리 3,4호기 정격출력 1400 MWe")
    assert prf["f1"] >= 0.99


def test_token_prf1_partial():
    prf = token_prf1("APR1400", "APR1400 노형 1400 MWe")
    assert 0 < prf["f1"] < 1


def test_retrieval_pr():
    r = retrieval_pr(retrieved_pages=[12, 13, 14, 15, 16], gold_pages=[13, 17])
    assert r["recall@k"] == 0.5
    assert abs(r["precision@k"] - 0.2) < 1e-9


def test_retrieval_pr_empty_gold():
    r = retrieval_pr(retrieved_pages=[1, 2], gold_pages=[])
    assert r["recall@k"] == 1.0
    assert r["precision@k"] == 0.0


if __name__ == "__main__":
    test_exact_match()
    test_keyword_hit_rate_substring()
    test_token_prf1_identical()
    test_token_prf1_partial()
    test_retrieval_pr()
    test_retrieval_pr_empty_gold()
    print("[OK] all tests passed")
