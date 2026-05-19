# 골드셋 (Q&A 정답셋)

`goldset.jsonl`은 20문항의 시드 골드셋이다. 카테고리는 4종:

- **사실검색**: 노형/출력/수명 등 단일 사실 추출
- **위치지정**: 절 번호·페이지 인용 능력 평가
- **요약**: 다문장 요약 품질
- **비교**: 다른 호기/노형 대비 차이점 추론

## 스키마

```jsonc
{
  "qid": "Q01",
  "category": "사실검색|위치지정|요약|비교",
  "question": "질문 본문",
  "gold_answer": "정답 본문",
  "gold_keywords": ["반드시 포함되어야 할 키워드들"],
  "gold_pages": [12, 13]   // 정답이 위치한 페이지 번호 (PDF 실제 페이지)
}
```

## 정답 보정 절차

1. PDF를 `data/raw/`에 배치하고 `python -m src.ingestion.build_index --reset` 실행
2. PDF를 직접 열어 각 질문의 **정답 페이지 번호**를 확인하여 `gold_pages` 필드를 채운다
   - `gold_pages`가 비어 있으면 검색 메트릭이 의미를 잃는다
3. `gold_answer`가 PDF 실제 내용과 다르면 수정 (시드는 일반 지식 기반의 추정치)
4. 골드셋이 20문항 미만이면 카테고리별 균형을 유지하며 추가

## 운영 팁

- `python -m src.ingestion.build_index` 실행 후 한두 질문에 대해 Streamlit UI로 정성 점검 → 명백한 오답 패턴이 보이면 정답을 보정
- 다수 문항을 한꺼번에 보정하려면 `report_*.csv`의 `retrieved_pages` 열을 참고하여 실제 페이지를 빠르게 식별 가능
