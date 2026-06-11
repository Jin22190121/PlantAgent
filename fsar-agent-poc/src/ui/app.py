"""Streamlit 비교 대시보드: System A(Gemini) vs System B(EXAONE)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# `streamlit run src/ui/app.py`로 직접 실행될 때 프로젝트 루트가
# sys.path에 없으므로 `from src.X import ...` 가 실패한다. 명시적으로 추가.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.factory import get_agent
from src.config import load_settings, resolve_path
from src.evaluation.human_eval import (
    ACCURACY_KO,
    ACCURACY_LEVELS,
    HumanEvalRecord,
    aggregate_stats,
    append_record,
    export_csv,
    export_markdown,
    load_records,
    log_path,
)
from src.evaluation.runner import run as run_eval
from src.retrieval.retriever import count as vector_count

st.set_page_config(
    page_title="FSAR Agent: Gemini vs EXAONE",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_settings()


def _system_box(col, system_key: str, label: str, model: str):
    with col:
        st.markdown(f"### {label}")
        st.caption(f"모델: `{model}`")


@st.cache_resource(show_spinner=False)
def _agent(system_key: str):
    return get_agent(system_key)


def page_query():
    st.title("FSAR AI Agent — 이중 LLM 비교")
    st.caption(
        "신고리 3,4호기 FSAR 1장에 대해 동일 질문을 두 시스템에 입력하고 결과를 비교합니다. "
        "임베딩·청킹·검색·프롬프트·온도는 모두 동일하게 통제됩니다."
    )

    n = vector_count()
    if n == 0:
        st.error(
            "ChromaDB에 인덱스된 청크가 없습니다. "
            "data/raw/ 에 PDF를 배치하고 `python -m src.ingestion.build_index` 를 먼저 실행하세요."
        )
        return

    st.success(f"ChromaDB 청크 수: **{n}**")

    sample_questions = [
        "신고리 3,4호기의 노형은 무엇이며 정격 전기출력은 얼마인가?",
        "신고리 3,4호기와 신고리 1,2호기의 주요 차이점은 무엇인가?",
        "신고리 3,4호기의 부지 특성을 3문장으로 요약하라.",
    ]
    q = st.text_area(
        "질문",
        value=sample_questions[0],
        height=80,
        help="동일 질문이 두 시스템에 그대로 입력됩니다.",
    )

    run_both = st.button("두 시스템 동시 실행", type="primary", use_container_width=True)
    examples = st.expander("예시 질문")
    with examples:
        for s in sample_questions:
            if st.button(s, key=f"ex-{s[:10]}"):
                st.session_state["_q"] = s
                st.rerun()
    if "_q" in st.session_state:
        q = st.session_state.pop("_q")

    if not run_both:
        return

    col_a, col_b = st.columns(2, gap="large")
    _system_box(col_a, "A", "System A — Gemini", cfg["system_a"]["model"])
    _system_box(col_b, "B", "System B — EXAONE", cfg["system_b"]["model"])

    responses = {}
    for sys_key, col in [("A", col_a), ("B", col_b)]:
        with col:
            with st.spinner(f"{sys_key} 응답 생성 중..."):
                try:
                    agent = _agent(sys_key)
                    responses[sys_key] = agent.ask(q)
                except Exception as e:
                    st.error(f"실패: {e}")
                    responses[sys_key] = None

    for sys_key, col in [("A", col_a), ("B", col_b)]:
        r = responses.get(sys_key)
        if r is None:
            continue
        with col:
            st.markdown("**답변**")
            st.write(r.answer)
            m1, m2, m3 = st.columns(3)
            m1.metric("응답 시간", f"{r.latency_ms:.0f} ms")
            m2.metric("입력 토큰", r.prompt_tokens)
            m3.metric("출력 토큰", r.completion_tokens)

            with st.expander(f"검색된 청크 ({len(r.retrieved)}개)"):
                for c in r.retrieved:
                    st.markdown(
                        f"- **p.{c.page} §{c.section}** ({c.section_title}) — score={c.score:.3f}"
                    )
                    st.caption(c.text[:200] + ("..." if len(c.text) > 200 else ""))


def page_eval():
    st.title("정량 평가 실행 및 리포트")

    goldset_path = resolve_path(cfg["paths"]["goldset"])
    if not goldset_path.exists():
        st.error(f"골드셋이 없습니다: {goldset_path}")
        return

    with open(goldset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    st.info(f"골드셋 문항 수: **{len(items)}**")

    systems = st.multiselect(
        "평가할 시스템",
        options=["A", "B"],
        default=["A", "B"],
    )
    if st.button("평가 실행", type="primary"):
        if not systems:
            st.warning("최소 하나의 시스템을 선택하세요.")
            return
        with st.spinner("평가 진행 중... (문항 수 × 시스템 수만큼 LLM 호출이 발생합니다)"):
            result = run_eval(systems)
        st.success("완료")
        _render_summary(result)

    st.markdown("---")
    st.subheader("기존 리포트 불러오기")
    report_dir = resolve_path(cfg["paths"]["report_dir"])
    reports = sorted(report_dir.glob("report_*.json"), reverse=True)
    if not reports:
        st.caption("아직 생성된 리포트가 없습니다.")
        return
    choice = st.selectbox("리포트 선택", [r.name for r in reports])
    if choice:
        with open(report_dir / choice, "r", encoding="utf-8") as f:
            result = json.load(f)
        _render_summary(result)


def _render_summary(result: dict):
    st.subheader(f"요약 — {result.get('timestamp', '')}")
    sumry = result.get("summary", {})

    if "A" in sumry or "B" in sumry:
        cols = st.columns(2)
        for i, sys_key in enumerate(["A", "B"]):
            if sys_key not in sumry:
                continue
            s = sumry[sys_key]
            with cols[i]:
                label = "Gemini (A)" if sys_key == "A" else "EXAONE (B)"
                st.markdown(f"#### {label}")
                st.metric("Token F1 (평균)", f"{s.get('token_f1_mean', 0):.3f}")
                st.metric("Keyword 적중률", f"{s.get('keyword_hit_rate_mean', 0):.3f}")
                st.metric("Retrieval Recall@k", f"{s.get('retrieval_recall@k_mean', 0):.3f}")
                st.metric("평균 지연(ms)", f"{s.get('latency_ms_mean', 0):.0f}")
                st.metric("총 비용 (USD)", f"{s.get('cost_usd_total', 0):.4f}")

    cat = result.get("per_category", {})
    if cat:
        st.subheader("카테고리별 Token F1")
        rows = []
        for sys_key, cats in cat.items():
            for c, m in cats.items():
                rows.append(
                    {"system": sys_key, "category": c, "token_f1": m.get("token_f1_mean", 0)}
                )
        if rows:
            df = pd.DataFrame(rows)
            fig = px.bar(
                df,
                x="category",
                y="token_f1",
                color="system",
                barmode="group",
                title="카테고리 × 시스템 Token F1",
            )
            st.plotly_chart(fig, use_container_width=True)

    rows = result.get("rows", {})
    if rows:
        st.subheader("문항별 결과")
        all_rows = []
        for sys_key, lst in rows.items():
            for r in lst:
                all_rows.append({**r, "_system": sys_key})
        df = pd.DataFrame(all_rows)
        cols_show = [
            "_system", "qid", "category", "question",
            "token_f1", "keyword_hit_rate",
            "retrieval_recall@k", "latency_ms",
            "prompt_tokens", "completion_tokens",
        ]
        st.dataframe(df[[c for c in cols_show if c in df.columns]], use_container_width=True)


def page_human_eval():
    """사용자가 PDF로 정답을 알고 두 시스템 답변을 직접 채점하는 대화형 페이지."""
    st.title("인간 평가 — 대화형 채점")
    st.caption(
        "사용자가 PDF로 정답을 학습한 뒤, 두 시스템(A·B)의 답변을 직접 채점합니다. "
        "같은 질문에 두 시스템이 동시 응답하며 라벨이 표시됩니다."
    )

    if vector_count() == 0:
        st.error("ChromaDB에 인덱스가 없습니다. 먼저 인덱싱을 실행하세요.")
        return

    # 세션 초기화
    if "he_messages" not in st.session_state:
        st.session_state.he_messages = []
    if "he_category" not in st.session_state:
        st.session_state.he_category = "사실검색"

    # ── 사이드바: 설정·통계·내보내기 ─────────────────
    with st.sidebar:
        st.markdown("### 인간 평가 설정")
        st.session_state.he_category = st.selectbox(
            "다음 질문의 카테고리",
            ["사실검색", "위치지정", "요약", "비교", "기타"],
            index=["사실검색", "위치지정", "요약", "비교", "기타"].index(
                st.session_state.he_category
            ),
        )
        if st.button("대화 리셋", use_container_width=True):
            st.session_state.he_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 누적 통계")
        records = load_records()
        stats = aggregate_stats(records)
        st.metric("누적 평가 수", stats.get("n_total", 0))
        for sys_key, s in stats.get("by_system", {}).items():
            label = "Gemini (A)" if sys_key == "A" else "EXAONE (B)"
            with st.expander(f"{label} — {s['n']}건", expanded=False):
                st.metric("정답률", f"{s['accuracy_rate']:.1%}")
                st.metric("부분정답률", f"{s['partial_rate']:.1%}")
                st.metric("오답률", f"{s['wrong_rate']:.1%}")
                st.metric("평균 평점", f"{s['rating_mean']:.2f} / 5")
                st.metric("평균 지연", f"{s['latency_ms_mean']:.0f} ms")

        st.markdown("---")
        st.markdown("### 내보내기")
        export_dir = resolve_path(cfg["paths"]["report_dir"])
        if st.button("CSV 내보내기", use_container_width=True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = export_csv(records, export_dir / f"human_eval_export_{ts}.csv")
            st.success(f"저장: {out.name}")
        if st.button("Markdown 리포트 내보내기", use_container_width=True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = export_markdown(
                records, stats, export_dir / f"human_eval_export_{ts}.md"
            )
            st.success(f"저장: {out.name}")
        st.caption(f"로그 파일: `{log_path()}`")

    # ── 메인: 대화 히스토리 + 입력 ──────────────────
    for i, msg in enumerate(st.session_state.he_messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                cat = msg.get("category", "")
                if cat:
                    st.markdown(f"_[{cat}]_")
                st.markdown(msg["content"])
        else:
            _render_assistant_message(i, msg)

    if q := st.chat_input("질문을 입력하세요 (예: 신고리 3,4호기의 정격 출력은?)"):
        st.session_state.he_messages.append(
            {
                "role": "user",
                "content": q,
                "category": st.session_state.he_category,
            }
        )
        for sys_key in ("A", "B"):
            with st.spinner(f"System {sys_key} 응답 생성 중..."):
                try:
                    agent = _agent(sys_key)
                    resp = agent.ask(q)
                    payload = {
                        "answer": resp.answer,
                        "retrieved_pages": resp.cited_pages,
                        "retrieved_detail": [
                            {
                                "page": c.page,
                                "section": c.section,
                                "section_title": c.section_title,
                                "score": c.score,
                                "preview": c.text[:200],
                            }
                            for c in resp.retrieved
                        ],
                        "latency_ms": resp.latency_ms,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                    }
                    error = None
                except Exception as e:
                    payload = {
                        "answer": f"[ERROR] {e}",
                        "retrieved_pages": [],
                        "retrieved_detail": [],
                        "latency_ms": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
                    error = str(e)
            st.session_state.he_messages.append(
                {
                    "role": "assistant",
                    "system": sys_key,
                    "category": st.session_state.he_category,
                    "question": q,
                    "response": payload,
                    "error": error,
                    "scored": False,
                    "scores": None,
                }
            )
        st.rerun()


def _render_assistant_message(i: int, msg: dict):
    sys_key = msg["system"]
    sys_name = "Gemini" if sys_key == "A" else "EXAONE"
    model = cfg["system_a" if sys_key == "A" else "system_b"]["model"]
    resp = msg["response"]
    with st.chat_message("assistant"):
        st.markdown(f"**System {sys_key} — {sys_name}** · `{model}`")
        st.write(resp["answer"])

        m1, m2, m3 = st.columns(3)
        m1.metric("지연", f"{resp['latency_ms']:.0f} ms")
        m2.metric("입력 토큰", resp["prompt_tokens"])
        m3.metric("출력 토큰", resp["completion_tokens"])

        detail = resp.get("retrieved_detail", [])
        if detail:
            with st.expander(f"검색 청크 ({len(detail)}개) · 인용 페이지 {resp['retrieved_pages']}"):
                for c in detail:
                    st.markdown(
                        f"- **p.{c['page']} §{c['section']}** "
                        f"({c['section_title']}) — score={c['score']:.3f}"
                    )
                    st.caption(c["preview"] + ("..." if len(c["preview"]) >= 200 else ""))

        if msg.get("scored"):
            sc = msg["scores"]
            st.success(
                f"✓ 정확성: **{ACCURACY_KO.get(sc['accuracy'], sc['accuracy'])}** · "
                f"평점: **{sc['rating']}/5**"
            )
            if sc.get("note"):
                st.caption(f"코멘트: {sc['note']}")
            return

        if msg.get("error"):
            st.warning("에러 발생 — 채점은 'na'로 자동 저장하거나 건너뛰세요.")

        with st.form(f"score_form_{i}", clear_on_submit=False):
            acc = st.radio(
                "정확성",
                ACCURACY_LEVELS,
                format_func=lambda x: ACCURACY_KO[x],
                horizontal=True,
                key=f"acc_{i}",
            )
            rating = st.slider("종합 평점 (1=나쁨, 5=완벽)", 1, 5, 3, key=f"rat_{i}")
            note = st.text_input("코멘트 (선택)", key=f"note_{i}")
            submitted = st.form_submit_button("채점 저장", type="primary")
            if submitted:
                record = HumanEvalRecord(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    question=msg.get("question", ""),
                    category=msg.get("category", "기타"),
                    system=sys_key,
                    system_model=model,
                    answer=resp["answer"],
                    retrieved_pages=resp["retrieved_pages"],
                    latency_ms=resp["latency_ms"],
                    prompt_tokens=resp["prompt_tokens"],
                    completion_tokens=resp["completion_tokens"],
                    accuracy=acc,
                    rating=int(rating),
                    note=note or "",
                )
                append_record(record)
                st.session_state.he_messages[i]["scored"] = True
                st.session_state.he_messages[i]["scores"] = {
                    "accuracy": acc,
                    "rating": int(rating),
                    "note": note or "",
                }
                st.rerun()


def main():
    with st.sidebar:
        st.markdown("### FSAR Agent PoC")
        st.caption("Gemini vs EXAONE · 신고리 3,4호기 FSAR 1장")
        page = st.radio(
            "페이지",
            ["질의 비교", "정량 평가", "인간 평가 (대화형)"],
            index=0,
        )
        st.markdown("---")
        st.caption(
            f"임베딩: `{cfg['embedding']['model_name']}`\n\n"
            f"청크: {cfg['ingestion']['chunk_size']}자 / overlap {cfg['ingestion']['chunk_overlap']}\n\n"
            f"top-k: {cfg['retrieval']['top_k']}\n\n"
            f"temperature: {cfg['generation']['temperature']}"
        )

    if page == "질의 비교":
        page_query()
    elif page == "정량 평가":
        page_eval()
    else:
        page_human_eval()


if __name__ == "__main__":
    main()
