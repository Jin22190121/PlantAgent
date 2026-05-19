"""Streamlit 비교 대시보드: System A(Gemini) vs System B(EXAONE)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.factory import get_agent
from src.config import load_settings, resolve_path
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


def main():
    with st.sidebar:
        st.markdown("### FSAR Agent PoC")
        st.caption("Gemini vs EXAONE · 신고리 3,4호기 FSAR 1장")
        page = st.radio("페이지", ["질의 비교", "정량 평가"], index=0)
        st.markdown("---")
        st.caption(
            f"임베딩: `{cfg['embedding']['model_name']}`\n\n"
            f"청크: {cfg['ingestion']['chunk_size']}자 / overlap {cfg['ingestion']['chunk_overlap']}\n\n"
            f"top-k: {cfg['retrieval']['top_k']}\n\n"
            f"temperature: {cfg['generation']['temperature']}"
        )

    if page == "질의 비교":
        page_query()
    else:
        page_eval()


if __name__ == "__main__":
    main()
