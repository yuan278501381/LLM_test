# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""教学证据、课程契约与参考资料的统一 Streamlit 组件。"""

from html import escape

import streamlit as st

from dashboard.constants.course import EVIDENCE_DESCRIPTIONS, LESSONS, EvidenceLevel


_EVIDENCE_COLORS: dict[EvidenceLevel, tuple[str, str, str]] = {
    EvidenceLevel.EXACT_COMPUTATION: ("#ecfdf5", "#047857", "#a7f3d0"),
    EvidenceLevel.TEACHING_SCALE: ("#eff6ff", "#1d4ed8", "#bfdbfe"),
    EvidenceLevel.SYNTHETIC_DATA: ("#fff7ed", "#b45309", "#fed7aa"),
    EvidenceLevel.SIMULATION: ("#fff1f2", "#be123c", "#fecdd3"),
    EvidenceLevel.ARCHITECTURE_ONLY: ("#f5f3ff", "#6d28d9", "#ddd6fe"),
    EvidenceLevel.PAPER_REPRODUCTION: ("#f0fdfa", "#0f766e", "#99f6e4"),
}


def evidence_badge(level: EvidenceLevel) -> str:
    """返回无脚本、可测试的证据标签 HTML。"""

    bg, fg, border = _EVIDENCE_COLORS[level]
    return (
        f'<span data-evidence-level="{escape(level.name)}" '
        f'style="display:inline-flex;align-items:center;padding:0.28rem 0.55rem;'
        f"margin:0 0.3rem 0.3rem 0;border-radius:999px;background:{bg};color:{fg};"
        f'border:1px solid {border};font-size:0.76rem;font-weight:800;">'
        f"{escape(level.value)}</span>"
    )


def render_lesson_evidence(lesson_id: str, *, show_contract: bool = False) -> None:
    """在页面顶部呈现证据性质、结论边界和可追溯参考资料。"""

    if lesson_id not in LESSONS:
        raise ValueError(f"未知课程编号: {lesson_id}")
    lesson = LESSONS[lesson_id]
    badges = "".join(evidence_badge(level) for level in lesson.evidence)
    descriptions = "；".join(EVIDENCE_DESCRIPTIONS[level] for level in lesson.evidence)
    primary_reference = lesson.references[0]
    st.markdown(
        '<section data-testid="lesson-evidence" '
        'style="background:#ffffff;border:1px solid #cbd5e1;border-left:4px solid #1d4ed8;'
        'border-radius:10px;padding:0.8rem 1rem;margin:0.35rem 0 1rem 0;">'
        f'<div style="font-weight:800;color:#0f172a;margin-bottom:0.45rem;">'
        f"{escape(lesson.lesson_id)} · EVIDENCE // 本页证据性质</div>{badges}"
        f'<div style="color:#334155;font-size:0.88rem;line-height:1.65;">{escape(descriptions)}</div>'
        f'<div style="color:#7c2d12;font-size:0.86rem;margin-top:0.4rem;">'
        f"<strong>结论边界：</strong>{escape(lesson.conclusion_boundary)}</div>"
        f'<div style="font-size:0.84rem;margin-top:0.35rem;"><strong>原始资料：</strong>'
        f'<a href="{escape(primary_reference.url, quote=True)}" target="_blank" rel="noopener noreferrer">'
        f"{escape(primary_reference.title)}</a></div></section>",
        unsafe_allow_html=True,
    )
    if show_contract:
        render_lesson_contract(lesson_id)


def render_lesson_contract(lesson_id: str) -> None:
    """呈现统一教学结构；重点页面可直接复用。"""

    if lesson_id not in LESSONS:
        raise ValueError(f"未知课程编号: {lesson_id}")
    lesson = LESSONS[lesson_id]
    with st.expander("LEARNING CONTRACT // 学习目标、失败案例与参考资料"):
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**先修知识**")
            for item in lesson.prerequisites:
                st.markdown(f"- {item}")
            st.markdown("**学习目标**")
            for item in lesson.objectives:
                st.markdown(f"- {item}")
            st.markdown(f"**前代瓶颈**：{lesson.predecessor_problem}")
        with col_right:
            st.markdown("**建议观察**")
            for item in lesson.observations:
                st.markdown(f"- {item}")
            st.markdown("**失败案例**")
            for item in lesson.failure_cases:
                st.markdown(f"- {item}")
            st.markdown(f"**历史影响**：{lesson.historical_impact}")
        st.markdown("**权威参考资料**")
        for ref in lesson.references:
            st.markdown(f"- [{ref.title}]({ref.url}) — {ref.note}")
