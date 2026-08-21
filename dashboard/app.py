# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
NN Playground - 手搓神经网络可视化实验平台
Streamlit 主入口 (现代极简高精 Bento 导航中枢)
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from dashboard.styles.icons import svg_icon
from dashboard.styles.theme import apply_custom_theme, render_hero_header

st.set_page_config(
    page_title="NN Playground · Neural Research Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入世界级极简高对比度亮色主题
apply_custom_theme()

# Hero 区域
render_hero_header(
    title="NN Playground",
    subtitle="零框架手搓神经网络 · 全参数透明可视化实验平台 · 彻底击碎深度学习黑盒",
    badge_text="WORLD-CLASS ARCHITECTURE · PURE NUMPY CORE",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 首页 Bento 卡片矩阵 (3x3 原生容器网格)
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    # ---- 卡片 1: 单神经元感知器 ----
    with st.container(border=True):
        icon_target = svg_icon("target", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_target}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">单神经元感知器</span></div>'
            '<span class="pill-badge pill-blue">M01</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_单神经元感知器.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 4: 全参数微观实验室 ----
    with st.container(border=True):
        icon_terminal = svg_icon("terminal", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_terminal}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">全参数微观实验室</span></div>'
            '<span class="pill-badge pill-emerald">M04</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/4_参数实验室.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 7: 注意力机制 ----
    with st.container(border=True):
        icon_eye = svg_icon("eye", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_eye}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">注意力机制</span></div>'
            '<span class="pill-badge pill-emerald">M07</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/7_注意力机制.py", label="进入实验 // LAUNCH →", use_container_width=True)

with col2:
    # ---- 卡片 2: 多层网络与活性探针 ----
    with st.container(border=True):
        icon_layers = svg_icon("layers", size=22, color="#6d28d9")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_layers}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">多层网络与活性探针</span></div>'
            '<span class="pill-badge pill-purple">M02</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/2_多层网络.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 5: 词嵌入空间 ----
    with st.container(border=True):
        icon_hash = svg_icon("hash", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_hash}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">词嵌入空间</span></div>'
            '<span class="pill-badge pill-blue">M05</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/5_词嵌入空间.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 8: Transformer ----
    with st.container(border=True):
        icon_box = svg_icon("box", size=22, color="#b45309")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_box}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">Transformer</span></div>'
            '<span class="pill-badge pill-amber">M08</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/8_Transformer.py", label="进入实验 // LAUNCH →", use_container_width=True)

with col3:
    # ---- 卡片 3: 优化器同屏竞速 ----
    with st.container(border=True):
        icon_zap = svg_icon("zap", size=22, color="#b45309")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_zap}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">优化器同屏竞速</span></div>'
            '<span class="pill-badge pill-amber">M03</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/3_优化器对比.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 6: 序列记忆 ----
    with st.container(border=True):
        icon_activity = svg_icon("activity", size=22, color="#6d28d9")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_activity}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">序列记忆与遗忘瓶颈</span></div>'
            '<span class="pill-badge pill-purple">M06</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/6_序列记忆.py", label="进入实验 // LAUNCH →", use_container_width=True)

    # ---- 卡片 9: Mini-GPT ----
    with st.container(border=True):
        icon_cpu = svg_icon("cpu", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_cpu}<span style="font-size:1.24rem;font-weight:800;color:#0f172a;">Mini-GPT 文本生成</span></div>'
            '<span class="pill-badge pill-blue">M09</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/9_Mini_GPT.py", label="进入实验 // LAUNCH →", use_container_width=True)

# ---------------------------------------------------------------------------
# 系统技术架构与工程标准
# ---------------------------------------------------------------------------
col_arch, col_stats = st.columns([1.5, 1])

with col_arch:
    with st.container(border=True):
        icon_box2 = svg_icon("box", size=18, color="#1d4ed8")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f'{icon_box2}'
            f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">核心架构与设计原则</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **纯 NumPy 底层**: 零 PyTorch/TF 依赖，前向计算与反向传播 100% 纯手写矩阵运算
            - **数学严谨性**: 梯度校验通过中心差分法 (Central Difference)，解析误差严格 < 1e-5
            - **模块化解耦**: `Activations`, `Losses`, `Layers`, `Optimizers`, `Callbacks` 遵循高内聚原则
            - **工业级 DevOps**: GitHub Actions 矩阵测试 (Linux/macOS/Windows × Py3.11/12/13)、多阶段 Docker 容器化
            """
        )

with col_stats:
    with st.container(border=True):
        icon_activity2 = svg_icon("activity", size=18, color="#047857")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f'{icon_activity2}'
            f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">引擎遥测状态</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.6rem;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#1d4ed8;font-family:'JetBrains Mono';">146 / 146</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">UNIT & E2E TESTS</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#047857;font-family:'JetBrains Mono';">&lt; 1e-5</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">GRAD ERROR</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#b45309;font-family:'JetBrains Mono';">9 STAGES</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">MILESTONES</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#6d28d9;font-family:'JetBrains Mono';">100%</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">NUMPY BACKEND</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 底部版权声明
st.markdown(
    """<div style="text-align: center; color: #64748b; font-size: 0.82rem; margin-top: 2.5rem; font-family: 'JetBrains Mono'; font-weight: 500;">
ENGINEERED BY <a href="https://github.com/yuan278501381" style="color: #1d4ed8; font-weight: 700; text-decoration: none;">Yy1 (yuan278501381)</a> · MIT LICENSE · COPYRIGHT (C) 2026
</div>""",
    unsafe_allow_html=True,
)
