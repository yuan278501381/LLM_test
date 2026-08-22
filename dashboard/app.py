# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
NN Playground - 手搓神经网络可视化实验平台
Streamlit 主入口（M00-M15 渐进式课程导航）
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from dashboard.styles.icons import svg_icon
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_section_heading,
)

st.set_page_config(
    page_title="NN Playground · Neural Research Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入统一的高对比度亮色主题
apply_custom_theme()

# Hero 区域
render_hero_header(
    title="NN Playground",
    subtitle="从数学、感知器与反向传播出发，用纯 NumPy 教学实现逐步理解 Transformer、多模态、训练、评估与强化学习",
    badge_text="EVIDENCE-LABELED · PURE NUMPY CORE · M00-M16",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 板块 1: 核心语言与网络基础 (3x3 原生容器网格)
# ---------------------------------------------------------------------------
render_section_heading(
    "CORE FOUNDATIONS & LLM // 神经网络底层基础与大语言模型核心", icon_name="cpu"
)

with st.container(border=True):
    icon_math = svg_icon("grid", size=22, color="#1d4ed8")
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_math}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">从这里开始：数学、Shape 与梯度检查</span></div>'
        '<span class="pill-badge pill-blue">M00</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/0_数学基础.py", label="进入零基础入口 // START →", width="stretch")

col1, col2, col3 = st.columns(3)

with col1:
    # ---- 卡片 1: 单神经元感知器 ----
    with st.container(border=True):
        icon_target = svg_icon("target", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_target}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">单神经元感知器</span></div>'
            '<span class="pill-badge pill-blue">M01</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/1_单神经元感知器.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 4: 全参数微观实验室 ----
    with st.container(border=True):
        icon_terminal = svg_icon("terminal", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_terminal}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">全参数微观实验室</span></div>'
            '<span class="pill-badge pill-emerald">M04</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/4_参数实验室.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 7: 注意力机制 ----
    with st.container(border=True):
        icon_eye = svg_icon("eye", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_eye}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">注意力机制</span></div>'
            '<span class="pill-badge pill-emerald">M07</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/7_注意力机制.py", label="进入实验 // LAUNCH →", width="stretch")

with col2:
    # ---- 卡片 2: 多层网络与活性探针 ----
    with st.container(border=True):
        icon_layers = svg_icon("layers", size=22, color="#6d28d9")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_layers}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">多层网络与活性探针</span></div>'
            '<span class="pill-badge pill-purple">M02</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/2_多层网络.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 5: 词嵌入空间 ----
    with st.container(border=True):
        icon_hash = svg_icon("hash", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_hash}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">词嵌入空间</span></div>'
            '<span class="pill-badge pill-blue">M05</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/5_词嵌入空间.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 8: Transformer ----
    with st.container(border=True):
        icon_box = svg_icon("box", size=22, color="#b45309")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_box}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">Transformer</span></div>'
            '<span class="pill-badge pill-amber">M08</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/8_Transformer.py", label="进入实验 // LAUNCH →", width="stretch")

with col3:
    # ---- 卡片 3: 优化器同屏竞速 ----
    with st.container(border=True):
        icon_zap = svg_icon("zap", size=22, color="#b45309")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_zap}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">优化器同屏竞速</span></div>'
            '<span class="pill-badge pill-amber">M03</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/3_优化器对比.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 6: 序列记忆 ----
    with st.container(border=True):
        icon_activity = svg_icon("activity", size=22, color="#6d28d9")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_activity}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">序列记忆与遗忘瓶颈</span></div>'
            '<span class="pill-badge pill-purple">M06</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/6_序列记忆.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 9: Mini-GPT ----
    with st.container(border=True):
        icon_cpu = svg_icon("cpu", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_cpu}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">Mini-GPT 文本生成</span></div>'
            '<span class="pill-badge pill-blue">M09</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/9_Mini_GPT.py", label="进入实验 // LAUNCH →", width="stretch")

# ---------------------------------------------------------------------------
# 板块 2: 多模态感知、世界模型与训练全生命周期 (3x2 网格)
# ---------------------------------------------------------------------------
render_section_heading(
    "ADVANCED MULTIMODAL & TRAINING // 进阶多模态、世界模型与全生命周期", icon_name="activity"
)

col_a1, col_a2, col_a3 = st.columns(3)

with col_a1:
    # ---- 卡片 10: 视觉感知 ----
    with st.container(border=True):
        icon_eye10 = svg_icon("eye", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_eye10}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">卷积与视觉感知</span></div>'
            '<span class="pill-badge pill-emerald">M10</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/10_视觉感知.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 13: 预训练范式 ----
    with st.container(border=True):
        icon_hash13 = svg_icon("hash", size=22, color="#1d4ed8")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_hash13}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">预训练范式全景</span></div>'
            '<span class="pill-badge pill-blue">M13</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/13_预训练范式.py", label="进入实验 // LAUNCH →", width="stretch")

with col_a2:
    # ---- 卡片 11: 音频感知 ----
    with st.container(border=True):
        icon_act11 = svg_icon("activity", size=22, color="#6d28d9")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_act11}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">音频信号与语音</span></div>'
            '<span class="pill-badge pill-purple">M11</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/11_音频感知.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 14: 后训练工程 ----
    with st.container(border=True):
        icon_target14 = svg_icon("target", size=22, color="#b45309")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_target14}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">后训练与对齐工程</span></div>'
            '<span class="pill-badge pill-amber">M14</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/14_后训练工程.py", label="进入实验 // LAUNCH →", width="stretch")

with col_a3:
    # ---- 卡片 12: 视频与世界模型 ----
    with st.container(border=True):
        icon_cpu12 = svg_icon("cpu", size=22, color="#be123c")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_cpu12}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">视频与世界模型</span></div>'
            '<span class="pill-badge pill-amber">M12</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/12_视频与世界模型.py", label="进入实验 // LAUNCH →", width="stretch")

    # ---- 卡片 15: 评估基准框架 ----
    with st.container(border=True):
        icon_term15 = svg_icon("terminal", size=22, color="#047857")
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">{icon_term15}<span style="font-size:1.2rem;font-weight:800;color:#0f172a;">评估基准框架</span></div>'
            '<span class="pill-badge pill-emerald">M15</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.page_link("pages/15_评估基准.py", label="进入实验 // LAUNCH →", width="stretch")

# ---------------------------------------------------------------------------
# 板块 3: 2026 前沿强化学习与自主智能体
# ---------------------------------------------------------------------------
render_section_heading(
    "REINFORCEMENT LEARNING & REASONING AGENTS // 强化学习与推理训练边界（含规则模拟）",
    icon_name="compass",
    subtext="从经典 GridWorld MDP 到 2025 DeepSeek-R1/R1-Zero 案例；GRPO 区域仅为规则曲线仿真",
)

with st.container(border=True):
    icon_reinf = svg_icon("compass", size=24, color="#be123c")
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:0.6rem;">{icon_reinf}'
        '<span style="font-size:1.25rem;font-weight:800;color:#0f172a;">强化学习与自主智能体实验室 (MDP · 贝尔曼曲面 · Q-Learning · GRPO)</span></div>'
        '<span class="pill-badge pill-rose">M16</span>'
        "</div>"
        '<div style="color:#475569;font-size:0.92rem;margin-bottom:0.8rem;line-height:1.6;">'
        "对照当前有限已知 GridWorld 的动态规划参考解与 Q-Learning 贪心路径；识别 DeepSeek-R1、R1-Zero 的训练差异，以及页内规则曲线与真实 GRPO 训练的边界。"
        "</div>",
        unsafe_allow_html=True,
    )
    st.page_link("pages/16_强化学习.py", label="进入强化学习实验室 // LAUNCH →", width="stretch")

# ---------------------------------------------------------------------------
# 系统技术架构与工程标准
# ---------------------------------------------------------------------------
col_arch, col_stats = st.columns([1.5, 1])

with col_arch:
    with st.container(border=True):
        icon_box2 = svg_icon("box", size=18, color="#1d4ed8")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f"{icon_box2}"
            f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">核心架构与设计原则</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **纯 NumPy 教学核心**: 基础层与关键模型组件不依赖 PyTorch/TensorFlow；高级主题可能是缩小实现或架构示意
            - **证据边界可见**: 每页明确区分真实计算、教学缩小版、合成数据、概率模拟与架构示意
            - **数学核对**: 部分核心算子在受控输入上用双侧中心差分与解析梯度比较；通过只表示当前点、步长、精度和容差下一致
            - **工程验证**: CI 配置覆盖跨平台测试、静态检查、构建与 Docker 验证
            """
        )

with col_stats:
    with st.container(border=True):
        icon_activity2 = svg_icon("activity", size=18, color="#047857")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">'
            f"{icon_activity2}"
            f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">引擎遥测状态</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.6rem;">
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#1d4ed8;font-family:'JetBrains Mono';">PYTEST [OK] </div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">RUN LOCALLY FOR COUNT</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#047857;font-family:'JetBrains Mono';">&lt; 1e-4</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">GRAD ERROR</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#b45309;font-family:'JetBrains Mono';">M00–M16</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">MILESTONES</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">
                    <div style="font-size:1.45rem;font-weight:800;color:#6d28d9;font-family:'JetBrains Mono';">5 LEVELS</div>
                    <div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">EVIDENCE LABELS USED</div>
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
