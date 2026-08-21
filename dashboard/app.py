# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
NN Playground - 手搓神经网络可视化实验平台
Streamlit 主入口 (交互式可点击 Bento 导航中枢)
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


def _render_bento_nav_card(
    page_route: str,
    icon_name: str,
    icon_color: str,
    tag: str,
    tag_class: str,
    title: str,
    desc: str,
    feature: str,
) -> str:
    """生成整卡可点击直达目标详情页的 Neo-Bento 导航卡片 HTML"""
    icon_html = svg_icon(icon_name, size=22, color=icon_color)
    return (
        f'<a href="{page_route}" target="_self" class="cyber-nav-card">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">'
        f'<div>{icon_html}</div>'
        f'<span class="pill-badge {tag_class}">{tag}</span>'
        f'</div>'
        f'<h3 style="margin:0 0 0.5rem 0;color:#0f172a;font-size:1.24rem;font-weight:800;">{title}</h3>'
        f'<p style="color:#475569;font-size:0.9rem;line-height:1.6;margin-bottom:0.8rem;">{desc}</p>'
        f'<div style="font-size:0.78rem;color:{icon_color};font-family:\'JetBrains Mono\';font-weight:700;margin-bottom:0.6rem;">'
        f'FEATURE: {feature}'
        f'</div>'
        f'<div class="nav-action-link" style="color:{icon_color};">'
        f'ENTER LAB // 点击进入实验 &rarr;'
        f'</div>'
        f'</a>'
    )


# ---------------------------------------------------------------------------
# 首页 Bento 卡片矩阵 (2x2 可点击交互导航网格)
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    card1_html = _render_bento_nav_card(
        page_route="单神经元感知器",
        icon_name="target",
        icon_color="#1d4ed8",
        tag="M01 // PERCEPTRON",
        tag_class="pill-blue",
        title="单神经元感知器",
        desc="解构最小计算单元：线性前向计算 Z=XW+b、交叉熵损失、反向传播链式推导与权重轨迹寻优。",
        feature="2D 线性分界面 · Loss 收敛曲线 · 权重参数空间轨迹",
    )
    st.markdown(card1_html, unsafe_allow_html=True)

    card3_html = _render_bento_nav_card(
        page_route="优化器对比",
        icon_name="zap",
        icon_color="#b45309",
        tag="M03 // OPTIMIZER ARENA",
        tag_class="pill-amber",
        title="优化器同屏竞速",
        desc="同一起跑线对比 SGD、Momentum、RMSProp 与 Adam，观测动量累积、自适应学习率与偏差修正的威力。",
        feature="多轨 Loss 对比 · 决策边界拟合速率 · 收敛步数排行榜",
    )
    st.markdown(card3_html, unsafe_allow_html=True)

with col2:
    card2_html = _render_bento_nav_card(
        page_route="多层网络",
        icon_name="layers",
        icon_color="#6d28d9",
        tag="M02 // TOPOLOGY & PROBE",
        tag_class="pill-purple",
        title="多层网络与活性探针",
        desc="理解「深度」的非线性流形折叠。内置单样本动态活性探针，实时观测信号在各层神经元中的点亮状态。",
        feature="神经元活性探针 · 激活热力图 · 梯度消失/爆炸直方图",
    )
    st.markdown(card2_html, unsafe_allow_html=True)

    card4_html = _render_bento_nav_card(
        page_route="参数实验室",
        icon_name="terminal",
        icon_color="#047857",
        tag="M04 // PARAMETER LAB",
        tag_class="pill-emerald",
        title="全参数微观实验室",
        desc="工业级四宫格微观监控台，支持逐步训练 (Step-by-Step)、快照回滚、A/B 对比与全量实验日志导出。",
        feature="四宫格全景遥测 · 逐步微调 · 实验快照 JSON",
    )
    st.markdown(card4_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 系统技术架构与工程标准
# ---------------------------------------------------------------------------
col_arch, col_stats = st.columns([1.5, 1])

with col_arch:
    icon_box = svg_icon("box", size=18, color="#1d4ed8")
    arch_html = (
        f'<div class="cyber-card">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem;">'
        f'{icon_box}'
        f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">核心架构与设计原则</span>'
        f'</div>'
        f'<ul style="color:#334155;font-size:0.9rem;line-height:1.9;margin-bottom:0;padding-left:1.2rem;">'
        f'<li><b>纯 NumPy 底层</b>: 零 PyTorch/TF 依赖，前向计算与反向传播 100% 纯手写矩阵运算</li>'
        f'<li><b>数学严谨性</b>: 梯度校验通过中心差分法 (Central Difference)，解析误差严格 &lt; 1e-5</li>'
        f'<li><b>模块化解耦</b>: <code>Activations</code>, <code>Losses</code>, <code>Layers</code>, <code>Optimizers</code>, <code>Callbacks</code> 遵循高内聚原则</li>'
        f'<li><b>工业级 DevOps</b>: GitHub Actions 矩阵测试 (Linux/macOS/Windows × Py3.11/12/13)、多阶段 Docker 容器化</li>'
        f'</ul>'
        f'</div>'
    )
    st.markdown(arch_html, unsafe_allow_html=True)

with col_stats:
    icon_activity = svg_icon("activity", size=18, color="#047857")
    stats_html = (
        f'<div class="cyber-card">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem;">'
        f'{icon_activity}'
        f'<span style="font-size:1.1rem;font-weight:700;color:#0f172a;">引擎遥测状态</span>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.8rem;">'
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:#1d4ed8;font-family:\'JetBrains Mono\';">85 / 85</div>'
        f'<div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">UNIT & E2E TESTS</div>'
        f'</div>'
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:#047857;font-family:\'JetBrains Mono\';">&lt; 1e-5</div>'
        f'<div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">GRAD ERROR</div>'
        f'</div>'
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:#b45309;font-family:\'JetBrains Mono\';">4 CLASS</div>'
        f'<div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">OPTIMIZERS</div>'
        f'</div>'
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;padding:0.8rem;border-radius:8px;text-align:center;">'
        f'<div style="font-size:1.45rem;font-weight:800;color:#6d28d9;font-family:\'JetBrains Mono\';">5 TYPES</div>'
        f'<div style="font-size:0.72rem;color:#64748b;font-weight:700;text-transform:uppercase;">TOPOLOGIES</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(stats_html, unsafe_allow_html=True)

# 底部版权声明
st.markdown(
    """<div style="text-align: center; color: #64748b; font-size: 0.82rem; margin-top: 2.5rem; font-family: 'JetBrains Mono'; font-weight: 500;">
ENGINEERED BY <a href="https://github.com/yuan278501381" style="color: #1d4ed8; font-weight: 700; text-decoration: none;">Yy1 (yuan278501381)</a> · MIT LICENSE · COPYRIGHT (C) 2026
</div>""",
    unsafe_allow_html=True,
)
