# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
🧠 NN Playground - 手搓神经网络可视化实验平台
Streamlit 主入口 (Neo-Bento 赛博玻璃拟态首页)
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from dashboard.styles.theme import apply_custom_theme, render_hero_header

st.set_page_config(
    page_title="NN Playground · 手搓神经网络实验台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入世界级深色玻璃拟态主题
apply_custom_theme()

# Hero 区域
render_hero_header(
    title="NN Playground",
    subtitle="零框架手搓神经网络 · 全参数透明可视化实验平台 · 彻底击碎深度学习黑盒",
    badge_text="WORLD-CLASS ARCHITECTURE · PURE NUMPY CORE",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 首页 Bento 卡片矩阵 (Bento Grid)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.2rem; margin-bottom: 2rem;">
        <!-- 卡片 1 -->
        <div class="cyber-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <span style="font-size: 1.5rem;">🎯</span>
                <span class="pill-badge pill-blue">MILESTONE 1</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #f8fafc; font-size: 1.25rem;">单神经元感知器</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1rem;">
                解构最小计算单元：前向计算 $Z=XW+b$、交叉熵损失、反向传播链式推导与权重轨迹寻优。
            </p>
            <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 600;">
                📊 核心观测: 2D 线性分界面 · Loss 收敛曲线 · 权重参数空间轨迹
            </div>
        </div>

        <!-- 卡片 2 -->
        <div class="cyber-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <span style="font-size: 1.5rem;">🧬</span>
                <span class="pill-badge pill-purple">MILESTONE 2</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #f8fafc; font-size: 1.25rem;">多层网络与活性探针</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1rem;">
                理解「深度」的非线性流形折叠。内置<b>单样本动态活性探针</b>，实时观测信号在各层神经元中的点亮状态。
            </p>
            <div style="font-size: 0.8rem; color: #c084fc; font-weight: 600;">
                🔥 核心观测: 神经元活性探针 · 激活热力图 · 梯度消失/爆炸直方图
            </div>
        </div>

        <!-- 卡片 3 -->
        <div class="cyber-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <span style="font-size: 1.5rem;">🏎️</span>
                <span class="pill-badge pill-amber">MILESTONE 3</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #f8fafc; font-size: 1.25rem;">优化器同屏竞速</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1rem;">
                同一起跑线对比 SGD、Momentum、RMSProp 与 Adam，观测动量累积、自适应学习率与偏差修正的威力。
            </p>
            <div style="font-size: 0.8rem; color: #fbbf24; font-weight: 600;">
                🏁 核心观测: 多轨 Loss 对比 · 决策边界拟合速率 · 收敛步数排行榜
            </div>
        </div>

        <!-- 卡片 4 -->
        <div class="cyber-card">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <span style="font-size: 1.5rem;">🔬</span>
                <span class="pill-badge pill-emerald">MILESTONE 4</span>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #f8fafc; font-size: 1.25rem;">全参数实验室</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 1rem;">
                工业级四宫格微观监控台，支持逐步训练（Step-by-Step）、快照回滚、A/B 对比与全量实验日志导出。
            </p>
            <div style="font-size: 0.8rem; color: #34d399; font-weight: 600;">
                🎛️ 核心观测: 四宫格全景遥测 · 逐步微调 · 实验快照 JSON
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 系统技术架构与工程标准
# ---------------------------------------------------------------------------
col_arch, col_stats = st.columns([1.5, 1])

with col_arch:
    st.markdown(
        """
        <div class="cyber-card">
            <h3 style="margin-top: 0; color: #f8fafc;">🏛️ 核心架构与设计原则</h3>
            <ul style="color: #94a3b8; font-size: 0.9rem; line-height: 1.8; margin-bottom: 0;">
                <li><b>纯 NumPy 底层</b>: 零 PyTorch/TF 依赖，前向计算与反向传播 100% 纯手写矩阵运算</li>
                <li><b>数学严谨性</b>: 梯度校验通过中心差分法（Central Difference），解析误差严格 &lt; 1e-5</li>
                <li><b>模块化解耦</b>: <code>Activations</code>, <code>Losses</code>, <code>Layers</code>, <code>Optimizers</code>, <code>Callbacks</code> 严格遵循 OCP 原则</li>
                <li><b>世界级 DevOps</b>: GitHub Actions 矩阵测试 (Linux/macOS/Windows × Py3.11/12/13)、多阶段 Docker 容器化</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_stats:
    st.markdown(
        """
        <div class="cyber-card">
            <h3 style="margin-top: 0; color: #f8fafc;">⚡ 引擎遥测状态</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 0.8rem;">
                <div style="background: rgba(8,12,20,0.6); padding: 0.8rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #38bdf8;">66 / 66</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">单元测试通过</div>
                </div>
                <div style="background: rgba(8,12,20,0.6); padding: 0.8rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #34d399;">&lt; 1e-5</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">梯度校验误差</div>
                </div>
                <div style="background: rgba(8,12,20,0.6); padding: 0.8rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #fbbf24;">4 大类</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">经典优化算法</div>
                </div>
                <div style="background: rgba(8,12,20,0.6); padding: 0.8rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #c084fc;">5 种</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">非线性数据集</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 底部版权声明
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 2rem;">
        Crafted with precision by <a href="https://github.com/yuan278501381" style="color: #38bdf8; text-decoration: none;">Yy1 (yuan278501381)</a>
        · MIT License · Copyright (c) 2026
    </div>
    """,
    unsafe_allow_html=True,
)
