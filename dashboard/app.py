# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
🧠 NN Playground - 手搓神经网络可视化实验平台

Streamlit 主入口 — 多页面应用的首页。
"""

import os
import sys

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="🧠 NN Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 首页内容
# ---------------------------------------------------------------------------
st.markdown(
    """
    # 🧠 NN Playground
    ### 手搓神经网络 · 可视化实验平台

    > **从零用纯 NumPy 手写神经网络，通过交互式仪表板观察每一个参数对结果的影响。**

    ---
    """
)

# 4 个里程碑卡片
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### 🎯 里程碑 1: 单神经元感知器
        理解最小计算单元的完整闭环：
        - 前向传播 $Z = XW + b$
        - 损失计算
        - 反向传播（梯度计算）
        - 参数更新

        **可视化**: 决策边界、Loss 曲线、权重轨迹
        """
    )

    st.markdown(
        """
        ### ⚙️ 里程碑 3: 优化器对比
        对比 4 种主流优化器的收敛行为：
        - SGD → Momentum → RMSProp → Adam
        - 同一问题，不同优化策略

        **可视化**: 多优化器 Loss 对比、决策边界并排
        """
    )

with col2:
    st.markdown(
        """
        ### 🧱 里程碑 2: 多层网络
        理解「深度」的力量：
        - 多层堆叠与链式法则
        - 梯度消失 / 爆炸
        - 权重初始化的影响

        **可视化**: 网络拓扑图、逐层激活热力图、梯度直方图
        """
    )

    st.markdown(
        """
        ### 🔬 里程碑 4: 参数实验室
        全参数交互式实验台：
        - 所有超参数均可实时调整
        - A/B 对比模式
        - 逐步训练 & 快照

        **可视化**: 四宫格全维度监控
        """
    )

st.markdown("---")

# 技术栈
st.markdown(
    """
    ### 🔧 技术栈

    | 组件 | 技术 | 说明 |
    |------|------|------|
    | 核心引擎 | **纯 NumPy** | 手写前向传播、反向传播、优化器 |
    | 可视化 | **Streamlit + Plotly** | 交互式仪表板 + 高质量图表 |
    | 数据集 | **sklearn** | 仅用于生成 2D 合成数据 |
    | 测试 | **pytest** | 梯度数值校验 + 单元测试 |

    ---

    <div style="text-align: center; color: gray; font-size: 0.9em;">
        Made with ❤️ by <a href="https://github.com/yuan278501381">Yy1 (yuan278501381)</a>
        · MIT License · Copyright (c) 2026
    </div>
    """,
    unsafe_allow_html=True,
)
