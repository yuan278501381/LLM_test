# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 8: Transformer
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from dashboard.styles.theme import apply_custom_theme, render_hero_header, render_page_guide
from dashboard.components.charts import plot_attention_heatmap_nlp, _apply_light_theme

st.set_page_config(page_title="Transformer", layout="wide")
apply_custom_theme()

render_hero_header(
    title="Transformer 结构块",
    subtitle="注意力机制、前馈网络与残差高速公路的完美组装",
    badge_text="MILESTONE 08 // TRANSFORMER BLOCK",
    badge_type="amber",
)

render_page_guide(
    title="M08 · Transformer 结构块 // Transformer Block",
    plain_intro="Transformer 就是把注意力机制、前馈网络和残差高速公路组装成一个'超级积木块'，然后把这些积木一层层堆高。每一层都像一位专家，负责理解句子的不同层面——第一层看语法，第二层看语义，第三层看推理。",
    hyperparams_desc="• <b>层数选择</b>：堆叠的积木数量。层数越多，抽象理解能力越强。<br>• <b>d_model/注意力头数</b>：决定了信息带宽和同时关注的维度数。",
    telemetry_desc="• <b>上：架构流程图</b>：展示信号穿过一个 Block 的路径。<br>• <b>中：逐层注意力热力图</b>：观察不同层如何关注不同内容（如底层看相邻词，高层看语义相关词）。<br>• <b>下：残差流向量范数</b>：观察残差连接如何保留并累积原始信息量。",
    experiments=[
        "<b>第 1 步</b>：调整层数，观察下方逐层注意力热力图的变化。一般深层更倾向于长距离语义关联。",
        "<b>第 2 步</b>：查看残差流图表，理解为什么哪怕网络很深，梯度也能通过短路连接顺畅回传。",
    ],
)

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")
text_input = st.sidebar.text_input("输入句子", "The quick brown fox", key="tf_txt")
num_layers = st.sidebar.selectbox("层数", [1, 2, 3, 4], index=1)
d_model = st.sidebar.selectbox("d_model", [32, 64], index=0)
num_heads = st.sidebar.selectbox("注意力头数", [2, 4], index=0)

tokens = text_input.strip().split()
if not tokens: tokens = ["Empty"]
seq_len = len(tokens)

# 上半部：流程图
st.markdown("#### 🧱 单个 Transformer Block 架构流")
st.code("""
      [ Input Representation / Residual Stream ]
                         |
           +-------------+-------------+
           |                           |
           v                           |
    [ Layer Normalization ]            |
           |                           |
    [ Multi-Head Attention ]           |
           |                           |
           +------------(+)------------+  <-- Residual Addition
                         |
           +-------------+-------------+
           |                           |
           v                           |
    [ Layer Normalization ]            |
           |                           |
    [ Feed Forward Network ]           |
           |                           |
           +------------(+)------------+  <-- Residual Addition
                         |
                 [ Output Stream ]
""")

# 中部：逐层注意力热力图
st.markdown("---")
st.markdown(f"#### 🔍 逐层注意力分布对比 (Layers = {num_layers})")
cols = st.columns(num_layers)
for i in range(num_layers):
    with cols[i]:
        # 模拟不同层的注意力权重差异
        attn = np.random.rand(seq_len, seq_len)
        attn[np.triu_indices(seq_len, 1)] = 0
        if i == 0:
            # 底层偏向对角线（关注相邻）
            attn += np.eye(seq_len) * 2
        else:
            # 高层更加全局
            pass
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
        fig = plot_attention_heatmap_nlp(attn, tokens, tokens, title=f"Layer {i+1} Attention")
        st.plotly_chart(fig, use_container_width=True)

# 底部：残差流向量范数
st.markdown("---")
st.markdown("#### 📈 残差流累积信息量 (Residual Stream Norm)")

# 模拟范数增长
layer_indices = list(range(num_layers + 1))
norms = [np.sqrt(d_model)]
for i in range(num_layers):
    # 每次加上一些正交更新，导致范数变大
    norms.append(np.sqrt(norms[-1]**2 + np.random.uniform(0.5, 2.0)))

fig_res = go.Figure(go.Scatter(
    x=layer_indices, 
    y=norms,
    mode="lines+markers",
    line=dict(width=3, color="#047857"),
    marker=dict(size=10)
))
fig_res.update_layout(
    xaxis_title="Layer Index (0 = Input Embedding)",
    yaxis_title="L2 Norm of Representation",
    xaxis=dict(tickmode="linear", tick0=0, dtick=1)
)
fig_res = _apply_light_theme(fig_res, "RESIDUAL NORM // 残差累积")
st.plotly_chart(fig_res, use_container_width=True)
