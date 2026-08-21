# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 7: 注意力机制
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.styles.theme import apply_custom_theme, render_hero_header, render_page_guide
from dashboard.components.charts import plot_attention_heatmap_nlp

try:
    from nn_core.attention import MultiHeadAttention
except ImportError:
    class MultiHeadAttention:
        def __init__(self, d_model, num_heads): 
            self.num_heads = num_heads
        def forward(self, q, k, v, use_scale=True, mask=True):
            seq_len = q.shape[0]
            attn = np.random.rand(seq_len, seq_len)
            if mask:
                attn[np.triu_indices(seq_len, 1)] = 0
            # normalize
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
            return np.random.randn(seq_len, q.shape[-1]), attn

st.set_page_config(page_title="注意力机制", layout="wide")
apply_custom_theme()

render_hero_header(
    title="注意力机制",
    subtitle="彻底打破距离束缚的动态信息路由",
    badge_text="MILESTONE 07 // ATTENTION MECHANISM",
    badge_type="emerald",
)

render_page_guide(
    title="M07 · 注意力机制 // Attention Mechanism",
    plain_intro="注意力机制就像考试时的'开卷查阅'——不用死记硬背所有内容，而是在需要时直接翻到最相关的那一页！每个词都能直接'回头看'所有之前的词，不受距离限制。",
    hyperparams_desc="• <b>输入句子</b>：观察不同长度词序列间的注意力分配。<br>• <b>注意力头数</b>：多个头就像多组不同的探照灯，从语法、语义等不同维度关注信息。<br>• <b>缩放因子开关</b>：观察除以 √d_k 是如何防止 Softmax 梯度消失的。",
    telemetry_desc="• <b>中：核心热力图</b>：Q与K的点积分布。因果掩码区域显示为无权重的灰色。<br>• <b>右：缩放对比</b>：观察未缩放时的分布极端化。<br>• <b>底：RNN vs Attention</b>：记忆能力的终极对比。",
    experiments=[
        "<b>第 1 步</b>：关闭缩放因子，观察权重分布是否变成极端的一个点（Softmax饱和）。",
        "<b>第 2 步</b>：调整注意力头数，想象每个头可能代表的不同关注逻辑（主谓匹配、代词指代等）。",
        "<b>第 3 步</b>：查看因果掩码（右上角），理解为什么在生成下一个词时，模型不能'偷看'未来的词。",
    ],
)

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")
text_input = st.sidebar.text_input("输入句子", "The quick brown fox jumps over the lazy dog", key="attn_txt")
num_heads = st.sidebar.selectbox("注意力头数", [1, 2, 4], index=0)
use_scale = st.sidebar.checkbox("缩放因子 (除以 √d_k)", value=True)
d_model = st.sidebar.selectbox("d_model", [16, 32, 64], index=1)

tokens = text_input.strip().split()
if not tokens: tokens = ["Empty"]
seq_len = len(tokens)

# 模拟输入嵌入
X = np.random.randn(seq_len, d_model)
attn_layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
_, attn_weights_scaled = attn_layer.forward(X, X, X, use_scale=True, mask=True)
_, attn_weights_unscaled = attn_layer.forward(X, X, X, use_scale=False, mask=True)

attn_to_plot = attn_weights_scaled if use_scale else attn_weights_unscaled

# 上半部：QKV 分解示意
st.markdown("#### 🔍 Q, K, V 投影分解")
col_q, col_k, col_v = st.columns(3)
with col_q:
    st.info("**Query (Q)**\n\n'我在寻找什么？'\n\n每个词生成一个查询向量，用来衡量它对其他词的需求。")
with col_k:
    st.warning("**Key (K)**\n\n'我包含什么信息？'\n\n每个词生成一个键向量，用来回应其他词的查询请求。")
with col_v:
    st.success("**Value (V)**\n\n'我的实际内容'\n\n当匹配成功后，提供给对方的实际信息负载。")

# 中部
st.markdown("---")
col_heat, col_cmp = st.columns([1.5, 1])
with col_heat:
    fig_attn = plot_attention_heatmap_nlp(attn_to_plot, tokens, tokens, title="CAUSAL ATTENTION WEIGHTS // 因果注意力矩阵")
    st.plotly_chart(fig_attn, use_container_width=True)

with col_cmp:
    st.markdown("#### ⚖️ 缩放因子效应对比 (Scale Factor)")
    st.markdown("未缩放 $QK^T$ 会导致 Softmax 趋近 one-hot，产生梯度消失。")
    fig_scale = plot_attention_heatmap_nlp(attn_weights_scaled, tokens, tokens, title="有缩放 (Smoother)")
    st.plotly_chart(fig_scale, use_container_width=True)
    fig_unscale = plot_attention_heatmap_nlp(attn_weights_unscaled, tokens, tokens, title="无缩放 (Sharper)")
    st.plotly_chart(fig_unscale, use_container_width=True)

# 底部对比
st.markdown("---")
st.markdown("#### ⚔️ 终极对决：RNN 遗忘瓶颈 vs Attention 全局路由")
col_rnn, col_att = st.columns(2)
with col_rnn:
    st.error("RNN: 随着距离增加，记忆呈指数级衰减。长句子头部信息几乎丢失。")
with col_att:
    st.success("Attention: 无论距离多远，只要 Q 和 K 匹配（内积大），信号就能以 O(1) 路径无损传达！")
