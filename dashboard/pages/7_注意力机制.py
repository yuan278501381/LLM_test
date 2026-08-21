# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 7: 注意力机制 (Attention Mechanism) - 零基础入门保姆级教学平台

解剖缩放点积自注意力 (Scaled Dot-Product Attention)、QKV 数据库检索隐喻、因果掩码 (Causal Mask) 与多头并行机制 (Multi-Head Attention)。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib
import numpy as np
import streamlit as st

import dashboard.components.charts
importlib.reload(dashboard.components.charts)

from dashboard.components.charts import plot_attention_heatmap_nlp
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention
from nn_core.embeddings import Embedding, get_mini_vocab, get_pretrained_embeddings

st.set_page_config(
    page_title="Attention Mechanism · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="注意力机制与动态路由",
    subtitle="解剖 Transformer 的核心灵魂：缩放点积注意力 $\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$、QKV 检索隐喻与因果掩码",
    badge_text="MILESTONE 07 // ATTENTION MECHANISM",
    badge_type="emerald",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="注意力机制入门",
    plain_intro=(
        "<b>注意力机制就像考试时的'开卷查阅'</b>。<br>"
        "它不再强迫模型死记硬背所有前文，而是让每一个词手握一个<b>搜索探针 (Query)</b>；<br>"
        "去和其他所有词的<b>索引标签 (Key)</b> 做点积匹配，算出相关度得分；<br>"
        "最后把最相关的词的<b>信息载荷 (Value)</b> 打包提取出来！<br>"
        "无论两个词相隔 5 个词还是 500 个词，<b>一步直达，毫无记忆损耗</b>！"
    ),
    hyperparams_desc=(
        "• <b>测试句子</b>：观察不同句子中词与词之间的注意力连线。<br>"
        "• <b>注意力头数 (Heads)</b>：相当于多组独立的探照灯，各自关注语法、语义或代词指代。<br>"
        "• <b>缩放因子开关 (1/√d_k)</b>：验证为什么高维点积必须除以 $\\sqrt{d_k}$ 防止 Softmax 极化失效。<br>"
        "• <b>因果掩码 (Causal Mask)</b>：确保生成模型只能看过去、绝不能偷看未来。"
    ),
    telemetry_desc=(
        "• <b>N×N 注意力热力图</b>：当前词（行）与被关注词（列）之间的注意力分配比例。<br>"
        "• <b>缩放 vs 未缩放对比</b>：直观对比数值稳定性技巧对概率分布平滑度的影响。<br>"
        "• <b>注意力信息熵</b>：衡量模型是把目光聚焦于少数关键词还是平均分摊。"
    ),
    experiments=[
        "<b>第 1 步【看懂聚光灯】</b>：观察默认句子中，词汇 <code>queen</code> 在处理时如何对 <code>king</code> 产生强烈的注意力响应（动宾与代词关联）！",
        "<b>第 2 步【体验缩放因子的神奇】</b>：在左侧关闭【缩放因子 (1/√d_k)】，观察右侧对比图：没有缩放时，矩阵瞬间极化为非 0 即 1 的极端值，梯度彻底消失！",
        "<b>第 3 步【观察因果掩码】</b>：注意右上角的灰色三角区域——这就是因果掩码（Causal Mask），严格禁止当前词偷看未来的内容！",
    ],
)

# ---------------------------------------------------------------------------
# 词表与嵌入层
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
embed_weights = get_pretrained_embeddings(len(vocab_words), d_model=32)

embedding_layer = Embedding(vocab_size=len(vocab_words), d_model=32)
embedding_layer.weights = embed_weights

# ---------------------------------------------------------------------------
# 侧边栏参数面板
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

sentence_options = {
    "代词指代与王族关系": "the king and queen ruled and she was happy",
    "动作与地点长程依赖": "the cat went to paris and then it sat on the big cold mat",
    "动物习性与状态": "the puppy and kitten run fast in china",
    "自定义输入 (Custom Input)...": "",
}

selected_sentence_label = st.sidebar.selectbox(
    "测试句子预设",
    list(sentence_options.keys()),
    index=0,
)

if "自定义" in selected_sentence_label:
    input_text = st.sidebar.text_input(
        "自定义英文句子",
        "the king and queen sleep on the mat",
    )
else:
    input_text = sentence_options[selected_sentence_label]

num_heads = st.sidebar.select_slider(
    "多头注意力头数 (Num Heads)",
    options=[1, 2, 4],
    value=2,
    help="Multi-Head Attention 的并行头数。每个头独立学习不同的投影空间。",
)

use_causal_mask = st.sidebar.checkbox(
    "启用因果掩码 (Causal Mask / 下三角)",
    value=True,
    help="用于 GPT 等自回归生成模型，确保第 t 个词只能关注位置 <= t 的前序词汇。",
)

enable_scale = st.sidebar.checkbox(
    "启用缩放因子 (Scale Factor 1/√d_k)",
    value=True,
    help="除以 sqrt(d_k) 使得点积方差保持为 1，防止高维向量点积数值过大导致 Softmax 梯度饱和进入死区。",
)

# ---------------------------------------------------------------------------
# Token 化与注意力前向传播
# ---------------------------------------------------------------------------
raw_tokens = [w.lower().strip(",.!?") for w in input_text.strip().split() if w.strip()]
if not raw_tokens:
    raw_tokens = ["the", "king", "queen"]

token_indices = [raw_vocab.get(w, 0) for w in raw_tokens]
tokens_array = np.array([token_indices])
seq_len = len(raw_tokens)

# 提取词向量: (1, seq_len, 32)
X_embed = embedding_layer.forward(tokens_array)

# 实例化 MultiHeadAttention
mha = MultiHeadAttention(d_model=32, num_heads=num_heads)

# 构造因果掩码
mask = causal_mask(seq_len) if use_causal_mask else None

# 前向计算
attn_output, attn_weights = mha.forward(X_embed, mask=mask)
# attn_weights shape: (1, num_heads, seq_len, seq_len)

# 选定当前展示的 Head 0 权重
head_0_weights = attn_weights[0, 0]

# 计算未缩放的注意力权重用于对比
d_k = 32 // num_heads
# 手动计算 Q, K
q = np.matmul(X_embed, mha.W_q)
k = np.matmul(X_embed, mha.W_k)
q_split = q.reshape(1, seq_len, num_heads, d_k).swapaxes(1, 2)
k_split = k.reshape(1, seq_len, num_heads, d_k).swapaxes(1, 2)

# 无缩放点积
scores_unscaled = np.matmul(q_split, np.swapaxes(k_split, -1, -2))
if mask is not None:
    scores_unscaled = np.where(mask == 0, -1e9, scores_unscaled)
exp_unscaled = np.exp(scores_unscaled - np.max(scores_unscaled, axis=-1, keepdims=True))
weights_unscaled = exp_unscaled / (np.sum(exp_unscaled, axis=-1, keepdims=True) + 1e-12)
head_0_unscaled = weights_unscaled[0, 0]

# 计算注意力信息熵: H = -sum(p * log(p))
eps = 1e-12
valid_mask = ~np.isnan(head_0_weights) & (head_0_weights > 0)
entropy = -float(np.mean(np.sum(head_0_weights * np.log(head_0_weights + eps), axis=-1)))
max_attn_val = float(np.max(head_0_weights))

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "MULTI-HEADS // 注意力头数",
        f"{num_heads} HEADS",
        delta=f"每个头 d_k={32//num_heads}",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "SCALE FACTOR // 缩放系数 1/√d_k",
        f"{1.0/np.sqrt(d_k):.3f}",
        delta="已启用 (STABLE)" if enable_scale else "已关闭 (SATURATED)",
        delta_type="positive" if enable_scale else "negative",
        icon_name="activity",
    )
    + render_metric_card(
        "ATTENTION ENTROPY // 注意力分布熵",
        f"{entropy:.2f} nats",
        delta="动态聚焦中",
        delta_type="positive" if entropy > 0.5 else "neutral",
        icon_name="target",
    )
    + render_metric_card(
        "MAX PEAK // 单点最大注意力",
        f"{max_attn_val * 100:.1f}%",
        delta="适度聚焦" if max_attn_val < 0.9 else "严重极化",
        delta_type="positive" if max_attn_val < 0.9 else "negative",
        icon_name="zap",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 主视图区 1：QKV 数据库检索隐喻
# ---------------------------------------------------------------------------
render_section_heading("QKV SEARCH METAPHOR // Query-Key-Value 数据库检索隐喻", icon_name="activity")

col_q, col_k, col_v = st.columns(3)
with col_q:
    with st.container(border=True):
        st.markdown(
            f"#### 🔍 Query (查询探针)\n"
            f"- **物理意义**：“当前词正在寻找什么？”\n"
            f"- **张量维度**：`({seq_len}, {32//num_heads})`\n"
            f"- **生成方式**：$Q = X \\cdot W_Q$\n"
            f"- **作用**：主动发起检索请求，与所有 Key 进行相似度点积。"
        )
with col_k:
    with st.container(border=True):
        st.markdown(
            f"#### 🏷️ Key (索引标签)\n"
            f"- **物理意义**：“我包含什么特征来响应查询？”\n"
            f"- **张量维度**：`({seq_len}, {32//num_heads})`\n"
            f"- **生成方式**：$K = X \\cdot W_K$\n"
            f"- **作用**：被动接受匹配，$Q \\cdot K^T$ 产生 $N \\times N$ 相似度分数。"
        )
with col_v:
    with st.container(border=True):
        st.markdown(
            f"#### 📦 Value (信息载荷)\n"
            f"- **物理意义**：“我的实际语义内容”\n"
            f"- **张量维度**：`({seq_len}, {32//num_heads})`\n"
            f"- **生成方式**：$V = X \\cdot W_V$\n"
            f"- **作用**：按照 Softmax 归一化权重 $\\alpha$ 进行加权求和输出。"
        )

# ---------------------------------------------------------------------------
# 主视图区 2：核心注意力矩阵热力图 & 缩放效应对比
# ---------------------------------------------------------------------------
render_section_heading("ATTENTION MATRIX & SCALE FACTOR // 核心注意力权重热力图与缩放效应", icon_name="target")

col_main_heat, col_cmp_heat = st.columns([1.3, 1])

with col_main_heat:
    fig_main = plot_attention_heatmap_nlp(
        attention_weights=head_0_weights,
        tokens_x=raw_tokens,
        tokens_y=raw_tokens,
        title=f"CAUSAL ATTENTION MATRIX // 因果自注意力权重矩阵 (Head 0 / {num_heads})",
    )
    st.plotly_chart(fig_main, use_container_width=True)

with col_cmp_heat:
    st.markdown("#### ⚖️ 缩放因子效应对比 (Scale Comparison)")
    st.markdown("当高维空间 $d_k$ 很大时，点积的方差膨胀为 $d_k$。如果不除以 $\\sqrt{d_k}$，Softmax 会将概率推向极端的 0 和 1，导致梯度归零。")
    
    fig_scaled_thumb = plot_attention_heatmap_nlp(
        head_0_weights, raw_tokens, raw_tokens, title="✅ 有缩放 1/√d_k (平滑分布)"
    )
    st.plotly_chart(fig_scaled_thumb, use_container_width=True)
    
    fig_unscaled_thumb = plot_attention_heatmap_nlp(
        head_0_unscaled, raw_tokens, raw_tokens, title="❌ 无缩放 (极化饱和 / 梯度消失)"
    )
    st.plotly_chart(fig_unscaled_thumb, use_container_width=True)

# ---------------------------------------------------------------------------
# 底部理论卡片：因果掩码 (Causal Mask)
# ---------------------------------------------------------------------------
render_section_heading("CAUSAL MASKING // 为什么自回归模型必须使用因果掩码？", icon_name="zap")

col_mask_info, col_mask_math = st.columns(2)
with col_mask_info:
    with st.container(border=True):
        st.markdown(
            """
            #### 🛡️ 因果约束与单向自回归
            - **生成法则**：在 GPT 生成下一个词时，模型只能依赖**已经生成的历史词汇**，绝对不能跨越时间线偷看未来的词；
            - **视觉呈现**：注意力矩阵右上角被严格置为灰色（即负无穷 $-\\infty$），Softmax 后概率恒为 0；
            - **双向模型 vs 单向模型**：
              - **BERT (双向编码器)**：无掩码，每个词能看全句（做理解任务）；
              - **GPT (因果解码器)**：带下三角掩码，单向自回归生成（做生成任务）。
            """
        )

with col_mask_math:
    with st.container(border=True):
        st.markdown(
            """
            #### 📐 因果掩码的数学实现
            $$
            M_{ij} = \\begin{cases} 0, & i \\ge j \\text{ (允许关注)} \\\\ -\\infty, & i < j \\text{ (遮蔽未来)} \\end{cases}
            $$
            $$
            \\text{Scores} = \\frac{QK^T}{\\sqrt{d_k}} + M
            $$
            经过 Softmax 后：$e^{-\\infty} = 0$，上三角权重被精确清零！
            """
        )
