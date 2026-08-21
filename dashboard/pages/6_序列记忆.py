# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 6: 序列记忆与遗忘瓶颈 (Sequence Memory & Forgetting) - 零基础入门保姆级教学平台

解剖循环神经网络 (RNN) 的隐藏状态传递机制、长期序列信息压缩与记忆衰减的数学瓶颈。
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

from dashboard.components.charts import plot_memory_decay_heatmap
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
    render_sequence_flow,
)
from nn_core.embeddings import Embedding, get_mini_vocab, get_pretrained_embeddings
from nn_core.rnn import RNNCell

st.set_page_config(
    page_title="Sequence Memory · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="序列记忆与遗忘瓶颈",
    subtitle="解剖循环神经网络 (RNN) 的时序记忆机理：隐藏状态递推 $h_t = \\tanh(x_t W_{xh} + h_{t-1} W_{hh} + b)$ 与长程遗忘瓶颈",
    badge_text="MILESTONE 06 // SEQUENCE & FORGETTING",
    badge_type="purple",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="序列记忆与遗忘瓶颈入门",
    plain_intro=(
        "<b>RNN 就像一个记性有限的听书人</b>。<br>"
        "当听到一句话时，它用一个固定大小的<b>隐藏状态向量 $h_t$（相当于短期大脑记忆）</b>顺次吸收每个词。<br>"
        "每读一个新词，旧的记忆就会被压缩、覆盖一部分。<br>"
        "⚠️ <b>致命缺陷</b>：当句子达到 10~20 个词以上时，最开头的关键信息（如主语是谁）就会被彻底稀释忘光！"
    ),
    hyperparams_desc=(
        "• <b>预设测试句子</b>：选择短句（5词）或长句（15+词）观察记忆衰减现象。<br>"
        "• <b>RNN 隐藏层维度 (Hidden Dim)</b>：相当于记忆背包的容量（8/16/32 维）。<br>"
        "• <b>记忆保持系数</b>：调节循环权重对历史状态的衰减速率。"
    ),
    telemetry_desc=(
        "• <b>时序流水线色块</b>：直观展示每一个时间步的词汇与记忆激活强度。<br>"
        "• <b>记忆留存衰减热力图</b>：展示第 $t$ 步时对第 $k$ 步历史词汇的记忆强度（越靠右上角颜色越浅代表遗忘越严重）。<br>"
        "• <b>句首主语留存率</b>：量化评估长程关键信息的衰减比例。"
    ),
    experiments=[
        "<b>第 1 步【体验短句记忆】</b>：在左侧选择 <code>短句测试 (5 词)</code>，观察下方热力图每个词之间都有深蓝色的连接，记忆留存率高达 80% 以上！",
        "<b>第 2 步【目睹长句遗忘灾难】</b>：切换为 <code>超长叙事句 (16 词)</code>，观察热力图右上角大面积变白！读到最后几个词时，对句首'king'的记忆几乎彻底归零！",
        "<b>第 3 步【尝试增大容量】</b>：把【隐藏层维度】调到 32，观察虽然略微改善，但根本无法解决固定容量压缩的瓶颈——从而理解<b>为什么 2017 年 Attention 机制颠覆了 RNN</b>！",
    ],
)

# ---------------------------------------------------------------------------
# 词表与模型初始化
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
embed_weights = get_pretrained_embeddings(len(vocab_words), d_model=32)

embedding_layer = Embedding(vocab_size=len(vocab_words), d_model=32)
embedding_layer.weights = embed_weights

# ---------------------------------------------------------------------------
# 侧边栏参数面板
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与输入")

sentence_presets = {
    "短句测试 (5 词 - 短程易记)": "the king and queen sleep",
    "中等长度句 (9 词 - 记忆开始衰减)": "the cat sat on the mat and sleep well",
    "超长叙事句 (16 词 - 严重遗忘瓶颈)": "the king of china went to paris and then the queen traveled to rome",
    "自定义输入 (Custom Input)...": "",
}

selected_preset = st.sidebar.selectbox(
    "测试句子预设",
    list(sentence_presets.keys()),
    index=2,
    help="对比不同长度文本在 RNN 中的记忆留存表现。",
)

if "自定义" in selected_preset:
    input_text = st.sidebar.text_input(
        "自定义英文句子",
        "the dog and cat run fast and then sleep on the big cold mat",
        help="请输入空格分隔的英文单词（优先使用词表内的常见词汇）。",
    )
else:
    input_text = sentence_presets[selected_preset]

hidden_dim = st.sidebar.select_slider(
    "RNN 隐藏状态容量 (Hidden Dim)",
    options=[8, 16, 32],
    value=16,
    help="循环神经网络隐藏层神经元数量。容量越大能记住的信息相对略多，但无法突破理论瓶颈。",
)

# ---------------------------------------------------------------------------
# 分词与 RNN 逐步前向计算
# ---------------------------------------------------------------------------
raw_tokens = [w.lower().strip(",.!?") for w in input_text.strip().split() if w.strip()]
if not raw_tokens:
    raw_tokens = ["the", "cat", "sleep"]

# 映射到词表
token_indices = [raw_vocab.get(w, 0) for w in raw_tokens]
tokens_array = np.array([token_indices])

# 提取词嵌入: (1, seq_len, 32)
embedded_seq = embedding_layer.forward(tokens_array)

# 实例化 RNN 单元并逐步处理
rnn_cell = RNNCell(input_size=32, hidden_size=hidden_dim)
hidden_states_list = rnn_cell.step_sequence(embedded_seq)

seq_len = len(raw_tokens)

# 计算句首词 (t=0) 在最后一步 (t=T-1) 的记忆余弦相似度
h_first = hidden_states_list[0].ravel()
h_last = hidden_states_list[-1].ravel()
retention_rate = float(
    np.abs(np.dot(h_first, h_last)) / (np.linalg.norm(h_first) * np.linalg.norm(h_last) + 1e-12)
)

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "SEQUENCE LENGTH // 序列总长度",
        f"{seq_len} TOKENS",
        delta="长文本模式" if seq_len > 10 else "短文本模式",
        delta_type="neutral",
        icon_name="database",
    )
    + render_metric_card(
        "HEAD RETENTION // 句首记忆留存率",
        f"{retention_rate * 100:.1f}%",
        delta="严重遗忘 (SEVERE)" if retention_rate < 0.35 else ("轻度衰减" if retention_rate < 0.65 else "留存良好"),
        delta_type="negative" if retention_rate < 0.35 else ("neutral" if retention_rate < 0.65 else "positive"),
        icon_name="activity",
    )
    + render_metric_card(
        "HIDDEN CAPACITY // 记忆背包维度",
        f"{hidden_dim}-DIM",
        delta="固定容量瓶颈",
        delta_type="neutral",
        icon_name="cpu",
    )
    + render_metric_card(
        "BOTTLENECK STATUS // 信息漏斗状态",
        "FORGETTING" if seq_len > 8 else "STABLE",
        delta="需 Attention 拯救" if seq_len > 8 else "信息未溢出",
        delta_type="negative" if seq_len > 8 else "positive",
        icon_name="target",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 主视图区 1：时序流水线逐步流动
# ---------------------------------------------------------------------------
render_section_heading("STEP-BY-STEP RECURRENT FLOW // RNN 隐状态顺次传递管道", icon_name="activity")
render_sequence_flow(raw_tokens, hidden_states_list)

# ---------------------------------------------------------------------------
# 主视图区 2：记忆衰减热力图
# ---------------------------------------------------------------------------
render_section_heading("MEMORY DECAY HEATMAP // 序列记忆与历史衰减矩阵", icon_name="target")

fig_decay = plot_memory_decay_heatmap(
    hidden_states=hidden_states_list,
    tokens=raw_tokens,
    title="RNN RECURRENT MEMORY DECAY // 隐状态历史关联度矩阵 (越靠右上角颜色越浅代表遗忘)",
)
st.plotly_chart(fig_decay, use_container_width=True)

# ---------------------------------------------------------------------------
# 底部理论对比卡片：RNN 遗忘瓶颈 vs Attention 动态路由
# ---------------------------------------------------------------------------
render_section_heading("ARCHITECTURAL EVOLUTION // 为什么必须从 RNN 进化到 Attention？", icon_name="zap")

col_rnn, col_attn = st.columns(2)
with col_rnn:
    with st.container(border=True):
        st.markdown(
            """
            #### ⚠️ 循环神经网络 (RNN) 的致命局限
            - **固定容量漏斗 (Information Bottleneck)**：
              无论句子是 5 个词还是 1000 个词，所有历史信息都被强制压缩到同一个固定大小的 $h_t$ 向量中；
            - **长程梯度消失 (Vanishing Gradients)**：
              反向传播需要跨越数十个时间步长做连乘（BPTT），前端梯度呈指数级衰减为零；
            - **无法并行计算**：
              第 $t$ 步的计算必须等待第 $t-1$ 步完成，完全无法利用现代 GPU 的数千个核心做矩阵并行加速。
            """
        )

with col_attn:
    with st.container(border=True):
        st.markdown(
            """
            #### 💡 注意力机制 (Attention) 的范式革命
            - **$O(1)$ 任意距离直达路由**：
              取消递归传递链，每一个词都能以光速直接“回头看”整个句子的所有词汇；
            - **全局矩阵并行共振**：
              所有 Token 的 Query/Key/Value 矩阵一次性送入 GPU，单步完成 $N \times N$ 全对全关联计算；
            - **动态聚光灯 (Dynamic Spotlight)**：
              根据上下文动态分配权重，彻底攻克长程记忆与代词回指难题！
            """
        )
