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
from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
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
render_lesson_evidence("M06")

render_hero_header(
    title="序列记忆与遗忘瓶颈",
    subtitle="解剖循环神经网络 (RNN) 的时序记忆机理：隐藏状态递推 $h_t = \\tanh(x_t W_{ih} + h_{t-1} W_{hh} + b_h)$ 与长程遗忘瓶颈",
    badge_text="MILESTONE 06 // SEQUENCE & FORGETTING",
    badge_type="purple",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="序列记忆与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "时序输入控制台",
            "desc": "在左侧侧边栏切换短句、长句或自定义句子与隐藏层维度",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解 RNN 隐藏状态背包与长程遗忘致命缺陷",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时记忆遥测",
            "desc": "显示当前序列长度、句首信息保留率与时间步记忆衰减",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "时序流水线色块",
            "desc": "按时间步展示每一个 Token 的隐藏状态向量流动与激活强度",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "记忆衰减热力图",
            "desc": "直观验证右上角大面积褪色白块，见证信息稀释与注意力诞生背景",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>RNN 就像一个记性有限的听书人</b>。<br>"
        f"当听到一句话时，它用一个固定大小的<b>隐藏状态向量 $h_t$（相当于短期大脑记忆）</b>顺次吸收每个词。<br>"
        f"每读一个新词，旧的记忆就会被压缩、覆盖一部分。<br>"
        f"[WARNING] <b>致命缺陷</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 换上长句时，最开头的关键信息（如主语是谁）在 {anchor_badge('[E. 记忆衰减热力图]', 'blue', target_id='region-e')} 就会被彻底稀释忘光！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>预设测试句子</b>：选择短句（5词）或长句（15+词）观察记忆衰减现象。<br>"
        f"• <b>RNN 隐藏层维度 (Hidden Dim)</b>：相当于记忆背包的容量（8/16/32 维）。<br>"
        f"• <b>记忆保持系数</b>：调节循环权重对历史状态的衰减速率。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 时序流水线]', 'purple', target_id='region-d')} 观察</b>：直观展示每一个时间步的词汇与记忆激活强度。<br>"
        f"• <b>在 {anchor_badge('[E. 记忆留存热力图]', 'blue', target_id='region-e')} 诊断</b>：展示第 $t$ 步时对第 $k$ 步历史词汇的记忆强度（越靠右上角颜色越浅代表遗忘越严重）。<br>"
        f"• <b>在 {anchor_badge('[C. 记忆遥测指标]', 'emerald', target_id='region-c')} 评估</b>：量化评估句首关键信息的衰减比例。"
    ),
    experiments=[
        f"<b>第 1 步【体验短句记忆】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 选择 <code>短句测试 (5 词)</code>，观察下方热力图每个词之间都有深蓝色的连接，记忆留存率高达 80% 以上！",
        f"<b>第 2 步【目睹长句遗忘灾难】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 切换为 <code>超长叙事句 (16 词)</code>，观察 {anchor_badge('[E. 记忆热力图]', 'blue', target_id='region-e')} 右上角大面积变白！读到最后几个词时，对句首'king'的记忆几乎彻底归零！",
        f"<b>第 3 步【尝试增大容量】</b>：把【隐藏层维度】调到 32，观察虽然略微改善，但根本无法解决固定容量压缩的瓶颈——从而理解<b>为什么 2017 年 Attention 机制颠覆了 RNN</b>！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>SEQUENCE CONTROLS // 时序输入控制台</b></div>',
    unsafe_allow_html=True,
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
first_token_retention = float(
    np.abs(np.dot(h_first, h_last)) / (np.linalg.norm(h_first) * np.linalg.norm(h_last) + 1e-12)
)
retention_rate = first_token_retention

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">MEMORY TELEMETRY // 序列记忆与衰减遥测看板</span>'
    f"</div>",
    unsafe_allow_html=True,
)
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
        delta="严重遗忘 (SEVERE)"
        if retention_rate < 0.35
        else ("轻度衰减" if retention_rate < 0.65 else "留存良好"),
        delta_type="negative"
        if retention_rate < 0.35
        else ("neutral" if retention_rate < 0.65 else "positive"),
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
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">STEP-BY-STEP RECURRENT FLOW // RNN 隐状态顺次传递流水线</span>'
    f"</div>",
    unsafe_allow_html=True,
)
render_sequence_flow(raw_tokens, hidden_states_list)

# ---------------------------------------------------------------------------
# 主视图区 2：记忆衰减热力图
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">MEMORY DECAY HEATMAP // 序列记忆与历史衰减矩阵</span>'
    f"</div>",
    unsafe_allow_html=True,
)

fig_decay = plot_memory_decay_heatmap(
    hidden_states=hidden_states_list,
    tokens=raw_tokens,
    title="RNN RECURRENT MEMORY DECAY // 隐状态历史关联度矩阵 (越靠右上角颜色越浅代表遗忘)",
)
st.plotly_chart(fig_decay, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 时间步传递与长程遗忘瓶颈", expanded=False):
    st.markdown(
        """
        * **横轴与纵轴**：句子中的时间步词汇（从第 1 个词到最后一个词）。每个方格的颜色代表第 $t$ 步的隐藏状态与之前第 $t-k$ 步状态的**余弦相似度（记忆保留度）**。
        * **对角线（亮黄色）**：自己与自己的相似度恒为 1.0。
        * [WARNING] **【读图重点：右上角褪色】**：
          * 观察矩阵的右上角区域：距离当前词越远的早期词汇，颜色越**暗淡变冷**（接近 0）；
          * 这直观证明了**RNN 具有严重的物理遗忘性**——随着时间推移，最早输入的词（如句首的“国王”）被后来输入的词冲淡冲没了！
        """
    )

# ---------------------------------------------------------------------------
# 底部理论对比卡片：RNN 遗忘瓶颈 vs Attention 动态路由
# ---------------------------------------------------------------------------
render_section_heading(
    "ARCHITECTURAL EVOLUTION // 为什么必须从 RNN 进化到 Attention？", icon_name="zap"
)

col_rnn, col_attn = st.columns(2)
with col_rnn:
    with st.container(border=True):
        st.markdown(
            """
            #### [BOTTLENECK // 遗忘瓶颈] 循环神经网络 (RNN) 的物理极限
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
            #### [PARADIGM SHIFT // 范式跃迁] 注意力机制 (Attention) 全局路由
            - **$O(1)$ 任意距离直达路由**：
              取消递归传递链，每一个词都能以光速直接“回头看”整个句子的所有词汇；
            - **全局矩阵并行共振**：
              所有 Token 的 Query/Key/Value 矩阵一次性送入 GPU，单步完成 $N \times N$ 全对全关联计算；
            - **动态聚光灯 (Dynamic Spotlight)**：
              根据上下文动态分配权重，缩短信息交互路径；长程记忆和指代仍取决于训练数据、容量与上下文长度。
            """
        )

# ---------------------------------------------------------------------------
# 零基础进阶：RNN 核心递归公式逐字拆解与名词通俗速查
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] RNN 核心公式拆解与序列时序名词通俗全解", expanded=False
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：RNN 单步隐状态递归传递
        $$h_t = \\tanh(x_t W_{ih} + h_{t-1} W_{hh} + b_h)$$
        
        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$x_t$** | **当前时间步输入词向量 (Current Input)** | $(1, d_{\\text{in}})$，如 $(1, 32)$ | 当前时刻读到的新单词（例如"猫咪"的嵌入向量）。 |
        | **$W_{ih}$** | **输入-隐层权重矩阵 (Input-to-Hidden)** | $(d_{\\text{in}}, d_h)$，如 $(32, 64)$ | 负责把当前新单词的特征提取转换并存入记忆。 |
        | **$h_{t-1}$** | **上一时刻的历史记忆 (Previous Memory)** | $(1, d_h)$，如 $(1, 64)$ | 截止到上一个词为止，大脑里已经记住的全部故事背景。 |
        | **$W_{hh}$** | **隐层-隐层递归权重矩阵 (Hidden-to-Hidden)** | $(d_h, d_h)$，如 $(64, 64)$ | **记忆过滤器**。决定把上一时刻的历史记忆保留多少、按什么方式融入当前时刻。 |
        | **$b_h$** | **循环偏置常数 (Bias)** | $(1, d_h)$，如 $(1, 64)$ | 记忆状态的基础偏移。 |
        | **$\\tanh$** | **双曲正切激活函数** | 逐元素 | 把加权后的总记忆平滑压缩到 $[-1, 1]$ 之间，防止记忆数值越滚越大发生数值爆炸。 |
        | **$h_t$** | **当前时刻综合新记忆 (New Memory)** | $(1, d_h)$，如 $(1, 64)$ | 融合了"新词 $x_t$"与"旧记忆 $h_{t-1}$"后的最新大脑状态，准备传给下一个时间步 $t+1$。 |
        
        ---
        
        ### 1. 什么是【BPTT (沿时间反向传播)】？—— “顺着时光隧道找原因”
        * **生活比喻**：期末考试做错了一道大题，倒推回去发现是上周讲的第 5 个公式用错了。
        * **本质机理**：把时间步像手风琴一样拉开，误差梯度顺着时间链条 $t \\to t-1 \\to t-2$ 一路倒推求导。链条越长，梯度越容易衰减为 0（长程遗忘）。
        """
    )
