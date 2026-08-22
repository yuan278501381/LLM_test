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
import plotly.graph_objects as go
import streamlit as st

import dashboard.components.charts

importlib.reload(dashboard.components.charts)

from dashboard.components.charts import _apply_light_theme, plot_attention_heatmap_nlp
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.attention import MultiHeadAttention, causal_mask
from nn_core.embeddings import Embedding, get_mini_vocab, get_synthetic_demo_embeddings

st.set_page_config(
    page_title="Attention Mechanism · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M07", show_contract=True)
render_core_result_evidence("M07")

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
    title="注意力机制与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "注意力控制台",
            "desc": "在左侧侧边栏调节句子输入、多头数与缩放因子开关",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解 QKV 数据库检索开卷查阅与动态软路由",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时注意力遥测",
            "desc": "监测头数、缩放系数 1/√d_k、分布熵与最大峰值",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "QKV 检索隐喻",
            "desc": "直观拆解 Query 探针、Key 标签与 Value 内容三位一体",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "多头注意力热力图",
            "desc": "观察多头聚光灯在词与词之间的关联权重与因果三角掩码",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        "<b>注意力机制就像考试时的'开卷查阅'</b>。<br>"
        "它不再强迫模型死记硬背所有前文，而是让每一个词手握一个<b>搜索探针 (Query)</b>；<br>"
        "去和其他所有词的<b>索引标签 (Key)</b> 做点积匹配，算出相关度得分；<br>"
        "最后把最相关的词的<b>信息载荷 (Value)</b> 打包提取出来！<br><br>"
        "<b>【2026 前沿拓展】：RoPE 与 GQA</b><br>"
        "现代大模型（如 Llama-3）改用 <b>RoPE 旋转位置编码</b> 使点积自带相对距离衰减；"
        "同时采用 <b>GQA 分组查询</b> 让多个 Query 头共享 Key/Value，极大降低了显存开销！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>测试句子</b>：观察不同句子中词与词之间的注意力连线。<br>"
        f"• <b>注意力头数 (Heads)</b>：将特征拆分到多个子空间独立计算后拼接。经训练的头可能出现专门化，但本页未训练权重不支持语法/语义标签。<br>"
        f"• <b>缩放因子开关 (1/√d_k)</b>：验证为什么高维点积必须除以 $\\sqrt{{d_k}}$ 防止 Softmax 极化失效。<br>"
        f"• <b>因果掩码 (Causal Mask)</b>：确保生成模型只能看过去、绝不能偷看未来。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[E. 注意力热力图]', 'blue', target_id='region-e')} 观测</b>：当前词（行）与被关注词（列）之间的注意力分配比例。<br>"
        f"• <b>在 {anchor_badge('[D. QKV 隐喻]', 'purple', target_id='region-d')} 拆解</b>：Query 探针与 Key 标签的匹配机制。<br>"
        f"• <b>在 {anchor_badge('[C. 注意力遥测]', 'emerald', target_id='region-c')} 评估</b>：分布熵与峰值极化状态。"
    ),
    experiments=[
        "<b>第 1 步【看懂聚光灯】</b>：观察默认句子中，词汇 <code>queen</code> 在处理时如何对 <code>king</code> 产生强烈的注意力响应！",
        "<b>第 2 步【体验缩放因子】</b>：关闭 <code>1/√d_k</code> 并比较 logits 方差与 softmax 熟化程度。在常用独立分量假设下，未缩放点积方差随 $d_k$ 增长，增加饱和和小梯度风险；不保证任意输入都变成 0/1。",
        "<b>第 3 步【探索前沿架构】</b>：滚动到底部实验室，对比 RoPE 的相对距离热力图，并切换 GQA/MHA 选项，观察显存压缩比如何从 1.0x 暴涨至 4.0x！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>ATTENTION CONFIG // 注意力控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 词表与嵌入层
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
embed_weights = get_synthetic_demo_embeddings(len(vocab_words), d_model=32)

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

sent_opts = list(sentence_options.keys())
selected_sentence_label = st.sidebar.segmented_control(
    "测试句子预设",
    options=sent_opts,
    default=sent_opts[0],
)
selected_sentence_label = selected_sentence_label or sent_opts[0]

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
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">ATTENTION TELEMETRY // 注意力分布与极化遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "MULTI-HEADS // 注意力头数",
        f"{num_heads} HEADS",
        delta=f"每个头 d_k={32 // num_heads}",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "SCALE FACTOR // 缩放系数 1/√d_k",
        f"{1.0 / np.sqrt(d_k):.3f}",
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
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">QKV SEARCH METAPHOR // Query-Key-Value 数据库检索隐喻</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_q, col_k, col_v = st.columns(3)
with col_q, st.container(border=True):
    st.markdown(
        f"#### [QUERY // 查询探针]\n"
        f"- **物理意义**：“当前词正在寻找什么？”\n"
        f"- **张量维度**：`({seq_len}, {32 // num_heads})`\n"
        f"- **生成方式**：$Q = X \\cdot W_Q$\n"
        f"- **作用**：主动发起检索请求，与所有 Key 进行相似度点积。"
    )
with col_k, st.container(border=True):
    st.markdown(
        f"#### [KEY // 索引标签]\n"
        f"- **物理意义**：“我包含什么特征来响应查询？”\n"
        f"- **张量维度**：`({seq_len}, {32 // num_heads})`\n"
        f"- **生成方式**：$K = X \\cdot W_K$\n"
        f"- **作用**：被动接受匹配，$Q \\cdot K^T$ 产生 $N \\times N$ 相似度分数。"
    )
with col_v, st.container(border=True):
    st.markdown(
        f"#### [VALUE // 信息载荷]\n"
        f"- **计算意义**：被注意力权重加权求和的特征向量；是否编码语义取决于训练\n"
        f"- **张量维度**：`({seq_len}, {32 // num_heads})`\n"
        f"- **生成方式**：$V = X \\cdot W_V$\n"
        f"- **作用**：按照 Softmax 归一化权重 $\\alpha$ 进行加权求和输出。"
    )

# ---------------------------------------------------------------------------
# 主视图区 2：核心注意力矩阵热力图 & 缩放效应对比
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">ATTENTION MATRIX // 核心注意力多头权重矩阵与因果掩码</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="ATTENTION DYNAMICS // 缩放点积与注意力分布微观参数",
    badges=[
        {"label": "Scale Factor 1/√d_k", "value": f"{1.0 / np.sqrt(d_k):.3f}", "color": "blue"},
        {"label": "Entropy H", "value": f"{entropy:.2f} nats", "color": "emerald"},
        {"label": "Max Peak α", "value": f"{max_attn_val * 100:.1f}%", "color": "purple"},
        {
            "label": "Mask Type",
            "value": "Causal" if use_causal_mask else "Full Bidirectional",
            "color": "amber",
        },
    ],
    metrics=[
        ("多头数量 h", f"{num_heads}"),
        ("单头维度 d_k", f"{d_k}"),
        ("缩放状态", "ENABLED" if enable_scale else "DISABLED"),
    ],
    tag=f"ROUTING: {seq_len}x{seq_len} ATTENTION MATRIX",
    tag_color="emerald",
)

col_main_heat, col_cmp_heat = st.columns([1.3, 1])

with col_main_heat:
    fig_main = plot_attention_heatmap_nlp(
        attention_weights=head_0_weights,
        tokens_x=raw_tokens,
        tokens_y=raw_tokens,
        title=f"CAUSAL ATTENTION MATRIX // 因果自注意力权重矩阵 (Head 0 / {num_heads})",
    )
    st.plotly_chart(fig_main, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 因果自注意力热力图", expanded=False):
        st.markdown(
            """
            * **纵轴 (Query)**：“发出查询的当前词”；**横轴 (Key)**：“被查询关注的历史词”。
            * **颜色深浅**：方格越亮（亮黄/亮绿），代表当前词对该历史词的**注意力权重占比越高**（每一行的概率总和严格等于 $1.0$）。
            * **右上角全灰**：因果下三角掩码生效，防止当前词穿越时空偷看未来词。
            * **[观察要点]**：你可以查看某行权重分布，但本页权重未经语言任务训练；高亮连接不能解读为代词指代的证据。
            """
        )

with col_cmp_heat:
    st.markdown("#### [SCALE COMPARISON // 缩放因子效应对比]")
    st.caption(
        "当高维空间 $d_k$ 很大时，点积方差膨胀为 $d_k$。如果不除以 $\\sqrt{d_k}$，Softmax 会将概率推向极端 0 和 1 导致梯度死锁。"
    )

    fig_scaled_thumb = plot_attention_heatmap_nlp(
        head_0_weights, raw_tokens, raw_tokens, title="[SCALED] 1/√d_k 缩放 (平滑聚焦)"
    )
    st.plotly_chart(fig_scaled_thumb, width="stretch")

    fig_unscaled_thumb = plot_attention_heatmap_nlp(
        head_0_unscaled, raw_tokens, raw_tokens, title="[UNSCALED] 未缩放 (极化失效)"
    )
    st.plotly_chart(fig_unscaled_thumb, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 缩放因子有效性对比", expanded=False):
        st.markdown(
            """
            * **上方 [SCALED]**：除了最相关的词高亮外，其余词保留适当注意力过渡（软性 Softmax），梯度反向传播顺畅。
            * **下方 [UNSCALED]**：对照 logits 和权重的分布。未缩放点积在较大 $d_k$ 或较大输入范数下更可能饱和，具体程度取决于输入和权重。
            """
        )

# ---------------------------------------------------------------------------
# 底部理论卡片：因果掩码 (Causal Mask)
# ---------------------------------------------------------------------------
render_section_heading("CAUSAL MASKING // 为什么自回归模型必须使用因果掩码？", icon_name="zap")

col_mask_info, col_mask_math = st.columns(2)
with col_mask_info:
    with st.container(border=True):
        st.markdown(
            """
            #### [CAUSALITY // 因果约束] 自回归遮蔽原则
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
            #### [MATHEMATICS // 形式化定义] 掩码矩阵方程
            $$
            M_{ij} = \\begin{cases} 0, & i \\ge j \\text{ (允许关注)} \\\\ -\\infty, & i < j \\text{ (遮蔽未来)} \\end{cases}
            $$
            $$
            \\text{Scores} = \\frac{QK^T}{\\sqrt{d_k}} + M
            $$
            经过 Softmax 后：$e^{-\\infty} = 0$，上三角权重被精确清零！
            """
        )

# ---------------------------------------------------------------------------
# 2026 前沿拓展：RoPE 旋转位置编码 vs GQA 分组查询注意力
# ---------------------------------------------------------------------------
render_section_heading(
    "2026 ATTENTION EVOLUTION // 现代注意力架构跃迁：RoPE 与 GQA", icon_name="activity"
)

col_rope, col_gqa = st.columns(2)

with col_rope:
    with st.container(border=True):
        st.markdown("#### [ROPE // 旋转位置编码] 现代 LLM 标配")
        st.caption(
            "LLaMA-3/Qwen-2.5 放弃了绝对正弦编码，采用 2D 复数旋转：使注意力打分天然只取决于相对距离 (m - n)。"
        )

        from nn_core.rope import RotaryPositionalEmbedding

        rope_engine = RotaryPositionalEmbedding(dim=16, max_seq_len=seq_len)
        decay_mat = rope_engine.compute_relative_decay_matrix(seq_len)

        fig_rope = go.Figure(
            data=go.Heatmap(
                z=decay_mat,
                x=raw_tokens,
                y=raw_tokens,
                colorscale="Viridis",
                showscale=True,
                xgap=2,
                ygap=2,
                colorbar=dict(
                    title=dict(text="Inner Prod", font=dict(size=10, color="#0f172a")),
                    thickness=10,
                    len=0.8,
                ),
                hovertemplate="Token A: %{y}<br>Token B: %{x}<br>相对内积: %{z:.3f}<extra></extra>",
            )
        )
        fig_rope.update_layout(
            xaxis=dict(side="bottom", tickangle=-25 if seq_len > 6 else 0),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        fig_rope = _apply_light_theme(fig_rope, "RoPE 相对位置内积衰减矩阵 (对角线为1)")
        st.plotly_chart(fig_rope, width="stretch")
        with st.expander("[HOW TO READ // 读图指南] RoPE 相对衰减矩阵", expanded=False):
            st.markdown(
                """
                * **主对角线 (值为 1.0)**：自己与自己的相对距离为 0，内积最大（最亮）。
                * **远离对角线 (平滑变暗)**：随着词与词之间的距离拉大，内积得分自然衰减。
                * **[机制与边界]**：RoPE 将相对位置信息编码进注意力内积；超出训练长度的效果取决于频率设计、训练分布与外推方法，不能仅由结构保证。
                """
            )

with col_gqa, st.container(border=True):
    st.markdown("#### [GQA // 分组查询注意力] 显存瓶颈破局者")
    st.caption("多个 Query 头共享同一组 Key/Value 头，大幅缩减推理时 KV-Cache 显存开销。")

    kv_heads_choice = st.radio(
        "选择 KV 头配置架构",
        options=[
            "MHA (8 Q / 8 KV) - 1.0× 无压缩",
            "GQA (8 Q / 2 KV) - 4.0× 压缩 [推荐]",
            "MQA (8 Q / 1 KV) - 8.0× 极速",
        ],
        index=1,
        horizontal=True,
    )

    n_kv = 8 if "MHA" in kv_heads_choice else (2 if "GQA" in kv_heads_choice else 1)
    from nn_core.gqa import GroupedQueryAttention

    gqa_engine = GroupedQueryAttention(d_model=32, num_heads=8, num_kv_heads=n_kv)
    gqa_stats = gqa_engine.get_kv_cache_savings()

    st.metric(
        label="KV-Cache 显存吞吐压缩比",
        value=f"{gqa_stats['compression_ratio']:.1f}× 压缩",
        delta=f"为每步自回归推理节省 {gqa_stats['memory_saved_percent']:.1f}% 显存带宽",
        delta_color="normal",
    )

    # 显示路由矩阵关系
    st.markdown(
        f"""
            - **Query 头数**：`8 个` (负责保持全量多角度表征能力)
            - **Key/Value 头数**：`{n_kv} 个` (负责紧凑缓存)
            - **广播倍数**：每组由 `{8 // n_kv} 个 Query 头` 共享 1 个 KV 键值对
            """
    )

# ---------------------------------------------------------------------------
# 零基础进阶：注意力机制核心公式拆解与通俗速查
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 缩放点积注意力核心公式拆解与大白话速查", expanded=True
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：缩放点积注意力 (Scaled Dot-Product Attention)
        $$\\text{Attention}(Q, K, V) = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V$$

        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$Q$** | **查询矩阵 (Query)** | $N \\times d_k$ | **“当前词发出的搜索需求”**。比如主语“猫咪”发出 Query：“我的谓语动词在哪里？” |
        | **$K$** | **键矩阵 (Key)** | $N \\times d_k$ | **“所有候选词贴上的身份标签”**。比如动词“睡觉”贴上 Key：“我是一个动作！” |
        | **$Q K^T$** | **点积相关度矩阵** | $N \\times N$ | 计算句子里**所有词两两之间的关联契合度**（打分越高代表关系越密切）。 |
        | **$\\sqrt{d_k}$** | **维度缩放因子 (Scale Factor)** | 标量 (如 $\\sqrt{16}=4$) | **防爆炸调节阀**。向量维度高时，点积结果容易变得巨大，除以 $\\sqrt{d_k}$ 可以把方差拉回 1.0，防止 Softmax 梯度饱和进入死区。 |
        | **$M$** | **因果掩码 (Causal Mask)** | $N \\times N$ | **防剧透挡板**。上三角全为 $-\\infty$（负无穷），禁止当前词偷看后文。 |
        | **$\\text{Softmax}$** | **归一化概率转换** | $N \\times N$ | 把打分转为百分比概率，**确保每一行加起来严格等于 100% (1.0)**。 |
        | **$V$** | **值矩阵 (Value)** | $N \\times d_v$ | 被 Softmax 权重加权求和的特征。数学上它不自动等于“真正语义”。 |

        ---

        ### 1. 什么是【多头注意力 (Multi-Head Attention)】？—— “戴多副不同度数的眼镜”
        * **结构**：不同头有独立的 Q/K/V 投影参数，各自计算特征混合，再拼接并线性投影。多头提供多个子空间，但不预先指定每个头的语言学角色。
        * **结论边界**：只有在经训练模型上做系统探针、干预和多方法验证，才可谨慎讨论头的专门化；注意力权重本身不是因果解释。
        """
    )
