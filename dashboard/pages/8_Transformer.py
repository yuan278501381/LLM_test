# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 8: Transformer 结构块 (Transformer Block & Residual Stream) - 零基础入门保姆级教学平台

解剖现代大模型的标准积木块：Pre-LN 架构、Multi-Head Attention、GELU FeedForward、残差高速公路 (Residual Stream) 与逐层特征精炼。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme, plot_attention_heatmap_nlp
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.embeddings import Embedding, get_mini_vocab, get_pretrained_embeddings
from nn_core.transformer import TransformerBlock

st.set_page_config(
    page_title="Transformer Block · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="Transformer 结构块与残差流",
    subtitle="解剖 ChatGPT 的核心积木：Pre-LN 架构、多头自注意力、GELU 前馈网络与残差高速公路 (Residual Stream) 逐层信息精炼",
    badge_text="MILESTONE 08 // TRANSFORMER BLOCK",
    badge_type="amber",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="Transformer 结构块与残差流入门",
    plain_intro=(
        "<b>Transformer Block 就是现代大语言模型的标准积木块</b>。<br>"
        "它把<b>自注意力机制 (MHA)</b>、<b>前馈全连接网络 (FFN)</b> 和<b>残差高速公路 (Residual Stream)</b> 封装在同一个盒子里。<br>"
        "主干道上的残差流像一块中央黑板，数据在流动过程中，MHA 负责在不同词之间搬运信息，FFN 负责非线性思考与知识提取。<br>"
        "每一层只在黑板上写下微小的增量，<b>层数堆得越多，模型理解越深刻！</b>"
    ),
    hyperparams_desc=(
        "• <b>Transformer 堆叠层数 (Num Layers)</b>：堆叠的积木数量（1~4层），层数越深抽象推理能力越强。<br>"
        "• <b>隐藏层维度 (d_model)</b>：残差流主干道的向量宽度（32/64 维）。<br>"
        "• <b>注意力头数 (Heads)</b>：每个结构块内的并行观察视角数。<br>"
        "• <b>测试输入句子</b>：观察多层网络在处理真实文本时的逐层注意力演进。"
    ),
    telemetry_desc=(
        "• <b>逐层注意力热力图</b>：同屏对比 Layer 1 到 Layer N 的注意力模式分工（底层看表面语法，高层看语义回指）。<br>"
        "• <b>残差流向量范数曲线</b>：展示特征向量范数如何随着层数稳步增长（证明每层都在积累知识）。<br>"
        "• <b>参数总量遥测</b>：实时统计当前多层 Transformer Block 的物理参数规模。"
    ),
    experiments=[
        "<b>第 1 步【对比逐层分工】</b>：在左侧把【堆叠层数】调到 3 或 4，观察下方逐层注意力热力图：第 1 层偏向关注相邻词（对角线），而高层注意力开始跨距离跳跃关注语义相关的词！",
        "<b>第 2 步【观察残差流范数】</b>：查看下方的残差流向量模长增长折线，体会为什么残差连接能让深层网络绝不发生梯度消失！",
        "<b>第 3 步【改变头数】</b>：切换注意力头数（2头 vs 4头），观察特征表达精细度的提升！",
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
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与架构")

sentence_options = {
    "经典长句 (王族与旅行)": "the king and queen went to paris and then they were very happy",
    "因果推理与状态": "the puppy was small and cold on the mat and sleep",
    "地理与动作": "the dog run fast in china and then sleep in beijing",
    "自定义输入...": "",
}

selected_sentence_key = st.sidebar.selectbox(
    "测试句子预设",
    list(sentence_options.keys()),
    index=0,
)

if "自定义" in selected_sentence_key:
    input_text = st.sidebar.text_input(
        "自定义英文句子",
        "the king and queen traveled to paris",
    )
else:
    input_text = sentence_options[selected_sentence_key]

num_layers = st.sidebar.select_slider(
    "Transformer 堆叠层数 (Num Layers)",
    options=[1, 2, 3, 4],
    value=3,
    help="堆叠的 Pre-LN Transformer 结构块数量。",
)

num_heads = st.sidebar.selectbox(
    "每个 Block 的注意力头数",
    [2, 4],
    index=0,
    help="多头注意力的并行头数。",
)

# ---------------------------------------------------------------------------
# Token 化与多层 Transformer 前向推理
# ---------------------------------------------------------------------------
raw_tokens = [w.lower().strip(",.!?") for w in input_text.strip().split() if w.strip()]
if not raw_tokens:
    raw_tokens = ["the", "king", "queen", "sleep"]

token_indices = [raw_vocab.get(w, 0) for w in raw_tokens]
tokens_array = np.array([token_indices])
seq_len = len(raw_tokens)

# 提取初始嵌入
x_stream = embedding_layer.forward(tokens_array)  # shape: (1, seq_len, 32)

# 记录残差流的 L2 范数变化
norms_history = [float(np.mean(np.linalg.norm(x_stream[0], axis=-1)))]

# 实例化多层 Transformer Block 并依次前向传播
blocks = [
    TransformerBlock(d_model=32, num_heads=num_heads, d_ff=128)
    for _ in range(num_layers)
]

layer_attn_weights = []

for block_idx, block in enumerate(blocks):
    x_stream, attn_w = block.forward(x_stream)
    layer_attn_weights.append(attn_w[0, 0])  # 取 Head 0
    # 记录经过本层后的残差流范数
    current_norm = float(np.mean(np.linalg.norm(x_stream[0], axis=-1)))
    norms_history.append(current_norm)

# 参数量统计
# 每一个 Block: MHA(4 * 32*32) + 2*LayerNorm(2*32*2) + FFN(32*128 + 128*32 + 128 + 32)
params_per_block = (4 * 32 * 32) + (4 * 32) + (32 * 128 + 128 * 32 + 128 + 32)
total_params = params_per_block * num_layers + (len(vocab_words) * 32)

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "STACKED BLOCKS // 堆叠层数",
        f"{num_layers} BLOCKS",
        delta="深层特征提取" if num_layers >= 3 else "浅层网络",
        delta_type="positive" if num_layers >= 3 else "neutral",
        icon_name="database",
    )
    + render_metric_card(
        "RESIDUAL STREAM NORM // 残差流范数",
        f"{norms_history[-1]:.2f}",
        delta=f"较初始增长 +{(norms_history[-1]/norms_history[0] - 1)*100:.1f}%",
        delta_type="positive",
        icon_name="trending-up",
    )
    + render_metric_card(
        "FFN EXPANSION // 前馈隐藏维度",
        "128-DIM (4×)",
        delta="GELU 非线性激活",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TOTAL PARAMETERS // 参数总量",
        f"{total_params:,} PARAMS",
        delta=f"{num_layers}× Transformer",
        delta_type="neutral",
        icon_name="target",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 主视图区 1：Pre-LN Transformer 结构块数据流拓扑
# ---------------------------------------------------------------------------
render_section_heading("PRE-LN ARCHITECTURE // Pre-LN Transformer 结构块内部数据流拓扑", icon_name="activity")

st.markdown(
    """
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1.2rem; font-family:'JetBrains Mono', monospace; font-size:0.85rem; line-height:1.6; color:#0f172a; box-shadow:0 2px 8px rgba(15,23,42,0.03);">
        <div style="display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div style="background:#eff6ff; border:1px solid #bfdbfe; padding:0.8rem 1.2rem; border-radius:8px; text-align:center;">
                <span style="color:#1d4ed8; font-weight:800;">INPUT STREAM x</span><br>
                <span style="font-size:0.75rem; color:#64748b;">(残差流主干道)</span>
            </div>
            <div style="color:#64748b; font-size:1.4rem;">➔</div>
            <div style="background:#f8fafc; border:1px solid #cbd5e1; padding:0.8rem 1.2rem; border-radius:8px; text-align:center;">
                <span style="color:#7c3aed; font-weight:800;">LayerNorm₁</span> ➔ <span style="color:#2563eb; font-weight:800;">Multi-Head Attention</span><br>
                <span style="color:#059669; font-weight:700;">x = x + MHA(LN₁(x))</span> <span style="font-size:0.75rem; color:#64748b;">(残差加法)</span>
            </div>
            <div style="color:#64748b; font-size:1.4rem;">➔</div>
            <div style="background:#f8fafc; border:1px solid #cbd5e1; padding:0.8rem 1.2rem; border-radius:8px; text-align:center;">
                <span style="color:#7c3aed; font-weight:800;">LayerNorm₂</span> ➔ <span style="color:#b45309; font-weight:800;">GELU FeedForward</span><br>
                <span style="color:#059669; font-weight:700;">x = x + FFN(LN₂(x))</span> <span style="font-size:0.75rem; color:#64748b;">(残差加法)</span>
            </div>
            <div style="color:#64748b; font-size:1.4rem;">➔</div>
            <div style="background:#ecfdf5; border:1px solid #a7f3d0; padding:0.8rem 1.2rem; border-radius:8px; text-align:center;">
                <span style="color:#047857; font-weight:800;">OUTPUT STREAM x</span><br>
                <span style="font-size:0.75rem; color:#64748b;">(进入下一层 Block)</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 主视图区 2：逐层注意力热力图同屏对比
# ---------------------------------------------------------------------------
render_section_heading(f"LAYER-WISE ATTENTION EVOLUTION // 逐层注意力模式演进同屏对比 (1 ~ {num_layers} 层)", icon_name="target")

cols = st.columns(num_layers)
for i in range(num_layers):
    with cols[i]:
        fig_layer = plot_attention_heatmap_nlp(
            attention_weights=layer_attn_weights[i],
            tokens_x=raw_tokens,
            tokens_y=raw_tokens,
            title=f"Block #{i+1} Attention (Head 0)",
        )
        st.plotly_chart(fig_layer, use_container_width=True)

# ---------------------------------------------------------------------------
# 主视图区 3：残差流范数增长曲线
# ---------------------------------------------------------------------------
render_section_heading("RESIDUAL STREAM CAPACITY ACCUMULATION // 残差流特征累积与范数增长", icon_name="trending-up")

col_curve, col_math = st.columns([1.4, 1])

with col_curve:
    fig_norm = go.Figure()
    fig_norm.add_trace(
        go.Scatter(
            x=[f"Layer {i}" if i > 0 else "Input (Embedding)" for i in range(num_layers + 1)],
            y=norms_history,
            mode="lines+markers+text",
            text=[f"{v:.2f}" for v in norms_history],
            textposition="top center",
            textfont=dict(size=11, family="JetBrains Mono", color="#0f172a"),
            line=dict(color="#1d4ed8", width=3),
            marker=dict(size=9, color="#1d4ed8", symbol="circle"),
            name="Residual L2 Norm",
            hovertemplate="<b>%{x}</b><br>Mean Vector Norm: %{y:.4f}<extra></extra>",
        )
    )
    fig_norm.update_layout(
        xaxis_title="Network Depth (网络层级深度)",
        yaxis_title="Mean Vector L2 Norm (平均特征向量范数)",
        showlegend=False,
    )
    fig_norm = _apply_light_theme(fig_norm, "RESIDUAL NORM GROWTH // 残差流模长递增趋势")
    st.plotly_chart(fig_norm, use_container_width=True)

with col_math:
    with st.container(border=True):
        st.markdown(
            """
            #### 💡 残差流 (Residual Stream) 的理论精髓
            1. **中央信息总线**：
               残差流就像一条高宽带总线，每个 Block 不重写整条总线，只以加法形式写入增量特征 $\\Delta x$；
            2. **梯度高速公路**：
               反向传播时，损失梯度可以直接通过加法分支无阻碍传回底层（$\\frac{\\partial}{\\partial x}(x + f(x)) = 1 + f'(x)$），彻底消除了梯度消失；
            3. **为什么采用 Pre-LN？**：
               现代大模型（LLaMA/GPT-3/4）全面淘汰 Post-LN，将 LayerNorm 放在子层之前，确保残差主干道完全无阻碍，使得训练哪怕几百层模型也能极其稳定。
            """
        )
