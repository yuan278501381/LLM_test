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

import importlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import dashboard.components.charts

importlib.reload(dashboard.components.charts)

from dashboard.components.charts import _apply_light_theme, plot_attention_heatmap_nlp
from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_architecture_flow_card,
    render_hero_header,
    render_live_param_status_bar,
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
render_lesson_evidence("M08", show_contract=True)

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
    title="Transformer 结构块与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "积木架构控制台",
            "desc": "在左侧侧边栏调节堆叠层数、隐藏层维度与输入测试句子",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解 Pre-LN 架构、残差高速公路与 SwiGLU 门控前馈",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时积木遥测",
            "desc": "监测堆叠层数、残差流向量模长增长、FFN 维度与总参数量",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "Pre-LN 数据流拓扑",
            "desc": "直观拆解 LayerNorm、MHA、GELU FFN 与主干残差流计算图",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "未训练注意力对比",
            "desc": "检查随机初始化层的权重形状、归一化与信息混合；不作语义解释",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>Transformer Block 就是现代大语言模型的标准积木块</b>。<br>"
        f"它把<b>自注意力机制 (MHA)</b>、<b>前馈全连接网络 (FFN)</b> 和<b>残差高速公路 (Residual Stream)</b> 封装在同一个盒子里。<br>"
        f"主干道上的残差流像一块中央黑板，数据在流动过程中，MHA 负责在不同词之间搬运信息，FFN 负责非线性思考与知识提取。<br>"
        f"每一层向残差流写入一个增量；只有经过合适数据与目标训练后，这些增量才可能形成有用表示。<br><br>"
        f"<b>【2026 前沿拓展】：SwiGLU 门控前馈网络</b><br>"
        f"现代 LLM 使用 SwiGLU 替代了传统的 GELU FFN，它引入了与输入相关的门控（Gate）机制："
        f"不仅决定激活强度，还直接调控信息流转通道，极大提升了同等参数量下的知识容量！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>Transformer 堆叠层数 (Num Layers)</b>：堆叠数量（1~4 层）；增加层数会增加容量和计算量，但不保证能力提升。<br>"
        f"• <b>隐藏层维度 (d_model)</b>：残差流主干道的向量宽度（32/64 维）。<br>"
        f"• <b>注意力头数 (Heads)</b>：每个结构块内的并行观察视角数。<br>"
        f"• <b>测试输入句子</b>：检查 token 经过随机初始化 Block 时的矩阵计算；结果不是训练后的语言理解。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[E. 逐层注意力热力图]', 'blue', target_id='region-e')} 观测</b>：同屏核对 Layer 1 到 Layer N 的未训练注意力数值。<br>"
        f"• <b>在 {anchor_badge('[D. Pre-LN 数据流]', 'purple', target_id='region-d')} 拆解</b>：主干残差流计算流程。<br>"
        f"• <b>在 {anchor_badge('[C. 积木遥测]', 'emerald', target_id='region-c')} 评估</b>：残差流向量模长增长折线。"
    ),
    experiments=[
        f"<b>第 1 步【核对随机初始化】</b>：把【堆叠层数】调到 3 或 4，检查 {anchor_badge('[E. 逐层热力图]', 'blue', target_id='region-e')} 每行是否归一化；不要把随机图案解释成语义分工。",
        f"<b>第 2 步【观察残差流范数】</b>：查看向量模长变化，理解恒等分支为何有利于信息与梯度传播，同时记录它并不保证任意深度下都稳定。",
        f"<b>第 3 步【对比前沿门控机制】</b>：滚动至底部实验室，体验 SwiGLU 的乘法门控机制如何通过 $x \\otimes \\text{{Swish}}(Wx)$ 实现对特征的高级非线性调制！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>BLOCK ARCHITECTURE // 积木架构控制台</b></div>',
    unsafe_allow_html=True,
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

sent_opts8 = list(sentence_options.keys())
selected_sentence_key = st.sidebar.segmented_control(
    "测试句子预设",
    options=sent_opts8,
    default=sent_opts8[0],
)
selected_sentence_key = selected_sentence_key or sent_opts8[0]

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

head_options = [
    (2, "双头注意力 (2 Heads)", "拆分 2 个独立特征子空间并行交互"),
    (4, "四头注意力 (4 Heads)", "拆分 4 个更细粒度注意力子空间"),
]

selected_head_card = st.sidebar.radio(
    "每个 Block 的注意力头数",
    options=head_options,
    format_func=lambda o: f"**{o[1]}**\n\n↳ *{o[2]}*",
    index=0,
    help="多头注意力的并行头数。",
)
num_heads = selected_head_card[0]

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
blocks = [TransformerBlock(d_model=32, num_heads=num_heads, d_ff=128) for _ in range(num_layers)]

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
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">BLOCK TELEMETRY // 积木架构与特征流遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
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
        delta=f"较初始增长 +{(norms_history[-1] / norms_history[0] - 1) * 100:.1f}%",
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
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">PRE-LN ARCHITECTURE // Pre-LN Transformer 结构块内部数据流拓扑</span>'
    f"</div>",
    unsafe_allow_html=True,
)
render_architecture_flow_card()

# ---------------------------------------------------------------------------
# 主视图区 2：逐层注意力热力图同屏对比
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">UNTRAINED ATTENTION // 随机初始化注意力数值对比 (1 ~ {num_layers} 层)</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="TRANSFORMER BLOCK TOPOLOGY // 结构块与残差流参数状态",
    badges=[
        {"label": "Layers L", "value": f"{num_layers}", "color": "blue"},
        {"label": "Heads h", "value": f"{num_heads}", "color": "amber"},
        {"label": "d_model", "value": "32", "color": "purple"},
        {"label": "d_ffn (4x)", "value": "128", "color": "emerald"},
    ],
    metrics=[
        ("初始残差范数 ‖x₀‖", f"{norms_history[0]:.2f}"),
        ("深层残差范数 ‖x_L‖", f"{norms_history[-1]:.2f}"),
        ("范数增长倍率", f"{norms_history[-1] / (norms_history[0] + 1e-12):.2f}x"),
    ],
    tag=f"PRE-LN RESIDUAL FLOW · {total_params:,} PARAMS",
    tag_color="emerald",
)

cols = st.columns(num_layers)
for i in range(num_layers):
    with cols[i]:
        fig_layer = plot_attention_heatmap_nlp(
            attention_weights=layer_attn_weights[i],
            tokens_x=raw_tokens,
            tokens_y=raw_tokens,
            title=f"Block #{i + 1} (Head 0)",
        )
        st.plotly_chart(fig_layer, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 逐层注意力演进热力图", expanded=False):
    st.markdown(
        """
        * **纵轴 (当前词)** 与 **横轴 (历史关注词)**：展示每个 Transformer 层的自注意力分布。
        * **[BOUNDARY // 结论边界]**：这些 Block 没有经过语料训练；热力图只用于检查计算与信息混合，不能证明逐层语义抽象。
          * **第 1 层 (浅层)**：注意力主要集中在邻近词（提取局部短语、标点等浅层语法特征）。
          * **深层 (第 2/3 层)**：注意力亮点开始跳跃式分布在远距离的关键核心词（完成深层语义理解与长程代词绑定）。
        """
    )

# ---------------------------------------------------------------------------
# 主视图区 3：残差流范数增长曲线
# ---------------------------------------------------------------------------
render_section_heading(
    "RESIDUAL STREAM CAPACITY ACCUMULATION // 残差流特征累积与范数增长", icon_name="trending-up"
)

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
        xaxis_title="网络深度 (Network Depth)",
        yaxis_title="平均特征向量 L2 范数 (Vector L2 Norm)",
        showlegend=False,
    )
    fig_norm = _apply_light_theme(fig_norm, "RESIDUAL NORM GROWTH // 残差流模长递增趋势")
    st.plotly_chart(fig_norm, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 残差流模长与特征累积折线图", expanded=False):
        st.markdown(
            """
            * **横轴【网络层级深度】**：从最底层的词嵌入输入层 $\\to$ Layer 1 $\\to$ Layer 2 $\\to$ 最终输出层。
            * **纵轴【特征向量 L2 模长】**：残差流中隐藏状态向量的能量总和。
            * **[OPTIMAL // 健康形态]**：模长随着网络层数平滑阶梯式稳步递增，说明每个 Block 都在以“加法”形式不断往中央总线注入新的抽象信息。
            """
        )

with col_math:
    with st.container(border=True):
        st.markdown(
            """
            #### [RESIDUAL STREAM // 理论精髓] 残差信息总线机制
            1. **中央信息总线**：
               残差流就像一条高宽带总线，每个 Block 不重写整条总线，只以加法形式写入增量特征 $\\Delta x$；
            2. **梯度高速公路**：
               加法分支为梯度提供恒等项（$\\frac{\\partial}{\\partial x}(x + f(x)) = 1 + f'(x)$），通常改善深层优化，但仍可能因初始化、归一化或数值尺度而不稳定；
            3. **为什么采用 Pre-LN？**：
               Pre-LN 把归一化放在子层之前，通常改善深层网络的训练稳定性；Post-LN 仍有研究与应用，具体选择依赖架构和训练方案。
            """
        )

# ---------------------------------------------------------------------------
# 2026 前沿拓展：GELU FFN vs SwiGLU 门控前馈网络
# ---------------------------------------------------------------------------
render_section_heading(
    "2026 FFN EVOLUTION // 前馈网络演进：经典 GELU vs 现代 SwiGLU 门控", icon_name="cpu"
)

col_gelu_box, col_swiglu_box = st.columns(2)

with col_gelu_box:
    with st.container(border=True):
        st.markdown("#### [CLASSIC FFN // 经典两层 MLP (GPT-2/3)]")
        st.code("h = GELU(x @ W1 + b1) @ W2 + b2", language="python")
        st.markdown(
            """
            - **计算路径**：升维 (4×)  GELU 激活  降维；
            - **参数量**：$2 \\times d_{model} \\times d_{ff}$；
            - **缺点**：缺少特征通道间的动态门控过滤。
            """
        )

with col_swiglu_box:
    with st.container(border=True):
        st.markdown("#### [MODERN SWIGLU // 现代门控 FFN (LLaMA-3/Gemma-2)]")
        st.code("out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down", language="python")
        st.markdown(
            """
            - **计算路径**：双重升维 (Gate & Up)  SiLU 门控相乘  降维；
            - **参数量**：$3 \\times d_{model} \\times \\frac{8}{3}d_{model}$ (等效总参数量)；
            - **核心特点**：元素级门控提供乘法交互；是否优于 GELU 取决于参数预算、数据和训练配置。
            """
        )

# ---------------------------------------------------------------------------
# 零基础进阶：Transformer Block 核心公式逐字拆解与名词通俗速查
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] Transformer Block 核心公式拆解与大模型底座名词全解", expanded=True
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：Pre-LN Transformer Block 标准计算流
        $$x^{(1)} = x + \\text{MHA}(\\text{LN}(x))$$
        $$x^{(2)} = x^{(1)} + \\text{FFN}(\\text{LN}(x^{(1)}))$$
        
        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$x$** | **输入残差主干信号 (Residual Stream)** | $N \\times d_{\\text{model}}$ | **中央高速公路上的原始行李箱**。装载着当前词汇的所有历史特征。 |
        | **$\\text{LN}$** | **层归一化 (Layer Normalization / RMSNorm)** | 函数映射 | **信号调音台**。把向量的均值拉到 0、方差拉到 1.0，防止数字在深度传递中过大或过小失真。 |
        | **$\\text{MHA}$** | **多头自注意力层 (Multi-Head Attention)** | $N \\times d_{\\text{model}}$ | **词与词之间的社交网络**。让当前词去查看句子中其他词的信息。 |
        | **$+$ (加法)** | **残差连接分支 (Residual Skip-Connection)** | 矩阵直接相加 | 提供恒等信息与梯度路径，通常缓解深层优化困难，但不是稳定性的无条件保证。 |
        | **$\\text{FFN}$** | **前馈感知网络 (Feed-Forward Network)** | $N \\times d_{\\text{model}}$ | **知识百科全书**。如果说 MHA 负责在词与词之间传纸条，FFN 则负责从记忆里检索这个词本身的百科知识。 |
        | **$x^{(2)}$** | **当前 Block 最终输出特征** | $N \\times d_{\\text{model}}$ | 经过本层注意力社交与知识补充后的新特征，直接送入下一层 Transformer Block。 |
        
        ---
        
        ### 1. Pre-LN 与 Post-LN 有什么取舍？
        * **Post-LN（原始 Transformer/BERT 常见做法）**：$x_{l+1} = \\text{LN}(x_l + f(x_l))$。深层训练通常更依赖 warm-up 和初始化，但并非不可训练。
        * **Pre-LN（许多现代 LLM 的做法）**：$x_{l+1} = x_l + f(\\text{LN}(x_l))$。主干保留加法路径，通常更易优化，但仍需合适的初始化、尺度控制和训练配置。
        """
    )
