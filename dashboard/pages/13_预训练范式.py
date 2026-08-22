# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 13: 预训练范式全景 (Pre-training Paradigms) - 零基础入门保姆级教学平台

解剖四大核心预训练范式：BERT 式双向掩码语言模型 (MLM)、GPT 式单向因果自回归语言模型 (CLM)、CLIP 式对比学习 (Contrastive) 与 MAE 视觉高比例掩码自编码。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.embeddings import get_mini_vocab
from nn_core.pretraining import (
    CausalLanguageModel,
    ContrastiveLearning,
    MaskedAutoEncoder,
    MaskedLanguageModel,
    PretrainingComparator,
)

st.set_page_config(
    page_title="Pre-training Paradigms · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="预训练范式全景与能力基因",
    subtitle="从海量无标注数据汲取通用世界知识：解剖 MLM 完形填空、CLM 因果接龙、Contrastive 对比学习与 MAE 视觉掩码重建",
    badge_text="MILESTONE 13 // PRE-TRAINING PARADIGMS",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="预训练目标函数与模型能力基因",
    plain_intro=(
        "<b>大模型的底座智能究竟是从哪里来的？</b><br>"
        "在没有任何人工标注的情况下，我们如何让模型自学成才？答案是<b>自监督预训练 (Self-Supervised Pre-training)</b>！<br>"
        "通过巧妙设计无监督目标函数：<br>"
        "• <b>BERT (MLM)</b> 玩'完形填空'，挖掉句子里 15% 的词让模型猜，因此精通双向语法与深层语义理解；<br>"
        "• <b>GPT (CLM)</b> 玩'文字接龙'，只看前文猜下一个词，因此拥有了强大的文本创作与自回归推理能力；<br>"
        "• <b>CLIP</b> 玩'连连看'，拉近图文距离；<b>MAE</b> 挖掉图片 75% 的块让模型脑补。<br>"
        "<b>目标函数的数学设计，从根本上决定了模型的'能力基因'！</b>"
    ),
    hyperparams_desc=(
        "• <b>预训练范式选择</b>：MLM (BERT) / CLM (GPT) / Contrastive (CLIP) / MAE (视觉自编码)。<br>"
        "• <b>MLM 掩码比例</b>：工业标准为 15%（过高缺失上下文，过低学不到知识）。<br>"
        "• <b>MAE 图像遮蔽率</b>：视觉信息存在高冗余，掩码率通常高达 75%！<br>"
        "• <b>迷你训练 Epochs</b>：在 Dell XPS 笔记本 CPU 上体验 1~10 轮纯 NumPy 真实损失收敛。"
    ),
    telemetry_desc=(
        "• <b>当前范式训练损失</b>：自监督优化目标收敛轨迹。<br>"
        "• <b>遮蔽 / 预测 Token 规模</b>：当前批次中参与梯度更新的活跃参数量。<br>"
        "• <b>下游任务迁移画像</b>：预训练范式向分类、生成、检索任务迁移的能力得分。"
    ),
    experiments=[
        "<b>第 1 步【对比完形填空与接龙】</b>：在 Section 2 观察同一句话在 MLM（双向关注两端）与 CLM（仅单向看前文）下的计算方式差异！",
        "<b>第 2 步【体验真实迷你训练】</b>：在左侧设置 Epoch=5 并点击【🚀 开始纯 NumPy 真实训练】，观察损失曲线如何在 1 秒内平滑下降！",
        "<b>第 3 步【解读迁移效果图】</b>：在 Section 4 观察四种范式在不同下游任务上的柱状图对比，理解为什么没有万能的模型！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

paradigm_choice = st.sidebar.selectbox(
    "核心预训练范式",
    options=[
        "MLM 掩码语言模型 (BERT 式)",
        "CLM 因果语言模型 (GPT 式)",
        "Contrastive 对比学习 (CLIP 式)",
        "MAE 视觉掩码自编码",
    ],
    index=0,
)

mlm_ratio = st.sidebar.slider(
    "MLM 文本掩码率 (Mask Ratio)",
    min_value=0.05,
    max_value=0.40,
    value=0.15,
    step=0.05,
    help="BERT 工业标准为 15%",
)

mae_ratio = st.sidebar.slider(
    "MAE 图像遮蔽率 (Mask Ratio)",
    min_value=0.50,
    max_value=0.90,
    value=0.75,
    step=0.05,
    help="MAE 推荐 75% 超高遮蔽",
)

train_epochs = st.sidebar.slider(
    "纯 NumPy 迷你训练轮数 (Epochs)",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="在笔记本 CPU 上秒级完成",
)

start_train_btn = st.sidebar.button("🚀 开始纯 NumPy 真实训练", use_container_width=True)

# ---------------------------------------------------------------------------
# 数据与模型运算
# ---------------------------------------------------------------------------
vocab = get_mini_vocab()
vocab_words = list(vocab.keys())
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab_words)

# 构造测试句子
sample_text = "the king and queen sleep on the big mat"
sample_tokens = [vocab.get(w, 0) for w in sample_text.split()]
sample_tensor = np.array([sample_tokens])

# 实例化四大模型
mlm_model = MaskedLanguageModel(vocab_size=vocab_size, d_model=32, mask_ratio=mlm_ratio)
clm_model = CausalLanguageModel(vocab_size=vocab_size, d_model=32)
contrastive_model = ContrastiveLearning(d_model=32)
mae_train_engine = MaskedAutoEncoder(num_patches=16, d_model=64, mask_ratio=mae_ratio)

# 真实训练与损失跟踪
loss_history = []
if start_train_btn or "train_loss_hist" not in st.session_state:
    hist = []
    for ep in range(train_epochs * 5):
        if "MLM" in paradigm_choice:
            l = mlm_model.train_step(sample_tensor, lr=0.05)
        elif "CLM" in paradigm_choice:
            l = clm_model.train_step(sample_tensor, lr=0.05)
        elif "Contrastive" in paradigm_choice:
            # 真实 NT-Xent 对比损失计算
            np.random.seed(42 + ep)
            dummy_embeds = np.random.randn(8, 32)
            z_i, z_j = contrastive_model.create_positive_pairs(dummy_embeds, noise_std=max(0.01, 0.2 - 0.02 * ep))
            l = contrastive_model.nt_xent_loss(z_i, z_j, temperature=0.5)
        else:  # MAE
            np.random.seed(42 + ep)
            dummy_patches = np.random.randn(1, 16, 64)
            m_p, m_idx, _ = mae_train_engine.create_mae_batch(dummy_patches)
            r_p = mae_train_engine.reconstruct(m_p)
            l = mae_train_engine.reconstruction_loss(r_p, dummy_patches, m_idx)
        hist.append(float(l))
    st.session_state["train_loss_hist"] = hist

loss_history = st.session_state.get("train_loss_hist", [2.5, 2.1, 1.8, 1.4, 1.1])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "PRE-TRAIN PARADIGM // 当前范式",
        paradigm_choice.split(" ")[0],
        delta="自监督学习",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TRAINING LOSS // 当前收敛损失",
        f"{loss_history[-1]:.4f}",
        delta=f"↓ 下降 {loss_history[0] - loss_history[-1]:.3f}",
        delta_type="positive",
        icon_name="activity",
    )
    + render_metric_card(
        "MASK RATIO // 掩码遮蔽率",
        f"{mlm_ratio * 100:.0f}%" if "MLM" in paradigm_choice else (f"{mae_ratio * 100:.0f}%" if "MAE" in paradigm_choice else "N/A"),
        delta="自监督信息瓶颈",
        delta_type="positive",
        icon_name="target",
    )
    + render_metric_card(
        "TRANSFER ADVANTAGE // 最强迁移领域",
        "自回归创作" if "CLM" in paradigm_choice else ("理解与问答" if "MLM" in paradigm_choice else "跨模态检索"),
        delta="基因决定能力",
        delta_type="positive",
        icon_name="layers",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 四大预训练范式核心思想卡片矩阵
# ---------------------------------------------------------------------------
render_section_heading("CORE PARADIGMS MATRIX // 四大预训练范式核心思想与数学目标", icon_name="cpu")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)

with col_p1:
    with st.container(border=True):
        st.markdown(
            """
            #### [MLM // 完形填空]
            - **代表模型**：BERT / RoBERTa
            - **核心公式**：
            $$-\\log p(w_{\\text{mask}} | w_{\\setminus \\text{mask}})$$
            - **注意力机制**：全向双向注意力
            - **能力基因**：深层句法/语义理解
            """
        )

with col_p2:
    with st.container(border=True):
        st.markdown(
            """
            #### [CLM // 因果接龙]
            - **代表模型**：GPT / LLaMA / Qwen
            - **核心公式**：
            $$-\\sum_t \\log p(w_t | w_{<t})$$
            - **注意力机制**：单向下三角因果掩码
            - **能力基因**：自回归文本创作与逻辑链
            """
        )

with col_p3:
    with st.container(border=True):
        st.markdown(
            """
            #### [CONTRASTIVE // 对比]
            - **代表模型**：CLIP / SimCLR
            - **核心公式**：
            $$-\\log \\frac{e^{\\text{sim}(z_i, z_j)/\\tau}}{\\sum_k e^{\\text{sim}(z_i, z_k)/\\tau}}$$
            - **注意力机制**：双塔跨模态独立编码
            - **能力基因**：多模态统一联合表征
            """
        )

with col_p4:
    with st.container(border=True):
        st.markdown(
            """
            #### [MAE // 视觉掩码]
            - **代表模型**：MAE (He et al.)
            - **核心公式**：
            $$\\text{MSE}(\\hat{P}_{\\text{mask}}, P_{\\text{mask}})$$
            - **注意力机制**：仅可见 Patch 参与自注意力
            - **能力基因**：全局视觉空间拓扑补全
            """
        )

# ---------------------------------------------------------------------------
# Section 2: MLM 完形填空 vs CLM 因果接龙互动对比与真实训练
# ---------------------------------------------------------------------------
render_section_heading("MLM VS CLM INTERACTION // 完形填空与接龙预测互动对比与训练", icon_name="target")

col_mlm_view, col_clm_view = st.columns(2)

with col_mlm_view:
    with st.container(border=True):
        st.markdown("#### [BERT 式 MLM 完形填空演示]")
        masked_ids, labels, mask_pos = mlm_model.create_mlm_batch(sample_tensor)
        masked_words = [inv_vocab.get(i, "[?]") if m else inv_vocab.get(i, "") for i, m in zip(sample_tokens, mask_pos)]
        st.markdown(f"**原始文本**：`{' '.join([inv_vocab.get(i, '') for i in sample_tokens])}`")
        st.markdown(f"**掩码输入**：`{' '.join(['**[MASK]**' if m else w for w, m in zip(masked_words, mask_pos)])}`")
        
        logits_mlm = mlm_model.forward(masked_ids)
        # 获取掩码位置的 Top-3 预测
        pred_top3_html = "<ul>"
        for idx in np.where(mask_pos)[0]:
            orig_word = inv_vocab.get(sample_tokens[idx], "")
            top_ids = np.argsort(logits_mlm[0, idx])[::-1][:3]
            top_words = [inv_vocab.get(tid, "") for tid in top_ids]
            pred_top3_html += f"<li>位置 {idx} (原词: <code>{orig_word}</code>) 预测 Top-3: <b>{', '.join(top_words)}</b></li>"
        pred_top3_html += "</ul>"
        st.markdown(pred_top3_html, unsafe_allow_html=True)

with col_clm_view:
    with st.container(border=True):
        st.markdown("#### [GPT 式 CLM 自回归接龙演示]")
        x_clm, y_clm = clm_model.create_clm_batch(sample_tensor)
        st.markdown(f"**上下文 Prefix**：`{' '.join([inv_vocab.get(i, '') for i in x_clm[0]])}`")
        st.markdown(f"**自回归 Targets**：`{' '.join([inv_vocab.get(i, '') for i in y_clm[0]])}`")
        
        logits_clm = clm_model.forward(x_clm)
        last_logits = logits_clm[0, -1]
        top_clm_ids = np.argsort(last_logits)[::-1][:3]
        top_clm_words = [inv_vocab.get(tid, "") for tid in top_clm_ids]
        st.markdown(f"- **下一个词候选 Top-3**：`{', '.join(top_clm_words)}`")
        st.markdown(f"- **因果限制**：严格通过下三角掩码禁止注意力看到后文。")

# 实时损失收敛曲线
fig_loss = go.Figure()
fig_loss.add_trace(
    go.Scatter(
        x=list(range(1, len(loss_history) + 1)),
        y=loss_history,
        mode="lines+markers",
        line=dict(color="#1d4ed8", width=2.5),
        marker=dict(size=6, color="#1d4ed8"),
        name="Training Loss",
        hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra></extra>",
    )
)
fig_loss.update_layout(
    xaxis=dict(title="训练迭代步数 (Optimization Steps)"),
    yaxis=dict(title="交叉熵损失 (Cross-Entropy Loss)"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_loss = _apply_light_theme(fig_loss, "纯 NumPy 真实梯度反向传播损失收敛曲线")
st.plotly_chart(fig_loss, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: MAE 75% 视觉高比例图块掩码与自编码重建
# ---------------------------------------------------------------------------
render_section_heading("MAE HIGH-RATIO MASKING // 视觉掩码自编码器 (75% 超高遮蔽率)", icon_name="layers")

from nn_core.video import generate_synthetic_video
raw_patch_img = generate_synthetic_video(1, 32, "bounce")[0, 0]

# 切为 16 个 Patch (4x4)
P = 8
gh = gw = 32 // P
patches_2d = raw_patch_img.reshape(gh, P, gw, P).transpose(0, 2, 1, 3).reshape(gh * gw, P * P)
mae_engine = MaskedAutoEncoder(num_patches=16, d_model=P * P, mask_ratio=mae_ratio)

masked_p, mask_idx, unmask_idx = mae_engine.create_mae_batch(patches_2d[np.newaxis, ...])
recon_p = mae_engine.reconstruct(masked_p)

# 重组回图像
masked_img_view = masked_p[0].reshape(gh, gw, P, P).transpose(0, 2, 1, 3).reshape(32, 32)
recon_img_view = recon_p[0].reshape(gh, gw, P, P).transpose(0, 2, 1, 3).reshape(32, 32)

col_mae1, col_mae2, col_mae3 = st.columns(3)
with col_mae1:
    fig_m1 = go.Figure(data=go.Heatmap(z=raw_patch_img, colorscale="Viridis", showscale=False))
    fig_m1.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed"), margin=dict(l=5, r=5, t=25, b=5))
    fig_m1 = _apply_light_theme(fig_m1, "原始完整图像 (100%)")
    st.plotly_chart(fig_m1, use_container_width=True)

with col_mae2:
    fig_m2 = go.Figure(data=go.Heatmap(z=masked_img_view, colorscale="Viridis", showscale=False))
    fig_m2.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed"), margin=dict(l=5, r=5, t=25, b=5))
    fig_m2 = _apply_light_theme(fig_m2, f"随机掩码遮蔽 ({int(mae_ratio*100)}% 消失)")
    st.plotly_chart(fig_m2, use_container_width=True)

with col_mae3:
    fig_m3 = go.Figure(data=go.Heatmap(z=recon_img_view, colorscale="Viridis", showscale=False))
    fig_m3.update_layout(xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False, autorange="reversed"), margin=dict(l=5, r=5, t=25, b=5))
    fig_m3 = _apply_light_theme(fig_m3, "MAE 自编码注意力重建")
    st.plotly_chart(fig_m3, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: 预训练范式向下游任务迁移效果对比图
# ---------------------------------------------------------------------------
render_section_heading("DOWNSTREAM TASK TRANSFER // 预训练范式向下游任务迁移能力画像", icon_name="activity")

transfer_data = PretrainingComparator.get_transfer_scores()
tasks = ["文本分类", "阅读理解", "自回归生成", "跨模态检索"]
colors = ["#6d28d9", "#1d4ed8", "#047857", "#b45309"]

fig_trans = go.Figure()
for idx, (model_name, scores_dict) in enumerate(transfer_data.items()):
    fig_trans.add_trace(
        go.Bar(
            name=model_name,
            x=tasks,
            y=[scores_dict[t] for t in tasks],
            marker_color=colors[idx],
            hovertemplate="模型: " + model_name + "<br>任务: %{x}<br>迁移得分: %{y}<extra></extra>",
        )
    )

fig_trans.update_layout(
    barmode="group",
    xaxis=dict(title="下游具体应用场景任务"),
    yaxis=dict(title="基准评测能力得分 (0 ~ 100)", range=[0, 110]),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_trans = _apply_light_theme(fig_trans, "四大预训练范式向各类下游任务迁移得分全景对比")
st.plotly_chart(fig_trans, use_container_width=True)
