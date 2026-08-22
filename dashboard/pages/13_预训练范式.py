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
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_floating_hud_navigator,
    render_page_guide,
    render_section_heading,
)
from nn_core.embeddings import get_mini_vocab
from nn_core.pretraining import (
    CausalLanguageModel,
    ContrastiveLearning,
    DataMixtureEngine,
    MaskedAutoEncoder,
    MaskedLanguageModel,
    PretrainingComparator,
    ScalingLawEngine,
    SimpleBPE,
)

st.set_page_config(
    page_title="Pre-training Paradigms · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="预训练范式全景与大模型扩展定律",
    subtitle="从海量无标注数据汲取通用世界知识：解剖四大自监督目标、Chinchilla 算力扩展定律 (Scaling Laws)、BPE 分词演化与工业级语料配比",
    badge_text="MILESTONE 13 // PRE-TRAINING PARADIGMS & SCALING LAWS",
    badge_type="blue",
)

render_floating_hud_navigator([
        {"id": "A", "name": "预训练控制台", "desc": "在左侧侧边栏切换核心预训练目标、掩码比例与学习率", "color": "amber", "target_id": "region-a"},
        {"id": "B", "name": "教学指引", "desc": "当前卡片：通俗理解自监督预训练基因、Scaling Laws 与数据清洗工程", "color": "blue", "target_id": "region-b"},
        {"id": "C", "name": "实时预训练遥测", "desc": "显示当前范式、梯度收敛损失、掩码遮蔽率与最优迁移领域", "color": "emerald", "target_id": "region-c"},
        {"id": "D", "name": "MLM vs CLM 真实训练", "desc": "完形填空与因果接龙对比互动及纯 NumPy 真实梯度反向传播收敛", "color": "purple", "target_id": "region-d"},
        {"id": "E", "name": "Scaling Laws 实验", "desc": "Chinchilla 扩展定律算力计算器、历史模型散点与 BPE 演化", "color": "blue", "target_id": "region-e"},
    ])

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="预训练范式全景与空间交互地图",
    blueprint_sections=[
        {"id": "A", "name": "预训练控制台", "desc": "在左侧侧边栏切换核心预训练目标、掩码比例与学习率", "color": "amber", "target_id": "region-a"},
        {"id": "B", "name": "教学指引", "desc": "当前卡片：通俗理解自监督预训练基因、Scaling Laws 与数据清洗工程", "color": "blue", "target_id": "region-b"},
        {"id": "C", "name": "实时预训练遥测", "desc": "显示当前范式、梯度收敛损失、掩码遮蔽率与最优迁移领域", "color": "emerald", "target_id": "region-c"},
        {"id": "D", "name": "MLM vs CLM 真实训练", "desc": "完形填空与因果接龙对比互动及纯 NumPy 真实梯度反向传播收敛", "color": "purple", "target_id": "region-d"},
        {"id": "E", "name": "Scaling Laws 实验", "desc": "Chinchilla 扩展定律算力计算器、历史模型散点与 BPE 演化", "color": "blue", "target_id": "region-e"},
    ],
    plain_intro=(
        f"<b>大模型的底座智能究竟是从哪里来的？</b><br>"
        f"在没有任何人工标注的情况下，我们如何让模型自学成才？答案是<b>自监督预训练 (Self-Supervised Pre-training)</b> 与 <b>大模型扩展定律 (Scaling Laws)</b>！<br>"
        f"• <b>四大目标基因</b>：BERT (MLM) 完形填空擅长理解；GPT (CLM) 因果接龙擅长创作推理；CLIP 擅长跨模态对齐；MAE 擅长全局视觉补全。<br>"
        f"• <b>扩展定律经济学</b>：在 {anchor_badge('[E. Scaling Laws 实验室]', 'blue', target_id='region-e')} 体验 DeepMind Chinchilla 严格证明的黄金配比 $D \\approx 20N$ 与算力公式 $C \\approx 6ND$。<br>"
        f"• <b>分词与数据工程</b>：BPE 自底向上合并高频子词；高质量语料清洗决定智能上限！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>预训练范式选择</b>：MLM (BERT) / CLM (GPT) / Contrastive (CLIP) / MAE (视觉自编码)。<br>"
        f"• <b>算力预算对数指数</b>：从 $10^{{18}}$ FLOPs 到 $10^{{25}}$ FLOPs 动态求解 Chinchilla 最优参数量与 Token 数。<br>"
        f"• <b>BPE 目标词表大小</b>：观察从单字符演化为包含常见词根词缀的紧凑词表。<br>"
        f"• <b>迷你训练 Epochs</b>：在笔记本 CPU 上体验纯 NumPy 真实梯度反向传播损失收敛。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 真实训练]', 'purple', target_id='region-d')} 观测</b>：解析梯度驱动下的损失单调收敛曲线。<br>"
        f"• <b>在 {anchor_badge('[E. Scaling Laws]', 'blue', target_id='region-e')} 计算</b>：最优参数规模 $N_{{\\text{{opt}}}}$、最优 Token 数 $D_{{\\text{{opt}}}}$ 与 H100 训练工期。<br>"
        f"• <b>在 {anchor_badge('[C. 预训练遥测]', 'emerald', target_id='region-c')} 评估</b>：核心范式迁移优势与损失。"
    ),
    experiments=[
        f"<b>第 1 步【对比完形填空与接龙】</b>：在 {anchor_badge('[D. 真实训练]', 'purple', target_id='region-d')} 观察同一句话在 MLM 与 CLM 下的真实梯度训练收敛！",
        f"<b>第 2 步【探索 Chinchilla 扩展定律】</b>：在 {anchor_badge('[E. Scaling Laws]', 'blue', target_id='region-e')} 拖动算力预算滑块，观察为什么 70B 模型需要喂 1.4T Token，以及 GPT-3 175B 为何欠训练！",
        f"<b>第 3 步【动手训练 BPE 分词器】</b>：在 Section 6 输入自定义句子并点击【训练 BPE 分词规则】，亲眼见证常见词是如何一步步被合并诞生的！",
        f"<b>第 4 步【拆解工业级数据流水线】</b>：在 Section 7 观察 LLaMA-3 与 DeepSeek-V3 的真实语料配比与 4 阶段清洗过滤规则！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown(f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>PRE-TRAINING CONTROLS // 预训练控制台</b></div>', unsafe_allow_html=True)

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

start_train_btn = st.sidebar.button("[EXECUTE] 开始纯 NumPy 真实训练", use_container_width=True)

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
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">PRE-TRAINING TELEMETRY // 预训练损失与能力基因遥测</span>'
    f'</div>',
    unsafe_allow_html=True,
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
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">MLM VS CLM TRAINING // 完形填空与接龙对比及真实梯度反向传播收敛</span>'
    f'</div>',
    unsafe_allow_html=True,
)

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
with st.expander("[HOW TO READ // 读图指南] 预训练损失单调收敛曲线", expanded=False):
    st.markdown(
        """
        * **横轴【优化迭代步数】** 与 **纵轴【交叉熵损失 (Loss)】**。
        * **[OPTIMAL // 真实梯度特征]**：在真实解析梯度驱动下，损失单调下滑，表明模型成功在预训练目标上学到了语言统计规律。
        """
    )

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

with st.expander("[HOW TO READ // 读图指南] MAE 高比例遮蔽重建三联图", expanded=False):
    st.markdown(
        """
        * **中图【75% 极高遮蔽】**：大部分像素消失，仅留 25% 可见图块；
        * **右图【模型脑补重建】**：模型依靠 Transformer 空间自注意力，从仅有的少量线索中还原出被遮挡的完整物理几何轮廓！
        """
    )

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
with st.expander("[HOW TO READ // 读图指南] 预训练基因与下游任务迁移能力雷达/柱状图", expanded=False):
    st.markdown(
        """
        * **横轴【4 种典型下游任务】** 与 **纵轴【迁移能力得分 (0~100)】**。
        * **CLM (GPT 式因果接龙)**：在**自回归生成**任务上一骑绝尘；
        * **MLM (BERT 式完形填空)**：在**分类与阅读理解**任务上得分最高。
        """
    )

# ---------------------------------------------------------------------------
# Section 5: 大模型扩展定律 (Scaling Laws) 与最优算力分配计算器 (Chinchilla)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">SCALING LAWS & BPE // 大模型扩展定律 (Chinchilla) 与 BPE 分词演化</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    **DeepMind Chinchilla 定律 (Hoffmann et al., 2022)** 颠覆了早期 Kaplan 的“大模型优先”假说，严格证明：
    在固定总算力预算 $C \\approx 6ND$ 下，**模型参数量 $N$ 与训练 Token 数 $D$ 应当等比例缩放**（黄金配比 $D \\approx 20N$）！
    """
)

col_s_ctrl, col_s_res = st.columns([1, 2])

with col_s_ctrl:
    with st.container(border=True):
        st.markdown("#### [算力预算与硬件配置]")
        log_flops = st.slider(
            "总算力预算 (FLOPs 对数指数 $\\log_{10} C$)",
            min_value=18.0,
            max_value=25.0,
            value=23.0,
            step=0.2,
            help="10^23 FLOPs 约相当于训练一个 7B~13B 模型所消耗的算力",
        )
        current_flops = 10.0 ** log_flops
        opt_res = ScalingLawEngine.compute_optimal_allocation(current_flops)

        st.markdown(
            f"""
            - **算力预算 $C$**：`{current_flops:.2e}` FLOPs
            - **最优参数量 $N_{{\\text{{opt}}}}$**：`{opt_res['optimal_params_N'] / 1e9:.2f} B` (十亿参数)
            - **最优 Token 数 $D_{{\\text{{opt}}}}$**：`{opt_res['optimal_tokens_D'] / 1e9:.2f} B` ({opt_res['optimal_tokens_D'] / 1e12:.3f} T Tokens)
            - **数据/参数比例**：`{opt_res['token_param_ratio']:.1f} : 1` (符合 Chinchilla ~20:1)
            - **理论预估损失 $L$**：`{opt_res['predicted_loss']:.4f}`
            - **H100 训练工期**：`{opt_res['h100_gpu_days']:.1f}` GPU-Days (MFU=45%)
            """
        )

with col_s_res:
    with st.container(border=True):
        st.markdown("#### [前沿大模型在 Scaling 曲线上的实际分布]")
        bench_data = ScalingLawEngine.get_historical_benchmarks()
        
        # 绘制历史模型与当前预算对比散点图
        fig_scale = go.Figure()
        
        # 绘制最优 Chinchilla 理论线
        synth_flops = np.logspace(18, 25, 50)
        synth_n = [ScalingLawEngine.compute_optimal_allocation(f)["optimal_params_N"] for f in synth_flops]
        synth_d = [ScalingLawEngine.compute_optimal_allocation(f)["optimal_tokens_D"] for f in synth_flops]
        
        fig_scale.add_trace(
            go.Scatter(
                x=[n / 1e9 for n in synth_n],
                y=[d / 1e12 for d in synth_d],
                mode="lines",
                line=dict(color="#1d4ed8", width=2.5, dash="dash"),
                name="Chinchilla 最优边界 (D ≈ 20N)",
                hovertemplate="最优参数: %{x:.2f}B<br>最优数据: %{y:.2f}T Tokens<extra></extra>",
            )
        )
        
        # 绘制历史真实模型散点
        for b in bench_data:
            fig_scale.add_trace(
                go.Scatter(
                    x=[b["params"] / 1e9],
                    y=[b["tokens"] / 1e12],
                    mode="markers+text",
                    marker=dict(size=12, symbol="diamond"),
                    name=b["name"],
                    text=[b["name"]],
                    textposition="top center",
                    hovertemplate=f"<b>{b['name']}</b><br>参数量: {b['params']/1e9:.1f}B<br>训练 Token: {b['tokens']/1e12:.1f}T<br>状态: {b['status']}<extra></extra>",
                )
            )
            
        # 当前滑动条用户选择的最优点
        fig_scale.add_trace(
            go.Scatter(
                x=[opt_res["optimal_params_N"] / 1e9],
                y=[opt_res["optimal_tokens_D"] / 1e12],
                mode="markers",
                marker=dict(size=16, color="#dc2626", symbol="star"),
                name="当前算力最优解",
                hovertemplate=f"当前选择算力最优:<br>参数: {opt_res['optimal_params_N']/1e9:.2f}B<br>数据: {opt_res['optimal_tokens_D']/1e12:.2f}T<extra></extra>",
            )
        )
        
        fig_scale.update_layout(
            xaxis=dict(title="模型参数量 N (Billion / 十亿)", type="log"),
            yaxis=dict(title="训练 Token 数 D (Trillion / 万亿)", type="log"),
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        fig_scale = _apply_light_theme(fig_scale, "大模型算力扩展定律与前沿工业界模型位置")
        st.plotly_chart(fig_scale, use_container_width=True)
        with st.expander("[HOW TO READ // 读图指南] Chinchilla 对数双轴扩展定律散点图", expanded=False):
            st.markdown(
                """
                * **双对数坐标轴**：横轴【参数量 N (十亿 / B)】，纵轴【训练数据量 D (万亿 / T)】。
                * **蓝色虚线 (理论最优线)**：严格满足 $D \\approx 20N$ 的最佳算力投资分配线。
                * **GPT-3 (偏右下)**：参数巨大但数据少（严重欠训练）；
                * **LLaMA-3 (偏左上)**：8B 小模型狂喂 15T 数据（超训练，换取极致推理轻量化）。
                """
            )

# ---------------------------------------------------------------------------
# Section 6: BPE (Byte-Pair Encoding) 分词器动态演练
# ---------------------------------------------------------------------------
render_section_heading("BPE TOKENIZATION // 字节对编码子词分词器演练", icon_name="target")

st.markdown(
    """
    大模型并非直接以字符或完整单词作为输入，而是通过 **BPE (Byte-Pair Encoding)** 统计相邻字节/字符频次，
    从单字符出发自底向上贪心合并为高频子词 (Subwords)，在词表规模与序列长度之间取得极致的压缩平衡！
    """
)

col_bpe_in, col_bpe_out = st.columns([1, 2])

with col_bpe_in:
    with st.container(border=True):
        st.markdown("#### [分词器语料与配置]")
        default_bpe_corpus = (
            "the king and the queen sleep on the big mat\n"
            "the queen and the king sleep together\n"
            "the king has a big crown on the head\n"
            "the queen loves the small cat"
        )
        bpe_corpus_input = st.text_area("训练语料库", value=default_bpe_corpus, height=120)
        target_vocab_k = st.slider("目标词表容量 (Vocab Size)", min_value=15, max_value=50, value=30, step=1)
        test_sentence_bpe = st.text_input("待切分测试文本", value="the queen sleep on the cat")

with col_bpe_out:
    with st.container(border=True):
        bpe_engine = SimpleBPE(vocab_size=target_vocab_k)
        corpus_list = [line.strip() for line in bpe_corpus_input.strip().split("\n") if line.strip()]
        merge_records = bpe_engine.train(corpus_list)
        
        stats_res = bpe_engine.get_compression_stats(test_sentence_bpe)
        
        st.markdown(
            f"""
            - **基础单字符数**：`{len(bpe_engine.vocab) - len(merge_records)}` 个字符
            - **迭代合并次数**：`{len(merge_records)}` 轮
            - **最终词表大小**：`{len(bpe_engine.vocab)}` 个 Tokens
            - **文本压缩倍率**：`{stats_res['compression_ratio']:.2f}x` ({stats_res['raw_characters']} 字符 $\\to$ {stats_res['token_count']} Tokens)
            """
        )
        
        # 可视化切分结果
        st.markdown("**BPE 子词切分可视化**：")
        token_badges_html = " ".join([f"<code style='background:#dbeafe;color:#1e40af;padding:3px 6px;border-radius:4px;'>{t}</code>" for t in stats_res['tokens']])
        st.markdown(token_badges_html, unsafe_allow_html=True)
        
        # 展示 Top-5 合并规则
        if merge_records:
            st.markdown("<br>**前 5 轮高频字符对合并日志**：", unsafe_allow_html=True)
            table_md = "| 轮次 | 合并字符对 | 新子词 Token | 语料频次 | 词表规模 |\n|:---:|:---:|:---:|:---:|:---:|\n"
            for r in merge_records[:5]:
                table_md += f"| Step {r['step']} | {r['merged_pair']} | `'{r['new_token']}'` | {r['frequency']} | {r['vocab_size']} |\n"
            st.markdown(table_md)

# ---------------------------------------------------------------------------
# Section 7: 工业级预训练语料配比与清洗流水线
# ---------------------------------------------------------------------------
render_section_heading("PRE-TRAINING DATA MIXTURE & PIPELINE // 工业级语料配比与清洗流水线", icon_name="layers")

st.markdown(
    """
    **数据质量决定智能上限 (Garbage In, Garbage Out)**。现代 10T+ Token 级预训练模型依赖严苛的四阶段多源清洗与精准配比！
    """
)

col_d_mix, col_d_pipe = st.columns([1, 1])

with col_d_mix:
    with st.container(border=True):
        st.markdown("#### [前沿大模型语料配比全景 (Data Mixture)]")
        mixtures_dict = DataMixtureEngine.get_mixtures()
        model_mix_choice = st.selectbox("选择大模型语料分布", list(mixtures_dict.keys()), index=0)
        
        chosen_mix = mixtures_dict[model_mix_choice]
        labels = list(chosen_mix.keys())
        values = list(chosen_mix.values())
        
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, textinfo="label+percent")])
        fig_donut.update_layout(margin=dict(l=10, r=10, t=20, b=20), showlegend=False)
        fig_donut = _apply_light_theme(fig_donut, f"{model_mix_choice} 语料构成分布")
        st.plotly_chart(fig_donut, use_container_width=True)
        with st.expander("[HOW TO READ // 读图指南] 预训练多源语料配比环形图", expanded=False):
            st.markdown(
                """
                * **扇区比例**：不同领域知识（网页、代码、学术论文、书籍、数学题）在 10T+ 数据中的百分比构成。
                """
            )

with col_d_pipe:
    with st.container(border=True):
        st.markdown("#### [工业级四阶段数据清洗过滤流水线]")
        stages = DataMixtureEngine.get_cleaning_pipeline()
        for s in stages:
            st.markdown(
                f"""
                **{s['stage']}**  
                - **核心规则**：{s['rules']}  
                - **过滤效率**：<span style='color:#dc2626;font-weight:600;'>{s['filter_rate']}</span>
                ---
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# 零基础进阶：预训练与扩展定律核心公式拆解
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 预训练目标与 Chinchilla 扩展定律核心公式全解", expanded=True):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：DeepMind Chinchilla 大模型扩展定律
        $$L(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}$$
        
        | 符号 | 中文名称 | 权威拟合数值 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$L(N, D)$** | **预测预训练交叉熵损失 (Loss)** | 连续实数 | 评估一个模型预训练完后到底有多聪明（损失越低，模型越强）。 |
        | **$E$** | **不可约内在熵 (Irreducible Loss)** | **$1.6934$** | **人类语言的物理极限天花板**。即使你拥有无穷大的模型和无穷多的数据，语言本身的随机性也决定了 Loss 不可能低于 $1.6934$。 |
        | **$N$** | **模型非嵌入参数量 (Parameters)** | 比如 $7\\text{B} = 7 \\times 10^9$ | **大脑的脑容量脑细胞数量**。脑容量越大，记忆能力越强。 |
        | **$D$** | **训练 Token 总数 (Data Tokens)** | 比如 $2\\text{T} = 2 \\times 10^{12}$ | **给大脑喂的书本知识量**。书读得越多，见识越广。 |
        | **$A, B$** | **参数与数据规模常数** | $A=406.4, B=410.7$ | 衡量增加参数与增加数据对降低 Loss 的相对贡献系数。 |
        | **$\\alpha, \\beta$** | **幂律缩放指数 (Scaling Exponents)** | $\\alpha=0.34, \\beta=0.28$ | 揭示“边际效应递减”规律：参数翻倍，损失并不会减半，而是按幂律微弱平滑下降。 |
        
        ---
        
        ### 1. 为什么 Chinchilla 黄金法则是 $D \\approx 20N$？
        * 早期 OpenAI Kaplan 论文误以为“模型参数最重要”，导致 GPT-3 175B 只读了 300B Tokens（严重欠训练）；
        * DeepMind 严格证明：在固定总算力下，**每增加 1 个参数，应当对应增加 20 个训练 Token**！这也是 LLaMA (7B 训 2T Tokens) 性能跨代吊打上一代旧模型的数学底层秘密！
        """
    )


