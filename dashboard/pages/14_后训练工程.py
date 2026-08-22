# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 14: 后训练与对齐工程 (Post-training & Alignment) - 零基础入门保姆级教学平台

解剖 SFT 监督微调、RLHF 人类偏好对齐、DPO 直接偏好优化与 LoRA 低秩轻量微调。
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
from nn_core.lora import compute_param_savings
from nn_core.posttraining import AlignmentPipeline, generate_before_after_examples
from nn_core.rlhf import DPOLoss, PPOClipObjective, RewardModel

st.set_page_config(
    page_title="Post-training Alignment · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="后训练对齐与轻量微调架构",
    subtitle="从'胡言乱语的接龙机器'蜕变为'温文尔雅的超级助手'：解剖 SFT 指令微调、RLHF 强化学习、DPO 偏好优化与 LoRA 低秩矩阵分解",
    badge_text="MILESTONE 14 // POST-TRAINING ALIGNMENT",
    badge_type="amber",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="后训练对齐与微调工程入门",
    plain_intro=(
        "<b>为什么预训练完的 GPT 还不能直接当 ChatGPT 用？</b><br>"
        "基座模型读完了互联网全部文章，但他只会'接着你的话往下猜'，既不懂礼貌、也不会听指令、甚至会教人做坏事；<br>"
        "<b>后训练 (Post-training)</b> 是赋予大模型'灵魂与教养'的蜕变工程：<br>"
        "1. <b>SFT (监督指令微调)</b>：用高质量问答对教模型学会'像助手一样说话'；<br>"
        "2. <b>RLHF / DPO (偏好对齐)</b>：通过奖励惩罚机制，让模型变得<b>有用 (Helpful)、诚实 (Honest)、无害 (Harmless)</b>；<br>"
        "3. <b>LoRA (低秩微调)</b>：只修改不到 1% 的旁路参数，在一张消费级显卡上就能让大模型掌握全新专业领域的知识！"
    ),
    hyperparams_desc=(
        "• <b>PPO Clip 截断系数 ($\\epsilon$)</b>：强化学习更新步长安全带，防止策略模型更新过猛崩溃。<br>"
        "• <b>DPO 温度系数 ($\\beta$)</b>：直接偏好优化中对参考模型的偏离惩罚项。<br>"
        "• <b>LoRA 秩 (Rank $r$)</b>：低秩矩阵的瓶颈维度。秩越小显存越省，秩越大表达能力越强。<br>"
        "• <b>主干矩阵维度 ($d$)</b>：待微调大模型隐藏层宽度。"
    ),
    telemetry_desc=(
        "• <b>LoRA 显存压缩比</b>：相较于全量微调，参数量节约的倍数。<br>"
        "• <b>六维能力演进画像</b>：模型在有用性、无害性、安全性等维度的综合雷达得分。<br>"
        "• <b>RLHF 奖励与 KL 轨迹</b>：强化学习训练过程中奖励上升与稳定性监控。"
    ),
    experiments=[
        "<b>第 1 步【对比同一问题三阶段回答】</b>：在 Section 2 查看 5 个真实问题，观察同一个模型在 Pretrain -> SFT -> RLHF 下回答质量的巨大跃迁！",
        "<b>第 2 步【观察六维能力雷达演进】</b>：在 Section 3 勾选不同的训练阶段，观察大模型如何从偏科的'野蛮天才'进化为'全能专家'！",
        "<b>第 3 步【体验 LoRA 暴跌的参数量】</b>：在左侧调整 LoRA Rank，观察 Section 5 中参数量如何直接缩减 98% 以上！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

ppo_eps = st.sidebar.slider(
    "PPO Clip 截断阈值 (Epsilon ε)",
    min_value=0.10,
    max_value=0.30,
    value=0.20,
    step=0.02,
    help="推荐 0.20",
)

dpo_beta = st.sidebar.slider(
    "DPO 隐式奖励系数 (Beta β)",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.02,
    help="推荐 0.10",
)

lora_rank_val = st.sidebar.select_slider(
    "LoRA 低秩瓶颈 (Rank r)",
    options=[1, 2, 4, 8, 16, 32],
    value=4,
)

lora_dmodel = st.sidebar.selectbox(
    "主干模型维度 (d_model)",
    options=[64, 128, 256, 512, 1024],
    index=3,
)

stage_options = list(AlignmentPipeline.STAGE_SCORES.keys())
selected_stages = st.sidebar.multiselect(
    "雷达图展示阶段",
    options=stage_options,
    default=stage_options,
)

# ---------------------------------------------------------------------------
# 计算指标
# ---------------------------------------------------------------------------
lora_stats = compute_param_savings(d_model=lora_dmodel, rank=lora_rank_val)
rlhf_traj = PPOClipObjective.simulate_rlhf_trajectory(n_steps=20)

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "LORA COMPRESSION // 参数压缩比",
        f"{lora_stats['compression_ratio']:.1f} ×",
        delta=f"节约 {lora_stats['saved_percent']:.1f}% 显存",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TRAINABLE PARAMS // 仅训练参数",
        f"{lora_stats['lora_params']:,}",
        delta=f"原参数量 {lora_stats['original_params']:,}",
        delta_type="positive",
        icon_name="layers",
    )
    + render_metric_card(
        "RLHF REWARD SCORE // 终态奖励分",
        f"+{rlhf_traj['reward'][-1]:.2f}",
        delta="人类价值观对齐",
        delta_type="positive",
        icon_name="target",
    )
    + render_metric_card(
        "DPO BETA STRENGTH // 偏好优化强度",
        f"β = {dpo_beta:.2f}",
        delta="零强化学习闭式解",
        delta_type="positive",
        icon_name="activity",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 全生命周期流水线 (Pre-train -> SFT -> RLHF -> DPO)
# ---------------------------------------------------------------------------
render_section_heading("TRAINING LIFECYCLE // 大语言模型四阶段全生命周期演进", icon_name="cpu")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    with st.container(border=True):
        st.markdown(
            """
            #### [1. 预训练 // PRE-TRAIN]
            - **数据**：万亿无标注互联网语料
            - **目标**：下一词预测 (CLM)
            - **状态**：**野蛮天才**
            - **缺点**：不听指令、答非所问
            """
        )

with col_s2:
    with st.container(border=True):
        st.markdown(
            """
            #### [2. 监督微调 // SFT]
            - **数据**：数十万精标指令问答对
            - **目标**：标准交叉熵微调
            - **状态**：**听话学徒**
            - **缺点**：易被诱导越狱、格式呆板
            """
        )

with col_s3:
    with st.container(border=True):
        st.markdown(
            """
            #### [3. 偏好对齐 // RLHF]
            - **数据**：人类偏好打分排序对
            - **目标**：PPO 策略梯度优化
            - **状态**：**得体助手**
            - **亮点**：有用、诚实、主动拒绝有害
            """
        )

with col_s4:
    with st.container(border=True):
        st.markdown(
            """
            #### [4. 直接优化 // DPO]
            - **数据**：同源 Chosen/Rejected 对
            - **目标**：隐式奖励对数比率损失
            - **状态**：**极速对齐**
            - **亮点**：无须训练奖励模型，极度稳健
            """
        )

# ---------------------------------------------------------------------------
# Section 2: 真实案例演进对比 (同一问题的三阶段质变)
# ---------------------------------------------------------------------------
render_section_heading("BEFORE VS AFTER CASE STUDY // 同一指令在不同阶段下的回答质变实录", icon_name="target")

cases = generate_before_after_examples()

for case_idx, case_data in enumerate(cases):
    with st.expander(f"📌 案例 {case_idx + 1}：【{case_data['category']}】— \"{case_data['prompt']}\"", expanded=(case_idx == 0)):
        c_pre, c_sft, c_rlhf = st.columns(3)
        with c_pre:
            with st.container(border=True):
                st.markdown("##### 🔴 预训练基座回答 (Pre-training)")
                st.caption("只会胡乱接龙，缺乏指令遵循意识：")
                st.info(case_data["pretrain"])
        with c_sft:
            with st.container(border=True):
                st.markdown("##### 🔵 SFT 指令微调回答 (Supervised FT)")
                st.caption("能听懂意图并回答，但深度与安全性欠佳：")
                st.warning(case_data["sft"])
        with c_rlhf:
            with st.container(border=True):
                st.markdown("##### 🟢 RLHF / DPO 对齐回答 (Aligned)")
                st.caption("结构严谨、通俗深刻、安全合规：")
                st.success(case_data["rlhf"])

# ---------------------------------------------------------------------------
# Section 3: 六维能力雷达图演进
# ---------------------------------------------------------------------------
render_section_heading("6D RADAR EVOLUTION // 模型能力画像六维雷达演进图", icon_name="activity")

categories = ["有用性", "无害性", "诚实性", "指令跟随", "创造力", "安全性"]
colors_map = {
    "Pre-training (基座接龙)": ("#64748b", "rgba(100, 116, 139, 0.15)"),
    "SFT (指令微调)": ("#1d4ed8", "rgba(29, 78, 216, 0.2)"),
    "RLHF (人类偏好对齐)": ("#047857", "rgba(4, 120, 87, 0.25)"),
    "DPO (直接偏好优化)": ("#6d28d9", "rgba(109, 40, 217, 0.25)"),
}

fig_radar = go.Figure()

for stg_name in selected_stages:
    scores = AlignmentPipeline.get_stage_scores(stg_name)
    r_vals = [scores[c] for c in categories] + [scores[categories[0]]]
    theta_vals = categories + [categories[0]]
    line_c, fill_c = colors_map.get(stg_name, ("#000000", "rgba(0,0,0,0.1)"))

    fig_radar.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill="toself",
            fillcolor=fill_c,
            line=dict(color=line_c, width=2.2),
            name=stg_name,
            hovertemplate="阶段: " + stg_name + "<br>维度: %{theta}<br>得分: %{r}<extra></extra>",
        )
    )

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]),
    ),
    showlegend=True,
    margin=dict(l=40, r=40, t=40, b=40),
)
fig_radar = _apply_light_theme(fig_radar, "大模型各训练阶段六维能力画像雷达对比")
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: RLHF PPO 轨迹 vs DPO 直接偏好机制
# ---------------------------------------------------------------------------
render_section_heading("RLHF VS DPO DYNAMICS // PPO 训练收敛轨迹与 DPO 隐式偏好代换", icon_name="activity")

col_ppo_plot, col_dpo_card = st.columns([1.3, 1])

with col_ppo_plot:
    fig_ppo = go.Figure()
    fig_ppo.add_trace(
        go.Scatter(
            x=rlhf_traj["step"],
            y=rlhf_traj["reward"],
            mode="lines+markers",
            name="Reward 得分 (↑)",
            line=dict(color="#047857", width=2.5),
        )
    )
    fig_ppo.add_trace(
        go.Scatter(
            x=rlhf_traj["step"],
            y=rlhf_traj["kl_div"],
            mode="lines+markers",
            name="KL 散度约束 (稳定受控)",
            line=dict(color="#b45309", width=2, dash="dot"),
        )
    )
    fig_ppo.update_layout(
        xaxis=dict(title="PPO 训练步数 (Training Steps)"),
        yaxis=dict(title="数值指标"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig_ppo = _apply_light_theme(fig_ppo, f"RLHF PPO 策略训练轨迹 (Clip={ppo_eps})")
    st.plotly_chart(fig_ppo, use_container_width=True)

with col_dpo_card:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [DPO 突破性革新 // 告别复杂 RL]
            - **传统 RLHF**：需要同时维护 4 个模型（Actor, Critic, Ref, Reward），显存消耗巨大，训练极其脆弱；
            - **DPO 巧思**：直接推导出隐式奖励封闭解：
            $$r^*(x, y) = \\beta \\log \\frac{{\\pi_\\theta(y|x)}}{{\\pi_{{ref}}(y|x)}}$$
            - **单步二元交叉熵**：直接计算偏好对损失，训练稳定如 SFT，显存立减 $50\\%$！
            """
        )

# ---------------------------------------------------------------------------
# Section 5: LoRA 低秩矩阵分解参数可视化
# ---------------------------------------------------------------------------
render_section_heading("LORA LOW-RANK DECOMPOSITION // LoRA 旁路低秩矩阵分解与参数量对比", icon_name="layers")

ranks_cmp = [1, 2, 4, 8, 16, 32]
params_list = [compute_param_savings(lora_dmodel, r)["lora_params"] for r in ranks_cmp]
orig_p = lora_dmodel * lora_dmodel

fig_lora = go.Figure()
fig_lora.add_trace(
    go.Bar(
        x=[f"Rank {r}" for r in ranks_cmp],
        y=params_list,
        marker_color="#1d4ed8",
        name="LoRA 可训练参数量",
        hovertemplate="配置: %{x}<br>可训练参数: %{y:,}<extra></extra>",
    )
)
fig_lora.add_hline(
    y=orig_p,
    line_dash="dash",
    line_color="#be123c",
    annotation_text=f"全量微调参数量 ({orig_p:,})",
    annotation_position="top left",
)
fig_lora.update_layout(
    xaxis=dict(title="LoRA 秩配置 (Rank)"),
    yaxis=dict(title="参数量 (Parameters)"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_lora = _apply_light_theme(fig_lora, f"LoRA 不同 Rank 下与全量微调参数量对比 (d_model={lora_dmodel})")
st.plotly_chart(fig_lora, use_container_width=True)

# ---------------------------------------------------------------------------
# 零基础进阶：后训练对齐与 LoRA 核心公式拆解
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 后训练对齐 (RLHF/DPO) 与 LoRA 微调核心公式全解", expanded=True):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：LoRA 低秩矩阵分解
        $$h = x W_0 + \\frac{\\alpha}{r} (x A) B$$
        
        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$W_0$** | **预训练原始冻结权重** | $d \\times d$ (如 $4096 \\times 4096$) | **被冰冻的大脑底座**。包含大模型千亿知识，训练时 100% 保持只读，不消耗反向梯度显存！ |
        | **$A$** | **降维投影旁路矩阵** | $d \\times r$ (如 $4096 \\times 8$) | **特征压缩漏斗**。高斯随机初始化，把 4096 维高维特征压缩到极窄的 $r=8$ 维低秩空间。 |
        | **$B$** | **升维重构旁路矩阵** | $r \\times d$ (如 $8 \\times 4096$) | **特征放大镜**。初始化为纯全零，把 $r=8$ 维特征重新放大回 4096 维。因为初始为 0，所以初始状态整个旁路输出为 0，完全不扰动原始模型！ |
        | **$r$** | **LoRA 秩 (Rank)** | 整数 (如 $4, 8, 16$) | **窄桥的宽度**。$r$ 越小（如 4 或 8），参数量越少（省 99% 显存）；$r$ 越大，微调拟合能力越强。 |
        | **$\\alpha / r$** | **缩放缩放因子 (Alpha)** | 标量常数 | **微调补丁的音量旋钮**。决定让新增的 LoRA 补丁在最终输出中占多大话语权。 |
        
        ---
        
        ### 1. 核心公式逐字拆解：DPO (Direct Preference Optimization) 偏好对齐损失
        $$L_{\\text{DPO}} = -\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w | x)}{\\pi_{\\text{ref}}(y_w | x)} - \\beta \\log \\frac{\\pi_\\theta(y_l | x)}{\\pi_{\\text{ref}}(y_l | x)}\\right)$$
        
        * **$y_w$ (Winner / 优秀回答)** vs **$y_l$ (Loser / 差劲回答)**：人类标注或 AI 评审出的胜出回答与败北回答；
        * **$\\pi_\\theta / \\pi_{\\text{ref}}$ (当前策略 vs 原始底座)**：评估模型在给出某个回答时，比原始模型更有信心还是更没信心；
        * **$\\beta$ (KL 惩罚系数)**：防止模型为了讨好人类而彻底胡言乱语（防止偏离原始底座知识太远）。
        """
    )

