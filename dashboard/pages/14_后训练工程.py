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

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_floating_hud_navigator,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.lora import compute_param_savings
from nn_core.posttraining import AlignmentPipeline, generate_before_after_examples
from nn_core.rlhf import PPOClipObjective

st.set_page_config(
    page_title="Post-training Alignment · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M14", show_contract=True)
render_core_result_evidence("M14")

render_hero_header(
    title="后训练对齐与轻量微调架构",
    subtitle="区分 SFT、RLHF/PPO、DPO 与 LoRA 的目标函数片段、完整训练协议和页内模拟",
    badge_text="MILESTONE 14 // POST-TRAINING ALIGNMENT",
    badge_type="amber",
)

render_floating_hud_navigator(
    [
        {
            "id": "A",
            "name": "对齐参数控制台",
            "desc": "在左侧侧边栏调节 PPO Clip 阈值、DPO 奖励系数与 LoRA Rank",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解从基座接龙到 SFT 听话学徒与 RLHF/DPO 对齐助手",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时对齐遥测",
            "desc": "显示 LoRA 参数节约比例、微调后总显存节省与当前对齐状态",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "三阶段回答实录",
            "desc": "对比人工编写的三阶段模板回答（非模型输出）",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "六维雷达画像",
            "desc": "有用性、无害性、诚实性与安全性等全方位能力画像演进雷达图",
            "color": "blue",
            "target_id": "region-e",
        },
    ]
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="后训练对齐与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "对齐参数控制台",
            "desc": "在左侧侧边栏调节 PPO Clip 阈值、DPO 奖励系数与 LoRA Rank",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解从基座接龙到 SFT 听话学徒与 RLHF/DPO 对齐助手",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时对齐遥测",
            "desc": "显示 LoRA 参数节约比例、微调后总显存节省与当前对齐状态",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "三阶段回答实录",
            "desc": "用预置模板对比三个训练阶段可能造成的回答形式差异",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "六维雷达画像",
            "desc": "有用性、无害性、诚实性与安全性等全方位能力画像演进雷达图",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        "<b>为什么预训练完的 GPT 还不能直接当 ChatGPT 用？</b><br>"
        "基座模型读完了互联网全部文章，但他只会'接着你的话往下猜'，既不懂礼貌、也不会听指令、甚至会教人做坏事；<br>"
        "<b>后训练 (Post-training)</b> 是赋予大模型'灵魂与教养'的蜕变工程：<br>"
        "1. <b>SFT (监督指令微调)</b>：用高质量问答对教模型学会'像助手一样说话'；<br>"
        "2. <b>RLHF / DPO (偏好优化)</b>：利用人类/规则偏好数据改变行为分布；不保证真实、安全或无偏。<br>"
        "3. <b>LoRA (低秩微调)</b>：冻结基座权重并学习低秩更新；参数节省比例取决于目标层、尺寸和 rank，不保证新知识或任务成功。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>PPO Clip 截断系数 ($\\epsilon$)</b>：强化学习更新步长安全带，防止策略模型更新过猛崩溃。<br>"
        f"• <b>DPO 温度系数 ($\\beta$)</b>：直接偏好优化中对参考模型的偏离惩罚项。<br>"
        f"• <b>LoRA 秩 (Rank $r$)</b>：低秩矩阵的瓶颈维度。秩越小显存越省，秩越大表达能力越强。<br>"
        f"• <b>主干矩阵维度 ($d$)</b>：待微调大模型隐藏层宽度。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[C. 对齐遥测]', 'emerald', target_id='region-c')} 评估</b>：LoRA 显存压缩比与参数量节约倍数。<br>"
        f"• <b>在 {anchor_badge('[E. 六维雷达]', 'blue', target_id='region-e')} 观测</b>：模型在有用性、无害性、安全性等维度的综合雷达得分。<br>"
        f"• <b>在 {anchor_badge('[D. 模板案例]', 'purple', target_id='region-d')} 对比</b>：预置文本的形式差异；它不是模型训练日志。"
    ),
    experiments=[
        f"<b>第 1 步【识别模板案例】</b>：在 {anchor_badge('[D. 案例实录]', 'purple', target_id='region-d')} 对比人工编写的阶段模板；它们用于讲解期望行为，不是同一模型的实测输出。",
        f"<b>第 2 步【观察六维能力雷达演进】</b>：在 {anchor_badge('[E. 六维雷达]', 'blue', target_id='region-e')} 勾选不同的训练阶段，观察大模型如何从偏科的'野蛮天才'进化为'全能专家'！",
        f"<b>第 3 步【体验 LoRA 暴跌的参数量】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调整 LoRA Rank，观察参数量如何直接缩减 98% 以上！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>ALIGNMENT CONTROLS // 对齐控制台</b></div>',
    unsafe_allow_html=True,
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

dmodel_opts = [64, 128, 256, 512, 1024]
lora_dmodel = st.sidebar.segmented_control(
    "主干模型维度 (d_model)",
    options=dmodel_opts,
    default=512,
)
lora_dmodel = lora_dmodel or 512

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

st.warning(
    "本页只计算 PPO clip、DPO loss 和 LoRA 参数量片段，并显示规则曲线/模板回答。"
    "完整 RLHF/PPO 还需要 rollout、reward model、reference policy、KL 约束、优势估计和优化循环；"
    "完整 DPO/LoRA 还需要偏好/任务数据、训练循环、验证集与基线评估。"
)

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
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">POST-TRAINING TELEMETRY // 对齐收益与显存压缩遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 全生命周期流水线 (Pre-train -> SFT -> RLHF -> DPO)
# ---------------------------------------------------------------------------
render_section_heading("TRAINING LIFECYCLE // 大语言模型四阶段全生命周期演进", icon_name="cpu")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1, st.container(border=True):
    st.markdown(
        """
            #### [1. 预训练 // PRE-TRAIN]
            - **数据**：万亿无标注互联网语料
            - **目标**：下一词预测 (CLM)
            - **状态**：**野蛮天才**
            - **缺点**：不听指令、答非所问
            """
    )

with col_s2, st.container(border=True):
    st.markdown(
        """
            #### [2. 监督微调 // SFT]
            - **数据**：数十万精标指令问答对
            - **目标**：标准交叉熵微调
            - **状态**：**听话学徒**
            - **缺点**：易被诱导越狱、格式呆板
            """
    )

with col_s3, st.container(border=True):
    st.markdown(
        """
            #### [3. 偏好对齐 // RLHF]
            - **数据**：人类偏好打分排序对
            - **目标**：PPO 策略梯度优化
            - **状态**：**得体助手**
            - **亮点**：有用、诚实、主动拒绝有害
            """
    )

with col_s4, st.container(border=True):
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
# Section 2: 预置案例对比（不是同一模型的真实训练轨迹）
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">TEMPLATE COMPARISON // 同一指令的预置阶段示例（概率模拟）</span>'
    f"</div>",
    unsafe_allow_html=True,
)

cases = generate_before_after_examples()

for case_idx, case_data in enumerate(cases):
    with st.expander(
        f'[CASE] 案例 {case_idx + 1}：【{case_data["category"]}】— "{case_data["prompt"]}"',
        expanded=(case_idx == 0),
    ):
        c_pre, c_sft, c_rlhf = st.columns(3)
        with c_pre, st.container(border=True):
            st.markdown("##### [FAILED] 预训练基座回答 (Pre-training)")
            st.caption("只会胡乱接龙，缺乏指令遵循意识：")
            st.info(case_data["pretrain"])
        with c_sft, st.container(border=True):
            st.markdown("##### [SFT] SFT 指令微调回答 (Supervised FT)")
            st.caption("能听懂意图并回答，但深度与安全性欠佳：")
            st.warning(case_data["sft"])
        with c_rlhf, st.container(border=True):
            st.markdown("##### [PASSED] RLHF / DPO 对齐回答 (Aligned)")
            st.caption("结构严谨、通俗深刻、安全合规：")
            st.success(case_data["rlhf"])

# ---------------------------------------------------------------------------
# Section 3: 六维能力雷达图演进
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">6D RADAR EVOLUTION // 模型能力画像六维雷达演进图</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="POST-TRAINING & LORA DYNAMICS // 后训练微调与对齐微观参数",
    badges=[
        {"label": "LoRA Rank r", "value": f"{lora_rank_val}", "color": "blue"},
        {"label": "d_model", "value": f"{lora_dmodel}", "color": "amber"},
        {"label": "PPO Clip ε", "value": f"{ppo_eps:.2f}", "color": "purple"},
        {"label": "DPO Beta β", "value": f"{dpo_beta:.2f}", "color": "emerald"},
    ],
    metrics=[
        ("参数压缩比", f"{lora_stats['compression_ratio']:.1f}x"),
        ("显存节省率", f"{lora_stats['saved_percent']:.1f}%"),
        ("对齐奖励峰值", f"+{rlhf_traj['reward'][-1]:.2f}"),
    ],
    tag=f"LORA SAVINGS: {lora_stats['saved_percent']:.1f}% VRAM",
    tag_color="emerald",
)

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
    theta_vals = [*categories, categories[0]]
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
st.plotly_chart(fig_radar, width="stretch")
with st.expander("[HOW TO READ // 读图指南] 六维对齐能力雷达图", expanded=False):
    st.markdown(
        """
        * **6 个顶点维度**：有用性、无害性、诚实性、指令跟随、创造力、安全性。
        * **[SIMULATION // 模拟边界]**：雷达图数值是预设分数，不是评测数据。实际 SFT/RLHF/DPO 可改变多种行为，也会存在能力回归、偏好偏差、reward hacking 和安全权衡。
        """
    )

# ---------------------------------------------------------------------------
# Section 4: RLHF PPO 轨迹 vs DPO 直接偏好机制
# ---------------------------------------------------------------------------
render_section_heading(
    "RLHF VS DPO DYNAMICS // PPO 训练收敛轨迹与 DPO 隐式偏好代换", icon_name="activity"
)

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
    st.plotly_chart(fig_ppo, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] RLHF PPO 策略强化学习轨迹", expanded=False):
        st.markdown(
            """
            * **绿线【奖励得分 (Reward)】** 稳步爬升；**棕虚线【KL 散度约束】** 稳定在低位未发生失控（防止模型对齐崩塌）。
            """
        )

with col_dpo_card:
    with st.container(border=True):
        st.markdown(
            """
            #### [DPO // 直接偏好目标]
            - **传统 RLHF**：需要同时维护 4 个模型（Actor, Critic, Ref, Reward），显存消耗巨大，训练极其脆弱；
            - **DPO 巧思**：直接推导出隐式奖励封闭解：
            $$r^*(x, y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{ref}(y|x)}$$
            - **单步二元交叉熵**：直接计算偏好对损失，训练稳定如 SFT，显存立减 $50\\%$！
            """
        )

# ---------------------------------------------------------------------------
# Section 5: LoRA 低秩矩阵分解参数可视化
# ---------------------------------------------------------------------------
render_section_heading(
    "LORA LOW-RANK DECOMPOSITION // LoRA 旁路低秩矩阵分解与参数量对比", icon_name="layers"
)

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
fig_lora = _apply_light_theme(
    fig_lora, f"LoRA 不同 Rank 下与全量微调参数量对比 (d_model={lora_dmodel})"
)
st.plotly_chart(fig_lora, width="stretch")
with st.expander("[HOW TO READ // 读图指南] LoRA 参数量削减对比柱状图", expanded=False):
    st.markdown(
        """
        * **红色虚线**：全量微调的可训练权重数；**蓝色柱子**：当前简化层中 LoRA 的可训练参数。参数减少不等于同比例显存节省；优化器状态、激活、量化和目标层都会影响实际显存。
        """
    )

# ---------------------------------------------------------------------------
# 零基础进阶：后训练对齐与 LoRA 核心公式拆解
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 后训练对齐 (RLHF/DPO) 与 LoRA 微调核心公式全解", expanded=True
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：LoRA 低秩矩阵分解
        $$h = x W_0 + \\frac{\\alpha}{r} (x A) B$$

        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$W_0$** | **预训练原始冻结权重** | $d \\times d$ (如 $4096 \\times 4096$) | LoRA 配置中不更新该矩阵参数；但前向激活、优化器状态及实现细节仍会占用显存。 |
        | **$A$** | **降维投影旁路矩阵** | $d \\times r$ (如 $4096 \\times 8$) | **特征压缩漏斗**。高斯随机初始化，把 4096 维高维特征压缩到极窄的 $r=8$ 维低秩空间。 |
        | **$B$** | **升维重构旁路矩阵** | $r \\times d$ (如 $8 \\times 4096$) | **特征放大镜**。初始化为纯全零，把 $r=8$ 维特征重新放大回 4096 维。因为初始为 0，所以初始状态整个旁路输出为 0，完全不扰动原始模型！ |
        | **$r$** | **LoRA 秩 (Rank)** | 整数 (如 $4, 8, 16$) | **窄桥的宽度**。$r$ 越小（如 4 或 8），参数量越少（省 99% 显存）；$r$ 越大，微调拟合能力越强。 |
        | **$\\alpha / r$** | **缩放缩放因子 (Alpha)** | 标量常数 | **微调补丁的音量旋钮**。决定让新增的 LoRA 补丁在最终输出中占多大话语权。 |

        ---

        ### 1. 核心公式逐字拆解：DPO (Direct Preference Optimization) 偏好对齐损失
        $$L_{\\text{DPO}} = -\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w | x)}{\\pi_{\\text{ref}}(y_w | x)} - \\beta \\log \\frac{\\pi_\\theta(y_l | x)}{\\pi_{\\text{ref}}(y_l | x)}\\right)$$

        * **$y_w$ (Winner / 优秀回答)** vs **$y_l$ (Loser / 差劲回答)**：人类标注或 AI 评审出的胜出回答与败北回答；
        * **$\\pi_\\theta / \\pi_{\\text{ref}}$ (当前策略 vs 原始底座)**：评估模型在给出某个回答时，比原始模型更有信心还是更没信心；
        * **$\\beta$ (KL 惩罚系数)**：惩罚策略相对参考模型的偏移；它可约束更新幅度，但不能单独保证事实性、安全性或避免奖励投机。

        ---

        ### 2. 核心公式逐字拆解：PPO-Clip 目标函数 (RLHF 策略更新)
        $$L_{\\text{CLIP}}(\\theta) = \\mathbb{E}_t \\left[ \\min\\left( r_t(\\theta) \\hat{A}_t, \\; \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]$$

        | 符号 | 中文名称 | 通俗大白话解释 |
        |:---:|:---:|:---|
        | **$r_t(\\theta) = \\frac{\\pi_\\theta(a_t \\| s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t \\| s_t)}$** | **概率比率 (Probability Ratio)** | 新策略在该步给出同样回答的概率 / 旧策略的概率。$r_t > 1$ 说明新策略更倾向该回答，$r_t < 1$ 说明新策略不太想给这个回答了。 |
        | **$\\hat{A}_t$** | **优势估计 (Advantage)** | **"这个回答比平均水平好了多少分"**。正值 = 超出预期的好回答，负值 = 低于预期的差回答。 |
        | **$\\epsilon$** | **代理目标裁剪范围** | 常见示例取 $\\epsilon=0.2$；裁剪作用于代理目标中的比率项，并不把更新后的实际概率比率硬限制在 $[0.8,1.2]$。 |
        | **$\\min(\\cdot, \\cdot)$** | **保守代理项** | 在未裁剪与裁剪后的代理项中取较小者，限制部分方向上的更新激励；它不保证训练永不发散或策略一定改进。 |
        """
    )
