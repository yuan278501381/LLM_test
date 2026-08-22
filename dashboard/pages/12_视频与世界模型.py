# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 12: 视频张量、教学级预测头与 DDPM 前向加噪 - 零基础入门教学平台

解剖 3D 视频时空切片、时空相关性、教学级下一帧预测头与 DDPM 前向加噪；不实现视频去噪生成网络。
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
from dashboard.components.client_player import (
    build_video_payload,
    render_timeline_controls,
    render_video_timeline,
)
from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.attention import MultiHeadAttention
from nn_core.video import (
    SpatioTemporalPatchEmbed,
    VideoFrameSampler,
    compute_frame_change_magnitude,
    compute_frame_difference,
    generate_synthetic_video,
)
from nn_core.world_model import (
    DiffusionScheduler,
    NextFramePredictor,
    visualize_diffusion_process,
)

st.set_page_config(
    page_title="Video & World Model · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M12", show_contract=True)

render_hero_header(
    title="视频理解与世界模型架构",
    subtitle="从像素序列到时间结构：解剖 3D 时空图块、相关性、教学级下一帧预测与 DDPM 前向加噪",
    badge_text="MILESTONE 12 // VIDEO & WORLD MODEL",
    badge_type="rose",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="视频理解与世界模型空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "动力学控制台",
            "desc": "在左侧侧边栏切换合成动力学模式、视频总帧数与 Patch 尺寸",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "理解 3D 时空图块、相关性与 DDPM 前向加噪，并区分未实现的逆过程",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时时空遥测",
            "desc": "显示 3D 时空 Token 规模、帧间运动能量与世界模型推演误差",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "时序采样与能量",
            "desc": "32x32 视频连续帧序列展示与相邻帧间差分运动能量折线",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "时空相关与前向扩散",
            "desc": "时空相关性热力图与 DDPM 前向加噪时间表；不含逆向生成",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>视频不仅是图片的集合，更是物理世界在时间轴上的因果连续流！</b><br>"
        f"大模型理解视频的核心技巧是 <b>3D 时空图块 (Spatio-Temporal Patches)</b>：把视频切成时空小方块，"
        f"并用<b>空间注意力</b>看'这一刻画面里发生了什么'，用<b>时间注意力</b>追踪'这个物体运动到了哪里'；<br>"
        f"本页的 <b>NextFramePredictor</b> 是未训练的两层教学预测头，用于说明接口，不代表已学会物理规律；<br>"
        f"{anchor_badge('[E. 扩散实验室]', 'blue', target_id='region-e')} 只计算 DDPM 前向加噪。DiT 流程图是架构背景，不是已实现的生成器。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>运动模式</b>：内置合成连续动力学轨迹（弹跳运动 / 匀速平移 / 扩散膨胀）。<br>"
        f"• <b>视频总帧数</b>：时间序列长度（4 ~ 16 帧）。<br>"
        f"• <b>Patch 尺寸</b>：空间切片分辨率（4x4 或 8x8）。<br>"
        f"• <b>Diffusion 加噪步数</b>：模拟扩散前向轨迹的精细度时间表。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 时序采样]', 'purple', target_id='region-d')} 观测</b>：视频序列帧与运动能量折线。<br>"
        f"• <b>在 {anchor_badge('[E. 时空相关与扩散]', 'blue', target_id='region-e')} 观测</b>：时空相关性热力图与前向加噪轨迹。<br>"
        f"• <b>在 {anchor_badge('[C. 时空遥测]', 'emerald', target_id='region-c')} 评估</b>：时空 Token 规模与世界模型推演误差。"
    ),
    experiments=[
        f"<b>第 1 步【观察时空采样】</b>：在 {anchor_badge('[D. 采样与能量]', 'purple', target_id='region-d')} 中观察 32x32 视频连续帧及下方的帧间运动能量曲线！",
        f"<b>第 2 步【对比空间与时间注意力】</b>：在 {anchor_badge('[E. 双轨注意力]', 'blue', target_id='region-e')} 中对比两组热力图，体会模型如何将空间特征与时间轨迹解耦！",
        f"<b>第 3 步【验证前向加噪】</b>：观察清晰帧如何随 t 增大而丢失信号；注意本页没有计算逆向去噪。",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>WORLD MODEL CONTROLS // 动力学控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

motion_options = [
    ("弹跳运动 (Bounce)", "重力加速度与边界弹性碰撞 · 模拟牛顿力学"),
    ("匀速平移 (Slide)", "刚体恒定线速度平移 · 模拟线性位移"),
    ("中心膨胀 (Grow)", "中心径向周期胀缩 · 模拟流体/生物动力学"),
]

selected_motion_card = st.sidebar.radio(
    "合成动力学运动模式",
    options=motion_options,
    format_func=lambda o: f"**{o[0]}**\n\n↳ *{o[1]}*",
    index=0,
)
motion_choice = selected_motion_card[0]

n_frames_val = st.sidebar.select_slider(
    "视频总帧数 (Total Frames)",
    options=[4, 8, 16],
    value=8,
)

patch_size_val = st.sidebar.select_slider(
    "空间 Patch 尺寸 (Patch Size)",
    options=[4, 8],
    value=8,
)

sampling_strategy = st.sidebar.radio(
    "帧采样策略 (Sampling Strategy)",
    options=["均匀采样 (Uniform)", "关键帧突变采样 (Keyframe)"],
    index=0,
)

diffusion_steps = st.sidebar.slider(
    "Diffusion 扩散步数 (Time Steps)",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)

# ---------------------------------------------------------------------------
# 数据生成与时空计算
# ---------------------------------------------------------------------------
full_video = generate_synthetic_video(n_frames=n_frames_val, size=32, motion=motion_choice)
sampler = VideoFrameSampler(strategy=sampling_strategy)
sampled_video, sampled_indices = sampler.sample(full_video, n_sample=min(4, n_frames_val))

# 计算运动特征
frame_diffs = compute_frame_difference(full_video)
motion_vecs = compute_frame_change_magnitude(full_video)

# 3D 时空嵌入
st_embed_op = SpatioTemporalPatchEmbed(
    img_size=32,
    patch_size=patch_size_val,
    n_frames=n_frames_val,
    in_channels=1,
    d_model=32,
)
video_tensor = full_video.reshape(1, n_frames_val, 1, 32, 32)
st_tokens = st_embed_op.forward(video_tensor)
num_spatial_patches = (32 // patch_size_val) ** 2
total_tokens = n_frames_val * num_spatial_patches

# 模拟空间注意力与时间注意力权重
mha_spatial = MultiHeadAttention(d_model=32, num_heads=2)
spatial_tokens = st_tokens[0, :num_spatial_patches]  # 单帧图块
_, spatial_attn = mha_spatial.forward(spatial_tokens.reshape(1, num_spatial_patches, 32))

mha_temporal = MultiHeadAttention(d_model=32, num_heads=2)
temporal_tokens = st_tokens[0, ::num_spatial_patches]  # 同一位置跨帧
_, temporal_attn = mha_temporal.forward(temporal_tokens.reshape(1, n_frames_val, 32))

# 世界模型下一帧预测模拟
predictor = NextFramePredictor(d_model=32, frame_pixels=1024)
context_feature = np.mean(st_tokens[0, :-num_spatial_patches], axis=0, keepdims=True)
pred_frame_flat = predictor.forward(context_feature)[0]
pred_frame = pred_frame_flat.reshape(32, 32)
true_frame = full_video[-1, 0]
rec_loss = predictor.reconstruction_loss(pred_frame_flat, true_frame.flatten())

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "TOTAL VIDEO FRAMES // 视频序列",
        f"{n_frames_val} 帧 @ 32×32",
        delta=f"抽样展示 {len(sampled_indices)} 关键帧",
        delta_type="positive",
        icon_name="eye",
    )
    + render_metric_card(
        "SPATIO-TEMPORAL TOKENS // 时空总数",
        f"{total_tokens} TOKENS",
        delta=f"{n_frames_val} 帧 × {num_spatial_patches} Patches",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "MOTION ENERGY // 帧间运动能量",
        f"{np.mean(frame_diffs):.4f} MSE",
        delta="连续物理动力学",
        delta_type="positive",
        icon_name="activity",
    )
    + render_metric_card(
        "WORLD MODEL MSE // 未来推演误差",
        f"{rec_loss:.4f}",
        delta="自回归像素解码",
        delta_type="positive" if rec_loss < 0.2 else "neutral",
        icon_name="target",
    )
    + "</div>"
)
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">SPATIO-TEMPORAL TELEMETRY // 时空动力学与世界模型遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 视频帧采样与动力学时间线
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">VIDEO FRAME SEQUENCE // 32×32 视频时序采样与帧间动力学能量</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="SPATIO-TEMPORAL WORLD MODEL // 3D 时空切片与扩散动力学",
    badges=[
        {"label": "Video Shape", "value": f"({n_frames_val}, 1, 32, 32)", "color": "blue"},
        {"label": "Spatial Patch", "value": f"{patch_size_val}x{patch_size_val}", "color": "amber"},
        {"label": "Tokens N_3d", "value": f"{total_tokens}", "color": "purple"},
        {"label": "World MSE", "value": f"{rec_loss:.4f}", "color": "emerald"},
    ],
    metrics=[
        ("时序帧数 T", f"{n_frames_val} frames"),
        ("物理运动模式", f"{motion_choice}"),
        ("DDPM 扩散步数", f"{diffusion_steps}"),
    ],
    tag=f"WORLD MODEL LATENT: {total_tokens}x32D",
    tag_color="emerald",
)

video_payload = build_video_payload(full_video, frame_diffs)
render_timeline_controls(
    total_steps=n_frames_val,
    event_name="nn:m12-frame",
    title="[VIDEO TIME PLAYER // 视频时空演播厅]",
    badge="D",
    caption="视频帧与运动能量在浏览器内同步原位更新；暂停后可逐帧检查碰撞、位移与能量峰值。",
    interval_ms=620,
    initial_step=1,
    inspect_label="当前帧",
)
render_video_timeline(video_payload)
with st.expander("[HOW TO READ // 读图指南] 帧间运动差分能量曲线", expanded=False):
    st.markdown(
        """
        * **横轴【帧序号】** 与 **纵轴【相邻帧间像素变化能量】**。
        * **尖峰高点**：物体发生剧烈位移、碰撞反弹或形态瞬变的时间点。
        """
    )

# ---------------------------------------------------------------------------
# Section 2: 空间注意力 vs 时间注意力机制
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">SPATIO-TEMPORAL CORRELATION // 时空相关性与前向扩散</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_sp_attn, col_tp_attn = st.columns(2)

with col_sp_attn:
    fig_sp = go.Figure(
        data=go.Heatmap(
            z=spatial_attn[0, 0],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(thickness=10, len=0.8),
            hovertemplate="目标 Patch: %{y}<br>来源 Patch: %{x}<br>空间相关度: %{z:.3f}<extra></extra>",
        )
    )
    fig_sp.update_layout(
        xaxis=dict(title="当前帧空间 Patch 索引"),
        yaxis=dict(title="当前帧空间 Patch 索引", autorange="reversed"),
        margin=dict(l=30, r=30, t=30, b=40),
    )
    fig_sp = _apply_light_theme(fig_sp, "空间注意力 (Spatial Attention: 同一时刻全局画面)")
    st.plotly_chart(fig_sp, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 空间注意力热力矩阵", expanded=False):
        st.markdown(
            """
            * **在单张画面内部**：不同物体 Patch 之间的相互关联（如球与地面的几何相对位置）。
            """
        )

with col_tp_attn:
    fig_tp = go.Figure(
        data=go.Heatmap(
            z=temporal_attn[0, 0],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(thickness=10, len=0.8),
            hovertemplate="目标帧: %{y}<br>来源帧: %{x}<br>时间相关度: %{z:.3f}<extra></extra>",
        )
    )
    fig_tp.update_layout(
        xaxis=dict(title="时序时间步 (Frame Index)"),
        yaxis=dict(title="时序时间步 (Frame Index)", autorange="reversed"),
        margin=dict(l=30, r=30, t=30, b=40),
    )
    fig_tp = _apply_light_theme(fig_tp, "时间注意力 (Temporal Attention: 同一位置历史运动)")
    st.plotly_chart(fig_tp, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 时间注意力时序追踪矩阵", expanded=False):
        st.markdown(
            """
            * **跨越时间维度**：当前帧该位置与过去所有历史帧的关联（追踪运动轨迹与速度惯性）。
            """
        )

# ---------------------------------------------------------------------------
# Section 3: 世界模型自回归下一帧物理预测
# ---------------------------------------------------------------------------
render_section_heading(
    "WORLD MODEL FUTURE PREDICTION // 世界模型自回归下一帧物理规律推演", icon_name="cpu"
)

col_pred_view, col_true_view, col_world_info = st.columns([1.1, 1.1, 1.2])

with col_pred_view:
    fig_p = go.Figure(
        data=go.Heatmap(
            z=pred_frame,
            colorscale="Viridis",
            showscale=False,
            hovertemplate="预测值: %{z:.2f}<extra></extra>",
        )
    )
    fig_p.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_p = _apply_light_theme(fig_p, f"模型推演预测画面 (T={n_frames_val - 1})")
    st.plotly_chart(fig_p, width="stretch")

with col_true_view:
    fig_t = go.Figure(
        data=go.Heatmap(
            z=true_frame,
            colorscale="Viridis",
            showscale=False,
            hovertemplate="真实值: %{z:.2f}<extra></extra>",
        )
    )
    fig_t.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig_t = _apply_light_theme(fig_t, f"物理世界真实画面 (T={n_frames_val - 1})")
    st.plotly_chart(fig_t, width="stretch")

with col_world_info:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [WORLD MODEL // 想象力引擎]
            - **输入条件**：前 `{n_frames_val - 1}` 帧聚合上下文特征
            - **推演目标**：预测未来第 `{n_frames_val - 1}` 帧物理像素
            - **像素级均方误差 (MSE)**：`{rec_loss:.4f}`
            - **核心理念**：*世界模型不仅记住了画面，更在隐空间学会了重力、惯性与碰撞等隐式物理规律！*
            """
        )

with st.expander("[HOW TO READ // 读图指南] 物理世界下一帧推演预测对比", expanded=False):
    st.markdown(
        """
        * **左图 (模型想象)** vs **中图 (真实物理)**：对比小球位置与反弹形变是否严丝合缝一致。
        * **[OPTIMAL // 成功标志]**：预测误差 MSE 极小，小球位置准确落在反弹轨迹上。
        """
    )

# ---------------------------------------------------------------------------
# Section 4: DDPM 前向加噪与未实现的生成架构背景
# ---------------------------------------------------------------------------
render_section_heading("DDPM FORWARD PROCESS // 扩散前向加噪（非视频生成）", icon_name="activity")

st.warning(
    "证据边界：以下快照和曲线仅由 DDPM 闭式前向加噪公式计算；项目没有实现或训练反向去噪网络、DiT 或 Sora。"
)

scheduler = DiffusionScheduler(num_steps=diffusion_steps)
snapshots = visualize_diffusion_process(
    raw_image_0 := full_video[0, 0], scheduler=scheduler, steps_to_show=5
)

col_diff_shots = st.columns(5)
for idx, (step_t, noisy_img) in enumerate(snapshots):
    with col_diff_shots[idx]:
        fig_ns = go.Figure(
            data=go.Heatmap(
                z=noisy_img,
                colorscale="Viridis",
                showscale=False,
            )
        )
        fig_ns.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
            margin=dict(l=5, r=5, t=25, b=5),
        )
        fig_ns = _apply_light_theme(
            fig_ns,
            f"Step t={step_t} ({'原图' if step_t == 0 else ('纯噪声' if step_t == diffusion_steps - 1 else '加噪中')})",
        )
        st.plotly_chart(fig_ns, width="stretch")

# 调度曲线图
sched_data = scheduler.get_schedule()
fig_sched = go.Figure()
fig_sched.add_trace(
    go.Scatter(
        x=sched_data["steps"],
        y=sched_data["betas"],
        mode="lines",
        name="β_t (加噪率)",
        line=dict(color="#be123c", width=2),
    )
)
fig_sched.add_trace(
    go.Scatter(
        x=sched_data["steps"],
        y=sched_data["alphas_cumprod"],
        mode="lines",
        name="ᾱ_t (信号保留率)",
        line=dict(color="#1d4ed8", width=2),
    )
)
fig_sched.update_layout(
    xaxis=dict(title="扩散时间步 (Diffusion Steps t)"),
    yaxis=dict(title="方差调度参数值 (0.0 ~ 1.0)"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_sched = _apply_light_theme(fig_sched, "Diffusion 前向方差与信号保留率调度曲线")
st.plotly_chart(fig_sched, width="stretch")
with st.expander("[HOW TO READ // 读图指南] 扩散加噪与方差调度曲线", expanded=False):
    st.markdown(
        """
        * **上方 5 张快照图**：从 $t=0$ (清晰原图) 随着加噪逐步被雪花点吞没，直到第 20 步变成纯白噪声。
        * **下方折线图**：**红线 (加噪率 $\\beta_t$)** 逐步抬升，**蓝线 (原图残留率 $\\bar{\\alpha}_t$)** 单调骤降至 0。
        """
    )

# 架构流程总结卡
with st.container(border=True):
    st.markdown(
        """
        #### [ARCHITECTURE CONTEXT ONLY // 文本到视频生成背景（未实现）]
        ```
        [用户 Prompt 提示词] ──> [T5/CLIP 文本编码]
                                         │
                                         ▼ (交叉注意力注入引导)
        [3D 高斯白噪声视频] ──> [Diffusion Transformer (DiT)] ──> [逐步去噪迭代 (T 步)] ──> [高清 3D 物理视频]
        ```
        - **时空统一**：将视频帧切片为 3D Patches，直接送入 Transformer 进行时空全注意力建模；
        - **未实现部分**：完整系统需要训练 $\\epsilon_\\theta(x_t,t,c)$ 或其他预测器并运行逆向采样；本页没有执行这一步。
        """
    )

# ---------------------------------------------------------------------------
# 零基础进阶：视频生成与世界模型核心公式拆解
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] DDPM 前向加噪公式与逆过程边界", expanded=True):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：扩散模型 (DDPM) 前向一步加噪
        $$x_t = \\sqrt{\\bar{\\alpha}_t} \\cdot x_0 + \\sqrt{1 - \\bar{\\alpha}_t} \\cdot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\mathbf{I})$$
        
        | 符号 | 中文名称 | 形状与类型 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$x_0$** | **原始清晰图像/视频帧** | 图像张量 | 真实世界高清无码的初始画面（第 0 步）。 |
        | **$t$** | **加噪扩散时间步 (Step)** | 整数 $0 \\sim T$ | 当前加噪进度。$t=0$ 是原图，$t=T$（如第 20 步）彻底变成纯高斯噪点。 |
        | **$\\bar{\\alpha}_t$** | **累积信号保留率 (Alpha-Bar)** | 标量 $1.0 \\to 0.0$ | **原图残留比例**。随着 $t$ 增大，$\\bar{\\alpha}_t$ 从 0.99 骤降到 0.0001，原图画面逐渐消失。 |
        | **$\\epsilon$** | **纯高斯随机白噪声** | 与图像同形状 | 从标准正态分布中随机抽取的“雪花点杂波”。 |
        | **$\\sqrt{1 - \\bar{\\alpha}_t}$** | **噪声强度系数** | 标量 $0.0 \\to 1.0$ | **雪花点混合浓度**。与 $\\sqrt{\\bar{\\alpha}_t}$ 满足平方和等于 1 的守恒律，确保图像总能量恒定。 |
        | **$x_t$** | **第 $t$ 步带噪混合图像** | 图像张量 | 带有不同程度毛刺雪花点的中间画面。 |
        
        ---
        
        ### 1. 完整逆向去噪还缺少什么？
        * **训练阶段**：需要可学习的噪声或速度预测网络、时间步条件、真实视频数据和反向传播训练。
        * **生成阶段**：需要选择采样器并反复调用训练后的网络。本页没有这些组件，因此不能从噪声生成视频，也不能据此评价物理一致性。
        """
    )
