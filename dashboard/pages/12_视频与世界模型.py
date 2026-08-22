# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 12: 视频理解与世界模型 (Video -> Sora) - 零基础入门保姆级教学平台

解剖 3D 视频帧时空切片、空间 vs 时间多头注意力、自回归下一帧物理世界推演与 Sora/DiT 扩散去噪调度。
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

render_hero_header(
    title="视频理解与世界模型架构",
    subtitle="从像素序列到物理规律推演：解剖 3D 时空图块、空间与时间双轨注意力、世界模型下一帧预测与 Sora 扩散调度",
    badge_text="MILESTONE 12 // VIDEO & WORLD MODEL",
    badge_type="rose",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="视频大模型与世界模型入门",
    plain_intro=(
        "<b>视频不仅是图片的集合，更是物理世界在时间轴上的因果连续流！</b><br>"
        "大模型理解视频的核心技巧是 <b>3D 时空图块 (Spatio-Temporal Patches)</b>：把视频切成时空小方块，"
        "并用<b>空间注意力</b>看'这一刻画面里发生了什么'，用<b>时间注意力</b>追踪'这个物体运动到了哪里'；<br>"
        "更进一步的 <b>世界模型 (World Model)</b> 像人类的大脑一样，具备在脑海中推演未来画面的能力；<br>"
        "而 <b>Sora</b> 则将 Transformer 与 Diffusion 扩散去噪相结合，在纯高斯白噪声的泥土中，一步步雕刻出符合物理规律的高清动态视频！"
    ),
    hyperparams_desc=(
        "• <b>运动模式</b>：内置合成连续动力学轨迹（弹跳运动 / 匀速平移 / 扩散膨胀）。<br>"
        "• <b>视频总帧数</b>：时间序列长度（4 ~ 16 帧）。<br>"
        "• <b>Patch 尺寸</b>：空间切片分辨率（4x4 或 8x8）。<br>"
        "• <b>Diffusion 加噪步数</b>：模拟扩散前向轨迹的精细度时间表。"
    ),
    telemetry_desc=(
        "• <b>时空 Token 规模</b>：$T \\times (H/P) \\times (W/P)$ 总计打包的高维表征向量数量。<br>"
        "• <b>帧间运动能量</b>：相邻时间步画面像素位移的均方差 (MSE)。<br>"
        "• <b>下一帧预测误差</b>：世界模型推演未来物理轨迹的准确度。"
    ),
    experiments=[
        "<b>第 1 步【观察时空采样】</b>：在 Section 1 中观察 32x32 视频连续帧及下方的帧间运动能量曲线！",
        "<b>第 2 步【对比空间与时间注意力】</b>：在 Section 2 中对比两组热力图，体会模型如何将空间特征与时间轨迹解耦！",
        "<b>第 3 步【观测 Sora 噪声雕刻】</b>：在 Section 4 观察清晰视频帧如何一步步衰减为纯高斯白噪声，并理解去噪逆过程！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

motion_choice = st.sidebar.selectbox(
    "合成动力学运动模式",
    options=["弹跳运动 (Bounce)", "匀速平移 (Slide)", "中心膨胀 (Grow)"],
    index=0,
)

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
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 视频帧采样与动力学时间线
# ---------------------------------------------------------------------------
render_section_heading("VIDEO FRAME SEQUENCE // 32×32 视频时序采样与帧间动力学能量", icon_name="eye")

col_frames = st.columns(len(sampled_indices))
for idx, frame_idx in enumerate(sampled_indices):
    with col_frames[idx]:
        fig_f = go.Figure(
            data=go.Heatmap(
                z=sampled_video[idx, 0],
                colorscale="Viridis",
                showscale=False,
            )
        )
        fig_f.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
            margin=dict(l=5, r=5, t=25, b=5),
        )
        fig_f = _apply_light_theme(fig_f, f"Frame #{frame_idx} (T={frame_idx})")
        st.plotly_chart(fig_f, use_container_width=True)

# 帧间运动能量折线图
time_x = [f"F{i}->F{i+1}" for i in range(len(frame_diffs))]
fig_energy = go.Figure()
fig_energy.add_trace(
    go.Scatter(
        x=time_x,
        y=frame_diffs,
        mode="lines+markers",
        line=dict(color="#be123c", width=2.5),
        marker=dict(size=8, color="#be123c"),
        name="帧间差分 MSE",
        hovertemplate="时间过渡: %{x}<br>像素差分能量: %{y:.4f}<extra></extra>",
    )
)
fig_energy.update_layout(
    xaxis=dict(title="相邻帧间时序过渡"),
    yaxis=dict(title="均方差像素能量 (MSE)"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_energy = _apply_light_theme(fig_energy, "视频时序动力学帧间差分能量曲线 (Motion Energy)")
st.plotly_chart(fig_energy, use_container_width=True)
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
render_section_heading("SPATIAL VS TEMPORAL ATTENTION // 空间与时间注意力双轨解耦", icon_name="target")

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
    st.plotly_chart(fig_sp, use_container_width=True)
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
    st.plotly_chart(fig_tp, use_container_width=True)
    with st.expander("[HOW TO READ // 读图指南] 时间注意力时序追踪矩阵", expanded=False):
        st.markdown(
            """
            * **跨越时间维度**：当前帧该位置与过去所有历史帧的关联（追踪运动轨迹与速度惯性）。
            """
        )

# ---------------------------------------------------------------------------
# Section 3: 世界模型自回归下一帧物理预测
# ---------------------------------------------------------------------------
render_section_heading("WORLD MODEL FUTURE PREDICTION // 世界模型自回归下一帧物理规律推演", icon_name="cpu")

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
    fig_p = _apply_light_theme(fig_p, f"模型推演预测画面 (T={n_frames_val-1})")
    st.plotly_chart(fig_p, use_container_width=True)

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
    fig_t = _apply_light_theme(fig_t, f"物理世界真实画面 (T={n_frames_val-1})")
    st.plotly_chart(fig_t, use_container_width=True)

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
# Section 4: Diffusion 噪声调度与 Sora 架构
# ---------------------------------------------------------------------------
render_section_heading("DIFFUSION NOISE SCHEDULE // 扩散加噪轨迹与 Sora 架构全景", icon_name="activity")

scheduler = DiffusionScheduler(num_steps=diffusion_steps)
snapshots = visualize_diffusion_process(raw_image_0 := full_video[0, 0], scheduler=scheduler, steps_to_show=5)

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
        fig_ns = _apply_light_theme(fig_ns, f"Step t={step_t} ({'原图' if step_t==0 else ('纯噪声' if step_t==diffusion_steps-1 else '加噪中')})")
        st.plotly_chart(fig_ns, use_container_width=True)

# 调度曲线图
sched_data = scheduler.get_schedule()
fig_sched = go.Figure()
fig_sched.add_trace(
    go.Scatter(x=sched_data["steps"], y=sched_data["betas"], mode="lines", name="β_t (加噪率)", line=dict(color="#be123c", width=2))
)
fig_sched.add_trace(
    go.Scatter(x=sched_data["steps"], y=sched_data["alphas_cumprod"], mode="lines", name="ᾱ_t (信号保留率)", line=dict(color="#1d4ed8", width=2))
)
fig_sched.update_layout(
    xaxis=dict(title="扩散时间步 (Diffusion Steps t)"),
    yaxis=dict(title="方差调度参数值 (0.0 ~ 1.0)"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_sched = _apply_light_theme(fig_sched, "Diffusion 前向方差与信号保留率调度曲线")
st.plotly_chart(fig_sched, use_container_width=True)
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
        #### [SORA GENERATIVE PIPELINE // 文本到视频生成全景]
        ```
        [用户 Prompt 提示词] ──> [T5/CLIP 文本编码]
                                         │
                                         ▼ (交叉注意力注入引导)
        [3D 高斯白噪声视频] ──> [Diffusion Transformer (DiT)] ──> [逐步去噪迭代 (T 步)] ──> [高清 3D 物理视频]
        ```
        - **时空统一**：将视频帧切片为 3D Patches，直接送入 Transformer 进行时空全注意力建模；
        - **逆向雕塑**：训练模型预测每一步注入的噪声 $\\epsilon_\\theta(x_t, t, c)$，在推理时从纯噪声中雕刻出符合物理定律的世界！
        """
    )

# ---------------------------------------------------------------------------
# 零基础进阶：视频生成与世界模型核心公式拆解
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 扩散模型与 Sora 视频生成核心公式全解", expanded=True):
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
        
        ### 1. 什么是【Sora / DiT 逆向去噪】？—— “大理石雕塑”
        * **训练阶段（砸碎雕像）**：我们给高清照片不断加噪，教神经网络学会识别“当前画面里哪部分是噪声”。
        * **生成阶段（雕刻成型）**：给模型一段纯高斯白噪声（一块未开垦的粗糙大理石）和一段文字 Prompt，模型像米开朗基罗一样，一步步削去多余的噪声，最终涌现出一张完全符合物理常识的高清视频！
        """
    )

