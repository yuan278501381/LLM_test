# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 11: 音频信号与语音理解 (FFT -> Mel -> 连续帧切片) - 零基础入门教学平台

解剖时域波形、STFT、梅尔滤波器组与连续频谱帧切片，并说明它与 Whisper 完整前端的边界。
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
from nn_core.audio import (
    SpectrogramFramePatcher,
    compute_mel_spectrogram,
    generate_chord,
    generate_waveform,
    mel_filterbank,
    numpy_to_wav_bytes,
)

st.set_page_config(
    page_title="Audio Perception · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M11", show_contract=True)
render_core_result_evidence("M11")

render_hero_header(
    title="音频信号与语音理解架构",
    subtitle="从空气振动到时频特征：解剖波形、STFT、梅尔滤波器组与连续频谱帧切片（非 Whisper 复刻）",
    badge_text="MILESTONE 11 // AUDIO PERCEPTION",
    badge_type="purple",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="音频感知架构与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "音频合成控制台",
            "desc": "在左侧侧边栏调节基频、波形类型、泛音叠加与梅尔频段数",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解声波连续振动、STFT 傅里叶分光与梅尔耳蜗滤波",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时声学遥测",
            "desc": "显示采样率、频谱帧数、梅尔频段数与连续特征块数量",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "时域示波器与播放",
            "desc": "微观连续振动波形与纯 NumPy 实时合成标准 WAV 音频试听",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "梅尔滤波与帧切片",
            "desc": "梅尔滤波器组、2D log-Mel 特征与连续帧分组；非离散 tokenizer",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>大模型是如何'听到'声音的？</b><br>"
        f"人类耳朵听到的声音，在物理上只是<b>空气压强的连续时间振动波形</b>；<br>"
        f"计算机通过 <b>傅里叶变换 (FFT)</b> 像三棱镜分解白光一样，将复杂的混合波形拆解为不同频率成分；<br>"
        f"再通过模拟人耳非线性听觉的 <b>梅尔滤波器组 (Mel Filterbank)</b>，在 {anchor_badge('[E. 梅尔频谱图]', 'blue', target_id='region-e')} 将一维声音压缩为一张 2D 的<b>'声音热力图'</b>；<br>"
        f"语音模型会继续用卷积或投影网络把 log-Mel 特征变换为隐藏表示；这一步不同于文本的离散 tokenization。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>基频 (Frequency)</b>：声音的音高。例如中央 C 约为 261.6 Hz，国际标准音 A4 为 440 Hz。<br>"
        f"• <b>波形类型 (Wave Type)</b>：正弦波（纯音）、方波（复古电子音）、锯齿波（明亮丰富）。<br>"
        f"• <b>梅尔频段数 (Mel Bins)</b>：频域分辨率通道数，工业标准通常为 80 或 128。<br>"
        f"• <b>FFT 窗口大小</b>：时频权衡。窗口越小时间越准，窗口越大频率越准（测不准原理）。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 时域示波器]', 'purple', target_id='region-d')} 观测</b>：微观连续振动波形与真实声音试听。<br>"
        f"• <b>在 {anchor_badge('[E. 梅尔频谱图]', 'blue', target_id='region-e')} 观测</b>：对数梅尔功率谱能量分布。<br>"
        f"• <b>在 {anchor_badge('[C. 声学遥测]', 'emerald', target_id='region-c')} 评估</b>：采样率、频谱帧数与连续特征块数量。"
    ),
    experiments=[
        f"<b>第 1 步【聆听与观察纯音】</b>：点击 {anchor_badge('[D. 示波器]', 'purple', target_id='region-d')} 旁边的播放器试听 440Hz 纯音，并观察上方示波器中优美的正弦波形！",
        f"<b>第 2 步【合成和弦与观察泛音】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 勾选【叠加泛音】，观察频谱图上如何瞬间冒出 2 倍频与 3 倍频的尖锐峰值！",
        f"<b>第 3 步【观测梅尔声学图】</b>：在 {anchor_badge('[E. 梅尔声学图]', 'blue', target_id='region-e')} 观察 2D 梅尔热力图与三角形滤波器形状，理解为什么低频区域分布密集而高频区域分布稀疏！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>AUDIO CONTROLS // 音频信号控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

freq_val = st.sidebar.slider(
    "声音基频 (Base Frequency Hz)",
    min_value=220,
    max_value=880,
    value=440,
    step=20,
    help="440 Hz = 国际标准音 A4",
)

wave_options = [
    ("正弦波 (Sine Wave)", "单频基波 · 频域呈现单条锐利谱线"),
    ("方波 (Square Wave)", "包含全部奇次谐波 · 模拟合成管乐"),
    ("锯齿波 (Sawtooth Wave)", "包含全部整数次谐波 · 富含明亮高频"),
]

selected_wave_card = st.sidebar.radio(
    "波形类型 (Wave Type)",
    options=wave_options,
    format_func=lambda o: f"**{o[0]}**\n\n↳ *{o[1]}*",
    index=0,
)
wave_choice = selected_wave_card[0]

add_harmonics = st.sidebar.checkbox(
    "叠加复合泛音 (Harmonics 2x, 3x)",
    value=False,
    help="模拟真实乐器：叠加 2 倍频与 3 倍频次泛音",
)

n_mels_val = st.sidebar.select_slider(
    "梅尔频段数 (Mel Bins)",
    options=[40, 80, 128],
    value=80,
    help="Whisper 标准采用 80 个 Mel 频带",
)

fft_card_options = [
    (256, "FFT = 256 (高时间分辨率)", "时间定位精准 · 频率分辨率较低"),
    (512, "FFT = 512 (标准基线 · 默认)", "经典时频分辨率平衡点"),
    (1024, "FFT = 1024 (高频率分辨率)", "精细解析临近音高基频 · 时间有平滑"),
]

selected_fft_card = st.sidebar.radio(
    "FFT 窗口大小 (Window Size)",
    options=fft_card_options,
    format_func=lambda o: f"**{o[1]}**\n\n↳ *{o[2]}*",
    index=1,
)
n_fft_val = selected_fft_card[0]

# ---------------------------------------------------------------------------
# 音频生成与特征计算
# ---------------------------------------------------------------------------
sr = 16000
duration = 1.0

if add_harmonics:
    raw_signal = generate_chord(
        [freq_val, freq_val * 2.0, freq_val * 3.0], duration=duration, sr=sr
    )
else:
    raw_signal = generate_waveform(freq=freq_val, duration=duration, sr=sr, wave_type=wave_choice)

# 计算梅尔频谱图
mel_spec = compute_mel_spectrogram(
    raw_signal, sr=sr, n_fft=n_fft_val, hop_length=n_fft_val // 2, n_mels=n_mels_val
)

# Token 化
tokenizer = SpectrogramFramePatcher(n_mels=n_mels_val, frame_width=4)
audio_tokens = tokenizer.tokenize(mel_spec)

# 打包 WAV 字节流
wav_bytes = numpy_to_wav_bytes(raw_signal, sr=sr)

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "SAMPLE RATE // 采样率",
        f"{sr:,} HZ",
        delta="16-bit PCM Mono",
        delta_type="positive",
        icon_name="activity",
    )
    + render_metric_card(
        "FFT FREQ BINS // 频域通道",
        f"{n_fft_val // 2 + 1} BINS",
        delta=f"窗长 {n_fft_val} ({n_fft_val / sr * 1000:.1f}ms)",
        delta_type="positive",
        icon_name="target",
    )
    + render_metric_card(
        "MEL FILTER BANDS // 梅尔频段",
        f"{n_mels_val} BANDS",
        delta="对数临界频带",
        delta_type="positive",
        icon_name="layers",
    )
    + render_metric_card(
        "AUDIO TOKENS // 序列长度",
        f"{audio_tokens.shape[0]} TOKENS",
        delta=f"每帧 {audio_tokens.shape[1]}-Dim",
        delta_type="positive",
        icon_name="cpu",
    )
    + "</div>"
)
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">AUDIO TELEMETRY // 音频特征与时频分解遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 时域连续波形示波器
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">TIME-DOMAIN OSCILLOSCOPE // 时域连续波形与声音播放器</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="AUDIO SIGNAL & MEL-FREQUENCY DYNAMICS // 音频时频微观分解参数",
    badges=[
        {"label": "Sampling Rate fs", "value": f"{sr} Hz", "color": "blue"},
        {"label": "FFT Bins", "value": f"{n_fft_val // 2 + 1}", "color": "amber"},
        {"label": "Mel Bands", "value": f"{n_mels_val}", "color": "purple"},
        {"label": "Hop Size", "value": f"{n_fft_val // 2}", "color": "emerald"},
    ],
    metrics=[
        ("时长 Duration", f"{duration:.1f} s"),
        ("总音频帧数", f"{audio_tokens.shape[0]} frames"),
        ("Nyquist 极限", f"{sr // 2} Hz"),
    ],
    tag=f"SPECTROGRAM: {n_mels_val}x{audio_tokens.shape[0]} TENSORS",
    tag_color="emerald",
)

col_wave_plot, col_audio_play = st.columns([1.5, 0.8])

with col_wave_plot:
    # 截取前 0.02 秒观察微观波形 (320 个采样点)
    view_samples = int(sr * 0.02)
    time_axis = np.linspace(0, 0.02 * 1000, view_samples)
    fig_wave = go.Figure()
    fig_wave.add_trace(
        go.Scatter(
            x=time_axis,
            y=raw_signal[:view_samples],
            mode="lines",
            line=dict(color="#1d4ed8", width=2.2),
            name="时域波形",
            hovertemplate="时间: %{x:.2f} ms<br>振幅: %{y:.3f}<extra></extra>",
        )
    )
    fig_wave.update_layout(
        xaxis=dict(title="时间 (毫秒 / ms)"),
        yaxis=dict(title="振幅 (-1.0 ~ +1.0)", range=[-1.1, 1.1]),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig_wave = _apply_light_theme(fig_wave, "微观时域振动连续波形 (前 20ms)")
    st.plotly_chart(fig_wave, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 微观时域振动波形图", expanded=False):
        st.markdown(
            """
            * **横轴【时间 (毫秒 / ms)】** 与 **纵轴【空气压强振幅 (-1.0 ~ +1.0)】**。
            * **波形密集度**：波峰与波谷交替越频繁，说明声音**频率越高 (音调尖锐)**。
            * **波形高低**：振幅绝对值越大，说明声音**响度越大 (音量震耳)**。
            """
        )

with col_audio_play, st.container(border=True):
    st.markdown(f"#### [AUDIO PLAYER // 真实播放]\n**{wave_choice.split(' ')[0]} @ {freq_val} Hz**")
    st.caption("由纯 NumPy 实时合成，手写 44 字节 RIFF/WAVE 标准文件头：")
    st.audio(wav_bytes, format="audio/wav")
    st.markdown(
        f"""
            - **信号时长**：`1.0 秒`
            - **采样总点数**：`{len(raw_signal):,}` 点
            - **泛音状态**：`{"已叠加 (2x, 3x)" if add_harmonics else "纯单频"}`
            """
    )

# ---------------------------------------------------------------------------
# Section 2: 离散傅里叶变换 (FFT) 频谱分解
# ---------------------------------------------------------------------------
render_section_heading(
    "FOURIER FREQUENCY DECOMPOSITION // 快速傅里叶变换 (FFT) 频域分解", icon_name="target"
)

fft_full = np.abs(np.fft.rfft(raw_signal))
freq_axis = np.fft.rfftfreq(len(raw_signal), 1.0 / sr)
# 只展示 0~3000 Hz 主要人耳可闻区域
mask_3k = freq_axis <= 3000

fig_fft = go.Figure()
fig_fft.add_trace(
    go.Bar(
        x=freq_axis[mask_3k],
        y=fft_full[mask_3k],
        marker=dict(color="#6d28d9"),
        name="频域幅度",
        hovertemplate="频率: %{x:.1f} Hz<br>能量幅度: %{y:.1f}<extra></extra>",
    )
)
fig_fft.update_layout(
    xaxis=dict(title="频率 (Hz)"),
    yaxis=dict(title="能量谱幅度"),
    margin=dict(l=40, r=20, t=30, b=40),
)
fig_fft = _apply_light_theme(fig_fft, "FFT 全局频谱能量分布图 (0 ~ 3000 Hz)")
st.plotly_chart(fig_fft, width="stretch")
with st.expander("[HOW TO READ // 读图指南] FFT 全局频谱能量柱状图", expanded=False):
    st.markdown(
        """
        * **横轴【物理频率 (Hz)】** 与 **纵轴【该频率下的谐波能量强度】**。
        * **柱子尖峰位置**：声音中包含哪些音调成分（如 440 Hz 基频，以及 880 Hz、1320 Hz 泛音倍频）。
        * **[物理意义]**：在所选时间窗与离散采样条件下，傅里叶变换把信号表示为不同频率复指数（或正弦/余弦）分量；频谱分辨率和泄漏受窗长与窗函数影响。
        """
    )

# ---------------------------------------------------------------------------
# Section 3: 梅尔声学滤波器组与 2D 频谱图
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">MEL FILTERBANK & SPECTROGRAM // 梅尔滤波器组与 2D 对数梅尔频谱图</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_fb_plot, col_mel_heat = st.columns(2)

with col_fb_plot:
    # 绘制梅尔三角滤波器组几何形状
    fb_mat = mel_filterbank(sr=sr, n_fft=n_fft_val, n_mels=n_mels_val)
    fft_hz = np.linspace(0, sr / 2, n_fft_val // 2 + 1)
    fig_fb = go.Figure()
    # 抽样绘制 16 个滤波器避免过密
    step = max(1, n_mels_val // 16)
    for idx in range(0, n_mels_val, step):
        fig_fb.add_trace(
            go.Scatter(
                x=fft_hz,
                y=fb_mat[idx],
                mode="lines",
                line=dict(width=1.2),
                showlegend=False,
                hovertemplate=f"Mel 通道 {idx}<br>频率: %{{x:.1f}} Hz<br>权重: %{{y:.3f}}<extra></extra>",
            )
        )
    fig_fb.update_layout(
        xaxis=dict(title="频率 (Hz)", range=[0, sr / 2]),
        yaxis=dict(title="三角带通响应权重"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig_fb = _apply_light_theme(fig_fb, f"梅尔三角滤波器组 ({n_mels_val} 临界频带几何重叠)")
    st.plotly_chart(fig_fb, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 梅尔三角滤波器组分布", expanded=False):
        st.markdown(
            """
            * **低频段密集、高频段稀疏**：模拟人耳耳蜗对低频声音变化极度敏感、对高频不敏感的非线性生理特性。
            """
        )

with col_mel_heat:
    fig_spec = go.Figure(
        data=go.Heatmap(
            z=mel_spec,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(title="Log-Mel (dB)", thickness=10, len=0.8),
            hovertemplate="时间帧: %{x}<br>Mel 频段: %{y}<br>能量: %{z:.2f} dB<extra></extra>",
        )
    )
    fig_spec.update_layout(
        xaxis=dict(title="时间帧 (Time Frames)"),
        yaxis=dict(title="梅尔频段索引 (0 ~ N_mels)"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    fig_spec = _apply_light_theme(fig_spec, "2D 对数梅尔功率频谱图 (Log-Mel Spectrogram)")
    st.plotly_chart(fig_spec, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 2D 对数梅尔时频谱图", expanded=False):
        st.markdown(
            """
            * **横轴【时间流逝】**、**纵轴【梅尔频段高低】**、**颜色深浅【能量强弱 (dB)】**。
            * **[MODEL INPUT // 模型输入]**：语音模型通常先用卷积或线性投影处理 log-Mel 特征；它不是把频谱直接交给视觉 Transformer。
            """
        )

# ---------------------------------------------------------------------------
# Section 4: 连续声学特征与 Whisper 架构边界
# ---------------------------------------------------------------------------
render_section_heading(
    "CONTINUOUS AUDIO FEATURES // 连续声学特征与 Whisper 架构边界", icon_name="cpu"
)

col_w_pipe, col_w_info = st.columns([1.2, 1])

with col_w_pipe:
    with st.container(border=True):
        st.markdown(
            """
            #### [WHISPER ARCHITECTURE // 端到端声学-语言统一架构]
            ```
            [原始音频 16kHz] ──> [STFT 变换] ──> [80-Mel 滤波器] ──> [2D 对数频谱图]
                                                                        │
                                                                        ▼
            [最终识别文本] <── [Cross-Attention 解码] <── [Transformer Encoder] <── [Conv1D 跨步下采样]
            ```
            """
        )

with col_w_info, st.container(border=True):
    st.markdown(
        f"""
            #### [FRAME PATCHING // 连续声学帧切片]
            - **输入频谱图尺寸**：`{mel_spec.shape[0]} 频段 × {mel_spec.shape[1]} 时间帧`
            - **分帧打包倍率 (Frame Width)**：`4 帧/特征块`
            - **连续特征块数量**：`{audio_tokens.shape[0]} 个`
            - **单块展平维度**：`{audio_tokens.shape[1]} 维`
            - **结论边界**：这些是连续浮点特征，不是离散 token id，也不是 Whisper tokenizer。
            """
    )

# ---------------------------------------------------------------------------
# 零基础进阶：音频语音核心公式逐字拆解与名词通俗速查
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 音频信号处理与 Whisper 语音识别核心公式全解", expanded=True
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：离散傅里叶变换 (FFT) 与 梅尔刻度 (Mel Scale)
        $$X[k] = \\sum_{n=0}^{N-1} x[n] \\cdot e^{-j \\frac{2\\pi k n}{N}}$$
        $$m = 2595 \\cdot \\log_{10}\\left(1 + \\frac{f}{700}\\right)$$

        | 符号 | 中文名称 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---|
        | **$x[n]$** | **原始时域离散音频采样点** | 麦克风录到的声波振幅序列（比如 1 秒钟有 16000 个数值）。 |
        | **$X[k]$** | **频域第 $k$ 个频率通道能量** | 傅里叶变换后的结果：把混在一起的声音拆解为“低音鼓声有多强、人声有多强、高音笛声有多强”。 |
        | **$f$** | **物理频率 (Hz)** | 客观物理世界的声波振动频率（比如人耳可听范围 20Hz ~ 20000Hz）。 |
        | **$m$** | **主观听觉梅尔频率 (Mel)** | **人耳听觉仿真器**。人耳对低频非常敏感（100Hz 到 200Hz 听起来音调翻倍），但对高频很迟钝（10000Hz 到 10100Hz 几乎听不出区别）。梅尔公式将物理频率非线性压缩，低频放大分辨率，高频粗糙化。 |

        ---

        ### 1. 什么是【时频谱 (Spectrogram)】？—— “声音的指纹乐谱”
        * **横轴是时间**（从前奏到副歌）；**纵轴是音高频率**（从低音贝斯到高音女高音）；**颜色越亮代表该时刻该音高的嗓门越大**！
        * 整个语音识别（ASR）本质上就是把这张“彩色乐谱”当成一张图片，用视觉 Transformer 读出里面的文字！
        """
    )
