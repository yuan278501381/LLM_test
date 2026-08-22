# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 11: 音频信号与语音理解 (FFT -> Mel -> Whisper) - 零基础入门保姆级教学平台

解剖时域连续振动波形、STFT 短时傅里叶变换、梅尔听觉滤波器组与现代语音模型 (Whisper) Token 化流程。
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
from nn_core.audio import (
    AudioTokenizer,
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

render_hero_header(
    title="音频信号与语音理解架构",
    subtitle="从空气分子振动到语音语义：解剖时域波形、STFT 短时傅里叶变换、梅尔滤波器组与 Whisper 语音 Token 化",
    badge_text="MILESTONE 11 // AUDIO PERCEPTION",
    badge_type="purple",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="音频信号处理与语音大模型入门",
    plain_intro=(
        "<b>大模型是如何'听到'声音的？</b><br>"
        "人类耳朵听到的声音，在物理上只是<b>空气压强的连续时间振动波形</b>；<br>"
        "计算机通过 <b>傅里叶变换 (FFT)</b> 像三棱镜分解白光一样，将复杂的混合波形拆解为不同频率成分；<br>"
        "再通过模拟人耳非线性听觉的 <b>梅尔滤波器组 (Mel Filterbank)</b>，将一维声音压缩为一张 2D 的<b>'声音热力图 (Mel Spectrogram)'</b>；<br>"
        "最后像 Whisper 这样的现代大模型，就可以直接<b>像阅读文本一样阅读这张声音频谱图</b>！"
    ),
    hyperparams_desc=(
        "• <b>基频 (Frequency)</b>：声音的音高。例如中央 C 约为 261.6 Hz，国际标准音 A4 为 440 Hz。<br>"
        "• <b>波形类型 (Wave Type)</b>：正弦波（纯音）、方波（复古电子音）、锯齿波（明亮丰富）。<br>"
        "• <b>梅尔频段数 (Mel Bins)</b>：频域分辨率通道数，工业标准通常为 80 或 128。<br>"
        "• <b>FFT 窗口大小</b>：时频权衡。窗口越小时间越准，窗口越大频率越准（测不准原理）。"
    ),
    telemetry_desc=(
        "• <b>采样率与波形点数</b>：16,000 Hz 工业标准单声道实时波形采样。<br>"
        "• <b>FFT 频域特征谱</b>：离散傅里叶变换提取的各频率分量振幅响应。<br>"
        "• <b>语音 Token 序列长度</b>：音频分帧打包为大模型输入向量的 Token 数量。"
    ),
    experiments=[
        "<b>第 1 步【聆听与观察纯音】</b>：点击 Section 1 的播放器试听 440Hz 纯音，并观察上方示波器中优美的正弦波形！",
        "<b>第 2 步【合成和弦与观察泛音】</b>：在左侧勾选【叠加泛音】，在 Section 2 观察频谱图上如何瞬间冒出 2 倍频与 3 倍频的尖锐峰值！",
        "<b>第 3 步【观测梅尔声学图】</b>：在 Section 3 观察 2D 梅尔热力图与三角形滤波器形状，理解为什么低频区域分布密集而高频区域分布稀疏！",
    ],
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

wave_choice = st.sidebar.selectbox(
    "波形类型 (Wave Type)",
    options=["正弦波 (Sine Wave)", "方波 (Square Wave)", "锯齿波 (Sawtooth Wave)"],
    index=0,
)

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

n_fft_val = st.sidebar.selectbox(
    "FFT 窗口大小 (Window Size)",
    options=[256, 512, 1024],
    index=1,
)

# ---------------------------------------------------------------------------
# 音频生成与特征计算
# ---------------------------------------------------------------------------
sr = 16000
duration = 1.0

if add_harmonics:
    raw_signal = generate_chord([freq_val, freq_val * 2.0, freq_val * 3.0], duration=duration, sr=sr)
else:
    raw_signal = generate_waveform(freq=freq_val, duration=duration, sr=sr, wave_type=wave_choice)

# 计算梅尔频谱图
mel_spec = compute_mel_spectrogram(
    raw_signal, sr=sr, n_fft=n_fft_val, hop_length=n_fft_val // 2, n_mels=n_mels_val
)

# Token 化
tokenizer = AudioTokenizer(n_mels=n_mels_val, frame_width=4)
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
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 时域连续波形示波器
# ---------------------------------------------------------------------------
render_section_heading("TIME-DOMAIN OSCILLOSCOPE // 时域连续振动波形示波器", icon_name="activity")

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
    fig_wave = _apply_light_theme(fig_wave, f"微观时域振动连续波形 (前 20ms)")
    st.plotly_chart(fig_wave, use_container_width=True)

with col_audio_play:
    with st.container(border=True):
        st.markdown(f"#### [AUDIO PLAYER // 真实播放]\n**{wave_choice.split(' ')[0]} @ {freq_val} Hz**")
        st.caption("由纯 NumPy 实时合成，手写 44 字节 RIFF/WAVE 标准文件头：")
        st.audio(wav_bytes, format="audio/wav")
        st.markdown(
            f"""
            - **信号时长**：`1.0 秒`
            - **采样总点数**：`{len(raw_signal):,}` 点
            - **泛音状态**：`{'已叠加 (2x, 3x)' if add_harmonics else '纯单频'}`
            """
        )

# ---------------------------------------------------------------------------
# Section 2: 离散傅里叶变换 (FFT) 频谱分解
# ---------------------------------------------------------------------------
render_section_heading("FOURIER FREQUENCY DECOMPOSITION // 快速傅里叶变换 (FFT) 频域分解", icon_name="target")

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
st.plotly_chart(fig_fft, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: 梅尔声学滤波器组与 2D 频谱图
# ---------------------------------------------------------------------------
render_section_heading("MEL FILTERBANK & SPECTROGRAM // 梅尔滤波器组与 2D 对数梅尔频谱图", icon_name="layers")

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
    st.plotly_chart(fig_fb, use_container_width=True)

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
    st.plotly_chart(fig_spec, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4: Whisper 语音模型端到端流水线 (2026 前沿)
# ---------------------------------------------------------------------------
render_section_heading("WHISPER AUDIO-TO-TOKEN PIPELINE // 现代语音模型 Token 化流水线", icon_name="cpu")

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

with col_w_info:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [AUDIO TOKENIZATION // 声学特征 Token 化]
            - **输入频谱图尺寸**：`{mel_spec.shape[0]} 频段 × {mel_spec.shape[1]} 时间帧`
            - **分帧打包倍率 (Frame Width)**：`4 帧/Token`
            - **最终大模型 Token 数量**：`{audio_tokens.shape[0]} 个连续表征向量`
            - **单 Token 嵌入维度**：`{audio_tokens.shape[1]} 维`
            - **核心启示**：*语音与文本在大模型深层没有本质区别，都是高维嵌入向量的序列自回归！*
            """
        )

# ---------------------------------------------------------------------------
# 零基础进阶：音频语音核心公式逐字拆解与名词通俗速查
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 音频信号处理与 Whisper 语音识别核心公式全解", expanded=True):
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

