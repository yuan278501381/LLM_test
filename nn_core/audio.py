# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.audio - 纯 NumPy 音频时域波形合成、STFT 短时傅里叶变换、梅尔滤波器组与语音 Token 化

包含：
- `generate_waveform` / `generate_chord`: 基础波形 (正弦/方波/锯齿波) 与复音和弦合成
- `stft`: 基于 Hanning 窗的短时离散傅里叶变换 (STFT)
- `hz_to_mel` / `mel_to_hz`: 赫兹与梅尔听觉感知尺度非线性映射
- `mel_filterbank`: 三角梅尔重叠带通滤波器组构建 (连续浮点三角插值，严防零行)
- `compute_mel_spectrogram`: 对数梅尔功率频谱图提取流水线
- `AudioTokenizer`: 类似 Whisper 的音频特征时域分帧 Token 化器
- `numpy_to_wav_bytes`: 零依赖纯 struct 序列化打包标准 RIFF/WAVE 字节流
"""

import io
import logging
import struct
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("nn_core.audio")


def generate_waveform(
    freq: float = 440.0,
    duration: float = 1.0,
    sr: int = 16000,
    wave_type: str = "sine"
) -> np.ndarray:
    """
    合成指定频率与类型的单声道时域连续波形信号。
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if wave_type == "square" or "方波" in wave_type:
        signal = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "sawtooth" or "锯齿" in wave_type:
        signal = 2.0 * (t * freq - np.floor(t * freq + 0.5))
    else:  # sine / 正弦波
        signal = np.sin(2 * np.pi * freq * t)
    return signal.astype(np.float32)


def generate_chord(
    freqs: list[float],
    duration: float = 1.0,
    sr: int = 16000
) -> np.ndarray:
    """
    多频谐波与和弦合成叠加。
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * t)
    # 振幅峰值归一化，防止削波失真
    max_val = np.max(np.abs(signal))
    if max_val > 1e-6:
        signal = signal / max_val * 0.9
    return signal.astype(np.float32)


def stft(
    signal: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """
    纯 NumPy 短时傅里叶变换 (Short-Time Fourier Transform)。

    数学原理：
        $$X(m, \\omega) = \\sum_{n=0}^{N-1} x(m \\cdot H + n) \\cdot w(n) \\cdot e^{-j \\omega n}$$
    返回复数频谱矩阵 shape: (n_fft // 2 + 1, n_frames)
    """
    window = np.hanning(n_fft)
    
    pad_amount = n_fft // 2
    padded_signal = np.pad(signal, (pad_amount, pad_amount), mode="reflect")
    
    num_frames = (len(padded_signal) - n_fft) // hop_length + 1
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    
    for t in range(num_frames):
        start = t * hop_length
        frames[t] = padded_signal[start : start + n_fft] * window
        
    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    return stft_matrix.T


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """赫兹频率转梅尔频标 (HTK 经典公式)"""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """梅尔频标转回赫兹频率"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80
) -> np.ndarray:
    """
    构建符合人类听觉耳蜗临界频带的重叠三角梅尔滤波器组矩阵 (基于标准连续频点三角插值)。
    返回 shape: (n_mels, n_fft // 2 + 1)
    """
    num_freq_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, num_freq_bins)

    f_min = 0.0
    f_max = sr / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    
    # 在 Mel 刻度上等间距采样 n_mels + 2 个锚点
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    filterbank = np.zeros((n_mels, num_freq_bins), dtype=np.float32)
    
    for m in range(1, n_mels + 1):
        f_left = hz_points[m - 1]
        f_center = hz_points[m]
        f_right = hz_points[m + 1]
        
        # 三角上升区与下降区
        up = (fft_freqs - f_left) / (f_center - f_left + 1e-8)
        down = (f_right - fft_freqs) / (f_right - f_center + 1e-8)
        tri = np.maximum(0.0, np.minimum(up, down))
        
        # 若由于离散分辨率导致峰值未采样，确保在最接近点至少有响应
        if np.sum(tri) == 0.0:
            closest_idx = np.argmin(np.abs(fft_freqs - f_center))
            tri[closest_idx] = 1.0
            
        # Slaney 归一化 (面积归一化)
        enorm = 2.0 / (f_right - f_left + 1e-8)
        filterbank[m - 1] = tri * enorm

    return filterbank


def compute_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 80
) -> np.ndarray:
    """
    计算对数梅尔功率频谱图 (Log-Mel Spectrogram)。
    返回 shape: (n_mels, n_frames)
    """
    spec_complex = stft(signal, n_fft=n_fft, hop_length=hop_length)
    power_spec = np.abs(spec_complex) ** 2
    
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(fb, power_spec)
    
    log_mel_spec = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel_spec.astype(np.float32)


class AudioTokenizer:
    """
    音频时域分帧 Token 化器（类似 OpenAI Whisper / Google AudioPaLM）。
    将 2D 梅尔频谱图沿时间轴切分打包为 1D 连续 Token 嵌入向量序列。
    """

    def __init__(self, n_mels: int = 80, frame_width: int = 4) -> None:
        self.n_mels = n_mels
        self.frame_width = frame_width

    def tokenize(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        mel_spec shape: (n_mels, n_frames)
        返回 token 序列: (num_tokens, n_mels * frame_width)
        """
        n_mels, n_frames = mel_spec.shape
        num_tokens = n_frames // self.frame_width
        
        truncated = mel_spec[:, : num_tokens * self.frame_width]
        tokens = truncated.reshape(n_mels, num_tokens, self.frame_width).transpose(1, 0, 2)
        return tokens.reshape(num_tokens, n_mels * self.frame_width)


def numpy_to_wav_bytes(signal: np.ndarray, sr: int = 16000) -> bytes:
    """
    零外部依赖！纯 struct 序列化生成符合标准规范的 PCM 16-bit RIFF/WAVE 格式二进制字节流。
    可以直接传递给 Streamlit 的 `st.audio(bytes, format='audio/wav')`。
    """
    clipped = np.clip(signal, -1.0, 1.0)
    pcm_data = (clipped * 32767.0).astype(np.int16).tobytes()
    
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sr * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm_data)
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sr,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_data
