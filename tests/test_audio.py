# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
tests/test_audio.py - 音频信号与语音感知模块单元测试
"""

import numpy as np

from nn_core.audio import (
    AudioTokenizer,
    SpectrogramFramePatcher,
    compute_mel_spectrogram,
    generate_chord,
    generate_waveform,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
    numpy_to_wav_bytes,
    stft,
)


def test_generate_waveform_shapes():
    """测试正弦波、方波与锯齿波生成形状"""
    sr = 16000
    duration = 0.5
    w_sine = generate_waveform(freq=440.0, duration=duration, sr=sr, wave_type="sine")
    assert len(w_sine) == int(sr * duration)
    assert np.max(np.abs(w_sine)) <= 1.0

    w_square = generate_waveform(freq=440.0, duration=duration, sr=sr, wave_type="square")
    assert len(w_square) == int(sr * duration)

    w_chord = generate_chord([261.6, 329.6, 392.0], duration=duration, sr=sr)
    assert len(w_chord) == int(sr * duration)


def test_stft_and_frequency_detection():
    """测试 STFT 形状与 440Hz 峰值频率位置"""
    sr = 16000
    freq = 440.0
    signal = generate_waveform(freq=freq, duration=1.0, sr=sr)

    n_fft = 512
    stft_mat = stft(signal, n_fft=n_fft, hop_length=256)
    assert stft_mat.shape[0] == n_fft // 2 + 1

    # 验证幅度谱峰值频率
    full_fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    peak_freq = freqs[np.argmax(full_fft)]
    assert np.abs(peak_freq - freq) < 2.0, f"FFT 频率检测偏差过大: {peak_freq} vs {freq}"


def test_mel_filterbank_properties():
    """测试梅尔滤波器组矩阵形状与有效性"""
    sr = 16000
    n_fft = 512
    n_mels = 80
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    assert fb.shape == (n_mels, n_fft // 2 + 1)
    # 确保没有全零行
    row_sums = np.sum(fb, axis=1)
    assert np.all(row_sums > 0.0)

    # 验证 hz <-> mel 双向转换一致性
    hz = np.array([100.0, 440.0, 1000.0, 4000.0])
    mel = hz_to_mel(hz)
    recovered_hz = mel_to_hz(mel)
    np.testing.assert_allclose(hz, recovered_hz, rtol=1e-5)


def test_mel_spectrogram_and_tokenizer():
    """测试梅尔频谱图与 Token 化器维度"""
    sr = 16000
    signal = generate_waveform(freq=440.0, duration=1.0, sr=sr)
    mel_spec = compute_mel_spectrogram(signal, sr=sr, n_fft=512, hop_length=256, n_mels=80)
    assert mel_spec.shape[0] == 80

    tokenizer = SpectrogramFramePatcher(n_mels=80, frame_width=4)
    tokens = tokenizer.tokenize(mel_spec)
    assert tokens.shape[1] == 80 * 4

    # 旧名称仅为兼容别名，行为保持一致。
    legacy_tokens = AudioTokenizer(n_mels=80, frame_width=4).tokenize(mel_spec)
    np.testing.assert_array_equal(tokens, legacy_tokens)
    assert tokens.shape[0] == mel_spec.shape[1] // 4


def test_numpy_to_wav_bytes():
    """测试纯 struct 生成的标准 WAV 文件头与字节流"""
    signal = np.random.randn(16000).astype(np.float32)
    wav_bytes = numpy_to_wav_bytes(signal, sr=16000)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) == 44 + 16000 * 2
    # 验证 RIFF 与 WAVE 标识
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "
    assert wav_bytes[36:40] == b"data"
