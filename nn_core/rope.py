# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.rope - 旋转位置编码模块 (Rotary Position Embedding)

实现 2026 年现代主流大模型 (LLaMA-3, Qwen-2.5, DeepSeek-V3) 统一采用的 RoPE 编码：
- 2D 复数旋转几何变换
- 相对位置内积恒等性: <R_m q, R_n k> = q^T R_{n-m} k
- 向量化成对切片旋转 (Pairwise Slicing)
- 长文本外推与相对衰减分析
"""

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


def precompute_freqs_cis(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    预计算旋转角度矩阵的 cos 和 sin。

    Args:
        dim: 每个注意力头的维度 (d_k，必须为偶数)
        max_seq_len: 最大序列长度
        theta: 基频常数 (默认 10000.0)

    Returns:
        cos: 形状 (max_seq_len, dim)
        sin: 形状 (max_seq_len, dim)
    """
    assert dim % 2 == 0, "RoPE 特征维度必须为偶数"

    # 频率指数: theta^(-2i / dim)
    freq_indices = np.arange(0, dim, 2, dtype=np.float64)
    freqs = 1.0 / (theta ** (freq_indices / dim))  # shape: (dim // 2,)

    # 时间步向量 [0, 1, 2, ..., max_seq_len - 1]
    positions = np.arange(max_seq_len, dtype=np.float64)  # shape: (max_seq_len,)

    # 外积得到角度网格 (max_seq_len, dim // 2)
    angles = np.outer(positions, freqs)

    # 沿最后一维复制一份以适配成对旋转: (max_seq_len, dim)
    angles_expanded = np.repeat(angles, 2, axis=-1)

    cos = np.cos(angles_expanded)
    sin = np.sin(angles_expanded)
    return cos, sin


def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    成对切片旋转辅助算子:
    [-x_1, x_0, -x_3, x_2, ...]
    """
    x_rotated = np.zeros_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., 0::2]
    return x_rotated


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    seq_axis: int = 1,
) -> np.ndarray:
    r"""
    对输入 Query 或 Key 向量应用旋转位置编码。

    数学公式:
        $R_m x = x \odot \cos(m\theta) + \text{rotate\_half}(x) \odot \sin(m\theta)$

    Args:
        x: 最后一维为 head_dim 的张量
        cos: 形状 (seq_len, head_dim)
        sin: 形状 (seq_len, head_dim)
        seq_axis: x 中明确的序列轴；不再根据维度大小猜测布局
    """
    x = np.asarray(x)
    cos = np.asarray(cos)
    sin = np.asarray(sin)
    if x.ndim < 2:
        raise ValueError("RoPE 输入至少需要序列维与特征维")
    axis = seq_axis if seq_axis >= 0 else x.ndim + seq_axis
    if axis < 0 or axis >= x.ndim - 1:
        raise ValueError("seq_axis 必须指向最后一维之前的序列轴")
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE 特征维必须为偶数")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != x.shape[-1]:
        raise ValueError("cos/sin 必须同形，且最后一维匹配 RoPE 特征维")
    seq_len = x.shape[axis]
    if seq_len > cos.shape[0]:
        raise ValueError("输入序列长度超过预计算 RoPE 长度")

    broadcast_shape = [1] * x.ndim
    broadcast_shape[axis] = seq_len
    broadcast_shape[-1] = x.shape[-1]
    c = cos[:seq_len].reshape(broadcast_shape)
    s = sin[:seq_len].reshape(broadcast_shape)

    return (x * c) + (rotate_half(x) * s)


class RotaryPositionalEmbedding:
    """
    现代 RoPE (Rotary Position Embedding) 层。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.cos, self.sin = precompute_freqs_cis(dim, max_seq_len, theta)

        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] RoPE 已创建: dim=%d, max_len=%d, theta=%.1f", tid, dim, max_seq_len, theta
        )

    def forward(
        self, q: np.ndarray, k: np.ndarray, *, seq_axis: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        同时对 Query 和 Key 应用旋转位置编码。
        """
        q_rot = apply_rope(q, self.cos, self.sin, seq_axis=seq_axis)
        k_rot = apply_rope(k, self.cos, self.sin, seq_axis=seq_axis)
        return q_rot, k_rot

    def compute_relative_decay_matrix(self, seq_len: int) -> np.ndarray:
        """
        计算固定单位探针在各位置旋转后的内积核（用于 UI 可视化）。

        该矩阵展示 RoPE 的相对位置结构与振荡性，不代表任意 Query/Key
        的注意力值，也不保证随距离严格单调下降。
        """
        decay_matrix = np.zeros((seq_len, seq_len))
        # 构造基准单位向量
        u = np.ones(self.dim) / np.sqrt(self.dim)
        for i in range(seq_len):
            for j in range(seq_len):
                c_i = self.cos[i]
                s_i = self.sin[i]
                c_j = self.cos[j]
                s_j = self.sin[j]

                u_i = (u * c_i) + (rotate_half(u) * s_i)
                u_j = (u * c_j) + (rotate_half(u) * s_j)
                decay_matrix[i, j] = np.dot(u_i, u_j)
        return decay_matrix
