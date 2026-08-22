# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.attention - 注意力机制模块

实现缩放点积注意力和多头注意力机制。
"""

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


def causal_mask(seq_len: int) -> np.ndarray:
    """
    返回一个因果掩码，用于自回归任务（防止看到未来的信息）。
    下三角矩阵。
    """
    return np.tril(np.ones((seq_len, seq_len)))


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    缩放点积注意力。

    数学公式:
        $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

    Args:
        q: (..., seq_len_q, d_k)
        k: (..., seq_len_k, d_k)
        v: (..., seq_len_v, d_v)
        mask: 掩码矩阵 (..., seq_len_q, seq_len_k)
    """
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q、k、v 至少需要序列维和特征维")
    if q.shape[-1] == 0 or q.shape[-2] == 0 or k.shape[-2] == 0:
        raise ValueError("注意力不支持空序列或零特征维")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q 与 k 的特征维必须一致")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k 与 v 的序列长度必须一致")
    try:
        np.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    except ValueError as exc:
        raise ValueError("q、k、v 的批次/头维无法广播") from exc

    d_k = q.shape[-1]

    # scores: (..., seq_len_q, seq_len_k)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_k)

    if mask is not None:
        valid_mask = np.asarray(mask, dtype=bool)
        try:
            valid_mask = np.broadcast_to(valid_mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask 无法广播到注意力分数形状") from exc
        if np.any(~np.any(valid_mask, axis=-1)):
            raise ValueError("mask 不能产生完全被遮蔽的查询行")
        scores = np.where(valid_mask, scores, -np.inf)

    # softmax over the last dimension
    shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = weights @ v
    return output, weights


class MultiHeadAttention:
    """
    多头注意力层。

    将查询、键、值映射到多个低维空间，并行进行注意力计算后再拼接。
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model 必须被 num_heads 整除"
        self.d_k = d_model // num_heads

        # 权重初始化
        scale = np.sqrt(2.0 / d_model)
        self.W_q: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_k: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_v: np.ndarray = np.random.randn(d_model, d_model) * scale
        self.W_o: np.ndarray = np.random.randn(d_model, d_model) * scale

        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] MultiHeadAttention 已创建: d_model=%d, num_heads=%d", tid, d_model, num_heads
        )

    def forward(
        self, x: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        前向传播。

        Args:
            x: (batch_size, seq_len, d_model)
            mask: 可选的注意力掩码
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影 (batch_size, seq_len, d_model)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 拆分为多头 (batch_size, seq_len, num_heads, d_k)
        # 转置为 (batch_size, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        if mask is not None and mask.ndim == 2:
            # 扩展后可广播至 (batch_size, num_heads, seq_len, seq_len)
            mask = mask[np.newaxis, np.newaxis, :, :]

        # 注意力计算
        output_heads, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 拼接回 (batch_size, seq_len, d_model)
        # output_heads: (batch_size, num_heads, seq_len, d_k)
        output_concat = output_heads.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model
        )

        # 最终线性投影
        output = output_concat @ self.W_o

        return output, attn_weights

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播。
        为教学简化，省略复杂的多头梯度的推导过程。
        """
        raise NotImplementedError("教学用模块：多头注意力的反向传播省略")
