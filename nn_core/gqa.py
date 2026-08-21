# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.gqa - 分组查询注意力模块 (Grouped Query Attention)

实现 2026 年现代开源大模型 (LLaMA-3, Mistral, Qwen-2.5) 标配的 GQA 架构：
- 从 MHA (多头) -> GQA (分组查询) -> MQA (多查询) 的全谱系演进
- Query 头与共享 Key/Value 头的张量广播机制
- KV-Cache 吞吐开销与显存压缩比理论计算
"""

import logging
import uuid
import numpy as np

from nn_core.attention import scaled_dot_product_attention

logger = logging.getLogger(__name__)


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """
    将 KV 头沿头维度重复 n_rep 次，使之与 Query 头数对齐。
    
    Args:
        x: 形状 (batch_size, num_kv_heads, seq_len, head_dim)
        n_rep: 重复倍数 (num_heads // num_kv_heads)
        
    Returns:
        形状 (batch_size, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    # (batch, n_kv_heads, 1, seq_len, head_dim) -> (batch, n_kv_heads, n_rep, seq_len, head_dim)
    x_expanded = np.repeat(x[:, :, np.newaxis, :, :], n_rep, axis=2)
    # reshape 回 (batch, n_kv_heads * n_rep, seq_len, head_dim)
    return x_expanded.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention:
    """
    分组查询注意力层 (Grouped Query Attention)。
    
    当 num_kv_heads == num_heads 时，等价于标准 MHA (Multi-Head Attention)；
    当 1 < num_kv_heads < num_heads 时，为 GQA (Grouped Query Attention)；
    当 num_kv_heads == 1 时，等价于 MQA (Multi-Query Attention)。
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        # 权重矩阵
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        # K 和 V 维度大大缩小！仅为 (num_kv_heads * head_dim)
        self.kv_dim = num_kv_heads * self.head_dim
        self.W_k = np.random.randn(d_model, self.kv_dim) * scale
        self.W_v = np.random.randn(d_model, self.kv_dim) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] GQA 已创建: d_model=%d, q_heads=%d, kv_heads=%d (压缩比: %d×)",
            tid, d_model, num_heads, num_kv_heads, self.num_queries_per_kv
        )

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        前向传播。
        
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (seq_len, seq_len) 或可广播掩码
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影
        Q = x @ self.W_q  # (B, S, d_model)
        K = x @ self.W_k  # (B, S, kv_dim)
        V = x @ self.W_v  # (B, S, kv_dim)

        # 2. 拆分多头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. 将 KV 头广播重复至与 Q 头数相同
        K_rep = repeat_kv(K, self.num_queries_per_kv)  # (B, num_heads, S, head_dim)
        V_rep = repeat_kv(V, self.num_queries_per_kv)  # (B, num_heads, S, head_dim)

        if mask is not None and mask.ndim == 2:
            mask = mask[np.newaxis, np.newaxis, :, :]

        # 4. 点积注意力
        attn_out, attn_weights = scaled_dot_product_attention(Q, K_rep, V_rep, mask)

        # 5. 拼接合并回 (B, S, d_model)
        attn_concat = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # 6. 输出投影
        output = attn_concat @ self.W_o
        return output, attn_weights

    def get_kv_cache_savings(self) -> dict[str, float]:
        """
        返回相较于标准 MHA 的 KV-Cache 显存节省指标。
        """
        mha_kv_params = 2 * self.num_heads * self.head_dim
        gqa_kv_params = 2 * self.num_kv_heads * self.head_dim
        savings_ratio = float(self.num_heads / self.num_kv_heads)
        memory_reduction = 1.0 - (1.0 / savings_ratio)
        return {
            "compression_ratio": savings_ratio,
            "memory_saved_percent": memory_reduction * 100.0,
            "mha_kv_dim": float(mha_kv_params),
            "gqa_kv_dim": float(gqa_kv_params),
        }
