# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.transformer - Transformer 模块

实现基于 Transformer 的前馈网络和解码器块。
"""

import logging
import uuid
import numpy as np

from nn_core.attention import MultiHeadAttention
from nn_core.layernorm import LayerNorm
from nn_core.gelu import GELU

logger = logging.getLogger(__name__)


class FeedForward:
    """
    两层感知机。
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.W1: np.ndarray = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1: np.ndarray = np.zeros(d_ff)
        self.W2: np.ndarray = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2: np.ndarray = np.zeros(d_model)
        
        self.gelu = GELU()
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] FeedForward 已创建: d_model=%d, d_ff=%d",
            tid, d_model, d_ff
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播: $GELU(x W_1 + b_1) W_2 + b_2$
        """
        h = x @ self.W1 + self.b1
        h_act = self.gelu.forward(h)
        out = h_act @ self.W2 + self.b2
        return out


class TransformerBlock:
    """
    Pre-LN Decoder Block (解码器块，先归一化)。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] TransformerBlock 已创建: d_model=%d, num_heads=%d, d_ff=%d",
            tid, d_model, num_heads, d_ff
        )

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        前向传播 (Pre-LN):
        1. $x = x + MHA(LN_1(x))$
        2. $x = x + FFN(LN_2(x))$
        """
        # 1. Multi-Head Attention 子层
        norm1 = self.ln1.forward(x)
        attn_out, attn_weights = self.mha.forward(norm1, mask=mask)
        x = x + attn_out
        
        # 2. Feed-Forward 子层
        norm2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(norm2)
        x = x + ffn_out
        
        return x, attn_weights
