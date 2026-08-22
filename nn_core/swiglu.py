# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.swiglu - 门控线性单元前馈网络模块 (SwiGLU FFN)

实现 2026 年现代主流大模型 (LLaMA-3, Gemma-2, DeepSeek-V3, Qwen-2.5) 标配的 SwiGLU 结构：
- SiLU (Swish-1) 平滑激活函数: SiLU(x) = x * sigmoid(x)
- 三矩阵门控架构: (SiLU(x @ W_gate + b_gate) ⊙ (x @ W_up + b_up)) @ W_down + b_down
- 与经典 2-Layer ReLU/GELU FFN 的对比分析
"""

import logging
import uuid

import numpy as np

from nn_core.tensor import safe_exp

logger = logging.getLogger(__name__)


def silu(x: np.ndarray) -> np.ndarray:
    r"""
    SiLU (Sigmoid Linear Unit / Swish-1) 激活函数。

    数学公式:
        $SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
    """
    sig = 1.0 / (1.0 + safe_exp(-x))
    return x * sig


class SwiGLU:
    r"""
    SwiGLU 门控前馈网络层 (Swish Gated Linear Unit)。

    相较于标准两层 MLP，SwiGLU 引入了 Gate 矩阵进行元素级门控相乘，
    具有更强的表达能力和更平滑的梯度流。

    数学公式:
        $h_{gate} = SiLU(x \cdot W_{gate} + b_{gate})$
        $h_{up}   = x \cdot W_{up} + b_{up}$
        $out      = (h_{gate} \odot h_{up}) \cdot W_{down} + b_{down}$
    """

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        self.d_model = d_model
        # 现代大模型通常将 SwiGLU 的 d_ff 设定为约 (8/3) * d_model，以保持参数量与标准 4x FFN 相当
        if d_ff is None:
            d_ff = int(2 * d_model * 4 / 3)
            # 向上取整到 8 的倍数以利于硬件对齐
            d_ff = ((d_ff + 7) // 8) * 8
        self.d_ff = d_ff

        # 初始化权重 (标准正态分布缩放)
        scale_in = np.sqrt(2.0 / d_model)
        scale_out = np.sqrt(2.0 / d_ff)

        self.W_gate = np.random.randn(d_model, d_ff) * scale_in
        self.b_gate = np.zeros(d_ff)

        self.W_up = np.random.randn(d_model, d_ff) * scale_in
        self.b_up = np.zeros(d_ff)

        self.W_down = np.random.randn(d_ff, d_model) * scale_out
        self.b_down = np.zeros(d_model)

        self.cache: dict[str, np.ndarray] = {}

        tid = uuid.uuid4().hex[:8]
        logger.info("[%s] SwiGLU 已创建: d_model=%d, d_ff=%d", tid, d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播。

        Args:
            x: 形状 (batch_size, seq_len, d_model) 或 (batch_size, d_model)
        """
        gate_proj = x @ self.W_gate + self.b_gate
        up_proj = x @ self.W_up + self.b_up

        gate_act = silu(gate_proj)
        gated_hidden = gate_act * up_proj  # 门控相乘

        out = gated_hidden @ self.W_down + self.b_down

        self.cache = {
            "x": x,
            "gate_proj": gate_proj,
            "gate_act": gate_act,
            "up_proj": up_proj,
            "gated_hidden": gated_hidden,
        }
        return out

    def get_params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回所有参数及其梯度"""
        grad_W_gate = np.zeros_like(self.W_gate)
        grad_b_gate = np.zeros_like(self.b_gate)
        grad_W_up = np.zeros_like(self.W_up)
        grad_b_up = np.zeros_like(self.b_up)
        grad_W_down = np.zeros_like(self.W_down)
        grad_b_down = np.zeros_like(self.b_down)
        return [
            (self.W_gate, grad_W_gate),
            (self.b_gate, grad_b_gate),
            (self.W_up, grad_W_up),
            (self.b_up, grad_b_up),
            (self.W_down, grad_W_down),
            (self.b_down, grad_b_down),
        ]
