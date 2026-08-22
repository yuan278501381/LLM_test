# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.lora - 低秩自适应微调 (Low-Rank Adaptation / LoRA) 纯 NumPy 计算层

包含：
- `LoRALayer`: 冻结主干权重并注入低秩分解旁路矩阵 $W_0 + \\frac{\\alpha}{r} A B$
- `compute_param_savings`: 评估 LoRA 微调在不同秩下的参数压缩比与显存节约率
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("nn_core.lora")


class LoRALayer:
    """
    LoRA 低秩矩阵分解旁路适配层。

    数学公式：
        $$h = x W_0 + \\Delta W x = x W_0 + \\frac{\\alpha}{r} (x A) B$$
        其中 $W_0 \\in \\mathbb{R}^{d_{in} \\times d_{out}}$ 保持完全冻结，
        $A \\in \\mathbb{R}^{d_{in} \\times r} \\sim \\mathcal{N}(0, \\sigma^2)$，
        $B \\in \\mathbb{R}^{r \\times d_{out}} = 0$ 确保初始状态输出 $h = x W_0$ 恒等无扰动。
    """

    def __init__(
        self,
        original_weight: np.ndarray,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        self.W = original_weight.copy()
        self.d_in, self.d_out = original_weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 旁路低秩矩阵初始化
        self.A = np.random.randn(self.d_in, rank) * 0.01
        self.B = np.zeros((rank, self.d_out), dtype=original_weight.dtype)

        # 梯度容器
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)

        self._x: Optional[np.ndarray] = None

    @property
    def num_trainable_params(self) -> int:
        """可训练低秩参数总数"""
        return self.A.size + self.B.size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播：主干冻结路径 + LoRA 低秩缩放旁路
        x shape: (B, ..., d_in)
        返回 shape: (B, ..., d_out)
        """
        self._x = x
        # 主干输出
        base_out = np.dot(x, self.W)
        # LoRA 旁路计算: (x @ A) @ B * scaling
        lora_out = np.dot(np.dot(x, self.A), self.B) * self.scaling
        return base_out + lora_out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播仅更新低秩矩阵 A 和 B 的梯度。
        """
        if self._x is None:
            raise RuntimeError("必须先执行 forward()")

        # 展平批次维度
        x_flat = self._x.reshape(-1, self.d_in)
        dout_flat = dout.reshape(-1, self.d_out)

        # grad_B: (rank, d_out) = (x @ A).T @ dout * scaling
        xA = np.dot(x_flat, self.A)
        self.grad_B = np.dot(xA.T, dout_flat) * self.scaling

        # grad_A: (d_in, rank) = x.T @ (dout @ B.T) * scaling
        dB = np.dot(dout_flat, self.B.T)
        self.grad_A = np.dot(x_flat.T, dB) * self.scaling

        # 输入梯度 dx
        dx_lora = np.dot(np.dot(dout_flat, self.B.T), self.A.T) * self.scaling
        dx_base = np.dot(dout_flat, self.W.T)
        dx = (dx_base + dx_lora).reshape(self._x.shape)
        return dx

    def merge(self) -> np.ndarray:
        """零推理延迟：将 LoRA 旁路权重直接永久融合进主干矩阵"""
        delta_w = np.dot(self.A, self.B) * self.scaling
        return self.W + delta_w

    def get_trainable_params(self) -> list[Tuple[np.ndarray, np.ndarray]]:
        """实现优化器参数协议"""
        return [
            (self.A, self.grad_A),
            (self.B, self.grad_B),
        ]


def compute_param_savings(d_model: int = 512, rank: int = 4) -> dict[str, float | int]:
    """
    计算线性投影层在指定秩下的 LoRA 参数量与压缩比。
    """
    orig_params = d_model * d_model
    lora_params = 2 * d_model * rank
    ratio = orig_params / lora_params
    saved_percent = (1.0 - lora_params / orig_params) * 100.0

    return {
        "d_model": d_model,
        "rank": rank,
        "original_params": orig_params,
        "lora_params": lora_params,
        "compression_ratio": ratio,
        "saved_percent": saved_percent,
    }
