# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.layernorm - 层归一化模块

实现 Layer Normalization (层归一化)。
"""

import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class LayerNorm:
    r"""
    层归一化层。
    
    数学公式:
        $\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$
        $\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2$
        $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
        $y = \gamma \hat{x} + \beta$
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        
        self.gamma: np.ndarray = np.ones(d_model)
        self.beta: np.ndarray = np.zeros(d_model)
        
        self.grad_gamma: np.ndarray = np.zeros_like(self.gamma)
        self.grad_beta: np.ndarray = np.zeros_like(self.beta)
        
        # 缓存
        self.cache: dict[str, np.ndarray] = {}
        
        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] LayerNorm 已创建: d_model=%d, eps=%e",
            tid, d_model, eps
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # x.shape = (batch_size, seq_len, d_model) 或 (batch_size, d_model)
        # 在最后一个维度上归一化
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        
        self.cache = {
            "x": x,
            "x_norm": x_norm,
            "mean": mean,
            "var": var,
        }
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播: 完整的归一化全微分链推导。
        """
        x_norm = self.cache["x_norm"]
        var = self.cache["var"]
        x = self.cache["x"]
        mean = self.cache["mean"]
        
        # dout shape = (..., d_model)
        # 保持与 gamma、beta 一致的维度进行求和
        sum_axis = tuple(range(dout.ndim - 1))
        self.grad_gamma = np.sum(dout * x_norm, axis=sum_axis)
        self.grad_beta = np.sum(dout, axis=sum_axis)
        
        D = self.d_model
        
        dx_norm = dout * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * (std_inv ** 3), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
        
        dx = dx_norm * std_inv + dvar * 2.0 * (x - mean) / D + dmean / D
        return dx

    def get_params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回所有可学习参数及其梯度的 (param, grad) 列表"""
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]
