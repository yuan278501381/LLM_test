# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.layers - 网络层模块

实现神经网络的基本构建单元。每个层遵循统一的 forward/backward 协议，
支持链式组装和自动梯度计算。

支持的层:
    - Dense: 全连接层 (Z = X·W + b)
    - Dropout: 随机丢弃层 (正则化)
"""

import logging
import uuid

import numpy as np

from nn_core.initializers import he_init, random_init, xavier_init, zeros_init
from nn_core.regularizers import L1, L2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 初始化策略注册表 — 通过字符串名称查找初始化函数
# ---------------------------------------------------------------------------
_INITIALIZER_REGISTRY: dict[str, callable] = {
    "zeros": zeros_init,
    "random": random_init,
    "xavier": xavier_init,
    "he": he_init,
}


class Dense:
    """
    全连接层 (Fully Connected / Linear Layer)。

    数学公式:
        forward:  Z = X · W + b
        backward:
            dW = X^T · dZ           (权重梯度)
            db = Σ dZ (按行求和)     (偏置梯度)
            dX = dZ · W^T           (传给前一层的梯度)

    Args:
        n_inputs: 输入特征维度
        n_outputs: 输出特征维度（即本层神经元数量）
        initializer: 权重初始化策略名称，可选 'zeros'|'random'|'xavier'|'he'
        regularizer: 正则化器实例（L1 或 L2），可选

    Attributes:
        weights: 权重矩阵，shape (n_inputs, n_outputs)
        biases: 偏置向量，shape (1, n_outputs)
        grad_weights: 权重梯度，shape 同 weights
        grad_biases: 偏置梯度，shape 同 biases
        input_cache: 前向传播时缓存的输入
        output_cache: 前向传播时缓存的输出
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        initializer: str = "xavier",
        regularizer: L1 | L2 | None = None,
    ) -> None:
        # ---- 参数校验 ----
        if n_inputs <= 0 or n_outputs <= 0:
            raise ValueError(f"层维度必须为正整数: n_inputs={n_inputs}, n_outputs={n_outputs}")
        if initializer not in _INITIALIZER_REGISTRY:
            raise ValueError(
                f"未知的初始化策略 '{initializer}'，可选: {list(_INITIALIZER_REGISTRY.keys())}"
            )

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.regularizer = regularizer

        # ---- 初始化权重和偏置 ----
        init_fn = _INITIALIZER_REGISTRY[initializer]
        self.weights: np.ndarray = init_fn((n_inputs, n_outputs))
        self.biases: np.ndarray = np.zeros((1, n_outputs))

        # ---- 梯度缓存 ----
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_biases: np.ndarray = np.zeros_like(self.biases)

        # ---- 前向缓存 ----
        self.input_cache: np.ndarray | None = None
        self.output_cache: np.ndarray | None = None

        tid = uuid.uuid4().hex[:8]
        logger.info(
            "[%s] Dense 层已创建: (%d → %d), 初始化=%s, 正则化=%s, 参数量=%d",
            tid,
            n_inputs,
            n_outputs,
            initializer,
            regularizer if regularizer else "无",
            n_inputs * n_outputs + n_outputs,
        )

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        前向传播: Z = X · W + b

        Args:
            x: 输入数据，shape (batch_size, n_inputs)
            training: 是否为训练模式（Dense 层不区分，但保持接口一致）

        Returns:
            输出数据，shape (batch_size, n_outputs)
        """
        self.input_cache = x
        output = x @ self.weights + self.biases
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播 — 计算梯度并返回传给前一层的梯度。

        梯度计算:
            grad_weights = X^T · dout      (对权重的梯度)
            grad_biases  = sum(dout, dim=0) (对偏置的梯度)
            dx           = dout · W^T      (传给前一层)

        如果存在正则化器，将正则化梯度累加到 grad_weights 上。

        Args:
            dout: 来自后一层的梯度，shape (batch_size, n_outputs)

        Returns:
            传给前一层的梯度，shape (batch_size, n_inputs)
        """
        # 计算参数梯度
        self.grad_weights = self.input_cache.T @ dout
        self.grad_biases = np.sum(dout, axis=0, keepdims=True)

        # 叠加正则化梯度
        if self.regularizer is not None:
            self.grad_weights = self.grad_weights + self.regularizer.gradient(self.weights)

        # 计算并返回传给前一层的梯度
        return dout @ self.weights.T

    def __repr__(self) -> str:
        reg_str = f", reg={self.regularizer}" if self.regularizer else ""
        return f"Dense({self.n_inputs} → {self.n_outputs}{reg_str})"


class Dropout:
    """
    Dropout 正则化层 (Inverted Dropout)。

    训练时随机将一部分神经元的输出置零，并对剩余输出进行缩放 (÷ (1-rate))，
    使得推理时无需额外缩放（即 inverted dropout）。

    数学公式:
        训练时: output = x · mask / (1 - rate)
                其中 mask ~ Bernoulli(1 - rate)
        推理时: output = x（直接透传）

    Args:
        rate: 丢弃概率，范围 [0, 1)。0 表示不丢弃，0.5 表示随机丢弃一半。

    注意:
        - rate=0 等价于恒等变换
        - rate 不应等于 1（会导致除以零）
    """

    def __init__(self, rate: float = 0.5) -> None:
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate 必须在 [0, 1) 范围内，收到: {rate}")
        self.rate = rate
        self.mask: np.ndarray | None = None
        self.input_cache: np.ndarray | None = None
        self.output_cache: np.ndarray | None = None

        logger.debug("Dropout 层已创建: rate=%.2f", rate)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        前向传播。

        Args:
            x: 输入数据
            training: True 时执行随机丢弃，False 时直接透传

        Returns:
            输出数据（训练时部分神经元被置零并缩放）
        """
        self.input_cache = x

        if training and self.rate > 0:
            # 生成伯努利掩码: 1 表示保留，0 表示丢弃
            self.mask = np.random.binomial(1, 1.0 - self.rate, size=x.shape).astype(np.float64)
            # Inverted dropout: 缩放保留的值，使期望不变
            output = x * self.mask / (1.0 - self.rate)
        else:
            self.mask = None
            output = x

        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播。

        被丢弃的神经元梯度为 0，保留的神经元梯度按 1/(1-rate) 缩放。

        Args:
            dout: 来自后一层的梯度

        Returns:
            传给前一层的梯度
        """
        if self.mask is not None:
            return dout * self.mask / (1.0 - self.rate)
        return dout

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"
