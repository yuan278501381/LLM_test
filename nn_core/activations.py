# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.activations - 激活函数模块

实现常用的非线性激活函数，每个激活函数都实现 forward() 和 backward() 方法，
遵循统一的接口协议。所有激活函数在 forward 时缓存必要数据，供 backward 反向传播使用。

支持的激活函数:
    - Sigmoid: σ(x) = 1 / (1 + e^(-x))
    - ReLU: f(x) = max(0, x)
    - Tanh: f(x) = tanh(x)
    - LeakyReLU: f(x) = x if x > 0 else α·x
    - Softmax: f(x_i) = e^(x_i) / Σe^(x_j)
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from nn_core.tensor import safe_exp

logger = logging.getLogger(__name__)


class Activation(ABC):
    """
    激活函数基类。

    所有激活函数必须实现 forward 和 backward 方法。
    forward 时自动缓存输入和输出，供 backward 和可视化使用。
    """

    def __init__(self) -> None:
        self.input_cache: np.ndarray | None = None  # 缓存 forward 输入
        self.output_cache: np.ndarray | None = None  # 缓存 forward 输出

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        ...

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Sigmoid(Activation):
    """
    Sigmoid 激活函数。

    数学公式:
        forward:  σ(x) = 1 / (1 + e^(-x))
        backward: dσ/dx = σ(x) · (1 - σ(x))

    特性:
        - 输出范围 (0, 1)，常用于二分类输出层
        - 梯度最大值为 0.25（在 x=0 处），容易导致梯度消失
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播: σ(x) = 1 / (1 + exp(-x))

        使用 safe_exp 防止数值溢出。
        """
        self.input_cache = x
        # 使用 safe_exp(-x) 避免大正数导致的溢出
        output = 1.0 / (1.0 + safe_exp(-x))
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播: dL/dx = dout · σ(x) · (1 - σ(x))

        直接从缓存的输出计算，避免重复计算 sigmoid。
        """
        s = self.output_cache
        if s is None:
            raise RuntimeError("Sigmoid.backward() 必须在 forward() 之后调用")
        return dout * s * (1.0 - s)


class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) 激活函数。

    数学公式:
        forward:  f(x) = max(0, x)
        backward: f'(x) = 1 if x > 0 else 0

    特性:
        - 计算高效，收敛速度快
        - 可能导致 "死亡神经元"（输入持续为负时梯度永远为 0）
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: f(x) = max(0, x)"""
        self.input_cache = x
        output = np.maximum(0, x)
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播: dL/dx = dout · (x > 0)"""
        x = self.input_cache
        if x is None:
            raise RuntimeError("ReLU.backward() 必须在 forward() 之后调用")
        return dout * (x > 0).astype(np.float64)


class Tanh(Activation):
    """
    Tanh (双曲正切) 激活函数。

    数学公式:
        forward:  f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        backward: f'(x) = 1 - tanh²(x)

    特性:
        - 输出范围 (-1, 1)，以零为中心
        - 比 Sigmoid 梯度更大，但仍可能梯度消失
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: f(x) = tanh(x)"""
        self.input_cache = x
        output = np.tanh(x)
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播: dL/dx = dout · (1 - tanh²(x))"""
        output = self.output_cache
        if output is None:
            raise RuntimeError("Tanh.backward() 必须在 forward() 之后调用")
        return dout * (1.0 - output**2)


class LeakyReLU(Activation):
    """
    LeakyReLU 激活函数。

    数学公式:
        forward:  f(x) = x if x > 0 else α·x
        backward: f'(x) = 1 if x > 0 else α

    Args:
        alpha: 负区间斜率，默认 0.01

    特性:
        - 解决 ReLU 的 "死亡神经元" 问题
        - 负区间仍有微小梯度，允许参数继续更新
    """

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: f(x) = x if x > 0 else α·x"""
        self.input_cache = x
        output = np.where(x > 0, x, self.alpha * x)
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播: dL/dx = dout · (1 if x > 0 else α)"""
        x = self.input_cache
        if x is None:
            raise RuntimeError("LeakyReLU.backward() 必须在 forward() 之后调用")
        return dout * np.where(x > 0, 1.0, self.alpha)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class Softmax(Activation):
    """
    Softmax 激活函数。

    数学公式:
        forward:  f(x_i) = e^(x_i - max(x)) / Σ e^(x_j - max(x))
        backward: 简化形式 dL/dx = dout · s - s · Σ(dout · s)
                  （其中 s 是 softmax 输出）

    特性:
        - 输出为概率分布（所有元素非负且求和为 1）
        - 减去 max(x) 保证数值稳定性
        - 常用于多分类输出层，与 CategoricalCrossEntropy 配合使用
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播: softmax(x)

        减去每行的最大值防止指数溢出（数值稳定技巧）。
        """
        self.input_cache = x
        # 减去每行最大值，防止 exp 溢出
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        self.output_cache = output
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播。

        使用简化形式:
            dL/dx_i = s_i · (dout_i - Σ_j(dout_j · s_j))
        等价于完整 Jacobian 计算，但更高效。
        """
        s = self.output_cache
        if s is None:
            raise RuntimeError("Softmax.backward() 必须在 forward() 之后调用")
        # 简化的向量化计算
        return s * (dout - np.sum(dout * s, axis=1, keepdims=True))
