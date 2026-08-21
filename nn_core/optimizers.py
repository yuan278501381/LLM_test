# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.optimizers - 优化器模块

实现梯度下降的多种变体。每个优化器通过 step() 方法就地更新网络层的参数。

支持的优化器:
    - SGD: 随机梯度下降
    - Momentum: 带动量的 SGD
    - RMSProp: 自适应学习率
    - Adam: Momentum + RMSProp + 偏差修正
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from nn_core.layers import Dense

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """优化器基类。所有优化器实现 step() 方法来更新参数。"""

    def __init__(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError(f"学习率必须为正数，收到: {learning_rate}")
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers: list) -> None:
        """
        执行一步参数更新。

        遍历所有 Dense 层，根据其 grad_weights 和 grad_biases 更新参数。

        Args:
            layers: 模型中所有层的列表（自动跳过非 Dense 层）
        """
        ...

    def _dense_layers(self, layers: list) -> list[Dense]:
        """从层列表中筛选出所有 Dense 层"""
        return [layer for layer in layers if isinstance(layer, Dense)]


class SGD(Optimizer):
    """
    随机梯度下降 (Stochastic Gradient Descent)。

    最朴素的优化器，每一步沿梯度反方向更新参数。

    更新公式:
        W ← W - lr · ∇W
        b ← b - lr · ∇b

    Args:
        learning_rate: 学习率，默认 0.01

    特性:
        - 简单直观，易于理解
        - 可能在鞍点或平坦区域收敛缓慢
        - 对学习率非常敏感
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)
        logger.debug("SGD 优化器: lr=%.6f", learning_rate)

    def step(self, layers: list) -> None:
        for layer in self._dense_layers(layers):
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.biases -= self.learning_rate * layer.grad_biases

    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate})"


class Momentum(Optimizer):
    """
    带动量的 SGD。

    引入「速度」变量累积历史梯度的指数移动平均，加速在一致方向上的更新，
    抑制在震荡方向上的更新。

    更新公式:
        v_W ← β · v_W + lr · ∇W
        W   ← W - v_W

    Args:
        learning_rate: 学习率，默认 0.01
        beta: 动量系数，默认 0.9（表示保留 90% 的历史速度）

    特性:
        - 加速收敛，尤其在「峡谷」型损失曲面上
        - 能冲过浅层局部最小值
        - β=0 退化为 vanilla SGD
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"动量系数 beta 必须在 [0, 1) 范围内，收到: {beta}")
        self.beta = beta
        # 每个 Dense 层的速度缓存，以 id(layer) 为 key
        self._velocity: dict[int, dict[str, np.ndarray]] = {}
        logger.debug("Momentum 优化器: lr=%.6f, beta=%.4f", learning_rate, beta)

    def _get_velocity(self, layer: Dense) -> dict[str, np.ndarray]:
        """获取或初始化某层的速度缓存"""
        lid = id(layer)
        if lid not in self._velocity:
            self._velocity[lid] = {
                "w": np.zeros_like(layer.weights),
                "b": np.zeros_like(layer.biases),
            }
        return self._velocity[lid]

    def step(self, layers: list) -> None:
        for layer in self._dense_layers(layers):
            v = self._get_velocity(layer)
            # 更新速度
            v["w"] = self.beta * v["w"] + self.learning_rate * layer.grad_weights
            v["b"] = self.beta * v["b"] + self.learning_rate * layer.grad_biases
            # 更新参数
            layer.weights -= v["w"]
            layer.biases -= v["b"]

    def __repr__(self) -> str:
        return f"Momentum(lr={self.learning_rate}, beta={self.beta})"


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation)。

    使用梯度平方的指数移动平均来自适应调整每个参数的学习率。
    对梯度大的参数降低学习率，对梯度小的参数提升学习率。

    更新公式:
        s_W ← β · s_W + (1-β) · (∇W)²
        W   ← W - lr · ∇W / (√s_W + ε)

    Args:
        learning_rate: 学习率，默认 0.001
        beta: 衰减率，默认 0.999
        epsilon: 防除零小常数，默认 1e-8

    特性:
        - 自适应学习率，每个参数有独立的有效学习率
        - 适合处理稀疏梯度和非平稳目标
        - 由 Hinton 在 Coursera 课程中提出（未正式发表论文）
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        logger.debug(
            "RMSProp 优化器: lr=%.6f, beta=%.4f, eps=%.1e",
            learning_rate,
            beta,
            epsilon,
        )

    def _get_cache(self, layer: Dense) -> dict[str, np.ndarray]:
        """获取或初始化某层的梯度平方缓存"""
        lid = id(layer)
        if lid not in self._cache:
            self._cache[lid] = {
                "s_w": np.zeros_like(layer.weights),
                "s_b": np.zeros_like(layer.biases),
            }
        return self._cache[lid]

    def step(self, layers: list) -> None:
        for layer in self._dense_layers(layers):
            c = self._get_cache(layer)
            # 更新梯度平方的移动平均
            c["s_w"] = self.beta * c["s_w"] + (1.0 - self.beta) * layer.grad_weights**2
            c["s_b"] = self.beta * c["s_b"] + (1.0 - self.beta) * layer.grad_biases**2
            # 自适应学习率更新参数
            layer.weights -= (
                self.learning_rate * layer.grad_weights / (np.sqrt(c["s_w"]) + self.epsilon)
            )
            layer.biases -= (
                self.learning_rate * layer.grad_biases / (np.sqrt(c["s_b"]) + self.epsilon)
            )

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.learning_rate}, beta={self.beta})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation)。

    结合 Momentum（一阶矩估计）和 RMSProp（二阶矩估计），
    并加入偏差修正以消除初始化偏差。当前最主流的优化器。

    更新公式:
        t   ← t + 1                                    (时间步)
        m_W ← β₁ · m_W + (1-β₁) · ∇W                  (一阶矩 / 动量)
        v_W ← β₂ · v_W + (1-β₂) · (∇W)²               (二阶矩 / 自适应)
        m̂_W ← m_W / (1 - β₁ᵗ)                          (偏差修正)
        v̂_W ← v_W / (1 - β₂ᵗ)                          (偏差修正)
        W   ← W - lr · m̂_W / (√v̂_W + ε)               (参数更新)

    Args:
        learning_rate: 学习率，默认 0.001
        beta1: 一阶矩衰减率，默认 0.9
        beta2: 二阶矩衰减率，默认 0.999
        epsilon: 防除零小常数，默认 1e-8

    特性:
        - 综合了 Momentum 和 RMSProp 的优点
        - 偏差修正确保初始更新步长合理
        - 几乎是「开箱即用」的最佳选择
        - 论文: Kingma & Ba, 2014
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t: int = 0  # 全局时间步
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        logger.debug(
            "Adam 优化器: lr=%.6f, beta1=%.4f, beta2=%.4f, eps=%.1e",
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )

    def _get_cache(self, layer: Dense) -> dict[str, np.ndarray]:
        """获取或初始化某层的一阶矩和二阶矩缓存"""
        lid = id(layer)
        if lid not in self._cache:
            self._cache[lid] = {
                "m_w": np.zeros_like(layer.weights),  # 一阶矩 (权重)
                "m_b": np.zeros_like(layer.biases),  # 一阶矩 (偏置)
                "v_w": np.zeros_like(layer.weights),  # 二阶矩 (权重)
                "v_b": np.zeros_like(layer.biases),  # 二阶矩 (偏置)
            }
        return self._cache[lid]

    def step(self, layers: list) -> None:
        self._t += 1  # 递增时间步

        for layer in self._dense_layers(layers):
            c = self._get_cache(layer)

            # ---- 更新一阶矩（动量）----
            c["m_w"] = self.beta1 * c["m_w"] + (1.0 - self.beta1) * layer.grad_weights
            c["m_b"] = self.beta1 * c["m_b"] + (1.0 - self.beta1) * layer.grad_biases

            # ---- 更新二阶矩（自适应学习率）----
            c["v_w"] = self.beta2 * c["v_w"] + (1.0 - self.beta2) * layer.grad_weights**2
            c["v_b"] = self.beta2 * c["v_b"] + (1.0 - self.beta2) * layer.grad_biases**2

            # ---- 偏差修正 ----
            # 初始时 m 和 v 偏向零，修正消除这个偏差
            bc1 = 1.0 - self.beta1**self._t
            bc2 = 1.0 - self.beta2**self._t
            m_w_hat = c["m_w"] / bc1
            m_b_hat = c["m_b"] / bc1
            v_w_hat = c["v_w"] / bc2
            v_b_hat = c["v_b"] / bc2

            # ---- 更新参数 ----
            layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            layer.biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def __repr__(self) -> str:
        return f"Adam(lr={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2})"
