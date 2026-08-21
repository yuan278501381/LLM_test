# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.optimizers - 优化器模块 (通用参数协议版)

实现梯度下降的多种变体。每个优化器通过 step() 方法就地更新网络层的可学习参数。
支持通用 `get_params_and_grads()` 协议，可无缝更新 Dense、LayerNorm、Embedding 等任意带参层。

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

        Args:
            layers: 模型中所有层的列表
        """
        ...

    def _extract_param_tuples(self, layers: list) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """
        从层列表中提取全部可学习参数及其梯度元组列表。
        返回: [(param_array, grad_array, param_unique_id), ...]
        """
        tuples = []
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, "get_params_and_grads"):
                for sub_idx, (p, g) in enumerate(layer.get_params_and_grads()):
                    uid = hash((layer_idx, sub_idx, id(p)))
                    tuples.append((p, g, uid))
            elif isinstance(layer, Dense):
                tuples.append((layer.weights, layer.grad_weights, hash((layer_idx, 0, id(layer.weights)))))
                tuples.append((layer.biases, layer.grad_biases, hash((layer_idx, 1, id(layer.biases)))))
        return tuples

    def _dense_layers(self, layers: list) -> list[Dense]:
        """向后兼容：从层列表中筛选出所有 Dense 层"""
        return [layer for layer in layers if isinstance(layer, Dense)]


class SGD(Optimizer):
    """
    随机梯度下降 (Stochastic Gradient Descent)。

    更新公式:
        θ ← θ - lr · ∇θ
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)
        logger.debug("SGD 优化器: lr=%.6f", learning_rate)

    def step(self, layers: list) -> None:
        for p, g, _ in self._extract_param_tuples(layers):
            p -= self.learning_rate * g

    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate})"


class Momentum(Optimizer):
    """
    带动量的 SGD。

    更新公式:
        v_θ ← β · v_θ + lr · ∇θ
        θ   ← θ - v_θ
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"动量系数 beta 必须在 [0, 1) 范围内，收到: {beta}")
        self.beta = beta
        self._velocity: dict[int, np.ndarray] = {}
        logger.debug("Momentum 优化器: lr=%.6f, beta=%.4f", learning_rate, beta)

    def _get_v(self, p: np.ndarray, uid: int) -> np.ndarray:
        if uid not in self._velocity:
            self._velocity[uid] = np.zeros_like(p)
        return self._velocity[uid]

    def step(self, layers: list) -> None:
        for p, g, uid in self._extract_param_tuples(layers):
            v = self._get_v(p, uid)
            v[:] = self.beta * v + self.learning_rate * g
            p -= v

    def __repr__(self) -> str:
        return f"Momentum(lr={self.learning_rate}, beta={self.beta})"


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation)。

    更新公式:
        s_θ ← β · s_θ + (1-β) · (∇θ)²
        θ   ← θ - lr · ∇θ / (√s_θ + ε)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self._cache: dict[int, np.ndarray] = {}
        logger.debug(
            "RMSProp 优化器: lr=%.6f, beta=%.4f, eps=%.1e",
            learning_rate,
            beta,
            epsilon,
        )

    def _get_s(self, p: np.ndarray, uid: int) -> np.ndarray:
        if uid not in self._cache:
            self._cache[uid] = np.zeros_like(p)
        return self._cache[uid]

    def step(self, layers: list) -> None:
        for p, g, uid in self._extract_param_tuples(layers):
            s = self._get_s(p, uid)
            s[:] = self.beta * s + (1.0 - self.beta) * (g ** 2)
            p -= self.learning_rate * g / (np.sqrt(s) + self.epsilon)

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.learning_rate}, beta={self.beta})"


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation)。

    更新公式:
        t   ← t + 1
        m_θ ← β₁ · m_θ + (1-β₁) · ∇θ
        v_θ ← β₂ · v_θ + (1-β₂) · (∇θ)²
        m̂_θ ← m_θ / (1 - β₁ᵗ)
        v̂_θ ← v_θ / (1 - β₂ᵗ)
        θ   ← θ - lr · m̂_θ / (√v̂_θ + ε)
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
        self._t: int = 0
        self._cache_m: dict[int, np.ndarray] = {}
        self._cache_v: dict[int, np.ndarray] = {}
        logger.debug(
            "Adam 优化器: lr=%.6f, beta1=%.4f, beta2=%.4f, eps=%.1e",
            learning_rate,
            beta1,
            beta2,
            epsilon,
        )

    def _get_mv(self, p: np.ndarray, uid: int) -> tuple[np.ndarray, np.ndarray]:
        if uid not in self._cache_m:
            self._cache_m[uid] = np.zeros_like(p)
            self._cache_v[uid] = np.zeros_like(p)
        return self._cache_m[uid], self._cache_v[uid]

    def step(self, layers: list) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        for p, g, uid in self._extract_param_tuples(layers):
            m, v = self._get_mv(p, uid)

            # 一阶矩
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            # 二阶矩
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

            m_hat = m / bc1
            v_hat = v / bc2

            p -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def __repr__(self) -> str:
        return f"Adam(lr={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2})"
