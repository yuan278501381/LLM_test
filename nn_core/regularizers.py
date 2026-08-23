# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.regularizers - 正则化模块

正则化通过在损失函数中添加惩罚项来约束模型复杂度，防止过拟合。

支持的正则化方式:
    - L1: Lasso 正则化（产生稀疏解）
    - L2: Ridge 正则化（权重衰减）
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class L1:
    """
    L1 (Lasso) 正则化。

    数学公式:
        惩罚项: λ · Σ|W|
        梯度:   λ · sign(W)

    特性:
        - 倾向于产生稀疏解（许多权重变为 0）
        - 可用于特征选择
    """

    def __init__(self, lambda_: float = 0.01) -> None:
        """
        Args:
            lambda_: 正则化强度，越大则惩罚越强
        """
        self.lambda_ = lambda_
        logger.debug("L1 正则化: lambda=%.4f", lambda_)

    def loss(self, weights: np.ndarray) -> float:
        """
        计算 L1 正则化惩罚项: λ · Σ|W|

        Args:
            weights: 权重矩阵

        Returns:
            正则化损失值（标量）
        """
        return self.lambda_ * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算 L1 正则化梯度: λ · sign(W)

        注意: sign(0) = 0，这是一个次梯度。

        Args:
            weights: 权重矩阵

        Returns:
            正则化梯度，与 weights 同形状
        """
        return self.lambda_ * np.sign(weights)

    def __repr__(self) -> str:
        return f"L1(lambda={self.lambda_})"


class L2:
    """
    L2 (Ridge / 权重衰减) 正则化。

    数学公式:
        惩罚项: 0.5 · λ · Σ(W²)
        梯度:   λ · W

    特性:
        - 使权重趋向于小值但不为零
        - 在标准 SGD（无动量）下等价于权重衰减：w <- (1 - lr·λ)·w - lr·grad；但在自适应学习率优化器（如 Adam）中并不等价，需使用解耦权重衰减（AdamW）
        - 在概率视角下等价于给权重施加高斯先验（Gaussian Prior / MAP 估计）
    """

    def __init__(self, lambda_: float = 0.01) -> None:
        """
        Args:
            lambda_: 正则化强度
        """
        self.lambda_ = lambda_
        logger.debug("L2 正则化: lambda=%.4f", lambda_)

    def loss(self, weights: np.ndarray) -> float:
        """
        计算 L2 正则化惩罚项: 0.5 · λ · Σ(W²)

        系数 0.5 使梯度更简洁: d/dW[0.5·λ·W²] = λ·W

        Args:
            weights: 权重矩阵

        Returns:
            正则化损失值（标量）
        """
        return 0.5 * self.lambda_ * np.sum(weights**2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算 L2 正则化梯度: λ · W

        Args:
            weights: 权重矩阵

        Returns:
            正则化梯度，与 weights 同形状
        """
        return self.lambda_ * weights

    def __repr__(self) -> str:
        return f"L2(lambda={self.lambda_})"
