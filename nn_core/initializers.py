# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.initializers - 权重初始化模块

提供多种权重初始化策略。好的初始化能加速收敛、避免梯度消失/爆炸。

支持的初始化方式:
    - zeros_init: 全零初始化（故意的坏选择，用于教学演示）
    - random_init: 小随机数初始化
    - xavier_init: Xavier/Glorot 初始化（适配 Sigmoid/Tanh）
    - he_init: He 初始化（适配 ReLU）
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def zeros_init(shape: tuple[int, int]) -> np.ndarray:
    """
    全零初始化。

    ️ 这是一个故意的「坏选择」——所有神经元的权重相同，
    导致对称性问题：所有神经元计算完全相同的梯度，永远无法学到不同的特征。

    用于教学：让学习者亲眼观察「为什么不能用零初始化」。

    Args:
        shape: (n_inputs, n_outputs)

    Returns:
        全零权重矩阵
    """
    logger.debug("零初始化: shape=%s（注意: 这会导致对称性问题）", shape)
    return np.zeros(shape)


def random_init(shape: tuple[int, int], scale: float = 0.01) -> np.ndarray:
    """
    小随机数初始化。

    从标准正态分布采样并乘以 scale 因子。

    Args:
        shape: (n_inputs, n_outputs)
        scale: 缩放因子，默认 0.01

    Returns:
        随机权重矩阵
    """
    logger.debug("随机初始化: shape=%s, scale=%.4f", shape, scale)
    return np.random.randn(*shape) * scale


def xavier_init(shape: tuple[int, int]) -> np.ndarray:
    """
    Xavier (Glorot) 初始化。

    数学公式:
        W ~ N(0, σ²)  其中 σ = √(2 / (n_in + n_out))

    设计目标: 保持前向和反向传播中信号的方差不变。
    适配: Sigmoid、Tanh 等饱和型激活函数。

    Args:
        shape: (n_inputs, n_outputs)

    Returns:
        Xavier 初始化的权重矩阵
    """
    n_in, n_out = shape
    std = np.sqrt(2.0 / (n_in + n_out))
    logger.debug("Xavier 初始化: shape=%s, std=%.4f", shape, std)
    return np.random.randn(*shape) * std


def he_init(shape: tuple[int, int]) -> np.ndarray:
    """
    He (Kaiming) 初始化。

    数学公式:
        W ~ N(0, σ²)  其中 σ = √(2 / n_in)

    设计目标: 专为 ReLU 系激活函数设计，补偿 ReLU 将一半神经元置零的效应。
    适配: ReLU、LeakyReLU 等非饱和型激活函数。

    Args:
        shape: (n_inputs, n_outputs)

    Returns:
        He 初始化的权重矩阵
    """
    n_in = shape[0]
    std = np.sqrt(2.0 / n_in)
    logger.debug("He 初始化: shape=%s, std=%.4f", shape, std)
    return np.random.randn(*shape) * std
