# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
datasets.generators - 合成数据集生成模块

提供多种 2D 可视化友好的合成数据集，每种数据集都具有不同的决策边界复杂度，
用于直观演示神经网络的分类能力。

数据集按复杂度排序:
    1. make_blobs: 线性可分（多分类）
    2. make_moons: 非线性但简单（半月形）
    3. make_circles: 需要径向基式的边界（同心圆）
    4. make_xor: 经典的异或问题
    5. make_spiral: 高度非线性（双螺旋，最难）
"""

import logging

import numpy as np
from sklearn.datasets import (
    make_blobs as _sk_blobs,
)
from sklearn.datasets import (
    make_circles as _sk_circles,
)
from sklearn.datasets import (
    make_moons as _sk_moons,
)
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def _normalize(X: np.ndarray, feature_range: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """将特征归一化到指定范围"""
    scaler = MinMaxScaler(feature_range=feature_range)
    return np.asarray(scaler.fit_transform(X), dtype=np.float64)


def make_moons(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    半月形数据集 — 两个交错的半月形。

    决策边界: 非线性但相对简单，单隐藏层即可拟合。

    Args:
        n_samples: 样本总数
        noise: 高斯噪声标准差
        random_state: 随机种子

    Returns:
        (X, y): X shape (n, 2), y shape (n, 1)
    """
    raw_X, raw_y = _sk_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X = np.asarray(raw_X, dtype=np.float64)
    y = np.asarray(raw_y)
    X = _normalize(X)
    y = y.reshape(-1, 1).astype(np.float64)
    logger.debug("make_moons: n=%d, noise=%.2f → X%s, y%s", n_samples, noise, X.shape, y.shape)
    return X, y


def make_circles(
    n_samples: int = 200,
    noise: float = 0.05,
    random_state: int = 42,
    factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    同心圆数据集 — 内环和外环。

    决策边界: 需要圆形/径向分割，线性模型完全无法解决。

    Args:
        n_samples: 样本总数
        noise: 高斯噪声标准差
        random_state: 随机种子
        factor: 内圆与外圆的半径比

    Returns:
        (X, y): X shape (n, 2), y shape (n, 1)
    """
    raw_X, raw_y = _sk_circles(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
        factor=factor,
    )
    X = np.asarray(raw_X, dtype=np.float64)
    y = np.asarray(raw_y)
    X = _normalize(X)
    y = y.reshape(-1, 1).astype(np.float64)
    logger.debug("make_circles: n=%d, noise=%.2f → X%s, y%s", n_samples, noise, X.shape, y.shape)
    return X, y


def make_xor(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    XOR (异或) 数据集 — 对角象限为同一类。

    经典的非线性可分问题:
        - 第 1、3 象限: 类别 1
        - 第 2、4 象限: 类别 0

    这是历史上证明感知器局限性的关键例子。

    Args:
        n_samples: 样本总数
        noise: 高斯噪声标准差
        random_state: 随机种子

    Returns:
        (X, y): X shape (n, 2), y shape (n, 1)
    """
    rng = np.random.RandomState(random_state)
    n_per_quadrant = n_samples // 4

    centers = [
        (0.5, 0.5),  # 第 1 象限 → 类别 1
        (-0.5, 0.5),  # 第 2 象限 → 类别 0
        (-0.5, -0.5),  # 第 3 象限 → 类别 1
        (0.5, -0.5),  # 第 4 象限 → 类别 0
    ]
    labels = [1, 0, 1, 0]

    X_parts = []
    y_parts = []

    for (cx, cy), label in zip(centers, labels, strict=True):
        # 确保最后一个象限包含剩余样本
        n = n_samples - sum(len(p) for p in X_parts) if len(X_parts) == 3 else n_per_quadrant

        x_part = rng.randn(n, 2) * noise + np.array([cx, cy])
        X_parts.append(x_part)
        y_parts.append(np.full((n, 1), label, dtype=np.float64))

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)

    # 打乱数据
    shuffle_idx = rng.permutation(len(X))
    X = _normalize(X[shuffle_idx])
    y = y[shuffle_idx]

    logger.debug("make_xor: n=%d, noise=%.2f → X%s, y%s", n_samples, noise, X.shape, y.shape)
    return X, y


def make_spiral(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    双螺旋数据集 — 两条交织的螺旋臂。

    这是最具挑战性的 2D 分类任务之一，需要较深的网络才能拟合。

    Args:
        n_samples: 样本总数
        noise: 高斯噪声标准差
        random_state: 随机种子

    Returns:
        (X, y): X shape (n, 2), y shape (n, 1)
    """
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    # 螺旋臂 1
    theta1 = np.linspace(0, 3 * np.pi, n_per_class)
    r1 = theta1 / (3 * np.pi)
    x1 = r1 * np.cos(theta1) + rng.randn(n_per_class) * noise * 0.1
    y1_coord = r1 * np.sin(theta1) + rng.randn(n_per_class) * noise * 0.1

    # 螺旋臂 2（旋转 π 弧度）
    n_class2 = n_samples - n_per_class
    theta2 = np.linspace(0, 3 * np.pi, n_class2)
    r2 = theta2 / (3 * np.pi)
    x2 = r2 * np.cos(theta2 + np.pi) + rng.randn(n_class2) * noise * 0.1
    y2_coord = r2 * np.sin(theta2 + np.pi) + rng.randn(n_class2) * noise * 0.1

    X = np.vstack(
        [
            np.column_stack([x1, y1_coord]),
            np.column_stack([x2, y2_coord]),
        ]
    )
    y = np.vstack(
        [
            np.zeros((n_per_class, 1)),
            np.ones((n_class2, 1)),
        ]
    )

    # 打乱数据
    shuffle_idx = rng.permutation(len(X))
    X = _normalize(X[shuffle_idx])
    y = y[shuffle_idx]

    logger.debug("make_spiral: n=%d, noise=%.2f → X%s, y%s", n_samples, noise, X.shape, y.shape)
    return X, y


def make_blobs(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
    n_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    高斯簇数据集 — 多个高斯分布的聚类。

    适用于多分类任务，标签自动转换为 one-hot 编码。

    Args:
        n_samples: 样本总数
        noise: 簇的标准差
        random_state: 随机种子
        n_classes: 类别数量，默认 3

    Returns:
        (X, y): X shape (n, 2), y shape (n, n_classes) (one-hot)
    """
    raw_result = _sk_blobs(
        n_samples=n_samples,
        centers=n_classes,
        cluster_std=noise + 0.5,  # 基线标准差 + 用户噪声
        random_state=random_state,
        n_features=2,
        return_centers=False,
    )
    raw_X, raw_y_int = raw_result[0], raw_result[1]
    X = np.asarray(raw_X, dtype=np.float64)
    y_int = np.asarray(raw_y_int, dtype=np.int64)
    X = _normalize(X)

    # 转换为 one-hot 编码
    y_onehot = np.zeros((len(y_int), n_classes), dtype=np.float64)
    y_onehot[np.arange(len(y_int)), y_int] = 1.0

    logger.debug(
        "make_blobs: n=%d, classes=%d, noise=%.2f → X%s, y%s",
        n_samples,
        n_classes,
        noise,
        X.shape,
        y_onehot.shape,
    )
    return X, y_onehot
