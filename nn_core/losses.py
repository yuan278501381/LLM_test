# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.losses - 损失函数模块

实现常用损失函数，每个损失函数实现 forward() 计算损失值和 backward() 计算梯度。
forward 时自动缓存预测值和真实值，供 backward 使用。

支持的损失函数:
    - MSE: 均方误差（回归）
    - BinaryCrossEntropy: 二元交叉熵（二分类）
    - CategoricalCrossEntropy: 分类交叉熵（多分类）
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from nn_core.tensor import safe_log

logger = logging.getLogger(__name__)


class Loss(ABC):
    """
    损失函数基类。

    所有损失函数必须实现 forward 和 backward 方法。
    forward 中缓存 y_pred 和 y_true 供 backward 使用。
    """

    def __init__(self) -> None:
        self.y_pred: np.ndarray | None = None  # 缓存模型预测值
        self.y_true: np.ndarray | None = None  # 缓存真实标签

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算损失值"""
        ...

    @abstractmethod
    def backward(self) -> np.ndarray:
        """计算损失函数对预测值的梯度 dL/dy_pred"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MSE(Loss):
    """
    均方误差 (Mean Squared Error) 损失函数。

    数学公式:
        forward:  L = (1/n) · Σ(y_pred - y_true)²
        backward: dL/dy_pred = 2 · (y_pred - y_true) / n

    适用场景: 回归任务
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算均方误差。

        Args:
            y_pred: 模型预测值，shape (n, d)
            y_true: 真实标签，shape (n, d)

        Returns:
            标量损失值
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        """
        计算 MSE 梯度: dL/dy_pred = 2(y_pred - y_true) / n
        """
        n = self.y_pred.size  # 总元素数 = 样本数 × 特征维度
        return 2.0 * (self.y_pred - self.y_true) / n


class BinaryCrossEntropy(Loss):
    """
    二元交叉熵损失函数。

    数学公式:
        forward:  L = -(1/n) · Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        backward: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n

    适用场景: 二分类任务（输出层使用 Sigmoid）
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算二元交叉熵损失。

        使用 safe_log 防止 log(0)。

        Args:
            y_pred: 模型输出概率，shape (n, 1)，范围 (0, 1)
            y_true: 真实标签，shape (n, 1)，值为 0 或 1

        Returns:
            标量损失值
        """
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(y_true * safe_log(y_pred) + (1.0 - y_true) * safe_log(1.0 - y_pred))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算 BCE 梯度: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n

        添加 epsilon 防止除以零。
        """
        n = self.y_pred.shape[0]
        eps = 1e-12
        denominator = self.y_pred * (1.0 - self.y_pred) + eps
        return (self.y_pred - self.y_true) / denominator / n


class CategoricalCrossEntropy(Loss):
    """
    分类交叉熵损失函数。

    数学公式:
        forward:  L = -(1/n) · Σ_i Σ_c [y_true_{i,c} · log(y_pred_{i,c})]
        backward: dL/dŷ = -y_true / (ŷ + ε) / n

    适用场景: 多分类任务（输出层使用 Softmax）

    注意: 当与 Softmax 配合使用时，联合梯度可简化为:
        dL/dz = (softmax_output - y_true) / n
        这个简化由 model.py 在训练循环中自动处理。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算分类交叉熵损失。

        Args:
            y_pred: 模型输出概率，shape (n, c)，每行和为 1
            y_true: one-hot 编码的真实标签，shape (n, c)

        Returns:
            标量损失值
        """
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(np.sum(y_true * safe_log(y_pred), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算 CCE 梯度: dL/dŷ = -y_true / (ŷ + ε) / n
        """
        n = self.y_pred.shape[0]
        eps = 1e-12
        return -self.y_true / (self.y_pred + eps) / n
