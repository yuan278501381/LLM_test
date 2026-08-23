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
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("MSE.backward() 必须在 forward() 之后调用")
        n = y_pred.size  # 总元素数 = 样本数 × 特征维度
        return 2.0 * (y_pred - y_true) / n


class BinaryCrossEntropy(Loss):
    """
    二元交叉熵损失函数（概率输入版本）。

    数学公式:
        forward:  L = -(1/n) · Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        backward: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n

    输入契约:
        要求输入 ŷ 为严格概率范围 [0, 1]（例如经过 Sigmoid 激活）。
        若输入超出 [0, 1] 定义域，将显式抛出 ValueError，杜绝掩盖契约错误。
        若使用未归一化的全实数 Logits，请使用数值更稳定的 BCEWithLogitsLoss。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算二元交叉熵损失。

        Args:
            y_pred: 模型输出概率，shape (n, 1) 或 (n,)，严格要求值域在 [0, 1]
            y_true: 真实标签，shape 与 y_pred 一致，取值为 0 或 1

        Returns:
            标量损失值 (>= 0)

        Raises:
            ValueError: 当 y_pred 超出 [0, 1] 合法概率定义域时抛出
        """
        min_val = float(np.min(y_pred))
        max_val = float(np.max(y_pred))
        eps_tol = 1e-6
        if min_val < -eps_tol or max_val > 1.0 + eps_tol:
            raise ValueError(
                f"BinaryCrossEntropy 期望输入为严格概率值 [0, 1]，实际输入超出范围: "
                f"min={min_val:.4f}, max={max_val:.4f}。若使用未激活的实数 Logits，请使用 BCEWithLogitsLoss。"
            )

        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        self.y_pred = y_pred_clipped
        self.y_true = y_true
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
        return max(0.0, float(loss))

    def backward(self) -> np.ndarray:
        """
        计算 BCE 概率梯度: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n
        """
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("BinaryCrossEntropy.backward() 必须在 forward() 之后调用")
        n = y_pred.shape[0] if y_pred.ndim > 0 else 1
        eps = 1e-12
        denominator = y_pred * (1.0 - y_pred) + eps
        return (y_pred - y_true) / denominator / n


class BCEWithLogitsLoss(Loss):
    """
    结合 Sigmoid 与二元交叉熵的数值稳定损失函数（Logits 实数域输入版本）。

    数学公式:
        forward:  L = (1/n) · Σ [ max(z, 0) - z·y + log(1 + exp(-|z|)) ]
        backward: dL/dz = (σ(z) - y) / n, 其中 σ(z) = 1 / (1 + exp(-z))

    优势:
        将 Sigmoid 与 Log-Loss 合并计算，利用 log-sum-exp 技巧消除上溢与下溢，
        支持全实数域 z ∈ (-∞, +∞)，且解析梯度与数值梯度在全定义域完全一致。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算带 Logits 的二元交叉熵损失。

        Args:
            y_pred: 未经过激活函数的线性 Logits z，shape (n, 1) 或 (n,)
            y_true: 真实二分类标签 y，取值为 0 或 1

        Returns:
            标量损失值 (>= 0)
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # 数值稳定二元交叉熵: max(z, 0) - z*y + log(1 + exp(-|z|))
        max_val = np.maximum(y_pred, 0.0)
        loss_elements = max_val - y_pred * y_true + np.log(1.0 + np.exp(-np.abs(y_pred)))
        return max(0.0, float(np.mean(loss_elements)))

    def backward(self) -> np.ndarray:
        """
        计算对 Logits 的梯度: dL/dz = (σ(z) - y) / n
        """
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("BCEWithLogitsLoss.backward() 必须在 forward() 之后调用")
        n = y_pred.shape[0] if y_pred.ndim > 0 else 1
        # 数值稳定 Sigmoid 计算
        pos_mask = y_pred >= 0
        sig_z = np.zeros_like(y_pred, dtype=float)
        sig_z[pos_mask] = 1.0 / (1.0 + np.exp(-y_pred[pos_mask]))
        exp_z = np.exp(y_pred[~pos_mask])
        sig_z[~pos_mask] = exp_z / (1.0 + exp_z)
        return (sig_z - y_true) / n


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
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("CategoricalCrossEntropy.backward() 必须在 forward() 之后调用")
        n = y_pred.shape[0]
        eps = 1e-12
        return -y_true / (y_pred + eps) / n
