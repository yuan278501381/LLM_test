# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.losses - 损失函数模块

实现常用损失函数，每个损失函数实现 forward() 计算损失值和 backward() 计算梯度。
forward 时自动缓存预测值和真实值，供 backward 使用。

支持的损失函数:
    - MSE: 均方误差（回归任务）
    - BinaryCrossEntropy: 二元交叉熵（概率域输入 [0, 1]）
    - BCEWithLogitsLoss: 带 Logits 的二元交叉熵（实数域输入，数值稳定）
    - CategoricalCrossEntropy: 多分类交叉熵（概率单纯形输入）
    - CategoricalCrossEntropyWithLogits: 带 Logits 的多分类交叉熵（实数域输入，数值稳定）
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


def _validate_basic_loss_inputs(y_pred: np.ndarray, y_true: np.ndarray, loss_name: str) -> None:
    """通用输入基础校验：非空、NumPy 数组、Shape 一致、全为有限数值（禁止 NaN/Inf）。"""
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise TypeError(
            f"{loss_name} 期望输入为 np.ndarray，实际得到: y_pred={type(y_pred)}, y_true={type(y_true)}"
        )
    if y_pred.size == 0 or y_true.size == 0:
        raise ValueError(
            f"{loss_name} 输入不能为空数组，实际得到: y_pred.shape={y_pred.shape}, y_true.shape={y_true.shape}"
        )
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"{loss_name} 预测值与真实标签形状不匹配: y_pred.shape={y_pred.shape} vs y_true.shape={y_true.shape}"
        )
    if not np.isfinite(y_pred).all():
        raise ValueError(f"{loss_name} 预测值 y_pred 包含 NaN 或 Inf 非有限数值")
    if not np.isfinite(y_true).all():
        raise ValueError(f"{loss_name} 真实标签 y_true 包含 NaN 或 Inf 非有限数值")


def _validate_binary_labels(y_true: np.ndarray, loss_name: str) -> None:
    """二分类标签合法性校验：标签必须严格取值为 0 或 1 (或 0.0/1.0)。"""
    is_zero = np.isclose(y_true, 0.0, atol=1e-7)
    is_one = np.isclose(y_true, 1.0, atol=1e-7)
    if not np.logical_or(is_zero, is_one).all():
        raise ValueError(
            f"{loss_name} 要求真实二分类标签 y_true 取值严格在 {{0, 1}} 集合内，实际存在非法标签值"
        )


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
        _validate_basic_loss_inputs(y_pred, y_true, "MSE")
        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()
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
    二元交叉熵损失函数（严格概率输入版本）。

    数学公式:
        forward:  L = -(1/n) · Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        backward: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n

    输入契约:
        要求输入 ŷ 为严格概率范围 [0.0, 1.0]（如经过 Sigmoid 激活）。
        若输入超出 [0, 1] 定义域，将显式抛出 ValueError，杜绝掩盖契约错误。
        若使用未归一化的全实数 Logits，请使用数值更稳定的 BCEWithLogitsLoss。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算二元交叉熵损失。

        Args:
            y_pred: 模型输出概率，shape (n, 1) 或 (n,)，严格要求值域在 [0.0, 1.0]
            y_true: 真实标签，shape 与 y_pred 一致，取值为 0 或 1

        Returns:
            标量损失值

        Raises:
            ValueError: 当 y_pred 超出 [0.0, 1.0] 或包含非法标签时抛出
        """
        _validate_basic_loss_inputs(y_pred, y_true, "BinaryCrossEntropy")
        _validate_binary_labels(y_true, "BinaryCrossEntropy")

        min_val = float(np.min(y_pred))
        max_val = float(np.max(y_pred))
        if min_val < 0.0 or max_val > 1.0:
            raise ValueError(
                f"BinaryCrossEntropy 期望输入为严格概率值 [0, 1]，实际输入超出范围: "
                f"min={min_val:.6f}, max={max_val:.6f}。若使用未激活的实数 Logits，请使用 BCEWithLogitsLoss。"
            )

        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()

        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算 BCE 概率梯度: dL/dŷ = (ŷ - y) / (ŷ·(1-ŷ) + ε) / n
        """
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("BinaryCrossEntropy.backward() 必须在 forward() 之后调用")
        n = y_pred.shape[0] if y_pred.ndim > 0 else 1
        eps = 1e-15
        denominator = np.clip(y_pred * (1.0 - y_pred), eps, None)
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
            标量损失值
        """
        _validate_basic_loss_inputs(y_pred, y_true, "BCEWithLogitsLoss")
        _validate_binary_labels(y_true, "BCEWithLogitsLoss")

        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()

        # 数值稳定二元交叉熵: max(z, 0) - z*y + log(1 + exp(-|z|))
        max_val = np.maximum(y_pred, 0.0)
        loss_elements = max_val - y_pred * y_true + np.log1p(np.exp(-np.abs(y_pred)))
        return float(np.mean(loss_elements))

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
    分类交叉熵损失函数（概率单纯形输入版本）。

    数学公式:
        forward:  L = -(1/n) · Σ_i Σ_c [y_true_{i,c} · log(y_pred_{i,c})]
        backward: dL/dŷ = -y_true / (ŷ + ε) / n

    适用场景: 多分类任务（输出层显式经过 Softmax 归一化）
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算分类交叉熵损失。

        Args:
            y_pred: 模型输出概率，shape (n, c)，每行和为 1 且全非负
            y_true: one-hot 或概率分布标签，shape (n, c)

        Returns:
            标量损失值
        """
        _validate_basic_loss_inputs(y_pred, y_true, "CategoricalCrossEntropy")

        if y_pred.ndim != 2:
            raise ValueError(
                f"CategoricalCrossEntropy 要求 2D 输入 (n_samples, n_classes)，实际 shape={y_pred.shape}"
            )
        if (y_pred < 0.0).any() or (y_pred > 1.0).any():
            raise ValueError("CategoricalCrossEntropy 期望输入为合法概率值 [0, 1]")
        if (y_true < 0.0).any():
            raise ValueError("CategoricalCrossEntropy 真实标签 y_true 必须非负")

        # 概率单纯形校验：每行和应约为 1.0
        pred_row_sums = np.sum(y_pred, axis=1)
        if not np.allclose(pred_row_sums, 1.0, atol=1e-2):
            raise ValueError("CategoricalCrossEntropy 期望 y_pred 每行概率和为 1.0")

        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()

        eps = 1e-15
        y_pred_safe = np.clip(y_pred, eps, 1.0)
        loss = -np.mean(np.sum(y_true * np.log(y_pred_safe), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """
        计算 CCE 梯度: dL/dŷ = -y_true / (ŷ + ε) / n
        """
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError("CategoricalCrossEntropy.backward() 必须在 forward() 之后调用")
        n = y_pred.shape[0]
        eps = 1e-15
        denominator = np.clip(y_pred, eps, None)
        return -y_true / denominator / n


class CategoricalCrossEntropyWithLogits(Loss):
    """
    带 Logits 的数值稳定多分类交叉熵损失函数（结合 Softmax 与 CCE）。

    数学公式:
        forward:  L = (1/n) · Σ_i [ log(Σ_c exp(z_{i,c})) - Σ_c y_{i,c} z_{i,c} ]
        backward: dL/dz = (Softmax(z) - y_true) / n

    优势:
        利用 Log-Sum-Exp (LSE) 消除实数 Logits 的数值溢出，提供精确光滑反向梯度。
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        计算带 Logits 的分类交叉熵损失。

        Args:
            y_pred: 未经过 Softmax 归一化的实数 Logits z，shape (n, c)
            y_true: one-hot 或概率分布标签，shape (n, c)

        Returns:
            标量损失值
        """
        _validate_basic_loss_inputs(y_pred, y_true, "CategoricalCrossEntropyWithLogits")
        if y_pred.ndim != 2:
            raise ValueError(
                f"CategoricalCrossEntropyWithLogits 要求 2D 输入 (n, c)，实际 shape={y_pred.shape}"
            )
        if (y_true < 0.0).any():
            raise ValueError("CategoricalCrossEntropyWithLogits 真实标签 y_true 必须非负")

        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()

        # 数值稳定 Log-Sum-Exp
        max_logits = np.max(y_pred, axis=1, keepdims=True)
        exp_shifted = np.exp(y_pred - max_logits)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        lse = max_logits + np.log(sum_exp)

        # loss = mean( LSE(z_i) - sum_c(y_ic * z_ic) )
        loss_elements = lse.ravel() - np.sum(y_true * y_pred, axis=1)
        return float(np.mean(loss_elements))

    def backward(self) -> np.ndarray:
        """
        计算对 Logits 的梯度: dL/dz = (Softmax(z) - y_true) / n
        """
        y_pred, y_true = self.y_pred, self.y_true
        if y_pred is None or y_true is None:
            raise RuntimeError(
                "CategoricalCrossEntropyWithLogits.backward() 必须在 forward() 之后调用"
            )
        n = y_pred.shape[0]

        max_logits = np.max(y_pred, axis=1, keepdims=True)
        exp_shifted = np.exp(y_pred - max_logits)
        softmax_probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return (softmax_probs - y_true) / n
