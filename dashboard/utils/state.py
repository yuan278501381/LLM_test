# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.utils.state - Streamlit Session State 统一管理

集中管理所有页面的会话状态，包括模型实例、训练历史、
实验快照等，确保跨页面状态一致性。
"""

import os
import sys
from typing import Any

import numpy as np
import streamlit as st

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh
from nn_core.layers import Dense, Dropout
from nn_core.losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from nn_core.model import Sequential
from nn_core.optimizers import SGD, Adam, Momentum, RMSProp
from nn_core.regularizers import L1, L2


def get_state(key: str, default: Any = None) -> Any:
    """安全获取 session state 值"""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """设置 session state 值"""
    st.session_state[key] = value


def reset_state() -> None:
    """清除所有训练相关的 session state"""
    keys_to_remove = [
        k for k in st.session_state.keys()
        if k.startswith(("model_", "history_", "snapshot_", "training_"))
    ]
    for k in keys_to_remove:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# 激活函数 / 优化器 / 损失函数 注册表
# ---------------------------------------------------------------------------
ACTIVATION_MAP: dict[str, type] = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "LeakyReLU": LeakyReLU,
}

OPTIMIZER_MAP: dict[str, type] = {
    "SGD": SGD,
    "Momentum": Momentum,
    "RMSProp": RMSProp,
    "Adam": Adam,
}

LOSS_MAP: dict[str, type] = {
    "MSE": MSE,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
}


def build_model(
    n_inputs: int,
    n_outputs: int,
    hidden_layers: list[int],
    activation: str = "ReLU",
    initializer: str = "xavier",
    regularizer_type: str | None = None,
    regularizer_strength: float = 0.01,
    dropout_rate: float = 0.0,
    output_activation: str | None = None,
) -> Sequential:
    """
    根据参数配置构建 Sequential 模型。

    Args:
        n_inputs: 输入特征维度
        n_outputs: 输出维度
        hidden_layers: 隐藏层神经元列表，如 [16, 8]
        activation: 隐藏层激活函数名称
        initializer: 权重初始化策略
        regularizer_type: 正则化类型 ('L1'|'L2'|None)
        regularizer_strength: 正则化强度
        dropout_rate: Dropout 比例 (0 = 不使用)
        output_activation: 输出层激活函数

    Returns:
        构建好的 Sequential 模型
    """
    act_cls = ACTIVATION_MAP.get(activation, ReLU)

    # 构建正则化器
    reg = None
    if regularizer_type == "L1":
        reg = L1(lambda_=regularizer_strength)
    elif regularizer_type == "L2":
        reg = L2(lambda_=regularizer_strength)

    model = Sequential()

    # 隐藏层
    prev_dim = n_inputs
    for n_neurons in hidden_layers:
        model.add(Dense(prev_dim, n_neurons, initializer=initializer, regularizer=reg))
        model.add(act_cls())
        if dropout_rate > 0:
            model.add(Dropout(rate=dropout_rate))
        prev_dim = n_neurons

    # 输出层
    model.add(Dense(prev_dim, n_outputs, initializer=initializer))

    # 输出层激活
    if output_activation:
        out_act_cls = {
            "Sigmoid": Sigmoid,
            "Softmax": Softmax,
        }.get(output_activation)
        if out_act_cls:
            model.add(out_act_cls())

    return model


def get_dataset(
    name: str,
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """统一获取 2D 合成数据集"""
    from datasets.generators import make_blobs, make_circles, make_moons, make_spiral, make_xor

    clean_name = name.lower().strip()
    if "moon" in clean_name:
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif "circle" in clean_name:
        return make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    elif "xor" in clean_name:
        return make_xor(n_samples=n_samples, noise=noise, random_state=random_state)
    elif "spiral" in clean_name:
        return make_spiral(n_samples=n_samples, noise=noise, random_state=random_state)
    elif "blob" in clean_name:
        return make_blobs(n_samples=n_samples, noise=noise, centers=2, random_state=random_state)
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

