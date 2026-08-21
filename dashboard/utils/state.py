# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.utils.state - Streamlit Session State 与元数据解析中枢 (Zero Hardcoding)

基于 dashboard.constants.knowledge 统一元数据驱动，提供无硬编码的：
- 激活函数解析
- 优化器解析
- 初始化器解析
- 正则化器解析
- 数据集获取
- Session State 安全管理
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

from dashboard.constants.knowledge import (
    ACTIVATIONS,
    DATASETS,
    INITIALIZERS,
    OPTIMIZERS,
    REGULARIZERS,
)
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
        if k.startswith(("model_", "history_", "snapshot_", "training_", "m4_"))
    ]
    for k in keys_to_remove:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# 映射注册表 (由 knowledge 元数据驱动，拒绝硬编码)
# ---------------------------------------------------------------------------
_RAW_ACTIVATION_CLASSES: dict[str, type] = {
    "ReLU": ReLU,
    "Sigmoid": Sigmoid,
    "Tanh": Tanh,
    "LeakyReLU": LeakyReLU,
    "Softmax": Softmax,
}

_RAW_OPTIMIZER_CLASSES: dict[str, type] = {
    "Adam": Adam,
    "SGD": SGD,
    "Momentum": Momentum,
    "RMSProp": RMSProp,
}

_RAW_LOSS_CLASSES: dict[str, type] = {
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "MSE": MSE,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
}

# 动态构建包含 ID 与 Label 的全局查询字典
ACTIVATION_MAP: dict[str, type] = {}
for act_id, meta in ACTIVATIONS.items():
    cls = _RAW_ACTIVATION_CLASSES[act_id]
    ACTIVATION_MAP[act_id] = cls
    ACTIVATION_MAP[meta.label] = cls

OPTIMIZER_MAP: dict[str, type] = {}
for opt_id, meta in OPTIMIZERS.items():
    cls = _RAW_OPTIMIZER_CLASSES[opt_id]
    OPTIMIZER_MAP[opt_id] = cls
    OPTIMIZER_MAP[meta.label] = cls

INITIALIZER_MAP: dict[str, str] = {}
for init_id, meta in INITIALIZERS.items():
    INITIALIZER_MAP[init_id] = init_id
    INITIALIZER_MAP[meta.label] = init_id

LOSS_MAP: dict[str, type] = {
    "MSE": MSE,
    "MSE (均方误差)": MSE,
    "BinaryCrossEntropy": BinaryCrossEntropy,
    "BinaryCrossEntropy (二元交叉熵)": BinaryCrossEntropy,
    "CategoricalCrossEntropy": CategoricalCrossEntropy,
    "CategoricalCrossEntropy (多元交叉熵)": CategoricalCrossEntropy,
}


def resolve_activation(name: str) -> type:
    """智能解析激活函数（支持 ID、中英双语 Label 及前缀）"""
    if name in ACTIVATION_MAP:
        return ACTIVATION_MAP[name]
    clean = name.split(" ")[0].strip()
    return ACTIVATION_MAP.get(clean, ReLU)


def resolve_optimizer(name: str) -> type:
    """智能解析优化器（支持 ID、中英双语 Label 及前缀）"""
    if name in OPTIMIZER_MAP:
        return OPTIMIZER_MAP[name]
    clean = name.split(" ")[0].strip()
    return OPTIMIZER_MAP.get(clean, Adam)


def resolve_initializer(name: str) -> str:
    """智能解析初始化器名称"""
    if name in INITIALIZER_MAP:
        return INITIALIZER_MAP[name]
    clean = name.split(" ")[0].strip().lower()
    return INITIALIZER_MAP.get(clean, "he")


def resolve_regularizer(name: str | None, strength: float = 0.01) -> Any:
    """智能解析正则化器"""
    if not name or "None" in name:
        return None
    if "L1" in name:
        return L1(lambda_=strength)
    if "L2" in name:
        return L2(lambda_=strength)
    return None


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
    """根据元数据构建 Sequential 模型"""
    act_cls = resolve_activation(activation)
    init_name = resolve_initializer(initializer)
    reg = resolve_regularizer(regularizer_type, regularizer_strength)

    model = Sequential()

    # 隐藏层
    prev_dim = n_inputs
    for n_neurons in hidden_layers:
        model.add(Dense(prev_dim, n_neurons, initializer=init_name, regularizer=reg))
        model.add(act_cls())
        if dropout_rate > 0:
            model.add(Dropout(rate=dropout_rate))
        prev_dim = n_neurons

    # 输出层
    model.add(Dense(prev_dim, n_outputs, initializer=init_name))

    # 输出层激活
    if output_activation:
        out_act_cls = resolve_activation(output_activation)
        model.add(out_act_cls())

    return model


def get_dataset(
    name: str,
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """统一获取 2D 合成数据集 (返回 y 为 (n, 1) 二分类标签)"""
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
        X, y_onehot = make_blobs(n_samples=n_samples, noise=noise, n_classes=2, random_state=random_state)
        y = np.argmax(y_onehot, axis=1).reshape(-1, 1).astype(np.float64)
        return X, y
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
