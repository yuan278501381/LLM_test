# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.param_panel - 世界级交互式参数控制中枢 (极简极客风格)

提供结构清晰、交互丝滑的侧边栏控制面板：
- 经典实验一键预设库 (Instant Presets)
- 数据集拓扑选择与噪声调节
- 深度神经网络架构配置 (层数/神经元/激活/初始化)
- 训练超参数与优化器选择
- 正则化与防过拟合策略
- 实时神经元活性探针点选取
"""

from typing import Any
import streamlit as st


# ---------------------------------------------------------------------------
# 经典实验预设库
# ---------------------------------------------------------------------------
PRESETS = {
    "自定义配置 (Custom)": {
        "desc": "自由配置所有超参数与网络架构",
        "dataset": "Moons (双月分布)",
        "n_samples": 250,
        "noise": 0.12,
        "n_layers": 2,
        "neurons": [8, 4],
        "activation": "ReLU",
        "initializer": "he",
        "optimizer": "Adam",
        "lr": 0.05,
        "epochs": 150,
    },
    "线性可分基准 (Linear Baseline)": {
        "desc": "单层感知机即可完美求解的经典线性分类问题",
        "dataset": "Blobs (高斯聚类)",
        "n_samples": 200,
        "noise": 0.10,
        "n_layers": 1,
        "neurons": [1],
        "activation": "Sigmoid",
        "initializer": "xavier",
        "optimizer": "SGD",
        "lr": 0.1,
        "epochs": 100,
    },
    "XOR 历史困境与破解 (XOR Problem)": {
        "desc": "明斯基异或难题：单层线性失效，双层隐藏层破解非线性决策",
        "dataset": "XOR (正交分布)",
        "n_samples": 300,
        "noise": 0.08,
        "n_layers": 2,
        "neurons": [8, 4],
        "activation": "Tanh",
        "initializer": "xavier",
        "optimizer": "Adam",
        "lr": 0.05,
        "epochs": 200,
    },
    "双螺旋奇点挑战 (Spiral Singularity)": {
        "desc": "高曲率流形分类，检验深度网络的非线性特征空间扭曲拟合能力",
        "dataset": "Spiral (双螺旋)",
        "n_samples": 400,
        "noise": 0.15,
        "n_layers": 3,
        "neurons": [16, 12, 8],
        "activation": "LeakyReLU",
        "initializer": "he",
        "optimizer": "Adam",
        "lr": 0.03,
        "epochs": 300,
    },
    "梯度消失复现与拯救 (Vanishing Gradient)": {
        "desc": "深层 Sigmoid + Random 初始化导致前端梯度归零 vs ReLU + He 救场",
        "dataset": "Circles (同心圆)",
        "n_samples": 300,
        "noise": 0.10,
        "n_layers": 4,
        "neurons": [12, 12, 12, 12],
        "activation": "Sigmoid",
        "initializer": "random",
        "optimizer": "SGD",
        "lr": 0.05,
        "epochs": 200,
    },
}

DATASET_LABEL_MAP = {
    "Moons (双月分布)": "moons",
    "Circles (同心圆)": "circles",
    "XOR (正交分布)": "xor",
    "Spiral (双螺旋)": "spiral",
    "Blobs (高斯聚类)": "blobs",
}


def render_presets_selector(key_prefix: str = "") -> dict[str, Any] | None:
    """渲染一键实验预设选择器"""
    st.sidebar.markdown("#### PRESET // 经典实验预设")
    preset_choice = st.sidebar.selectbox(
        "选择预设模板",
        list(PRESETS.keys()),
        key=f"{key_prefix}preset_choice",
    )
    preset_data = PRESETS[preset_choice]

    if preset_choice != "自定义配置 (Custom)":
        st.sidebar.caption(f"{preset_data['desc']}")
        return preset_data
    return None


def render_dataset_selector(
    key_prefix: str = "", default_dataset: str = "Moons (双月分布)"
) -> tuple[str, int, float, int]:
    """渲染数据集选择与参数卡片"""
    st.sidebar.markdown("#### DATASET // 数据集拓扑")

    dataset_options = list(DATASET_LABEL_MAP.keys())
    default_idx = dataset_options.index(default_dataset) if default_dataset in dataset_options else 0

    dataset_label = st.sidebar.selectbox(
        "分布类型",
        dataset_options,
        index=default_idx,
        key=f"{key_prefix}dataset",
    )
    dataset_name = DATASET_LABEL_MAP[dataset_label]

    col1, col2 = st.sidebar.columns(2)
    with col1:
        n_samples = st.slider("样本量 (N)", 50, 1000, 250, step=50, key=f"{key_prefix}n_samples")
    with col2:
        noise = st.slider("噪声比 (Noise)", 0.0, 0.4, 0.1, step=0.02, key=f"{key_prefix}noise")

    random_state = st.sidebar.number_input("随机种子 (Seed)", 0, 9999, 42, key=f"{key_prefix}random_state")

    return dataset_name, n_samples, noise, random_state


def render_network_params(
    allow_multi_layer: bool = True,
    key_prefix: str = "",
    default_layers: int = 2,
    default_neurons: list[int] | None = None,
    default_act: str = "ReLU",
    default_init: str = "he",
) -> dict[str, Any]:
    """渲染网络架构参数中枢"""
    st.sidebar.markdown("#### ARCHITECTURE // 网络架构")

    if allow_multi_layer:
        n_layers = st.sidebar.slider(
            "隐藏层深度 (Depth)", 1, 5, default_layers,
            key=f"{key_prefix}n_layers",
            help="增加网络深度以学习更复杂的非线性拓扑折叠",
        )
    else:
        n_layers = 1

    neurons_per_layer = []
    defaults = default_neurons or [8, 4, 4, 4, 4]

    for i in range(n_layers):
        default_val = defaults[i] if i < len(defaults) else 4
        n = st.sidebar.slider(
            f"隐藏层 #{i+1} 神经元数",
            1 if not allow_multi_layer else 2, 64, default_val,
            key=f"{key_prefix}neurons_{i}",
        )
        neurons_per_layer.append(n)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        act_options = ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]
        act_idx = act_options.index(default_act) if default_act in act_options else 0
        activation = st.selectbox(
            "激活函数",
            act_options,
            index=act_idx,
            key=f"{key_prefix}activation",
        )
    with col2:
        init_options = ["he", "xavier", "random", "zeros"]
        init_idx = init_options.index(default_init) if default_init in init_options else 0
        initializer = st.selectbox(
            "参数初始化",
            init_options,
            index=init_idx,
            key=f"{key_prefix}initializer",
        )

    return {
        "n_layers": n_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation": activation,
        "initializer": initializer,
    }


def render_training_params(
    key_prefix: str = "",
    default_opt: str = "Adam",
    default_lr: float = 0.05,
    default_epochs: int = 150,
) -> dict[str, Any]:
    """渲染训练超参数控制台"""
    st.sidebar.markdown("#### OPTIMIZER // 优化算法")

    opt_options = ["Adam", "SGD", "Momentum", "RMSProp"]
    opt_idx = opt_options.index(default_opt) if default_opt in opt_options else 0
    optimizer = st.sidebar.selectbox(
        "算法类型",
        opt_options,
        index=opt_idx,
        key=f"{key_prefix}optimizer",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        lr = st.number_input(
            "学习率 (LR)",
            min_value=0.0001,
            max_value=2.0,
            value=default_lr,
            step=0.01,
            format="%.4f",
            key=f"{key_prefix}learning_rate",
        )
    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128, 256, 0],
            format_func=lambda x: "Full (全量)" if x == 0 else str(x),
            index=1,
            key=f"{key_prefix}batch_size",
        )

    epochs = st.sidebar.slider(
        "训练轮数 (Epochs)",
        10, 800, default_epochs, step=10,
        key=f"{key_prefix}epochs",
    )

    return {
        "optimizer": optimizer,
        "learning_rate": lr,
        "batch_size": batch_size if batch_size > 0 else None,
        "epochs": epochs,
    }


def render_probe_point_selector(key_prefix: str = "") -> tuple[float, float]:
    """渲染单样本活性探针坐标微调器"""
    st.sidebar.markdown("#### PROBE // 神经元活性探针")
    st.sidebar.caption("设置测试样本坐标，捕获前向传递中各层神经元激活强度：")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        px = st.number_input("探针坐标 x₁", -3.0, 3.0, 0.5, step=0.1, key=f"{key_prefix}probe_x")
    with col2:
        py = st.number_input("探针坐标 x₂", -3.0, 3.0, 0.5, step=0.1, key=f"{key_prefix}probe_y")
    return float(px), float(py)
