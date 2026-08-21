# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.param_panel - 可复用的参数控制面板

提供标准化的 Streamlit 侧边栏控件集合，确保各页面参数控件一致。
"""

import streamlit as st


def render_dataset_selector(key_prefix: str = "") -> tuple[str, int, float, int]:
    """
    渲染数据集选择控件。

    Returns:
        (dataset_name, n_samples, noise, random_state)
    """
    st.sidebar.subheader("📊 数据集")
    dataset = st.sidebar.selectbox(
        "选择数据集",
        ["🌙 Moons", "⭕ Circles", "❌ XOR", "🌀 Spiral"],
        key=f"{key_prefix}dataset",
    )
    dataset_name = dataset.split(" ")[1].lower()

    n_samples = st.sidebar.slider(
        "样本数量", 50, 1000, 200, step=50,
        key=f"{key_prefix}n_samples",
    )
    noise = st.sidebar.slider(
        "噪声强度", 0.0, 0.5, 0.1, step=0.01,
        key=f"{key_prefix}noise",
    )
    random_state = st.sidebar.number_input(
        "随机种子", 0, 9999, 42,
        key=f"{key_prefix}random_state",
    )

    return dataset_name, n_samples, noise, random_state


def render_network_params(
    allow_multi_layer: bool = True,
    key_prefix: str = "",
) -> dict:
    """
    渲染网络结构控件。

    Returns:
        {
            'n_layers': int,
            'neurons_per_layer': list[int],
            'activation': str,
            'initializer': str,
        }
    """
    st.sidebar.subheader("🧱 网络结构")

    if allow_multi_layer:
        n_layers = st.sidebar.slider(
            "隐藏层数", 1, 6, 2,
            key=f"{key_prefix}n_layers",
        )
    else:
        n_layers = 1

    neurons_per_layer = []
    for i in range(n_layers):
        n = st.sidebar.slider(
            f"第 {i+1} 层神经元数",
            2, 64, 8 if i == 0 else 4,
            key=f"{key_prefix}neurons_{i}",
        )
        neurons_per_layer.append(n)

    activation = st.sidebar.selectbox(
        "激活函数",
        ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"],
        key=f"{key_prefix}activation",
    )

    initializer = st.sidebar.selectbox(
        "权重初始化",
        ["xavier", "he", "random", "zeros"],
        key=f"{key_prefix}initializer",
    )

    return {
        "n_layers": n_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation": activation,
        "initializer": initializer,
    }


def render_training_params(key_prefix: str = "") -> dict:
    """
    渲染训练超参数控件。

    Returns:
        {
            'learning_rate': float,
            'epochs': int,
            'batch_size': int,
            'optimizer': str,
            'loss': str,
        }
    """
    st.sidebar.subheader("⚙️ 训练参数")

    optimizer = st.sidebar.selectbox(
        "优化器",
        ["Adam", "SGD", "Momentum", "RMSProp"],
        key=f"{key_prefix}optimizer",
    )

    learning_rate = st.sidebar.select_slider(
        "学习率",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        value=0.01,
        key=f"{key_prefix}lr",
    )

    epochs = st.sidebar.slider(
        "训练轮数 (Epochs)", 10, 2000, 200, step=10,
        key=f"{key_prefix}epochs",
    )

    batch_size = st.sidebar.select_slider(
        "批大小 (Batch Size)",
        options=[4, 8, 16, 32, 64, 128, 256],
        value=32,
        key=f"{key_prefix}batch_size",
    )

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
    }


def render_regularization_params(key_prefix: str = "") -> dict:
    """
    渲染正则化控件。

    Returns:
        {
            'type': str | None,
            'strength': float,
            'dropout_rate': float,
        }
    """
    st.sidebar.subheader("🛡️ 正则化")

    reg_type = st.sidebar.selectbox(
        "正则化类型",
        ["无", "L1", "L2"],
        key=f"{key_prefix}reg_type",
    )

    strength = 0.01
    if reg_type != "无":
        strength = st.sidebar.select_slider(
            "正则化强度 (λ)",
            options=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
            value=0.01,
            key=f"{key_prefix}reg_strength",
        )

    dropout_rate = st.sidebar.slider(
        "Dropout 比例", 0.0, 0.8, 0.0, step=0.05,
        key=f"{key_prefix}dropout_rate",
    )

    return {
        "type": reg_type if reg_type != "无" else None,
        "strength": strength,
        "dropout_rate": dropout_rate,
    }
