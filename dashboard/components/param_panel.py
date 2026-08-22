# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.components.param_panel - 世界级参数控制中枢 (Zero Hardcoding · 深度学习提示与案例解析)

所有选项、标签、数学公式、详细含义与学习案例均由 dashboard.constants.knowledge 元数据驱动。
为每个组件注入详尽的 Tooltip (`help`)，帮助用户透彻理解每个超参数的几何物理意义与实际训练影响。
"""

from typing import Any

import streamlit as st

from dashboard.constants.knowledge import (
    ACTIVATIONS,
    DATASETS,
    INITIALIZERS,
    OPTIMIZERS,
    PRESETS_REGISTRY,
    REGULARIZERS,
)

# 导出兼容别名
PRESETS = PRESETS_REGISTRY


DATASET_HINTS = {
    "双半月弯 (Moons)": "非线性交错流形 · 考验隐藏层非线性拟合能力",
    "同心圆环 (Circles)": "完全对称嵌套流形 · 线性分类器彻底失效的试金石",
    "双螺旋线 (Spiral)": "高曲率混沌奇点 · 考验深层网络容量与局部感知",
    "高斯双簇 (Blobs)": "线性可分基准流形 · 检验单层感知器收敛速度",
}

ACTIVATION_HINTS = {
    "ReLU (线性整流函数)": "正向恒等 / 负向截断 · 缓解深层梯度消失的工业界基石",
    "Sigmoid (S型激活函数)": "二分类概率压缩 (0, 1) · 易发生饱和与梯度消失",
    "Tanh (双曲正切函数)": "零中心化平滑映射 (-1, 1) · 梯度传播优于 Sigmoid",
    "LeakyReLU (带泄露线性整流)": "负区间保留微小斜率 (0.01) · 彻底避免神经元坏死",
    "GELU (高斯误差线性单元)": "概率随机正则化平滑门控 · Transformer 标配",
    "Linear (纯线性恒等变换)": "无非线性变换 · 多层网络将退化为单层仿射变换",
}

INITIALIZER_HINTS = {
    "He (Kaiming 正态分布)": "方差 2/Din · 专为 ReLU / LeakyReLU 设计的抗梯度消失方案",
    "Xavier (Glorot 均匀分布)": "方差 2/(Din+Dout) · 专为 Sigmoid / Tanh 设计的方差均衡方案",
    "标准高斯分布 (Random Normal)": "均值 0 方差 1 · 深层网络极易迅速导致梯度爆炸或弥散",
    "全零初始化 (Zeros)": "所有权重为 0 · 神经元完全对称，反向传播无法学习分化",
}

OPTIMIZER_HINTS = {
    "Adam (自适应矩估计)": "一阶动量 + 二阶方差自校准 · 深度学习工业界默认标配",
    "SGD (标准随机梯度下降)": "纯沿瞬时小批量梯度方向更新 · 容易在鞍点与峡谷震荡",
    "Momentum (动量梯度下降)": "累积历史速度惯性 · 有效冲出局部平坦区与高频震荡",
    "RMSprop (均方根传播)": "指数滑动平均调节学习率 · 专为非平稳目标与 RNN 设计",
}


def render_presets_selector(key_prefix: str = "") -> dict[str, Any] | None:
    """渲染一键实验预设选择器 (Visual Radio Cards)"""
    st.sidebar.markdown("#### PRESET // 经典实验预设")

    preset_options = list(PRESETS_REGISTRY.keys())
    preset_choice = st.sidebar.radio(
        "选择实验预设方案",
        options=preset_options,
        format_func=lambda o: f"**{o}**\n\n↳ *{PRESETS_REGISTRY[o]['desc']}*",
        help="点击一键载入深度学习历史上的经典实验配置（如明斯基 XOR 困境、双螺旋奇点、梯度消失等）。",
        key=f"{key_prefix}preset_choice",
    )
    preset_data = PRESETS_REGISTRY[preset_choice]

    if preset_choice != "自定义配置 (Custom)":
        st.sidebar.caption(f"{preset_data['desc']}")
        return preset_data
    return None


def render_dataset_selector(
    key_prefix: str = "", default_dataset: str = "moons"
) -> tuple[str, int, float, int]:
    """渲染数据集选择与参数卡片 (Visual Radio Cards)"""
    st.sidebar.markdown("#### DATASET // 数据集拓扑")

    dataset_labels = [meta.label for meta in DATASETS.values()]
    dataset_ids = list(DATASETS.keys())

    clean_default = default_dataset.lower()
    default_idx = next(
        (i for i, d_id in enumerate(dataset_ids) if d_id in clean_default or clean_default in d_id),
        0,
    )

    selected_label = st.sidebar.radio(
        "分布类型 (Distribution)",
        options=dataset_labels,
        format_func=lambda o: f"**{o}**\n\n↳ *{DATASET_HINTS.get(o, '2D 特征空间流形')}*",
        index=default_idx,
        help="选择 2D 特征空间的数据集流形结构。不同的几何拓扑对网络深度与激活函数非线性有不同的要求。",
        key=f"{key_prefix}dataset",
    )

    # 找到选中的 metadata 并显示其详尽学习提示
    selected_meta = next(m for m in DATASETS.values() if m.label == selected_label)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        n_samples = st.slider(
            "样本量 (N)",
            50,
            1000,
            250,
            step=50,
            help="数据集中的总样本点数量。样本量越大，决策边界泛化越平滑，单轮计算量成正比增加。",
            key=f"{key_prefix}n_samples",
        )
    with col2:
        noise = st.slider(
            "噪声比 (Noise)",
            0.0,
            0.4,
            0.1,
            step=0.02,
            help="高斯噪声标准差。噪声越大，两类数据在边界处重叠越严重，考验模型的正则化抗过拟合能力。",
            key=f"{key_prefix}noise",
        )

    random_state = st.sidebar.number_input(
        "随机种子 (Seed)",
        0,
        9999,
        42,
        help="控制伪随机数发生器起点，确保数据生成与实验结果可 100% 稳定复现。",
        key=f"{key_prefix}random_state",
    )

    return selected_meta.id, n_samples, noise, int(random_state)


def render_network_params(
    allow_multi_layer: bool = True,
    key_prefix: str = "",
    default_layers: int = 2,
    default_neurons: list[int] | None = None,
    default_act: str = "ReLU",
    default_init: str = "he",
) -> dict[str, Any]:
    """渲染网络架构参数中枢 (Visual Radio Cards)"""
    st.sidebar.markdown("#### ARCHITECTURE // 网络架构")

    if allow_multi_layer:
        n_layers = st.sidebar.slider(
            "隐藏层深度 (Depth)",
            1,
            5,
            default_layers,
            help="隐藏层数量。网络越深，能够执行的「非线性空间流形折叠」次数越多，但对梯度反向传播的要求也越高。",
            key=f"{key_prefix}n_layers",
        )
    else:
        n_layers = 1

    neurons_per_layer = []
    defaults = default_neurons or [8, 4, 4, 4, 4]

    for i in range(n_layers):
        default_val = defaults[i] if i < len(defaults) else 4
        n = st.sidebar.slider(
            f"隐藏层 #{i + 1} 神经元数",
            1 if not allow_multi_layer else 2,
            64,
            default_val,
            help=f"第 {i + 1} 隐藏层的特征维度。神经元数量越多，该层对复杂空间分界面的拟合容量越大。",
            key=f"{key_prefix}neurons_{i}",
        )
        neurons_per_layer.append(n)

    act_labels = [meta.label for meta in ACTIVATIONS.values() if meta.id != "Softmax"]
    act_ids = [meta.id for meta in ACTIVATIONS.values() if meta.id != "Softmax"]
    act_default_idx = next(
        (i for i, a_id in enumerate(act_ids) if a_id.lower() in default_act.lower()),
        0,
    )

    init_labels = [meta.label for meta in INITIALIZERS.values()]
    init_ids = list(INITIALIZERS.keys())
    init_default_idx = next(
        (i for i, i_id in enumerate(init_ids) if i_id.lower() in default_init.lower()),
        0,
    )

    selected_act_label = st.radio(
        "激活函数",
        options=act_labels,
        format_func=lambda o: f"**{o}**\n\n↳ *{ACTIVATION_HINTS.get(o, '非线性变换')}*",
        index=act_default_idx,
        help="赋予神经网络非线性表达能力的数学变换。没有激活函数的多层网络会退化为单层线性变换。",
        key=f"{key_prefix}activation",
    )

    selected_init_label = st.radio(
        "权重初始化",
        options=init_labels,
        format_func=lambda o: f"**{o}**\n\n↳ *{INITIALIZER_HINTS.get(o, '参数初始分布')}*",
        index=init_default_idx,
        help="网络初始权重的概率分布。合理的方差缩放初始化（如 He / Xavier）是深层网络避免梯度消失的关键。",
        key=f"{key_prefix}initializer",
    )

    act_meta = next(m for m in ACTIVATIONS.values() if m.label == selected_act_label)
    init_meta = next(m for m in INITIALIZERS.values() if m.label == selected_init_label)

    return {
        "n_layers": n_layers,
        "neurons_per_layer": neurons_per_layer,
        "activation": act_meta.id,
        "activation_label": act_meta.label,
        "initializer": init_meta.id,
        "initializer_label": init_meta.label,
    }


def render_training_params(
    key_prefix: str = "",
    default_opt: str = "Adam",
    default_lr: float = 0.05,
    default_epochs: int = 150,
) -> dict[str, Any]:
    """渲染训练超参数控制台 (Visual Radio Cards)"""
    st.sidebar.markdown("#### OPTIMIZER // 优化算法")

    opt_labels = [meta.label for meta in OPTIMIZERS.values()]
    opt_ids = list(OPTIMIZERS.keys())
    opt_default_idx = next(
        (i for i, o_id in enumerate(opt_ids) if o_id.lower() in default_opt.lower()),
        0,
    )

    selected_opt_label = st.sidebar.radio(
        "算法类型 (Algorithm)",
        options=opt_labels,
        format_func=lambda o: f"**{o}**\n\n↳ *{OPTIMIZER_HINTS.get(o, '参数梯度优化算法')}*",
        index=opt_default_idx,
        help="指导参数沿着损失曲面梯度方向寻优的更新策略（动量累积、自适应步长等）。",
        key=f"{key_prefix}optimizer",
    )

    opt_meta = next(m for m in OPTIMIZERS.values() if m.label == selected_opt_label)

    lr = st.sidebar.number_input(
        "学习率 (LR)",
        min_value=0.0001,
        max_value=2.0,
        value=default_lr,
        step=0.01,
        format="%.4f",
        help="参数更新步长 $\\eta$。过大易导致损失震荡发散，过小会导致收敛缓慢陷入局部极小。",
        key=f"{key_prefix}learning_rate",
    )

    batch_size = st.sidebar.radio(
        "批大小 (Batch)",
        options=[16, 32, 64, 128, 256, 0],
        format_func=lambda x: (
            "**Full (全量)**\n\n↳ *全数据梯度精确计算*"
            if x == 0
            else f"**Mini-Batch ({x})**\n\n↳ *兼顾速度与逃逸鞍点*"
        ),
        index=1,
        help="每次参数更新所使用的样本数量。Mini-batch 兼具计算效率与适度随机扰动（利于逃逸鞍点）。",
        key=f"{key_prefix}batch_size",
    )

    epochs = st.sidebar.slider(
        "训练轮数 (Epochs)",
        10,
        800,
        default_epochs,
        step=10,
        help="全量数据集被前向与反向遍历的总循环次数。",
        key=f"{key_prefix}epochs",
    )

    return {
        "optimizer": opt_meta.id,
        "optimizer_label": opt_meta.label,
        "learning_rate": float(lr),
        "batch_size": batch_size if batch_size > 0 else None,
        "epochs": int(epochs),
    }


def render_probe_point_selector(key_prefix: str = "") -> tuple[float, float]:
    """渲染单样本活性探针坐标微调器"""
    st.sidebar.markdown("#### PROBE // 神经元活性探针")
    st.sidebar.caption("设置虚拟测试样本 (x₁, x₂)，实时捕获信号在前向各层神经元中的激发状态：")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        px = st.number_input(
            "探针坐标 x₁",
            -3.0,
            3.0,
            0.5,
            step=0.1,
            help="探针样本在特征 x₁ 轴的坐标值",
            key=f"{key_prefix}probe_x",
        )
    with col2:
        py = st.number_input(
            "探针坐标 x₂",
            -3.0,
            3.0,
            0.5,
            step=0.1,
            help="探针样本在特征 x₂ 轴的坐标值",
            key=f"{key_prefix}probe_y",
        )
    return float(px), float(py)


def render_deep_dive_card(
    title: str,
    meta_items: list[Any],
    icon_name: str = "book-open",
) -> None:
    """渲染富文本微观原理解析与学习案例卡片 (Collapsible Deep Dive, 100% 矢量图标)"""
    with st.expander(f"DEEP DIVE // 微观机理与案例解析: {title}", expanded=False):
        for item in meta_items:
            formula_line = f"`{item.formula}`" if hasattr(item, "formula") else ""
            impact_val = getattr(item, "impact", getattr(item, "difficulty", ""))
            st.markdown(
                f"##### **{item.label}** {formula_line}\n"
                f"- **[CONCEPT // 详细含义]**: {item.desc}\n"
                f"- **[IMPACT // 动态影响]**: {impact_val}\n"
                f"- **[BENCHMARK // 实战案例]**: {item.example}\n"
            )


def render_regularization_params(key_prefix: str = "") -> dict[str, Any]:
    """渲染正则化与惩罚项参数 (Visual Radio Cards)"""
    st.sidebar.markdown("#### REGULARIZATION // 正则化机制")

    reg_labels = [meta.label for meta in REGULARIZERS.values()]
    reg_hints = {
        "None (无正则)": "纯经验风险最小化 · 自由拟合无参数惩罚",
        "L1 正则 (Lasso)": "曼哈顿范数约束 · 驱动权重绝对稀疏化 (产生 0 权重)",
        "L2 正则 (Ridge)": "欧氏范数约束 · 惩罚极端大权重 · 提升泛化",
    }

    selected_reg_label = st.sidebar.radio(
        "正则化类型",
        options=reg_labels,
        format_func=lambda o: f"**{o}**\n\n↳ *{reg_hints.get(o, '泛化约束')}*",
        index=0,
        help="通过在损失函数中增加权重范数惩罚，约束模型复杂度，防止过拟合。",
        key=f"{key_prefix}regularization",
    )

    reg_meta = next(m for m in REGULARIZERS.values() if m.label == selected_reg_label)

    reg_lambda = 0.0
    if reg_meta.id != "None":
        reg_lambda = st.sidebar.slider(
            "惩罚强度 (Lambda \\lambda)",
            0.0001,
            0.1,
            0.01,
            step=0.001,
            format="%.4f",
            help="正则化损失项的权重系数。值越大，对网络权重的惩罚越严厉，边界越平滑简单。",
            key=f"{key_prefix}reg_lambda",
        )

    return {
        "regularization": reg_meta.id,
        "regularization_label": reg_meta.label,
        "lambda": reg_lambda,
    }
