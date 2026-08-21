# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 1: 单神经元感知器 (Single Neuron Perceptron) - 无硬编码 · 知识图谱深度解析

微观解构最小计算单元：线性加权 Z = XW + b、激活函数非线性映射、交叉熵损失、反向传播链式推导与权重轨迹寻优。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_loss_curve,
    plot_weight_trajectory,
)
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_deep_dive_card,
)
from dashboard.constants.knowledge import ACTIVATIONS, OPTIMIZERS
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_section_heading,
)
from dashboard.utils.state import get_dataset, resolve_activation, resolve_optimizer
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Single Perceptron · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="单神经元感知器",
    subtitle="解剖深度学习最小计算基元：线性加权 $Z = XW + b$、非线性激活与梯度下降的几何本质",
    badge_text="MILESTONE 01 // ATOMIC PERCEPTRON",
    badge_type="blue",
)

st.info(
    "💡 **极速概念辨析与入门指引**：\n"
    "- 👈 **左侧边栏**：是您作为人类架构师手动控制的 **【超参数 (Hyperparameters)】**（例如：学习率、训练轮数、激活函数、样本量）。\n"
    "- 👆 **上方 4 张仪表盘卡片**：是神经网络通过反向传播算法，**自主训练计算出来的【最终学习成果 (Learned Parameters)】**（如最终损失、准确率、模型算出的直线方程参数），您无需手动填写它们。"
)

# ---------------------------------------------------------------------------
# 侧边栏参数面板 (由 knowledge 元数据驱动，带丰富 Tooltip)
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m1_", default_dataset="blobs"
)

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")

act_list = [m for m in ACTIVATIONS.values() if m.id != "Softmax"]
act_labels = [m.label for m in act_list]
selected_act_label = st.sidebar.selectbox(
    "激活函数",
    act_labels,
    help="非线性激活函数。对单神经元而言，Sigmoid 将实数加权值映射为 (0,1) 的二分类置信概率。",
    key="m1_act",
)
act_meta = next(m for m in act_list if m.label == selected_act_label)

opt_list = list(OPTIMIZERS.values())
opt_labels = [m.label for m in opt_list]
selected_opt_label = st.sidebar.selectbox(
    "优化器",
    opt_labels,
    help="梯度下降算法。负责根据损失对权重 (w₁, w₂) 和偏置 b 的梯度调整参数大小。",
    key="m1_opt",
)
opt_meta = next(m for m in opt_list if m.label == selected_opt_label)

col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input(
        "学习率 (LR)",
        0.001, 2.0, 0.1, step=0.01, format="%.3f",
        help="参数更新步长 $\\eta$。过大导致跳过最优点剧烈震荡，过小导致收敛缓慢。",
        key="m1_lr",
    )
with col2:
    epochs = st.slider(
        "训练轮数",
        10, 500, 100, step=10,
        help="全量数据集迭代轮数。",
        key="m1_epochs",
    )

# ---------------------------------------------------------------------------
# 数据生成与模型训练
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

# 构建单神经元模型: Input(2) -> Dense(2, 1) -> Activation
model = Sequential()
dense_layer = Dense(2, 1, initializer="random")
model.add(dense_layer)

act_cls = resolve_activation(act_meta.id)
model.add(act_cls())

loss_fn = BinaryCrossEntropy()
opt_cls = resolve_optimizer(opt_meta.id)
optimizer = opt_cls(learning_rate=float(lr))

# 记录权重轨迹
weight_trajectory = [dense_layer.weights.copy()]

# 训练循环并记录历史
history: dict[str, list[float]] = {"loss": [], "accuracy": []}

for _ in range(int(epochs)):
    y_pred = model.forward(X, training=True)
    loss = loss_fn.forward(y_pred, y)
    acc = float(np.mean((y_pred >= 0.5).astype(float) == y))

    history["loss"].append(loss)
    history["accuracy"].append(acc)

    dloss = loss_fn.backward()
    model.backward(dloss)

    optimizer.step(model.layers)
    weight_trajectory.append(dense_layer.weights.copy())

final_loss = history["loss"][-1] if history["loss"] else 0.0
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
w_final = dense_layer.weights.ravel()
b_final = float(dense_layer.biases.ravel()[0])

# ---------------------------------------------------------------------------
# 遥测指标卡 (模型自动学习输出结果)
# ---------------------------------------------------------------------------
sign_b = "+" if b_final >= 0 else "-"
abs_b = abs(b_final)
hyperplane_str = f"{w_final[0]:.2f}x₁ + {w_final[1]:.2f}x₂ {sign_b} {abs_b:.2f} = 0"

grid_html = (
    '<div class="metric-grid">'
    + render_metric_card("FINAL LOSS // 最终训练损失", f"{final_loss:.4f}", delta="已收敛 (CONVERGED)" if final_loss < 0.2 else "训练中 (TRAINING)", delta_type="positive" if final_loss < 0.2 else "neutral", icon_name="trending-down")
    + render_metric_card("ACCURACY // 分类准确率", f"{final_acc:.1%}", delta="达标 (OPTIMAL)" if final_acc >= 0.95 else "收敛中", delta_type="positive" if final_acc >= 0.9 else "neutral", icon_name="target")
    + render_metric_card("LEARNED WEIGHTS // 模型自主学得权重", f"[{w_final[0]:.2f}, {w_final[1]:.2f}]", delta=f"偏置截距 b = {b_final:.2f}", delta_type="neutral", icon_name="sliders")
    + render_metric_card("DECISION LINE // 学得的直线方程", f'<span style="font-size:1.05rem;">{hyperplane_str}</span>', delta="模型自动求解的决策分界面", delta_type="positive", icon_name="activity")
    + '</div>'
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 可视化布局
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    fig_boundary = plot_decision_boundary(
        model, X, y, title=f"DECISION BOUNDARY // {act_meta.id.upper()} 空间决策流形"
    )
    st.plotly_chart(fig_boundary, use_container_width=True)

with col_right:
    fig_loss = plot_loss_curve(history, title="TRAINING DYNAMICS // 损失与准确率收敛")
    st.plotly_chart(fig_loss, use_container_width=True)

# 底部权重轨迹图
render_section_heading("权重参数空间寻优轨迹 (Weight Trajectory)", icon_name="crosshair", subtext="参数 (w₁, w₂) 从初始位置沿损失梯度向全局最优收敛的连续路径：")
fig_traj = plot_weight_trajectory(weight_trajectory, title="PARAMETER TRAJECTORY // 参数空间梯度搜索路径")
st.plotly_chart(fig_traj, use_container_width=True)

# 深度知识学习指南 (折叠微观原理解析)
render_deep_dive_card("单神经元感知器核心参数与激活函数", [act_meta, opt_meta])
