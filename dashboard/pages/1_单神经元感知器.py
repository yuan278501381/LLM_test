# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
🎯 里程碑 1: 单神经元感知器 (Single Neuron Perceptron)

微观解构最小计算单元：前向传播、损失函数、反向传播与权重空间寻优。
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
from dashboard.components.param_panel import render_dataset_selector
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
)
from dashboard.utils.state import ACTIVATION_MAP, OPTIMIZER_MAP, get_dataset
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="单神经元感知器 · NN Playground",
    page_icon="🎯",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="🎯 单神经元感知器",
    subtitle="解剖深度学习最小计算基元：线性加权 $Z = XW + b$、非线性激活与梯度下降的几何本质",
    badge_text="MILESTONE 1 · ATOMIC COMPUTATION",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 侧边栏参数面板
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m1_", default_dataset="🫧 Blobs"
)

st.sidebar.markdown("### 🧬 神经元超参数")
activation_name = st.sidebar.selectbox("激活函数", ["Sigmoid", "ReLU", "Tanh", "LeakyReLU"], key="m1_act")
optimizer_name = st.sidebar.selectbox("优化器", ["SGD", "Momentum", "RMSProp", "Adam"], key="m1_opt")

col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input("学习率 (LR)", 0.001, 2.0, 0.1, step=0.01, format="%.3f", key="m1_lr")
with col2:
    epochs = st.slider("训练轮数", 10, 500, 100, step=10, key="m1_epochs")

# ---------------------------------------------------------------------------
# 数据生成与模型训练
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

# 构建单神经元模型: Input(2) -> Dense(2, 1) -> Activation
model = Sequential()
dense_layer = Dense(2, 1, initializer="random")
model.add(dense_layer)
model.add(ACTIVATION_MAP[activation_name]())

loss_fn = BinaryCrossEntropy()
optimizer = OPTIMIZER_MAP[optimizer_name](learning_rate=lr)

# 记录权重轨迹
weight_trajectory = [dense_layer.weights.copy()]

# 训练循环并记录历史
history: dict[str, list[float]] = {"loss": [], "accuracy": []}

for _ in range(epochs):
    # 前向
    y_pred = model.forward(X, training=True)
    loss = loss_fn.forward(y_pred, y)
    acc = float(np.mean((y_pred >= 0.5).astype(float) == y))

    history["loss"].append(loss)
    history["accuracy"].append(acc)

    # 反向
    dloss = loss_fn.backward()
    model.backward(dloss)

    # 优化更新
    optimizer.step(model.layers)
    weight_trajectory.append(dense_layer.weights.copy())

final_loss = history["loss"][-1] if history["loss"] else 0.0
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
w_final = dense_layer.weights.ravel()
b_final = float(dense_layer.biases.ravel()[0])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="metric-grid">
        {render_metric_card("最终损失 (Loss)", f"{final_loss:.4f}", delta="收敛状态", delta_type="positive" if final_loss < 0.2 else "neutral", icon="📉")}
        {render_metric_card("分类准确率 (Acc)", f"{final_acc:.1%}", delta="+100%" if final_acc >= 0.95 else "+提升中", delta_type="positive" if final_acc >= 0.9 else "neutral", icon="🎯")}
        {render_metric_card("权重向量 [w₁, w₂]", f"[{w_final[0]:.2f}, {w_final[1]:.2f}]", delta=f"偏置 b = {b_final:.2f}", delta_type="neutral", icon="⚖️")}
        {render_metric_card("分界面方程", f"{w_final[0]:.2f}x₁ + {w_final[1]:.2f}x₂ + {b_final:.2f} = 0", delta="超平面", delta_type="neutral", icon="📐")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 可视化布局
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    fig_boundary = plot_decision_boundary(
        model, X, y, title=f"🗺️ 决策边界 · {activation_name} 激活空间"
    )
    st.plotly_chart(fig_boundary, use_container_width=True)

with col_right:
    fig_loss = plot_loss_curve(history, title="📈 训练损失与准确率演进")
    st.plotly_chart(fig_loss, use_container_width=True)

# 底部权重轨迹图
st.markdown("### 🌀 权重参数空间寻优轨迹")
st.caption("展示参数 $(w_1, w_2)$ 从随机初始位置沿损失梯度向最优解移动的完整流线：")
fig_traj = plot_weight_trajectory(weight_trajectory, title="参数空间 (w₁, w₂) 梯度下降路径")
st.plotly_chart(fig_traj, use_container_width=True)
