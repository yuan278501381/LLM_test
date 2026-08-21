# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 1: 单神经元感知器 — 理解最小计算单元

本页面使用一个最简单的网络（Dense(2,1) + Sigmoid）来演示:
    - 前向传播的计算过程
    - 损失函数如何衡量预测误差
    - 反向传播如何计算梯度
    - 参数更新如何改善预测

通过拖动学习率滑块，直观感受学习率对收敛的影响。
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_loss_curve,
    plot_weight_trajectory,
)
from datasets.generators import make_circles, make_moons, make_xor
from nn_core.activations import LeakyReLU, ReLU, Sigmoid, Tanh
from nn_core.callbacks import TrainingHistory
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential
from nn_core.optimizers import SGD
from nn_core.tensor import set_seed

st.set_page_config(page_title="🎯 单神经元感知器", layout="wide")
st.title("🎯 里程碑 1: 单神经元感知器")
st.markdown("**理解最小计算单元** — 一个神经元如何学会在二维空间中画一条分割线。")

# ---------------------------------------------------------------------------
# 侧边栏参数
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ 参数控制")

# 数据集
st.sidebar.subheader("📊 数据集")
dataset_choice = st.sidebar.selectbox(
    "选择数据集",
    ["🌙 Moons", "⭕ Circles", "❌ XOR"],
    key="m1_dataset",
)

n_samples = st.sidebar.slider("样本数量", 50, 500, 200, step=50, key="m1_n")
noise = st.sidebar.slider("噪声强度", 0.0, 0.5, 0.1, step=0.01, key="m1_noise")
seed = st.sidebar.number_input("随机种子", 0, 9999, 42, key="m1_seed")

# 超参数
st.sidebar.subheader("⚙️ 训练参数")
activation_name = st.sidebar.selectbox(
    "激活函数", ["Sigmoid", "Tanh", "ReLU", "LeakyReLU"], key="m1_act"
)
learning_rate = st.sidebar.select_slider(
    "学习率",
    options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    value=0.1, key="m1_lr",
)
epochs = st.sidebar.slider("训练轮数", 10, 1000, 200, step=10, key="m1_epochs")

# ---------------------------------------------------------------------------
# 生成数据
# ---------------------------------------------------------------------------
set_seed(seed)

dataset_map = {
    "🌙 Moons": make_moons,
    "⭕ Circles": make_circles,
    "❌ XOR": make_xor,
}
X, y = dataset_map[dataset_choice](n_samples=n_samples, noise=noise, random_state=seed)

# ---------------------------------------------------------------------------
# 训练按钮
# ---------------------------------------------------------------------------
train_btn = st.sidebar.button("🚀 开始训练", type="primary", use_container_width=True)

if train_btn:
    set_seed(seed)

    # 构建单神经元模型
    act_map = {"Sigmoid": Sigmoid, "Tanh": Tanh, "ReLU": ReLU, "LeakyReLU": LeakyReLU}
    model = Sequential()
    model.add(Dense(2, 1, initializer="xavier"))
    model.add(act_map[activation_name]())

    loss_fn = BinaryCrossEntropy()
    optimizer = SGD(learning_rate=learning_rate)
    history_cb = TrainingHistory(snapshot_interval=max(1, epochs // 50))

    # 训练
    with st.spinner("🔄 训练中..."):
        t0 = time.perf_counter()
        history = model.train(
            X, y, epochs=epochs, batch_size=n_samples,
            loss_fn=loss_fn, optimizer=optimizer,
            callbacks=[history_cb],
        )
        elapsed = time.perf_counter() - t0

    # ---- 指标卡片 ----
    m1, m2, m3 = st.columns(3)
    m1.metric("最终 Loss", f"{history['loss'][-1]:.4f}")
    m2.metric("最终 Accuracy", f"{history['accuracy'][-1]:.1%}")
    m3.metric("训练耗时", f"{elapsed:.2f}s")

    # ---- 可视化 (2x2 布局) ----
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_decision_boundary(model, X, y, title="🗺️ 决策边界"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            plot_loss_curve(history, title="📈 训练曲线"),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        if history_cb.snapshots:
            st.plotly_chart(
                plot_weight_trajectory(history_cb.snapshots, layer_idx=0, title="🛤️ 权重变化轨迹"),
                use_container_width=True,
            )

    with col4, st.expander("💡 概念说明", expanded=True):
        st.markdown(f"""
            **当前配置**:
            - 网络: Dense(2→1) + {activation_name}
            - 学习率: {learning_rate}
            - 优化器: SGD

            **关键观察**:
            1. **决策边界**: 单神经元只能画一条「线」来分割空间。
               对于 Moons/XOR 这类非线性问题，它无法完美分类。
            2. **学习率的影响**:
               - 太小 (0.001): Loss 下降极慢
               - 合适 (0.01-0.1): 平稳收敛
               - 太大 (1.0): Loss 震荡甚至发散
            3. **这就是为什么我们需要多层网络** → 前往里程碑 2
            """)
else:
    st.info("👈 调整左侧参数后，点击 **「🚀 开始训练」** 按钮开始实验。")
