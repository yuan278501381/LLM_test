# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 2: 多层网络 — 理解深度的力量

演示多层堆叠如何解决单神经元无法处理的非线性问题，
以及权重初始化、梯度消失等深层网络的关键挑战。
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from dashboard.components.charts import (
    plot_activation_heatmap,
    plot_decision_boundary,
    plot_gradient_histograms,
    plot_loss_curve,
    plot_weight_histograms,
)
from dashboard.components.network_viz import plot_network_topology
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_network_params,
    render_training_params,
)
from dashboard.utils.state import OPTIMIZER_MAP, build_model
from datasets.generators import make_circles, make_moons, make_spiral, make_xor
from nn_core.callbacks import TrainingHistory
from nn_core.tensor import set_seed

st.set_page_config(page_title="🧱 多层网络", layout="wide")
st.title("🧱 里程碑 2: 多层网络")
st.markdown("**理解深度的力量** — 多层堆叠如何拟合复杂的非线性决策边界。")

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ 参数控制")
dataset_name, n_samples, noise, seed = render_dataset_selector(key_prefix="m2_")
net_params = render_network_params(allow_multi_layer=True, key_prefix="m2_")
train_params = render_training_params(key_prefix="m2_")

train_btn = st.sidebar.button("🚀 开始训练", type="primary", use_container_width=True, key="m2_train")

# ---------------------------------------------------------------------------
# 生成数据
# ---------------------------------------------------------------------------
set_seed(seed)
dataset_fn = {"moons": make_moons, "circles": make_circles, "xor": make_xor, "spiral": make_spiral}
X, y = dataset_fn.get(dataset_name, make_moons)(n_samples=n_samples, noise=noise, random_state=seed)

if train_btn:
    set_seed(seed)

    # 构建模型
    model = build_model(
        n_inputs=2, n_outputs=1,
        hidden_layers=net_params["neurons_per_layer"],
        activation=net_params["activation"],
        initializer=net_params["initializer"],
        output_activation="Sigmoid",
    )

    # 优化器 & 损失函数
    opt_cls = OPTIMIZER_MAP[train_params["optimizer"]]
    optimizer = opt_cls(learning_rate=train_params["learning_rate"])
    from nn_core.losses import BinaryCrossEntropy
    loss_fn = BinaryCrossEntropy()

    history_cb = TrainingHistory(snapshot_interval=max(1, train_params["epochs"] // 30))

    with st.spinner("🔄 训练中..."):
        t0 = time.perf_counter()
        history = model.train(
            X, y, epochs=train_params["epochs"],
            batch_size=train_params["batch_size"],
            loss_fn=loss_fn, optimizer=optimizer,
            callbacks=[history_cb],
        )
        elapsed = time.perf_counter() - t0

    # 指标
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最终 Loss", f"{history['loss'][-1]:.4f}")
    m2.metric("最终 Accuracy", f"{history['accuracy'][-1]:.1%}")
    m3.metric("网络深度", f"{net_params['n_layers']} 层")
    m4.metric("训练耗时", f"{elapsed:.2f}s")

    # ---- 网络结构 + 决策边界 ----
    col1, col2 = st.columns(2)

    with col1:
        layer_sizes = [2] + net_params["neurons_per_layer"] + [1]
        # 提取权重用于可视化
        dense_weights = [
            info["weights"] for info in model.get_snapshot()["layers"]
            if info["type"] == "Dense" and info["weights"] is not None
        ]
        st.plotly_chart(
            plot_network_topology(layer_sizes, weights=dense_weights, title="🧬 网络结构"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            plot_decision_boundary(model, X, y, title="🗺️ 决策边界"),
            use_container_width=True,
        )

    # ---- Loss + 权重分布 ----
    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(
            plot_loss_curve(history, title="📈 训练曲线"),
            use_container_width=True,
        )

    with col4:
        snapshot = model.get_snapshot()
        st.plotly_chart(
            plot_weight_histograms(snapshot, title="📊 权重分布"),
            use_container_width=True,
        )

    # ---- 梯度分布 + 激活热力图 ----
    col5, col6 = st.columns(2)

    with col5:
        st.plotly_chart(
            plot_gradient_histograms(snapshot, title="🔥 梯度分布"),
            use_container_width=True,
        )

    with col6:
        # 找到第一个激活函数层的索引
        act_indices = [
            i for i, info in enumerate(snapshot["layers"])
            if info["type"] in ("ReLU", "Sigmoid", "Tanh", "LeakyReLU")
            and info["output"] is not None
        ]
        if act_indices:
            st.plotly_chart(
                plot_activation_heatmap(snapshot, layer_idx=act_indices[0], title="⚡ 激活值热力图"),
                use_container_width=True,
            )

    # 概念说明
    with st.expander("💡 关键观察", expanded=False):
        st.markdown("""
        1. **层数与拟合能力**: 单层网络只能画直线；2层能画曲线；3+层能拟合任意复杂边界
        2. **权重初始化的影响**:
           - `zeros`: 所有神经元学到相同特征（对称性问题），训练失败
           - `xavier`: 适配 Sigmoid/Tanh，保持信号方差稳定
           - `he`: 适配 ReLU，补偿一半神经元被置零
        3. **梯度直方图**: 如果深层的梯度趋近于 0 → 梯度消失；趋近于极大值 → 梯度爆炸
        4. **激活热力图**: 观察每层输出的分布是否健康（不全为0，不全饱和）
        """)
else:
    st.info("👈 调整参数后，点击 **「🚀 开始训练」** 开始实验。")
