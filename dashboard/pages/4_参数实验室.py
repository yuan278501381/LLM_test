# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 4: 参数实验室 — 全参数交互实验

所有超参数均可自由调节的终极实验台，支持:
    - 全参数控制
    - A/B 对比模式
    - 逐步训练（单步 epoch）
    - 四宫格全维度监控
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
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_network_params,
    render_regularization_params,
    render_training_params,
)
from dashboard.utils.state import OPTIMIZER_MAP, build_model
from datasets.generators import make_circles, make_moons, make_spiral, make_xor
from nn_core.callbacks import ExperimentLogger, TrainingHistory
from nn_core.losses import BinaryCrossEntropy
from nn_core.tensor import set_seed

st.set_page_config(page_title="🔬 参数实验室", layout="wide")
st.title("🔬 里程碑 4: 参数实验室")
st.markdown("**全参数交互实验台** — 自由探索每个超参数对训练结果的影响。")

# ---------------------------------------------------------------------------
# 侧边栏 — 全参数控制
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ 完整参数控制")

dataset_name, n_samples, noise, seed = render_dataset_selector(key_prefix="m4_")
net_params = render_network_params(allow_multi_layer=True, key_prefix="m4_")
train_params = render_training_params(key_prefix="m4_")
reg_params = render_regularization_params(key_prefix="m4_")

# 训练模式选择
st.sidebar.subheader("🎮 训练模式")
train_mode = st.sidebar.radio(
    "模式",
    ["🚀 完整训练", "🔄 逐步训练 (每次 1 epoch)"],
    key="m4_mode",
)

# 按钮
if train_mode == "🚀 完整训练":
    train_btn = st.sidebar.button("🚀 开始训练", type="primary", use_container_width=True, key="m4_train")
    step_btn = False
else:
    train_btn = False
    col_a, col_b = st.sidebar.columns(2)
    step_btn = col_a.button("⏩ 单步", key="m4_step")
    reset_btn = col_b.button("🔄 重置", key="m4_reset")
    if reset_btn:
        for k in list(st.session_state.keys()):
            if k.startswith("m4_state_"):
                del st.session_state[k]
        st.rerun()

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
dataset_fn = {"moons": make_moons, "circles": make_circles, "xor": make_xor, "spiral": make_spiral}


def _run_training(full_train: bool = True):
    """执行训练（完整或单步）"""

    if full_train or "m4_state_model" not in st.session_state:
        # 初始化模型
        set_seed(seed)
        X, y = dataset_fn.get(dataset_name, make_moons)(
            n_samples=n_samples, noise=noise, random_state=seed,
        )

        set_seed(seed)
        model = build_model(
            n_inputs=2, n_outputs=1,
            hidden_layers=net_params["neurons_per_layer"],
            activation=net_params["activation"],
            initializer=net_params["initializer"],
            regularizer_type=reg_params["type"],
            regularizer_strength=reg_params["strength"],
            dropout_rate=reg_params["dropout_rate"],
            output_activation="Sigmoid",
        )

        opt_cls = OPTIMIZER_MAP[train_params["optimizer"]]
        optimizer = opt_cls(learning_rate=train_params["learning_rate"])
        loss_fn = BinaryCrossEntropy()

        if full_train:
            # 完整训练
            history_cb = TrainingHistory(
                snapshot_interval=max(1, train_params["epochs"] // 30)
            )

            with st.spinner("🔄 训练中..."):
                t0 = time.perf_counter()
                history = model.train(
                    X, y, epochs=train_params["epochs"],
                    batch_size=train_params["batch_size"],
                    loss_fn=loss_fn, optimizer=optimizer,
                    callbacks=[history_cb],
                )
                elapsed = time.perf_counter() - t0

            return model, history, history_cb, X, y, elapsed
        else:
            # 保存状态到 session_state
            st.session_state["m4_state_model"] = model
            st.session_state["m4_state_optimizer"] = optimizer
            st.session_state["m4_state_loss_fn"] = loss_fn
            st.session_state["m4_state_X"] = X
            st.session_state["m4_state_y"] = y
            st.session_state["m4_state_history"] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
            st.session_state["m4_state_snapshots"] = []
            st.session_state["m4_state_epoch"] = 0

    # 单步训练
    model = st.session_state["m4_state_model"]
    optimizer = st.session_state["m4_state_optimizer"]
    loss_fn = st.session_state["m4_state_loss_fn"]
    X = st.session_state["m4_state_X"]
    y = st.session_state["m4_state_y"]
    history = st.session_state["m4_state_history"]
    epoch = st.session_state["m4_state_epoch"]

    # 执行 1 个 epoch
    h = model.train(
        X, y, epochs=1, batch_size=train_params["batch_size"],
        loss_fn=loss_fn, optimizer=optimizer,
    )
    history["loss"].extend(h["loss"])
    history["accuracy"].extend(h["accuracy"])
    st.session_state["m4_state_epoch"] = epoch + 1

    # 保存快照
    snapshot = model.get_snapshot()
    snapshot["epoch"] = epoch + 1
    st.session_state["m4_state_snapshots"].append(snapshot)

    return model, history, None, X, y, 0


def _render_results(model, history, history_cb, X, y, elapsed):
    """渲染可视化结果"""

    # 指标卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最终 Loss", f"{history['loss'][-1]:.4f}")
    m2.metric("最终 Accuracy", f"{history['accuracy'][-1]:.1%}")
    m3.metric("已训练 Epoch", f"{len(history['loss'])}")
    if elapsed > 0:
        m4.metric("训练耗时", f"{elapsed:.2f}s")
    else:
        m4.metric("当前 Epoch", f"{st.session_state.get('m4_state_epoch', 0)}")

    # 四宫格布局
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_loss_curve(history, title="📈 训练曲线"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            plot_decision_boundary(model, X, y, title="🗺️ 决策边界"),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    snapshot = model.get_snapshot()

    with col3:
        st.plotly_chart(
            plot_weight_histograms(snapshot, title="📊 权重分布"),
            use_container_width=True,
        )

    with col4:
        st.plotly_chart(
            plot_gradient_histograms(snapshot, title="🔥 梯度分布"),
            use_container_width=True,
        )

    # 激活热力图
    act_indices = [
        i for i, info in enumerate(snapshot["layers"])
        if info["type"] in ("ReLU", "Sigmoid", "Tanh", "LeakyReLU")
        and info["output"] is not None
    ]
    if act_indices:
        st.subheader("⚡ 逐层激活值")
        act_cols = st.columns(min(len(act_indices), 3))
        for i, idx in enumerate(act_indices[:3]):
            with act_cols[i]:
                st.plotly_chart(
                    plot_activation_heatmap(snapshot, layer_idx=idx),
                    use_container_width=True,
                )

    # 实验日志保存
    st.sidebar.markdown("---")
    if st.sidebar.button("📋 保存实验日志", key="m4_save"):
        logger = ExperimentLogger(log_dir="logs/")
        params = {
            "dataset": dataset_name,
            "n_samples": n_samples,
            "noise": noise,
            "hidden_layers": net_params["neurons_per_layer"],
            "activation": net_params["activation"],
            "initializer": net_params["initializer"],
            "optimizer": train_params["optimizer"],
            "learning_rate": train_params["learning_rate"],
            "epochs": len(history["loss"]),
            "batch_size": train_params["batch_size"],
            "regularization": reg_params["type"],
            "dropout": reg_params["dropout_rate"],
        }
        results = {
            "final_loss": history["loss"][-1],
            "final_accuracy": history["accuracy"][-1],
        }
        filepath = logger.log_experiment(params, results)
        st.sidebar.success(f"✅ 已保存: {filepath}")


# ---------------------------------------------------------------------------
# 执行逻辑
# ---------------------------------------------------------------------------
if train_btn:
    result = _run_training(full_train=True)
    _render_results(*result)
elif step_btn:
    result = _run_training(full_train=False)
    _render_results(*result)
elif "m4_state_model" in st.session_state:
    # 已有模型状态，直接渲染
    model = st.session_state["m4_state_model"]
    history = st.session_state["m4_state_history"]
    X = st.session_state["m4_state_X"]
    y = st.session_state["m4_state_y"]
    if history["loss"]:
        _render_results(model, history, None, X, y, 0)
    else:
        st.info("👈 点击 **「⏩ 单步」** 开始逐步训练。")
else:
    st.info("👈 调整左侧全部参数后，点击 **「🚀 开始训练」** 或切换到逐步模式。")
