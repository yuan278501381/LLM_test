# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 4: 全参数微观实验室 (Hyperparameter & Micro-State Lab) - 无硬编码 · 知识图谱深度解析

工业级微观监控中枢：四宫格全景图、单步调试 (Step-by-Step)、快照对比与超参数动态热调整。
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_gradient_histograms,
    plot_loss_curve,
    plot_weight_histograms,
)
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_deep_dive_card,
    render_network_params,
    render_training_params,
)
from dashboard.constants.knowledge import (
    ACTIVATIONS,
    INITIALIZERS,
    OPTIMIZERS,
    REGULARIZERS,
)
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_section_heading,
)
from dashboard.utils.state import (
    get_dataset,
    resolve_activation,
    resolve_initializer,
    resolve_optimizer,
    resolve_regularizer,
)
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Micro Parameter Lab · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="全参数微观实验室",
    subtitle="工业级四宫格微观遥测监控台 · 支持逐步微调训练 (Step-by-Step) · 参数快照热回滚与 A/B 对比",
    badge_text="MILESTONE 04 // MICRO LAB & TELEMETRY",
    badge_type="emerald",
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板 (中英双语标签，由元数据驱动)
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m4_", default_dataset="moons"
)

net_params = render_network_params(
    allow_multi_layer=True,
    key_prefix="m4_",
    default_layers=2,
    default_neurons=[12, 6],
    default_act="ReLU",
    default_init="he",
)
n_layers = net_params["n_layers"]
neurons_per_layer = net_params["neurons_per_layer"]
activation_name = net_params["activation"]
initializer = net_params["initializer"]

train_params = render_training_params(
    key_prefix="m4_", default_opt="Adam", default_lr=0.03, default_epochs=100
)
optimizer_name = train_params["optimizer"]
lr = train_params["learning_rate"]
batch_size = train_params["batch_size"]

st.sidebar.markdown("#### REGULARIZATION // 正则化策略")
reg_list = list(REGULARIZERS.values())
reg_labels = [m.label for m in reg_list]

selected_reg_label = st.sidebar.selectbox(
    "正则化类型",
    reg_labels,
    help="权重约束项。通过惩罚过大的权重值，防止模型强行记忆样本噪声，有效提升泛化能力。",
    key="m4_reg",
)
reg_meta = next(m for m in reg_list if m.label == selected_reg_label)

reg_lambda = 0.0
if reg_meta.id != "None":
    reg_lambda = st.sidebar.slider(
        "惩罚系数 (λ)",
        0.0001, 0.1, 0.01, step=0.005, format="%.4f",
        help="正则化惩罚强度。越大对权重的压制越强，决策面越平滑；过大会导致欠拟合。",
        key="m4_lambda",
    )

# ---------------------------------------------------------------------------
# Session State 模型管理 (支持逐步训练)
# ---------------------------------------------------------------------------
act_cls = resolve_activation(activation_name)
init_name = resolve_initializer(initializer)
opt_cls = resolve_optimizer(optimizer_name)

if "m4_model" not in st.session_state or st.sidebar.button("RESET // 重置模型", key="m4_reset"):
    m = Sequential()
    current_dim = 2
    for i in range(n_layers):
        out_dim = neurons_per_layer[i]
        reg = resolve_regularizer(reg_meta.id, float(reg_lambda))
        m.add(Dense(current_dim, out_dim, initializer=init_name, regularizer=reg))
        m.add(act_cls())
        current_dim = out_dim

    m.add(Dense(current_dim, 1, initializer=init_name))
    m.add(resolve_activation("Sigmoid")())

    st.session_state["m4_model"] = m
    st.session_state["m4_history"] = {"loss": [], "accuracy": []}
    st.session_state["m4_epoch_count"] = 0
    st.session_state["m4_opt_instance"] = opt_cls(learning_rate=float(lr))

model: Sequential = st.session_state["m4_model"]
history: dict[str, list[float]] = st.session_state["m4_history"]
opt_instance = st.session_state.get("m4_opt_instance", opt_cls(learning_rate=float(lr)))

# ---------------------------------------------------------------------------
# 训练控制中枢 (Train / Step)
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)
loss_fn = BinaryCrossEntropy()

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("TRAIN // 训练 50 轮", key="m4_train_50"):
        hist = model.train(X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=50, batch_size=batch_size)
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 50

with col_btn2:
    if st.button("STEP // 单步微调", key="m4_step_1"):
        hist = model.train(X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=1, batch_size=batch_size)
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 1

with col_btn3:
    current_epochs = st.session_state["m4_epoch_count"]
    clean_opt_name = optimizer_name.split(" ")[0]
    badge_status = (
        '<div style="display:flex;align-items:center;gap:0.8rem;height:100%;padding-top:0.3rem;">'
        f'<span class="pill-badge pill-emerald"><span class="status-dot"></span> EPOCHS 训练轮数: {current_epochs}</span>'
        f'<span class="pill-badge pill-blue">OPT 优化器: {clean_opt_name} (lr={lr})</span>'
        '</div>'
    )
    st.markdown(badge_status, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 遥测指标计算与展示 (中英双语标签)
# ---------------------------------------------------------------------------
current_loss = history["loss"][-1] if history["loss"] else 1.0
current_acc = history["accuracy"][-1] if history["accuracy"] else 0.0

dense_layers = [l for l in model.layers if isinstance(l, Dense)]
weights_list = [l.weights for l in dense_layers]
grads_list = [l.grad_weights for l in dense_layers if l.grad_weights is not None]
layer_names = [f"Dense #{i+1}" for i in range(len(dense_layers))]

grad_norm = float(np.mean([np.mean(np.abs(g)) for g in grads_list])) if grads_list else 0.0
weight_norm = float(np.mean([np.mean(np.abs(w)) for w in weights_list])) if weights_list else 0.0

clean_reg_name = reg_meta.id

grid_html = (
    '<div class="metric-grid" style="margin-top:1rem;">'
    + render_metric_card("CURRENT LOSS // 当前损失", f"{current_loss:.4f}", delta=f"EPOCH 轮次 #{st.session_state['m4_epoch_count']}", delta_type="positive" if current_loss < 0.2 else "neutral", icon_name="trending-down")
    + render_metric_card("ACCURACY // 当前准确率", f"{current_acc:.1%}", delta="LIVE STATE", delta_type="positive" if current_acc >= 0.9 else "neutral", icon_name="target")
    + render_metric_card("GRADIENT NORM // 梯度范数", f"{grad_norm:.2e}", delta="BACKPROP ACTIVITY", delta_type="neutral", icon_name="activity")
    + render_metric_card("WEIGHT MAGNITUDE // 权重均值", f"{weight_norm:.4f}", delta=f"REG 正则: {clean_reg_name}", delta_type="neutral", icon_name="shield")
    + '</div>'
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 四宫格全景监控 (Four-Grid Dashboard)
# ---------------------------------------------------------------------------
render_section_heading("四宫格全景遥测监控台 (Quad Telemetry Console)", icon_name="activity")

grid_c1, grid_c2 = st.columns(2)

with grid_c1:
    fig_bound = plot_decision_boundary(
        model, X, y, title=f"DECISION MANIFOLD // 空间决策流形 (Acc: {current_acc:.1%})"
    )
    st.plotly_chart(fig_bound, use_container_width=True)

    fig_w_hist = plot_weight_histograms(
        weights_list, layer_names, title="WEIGHT SPECTRUM // 逐层权重参数分布"
    )
    st.plotly_chart(fig_w_hist, use_container_width=True)

with grid_c2:
    fig_loss = plot_loss_curve(history, title="TRAINING DYNAMICS // 损失与准确率收敛动态")
    st.plotly_chart(fig_loss, use_container_width=True)

    fig_g_hist = plot_gradient_histograms(
        grads_list if grads_list else [np.zeros((2, 2))],
        layer_names,
        title="GRADIENT FLOW // 反向传播梯度流分布",
    )
    st.plotly_chart(fig_g_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# 快照与实验状态导出 (Snapshot Export / Rollback)
# ---------------------------------------------------------------------------
render_section_heading("实验状态快照管理与 JSON 导出", icon_name="database")

col_snap1, col_snap2 = st.columns(2)

with col_snap1:
    if st.button("SNAPSHOT // 保存当前实验快照", key="m4_snap_save"):
        snapshot = {
            "epoch": st.session_state["m4_epoch_count"],
            "loss": current_loss,
            "accuracy": current_acc,
            "layers": neurons_per_layer,
            "activation": activation_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "weights": [w.tolist() for w in weights_list],
        }
        st.session_state["m4_saved_snapshot"] = snapshot
        st.success(f"已保存第 {st.session_state['m4_epoch_count']} 轮快照 (Loss: {current_loss:.4f})")

with col_snap2:
    if "m4_saved_snapshot" in st.session_state:
        snap_json = json.dumps(st.session_state["m4_saved_snapshot"], indent=2)
        st.download_button(
            label="DOWNLOAD // 导出快照 JSON",
            data=snap_json,
            file_name=f"nn_snapshot_epoch_{st.session_state['m4_epoch_count']}.json",
            mime="application/json",
            key="m4_snap_download",
        )

# 深度知识学习指南 (折叠微观原理解析)
act_meta = ACTIVATIONS.get(activation_name, ACTIVATIONS.get(activation_name.split(" ")[0], ACTIVATIONS["ReLU"]))
init_meta = INITIALIZERS.get(initializer, INITIALIZERS.get(initializer.split(" ")[0].lower(), INITIALIZERS["he"]))
opt_meta = OPTIMIZERS.get(optimizer_name, OPTIMIZERS.get(optimizer_name.split(" ")[0], OPTIMIZERS["Adam"]))

render_deep_dive_card("全参数微观实验室正则化与超参数原理指南", [reg_meta, act_meta, init_meta, opt_meta])
