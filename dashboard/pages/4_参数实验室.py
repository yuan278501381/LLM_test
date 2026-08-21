# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
🔬 里程碑 4: 全参数微观实验室 (Parameter Laboratory)

全维度交互式参数调控台：支持逐步微调训练 (Step-by-Step)、四宫格全景监控与快照导出。
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
    render_network_params,
    render_training_params,
)
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
)
from dashboard.utils.state import ACTIVATION_MAP, OPTIMIZER_MAP, get_dataset
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential
from nn_core.regularizers import L1, L2

st.set_page_config(
    page_title="全参数微观实验室 · NN Playground",
    page_icon="🔬",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="🔬 全参数微观实验室",
    subtitle="工业级四宫格全景监控台 · 逐步微调训练 · 正则化与防过拟合深度诊断 · 实验快照导出",
    badge_text="MILESTONE 4 · TELEMETRY WORKBENCH",
    badge_type="emerald",
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m4_", default_dataset="🌙 Moons"
)

net_params = render_network_params(
    allow_multi_layer=True, key_prefix="m4_", default_layers=2, default_neurons=[12, 6]
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

# 正则化
st.sidebar.markdown("### 🛡️ 正则化与防过拟合")
reg_type = st.sidebar.selectbox("正则化类型", ["None", "L2", "L1"], key="m4_reg")
reg_lambda = 0.0
if reg_type != "None":
    reg_lambda = st.sidebar.slider("正则化强度 (λ)", 0.0001, 0.1, 0.01, step=0.005, format="%.4f", key="m4_lambda")

# ---------------------------------------------------------------------------
# Session State 模型管理 (支持逐步训练)
# ---------------------------------------------------------------------------
if "m4_model" not in st.session_state or st.sidebar.button("🔄 重置实验模型", key="m4_reset"):
    # 构建新模型
    m = Sequential()
    current_dim = 2
    for i in range(n_layers):
        out_dim = neurons_per_layer[i]
        reg = L2(reg_lambda) if reg_type == "L2" else (L1(reg_lambda) if reg_type == "L1" else None)
        m.add(Dense(current_dim, out_dim, initializer=initializer, regularizer=reg))
        m.add(ACTIVATION_MAP[activation_name]())
        current_dim = out_dim

    m.add(Dense(current_dim, 1, initializer=initializer))
    m.add(ACTIVATION_MAP["Sigmoid"]())

    st.session_state["m4_model"] = m
    st.session_state["m4_history"] = {"loss": [], "accuracy": []}
    st.session_state["m4_epoch_count"] = 0
    st.session_state["m4_optimizer"] = OPTIMIZER_MAP[optimizer_name](learning_rate=lr)

model: Sequential = st.session_state["m4_model"]
history: dict[str, list[float]] = st.session_state["m4_history"]
opt_instance = st.session_state["m4_optimizer"]

# ---------------------------------------------------------------------------
# 训练控制中枢 (Play / Step / Train)
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)
loss_fn = BinaryCrossEntropy()

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("▶️ 训练 50 Epochs", key="m4_train_50"):
        hist = model.train(X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=50, batch_size=batch_size, verbose=False)
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 50

with col_btn2:
    if st.button("🦶 单步微调 (Step 1)", key="m4_step_1"):
        hist = model.train(X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=1, batch_size=batch_size, verbose=False)
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 1

with col_btn3:
    current_epochs = st.session_state["m4_epoch_count"]
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 0.8rem; height: 100%; padding-top: 0.3rem;">
            <span class="pill-badge pill-emerald"><span class="status-dot"></span> 累计训练轮数: {current_epochs}</span>
            <span class="pill-badge pill-blue">优化器: {optimizer_name} (lr={lr})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 遥测指标计算与展示
# ---------------------------------------------------------------------------
current_loss = history["loss"][-1] if history["loss"] else 1.0
current_acc = history["accuracy"][-1] if history["accuracy"] else 0.0

dense_layers = [l for l in model.layers if isinstance(l, Dense)]
weights_list = [l.weights for l in dense_layers]
grads_list = [l.grad_weights for l in dense_layers if l.grad_weights is not None]
layer_names = [f"Dense #{i+1}" for i in range(len(dense_layers))]

grad_norm = float(np.mean([np.mean(np.abs(g)) for g in grads_list])) if grads_list else 0.0
weight_norm = float(np.mean([np.mean(np.abs(w)) for w in weights_list])) if weights_list else 0.0

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 1rem;">
        {render_metric_card("当前 Loss", f"{current_loss:.4f}", delta=f"Epoch #{st.session_state['m4_epoch_count']}", delta_type="positive" if current_loss < 0.2 else "neutral", icon="📉")}
        {render_metric_card("当前准确率", f"{current_acc:.1%}", delta="实时模型状态", delta_type="positive" if current_acc >= 0.9 else "neutral", icon="🎯")}
        {render_metric_card("平均梯度范数", f"{grad_norm:.2e}", delta="反向传播活性", delta_type="neutral", icon="🌊")}
        {render_metric_card("平均权重幅度", f"{weight_norm:.4f}", delta=f"正则化: {reg_type}", delta_type="neutral", icon="⚖️")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 四宫格全景监控 (Four-Grid Dashboard)
# ---------------------------------------------------------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig_boundary = plot_decision_boundary(model, X, y, title="🗺️ [象限 1] 实时空间决策流形")
    st.plotly_chart(fig_boundary, use_container_width=True)

with row1_col2:
    fig_loss = plot_loss_curve(history, title="📈 [象限 2] 全程损失与准确率收敛")
    st.plotly_chart(fig_loss, use_container_width=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    fig_w = plot_weight_histograms(weights_list, layer_names, title="⚖️ [象限 3] 各层权重参数分布")
    st.plotly_chart(fig_w, use_container_width=True)

with row2_col2:
    if grads_list:
        fig_g = plot_gradient_histograms(grads_list, layer_names, title="🌊 [象限 4] 各层反向传播梯度分布")
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("💡 运行至少 1 次训练后即可捕获反向传播梯度流分布。")

# ---------------------------------------------------------------------------
# 实验快照与导出
# ---------------------------------------------------------------------------
with st.expander("📦 查看与导出实验快照 (JSON)"):
    snapshot = {
        "dataset": dataset_name,
        "n_samples": n_samples,
        "noise": noise,
        "architecture": {
            "n_layers": n_layers,
            "neurons_per_layer": neurons_per_layer,
            "activation": activation_name,
            "initializer": initializer,
            "regularizer": reg_type,
            "lambda": reg_lambda,
        },
        "training": {
            "optimizer": optimizer_name,
            "lr": lr,
            "epochs_trained": st.session_state["m4_epoch_count"],
            "final_loss": current_loss,
            "final_accuracy": current_acc,
        },
    }
    st.json(snapshot)
    st.download_button(
        "📥 下载当前实验元数据 JSON",
        data=json.dumps(snapshot, indent=2, ensure_ascii=False),
        file_name=f"nn_experiment_{dataset_name}_ep{st.session_state['m4_epoch_count']}.json",
        mime="application/json",
    )
