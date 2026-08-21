# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 2: 多层网络与活性探针 (Deep Networks & Neuron Probe)

理解「深度」的特征流形扭曲力量：多层链式法则、神经元动态激活探针、梯度消失/爆炸诊断。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.components.charts import (
    plot_activation_heatmap,
    plot_decision_boundary,
    plot_gradient_histograms,
)
from dashboard.components.network_viz import plot_network_topology
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_network_params,
    render_presets_selector,
    render_probe_point_selector,
    render_training_params,
)
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_preset_badge,
)
from dashboard.utils.state import ACTIVATION_MAP, OPTIMIZER_MAP, get_dataset
from nn_core.activations import Activation
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Deep Topology & Probe · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="多层网络与动态活性探针",
    subtitle="探索深度非线性流形折叠 · 实时探针微观捕获信号在神经元间的点亮与流动 · 梯度健康诊断",
    badge_text="MILESTONE 02 // DEEP TOPOLOGY & PROBE",
    badge_type="purple",
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板 (含预设)
# ---------------------------------------------------------------------------
preset = render_presets_selector(key_prefix="m2_")

if preset:
    render_preset_badge(
        "已激活预设",
        f"数据集: {preset['dataset']} | 深度: {preset['n_layers']} 层 | 激活: {preset['activation']} | 优化器: {preset['optimizer']}",
    )
    dataset_name = preset["dataset"].split(" ")[0].lower()
    n_samples = preset["n_samples"]
    noise = preset["noise"]
    random_state = 42
    n_layers = preset["n_layers"]
    neurons_per_layer = preset["neurons"]
    activation_name = preset["activation"]
    initializer = preset["initializer"]
    optimizer_name = preset["optimizer"]
    lr = preset["lr"]
    epochs = preset["epochs"]
    batch_size = None
else:
    dataset_name, n_samples, noise, random_state = render_dataset_selector(
        key_prefix="m2_", default_dataset="Moons (双月分布)"
    )
    net_params = render_network_params(
        allow_multi_layer=True, key_prefix="m2_", default_layers=2, default_neurons=[8, 4]
    )
    n_layers = net_params["n_layers"]
    neurons_per_layer = net_params["neurons_per_layer"]
    activation_name = net_params["activation"]
    initializer = net_params["initializer"]

    train_params = render_training_params(key_prefix="m2_", default_opt="Adam", default_lr=0.05, default_epochs=150)
    optimizer_name = train_params["optimizer"]
    lr = train_params["learning_rate"]
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]

# 探针坐标选择器
probe_x, probe_y = render_probe_point_selector(key_prefix="m2_")
probe_pt = (probe_x, probe_y)

# ---------------------------------------------------------------------------
# 数据与模型训练
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

# 构建多层网络
model = Sequential()
current_dim = 2
for i in range(n_layers):
    out_dim = neurons_per_layer[i]
    model.add(Dense(current_dim, out_dim, initializer=initializer))
    model.add(ACTIVATION_MAP[activation_name]())
    current_dim = out_dim

# 输出层
model.add(Dense(current_dim, 1, initializer=initializer))
model.add(ACTIVATION_MAP["Sigmoid"]())

loss_fn = BinaryCrossEntropy()
optimizer = OPTIMIZER_MAP[optimizer_name](learning_rate=lr)

# 训练网络
history = model.train(
    X, y, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, batch_size=batch_size, verbose=False
)

final_loss = history["loss"][-1] if history["loss"] else 0.0
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0

# ---------------------------------------------------------------------------
# 神经元动态活性探针 (Single-Sample Forward Telemetry)
# ---------------------------------------------------------------------------
probe_input = np.array([[probe_x, probe_y]])
probe_activations: list[np.ndarray] = [probe_input]

curr_signal = probe_input
all_activations: list[np.ndarray] = []
dense_weights: list[np.ndarray] = []
dense_grads: list[np.ndarray] = []
layer_names: list[str] = []

dense_count = 0
for layer in model.layers:
    if isinstance(layer, Dense):
        dense_count += 1
        curr_signal = layer.forward(curr_signal, training=False)
        dense_weights.append(layer.weights)
        if layer.grad_weights is not None:
            dense_grads.append(layer.grad_weights)
            layer_names.append(f"Dense #{dense_count}")
    elif isinstance(layer, Activation):
        curr_signal = layer.forward(curr_signal)
        probe_activations.append(curr_signal.copy())
        full_act = layer.forward(layer.input_cache) if hasattr(layer, "input_cache") else curr_signal
        all_activations.append(full_act)

probe_prob = float(curr_signal.ravel()[0])
probe_pred_class = 1 if probe_prob >= 0.5 else 0

layer_sizes = [2] + neurons_per_layer + [1]

min_grad_norm = min([float(np.mean(np.abs(g))) for g in dense_grads]) if dense_grads else 0.0
if min_grad_norm < 1e-4:
    grad_health_status = "VANISHING GRADIENT"
elif min_grad_norm > 50.0:
    grad_health_status = "EXPLODING GRADIENT"
else:
    grad_health_status = "HEALTHY GRADIENT"

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="metric-grid">
        {render_metric_card("FINAL LOSS", f"{final_loss:.4f}", delta="CONVERGED", delta_type="positive" if final_loss < 0.2 else "neutral", icon_name="trending-down")}
        {render_metric_card("ACCURACY", f"{final_acc:.1%}", delta=dataset_name.upper(), delta_type="positive" if final_acc >= 0.9 else "neutral", icon_name="target")}
        {render_metric_card("PROBE RESPONSE", f"{probe_prob:.1%}", delta=f"CLASS {probe_pred_class}", delta_type="positive" if probe_pred_class == 1 else "neutral", icon_name="crosshair")}
        {render_metric_card("GRADIENT NORM", f"{min_grad_norm:.2e}", delta=grad_health_status, delta_type="positive" if "HEALTHY" in grad_health_status else "negative", icon_name="activity")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 可视化布局 (双栏联动)
# ---------------------------------------------------------------------------
col_topo, col_bound = st.columns([1.1, 1])

with col_topo:
    fig_topo = plot_network_topology(
        layer_sizes=layer_sizes,
        weights=dense_weights,
        neuron_activations=probe_activations,
        title=f"TOPOLOGY & PROBE // 激活探针响应 (x₁={probe_x:.2f}, x₂={probe_y:.2f})",
    )
    st.plotly_chart(fig_topo, use_container_width=True)

with col_bound:
    fig_bound = plot_decision_boundary(
        model, X, y, probe_point=probe_pt, title="DECISION MANIFOLD // 空间决策流形与探针定位"
    )
    st.plotly_chart(fig_bound, use_container_width=True)

# ---------------------------------------------------------------------------
# 深度诊断：逐层激活热力图 & 梯度直方图
# ---------------------------------------------------------------------------
col_heat, col_grad = st.columns(2)

with col_heat:
    if all_activations:
        fig_heat = plot_activation_heatmap(all_activations, title="ACTIVATION HEATMAP // 逐层神经元激活分布")
        st.plotly_chart(fig_heat, use_container_width=True)

with col_grad:
    if dense_grads:
        fig_grad = plot_gradient_histograms(dense_grads, layer_names, title="GRADIENT FLOW // 反向传播梯度流分布")
        st.plotly_chart(fig_grad, use_container_width=True)
