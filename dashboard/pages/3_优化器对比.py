# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 3: 优化器多轨竞速对比 (Optimizer Arena) - 无硬编码 · 知识图谱深度解析

同一起跑线对比 SGD, Momentum, RMSProp 与 Adam 的收敛效率、鞍面逃逸与超参数敏感度。
"""

import copy
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_multi_loss_curves,
)
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_deep_dive_card,
    render_network_params,
)
from dashboard.constants.knowledge import OPTIMIZERS
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
)
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Optimizer Arena · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="优化器多轨竞速对比",
    subtitle="同一网络拓扑与初始权重起点，全方位对比 SGD / Momentum / RMSProp / Adam 的收敛动力学",
    badge_text="MILESTONE 03 // CONVERGENCE DYNAMICS",
    badge_type="amber",
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m3_", default_dataset="moons"
)

net_params = render_network_params(
    allow_multi_layer=True, key_prefix="m3_", default_layers=2, default_neurons=[8, 4]
)
n_layers = net_params["n_layers"]
neurons_per_layer = net_params["neurons_per_layer"]
activation_name = net_params["activation"]
initializer = net_params["initializer"]

st.sidebar.markdown("#### HYPERPARAMETERS // 竞速超参数")
col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input(
        "基准学习率 (LR)",
        0.001, 1.0, 0.03, step=0.01, format="%.3f",
        help="所有优化器共享的基准更新步长 $\\eta$。可观察不同优化器在相同学习率下的抗震荡与自适应特性。",
        key="m3_lr",
    )
with col2:
    epochs = st.slider(
        "竞速轮数 (Epochs)",
        20, 500, 150, step=10,
        help="同台竞速迭代的总轮数。",
        key="m3_epochs",
    )

# ---------------------------------------------------------------------------
# 数据与基准模型构建
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

act_cls = resolve_activation(activation_name)
init_name = resolve_initializer(initializer)

base_model = Sequential()
current_dim = 2
for i in range(n_layers):
    out_dim = neurons_per_layer[i]
    base_model.add(Dense(current_dim, out_dim, initializer=init_name))
    base_model.add(act_cls())
    current_dim = out_dim

base_model.add(Dense(current_dim, 1, initializer=init_name))
base_model.add(resolve_activation("Sigmoid")())

# ---------------------------------------------------------------------------
# 并行训练 4 种优化器 (由 OPTIMIZERS 元数据驱动)
# ---------------------------------------------------------------------------
optimizer_items = list(OPTIMIZERS.values())
histories: dict[str, dict[str, list[float]]] = {}
trained_models: dict[str, Sequential] = {}
leaderboard_rows = []

for opt_meta in optimizer_items:
    model_clone = copy.deepcopy(base_model)
    opt_cls = resolve_optimizer(opt_meta.id)
    opt_instance = opt_cls(learning_rate=float(lr))
    loss_fn = BinaryCrossEntropy()

    hist = model_clone.train(
        X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=int(epochs), batch_size=32, verbose=False
    )

    histories[opt_meta.id] = hist
    trained_models[opt_meta.id] = model_clone

    final_loss = hist["loss"][-1] if hist["loss"] else 1.0
    final_acc = hist["accuracy"][-1] if hist["accuracy"] else 0.0

    converged_epoch = next((i + 1 for i, l in enumerate(hist["loss"]) if l < 0.25), "未收敛")

    leaderboard_rows.append({
        "优化器算法": opt_meta.label,
        "最终 Loss": f"{final_loss:.4f}",
        "最终准确率": f"{final_acc:.1%}",
        "收敛步数 (Loss < 0.25)": str(converged_epoch),
        "_raw_loss": final_loss,
    })

leaderboard_rows.sort(key=lambda x: x["_raw_loss"])
champion = leaderboard_rows[0]["优化器算法"]
champion_loss = leaderboard_rows[0]["最终 Loss"]
champion_acc = leaderboard_rows[0]["最终准确率"]

# ---------------------------------------------------------------------------
# 遥测指标卡 (中英双语标签)
# ---------------------------------------------------------------------------
grid_html = (
    '<div class="metric-grid">'
    + render_metric_card("BENCHMARK WINNER // 竞速冠军", champion, delta=f"Loss: {champion_loss} | Acc: {champion_acc}", delta_type="positive", icon_name="award")
    + render_metric_card("TOPOLOGY // 数据集拓扑", dataset_name.upper(), delta=f"N={n_samples} | NOISE={noise}", delta_type="neutral", icon_name="database")
    + render_metric_card("INITIALIZER // 初始化器", init_name.upper(), delta=f"{n_layers} HIDDEN LAYERS", delta_type="neutral", icon_name="layers")
    + render_metric_card("TOTAL EPOCHS // 总轮数", f"{epochs} EPOCHS", delta=f"LR = {lr}", delta_type="neutral", icon_name="activity")
    + '</div>'
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 核心可视化：多轨 Loss 对比图
# ---------------------------------------------------------------------------
fig_multi_loss = plot_multi_loss_curves(histories, title="OPTIMIZER BENCHMARK // 优化器多轨收敛对比")
st.plotly_chart(fig_multi_loss, use_container_width=True)

# ---------------------------------------------------------------------------
# 排行榜与四分屏决策边界对比
# ---------------------------------------------------------------------------
render_section_heading("优化器收敛效能排行榜", icon_name="award")
df_board = pd.DataFrame(leaderboard_rows).drop(columns=["_raw_loss"])
st.dataframe(df_board, use_container_width=True, hide_index=True)

render_section_heading("四大优化器最终决策边界并排透视", icon_name="crosshair")
cols = st.columns(4)

for idx, opt_meta in enumerate(optimizer_items):
    with cols[idx]:
        m = trained_models[opt_meta.id]
        acc_text = next(r["最终准确率"] for r in leaderboard_rows if r["优化器算法"] == opt_meta.label)
        fig_b = plot_decision_boundary(
            m, X, y, resolution=70, title=f"{opt_meta.id} (Acc: {acc_text})"
        )
        st.plotly_chart(fig_b, use_container_width=True)

# 深度知识学习指南 (折叠微观原理解析)
render_deep_dive_card("四大优化算法动力学原理与更新公式深度对比", optimizer_items)
