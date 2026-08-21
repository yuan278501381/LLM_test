# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
🏎️ 里程碑 3: 优化器多轨竞速对比 (Optimizer Arena)

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
    render_network_params,
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

st.set_page_config(
    page_title="优化器多轨竞速 · NN Playground",
    page_icon="🏎️",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="🏎️ 优化器多轨竞速对比",
    subtitle="同一网络拓扑与初始权重起点，全方位对比 SGD / Momentum / RMSProp / Adam 的收敛动力学",
    badge_text="MILESTONE 3 · CONVERGENCE DYNAMICS",
    badge_type="amber",
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m3_", default_dataset="🌙 Moons"
)

net_params = render_network_params(
    allow_multi_layer=True, key_prefix="m3_", default_layers=2, default_neurons=[8, 4]
)
n_layers = net_params["n_layers"]
neurons_per_layer = net_params["neurons_per_layer"]
activation_name = net_params["activation"]
initializer = net_params["initializer"]

st.sidebar.markdown("### 🏎️ 竞速超参数")
col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input("基准学习率 (LR)", 0.001, 1.0, 0.03, step=0.01, format="%.3f", key="m3_lr")
with col2:
    epochs = st.slider("竞速轮数 (Epochs)", 20, 500, 150, step=10, key="m3_epochs")

# ---------------------------------------------------------------------------
# 数据与基准模型构建
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

base_model = Sequential()
current_dim = 2
for i in range(n_layers):
    out_dim = neurons_per_layer[i]
    base_model.add(Dense(current_dim, out_dim, initializer=initializer))
    base_model.add(ACTIVATION_MAP[activation_name]())
    current_dim = out_dim

base_model.add(Dense(current_dim, 1, initializer=initializer))
base_model.add(ACTIVATION_MAP["Sigmoid"]())

# ---------------------------------------------------------------------------
# 并行训练 4 种优化器
# ---------------------------------------------------------------------------
optimizer_names = ["SGD", "Momentum", "RMSProp", "Adam"]
histories: dict[str, dict[str, list[float]]] = {}
trained_models: dict[str, Sequential] = {}
leaderboard_rows = []

for opt_name in optimizer_names:
    model_clone = copy.deepcopy(base_model)
    opt_instance = OPTIMIZER_MAP[opt_name](learning_rate=lr)
    loss_fn = BinaryCrossEntropy()

    hist = model_clone.train(
        X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=epochs, batch_size=32, verbose=False
    )

    histories[opt_name] = hist
    trained_models[opt_name] = model_clone

    final_loss = hist["loss"][-1] if hist["loss"] else 1.0
    final_acc = hist["accuracy"][-1] if hist["accuracy"] else 0.0

    # 寻找首次达到 Loss < 0.2 的 Epoch
    converged_epoch = next((i + 1 for i, l in enumerate(hist["loss"]) if l < 0.25), "未收敛")

    leaderboard_rows.append({
        "优化器": opt_name,
        "最终 Loss": f"{final_loss:.4f}",
        "最终准确率": f"{final_acc:.1%}",
        "收敛步数 (Loss < 0.25)": str(converged_epoch),
        "_raw_loss": final_loss,
    })

# 排序得出冠军
leaderboard_rows.sort(key=lambda x: x["_raw_loss"])
champion = leaderboard_rows[0]["优化器"]
champion_loss = leaderboard_rows[0]["最终 Loss"]
champion_acc = leaderboard_rows[0]["最终准确率"]

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="metric-grid">
        {render_metric_card("🏆 竞速冠军", champion, delta=f"Loss: {champion_loss} | Acc: {champion_acc}", delta_type="positive", icon="🥇")}
        {render_metric_card("基准数据分布", dataset_name.upper(), delta=f"N={n_samples} | 噪声={noise}", delta_type="neutral", icon="📊")}
        {render_metric_card("统一初始权重", initializer.upper(), delta=f"{n_layers} 隐藏层", delta_type="neutral", icon="⚖️")}
        {render_metric_card("竞速总轮数", f"{epochs} Epochs", delta=f"统一学习率 LR={lr}", delta_type="neutral", icon="⏱️")}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 核心可视化：多轨 Loss 对比图
# ---------------------------------------------------------------------------
fig_multi_loss = plot_multi_loss_curves(histories, title="🏎️ 多优化器 Loss 对数收敛曲线")
st.plotly_chart(fig_multi_loss, use_container_width=True)

# ---------------------------------------------------------------------------
# 排行榜与四分屏决策边界对比
# ---------------------------------------------------------------------------
st.markdown("### 🏁 优化器性能排行榜")
df_board = pd.DataFrame(leaderboard_rows).drop(columns=["_raw_loss"])
st.dataframe(df_board, use_container_width=True, hide_index=True)

st.markdown("### 🗺️ 四大优化器最终决策边界并排透视")
cols = st.columns(4)

for idx, opt_name in enumerate(optimizer_names):
    with cols[idx]:
        m = trained_models[opt_name]
        acc_text = next(r["最终准确率"] for r in leaderboard_rows if r["优化器"] == opt_name)
        fig_b = plot_decision_boundary(
            m, X, y, resolution=70, title=f"{opt_name} (Acc: {acc_text})"
        )
        st.plotly_chart(fig_b, use_container_width=True)
