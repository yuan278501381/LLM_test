# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 3: 优化器对比 — 理解训练动力学

同一网络结构 + 同一数据集，用不同优化器训练，
直观对比 SGD、Momentum、RMSProp、Adam 的收敛速度和稳定性。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_multi_loss_curves,
)
from dashboard.components.param_panel import render_dataset_selector, render_network_params
from dashboard.utils.state import build_model
from datasets.generators import make_circles, make_moons, make_spiral, make_xor
from nn_core.losses import BinaryCrossEntropy
from nn_core.optimizers import SGD, Adam, Momentum, RMSProp
from nn_core.tensor import set_seed

st.set_page_config(page_title="⚙️ 优化器对比", layout="wide")
st.title("⚙️ 里程碑 3: 优化器对比")
st.markdown("**同一网络，不同优化器** — 观察 SGD / Momentum / RMSProp / Adam 的收敛行为差异。")

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ 参数控制")
dataset_name, n_samples, noise, seed = render_dataset_selector(key_prefix="m3_")
net_params = render_network_params(allow_multi_layer=True, key_prefix="m3_")

st.sidebar.subheader("⚙️ 训练参数")
learning_rate = st.sidebar.select_slider(
    "学习率",
    options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    value=0.01, key="m3_lr",
)
epochs = st.sidebar.slider("训练轮数", 10, 1000, 200, step=10, key="m3_epochs")
batch_size = st.sidebar.select_slider(
    "批大小", options=[4, 8, 16, 32, 64, 128], value=32, key="m3_bs",
)

st.sidebar.subheader("🔀 选择优化器")
use_sgd = st.sidebar.checkbox("SGD", value=True, key="m3_sgd")
use_momentum = st.sidebar.checkbox("Momentum", value=True, key="m3_mom")
use_rmsprop = st.sidebar.checkbox("RMSProp", value=True, key="m3_rms")
use_adam = st.sidebar.checkbox("Adam", value=True, key="m3_adam")

train_btn = st.sidebar.button("🚀 开始对比", type="primary", use_container_width=True, key="m3_train")

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
dataset_fn = {"moons": make_moons, "circles": make_circles, "xor": make_xor, "spiral": make_spiral}

if train_btn:
    # 收集选中的优化器
    optimizers_to_test: dict[str, type] = {}
    if use_sgd:
        optimizers_to_test["SGD"] = SGD
    if use_momentum:
        optimizers_to_test["Momentum"] = Momentum
    if use_rmsprop:
        optimizers_to_test["RMSProp"] = RMSProp
    if use_adam:
        optimizers_to_test["Adam"] = Adam

    if not optimizers_to_test:
        st.warning("请至少选择一个优化器！")
        st.stop()

    # 训练每个优化器
    all_histories: dict[str, dict] = {}
    all_models: dict[str, object] = {}

    progress = st.progress(0, text="准备训练...")
    total = len(optimizers_to_test)

    for idx, (opt_name, opt_cls) in enumerate(optimizers_to_test.items()):
        progress.progress((idx) / total, text=f"训练 {opt_name}...")

        # 每个优化器使用相同的随机种子，确保公平对比
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
            output_activation="Sigmoid",
        )

        optimizer = opt_cls(learning_rate=learning_rate)
        loss_fn = BinaryCrossEntropy()

        history = model.train(
            X, y, epochs=epochs, batch_size=batch_size,
            loss_fn=loss_fn, optimizer=optimizer,
        )

        all_histories[opt_name] = history
        all_models[opt_name] = model

    progress.progress(1.0, text="✅ 训练完成！")

    # ---- Loss 对比曲线 ----
    st.plotly_chart(
        plot_multi_loss_curves(all_histories, title="📈 优化器 Loss 对比"),
        use_container_width=True,
    )

    # ---- 指标对比表格 ----
    st.subheader("📋 最终指标对比")
    metrics_data = []
    for name, h in all_histories.items():
        metrics_data.append({
            "优化器": name,
            "最终 Loss": f"{h['loss'][-1]:.6f}",
            "最终 Accuracy": f"{h['accuracy'][-1]:.2%}",
            "收敛速度 (Loss<0.5 的 Epoch)": next(
                (i+1 for i, l in enumerate(h["loss"]) if l < 0.5), "未收敛"
            ),
        })
    st.table(metrics_data)

    # ---- 决策边界并排对比 ----
    st.subheader("🗺️ 决策边界对比")

    # 重新生成数据用于可视化（固定种子）
    set_seed(seed)
    X_viz, y_viz = dataset_fn.get(dataset_name, make_moons)(
        n_samples=n_samples, noise=noise, random_state=seed,
    )

    n_opts = len(all_models)
    cols = st.columns(min(n_opts, 4))
    for i, (name, model) in enumerate(all_models.items()):
        with cols[i % len(cols)]:
            st.plotly_chart(
                plot_decision_boundary(model, X_viz, y_viz, title=f"{name}"),
                use_container_width=True,
            )

    with st.expander("💡 关键观察", expanded=False):
        st.markdown("""
        1. **SGD**: 最简单但最慢，容易卡在鞍点
        2. **Momentum**: 利用历史动量加速收敛，在一致方向上「滚动」更快
        3. **RMSProp**: 自适应学习率，每个参数有独立的有效步长
        4. **Adam**: 综合 Momentum + RMSProp + 偏差修正，通常是「开箱即用」的最佳选择
        5. **学习率的影响**: 相同学习率下，自适应优化器（Adam/RMSProp）更鲁棒
        """)
else:
    st.info("👈 选择要对比的优化器，点击 **「🚀 开始对比」** 开始实验。")
