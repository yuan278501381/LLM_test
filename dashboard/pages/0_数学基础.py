# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""里程碑 M00：神经网络必需的数学、shape 与梯度检查基础。"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_section_heading,
)

st.set_page_config(page_title="M00 数学基础 · NN Playground", layout="wide")
apply_custom_theme()
render_lesson_evidence("M00", show_contract=True)

render_hero_header(
    title="M00 · 神经网络数学与计算基础",
    subtitle="从 shape、矩阵乘法到链式法则：亲手用有限差分核对解析梯度，并理解数值误差来自哪里",
    badge_text="MILESTONE 00 // MATH FOUNDATIONS",
    badge_type="blue",
)

st.info("学习顺序：先预测 shape 和梯度方向，再运行计算；看懂误差来源后再进入 M01。")

render_section_heading("TENSOR SHAPES // 张量形状与矩阵乘法", icon_name="grid")

col_shape, col_broadcast = st.columns(2)
with col_shape:
    st.markdown(
        """
        #### 线性层为什么写作 $Z=XW+b$？

        - $X$：`(batch, input_features)`
        - $W$：`(input_features, output_features)`
        - $XW$：`(batch, output_features)`
        - $b$：`(output_features,)`，沿 batch 维广播

        相邻的 `input_features` 必须相同；结果保留外侧两个维度。
        """
    )
with col_broadcast:
    batch_size = st.slider("batch size", 1, 8, 4)
    input_features = st.slider("input features", 1, 8, 3)
    output_features = st.slider("output features", 1, 8, 2)
    x_matrix = np.arange(batch_size * input_features, dtype=float).reshape(
        batch_size, input_features
    )
    weights = np.ones((input_features, output_features))
    bias = np.arange(output_features, dtype=float)
    z_matrix = x_matrix @ weights + bias
    st.code(
        f"X{x_matrix.shape} @ W{weights.shape} + b{bias.shape} -> Z{z_matrix.shape}",
        language="text",
    )
    st.dataframe(z_matrix, width="stretch")

render_section_heading("CHAIN RULE LAB // 链式法则与梯度检查", icon_name="activity")

st.markdown(
    r"""
考虑单样本线性回归：$z=wx+b$，$L=(z-y)^2$。

解析梯度由链式法则得到：
$\frac{\partial L}{\partial w}=2(z-y)x$，
$\frac{\partial L}{\partial b}=2(z-y)$。
有限差分则重新计算两侧损失：
$\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$。
"""
)

control_col, result_col = st.columns([1, 1.5])
with control_col:
    x_value = st.slider("输入 x", -3.0, 3.0, 1.5, 0.1)
    weight_value = st.slider("权重 w", -3.0, 3.0, 0.8, 0.1)
    bias_value = st.slider("偏置 b", -2.0, 2.0, 0.2, 0.1)
    target_value = st.slider("目标 y", -3.0, 3.0, 1.0, 0.1)
    epsilon_exp = st.slider("log10(ε)", -12, -1, -5)

epsilon = 10.0**epsilon_exp


def scalar_loss(weight: float, bias: float) -> float:
    """当前交互样本的平方误差。"""

    prediction = weight * x_value + bias
    return float((prediction - target_value) ** 2)


prediction = weight_value * x_value + bias_value
loss_value = scalar_loss(weight_value, bias_value)
analytic_dw = 2.0 * (prediction - target_value) * x_value
analytic_db = 2.0 * (prediction - target_value)
numeric_dw = (
    scalar_loss(weight_value + epsilon, bias_value)
    - scalar_loss(weight_value - epsilon, bias_value)
) / (2.0 * epsilon)
numeric_db = (
    scalar_loss(weight_value, bias_value + epsilon)
    - scalar_loss(weight_value, bias_value - epsilon)
) / (2.0 * epsilon)
relative_error = abs(analytic_dw - numeric_dw) / (abs(analytic_dw) + abs(numeric_dw) + 1e-12)

with result_col:
    st.markdown(
        '<div class="metric-grid">'
        + render_metric_card("PREDICTION // 预测", f"{prediction:.5f}", icon_name="target")
        + render_metric_card("LOSS // 损失", f"{loss_value:.6f}", icon_name="activity")
        + render_metric_card("ANALYTIC dL/dw", f"{analytic_dw:.6f}", icon_name="cpu")
        + render_metric_card("NUMERIC dL/dw", f"{numeric_dw:.6f}", icon_name="database")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.code(
        f"dL/dw: analytic={analytic_dw:.10f}, numeric={numeric_dw:.10f}\n"
        f"dL/db: analytic={analytic_db:.10f}, numeric={numeric_db:.10f}\n"
        f"relative error={relative_error:.3e}",
        language="text",
    )
    if relative_error < 1e-5:
        st.success("解析梯度与有限差分在当前步长下吻合。")
    else:
        st.warning("误差较大。尝试调节 ε：过大产生截断误差，过小放大浮点舍入误差。")

epsilon_grid = np.logspace(-12, -1, 80)
errors = []
for eps in epsilon_grid:
    numeric = (
        scalar_loss(weight_value + eps, bias_value) - scalar_loss(weight_value - eps, bias_value)
    ) / (2.0 * eps)
    errors.append(abs(analytic_dw - numeric) / (abs(analytic_dw) + abs(numeric) + 1e-12))

fig_error = go.Figure(
    go.Scatter(x=epsilon_grid, y=np.maximum(errors, 1e-18), mode="lines", name="relative error")
)
fig_error.update_xaxes(type="log", title="有限差分步长 ε")
fig_error.update_yaxes(type="log", title="相对误差")
fig_error.update_layout(height=340, margin=dict(l=30, r=20, t=35, b=35))
st.plotly_chart(fig_error, width="stretch")

render_section_heading("CHECK YOUR MODEL // 形成性小测验", icon_name="check-circle")
answer = st.radio(
    "若解析梯度 dL/dw > 0，普通梯度下降下一步通常怎样更新 w？",
    ("增大 w", "减小 w", "保持不变", "仅改变 b"),
    index=None,
)
if answer == "减小 w":
    st.success("正确：w ← w - η·dL/dw；当梯度为正且 η>0 时，w 会减小。")
elif answer is not None:
    st.error("再看一次更新式 w ← w - η·dL/dw，注意前面的负号。")

st.caption("完成标准：能解释矩阵 shape、链式法则路径，以及有限差分步长为何不能无限缩小。")
