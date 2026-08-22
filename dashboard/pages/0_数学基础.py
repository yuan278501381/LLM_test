# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard/pages/0_数学基础.py - 里程碑 M00：神经网络必需的数学、张量 Shape 契约与梯度检查基础。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_formula_breakdown_card,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
)

st.set_page_config(
    page_title="M00 数学基础 · NN Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()

# Hero 标题
render_hero_header(
    title="M00: 神经网络数学与计算基础",
    subtitle="从 Tensor Shape 矩阵乘法消维到微积分链式法则：亲手用有限差分核对解析梯度，攻克初学者第一道数学难关",
    badge_text="MILESTONE 00 // MATH FOUNDATIONS & TENSOR CONTRACT",
    badge_type="blue",
)

# 核心教学论据
render_lesson_evidence("M00", show_contract=True)
render_core_result_evidence("M00")

# ---------------------------------------------------------------------------
# [A] 教学指引与蓝图导航
# ---------------------------------------------------------------------------
blueprint_sections = [
    {
        "id": "A",
        "name": "教学指引与蓝图",
        "desc": "掌握张量形状、矩阵相消与梯度检查核心思想",
        "color": "blue",
        "target_id": "region-a",
    },
    {
        "id": "B",
        "name": "张量形状实验室",
        "desc": "实机调节 batch, input_features, output_features 见证矩阵乘法与广播",
        "color": "amber",
        "target_id": "region-b",
    },
    {
        "id": "C",
        "name": "链式法则与梯度检查",
        "desc": "解析梯度 vs 有限差分数值梯度，探究误差 U 型曲线",
        "color": "emerald",
        "target_id": "region-c",
    },
    {
        "id": "D",
        "name": "概率与交叉熵",
        "desc": "从 logits 经 log-sum-exp 与 softmax 得到概率和交叉熵",
        "color": "rose",
        "target_id": "region-d",
    },
    {
        "id": "E",
        "name": "形成性小测验",
        "desc": "快速检验反向传播参数更新直觉",
        "color": "purple",
        "target_id": "region-e",
    },
    {
        "id": "F",
        "name": "通关标准",
        "desc": "确认能够解释 shape、梯度、概率与交叉熵",
        "color": "blue",
        "target_id": "region-f",
    },
]

render_page_guide(
    title="神经网络数学基础与张量契约全景指南",
    plain_intro="神经网络训练需要同时理解张量运算、链式法则与概率损失。这里先建立最小数学语言：用 shape 追踪数据，用梯度描述局部变化，再把 logits 转换为概率并用交叉熵衡量预测。",
    hyperparams_desc="• batch size (N)：批次样本数，并行喂给模型的独立数据行数；\n• input features (Din)：输入特征数，每个样本包含的原始指标数量；\n• output features (Dout)：输出特征数，当前层神经元提炼出的新特征维度；\n• 有限差分步长 ε：微小扰动步长，用于数值梯度校验。",
    telemetry_desc="• 矩阵乘法产物 Z：形状为 (batch, output_features) 的输出张量；\n• 解析梯度：按当前公式求得的导数；\n• 有限差分梯度：在当前输入和步长附近得到的数值近似；\n• 绝对/相对误差：只说明本次局部检查的一致程度；\n• Softmax 与交叉熵：把 logits 转成归一化概率，并衡量目标类别的负对数概率。",
    experiments=[
        "在 Section B 拖动 input features 滑块：观察权重矩阵 W 的行数与输入矩阵 X 的列数如何保持同步变化！",
        "在 Section B 观察广播机制：为什么偏置 b 只有一维 (output_features,)，却能加到全部 batch 个样本上？",
        "在 Section C 拖动 log10(ε) 从 -1 到 -12：观察相对误差为什么先变小后急剧变大（形成 U 型误差曲线）？",
        "在 Section D 调节 temperature：先预测概率分布会变尖还是变平，再观察交叉熵如何变化。",
    ],
    blueprint_sections=blueprint_sections,
    guide_region_id="region-a",
)

# ---------------------------------------------------------------------------
# [B] 张量形状与矩阵乘法契约实验室
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-b" class="interactive-region" style="margin-top:1.2rem;margin-bottom:0.6rem;padding:0.45rem 0.75rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f"{anchor_badge('B', 'amber')} <b>TENSOR SHAPES & MATRIX MULTIPLICATION // 张量形状与矩阵乘法契约</b>"
    f"</div>",
    unsafe_allow_html=True,
)

col_shape, col_broadcast = st.columns([1.1, 1.4])
with col_shape:
    st.markdown(
        """
        ##### 核心概念白话通俗说明 (Beginner Guide)

        在神经网络中，数据是以二维或高维矩阵（张量）流动的：

        * **`batch size` (批次大小 / 样本数 $N$)**：
          * **通俗理解**：**一次打包、同时送给模型计算的数据条数**。
          * *例子*：一次性批量评估 4 套房产、识别 4 张图片、或翻译 4 句话。
        * **`input features` (输入特征数 / 输入维度 $D_{in}$)**：
          * **通俗理解**：**每个样本自身携带的输入指标或属性数量**。
          * *例子*：每套房产有 3 个指标 `[建筑面积, 房间数, 距离地铁距离]`。
        * **`output features` (输出特征数 / 神经元个数 $D_{out}$)**：
          * **通俗理解**：**当前层神经网络提炼出多少个新的高阶表征或最终结论**。
          * *例子*：模型需要预测 2 个结论 `[预估总价, 租金回报率]`。
        """
    )

    render_formula_breakdown_card(
        formula_latex=r"Z = X \cdot W + b",
        math_principle="矩阵乘法要求相邻维度严格对齐相消：(N, Din) × (Din, Dout) = (N, Dout)。偏置向量 b 沿 batch 维度自动广播相加。",
        params_breakdown=[
            {
                "param": "X",
                "shape": "(batch, input_features)",
                "role": "输入特征矩阵：每行代表一个独立样本，每列代表一项输入属性",
            },
            {
                "param": "W",
                "shape": "(input_features, output_features)",
                "role": "权重矩阵：把 input_features 线性组合映射为 output_features",
            },
            {
                "param": "XW",
                "shape": "(batch, output_features)",
                "role": "矩阵点积结果：内部的 input_features 维度被求和相消，保留外侧两维",
            },
            {
                "param": "b",
                "shape": "(output_features,)",
                "role": "偏置向量：每个输出神经元的基准偏置，沿第 0 维 (batch) 自动广播复制",
            },
            {
                "param": "Z",
                "shape": "(batch, output_features)",
                "role": "最终线性输出张量：供后续激活函数激活或作为下一层网络的输入",
            },
        ],
    )

with col_broadcast:
    st.markdown("##### 交互控制台与实时张量遥测")
    c1, c2, c3 = st.columns(3)
    with c1:
        batch_size = st.slider("batch size (样本数 N)", 1, 8, 4, help="同时并行计算的数据行数")
    with c2:
        input_features = st.slider(
            "input features (输入特征 Din)", 1, 8, 3, help="每个样本拥有的输入属性维度"
        )
    with c3:
        output_features = st.slider(
            "output features (输出特征 Dout)", 1, 8, 2, help="当前层神经元个数 / 输出特征维度"
        )

    x_matrix = np.arange(batch_size * input_features, dtype=float).reshape(
        batch_size, input_features
    )
    weights = np.ones((input_features, output_features))
    bias = np.arange(output_features, dtype=float)
    z_matrix = x_matrix @ weights + bias

    render_live_param_status_bar(
        title="TENSOR SHAPE CONTRACT // 矩阵维度契约",
        badges=[
            {"label": "X (输入)", "value": f"({batch_size}, {input_features})", "color": "blue"},
            {
                "label": "W (权重)",
                "value": f"({input_features}, {output_features})",
                "color": "amber",
            },
            {"label": "b (偏置)", "value": f"({output_features},)", "color": "purple"},
            {
                "label": "Z (输出)",
                "value": f"({batch_size}, {output_features})",
                "color": "emerald",
            },
        ],
        metrics=[
            ("内部相消维", f"{input_features}"),
            ("外部保留维", f"({batch_size}, {output_features})"),
            ("矩阵乘法乘法次数", f"{batch_size * input_features * output_features}"),
        ],
        tag=f"DIMS: ({batch_size}, {output_features})",
        tag_color="emerald",
    )

    st.markdown(
        f"**计算公式与实时形状映射**：`X({batch_size}, {input_features}) @ W({input_features}, {output_features}) + b({output_features},) -> Z({batch_size}, {output_features})`"
    )

    st.markdown("**实时输出张量数值矩阵 $Z$**：")
    st.dataframe(z_matrix, width="stretch")

    with st.expander("SHAPE FAILURE // 为什么错误维度不能相乘", expanded=False):
        st.markdown(
            f"当前 `X` 的最后一维是 `{input_features}`。如果 `W` 的第一维改成 "
            f"`{input_features + 1}`，内部维不相等，`X @ W` 没有定义；NumPy 会报维度错误，"
            "而不是自动猜测应删除或补齐哪个特征。"
        )

# ---------------------------------------------------------------------------
# [C] 链式法则与数值梯度检查实验室
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="margin-top:1.5rem;margin-bottom:0.6rem;padding:0.45rem 0.75rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f"{anchor_badge('C', 'emerald')} <b>CHAIN RULE LAB // 链式法则与数值梯度检查</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    r"""
    考虑单样本线性回归：预测值 $z = wx + b$，损失函数 $L = (z - y)^2$。

    * **解析梯度 (Analytic Gradient)**：通过微积分链式法则直接求导：
      $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = 2(z - y) \cdot x, \quad \frac{\partial L}{\partial b} = 2(z - y)$$
    * **有限差分数值梯度 (Numeric Gradient / Finite Difference)**：不依赖求导公式，给参数加上极其微小的扰动 $\epsilon$：
      $$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$
    * **为什么需要梯度检查？** 有限差分可以在选定输入附近比较解析梯度与数值近似，帮助发现反向传播错误；一次通过不能排除其他输入、不可导点、随机运算或精度条件下的问题。
    """
)

control_col, result_col = st.columns([1, 1.4])
with control_col:
    st.markdown("##### 样本与步长调节")
    x_value = st.slider("输入 x", -3.0, 3.0, 1.5, 0.1, help="单样本输入数值")
    weight_value = st.slider("权重 w", -3.0, 3.0, 0.8, 0.1, help="线性层权重斜率")
    bias_value = st.slider("偏置 b", -2.0, 2.0, 0.2, 0.1, help="线性层截距偏置")
    target_value = st.slider("真实目标 y", -3.0, 3.0, 1.0, 0.1, help="样本期望标签")
    epsilon_exp = st.slider("微扰步长指数 log10(ε)", -12, -1, -5, help="微小扰动步长 ε = 10^k")

epsilon = 10.0**epsilon_exp


def scalar_loss(weight: float, bias: float) -> float:
    """当前交互样本的平方误差损失。"""
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
absolute_error = abs(analytic_dw - numeric_dw)
relative_error = absolute_error / max(abs(analytic_dw), abs(numeric_dw), 1e-8)

with result_col:
    render_live_param_status_bar(
        title="CHAIN RULE & GRADIENT TELEMETRY // 链式法则微观梯度",
        badges=[
            {"label": "x", "value": f"{x_value:.2f}", "color": "blue"},
            {"label": "w", "value": f"{weight_value:.2f}", "color": "amber"},
            {"label": "b", "value": f"{bias_value:.2f}", "color": "purple"},
            {"label": "y", "value": f"{target_value:.2f}", "color": "rose"},
        ],
        metrics=[
            ("预测误差 (z-y)", f"{prediction - target_value:+.4f}"),
            ("解析梯度 ∂L/∂w", f"{analytic_dw:+.4f}"),
            ("数值梯度 (有限差分)", f"{numeric_dw:+.4f}"),
            ("绝对误差", f"{absolute_error:.2e}"),
            ("相对误差", f"{relative_error:.2e}"),
        ],
        tag="EXACT MATCH [PASS]" if relative_error < 1e-5 else "STEP TUNING [WARN]",
        tag_color="emerald" if relative_error < 1e-5 else "amber",
    )

    st.markdown(
        '<div class="metric-grid">'
        + render_metric_card("PREDICTION // 预测值 z", f"{prediction:.5f}", icon_name="target")
        + render_metric_card("LOSS // 均方误差 L", f"{loss_value:.6f}", icon_name="activity")
        + render_metric_card("ANALYTIC dL/dw // 解析导数", f"{analytic_dw:.6f}", icon_name="cpu")
        + render_metric_card("NUMERIC dL/dw // 有限差分", f"{numeric_dw:.6f}", icon_name="database")
        + "</div>",
        unsafe_allow_html=True,
    )

    if relative_error < 1e-5:
        st.success(
            f"[PASS] 在当前输入、float64 精度和 ε = 10^{epsilon_exp} 下，解析梯度与有限差分近似一致（相对误差 {relative_error:.2e} < 1e-5）。这是一项局部检查，不是对所有输入的证明。"
        )
    else:
        st.warning(
            f"[WARN] 相对误差较大 ({relative_error:.2e})。原因：当 ε 过大（>-2）时泰勒展开截断误差显著；当 ε 过小（<-9）时 IEEE 754 浮点数相减导致有效数字丢失（灾难性消去）。"
        )

# 误差 U 型曲线
st.markdown("##### 误差 U 型曲线：观察截断误差与浮点舍入误差的数值边界")
epsilon_grid = np.logspace(-12, -1, 80)
errors = []
for eps in epsilon_grid:
    numeric = (
        scalar_loss(weight_value + eps, bias_value) - scalar_loss(weight_value - eps, bias_value)
    ) / (2.0 * eps)
    errors.append(abs(analytic_dw - numeric) / (abs(analytic_dw) + abs(numeric) + 1e-12))

fig_error = go.Figure()
fig_error.add_trace(
    go.Scatter(
        x=epsilon_grid,
        y=np.maximum(errors, 1e-18),
        mode="lines+markers",
        name="相对误差 (Relative Error)",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=4),
    )
)
fig_error.add_trace(
    go.Scatter(
        x=[epsilon],
        y=[max(relative_error, 1e-18)],
        mode="markers",
        name="当前选中 ε 位置",
        marker=dict(color="#dc2626", size=11, symbol="diamond"),
    )
)

fig_error.update_xaxes(type="log", title="有限差分扰动步长 ε (对数坐标)")
fig_error.update_yaxes(type="log", title="相对误差 (对数坐标)")
fig_error.update_layout(
    height=320,
    margin=dict(l=30, r=20, t=35, b=35),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.8)",
)
st.plotly_chart(fig_error, width="stretch")

# ---------------------------------------------------------------------------
# [D] 概率、log-sum-exp、Softmax 与交叉熵
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="margin-top:1.5rem;margin-bottom:0.6rem;padding:0.45rem 0.75rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f"{anchor_badge('D', 'rose')} <b>PROBABILITY & CROSS-ENTROPY // 概率与交叉熵</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    r"""
    分类网络先输出任意实数 **logits**，再用 Softmax 得到和为 1 的概率：
    $$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$
    直接计算很大的 $e^{z_i}$ 可能溢出，因此先减去最大 logit；这不会改变最终概率。
    对目标类别 $y$，单样本交叉熵是 $L=-\log p_y$。它只评价分配给目标类别的概率，
    不是“模型是否真正理解”的直接度量。
    """
)

prob_col, prob_result = st.columns([1, 1.4])
with prob_col:
    temperature = st.slider("Softmax temperature", 0.25, 2.0, 1.0, 0.25)
    target_class = st.radio("目标类别", ("类别 A", "类别 B", "类别 C"), horizontal=True)

base_logits = np.array([2.0, 1.0, -1.0], dtype=np.float64) / temperature
logsumexp = float(np.max(base_logits) + np.log(np.sum(np.exp(base_logits - np.max(base_logits)))))
log_probs = base_logits - logsumexp
probabilities = np.exp(log_probs)
target_idx = ("类别 A", "类别 B", "类别 C").index(target_class)
cross_entropy = float(-log_probs[target_idx])

with prob_result:
    st.dataframe(
        {
            "类别": ["A", "B", "C"],
            "缩放后 logit": base_logits,
            "Softmax 概率": probabilities,
        },
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        f"概率和：`{probabilities.sum():.12f}`　｜　目标概率：`{probabilities[target_idx]:.6f}`　｜　"
        f"交叉熵：`{cross_entropy:.6f}`"
    )
    st.caption("较低 temperature 通常让同一组 logits 的分布更尖；这不保证预测更正确或更有创造力。")

# ---------------------------------------------------------------------------
# [E] 形成性小测验
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="margin-top:1.5rem;margin-bottom:0.6rem;padding:0.45rem 0.75rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f"{anchor_badge('E', 'purple')} <b>CHECK YOUR MODEL // 形成性小测验</b>"
    f"</div>",
    unsafe_allow_html=True,
)

answer = st.radio(
    "若解析梯度 dL/dw > 0，在学习率 η > 0 时，标准梯度下降下一步通常怎样更新权重 w？",
    ("增大 w", "减小 w", "保持不变", "仅改变偏置 b"),
    index=None,
)
if answer == "减小 w":
    st.success(
        "[PASS] 回答正确！根据梯度下降更新公式 w ← w - η · (∂L/∂w)，当导数为正时，减去正数使得 w 向左（变小）移动，朝着损失下降的方向前进。"
    )
elif answer is not None:
    st.error(
        "[FAIL] 再看一次更新公式：w ← w - η · (∂L/∂w)。注意前面的负号！负导数方向才是损失下降最快的方向。"
    )

# ---------------------------------------------------------------------------
# [F] 先修基础总结与进阶通关要求
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-f" class="interactive-region" style="margin-top:1.5rem;margin-bottom:0.6rem;padding:0.45rem 0.75rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f"{anchor_badge('F', 'blue')} <b>MILESTONE CRITERIA // 先修基础总结与通关标准</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.caption(
    "里程碑 M00 完成标准：能解释 batch/input/output shape、链式法则与有限差分的局限，并能从 logits 稳定计算 Softmax 概率和交叉熵。"
)
