# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 1: 单神经元感知器 (Single Neuron Perceptron) - 零基础入门保姆级教学平台

微观解构最小计算单元：线性加权 Z = XW + b、激活函数非线性映射、交叉熵损失、反向传播链式推导与权重轨迹寻优。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import (
    plot_loss_curve,
)
from dashboard.components.client_player import (
    build_perceptron_payload,
    render_boundary_canvas,
    render_player_controls,
    render_trajectory_canvas,
)
from dashboard.components.param_panel import (
    OPTIMIZER_HINTS,
    get_visual_hint,
    render_dataset_selector,
    render_deep_dive_card,
)
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.constants.knowledge import ACTIVATIONS, DATASETS, OPTIMIZERS
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_floating_hud_navigator,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
)
from dashboard.utils.state import get_dataset, resolve_optimizer
from nn_core.activations import Sigmoid
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Single Perceptron · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M01", show_contract=True)
render_core_result_evidence("M01")

render_hero_header(
    title="单神经元感知器",
    subtitle="解剖深度学习最小计算基元：线性加权 $Z = XW + b$、非线性激活与梯度下降的几何本质",
    badge_text="MILESTONE 01 // ATOMIC PERCEPTRON",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="单神经元感知器入门与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "控制台面板",
            "desc": "在左侧侧边栏切换数据集、调节学习率与训练轮数",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解单神经元画直线分界线的物理机理",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时遥测指标",
            "desc": "观察模型自主训练后的 Loss 扣分与 100% 准确率",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "空间决策流形",
            "desc": "观察特征平面上那根黑色分界实线如何精准切分红蓝点",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "损失收敛曲线",
            "desc": "验证做错题扣分 (Loss) 是否如大滑梯般平滑下降至 0",
            "color": "blue",
            "target_id": "region-e",
        },
        {
            "id": "F",
            "name": "权重寻优轨迹",
            "desc": "俯瞰参数 (w₁, w₂) 沿损失坡度滚入盆地最低点的路径",
            "color": "rose",
            "target_id": "region-f",
        },
    ],
    plain_intro=(
        f"<b>1. 单神经元像一个拿尺子画直线的机器人</b>：输入特征 $(x_1, x_2)$ 是数据在地图上的横纵坐标。<br>"
        f"机器人的唯一任务就是在 {anchor_badge('[D. 空间决策流形图]', 'purple', target_id='region-d')} 上画出一根直线 <code>w₁·x₁ + w₂·x₂ + b = 0</code>，"
        f"把<b>蓝色点</b>（类别 0）和<b>红色点</b>（类别 1）干净利落地切开！<br><br>"
        f"<b>2. 怎么看主视窗图表？</b><br>"
        f"• 请看 {anchor_badge('[D. 空间决策流形图]', 'purple', target_id='region-d')}：<b>加粗的黑色实线 <code>[DECISION LINE] (Line: P=0.5)</code></b> 就是机器人真正画出的<b>唯一那根分界线</b>！线两侧的概率正好是 50% 临界点。<br>"
        f"• 请看 {anchor_badge('[E. 损失收敛曲线]', 'blue', target_id='region-e')}：观察左边蓝线是否像滑滑梯一样俯冲到 0，右边绿线是否快速飙升至 100%。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 左侧控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>剧情化一键关卡</b>：新手推荐直接点击关卡按钮快速体验！<br>"
        f"• <b>分布类型</b>：选择不同几何形状的数据集（如 Blobs 简单，Moons 弯曲）。<br>"
        f"• <b>激活函数</b>：如 <code>Sigmoid</code>，把输出压缩到 0~1 之间表示概率。<br>"
        f"• <b>学习率 (LR)</b>：机器人每次看错后调整尺子的步子大小。<br>"
        f"• <b>训练轮数</b>：机器人总共练习画线的迭代次数。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[C. 实时遥测指标]', 'emerald', target_id='region-c')} 验收成果</b>：<br>"
        f"• <b>最终训练损失 (Loss)</b>：做错题的惩罚分，<b>越接近 0 代表分得越准</b>。<br>"
        f"• <b>分类准确率 (Acc)</b>：做对的题目比例，<b>100% 代表完全切对</b>。<br>"
        f"• <b>学得权重 [w₁, w₂] 与偏置 b</b>：机器人经过训练后自己算出的直线参数（w 决定直线的斜率方向，b 决定直线上下平移）。<br>"
        f"• <b>学得的直线方程</b>：在流形图上绘制出的黑色决策分界实线公式。"
    ),
    experiments=[
        f"<b>第 1 步【寻找那根线】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 选择 <code>Blobs (高斯聚类)</code>，观察 {anchor_badge('[D. 决策流形图]', 'purple', target_id='region-d')} 中间那根<b>加粗的黑色分界线</b>，看它如何恰好把红蓝两团点隔开！",
        f"<b>第 2 步【体验调参】</b>：试着在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把【学习率】改成 <code>0.001</code>（步子太小），看 {anchor_badge('[E. 收敛曲线]', 'blue', target_id='region-e')} 需要很多轮才能画准；再改成 <code>1.5</code>（步子太大），看看直线是不是开始剧烈晃动！",
        "<b>第 3 步【见证单神经元的局限】</b>：把【分布类型】换成 <code>XOR (正交异或)</code> 或 <code>Moons (双月形)</code>。你会发现一根笔直的线无论怎么转都无法切开它们！这就是为什么我们需要<b>多层网络（深度学习）</b>！",
    ],
)

render_floating_hud_navigator(
    [
        {
            "id": "A",
            "name": "控制台面板",
            "desc": "在左侧侧边栏切换数据集、调节学习率与训练轮数",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解单神经元画直线分界线的物理机理",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时遥测指标",
            "desc": "观察模型自主训练后的 Loss 扣分与 100% 准确率",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "空间决策流形",
            "desc": "观察特征平面上那根黑色分界实线如何精准切分红蓝点",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "损失收敛曲线",
            "desc": "验证做错题扣分 (Loss) 是否如大滑梯般平滑下降至 0",
            "color": "blue",
            "target_id": "region-e",
        },
        {
            "id": "F",
            "name": "权重寻优轨迹",
            "desc": "俯瞰参数 (w₁, w₂) 沿损失坡度滚入盆地最低点的路径",
            "color": "rose",
            "target_id": "region-f",
        },
    ]
)

# ---------------------------------------------------------------------------
# 侧边栏参数面板 (由 knowledge 元数据驱动，带丰富 Tooltip 与剧情化一键关卡)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>NARRATIVE QUESTS // 剧情化探索关卡</b></div>',
    unsafe_allow_html=True,
)
c_q1, c_q2, c_q3 = st.sidebar.columns(3)
if c_q1.button(
    "关卡 1\n初出茅庐",
    help="Blobs 线性可分简单题，见证单神经元在标准配置下实现线性分割 (高准确率收敛)",
    width="stretch",
):
    st.session_state["m1_dataset"] = DATASETS["blobs"].label
    st.session_state["m1_act"] = ACTIVATIONS["Sigmoid"].label
    st.session_state["m1_opt"] = OPTIMIZERS["Adam"].label
    st.session_state["m1_lr"] = 0.100
    st.session_state["m1_epochs"] = 100
    st.rerun()

if c_q2.button(
    "关卡 2\n调参翻车",
    help="超大学习率 LR=1.8，步子太大扯到蛋，见证直线剧烈翻滚与损失震荡！",
    width="stretch",
):
    st.session_state["m1_dataset"] = DATASETS["blobs"].label
    st.session_state["m1_act"] = ACTIVATIONS["Sigmoid"].label
    st.session_state["m1_opt"] = OPTIMIZERS["SGD"].label
    st.session_state["m1_lr"] = 1.800
    st.session_state["m1_epochs"] = 60
    st.rerun()

if c_q3.button(
    "关卡 3\n绝望困境",
    help="XOR 异或难题，直线转到天荒地老也只能瞎猜 50%，见证单神经元的物理极限！",
    width="stretch",
):
    st.session_state["m1_dataset"] = DATASETS["xor"].label
    st.session_state["m1_act"] = ACTIVATIONS["Sigmoid"].label
    st.session_state["m1_opt"] = OPTIMIZERS["Adam"].label
    st.session_state["m1_lr"] = 0.150
    st.session_state["m1_epochs"] = 150
    st.rerun()

st.sidebar.divider()

dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m1_", default_dataset="blobs"
)

st.sidebar.markdown(
    f'<div style="margin-top:0.6rem;margin-bottom:0.4rem;">{anchor_badge("A", "amber")} <span style="font-size:0.85rem;font-weight:700;color:#0f172a;">HYPERPARAMETERS // 超参数配置</span></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:0.4rem 0.6rem;font-size:0.75rem;color:#334155;margin-bottom:0.6rem;">
    <b>[神经元架构契约]</b>：可微逻辑神经元 (Logistic Neuron)<br/>
    <code>Input(2) → Dense(2, 1) → Sigmoid → BinaryCrossEntropy</code><br/>
    以 $P(y=1|x) \\ge 0.5$ 作为严格分类决策面 ($z=0$)。
    </div>""",
    unsafe_allow_html=True,
)

act_meta = ACTIVATIONS["Sigmoid"]

opt_list = list(OPTIMIZERS.values())
opt_labels = [m.label for m in opt_list]
selected_opt_label = st.sidebar.radio(
    "优化器",
    options=opt_labels,
    format_func=lambda o: f"**{o}**\n\n↳ *{get_visual_hint(o, OPTIMIZER_HINTS, OPTIMIZERS)}*",
    help="梯度下降算法。负责根据损失对权重 (w₁, w₂) 和偏置 b 的梯度调整参数大小。",
    key="m1_opt",
)
opt_meta = next(m for m in opt_list if m.label == selected_opt_label)

col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input(
        "学习率 (LR)",
        0.001,
        2.0,
        float(st.session_state.get("m1_lr", 0.1)),
        step=0.01,
        format="%.3f",
        help="参数更新步长 $\\eta$。过大导致跳过最优点剧烈震荡，过小导致收敛缓慢。",
        key="m1_lr",
    )
with col2:
    epochs = st.slider(
        "训练轮数",
        10,
        500,
        int(st.session_state.get("m1_epochs", 100)),
        step=10,
        help="全量数据集迭代轮数。",
        key="m1_epochs",
    )

# ---------------------------------------------------------------------------
# 数据生成与模型训练
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

# 构建单神经元可微模型: Input(2) -> Dense(2, 1) -> Sigmoid -> BCE
model = Sequential()
dense_layer = Dense(2, 1, initializer="random")
model.add(dense_layer)
model.add(Sigmoid())

# 损失函数与决策规则完全自洽:
# Sigmoid 激活输出严格概率 P ∈ (0, 1)，搭配 BinaryCrossEntropy 损失与 P >= 0.5 决策阈值
loss_fn = BinaryCrossEntropy()

opt_cls = resolve_optimizer(opt_meta.id)
optimizer = opt_cls(learning_rate=float(lr))

# 记录权重轨迹与偏置轨迹快照
weight_trajectory = [dense_layer.weights.copy()]
bias_trajectory = [dense_layer.biases.copy()]

# 训练循环并记录历史
history: dict[str, list[float]] = {"loss": [], "accuracy": []}

for _ in range(int(epochs)):
    y_pred = model.forward(X, training=True)
    loss = loss_fn.forward(y_pred, y)
    acc = float(np.mean((y_pred >= 0.5).astype(float) == y))

    history["loss"].append(loss)
    history["accuracy"].append(acc)

    dloss = loss_fn.backward()
    model.backward(dloss)

    optimizer.step(model.layers)
    weight_trajectory.append(dense_layer.weights.copy())
    bias_trajectory.append(dense_layer.biases.copy())

final_loss = history["loss"][-1] if history["loss"] else 0.0
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0
w_final = dense_layer.weights.ravel()
b_final = float(dense_layer.biases.ravel()[0])

# ---------------------------------------------------------------------------
# 遥测指标卡 (模型自动学习输出结果)
# ---------------------------------------------------------------------------
sign_b = "+" if b_final >= 0 else "-"
abs_b = abs(b_final)
hyperplane_str = f"{w_final[0]:.2f}x₁ + {w_final[1]:.2f}x₂ {sign_b} {abs_b:.2f} = 0"

grid_html = (
    f'<div id="region-c" class="interactive-region" style="margin-bottom:0.5rem;padding:0.4rem 0.6rem;border-radius:8px;border:1px solid #e2e8f0;background:#ffffff;">'
    f'{anchor_badge("C", "emerald")} <span style="font-size:0.82rem;font-weight:800;color:#047857;letter-spacing:0.04em;text-transform:uppercase;">TELEMETRY BENCHMARK // 实时模型自学习遥测成果</span>'
    f"</div>"
    f'<div class="metric-grid">'
    + render_metric_card(
        "FINAL LOSS // 最终训练损失",
        f"{final_loss:.4f}",
        delta="已收敛 (CONVERGED)" if final_loss < 0.2 else "训练中 (TRAINING)",
        delta_type="positive" if final_loss < 0.2 else "neutral",
        icon_name="trending-down",
    )
    + render_metric_card(
        "ACCURACY // 分类准确率",
        f"{final_acc:.1%}",
        delta="本训练集达到 95%" if final_acc >= 0.95 else "本训练集未达 95%",
        delta_type="positive" if final_acc >= 0.9 else "neutral",
        icon_name="target",
    )
    + render_metric_card(
        "LEARNED WEIGHTS // 模型自主学得权重",
        f"[{w_final[0]:.2f}, {w_final[1]:.2f}]",
        delta=f"偏置截距 b = {b_final:.2f}",
        delta_type="neutral",
        icon_name="sliders",
    )
    + render_metric_card(
        "DECISION LINE // 学得的直线方程",
        f'<span style="font-size:1.02rem;">{hyperplane_str}</span>',
        delta="模型自动求解的决策分界面",
        delta_type="positive",
        icon_name="activity",
    )
    + "</div>"
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 智能调参诊断与物理瓶颈分析黄色提示框 (Diagnostic Alert Banner)
# ---------------------------------------------------------------------------
if final_acc < 0.90 or final_loss > 0.5:
    if dataset_name.lower() in ["xor", "moons", "circles", "spiral"]:
        st.warning(
            f"**[DIAGNOSTIC HINT // 智能诊断：单神经元物理极限]**\n\n"
            f"当前模型未能将数据集完全分开（当前准确率：**{final_acc:.1%}**，最终损失：**{final_loss:.4f}**）。\n\n"
            f"• **为什么画不准？** 当前选用的是 **{dataset_name.upper()} 非线性流形**。单神经元的数学本质是单个线性超平面（`w₁·x₁ + w₂·x₂ + b = 0`），只能在空间中画出一根笔直的直线。明斯基（Minsky）在 1969 年证明：单层线性感知机在数学上**绝对无法**将非线性流形（如异或 XOR、双月形）完全切开！\n\n"
            f"• **如何跑出 100% 正确结果？**\n"
            f"  1. **体验单神经元满分分类**：在左侧控制台将【分布类型】切换为 **Blobs (高斯聚类线性分布)**，或直接点击左上方 **【关卡 1 初出茅庐】**；\n"
            f"  2. **攻克非线性难题**：点击左侧导航栏前往 **M02·多层网络 (Multi-Layer Network)**，引入隐藏层神经元实现特征空间的非线性空间流形折叠！"
        )
    elif act_meta.id != "Sigmoid":
        st.warning(
            f"**[DIAGNOSTIC HINT // 智能诊断：激活函数特性影响]**\n\n"
            f"当前模型在线性数据集上未达最优收敛（当前准确率：**{final_acc:.1%}**，最终损失：**{final_loss:.4f}**）。\n\n"
            f"• **为什么未达 100%？** 当前激活函数选用了 **{act_meta.label}**。二分类感知器的标准输出应是将实数线性加权值平滑压缩至 `(0, 1)` 概率区间的 **Sigmoid** 激活函数。使用 ReLU 时，负向区域导数完全置零，可能引发神经元死区（Dying ReLU）导致直线停滞。\n\n"
            f"• **如何跑出 100% 正确结果？**\n"
            f"  1. 在左侧控制台将【激活函数】切换为 **Sigmoid (S型激活函数)**；\n"
            f"  2. 将【学习率】设为 **0.05 ~ 0.2** 黄金区间，点击即可见证 100% 满分收敛！"
        )
    elif lr > 1.0:
        st.warning(
            f"**[DIAGNOSTIC HINT // 智能诊断：学习率过大导致震荡]**\n\n"
            f"当前模型发生剧烈翻滚震荡（当前准确率：**{final_acc:.1%}**，最终损失：**{final_loss:.4f}**）。\n\n"
            f"• **为什么损失震荡？** 当前学习率过大（**LR = {lr}**），导致参数每次更新迈出的步长过大，在损失盆地两侧剧烈跳跃，无法沉降至全局最低点。\n\n"
            f"• **如何跑出正确结果？**\n"
            f"  1. 在左侧控制台将【学习率】下调至 **0.05 ~ 0.2**；\n"
            f"  2. 重新观察右侧损失曲线，将平滑单调俯冲至 0，准确率迅速攀升至 100%。"
        )
    else:
        st.warning(
            f"**[DIAGNOSTIC HINT // 智能诊断：训练尚未充分收敛]**\n\n"
            f"当前模型尚未完成充分训练（当前准确率：**{final_acc:.1%}**，最终损失：**{final_loss:.4f}**）。\n\n"
            f"• **优化建议**：适当提高【训练轮数】至 100~200 轮，或将【学习率】微调至 0.1 左右，帮助模型完成参数收敛。"
        )
# ---------------------------------------------------------------------------
total_steps = len(weight_trajectory)
if "m1_scrub_step" not in st.session_state or st.session_state["m1_scrub_step"] > total_steps:
    st.session_state["m1_scrub_step"] = total_steps
if "m1_player_state" not in st.session_state:
    st.session_state["m1_player_state"] = "idle"

# 计算全局特征范围 (固定坐标轴，永不自适应缩放抖动)
x_margin = 0.3
x_min, x_max = float(X[:, 0].min() - x_margin), float(X[:, 0].max() + x_margin)
y_min, y_max = float(X[:, 1].min() - x_margin), float(X[:, 1].max() + x_margin)

# 预先生成静态概率网格坐标 (40x40 极致流畅网格)
grid_res = 40
xx_static, yy_static = np.meshgrid(
    np.linspace(x_min, x_max, grid_res),
    np.linspace(y_min, y_max, grid_res),
)
grid_static = np.c_[xx_static.ravel(), yy_static.ravel()]

# 预提取分类散点数据
labels_static = y.ravel()
mask_0 = labels_static == 0
mask_1 = labels_static == 1


def get_analytical_line_points(w1: float, w2: float, b: float):
    """计算直线 w1*x1 + w2*x2 + b = 0 在显示范围内的精确两个端点"""
    pts = []
    if abs(w2) > 1e-6:
        y1 = -(w1 * x_min + b) / w2
        if y_min <= y1 <= y_max:
            pts.append((x_min, y1))
        y2 = -(w1 * x_max + b) / w2
        if y_min <= y2 <= y_max:
            pts.append((x_max, y2))
    if abs(w1) > 1e-6:
        x1 = -(w2 * y_min + b) / w1
        if x_min <= x1 <= x_max:
            pts.append((x1, y_min))
        x2 = -(w2 * y_max + b) / w1
        if x_min <= x2 <= x_max:
            pts.append((x2, y_max))

    uniq = []
    for p in pts:
        if not any(abs(p[0] - u[0]) < 1e-4 and abs(p[1] - u[1]) < 1e-4 for u in uniq):
            uniq.append(p)
    if len(uniq) >= 2:
        return [uniq[0][0], uniq[1][0]], [uniq[0][1], uniq[1][1]]
    return None, None


def make_boundary_figure(step_num: int):
    cur_w = weight_trajectory[step_num - 1]
    cur_b = bias_trajectory[step_num - 1]
    w1, w2 = float(cur_w.ravel()[0]), float(cur_w.ravel()[1])
    b = float(cur_b.ravel()[0])

    # 1. 极速概率场计算
    z_raw = grid_static @ cur_w + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(z_raw, -30, 30)))
    zz = probs.reshape(xx_static.shape)

    colorscale = [
        [0.0, "rgba(29, 78, 216, 0.25)"],  # 类别 0 (蓝)
        [0.5, "rgba(241, 245, 249, 0.5)"],  # 决策临界线
        [1.0, "rgba(190, 18, 60, 0.25)"],  # 类别 1 (红)
    ]

    fig = go.Figure()

    # 概率等高填充 (无轮廓线，零 SVG 重绘抖动)
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, grid_res),
            y=np.linspace(y_min, y_max, grid_res),
            z=zz,
            colorscale=colorscale,
            showscale=False,
            contours=dict(showlines=False, coloring="fill"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 解析几何直线 (GPU 级毫秒渲染)
    lx, ly = get_analytical_line_points(w1, w2, b)
    if lx is not None:
        fig.add_trace(
            go.Scatter(
                x=lx,
                y=ly,
                mode="lines",
                line=dict(color="#0f172a", width=3.5),
                name="决策分界线 (Line: P=0.5)",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # 散点
    fig.add_trace(
        go.Scatter(
            x=X[mask_0, 0],
            y=X[mask_0, 1],
            mode="markers",
            name="Class 0 (蓝)",
            marker=dict(
                size=8, color="#1d4ed8", line=dict(width=1.5, color="#ffffff"), opacity=0.9
            ),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[mask_1, 0],
            y=X[mask_1, 1],
            mode="markers",
            name="Class 1 (红)",
            marker=dict(
                size=8, color="#be123c", line=dict(width=1.5, color="#ffffff"), opacity=0.9
            ),
            hoverinfo="skip",
        )
    )

    # 彻底固化布局与坐标轴 (禁止任何边距抖动与自适应缩放)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        height=400,
        transition=dict(duration=480, easing="cubic-in-out"),
        margin=dict(l=40, r=20, t=10, b=35),
        uirevision="locked_view",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=10, family="JetBrains Mono"),
        ),
        xaxis=dict(
            range=[x_min, x_max],
            fixedrange=True,
            autorange=False,
            gridcolor="rgba(15, 23, 42, 0.05)",
            tickfont=dict(size=10, family="JetBrains Mono"),
            title=dict(text="Feature x₁", font=dict(size=11, family="JetBrains Mono")),
        ),
        yaxis=dict(
            range=[y_min, y_max],
            fixedrange=True,
            autorange=False,
            gridcolor="rgba(15, 23, 42, 0.05)",
            tickfont=dict(size=10, family="JetBrains Mono"),
            title=dict(text="Feature x₂", font=dict(size=11, family="JetBrains Mono")),
        ),
    )
    return fig


def render_line_equation_html(step_num: int):
    cur_w = weight_trajectory[step_num - 1]
    cur_b = bias_trajectory[step_num - 1]
    w1, w2 = float(cur_w.ravel()[0]), float(cur_w.ravel()[1])
    b = float(cur_b.ravel()[0])
    cur_sign = "+" if b >= 0 else "-"
    cur_line_eq = f"{w1:.2f}x₁ + {w2:.2f}x₂ {cur_sign} {abs(b):.2f} = 0"

    return (
        f'<div style="font-family: monospace; font-size: 0.82rem; font-weight: 700; color: #1e40af; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.3rem 0.6rem; margin-bottom: 0.4rem; height: 28px; line-height: 20px; box-sizing: border-box;">'
        f'分界实线方程: <span style="color: #0f172a; font-weight: 800;">{cur_line_eq}</span>'
        f"</div>"
    )


def render_status_html(step_num: int, player_state: str = "idle"):
    status_styles = {
        "playing": ("#1d4ed8", "#eff6ff", "#bfdbfe", "▶ 演播中"),
        "paused": ("#92400e", "#fffbeb", "#fde68a", "Ⅱ 已暂停 · 可观察当前参数"),
        "idle": ("#047857", "#ecfdf5", "#a7f3d0", "准备就绪"),
    }
    badge_color, badge_bg, badge_border, label = status_styles.get(
        player_state, status_styles["idle"]
    )
    play_status = f"{label} · STEP {step_num}/{total_steps}"

    return (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem 1rem;margin-bottom:0.8rem;box-shadow:0 2px 8px rgba(15,23,42,0.03);">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.78rem;font-weight:800;color:#1e40af;text-transform:uppercase;letter-spacing:0.04em;">'
        f"{anchor_badge('D', 'purple')} [TIME-TRAVEL PLAYER // 训练时空演播厅]"
        f"</div>"
        f'<span style="font-size:0.75rem;font-weight:700;color:{badge_color};font-family:monospace;background:{badge_bg};border:1px solid {badge_border};padding:0.12rem 0.5rem;border-radius:4px;">{play_status}</span>'
        f"</div>"
        f"</div>"
    )


player_payload = build_perceptron_payload(
    weight_trajectory,
    bias_trajectory,
    X,
    mask_0,
    mask_1,
    (x_min, x_max),
    (y_min, y_max),
)
render_player_controls(player_payload)

final_w_arr = np.asarray(weight_trajectory[-1]).ravel()
final_b_arr = np.asarray(bias_trajectory[-1]).ravel()
w1_val = float(final_w_arr[0]) if len(final_w_arr) > 0 else 0.0
w2_val = float(final_w_arr[1]) if len(final_w_arr) > 1 else 0.0
b_val = float(final_b_arr[0]) if len(final_b_arr) > 0 else 0.0
final_loss = history["loss"][-1] if history["loss"] else 0.0
final_acc = history["accuracy"][-1] if history["accuracy"] else 0.0

render_live_param_status_bar(
    title="PERCEPTRON DYNAMICS // 空间决策方程与收敛参数",
    badges=[
        {"label": "w₁", "value": f"{w1_val:+.3f}", "color": "blue"},
        {"label": "w₂", "value": f"{w2_val:+.3f}", "color": "amber"},
        {"label": "b", "value": f"{b_val:+.3f}", "color": "purple"},
    ],
    metrics=[
        ("学习率 η", f"{lr}"),
        ("最终 Loss", f"{final_loss:.4f}"),
        ("最终 Acc", f"{final_acc:.1%}"),
    ],
    tag=f"CONVERGED · {len(weight_trajectory)} STEPS",
    tag_color="emerald" if final_acc >= 0.9 else "amber",
)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown(
        f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
        f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">DECISION MANIFOLD // 空间决策流形与分界实线演进</span></div>',
        unsafe_allow_html=True,
    )
    render_boundary_canvas(player_payload)
    with st.expander("[HOW TO READ // 读图指南] [D] 空间决策流形与分界实线"):
        st.markdown(
            """
            * **红蓝圆点**是两类训练样本；**黑色实线**是预测概率 $P=0.5$ 的分界线。
            * **背景淡色流形**表示各位置的预测概率。
            * 暂停后对照上方方程，观察 $w_1$、$w_2$、$b$ 如何旋转和平移直线。
            """
        )

with col_right:
    st.markdown(
        f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
        f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">LOSS & ACCURACY CONVERGENCE // 损失与准确率收敛曲线</span></div>',
        unsafe_allow_html=True,
    )
    fig_loss = plot_loss_curve(history)
    fig_loss.update_layout(height=432, margin=dict(l=40, r=20, t=25, b=35))
    st.plotly_chart(fig_loss, width="stretch", key="m1_loss_player")
    with st.expander("[HOW TO READ // 读图指南] [E] 损失与准确率曲线"):
        st.markdown(
            """
            * **Loss** 是预测误差，越低越好；**Accuracy** 是答对比例，越高越好。
            * 暂停后用当前 Step 对照曲线横轴，判断参数变化发生在收敛的哪个阶段。
            """
        )

st.markdown(
    f'<div id="region-f" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-top:1.2rem;margin-bottom:0.5rem;">'
    f'{anchor_badge("F", "rose")} <span style="font-weight:800;color:#9f1239;font-size:0.86rem;">WEIGHT TRAJECTORY // 权重参数空间寻优轨迹</span></div>',
    unsafe_allow_html=True,
)
render_trajectory_canvas(player_payload)
with st.expander("[HOW TO READ // 读图指南] [F] 权重参数空间寻优轨迹图"):
    st.markdown(
        """
        * **横轴 $w_1$ 与纵轴 $w_2$**是两个核心权重；等高线表示损失盆地。
        * 连续折线展示参数沿梯度下坡的过程，暂停可检查任意中间位置。
        """
    )


# 深度知识学习指南 (折叠微观原理解析)
render_deep_dive_card("单神经元感知器核心参数与激活函数", [act_meta, opt_meta])

# ---------------------------------------------------------------------------
# 零基础进阶专家必读：深度学习核心名词通俗速查
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 核心公式逐字拆解与深度学习名词通俗全解", expanded=True
):
    st.markdown(
        r"""
        ### 0. 核心公式逐字拆解：$Z = XW + b$ 与 $\hat{y} = \sigma(Z)$
        这是整个深度学习世界最基础、最通用的**第一核心公式**：

        | 符号 | 中文名称 | 矩阵形状 (Shape) | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$X$** | **输入特征矩阵 (Inputs)** | $N \\times 2$ (样本数 × 特征数) | **你喂给机器人的原始数据**。比如 $N$ 个数据点，每个点有两个属性：$x_1$ 是横坐标，$x_2$ 是纵坐标。 |
        | **$W$** | **权重矩阵 (Weights)** | $2 \\times 1$ (特征数 × 输出数) | **直尺的旋转角度（重要性系数）**。包含 $[w_1, w_2]$ 两个数，分别给 $x_1$ 和 $x_2$ 乘以权重，决定了直线在平面上的**倾斜斜率**。 |
        | **$b$** | **偏置项 (Bias)** | $1 \\times 1$ (标量常数) | **直尺的上下平移距离（截距）**。若没有 $b$，直线被强行绑在原点 $(0,0)$ 动弹不得；有了 $b$，直线才能在平面任意位置平移！ |
        | **$Z$** | **线性加权和 (Logits)** | $N \\times 1$ | $x_1 w_1 + x_2 w_2 + b$ 的计算结果。正数代表在直线某一侧，负数代表在另一侧，数值绝对值代表距离直线的远近。 |
        | **$\\sigma$** | **激活函数 (Activation)** | 函数映射 $\\mathbb{R} \\to (0,1)$ | **概率压缩器**。比如 Sigmoid $\\sigma(Z) = \\frac{1}{1 + e^{-Z}}$，把从 $-\\infty$ 到 $+\\infty$ 的任意实数平滑压缩到 $0 \\sim 1$ 之间。 |
        | **$\\hat{y}$** | **最终预测概率 (Output)** | $N \\times 1$ | 模型对每个点的最终预测结果。$\\hat{y} \\ge 0.5$ 判定为红点，$\\hat{y} < 0.5$ 判定为蓝点。 |

        ---

        ### 1. 什么是【梯度 (Gradient)】？—— “最陡的下山方向”
        * **生活比喻**：想象你被蒙上双眼放在大雾弥漫的崇山峻岭中，任务是走到山谷最低处（损失最小处）。你看不清全貌，但用脚踩一踩脚下的地面，能感受到**哪个方向坡度最陡、往哪里下坠最快**。这个“坡度最陡的指向”就是**梯度**！
        * **本质机理**：梯度是一个由偏导数组成的向量 $\\nabla L = [\\frac{\\partial L}{\\partial w_1}, \\frac{\\partial L}{\\partial w_2}, \\frac{\\partial L}{\\partial b}]$，它告诉你**权重微调一点点，总损失会往哪个方向变化**。沿着梯度的**反方向**走一步，损失就会减少！

        ---

        ### 2. 什么是【损失 (Loss)】？—— “做错题的扣分罚分”
        * **生活比喻**：相当于老师给你的试卷打分。如果你把红点猜成蓝点，老师就扣你 10 分；如果你百分之百猜对，扣分就是 0。
        * **本质机理**：衡量模型当前预测与真实标签之间的差距（误差）。训练模型的目标就是通过不断调整参数，让 **Loss 越接近 0 越好**。

        ---

        ### 3. 什么是【权重 (Weights) $w$】与【偏置 (Bias) $b$】？—— “直线的旋转与平移”
        * **生活比喻**：机器人手里拿着一根直尺：
          * **权重 $w_1, w_2$**：决定直尺在平面上的**倾斜角度（斜率）**；
          * **偏置 $b$**：决定直尺在平面上的**平行移动距离（截距）**。
        * **两者配合**：通过旋转和平移直尺，尝试找到能降低当前训练损失的分界位置；含噪或不可线性分的数据未必能被一条直线全部分对。

        ---

        ### 4. 什么是【学习率 (Learning Rate / LR)】？—— “每一步迈出的步长”
        * **生活比喻**：下山时，你每一步迈出多大距离：
          * **步子太小 (如 0.0001)**：像蚂蚁爬行，走了一万步还在山顶；
          * **步子太大 (如 2.0)**：像一步跨出两米，直接从山左侧飞到了山右侧峭壁，在山谷两头疯狂乱跳（无法收敛）；
          * **中间步长 (本示例可试 0.05~0.1)**：在当前数据与尺度下可能较稳定；换模型、批量或归一化后需重新比较。

        ---

        ### 5. 什么是【前向传播 (Forward)】与【反向传播 (Backward)】？
        * **前向传播 (Forward)**：**做题过程**。输入特征 $(x_1, x_2)$ 进入神经元，计算出预测结果 $\hat{y}$，并算出当前扣了多少分 (Loss)。
        * **反向传播 (Backward)**：**订正错题过程**。拿着扣分结果，沿着数学公式倒推，找出究竟是 $w_1$ 调偏了还是 $b$ 调高了，计算出每个参数的梯度。
        * **参数更新 (Step)**：根据梯度，真正把 $w$ 和 $b$ 往正确方向微调一步。

        ---

        ### 6. 经典 Rosenblatt 感知器 vs. 现代可微逻辑神经元 (Logistic Neuron)
        * **经典 Rosenblatt 感知器 (1958)**：采用非连续阶跃激活函数 $f(z) = \text{sign}(z)$，输出硬分类标签 $\pm 1$。因为阶跃函数的导数在非零处恒为 0，无法使用微积分梯度下降，依赖于分类错误时的离散纠错规则 $w \leftarrow w + \eta (y - \hat{y}) x$。仅在线性可分数据集上保证有限步收敛（感知器收敛定理）。
        * **现代逻辑神经元 (Logistic Neuron)**：采用光滑可微的 Sigmoid 激活函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$，将实数加权映射为连续后验概率 $P(Y=1|X) \in (0, 1)$，并通过二元交叉熵 (BCE) 损失进行端到端微积分反向传播（极大似然估计）。
        """
    )
