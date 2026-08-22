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
import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_loss_curve,
    plot_weight_trajectory,
)
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_deep_dive_card,
)
from dashboard.constants.knowledge import ACTIVATIONS, OPTIMIZERS
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from dashboard.utils.state import get_dataset, resolve_activation, resolve_optimizer
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Single Perceptron · NN Playground",
    layout="wide",
)

apply_custom_theme()

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
    title="单神经元感知器入门与读图指南",
    plain_intro=(
        "<b>1. 单神经元像一个拿尺子画直线的机器人</b>：输入特征 $(x_1, x_2)$ 是数据在地图上的横纵坐标。<br>"
        "机器人的任务就是在二维平面上画出一根直线 <code>w₁·x₁ + w₂·x₂ + b = 0</code>，"
        "把<b>蓝色点</b>（类别 0）和<b>红色点</b>（类别 1）干净利落地切开！<br><br>"
        "<b>2. 怎么看左边的图？</b><br>"
        "• <b>背景的蓝色/红色渐变区域</b>：代表模型对整个平面每个位置的【预测信心】（越深蓝越确信是蓝点，越深红越确信是红点）。<br>"
        "• <b>中间唯一的粗黑实线 <code>[DECISION LINE] 决策分界线 (Line: P=0.5)</code></b>：这就是机器人真正画出的<b>唯一那根分界线</b>！线两侧的概率正好是 50% 临界点。"
    ),
    hyperparams_desc=(
        "• <b>分布类型</b>：选择不同形状的数据集（如 Blobs 简单，Moons 弯曲）。<br>"
        "• <b>激活函数</b>：如 <code>Sigmoid</code>，把输出压缩到 0~1 之间表示概率。<br>"
        "• <b>学习率 (LR)</b>：机器人每次看错后调整尺子的步子大小。<br>"
        "• <b>训练轮数</b>：机器人总共练习画线的迭代次数。"
    ),
    telemetry_desc=(
        "• <b>最终训练损失 (Loss)</b>：做错题的惩罚分，<b>越接近 0 代表分得越准</b>。<br>"
        "• <b>分类准确率 (Acc)</b>：做对的题目比例，<b>100% 代表完全切对</b>。<br>"
        "• <b>学得权重 [w₁, w₂] 与偏置 b</b>：机器人经过训练后自己算出的直线参数（w 决定直线的斜率方向，b 决定直线上下平移）。<br>"
        "• <b>学得的直线方程</b>：在左侧图表上绘制出的黑色决策分界实线公式。"
    ),
    experiments=[
        "<b>第 1 步【寻找那根线】</b>：在左侧选择 <code>Blobs (高斯聚类)</code>，观察左侧图表中间那根<b>加粗的黑色决策分界线</b>，看它如何恰好把红蓝两团点隔开！",
        "<b>第 2 步【体验调参】</b>：试着把左侧【学习率】改成 <code>0.001</code>（步子太小），看看需要很多轮才能画准；再改成 <code>1.5</code>（步子太大），看看直线是不是开始乱晃！",
        "<b>第 3 步【见证单神经元的局限】</b>：把【分布类型】换成 <code>XOR (正交异或)</code> 或 <code>Moons (双月形)</code>。你会发现一根笔直的线无论怎么转都无法完美切开它们！这就是为什么我们需要<b>多层网络（深度学习）</b>！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数面板 (由 knowledge 元数据驱动，带丰富 Tooltip)
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m1_", default_dataset="blobs"
)

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")

act_list = [m for m in ACTIVATIONS.values() if m.id != "Softmax"]
act_labels = [m.label for m in act_list]
selected_act_label = st.sidebar.selectbox(
    "激活函数",
    act_labels,
    help="非线性激活函数。对单神经元而言，Sigmoid 将实数加权值映射为 (0,1) 的二分类置信概率。",
    key="m1_act",
)
act_meta = next(m for m in act_list if m.label == selected_act_label)

opt_list = list(OPTIMIZERS.values())
opt_labels = [m.label for m in opt_list]
selected_opt_label = st.sidebar.selectbox(
    "优化器",
    opt_labels,
    help="梯度下降算法。负责根据损失对权重 (w₁, w₂) 和偏置 b 的梯度调整参数大小。",
    key="m1_opt",
)
opt_meta = next(m for m in opt_list if m.label == selected_opt_label)

col1, col2 = st.sidebar.columns(2)
with col1:
    lr = st.number_input(
        "学习率 (LR)",
        0.001, 2.0, 0.1, step=0.01, format="%.3f",
        help="参数更新步长 $\\eta$。过大导致跳过最优点剧烈震荡，过小导致收敛缓慢。",
        key="m1_lr",
    )
with col2:
    epochs = st.slider(
        "训练轮数",
        10, 500, 100, step=10,
        help="全量数据集迭代轮数。",
        key="m1_epochs",
    )

# ---------------------------------------------------------------------------
# 数据生成与模型训练
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)

# 构建单神经元模型: Input(2) -> Dense(2, 1) -> Activation
model = Sequential()
dense_layer = Dense(2, 1, initializer="random")
model.add(dense_layer)

act_cls = resolve_activation(act_meta.id)
model.add(act_cls())

loss_fn = BinaryCrossEntropy()
opt_cls = resolve_optimizer(opt_meta.id)
optimizer = opt_cls(learning_rate=float(lr))

# 记录权重轨迹
weight_trajectory = [dense_layer.weights.copy()]

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
    '<div class="metric-grid">'
    + render_metric_card("FINAL LOSS // 最终训练损失", f"{final_loss:.4f}", delta="已收敛 (CONVERGED)" if final_loss < 0.2 else "训练中 (TRAINING)", delta_type="positive" if final_loss < 0.2 else "neutral", icon_name="trending-down")
    + render_metric_card("ACCURACY // 分类准确率", f"{final_acc:.1%}", delta="达标 (OPTIMAL)" if final_acc >= 0.95 else "收敛中", delta_type="positive" if final_acc >= 0.9 else "neutral", icon_name="target")
    + render_metric_card("LEARNED WEIGHTS // 模型自主学得权重", f"[{w_final[0]:.2f}, {w_final[1]:.2f}]", delta=f"偏置截距 b = {b_final:.2f}", delta_type="neutral", icon_name="sliders")
    + render_metric_card("DECISION LINE // 学得的直线方程", f'<span style="font-size:1.02rem;">{hyperplane_str}</span>', delta="模型自动求解的决策分界面", delta_type="positive", icon_name="activity")
    + '</div>'
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 可视化布局
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    fig_boundary = plot_decision_boundary(
        model, X, y, title=f"DECISION MANIFOLD // {act_meta.id.upper()} 决策流形"
    )
    st.plotly_chart(fig_boundary, use_container_width=True)
    with st.expander("[HOW TO READ // 读图指南] 空间决策流形与分界实线", expanded=False):
        st.markdown(
            """
            * **横轴 $X_1$ 与纵轴 $X_2$**：样本的两个特征坐标（例如身高与体重、温度与湿度）。
            * **红蓝圆点**：两类真实的训练样本点（红色代表类别 0，蓝色代表类别 1）。
            * **加粗黑色实线 `[DECISION LINE]`**：模型画出的分类决策分界线（满足预测概率 $P=0.5$ 的分水岭）。
            * **背景淡色流形**：模型对全平面每一个坐标点的预测概率（越红说明模型越有把握判定为类别 0，越蓝越有把握为类别 1）。
            * **[OPTIMAL // 最优形态]**：黑色实线稳健地横穿在红蓝两堆点正中央，没有任何错分点。
            """
        )

with col_right:
    fig_loss = plot_loss_curve(history)
    st.plotly_chart(fig_loss, use_container_width=True)
    with st.expander("[HOW TO READ // 读图指南] 损失收敛 (Loss) 与准确率 (Accuracy) 曲线", expanded=False):
        st.markdown(
            """
            * **两图横轴【训练轮数 (Epoch)】**：
              - 模型把全部训练样本**从头到尾完整做过几遍试卷**（100 轮代表重温并反思了 100 遍错题）。
            * **左子图纵轴【损失误差 (Loss)】**：
              - **模型做错题目的扣分罚分**。Loss 越小，说明预测概率与真实答案越接近（0 代表完全没有误差）。
              - **[OPTIMAL // 最优形态]**：呈现**"大滑梯型"**，从最初的高误差急剧下跌并平稳贴近 0。绿色菱形标出全过程最低损失点。
            * **右子图纵轴【分类准确率 (Accuracy)】**：
              - **模型答对题目的百分比**（$0.0 \\sim 1.0$，即 $0\\% \\sim 100\\%$）。
              - **[OPTIMAL // 最优形态]**：从初始约 $0.5$（$50\\%$ 随机瞎猜）迅速爬坡，最终平稳锁定在 $1.0$（$100\\%$ 全对）。
            """
        )

# 底部权重轨迹图
render_section_heading("权重参数空间寻优轨迹 (Weight Trajectory)", icon_name="crosshair", subtext="参数 (w₁, w₂) 从初始位置沿损失梯度向全局最优收敛的连续路径：")
fig_traj = plot_weight_trajectory(weight_trajectory)
st.plotly_chart(fig_traj, use_container_width=True)
with st.expander("[HOW TO READ // 读图指南] 权重参数空间寻优轨迹图", expanded=False):
    st.markdown(
        """
        * **横轴 $w_1$ 与纵轴 $w_2$**：单神经元的两个核心权重参数。
        * **等高线椭圆**：损失函数在地表形成的盆地地形（越靠中央，损失 Loss 越低）。
        * **连续折线轨迹**：参数从随机出发点（小圆圈）顺着梯度下坡滚入盆地最低点（全局最优解）的过程。
        """
    )

# 深度知识学习指南 (折叠微观原理解析)
render_deep_dive_card("单神经元感知器核心参数与激活函数", [act_meta, opt_meta])

# ---------------------------------------------------------------------------
# 零基础进阶专家必读：深度学习核心名词通俗速查
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 核心公式逐字拆解与深度学习名词通俗全解", expanded=True):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：$Z = XW + b$ 与 $\\hat{y} = \\sigma(Z)$
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
        * **两者配合**：通过旋转和平移直尺，直到找到一个完美的角度与位置，把红蓝两团数据一刀切开！
        
        ---
        
        ### 4. 什么是【学习率 (Learning Rate / LR)】？—— “每一步迈出的步长”
        * **生活比喻**：下山时，你每一步迈出多大距离：
          * **步子太小 (如 0.0001)**：像蚂蚁爬行，走了一万步还在山顶；
          * **步子太大 (如 2.0)**：像一步跨出两米，直接从山左侧飞到了山右侧峭壁，在山谷两头疯狂乱跳（无法收敛）；
          * **黄金步长 (如 0.05~0.1)**：既能稳步下山，又不会踏空跌落。
        
        ---
        
        ### 5. 什么是【前向传播 (Forward)】与【反向传播 (Backward)】？
        * **前向传播 (Forward)**：**做题过程**。输入特征 $(x_1, x_2)$ 进入神经元，计算出预测结果 $\\hat{y}$，并算出当前扣了多少分 (Loss)。
        * **反向传播 (Backward)**：**订正错题过程**。拿着扣分结果，沿着数学公式倒推，找出究竟是 $w_1$ 调偏了还是 $b$ 调高了，计算出每个参数的梯度。
        * **参数更新 (Step)**：根据梯度，真正把 $w$ 和 $b$ 往正确方向微调一步。
        """
    )

