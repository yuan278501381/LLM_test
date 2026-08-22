# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 2: 多层网络与活性探针 (Deep Networks & Neuron Probe) - 零基础入门保姆级教学平台

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
    render_deep_dive_card,
    render_network_params,
    render_presets_selector,
    render_probe_point_selector,
    render_training_params,
)
from dashboard.constants.knowledge import ACTIVATIONS, INITIALIZERS, OPTIMIZERS
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_preset_badge,
)
from dashboard.utils.state import (
    get_dataset,
    resolve_activation,
    resolve_initializer,
    resolve_optimizer,
)
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
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="多层神经网络与活性探针入门",
    plain_intro=(
        "<b>单神经元只能画 1 根直线，而多层网络像一双能折纸的手！</b><br>"
        "通过把很多个神经元串联成「隐藏层（Hidden Layers）」，模型可以把多根直线折叠拼装成<b>弯曲的圆环、螺旋和不规则多边形</b>，"
        "轻松解决所有复杂的非线性分类问题。"
    ),
    hyperparams_desc=(
        "• <b>经典实验预设</b>：一键载入历史著名场景（如 XOR 难题、双螺旋等）。<br>"
        "• <b>隐藏层深度 (Depth)</b>：串联多少层神经元，网络越深，折叠能力越强。<br>"
        "• <b>神经元数 (Neurons)</b>：每一层有多少个节点（相当于多少根基础折线）。<br>"
        "• <b>探针坐标 (x₁, x₂)</b>：在地图上插一根虚拟探针，像听诊器一样实时听取内部反应。"
    ),
    telemetry_desc=(
        "• <b>左图（网络拓扑与探针响应）</b>：展示信号从左往右流过的过程，<b>深蓝色代表该神经元被强烈点亮激活</b>。<br>"
        "• <b>右图（空间决策流形）</b>：展示模型最终在空间中折叠出的弯曲分界面与探针定位。<br>"
        "• <b>底图（梯度流分布）</b>：检查深层网络信号是否健康传递（防止梯度归零消失）。"
    ),
    experiments=[
        "<b>第 1 步【一键破解 XOR 难题】</b>：在左侧【经典实验预设】中选择 <code>XOR 历史困境与破解</code>，观察 2 层网络如何用弯曲的曲线完美切开对角分布的异或数据！",
        "<b>第 2 步【体验探针听诊】</b>：在左侧拖动【探针坐标 x₁、x₂】滑动条，观察左下角拓扑图中，哪些神经元被染成了深蓝色（被强力激活）！",
        "<b>第 3 步【挑战双螺旋极限】</b>：选择预设 <code>双螺旋奇点挑战</code>，见证 3 层深度网络如何像拉花一样把极其复杂的双螺旋流形完全解开！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板 (含预设，由 knowledge 元数据驱动)
# ---------------------------------------------------------------------------
preset = render_presets_selector(key_prefix="m2_")

if preset:
    dataset_name = preset["dataset"]
    render_preset_badge(
        "已激活经典预设方案",
        f"数据集: {dataset_name} | 深度: {preset['n_layers']} 层 | 激活: {preset['activation']} | 初始化: {preset['initializer']} | 优化器: {preset['optimizer']}",
    )
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
        key_prefix="m2_", default_dataset="moons"
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

act_cls = resolve_activation(activation_name)
init_name = resolve_initializer(initializer)
opt_cls = resolve_optimizer(optimizer_name)

# 构建多层网络
model = Sequential()
current_dim = 2
for i in range(n_layers):
    out_dim = neurons_per_layer[i]
    model.add(Dense(current_dim, out_dim, initializer=init_name))
    model.add(act_cls())
    current_dim = out_dim

# 输出层
model.add(Dense(current_dim, 1, initializer=init_name))
model.add(resolve_activation("Sigmoid")())

loss_fn = BinaryCrossEntropy()
optimizer = opt_cls(learning_rate=float(lr))

# 训练网络
history = model.train(
    X, y, loss_fn=loss_fn, optimizer=optimizer, epochs=int(epochs), batch_size=batch_size
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
    grad_health_status = "VANISHING (梯度消失)"
elif min_grad_norm > 50.0:
    grad_health_status = "EXPLODING (梯度爆炸)"
else:
    grad_health_status = "HEALTHY (梯度健康)"

# ---------------------------------------------------------------------------
# 遥测指标卡 (中英双语标签)
# ---------------------------------------------------------------------------
grid_html = (
    '<div class="metric-grid">'
    + render_metric_card("FINAL LOSS // 最终损失", f"{final_loss:.4f}", delta="CONVERGED" if final_loss < 0.2 else "TRAINING", delta_type="positive" if final_loss < 0.2 else "neutral", icon_name="trending-down")
    + render_metric_card("ACCURACY // 准确率", f"{final_acc:.1%}", delta=dataset_name.upper(), delta_type="positive" if final_acc >= 0.9 else "neutral", icon_name="target")
    + render_metric_card("PROBE RESPONSE // 探针响应", f"{probe_prob:.1%}", delta=f"CLASS 预测类别 {probe_pred_class}", delta_type="positive" if probe_pred_class == 1 else "neutral", icon_name="crosshair")
    + render_metric_card("GRADIENT NORM // 梯度范数", f"{min_grad_norm:.2e}", delta=grad_health_status, delta_type="positive" if "HEALTHY" in grad_health_status else "negative", icon_name="activity")
    + '</div>'
)
st.markdown(grid_html, unsafe_allow_html=True)

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

with st.expander("[HOW TO READ // 读图指南] 网络拓扑、神经元点亮与决策分界线", expanded=False):
    st.markdown(
        """
        * **左图【网络拓扑图】**：
          * **圆圈（神经元）**：从左往右是输入层 $\\to$ 隐藏层 $\\to$ 输出层。**圆圈被染成深蓝色**，代表当前探针位置 $(x_1, x_2)$ 强烈**点亮（激活）**了该神经元！
          * **连线（权重连接）**：**蓝色连线**代表正权重（促进兴奋），**红色连线**代表负权重（抑制信号），**线条越粗**代表权重数值越大、影响力越强。
        * **右图【空间决策流形】**：
          * **黑色加粗实线 (Line: P=0.5)**：这是多层网络通过非线性折叠拼出的**弯曲分界线**，不再是单神经元死板的一根直线！
          * **黄色十字交叉点 (PROBE)**：你在左侧拖动的探针定位点，右图显示它位于哪个分类区域，左图同步显示该点激发的神经元通路！
        * 🎯 **【最优趋势】**：弯曲的黑色决策线把红蓝两类完全分隔开，准确率达到 95%+；探针在不同颜色区域时，左侧网络能清晰看到不同的激活通路切换。
        """
    )

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

with st.expander("[HOW TO READ // 读图指南] 神经元失活、梯度消失与梯度爆炸诊断", expanded=False):
    st.markdown(
        """
        * **左图【逐层神经元激活热力图】**：
          * **横轴**是该层神经元编号，**纵轴**是输入样本。颜色越亮代表激活值越高。
          * ⚠️ **【异常现象：Dead ReLU (神经元死亡)】**：如果整列都呈现暗淡无光或全灰（数值为 0），说明该神经元进入负值区彻底“死掉”了，不再参与学习！
          * [NOTE] **【调优方案】**：换用 `LeakyReLU` / `GELU`，或者调小初始学习率。
        * **右图【梯度流分布直方图】**：
          * 展示反向传播时每一层权重接收到的梯度大小。
          * 🎯 **【健康趋势】**：各层梯度处于相近的数量级（如 $0.01 \\sim 1.0$ 之间），信号能顺畅传递到第一层。
          * ⚠️ **【异常现象：梯度消失 (Vanishing)】**：浅层（靠近输入的层）梯度极小（如 $< 10^{-5}$），柱子几乎贴底，网络根本学不动 $\\implies$ 换用 `He/Xavier` 初始化与 `ReLU`。
          * ⚠️ **【异常现象：梯度爆炸 (Exploding)】**：深层梯度突然达到几十上百，导致 Loss 变成 NaN $\\implies$ 调小学习率或开启梯度裁剪。
        """
    )

# 深度知识学习指南 (折叠微观原理解析)
act_meta = ACTIVATIONS.get(activation_name, ACTIVATIONS.get(activation_name.split(" ")[0], ACTIVATIONS["ReLU"]))
init_meta = INITIALIZERS.get(initializer, INITIALIZERS.get(initializer.split(" ")[0].lower(), INITIALIZERS["he"]))
opt_meta = OPTIMIZERS.get(optimizer_name, OPTIMIZERS.get(optimizer_name.split(" ")[0], OPTIMIZERS["Adam"]))

render_deep_dive_card("多层网络拓扑、激活与初始化微观解析", [act_meta, init_meta, opt_meta])

# ---------------------------------------------------------------------------
# 零基础进阶：多层网络通俗名词速查
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 多层深度网络核心名词通俗全解", expanded=False):
    st.markdown(
        """
        ### 1. 什么是【隐藏层 (Hidden Layer)】？—— “特征加工车间”
        * **生活比喻**：做菜时，输入是生肉和蔬菜，输出是美味佳肴。中间的隐藏层就是“洗菜、切菜、调味、煎炒”一道道加工工序。
        * **本质机理**：第一层把点连成线，第二层把线拼成拐角和多边形，第三层拼成复杂的闭合环。网络越深，能抽取的特征越抽象！
        
        ---
        
        ### 2. 什么是【非线性激活函数】？—— “给笔装上拐弯功能”
        * **生活比喻**：如果没有激活函数，无论你叠 100 层网络，数学上 $W_3(W_2(W_1 x + b_1) + b_2) + b_3$ 依然只是一根笔直的直线（线性组合合并）。
        * **本质机理**：`ReLU`、`Sigmoid` 等激活函数给直线加入了“折叠和弯曲能力”，让网络能够画出任意复杂的曲线！
        
        ---
        
        ### 3. 什么是【梯度消失 (Vanishing Gradient)】？—— “话筒声音传丢了”
        * **生活比喻**：大队长给中队长讲话，中队长给小队长转达，小队长给队员传达。如果每次传达声音都衰减一半，传到第 10 个人时已经完全听不清在说什么了。
        * **本质机理**：反向传播链式求导时，多个小导数连乘导致浅层梯度趋近于 0，靠近输入的权重完全停止更新。
        """
    )

