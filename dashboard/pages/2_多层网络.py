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
    anchor_badge,
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
    title="多层神经网络与空间交互地图",
    blueprint_sections=[
        {"id": "A", "name": "控制台与探针", "desc": "在左侧侧边栏调节网络层数、神经元数与虚拟探针坐标", "color": "amber", "target_id": "region-a"},
        {"id": "B", "name": "教学指引", "desc": "当前卡片：通俗理解隐藏层如何像折纸一样折叠出复杂弯曲分界面", "color": "blue", "target_id": "region-b"},
        {"id": "C", "name": "实时遥测指标", "desc": "监测参数量、最终损失与全网神经元平均激活度", "color": "emerald", "target_id": "region-c"},
        {"id": "D", "name": "网络拓扑点亮", "desc": "微观观察探针信号在层层神经元之间被点亮激发的物理过程", "color": "purple", "target_id": "region-d"},
        {"id": "E", "name": "弯曲决策流形", "desc": "观察多层网络如何画出优雅的弯曲圆环包裹住复杂月牙数据", "color": "blue", "target_id": "region-e"},
    ],
    plain_intro=(
        f"<b>单神经元只能画 1 根直线，而多层网络像一双能折纸的手！</b><br>"
        f"通过在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把多个神经元串联成「隐藏层」，模型可以在 {anchor_badge('[E. 决策流形图]', 'blue', target_id='region-e')} 把多根直线折叠拼装成<b>弯曲的圆环、螺旋和复杂多边形</b>，"
        f"轻松解决所有单层网络无法切开的非线性难题。<br>"
        f"同时在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 插入一根虚拟探针，就能在 {anchor_badge('[D. 拓扑图]', 'purple', target_id='region-d')} 实时像听诊器一样看到每一个神经元是否被点亮！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 左侧控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>经典实验预设</b>：一键载入历史著名场景（如 XOR 难题、双螺旋等）。<br>"
        f"• <b>隐藏层深度 (Depth)</b>：串联多少层神经元，网络越深，折叠能力越强。<br>"
        f"• <b>神经元数 (Neurons)</b>：每一层有多少个节点（相当于多少根基础折线）。<br>"
        f"• <b>探针坐标 (x₁, x₂)</b>：在地图上插一根虚拟探针，实时听取神经元反应。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[C. 实时指标]', 'emerald', target_id='region-c')} 查看</b>：损失、准确率与探针响应。<br>"
        f"• <b>在 {anchor_badge('[D. 拓扑图]', 'purple', target_id='region-d')} 观察</b>：探针微观激活状态。<br>"
        f"• <b>底图（梯度流分布）</b>：检查深层网络信号是否健康传递（防止梯度归零消失）。"
    ),
    experiments=[
        f"<b>第 1 步【一键破解 XOR 难题】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')}【经典实验预设】中选择 <code>XOR 历史困境与破解</code>，观察 2 层网络如何用弯曲的曲线完美切开对角分布的异或数据！",
        f"<b>第 2 步【体验探针听诊】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 拖动【探针坐标 x₁、x₂】滑动条，观察 {anchor_badge('[D. 拓扑图]', 'purple', target_id='region-d')} 中，哪些神经元被染成了深蓝色（被强力激活）！",
        f"<b>第 3 步【挑战双螺旋极限】</b>：选择预设 <code>双螺旋奇点挑战</code>，见证 3 层深度网络如何像拉花一样把极其复杂的双螺旋流形完全解开！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板 (含预设，由 knowledge 元数据驱动)
# ---------------------------------------------------------------------------
st.sidebar.markdown(f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>HYPERPARAMETERS // 控制台与探针</b></div>', unsafe_allow_html=True)
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
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">TELEMETRY BENCHMARK // 实时遥测指标看板</span>'
    f'</div>',
    unsafe_allow_html=True,
)
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
    st.markdown(
        f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
        f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">TOPOLOGY & PROBE // 激活探针与网络拓扑</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig_topo = plot_network_topology(
        layer_sizes=layer_sizes,
        weights=dense_weights,
        neuron_activations=probe_activations,
        title=f"TOPOLOGY & PROBE // 激活探针响应 (x₁={probe_x:.2f}, x₂={probe_y:.2f})",
    )
    st.plotly_chart(fig_topo, use_container_width=True)
    with st.expander("[HOW TO READ // 读图指南] 网络拓扑与激活点亮", expanded=False):
        st.markdown(
            """
            * **从左到右三层**：输入特征层 $(x_1, x_2) \\to$ 中间隐藏层 $\\to$ 输出层 $(\\hat{y})$。
            * **圆圈高亮**：当前探针点强力激活点亮了哪个神经元（深蓝代表高度兴奋）。
            * **连线粗细与颜色**：**蓝色**为正权重（兴奋），**红色**为负权重（抑制），**越粗**影响力越大。
            * **[OPTIMAL // 观察要点]**：切换不同探针位置，观察不同特征是如何通过不同神经元通路组合决策的。
            """
        )

with col_bound:
    st.markdown(
        f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
        f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">DECISION MANIFOLD // 空间决策流形与探针定位</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig_bound = plot_decision_boundary(
        model, X, y, probe_point=probe_pt, title="DECISION MANIFOLD // 空间决策流形与探针定位"
    )
    st.plotly_chart(fig_bound, use_container_width=True)
    with st.expander("[HOW TO READ // 读图指南] 空间非线性弯曲决策分界线", expanded=False):
        st.markdown(
            """
            * **横轴 $X_1$ 与纵轴 $X_2$**：样本二维特征坐标。
            * **黑色加粗实线 `[DECISION LINE]`**：多层网络折叠拼出的**弯曲分界线**（不再是单层死板的直线）。
            * **黄色十字交叉点 (PROBE)**：你在左侧拖动的探针定位点，右图显示它所在的分类概率区域。
            * **[OPTIMAL // 最优形态]**：弯曲的黑线完美环绕或切分非线性数据集（如月牙/双圈）。
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
        with st.expander("[HOW TO READ // 读图指南] 逐层神经元激活热力图", expanded=False):
            st.markdown(
                """
                * **横轴**：该隐藏层各神经元编号；**纵轴**：前 30 个输入样本。
                * **颜色深浅**：颜色越亮代表神经元激活输出值越大。
                * **[WARNING // Dead ReLU 诊断]**：若某整列全黑全灰（值为 0），说明该神经元已死，不再传递信息。可切换为 `LeakyReLU` / `GELU`。
                """
            )

with col_grad:
    if dense_grads:
        fig_grad = plot_gradient_histograms(dense_grads, layer_names, title="GRADIENT FLOW // 反向传播梯度流分布")
        st.plotly_chart(fig_grad, use_container_width=True)
        with st.expander("[HOW TO READ // 读图指南] 反向传播梯度流分布直方图", expanded=False):
            st.markdown(
                """
                * **横轴**：梯度数值范围；**纵轴**：落入该区间的权重参数数量。
                * **[OPTIMAL // 健康形态]**：各层梯度幅度相近（如 $10^{-3} \\sim 1.0$），呈钟形平滑分布。
                * **[WARNING // 梯度消失]**：浅层梯度柱子全部死死贴在 0 附近（$< 10^{-6}$），说明深层误差无法回传给前层。
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

