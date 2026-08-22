# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 4: 全参数微观实验室 (Hyperparameter & Micro-State Lab) - 零基础入门保姆级教学平台

工业级微观监控中枢：四宫格全景图、单步调试 (Step-by-Step)、快照对比与超参数动态热调整。
"""

import json
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.components.charts import (
    plot_decision_boundary,
    plot_gradient_histograms,
    plot_loss_curve,
    plot_weight_histograms,
)
from dashboard.components.param_panel import (
    render_dataset_selector,
    render_deep_dive_card,
    render_network_params,
    render_training_params,
)
from dashboard.constants.knowledge import (
    ACTIVATIONS,
    INITIALIZERS,
    OPTIMIZERS,
    REGULARIZERS,
)
from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from dashboard.utils.state import (
    get_dataset,
    resolve_activation,
    resolve_initializer,
    resolve_optimizer,
    resolve_regularizer,
)
from nn_core.layers import Dense
from nn_core.losses import BinaryCrossEntropy
from nn_core.model import Sequential

st.set_page_config(
    page_title="Micro Parameter Lab · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M04")

render_hero_header(
    title="全参数微观实验室",
    subtitle="工业级四宫格微观遥测监控台 · 支持逐步微调训练 (Step-by-Step) · 参数快照热回滚与 A/B 对比",
    badge_text="MILESTONE 04 // MICRO LAB & TELEMETRY",
    badge_type="emerald",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="全参数微观实验室与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "控制台面板",
            "desc": "在左侧侧边栏调节网络结构、正则化项与优化器参数",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解单步微调、过拟合与 L1/L2 正则化紧箍咒",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "步进训练与遥测",
            "desc": "点击单步走或练50轮，实时观测损失与权重梯度均值",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "四宫格微观全景",
            "desc": "流形面、收敛线、权重谱与梯度流四重视窗同步透视",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "快照 A/B 对比",
            "desc": "保存当前网络快照并与历史权重进行无缝对比",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>这里是神经网络的「显微镜调试台」！</b><br>"
        f"在工业界，我们不仅要看最终准不准，还要微观检查每一个神经元的健康状态：<br>"
        f"• <b>单步微调 (STEP)</b>：在 {anchor_badge('[C. 步进控制台]', 'emerald', target_id='region-c')} 像按电影逐帧播放键一样，每按一次，模型只向前走 1 步，让你看清权重和梯度如何微小蠕动！<br>"
        f"• <b>过拟合与正则化 (Regularization)</b>：数据噪声大时，模型容易「死记硬背每个噪点」导致分界线千疮百孔（过拟合）。"
        f"在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 开启 <code>L2 权重衰减</code> 可以给模型套上紧箍咒，让 {anchor_badge('[D. 四宫格流形]', 'purple', target_id='region-d')} 重新变得光滑自然！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>正则化类型 (L1/L2)</b>：惩罚过大的权重，防止过拟合。<br>"
        f"• <b>惩罚系数 (λ)</b>：紧箍咒的威力大小。<br>"
        f"• <b>在 {anchor_badge('[C. 控制台]', 'emerald', target_id='region-c')} 操作</b>：<code>TRAIN (练50轮)</code>、<code>STEP (单步走)</code>、<code>RESET (重置)</code>。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 四宫格全景图]', 'purple', target_id='region-d')} 实时监控</b>：<br>"
        f"  1. 空间决策面（分界线长啥样）<br>"
        f"  2. 损失收敛曲线（成绩提升过程）<br>"
        f"  3. 逐层权重直方图（参数胖瘦分布）<br>"
        f"  4. 反向传播梯度直方图（学习推动力大小）。"
    ),
    experiments=[
        f"<b>第 1 步【单步微观调试】</b>：在 {anchor_badge('[C. 控制区]', 'emerald', target_id='region-c')} 点击 <code>STEP // 单步微调</code> 按钮 2~3 次，仔细观察上方训练轮数 +1，以及四宫格中直方图的细微移动！",
        f"<b>第 2 步【制造过拟合】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把【噪声比 Noise】拉大到 <code>0.3</code>，正则化选 <code>None</code>，点击 <code>TRAIN // 训练 50 轮</code>，看看分界线是不是变得坑坑洼洼？",
        f"<b>第 3 步【正则化救场】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把【正则化类型】切换为 <code>L2 (Weight Decay)</code>，再次点击训练，见证分界线如何瞬间恢复平滑优雅！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>HYPERPARAMETERS // 控制台配置</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板 (中英双语标签，由元数据驱动)
# ---------------------------------------------------------------------------
dataset_name, n_samples, noise, random_state = render_dataset_selector(
    key_prefix="m4_", default_dataset="moons"
)

net_params = render_network_params(
    allow_multi_layer=True,
    key_prefix="m4_",
    default_layers=2,
    default_neurons=[12, 6],
    default_act="ReLU",
    default_init="he",
)
n_layers = net_params["n_layers"]
neurons_per_layer = net_params["neurons_per_layer"]
activation_name = net_params["activation"]
initializer = net_params["initializer"]

train_params = render_training_params(
    key_prefix="m4_", default_opt="Adam", default_lr=0.03, default_epochs=100
)
optimizer_name = train_params["optimizer"]
lr = train_params["learning_rate"]
batch_size = train_params["batch_size"]

st.sidebar.markdown("#### REGULARIZATION // 正则化策略")
reg_list = list(REGULARIZERS.values())
reg_labels = [m.label for m in reg_list]
reg_hints = {
    "None (无正则)": "纯经验风险最小化 · 自由拟合无约束",
    "L1 正则 (Lasso)": "曼哈顿范数约束 · 驱动权重绝对稀疏化 (产生 0 权重)",
    "L2 正则 (Ridge)": "欧氏范数约束 · 惩罚极端大权重 · 提升泛化",
}

selected_reg_label = st.sidebar.radio(
    "正则化类型",
    options=reg_labels,
    format_func=lambda o: f"**{o}**\n\n↳ *{reg_hints.get(o, '泛化约束')}*",
    help="权重约束项。通过惩罚过大的权重值，防止模型强行记忆样本噪声，有效提升泛化能力。",
    key="m4_reg",
)
reg_meta = next(m for m in reg_list if m.label == selected_reg_label)

reg_lambda = 0.0
if reg_meta.id != "None":
    reg_lambda = st.sidebar.slider(
        "惩罚系数 (λ)",
        0.0001,
        0.1,
        0.01,
        step=0.005,
        format="%.4f",
        help="正则化惩罚强度。越大对权重的压制越强，决策面越平滑；过大会导致欠拟合。",
        key="m4_lambda",
    )

# ---------------------------------------------------------------------------
# Session State 模型管理 (支持逐步训练)
# ---------------------------------------------------------------------------
act_cls = resolve_activation(activation_name)
init_name = resolve_initializer(initializer)
opt_cls = resolve_optimizer(optimizer_name)

if "m4_model" not in st.session_state or st.sidebar.button("RESET // 重置模型", key="m4_reset"):
    m = Sequential()
    current_dim = 2
    for i in range(n_layers):
        out_dim = neurons_per_layer[i]
        reg = resolve_regularizer(reg_meta.id, float(reg_lambda))
        m.add(Dense(current_dim, out_dim, initializer=init_name, regularizer=reg))
        m.add(act_cls())
        current_dim = out_dim

    m.add(Dense(current_dim, 1, initializer=init_name))
    m.add(resolve_activation("Sigmoid")())

    st.session_state["m4_model"] = m
    st.session_state["m4_history"] = {"loss": [], "accuracy": []}
    st.session_state["m4_epoch_count"] = 0
    st.session_state["m4_opt_instance"] = opt_cls(learning_rate=float(lr))

model: Sequential = st.session_state["m4_model"]
history: dict[str, list[float]] = st.session_state["m4_history"]
opt_instance = st.session_state.get("m4_opt_instance", opt_cls(learning_rate=float(lr)))

# ---------------------------------------------------------------------------
# 训练控制中枢 (Train / Step)
# ---------------------------------------------------------------------------
X, y = get_dataset(dataset_name, n_samples, noise, random_state)
loss_fn = BinaryCrossEntropy()

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("TRAIN // 训练 50 轮", key="m4_train_50"):
        hist = model.train(
            X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=50, batch_size=batch_size
        )
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 50

with col_btn2:
    if st.button("STEP // 单步微调", key="m4_step_1"):
        hist = model.train(
            X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=1, batch_size=batch_size
        )
        history["loss"].extend(hist["loss"])
        history["accuracy"].extend(hist["accuracy"])
        st.session_state["m4_epoch_count"] += 1

with col_btn3:
    current_epochs = st.session_state["m4_epoch_count"]
    clean_opt_name = optimizer_name.split(" ")[0]
    badge_status = (
        '<div style="display:flex;align-items:center;gap:0.8rem;height:100%;padding-top:0.3rem;">'
        f'<span class="pill-badge pill-emerald"><span class="status-dot"></span> EPOCHS 训练轮数: {current_epochs}</span>'
        f'<span class="pill-badge pill-blue">OPT 优化器: {clean_opt_name} (lr={lr})</span>'
        "</div>"
    )
    st.markdown(badge_status, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 遥测指标计算与展示 (中英双语标签)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">STEP CONTROLLER & TELEMETRY // 步进训练中枢与遥测指标</span>'
    f"</div>",
    unsafe_allow_html=True,
)
current_loss = history["loss"][-1] if history["loss"] else 1.0
current_acc = history["accuracy"][-1] if history["accuracy"] else 0.0

dense_layers = [l for l in model.layers if isinstance(l, Dense)]
weights_list = [l.weights for l in dense_layers]
grads_list = [l.grad_weights for l in dense_layers if l.grad_weights is not None]
layer_names = [f"Dense #{i + 1}" for i in range(len(dense_layers))]

grad_norm = float(np.mean([np.mean(np.abs(g)) for g in grads_list])) if grads_list else 0.0
weight_norm = float(np.mean([np.mean(np.abs(w)) for w in weights_list])) if weights_list else 0.0

clean_reg_name = reg_meta.id

grid_html = (
    '<div class="metric-grid" style="margin-top:0.4rem;">'
    + render_metric_card(
        "CURRENT LOSS // 当前损失",
        f"{current_loss:.4f}",
        delta=f"EPOCH 轮次 #{st.session_state['m4_epoch_count']}",
        delta_type="positive" if current_loss < 0.2 else "neutral",
        icon_name="trending-down",
    )
    + render_metric_card(
        "ACCURACY // 当前准确率",
        f"{current_acc:.1%}",
        delta="LIVE STATE",
        delta_type="positive" if current_acc >= 0.9 else "neutral",
        icon_name="target",
    )
    + render_metric_card(
        "GRADIENT NORM // 梯度范数",
        f"{grad_norm:.2e}",
        delta="BACKPROP ACTIVITY",
        delta_type="neutral",
        icon_name="activity",
    )
    + render_metric_card(
        "WEIGHT MAGNITUDE // 权重均值",
        f"{weight_norm:.4f}",
        delta=f"REG 正则: {clean_reg_name}",
        delta_type="neutral",
        icon_name="shield",
    )
    + "</div>"
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 四宫格全景监控 (Four-Grid Dashboard)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">QUAD TELEMETRY CONSOLE // 四宫格微观全景遥测监控台</span>'
    f"</div>",
    unsafe_allow_html=True,
)

first_layer_w = weights_list[0] if weights_list else None
w11_str = f"{first_layer_w[0, 0]:.3f}" if first_layer_w is not None and first_layer_w.size > 0 else "0.000"
w12_str = f"{first_layer_w[1, 0]:.3f}" if first_layer_w is not None and first_layer_w.shape[0] > 1 else "0.000"
total_params_count = sum(w.size for w in weights_list)

render_live_param_status_bar(
    title="LIVE WEIGHT PARAMETERS // 实时微观参数与网络状态",
    badges=[
        {"label": "w₁₁", "value": w11_str, "color": "blue"},
        {"label": "w₁₂", "value": w12_str, "color": "amber"},
    ],
    metrics=[
        ("权重均值 ‖W‖", f"{weight_norm:.4f}"),
        ("梯度均值 ‖∇W‖", f"{grad_norm:.2e}"),
        ("正则惩罚 λ", f"{reg_lambda:.4f}"),
    ],
    tag=f"TOTAL PARAMS: {total_params_count} 个权重参数",
    tag_color="emerald",
)

grid_c1, grid_c2 = st.columns(2)

with grid_c1:
    fig_bound = plot_decision_boundary(
        model, X, y, title=f"DECISION MANIFOLD // 空间决策流形 (Acc: {current_acc:.1%})"
    )
    st.plotly_chart(fig_bound, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 空间决策流形与分界线", expanded=False):
        st.markdown(
            """
            * **横轴 $X_1$ 与纵轴 $X_2$**：样本的两个特征坐标。
            * **黑色加粗实线 `[DECISION LINE]`**：当前模型画出的二分类决策分水岭 ($P=0.5$)。
            * **[OPTIMAL // 最优形态]**：黑色实线随着训练轮数增加，优雅包裹月牙或环形数据，准确率达 95%+。
            """
        )

    fig_w_hist = plot_weight_histograms(
        weights_list, layer_names, title="WEIGHT SPECTRUM // 逐层权重参数分布"
    )
    st.plotly_chart(fig_w_hist, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 逐层权重参数直方图", expanded=False):
        st.markdown(
            """
            * **横轴**：权重参数的具体数值范围（如 $-1.0 \\sim +1.0$）；**纵轴**：落入该数值区间的参数个数。
            * **[OPTIMAL // 健康形态]**：以 0 为中心呈紧凑的**钟形高斯分布**。
            * **[WARNING // 过拟合风险]**：权重扩散到 $[-10, +10]$ 以上，说明权重膨胀过大，建议调大 L2 正则化系数 $\\lambda$。
            """
        )

with grid_c2:
    fig_loss = plot_loss_curve(history, title="TRAINING DYNAMICS // 损失与准确率收敛动态")
    st.plotly_chart(fig_loss, width="stretch")
    with st.expander(
        "[HOW TO READ // 读图指南] 损失收敛 (Loss) 与准确率 (Accuracy)", expanded=False
    ):
        st.markdown(
            """
            * **横轴**：训练轮数 (Epochs)；**左纵轴**：损失 (Loss，越低越好)；**右纵轴**：准确率 (Accuracy，越高越好)。
            * **[OPTIMAL // 最优形态]**：Loss 大滑梯式俯冲至 0，Accuracy 单调攀升至 100%。
            * **[WARNING // 震荡]**：若 Loss 剧烈锯齿波动，说明学习率 LR 过大，需调小。
            """
        )

    fig_g_hist = plot_gradient_histograms(
        grads_list if grads_list else [np.zeros((2, 2))],
        layer_names,
        title="GRADIENT FLOW // 反向传播梯度流分布",
    )
    st.plotly_chart(fig_g_hist, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 反向传播梯度流分布", expanded=False):
        st.markdown(
            """
            * **横轴**：梯度数值范围；**纵轴**：参数个数。
            * **[OPTIMAL // 健康形态]**：各层梯度稳定在 $10^{-3} \\sim 1.0$ 数量级，信号能顺畅直达浅层。
            * **[WARNING // 梯度消失]**：浅层梯度柱子全部贴在 0 附近，需更换激活函数 (ReLU/GELU) 或采用 He 初始化。
            """
        )

# ---------------------------------------------------------------------------
# 快照与实验状态导出 (Snapshot Export / Rollback)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">SNAPSHOT ARCHIVE // 实验状态快照管理与 JSON 导出</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_snap1, col_snap2 = st.columns(2)

with col_snap1:
    if st.button("SNAPSHOT // 保存当前实验快照", key="m4_snap_save"):
        snapshot = {
            "epoch": st.session_state["m4_epoch_count"],
            "loss": current_loss,
            "accuracy": current_acc,
            "layers": neurons_per_layer,
            "activation": activation_name,
            "optimizer": optimizer_name,
            "lr": lr,
            "weights": [w.tolist() for w in weights_list],
        }
        st.session_state["m4_saved_snapshot"] = snapshot
        st.success(
            f"已保存第 {st.session_state['m4_epoch_count']} 轮快照 (Loss: {current_loss:.4f})"
        )

with col_snap2:
    if "m4_saved_snapshot" in st.session_state:
        snap_json = json.dumps(st.session_state["m4_saved_snapshot"], indent=2)
        st.download_button(
            label="DOWNLOAD // 导出快照 JSON",
            data=snap_json,
            file_name=f"nn_snapshot_epoch_{st.session_state['m4_epoch_count']}.json",
            mime="application/json",
            key="m4_snap_download",
        )

# 深度知识学习指南 (折叠微观原理解析)
act_meta = ACTIVATIONS.get(
    activation_name, ACTIVATIONS.get(activation_name.split(" ")[0], ACTIVATIONS["ReLU"])
)
init_meta = INITIALIZERS.get(
    initializer, INITIALIZERS.get(initializer.split(" ")[0].lower(), INITIALIZERS["he"])
)
opt_meta = OPTIMIZERS.get(
    optimizer_name, OPTIMIZERS.get(optimizer_name.split(" ")[0], OPTIMIZERS["Adam"])
)

render_deep_dive_card(
    "全参数微观实验室正则化与超参数原理指南", [reg_meta, act_meta, init_meta, opt_meta]
)

# ---------------------------------------------------------------------------
# 零基础进阶：正则化与参数微观名词通俗速查 (含公式拆解)
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 正则化核心公式拆解与微观参数名词通俗全解", expanded=False
):
    st.markdown(
        r"""
        ### 0. 核心公式逐字拆解：L1 / L2 权重正则化
        $$L_{\\text{total}} = L_{\\text{task}} + \\lambda \\cdot \\Omega(W)$$
        
        | 符号 | 中文名称 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---|
        | **$L_{\\text{total}}$** | **总优化目标损失** | 最终要最小化的总罚分（任务预测误差 + 权重过大罚分）。 |
        | **$L_{\\text{task}}$** | **原始任务损失** | 模型在分类任务上做错题的交叉熵扣分。 |
        | **$\\lambda$** | **正则化惩罚系数 (Lambda)** | **戒尺的严厉程度**。$\\lambda=0$ 不管不顾；$\\lambda=0.01$ 适度约束；$\\lambda$ 过大（如 1.0）会把所有权重打成 0 导致欠拟合。 |
        | **$\\Omega(W)$** | **权重复杂度惩罚项** | • **L1 正则 (Lasso, $\\sum \|w\|$ )**：倾向于把不重要的权重**直接削成纯零 (稀疏特征选择)**；<br>• **L2 正则 (Ridge / Weight Decay, $\\frac{1}{2}\\sum w^2$ )**：倾向于让所有权重都**变得很小且均匀分布**，防止单个权重独大。 |
        
        ---
        
        ### 1. 什么是【过拟合 (Overfitting)】？—— “死记硬背不通变”
        * **生活比喻**：学生刷题时把题号和选项顺序全背下来了，模拟考打 100 分；但高考时换了一道数字不同的新题，当场交白卷。
        * **在图上的表现**：决策分界线变得极度扭曲崎岖，死扣每一个孤立的噪点。
        
        ---
        
        ### 2. 什么是【Update-to-Weight 步长比】？—— “每步微调的相对幅度”
        * **黄金准则**：每一步更新的量 $\\|\\Delta W\\|$ 与当前权重大小 $\\|W\\|$ 的比值在 **$10^{-3} \\approx 0.001$** 附近为最健康的训练状态。
        """
    )
