# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 3: 优化器多轨竞速对比 (Optimizer Arena) - 零基础入门保姆级教学平台

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
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.constants.knowledge import OPTIMIZERS
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
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
render_lesson_evidence("M03", show_contract=True)
render_core_result_evidence("M03")

render_hero_header(
    title="优化器多轨竞速对比",
    subtitle="同一网络拓扑与初始权重起点，全方位对比 SGD / Momentum / RMSProp / Adam 的收敛动力学",
    badge_text="MILESTONE 03 // CONVERGENCE DYNAMICS",
    badge_type="amber",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="优化器多轨竞速对比与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "控制台面板",
            "desc": "在左侧侧边栏调节基准学习率、网络结构与训练轮数",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解 SGD/Momentum/RMSProp/Adam 算法本质",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时竞速遥测",
            "desc": "查看四大优化器最终的做题扣分与登顶冠军",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "多轨收敛对比图",
            "desc": "同一起跑线 4 色多轨竞速曲线，看谁最先贴底收敛",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "排行榜与四分屏",
            "desc": "并排对比四大算法最终画出的空间决策分界质量",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        "<b>优化器（Optimizer）就是模型下山寻宝的「导航算法」！</b><br>"
        "模型的目标是从高山（Loss 损失很大）一路滑到谷底（Loss 为 0）。<br>"
        "• <b>SGD（普通梯度下降）</b>：像近视眼，只看脚下哪边陡就走哪边，容易在峡谷里来回撞墙。<br>"
        "• <b>Momentum（动量加速）</b>：给机器人加了滑雪惯性，能在平缓处加速冲刺，冲过小土坡。<br>"
        "• <b>RMSProp（自适应步长）</b>：经常大跳的参数自动减速，平缓的参数自动加速。<br>"
        "• <b>Adam</b>：结合一阶动量与二阶矩缩放，是常用起点；实际训练也常用 AdamW、SGD 等，结果依赖任务与超参数。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>基准学习率 (LR)</b>：4 种优化器共同遵守的初始油门大小。<br>"
        f"• <b>分布类型</b>：选择不同的地形复杂度（如 Moons 复杂曲面，Spiral 极度扭曲）。<br>"
        f"• <b>竞速轮数 (Epochs)</b>：让 4 辆赛车在同一起跑线上跑多少圈。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 多轨收敛图]', 'purple', target_id='region-d')} 观测</b>：4 根不同颜色的收敛曲线，对比各优化器在当前超参数下的下降速率与平稳度。<br>"
        f"• <b>在 {anchor_badge('[C. 竞速遥测指标]', 'emerald', target_id='region-c')} 统计</b>：量化比较最终损失值、收敛耗时与准确率。<br>"
        f"• <b>在 {anchor_badge('[E. 四分屏分界]', 'blue', target_id='region-e')} 验收</b>：并排展示 4 种算法各自最终画出的分界面质量。"
    ),
    experiments=[
        f"<b>第 1 步【对比多轨收敛】</b>：观察 {anchor_badge('[D. 多轨对比图]', 'purple', target_id='region-d')}，对比 SGD、Momentum、RMSProp 与 Adam 在当前地形下的收敛动态与震荡特性。",
        f"<b>第 2 步【学习率敏感度探究】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节【基准学习率】（如 0.01 vs 0.1）。观察在不同超参数配置下各优化器的收敛稳定性与边界行为（自适应优化器通常适应面更广，但在极端学习率下同样可能发生震荡或发散）。",
        f"<b>第 3 步【四分屏形态对比】</b>：在 {anchor_badge('[E. 四分屏分界]', 'blue', target_id='region-e')} 对比 4 种优化器最终拟合的决策流形平滑度与抗噪能力。",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>HYPERPARAMETERS // 控制台配置</b></div>',
    unsafe_allow_html=True,
)
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
        0.001,
        1.0,
        0.03,
        step=0.01,
        format="%.3f",
        help="所有优化器共享的基准更新步长 $\\eta$。可观察不同优化器在相同学习率下的抗震荡与自适应特性。",
        key="m3_lr",
    )
with col2:
    epochs = st.slider(
        "竞速轮数 (Epochs)",
        20,
        500,
        150,
        step=10,
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
        X, y, loss_fn=loss_fn, optimizer=opt_instance, epochs=int(epochs), batch_size=32
    )

    histories[opt_meta.id] = hist
    trained_models[opt_meta.id] = model_clone

    final_loss = hist["loss"][-1] if hist["loss"] else 1.0
    final_acc = hist["accuracy"][-1] if hist["accuracy"] else 0.0

    converged_epoch = next((i + 1 for i, l in enumerate(hist["loss"]) if l < 0.25), "未收敛")

    leaderboard_rows.append(
        {
            "优化器算法": opt_meta.label,
            "最终 Loss": f"{final_loss:.4f}",
            "最终准确率": f"{final_acc:.1%}",
            "收敛步数 (Loss < 0.25)": str(converged_epoch),
            "_raw_loss": final_loss,
        }
    )

leaderboard_rows.sort(key=lambda x: x["_raw_loss"])
champion = leaderboard_rows[0]["优化器算法"]
champion_loss = leaderboard_rows[0]["最终 Loss"]
champion_acc = leaderboard_rows[0]["最终准确率"]

# ---------------------------------------------------------------------------
# 遥测指标卡 (中英双语标签)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">TELEMETRY BENCHMARK // 实时竞速遥测看板</span>'
    f"</div>",
    unsafe_allow_html=True,
)
grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "BENCHMARK WINNER // 竞速冠军",
        champion,
        delta=f"Loss: {champion_loss} | Acc: {champion_acc}",
        delta_type="positive",
        icon_name="award",
    )
    + render_metric_card(
        "TOPOLOGY // 数据集拓扑",
        dataset_name.upper(),
        delta=f"N={n_samples} | NOISE={noise}",
        delta_type="neutral",
        icon_name="database",
    )
    + render_metric_card(
        "INITIALIZER // 初始化器",
        init_name.upper(),
        delta=f"{n_layers} HIDDEN LAYERS",
        delta_type="neutral",
        icon_name="layers",
    )
    + render_metric_card(
        "TOTAL EPOCHS // 总轮数",
        f"{epochs} EPOCHS",
        delta=f"LR = {lr}",
        delta_type="neutral",
        icon_name="activity",
    )
    + "</div>"
)
st.markdown(grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 核心可视化：多轨 Loss 对比图
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">MULTI-LOSS BENCHMARK // 多轨收敛竞速对比图</span>'
    f"</div>",
    unsafe_allow_html=True,
)

best_opt = min(histories.keys(), key=lambda k: histories[k]["loss"][-1])
best_loss = histories[best_opt]["loss"][-1]
best_acc = histories[best_opt]["accuracy"][-1]

render_live_param_status_bar(
    title="OPTIMIZER ARENA DYNAMICS // 优化器多轨竞速微观参数",
    badges=[
        {"label": "TOP OPT", "value": f"{best_opt}", "color": "emerald"},
        {"label": "MIN LOSS", "value": f"{best_loss:.4f}", "color": "blue"},
        {"label": "PEAK ACC", "value": f"{best_acc:.1%}", "color": "purple"},
    ],
    metrics=[
        ("学习率 η", f"{lr}"),
        ("对比算法数", f"{len(histories)}"),
        ("总训练步数", f"{epochs} Epochs"),
    ],
    tag=f"WINNER: {best_opt.upper()}",
    tag_color="emerald",
)

fig_multi_loss = plot_multi_loss_curves(
    histories, title="OPTIMIZER BENCHMARK // 优化器多轨收敛对比"
)
st.plotly_chart(fig_multi_loss, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 判断优化器速度与稳定性", expanded=False):
    st.markdown(
        """
        * **横轴**：训练轮数 (Epochs)；**纵轴**：损失 (Loss)。
        * **【如何比较】**：在相同初始化、数据顺序、计算预算与调参预算下，比较验证损失、方差和达到目标所需步数；训练损失下降最快不等于泛化最好。
          * 曲线抖动可能来自随机小批量，不自动表示优化失败；平滑也不证明每一步方向都正确。
        * [WARNING] **【典型现象对比】**：
          * **纯 SGD**：下降像爬行，常常停在半山腰或鞍点平原迟迟不动；
          * **Momentum (冲量)**：初期冲得极快，但在急转弯山谷容易“惯性冲过头”出现局部上扬，随后迅速回调；
          * **RMSProp / Adam**：具有自适应步长，在平缓地带大步流星，在陡峭峡谷自动减速防翻车。
        """
    )

# ---------------------------------------------------------------------------
# 排行榜与四分屏决策边界对比
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">LEADERBOARD & 4-QUADRANT VIEW // 效能排行榜与四分屏决策边界并排透视</span>'
    f"</div>",
    unsafe_allow_html=True,
)
df_board = pd.DataFrame(leaderboard_rows).drop(columns=["_raw_loss"])
st.dataframe(df_board, width="stretch", hide_index=True)

cols = st.columns(4)

for idx, opt_meta in enumerate(optimizer_items):
    with cols[idx]:
        m = trained_models[opt_meta.id]
        acc_text = next(
            r["最终准确率"] for r in leaderboard_rows if r["优化器算法"] == opt_meta.label
        )
        fig_b = plot_decision_boundary(
            m, X, y, resolution=70, title=f"{opt_meta.id} (Acc: {acc_text})"
        )
        st.plotly_chart(fig_b, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 直观验证最终划分能力", expanded=False):
    st.markdown(
        """
        * **对比重点**：在相同轮数下，哪一个优化器画出的黑色决策线最平滑、最准确地把两类点包裹切分。
        * [CONVERGED] **【当前样本上的收敛形态】**：黑色边界在这次合成训练集上分开多数样本；不能据此证明测试集最优；
        * [WARNING] **【欠训练形态】**：如果某个优化器的图里黑色分界线依然是一根僵硬的斜线甚至切错了大部分点，说明它在有限轮数内没能逃出平原（如学习率未调准的纯 SGD）。
        """
    )

# 深度知识学习指南 (折叠微观原理解析)
render_deep_dive_card("四大优化算法动力学原理与更新公式深度对比", optimizer_items)

# ---------------------------------------------------------------------------
# 零基础进阶：优化器核心名词通俗速查
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 优化器核心机制通俗全解", expanded=False):
    st.markdown(
        r"""
        ### 1. 什么是【动量 (Momentum)】？—— “下山推滚铁球”
        * **生活比喻**：普通 SGD 像一只没有记忆的青蛙，每跳一步都要重新看一眼脚下；而 Momentum 像一个顺着山坡滚下的大铁球，**积攒了前面的惯性速度**。遇到平地它能靠惯性冲过去，遇到小坑小洼不会被卡住！
        * **本质机理**：标准工程实现采用物理动量递推 $v_t = \beta v_{t-1} + \eta g_t, W \leftarrow W - v_t$（等效稳态步长为 $\frac{\eta}{1-\beta}$）；统计学中亦常表述为梯度指数移动平均 (EMA) $v_t = \beta v_{t-1} + (1-\beta) g_t, W \leftarrow W - \eta v_t$。两者经步长重缩放后完全等价，核心均在于累积历史一阶矩以抑制正交高频震荡。

        ---

        ### 2. 什么是【自适应学习率 (Adaptive LR / RMSProp)】？—— “自适应油门”
        * **生活比喻**：汽车开在平坦高速公路上就猛踩油门加速，开在九曲十八弯的盘山险道就轻踩刹车减速。
        * **本质机理**：记录历史梯度的未中心化平方和（二阶矩 $s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2$）。频繁剧烈振荡的维度因二阶矩累积而自动减小有效步长，稀疏平缓的维度自动放大步长。

        ---

        ### 3. 为什么【Adam】是深度学习广泛采用的基准优化器？
        * **机制组合**：**Adam = 一阶动量惯性 (Momentum) + 二阶自适应步长 (RMSProp) + 初始冷启动零偏校正**。
        * **收敛边界**：在很多非凸复杂地形中提供了良好的开箱即用性，但并不能消除超参数敏感性；在重尾噪声或对抗训练中仍可能遭遇振荡，实际训练超大模型通常需严格配合学习率预热（Warmup）、余弦退火与权重衰减解耦 (AdamW)。
        """
    )
