# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
NN Playground - M16: 强化学习与自主智能体实验室 (Reinforcement Learning & Reasoning Agents)

核心教学与实验目标：
1. 马尔可夫决策过程 (MDP) 与贝尔曼最优方程 (Bellman Equation) 解析求解
2. 时序差分 Q-Learning 试错演进、ε-greedy 探索与悬崖避障
3. 状态价值热力面 V(s) 逆向反推与策略箭头场收敛
4. 2026 前沿：DeepSeek-R1 式 GRPO 纯强化学习推理涌现与“顿悟 (Aha Moment)”长思维链进化
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.icons import svg_icon
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_floating_hud_navigator,
    render_hero_header,
    render_interactive_region_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.reinforcement import (
    BellmanSolver,
    GridWorldEnv,
    GRPORunner,
    QLearningAgent,
)

st.set_page_config(
    page_title="M16: 强化学习与自主智能体 · NN Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入全局高对比亮色主题
apply_custom_theme()

# 页面空间 HUD
render_floating_hud_navigator(
    [
        {"id": "A", "name": "参数控制台", "desc": "在左侧侧边栏调节学习率、折扣因子与探索率"},
        {"id": "B", "name": "教学指引与蓝图", "desc": "马尔可夫决策过程与贝尔曼方程原理解析"},
        {"id": "C", "name": "强化学习遥测卡", "desc": "实时监测累积回报、TD 误差与收敛指标"},
        {"id": "D", "name": "网格世界寻路演播", "desc": "智能体避障寻路并抵达宝藏终点"},
        {"id": "E", "name": "贝尔曼价值曲面", "desc": "2D/3D 状态价值流形热力分布"},
        {"id": "F", "name": "策略决策箭头场", "desc": "各网格最优动作流向矢量图"},
        {"id": "G", "name": "R1 推理涌现演化", "desc": "DeepSeek-R1 GRPO 组相对优化与长思维链顿悟"},
    ]
)

# Hero 标题
render_hero_header(
    title="M16: 强化学习与自主智能体",
    subtitle="从 MDP 贝尔曼方程、Q-Learning 试错寻路，到 2026 前沿 DeepSeek-R1 GRPO 纯强化学习推理涌现",
    badge_text="PURE NUMPY CORE · REINFORCEMENT LEARNING · BELLMAN & GRPO",
    badge_type="rose",
)

# 核心教学论据
render_lesson_evidence("M16")

# ---------------------------------------------------------------------------
# [A] 侧边栏参数控制台
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">'
    f'{anchor_badge("A", "amber")} <b>RL CONTROLS // 强化学习控制台</b>'
    f"</div>",
    unsafe_allow_html=True,
)

grid_card_options = [
    ("4x6 悬崖漫步 (Cliff Walking)", "高风险地形 · 负向惩罚试错 · 逼近悬崖最优解", "cliff"),
    ("5x5 迷宫障碍 (Maze)", "实体墙壁阻隔 · 复杂拓扑 · 自动绕障寻路", "maze"),
    ("4x4 经典网格 (Simple 4x4)", "基准几何拓扑 · 无障碍 · 贝尔曼最优求解", "simple"),
]

selected_grid_card = st.sidebar.radio(
    "环境类型 (Grid Environment)",
    options=grid_card_options,
    format_func=lambda o: f"**{o[0]}**\n\n↳ *{o[1]}*",
    index=0,
)
grid_choice = selected_grid_card[0]
env_key = selected_grid_card[2]
# env_key defined above

lr_val = st.sidebar.slider(
    "学习率 (Learning Rate α)",
    min_value=0.01,
    max_value=0.50,
    value=0.15,
    step=0.01,
)

gamma_val = st.sidebar.slider(
    "折扣因子 (Discount Factor γ)",
    min_value=0.50,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="越接近 1，智能体越看重长期未来回报；越小则越目光短浅",
)

epsilon_val = st.sidebar.slider(
    "探索率 (Exploration Rate ε)",
    min_value=0.05,
    max_value=1.00,
    value=0.30,
    step=0.05,
    help="ε-greedy 探索概率：以 ε 概率随机探索新动作，以 1-ε 概率利用当前最优 Q 值",
)

episodes_val = st.sidebar.select_slider(
    "训练轮数 (Training Episodes)",
    options=[20, 50, 80, 100, 150, 200],
    value=80,
)

# ---------------------------------------------------------------------------
# 环境初始化与贝尔曼最优求解
# ---------------------------------------------------------------------------
env = GridWorldEnv(grid_type=env_key)
solver = BellmanSolver(env=env, gamma=gamma_val, theta=1e-4)
opt_V, opt_policy, bellman_iters = solver.solve()

# Q-Learning 训练仿真
np.random.seed(42)
agent = QLearningAgent(
    height=env.height,
    width=env.width,
    lr=lr_val,
    gamma=gamma_val,
    epsilon=epsilon_val,
)
train_history = agent.train_episodes(env, n_episodes=episodes_val)

# 提取最终贪心路径
agent_path = []
curr_s = env.reset()
agent_path.append(curr_s)
for _ in range(30):
    best_a = int(np.argmax(agent.q_table[curr_s[0], curr_s[1]]))
    next_s, _, done, _ = env.step(best_a)
    agent_path.append(next_s)
    curr_s = next_s
    if done:
        break

# ---------------------------------------------------------------------------
# [B] 教学指引与蓝图导航
# ---------------------------------------------------------------------------
blueprint_sections = [
    {
        "id": "A",
        "name": "参数控制台",
        "desc": "调节学习率 α、折扣因子 γ、探索率 ε 与环境类型",
        "color": "amber",
        "target_id": "region-a",
    },
    {
        "id": "B",
        "name": "教学指引",
        "desc": "马尔可夫决策过程、贝尔曼方程与 2026 前沿推理 RL 理论拆解",
        "color": "blue",
        "target_id": "region-b",
    },
    {
        "id": "C",
        "name": "强化遥测",
        "desc": "实时监测累积回报、TD 误差与探索衰减率",
        "color": "emerald",
        "target_id": "region-c",
    },
    {
        "id": "D",
        "name": "网格世界寻路",
        "desc": "二维网格交互演播，智能体避开陷阱直达宝藏终点",
        "color": "blue",
        "target_id": "region-d",
    },
    {
        "id": "E",
        "name": "贝尔曼价值曲面",
        "desc": "展示最优状态价值 V*(s) 如何像水波一样从终点反向浸润",
        "color": "purple",
        "target_id": "region-e",
    },
    {
        "id": "F",
        "name": "策略箭头场",
        "desc": "可视化每个格子的最优动作流向分布",
        "color": "blue",
        "target_id": "region-f",
    },
    {
        "id": "G",
        "name": "R1 推理涌现演化",
        "desc": "2026 DeepSeek-R1 GRPO 组相对优化与长思考链顿悟模拟",
        "color": "rose",
        "target_id": "region-g",
    },
]

render_page_guide(
    title="强化学习 (RL) 与智能体演进全景指南",
    plain_intro="强化学习是 AI 实现自主决策与推理的终极武器。智能体在一个未知环境中，通过不断尝试动作、观察状态转移并接受环境给出的即时奖惩（Reward），自主探索并学会最大化长期累积回报的最优策略。",
    hyperparams_desc="• 学习率 α：时序差分单步更新权重；\n• 折扣因子 γ：未来奖励的时间折现比率；\n• 探索率 ε：探索未知与利用已知经验的权衡；\n• 训练轮数：智能体在环境中的试错总 Episode 数。",
    telemetry_desc="• 累积回报 (Return)：单轮获得的所有即时奖励总和；\n• TD-Error：贝尔曼目标与当前 Q 值的预测误差；\n• 贝尔曼收敛轮数：动态规划值迭代达到自洽的步数。",
    experiments=[
        "尝试调大探索率 ε 到 0.8，观察智能体早期频繁跌落悬崖，但最终能否探索出更稳妥的路径？",
        "观察贝尔曼价值曲面：为什么离终点越近的格子颜色越亮、数值越高？",
        "切换到【5x5 迷宫障碍】，观察策略箭头场如何自动绕过实体墙壁直达终点！",
        "在 Section G 拖动【GRPO 训练轮数】，见证思考链 Token 长度如何从几十暴涨到上千，并自发产生 Aha Moment！",
    ],
    blueprint_sections=blueprint_sections,
)

# ---------------------------------------------------------------------------
# [C] 强化学习微观遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">REINFORCEMENT TELEMETRY // 强化学习收敛遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)

final_return = train_history["returns"][-1]
avg_td = float(np.mean(train_history["td_errors"][-10:]))
path_len = len(agent_path) - 1

metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "EPISODE RETURN // 最终轮回报",
        f"{final_return:+.2f}",
        delta="已收敛达标" if final_return > 0 else "探索中",
        delta_type="positive" if final_return > 0 else "negative",
        icon_name="target",
    )
    + render_metric_card(
        "BELLMAN ITERS // 贝尔曼收敛轮数",
        f"{bellman_iters} ITERS",
        delta="动态规划已自洽",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TD-ERROR // 平均时序差分误差",
        f"{avg_td:.4f}",
        delta=f"学习率 α={lr_val}",
        delta_type="neutral",
        icon_name="activity",
    )
    + render_metric_card(
        "OPTIMAL PATH // 最优寻路步数",
        f"{path_len} STEPS",
        delta="避障直达终点",
        delta_type="positive",
        icon_name="compass",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 实时微观参数透视状态栏
# ---------------------------------------------------------------------------
render_live_param_status_bar(
    title="RL DYNAMICS // 贝尔曼最优性与 Q-Learning 微观参数",
    badges=[
        {"label": "学习率 α", "value": f"{lr_val:.2f}", "color": "blue"},
        {"label": "折扣因子 γ", "value": f"{gamma_val:.2f}", "color": "amber"},
        {"label": "探索率 ε", "value": f"{agent.epsilon:.2f}", "color": "purple"},
        {"label": "网格尺寸", "value": f"{env.height}x{env.width}", "color": "emerald"},
    ],
    metrics=[
        ("状态空间 |S|", f"{env.height * env.width} states"),
        ("动作空间 |A|", "4 (↑ → ↓ ←)"),
        ("最终探索衰减", f"{agent.epsilon:.3f}"),
    ],
    tag=f"BELLMAN OPTIMAL: {bellman_iters} ITERS",
    tag_color="emerald",
)

# ---------------------------------------------------------------------------
# [D] 核心可视化 1: 网格世界寻路演播
# ---------------------------------------------------------------------------
render_section_heading(
    "GRIDWORLD ARENA // 网格世界环境与智能体避障寻路轨迹",
    icon_name="grid",
    subtext="智能体从起点出发，在试错中学会避开悬崖/陷阱与墙壁，以最短路径直达目标宝藏",
)

st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("D", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">GRIDWORLD PLAYBACK // 二维网格动态演播</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_grid, col_curves = st.columns([1.3, 1])

with col_grid:
    # 绘制网格地貌
    grid_display = np.copy(env.grid).astype(float)
    # 将路径叠加标记
    for r, c in agent_path:
        if grid_display[r, c] == 0:
            grid_display[r, c] = 0.5  # 路径标记

    fig_grid = go.Figure()
    fig_grid.add_trace(
        go.Heatmap(
            z=grid_display,
            colorscale=[
                [0.0, "#f8fafc"],    # 通路
                [0.2, "#e2e8f0"],    # 墙壁
                [0.5, "#bfdbfe"],    # 路径痕迹
                [0.7, "#fecdd3"],    # 悬崖/陷阱
                [1.0, "#bbf7d0"],    # 宝藏终点
            ],
            showscale=False,
        )
    )

    # 标注文字
    annotations = []
    for r in range(env.height):
        for c in range(env.width):
            val = env.grid[r, c]
            if (r, c) == tuple(agent_path[0]):
                text = "START (起点)"
                color = "#1e40af"
            elif val == 3:
                text = "GOAL (宝藏 +1)"
                color = "#065f46"
            elif val == 2:
                text = "CLIFF (悬崖 -1)"
                color = "#9f1239"
            elif val == 1:
                text = "WALL (墙壁)"
                color = "#475569"
            elif (r, c) in agent_path:
                text = "PATH (路线)"
                color = "#2563eb"
            else:
                text = f"({r},{c})"
                color = "#94a3b8"

            annotations.append(
                dict(
                    x=c,
                    y=r,
                    text=text,
                    showarrow=False,
                    font=dict(color=color, size=11, family="JetBrains Mono"),
                )
            )

    fig_grid.update_layout(
        title="智能体最终贪心策略路线 (Agent Optimal Trajectory)",
        annotations=annotations,
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_grid, use_container_width=True)

with col_curves:
    # 绘制训练回报与 TD-Error 曲线
    fig_train = go.Figure()
    fig_train.add_trace(
        go.Scatter(
            y=train_history["returns"],
            mode="lines",
            name="单轮回报 (Return)",
            line=dict(color="#1d4ed8", width=2),
        )
    )
    fig_train.add_trace(
        go.Scatter(
            y=train_history["td_errors"],
            mode="lines",
            name="TD 误差 (TD-Error)",
            line=dict(color="#be123c", width=1.5, dash="dot"),
            yaxis="y2",
        )
    )
    fig_train.update_layout(
        title="Q-Learning 训练收敛曲线 (Episode Returns & TD Errors)",
        xaxis_title="训练轮数 (Episodes)",
        yaxis=dict(title=dict(text="累积回报 (Return)", font=dict(color="#1d4ed8"))),
        yaxis2=dict(
            title=dict(text="TD 误差", font=dict(color="#be123c")),
            overlaying="y",
            side="right",
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(x=0.05, y=0.95),
    )
    st.plotly_chart(fig_train, use_container_width=True)

# ---------------------------------------------------------------------------
# [E] 核心可视化 2: 贝尔曼状态价值曲面 V(s)
# ---------------------------------------------------------------------------
render_section_heading(
    "BELLMAN VALUE SURFACE // 贝尔曼最优状态价值分布 V*(s)",
    icon_name="activity",
    subtext="展示动态规划求解的绝对真值：越靠近终点价值越高，负向惩罚像波纹一样向后反向递减",
)

st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("E", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">VALUE FUNCTION HEATMAP // 状态价值热力面</span>'
    f"</div>",
    unsafe_allow_html=True,
)

col_val2d, col_val3d = st.columns([1, 1.2])

with col_val2d:
    fig_v2d = go.Figure(
        data=go.Heatmap(
            z=opt_V,
            colorscale="Viridis",
            text=[[f"{opt_V[r, c]:.2f}" for c in range(env.width)] for r in range(env.height)],
            texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=11, color="#ffffff"),
            colorbar=dict(title="V*(s)"),
        )
    )
    fig_v2d.update_layout(
        title="2D 贝尔曼最优状态价值矩阵 V*(s)",
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_v2d, use_container_width=True)

with col_val3d:
    # 3D 价值曲面
    x_grid = np.arange(env.width)
    y_grid = np.arange(env.height)
    fig_v3d = go.Figure(
        data=[
            go.Surface(
                z=opt_V,
                colorscale="Viridis",
                showscale=False,
            )
        ]
    )
    fig_v3d.update_layout(
        title="3D 贝尔曼价值地貌流形 (Value Manifold Surface)",
        scene=dict(
            xaxis_title="列 (X)",
            yaxis_title="行 (Y)",
            zaxis_title="V*(s)",
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_v3d, use_container_width=True)

# ---------------------------------------------------------------------------
# [F] 核心可视化 3: 策略箭头场与 Q 值矩阵
# ---------------------------------------------------------------------------
render_section_heading(
    "POLICY QUIVER & Q-TABLE // 最优策略决策箭头场与动作流向",
    icon_name="compass",
    subtext="每个格子的箭头代表当前状态下 Q 值最大的最优动作，直观展现决策流向",
)

st.markdown(
    f'<div id="region-f" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("F", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">POLICY QUIVER MAP // 策略箭头矢量分布</span>'
    f"</div>",
    unsafe_allow_html=True,
)

arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}
arrow_annotations = []

for r in range(env.height):
    for c in range(env.width):
        cell_type = env.grid[r, c]
        if cell_type == 3:
            sym = " 终点"
            col = "#047857"
        elif cell_type == 2:
            sym = " 悬崖"
            col = "#be123c"
        elif cell_type == 1:
            sym = "■ 墙壁"
            col = "#64748b"
        else:
            best_act = opt_policy[r, c]
            sym = arrow_map[best_act]
            col = "#1d4ed8"

        arrow_annotations.append(
            dict(
                x=c,
                y=r,
                text=sym,
                showarrow=False,
                font=dict(color=col, size=16, family="JetBrains Mono", weight=800),
            )
        )

fig_quiver = go.Figure(
    data=go.Heatmap(
        z=np.zeros((env.height, env.width)),
        colorscale=[[0, "#f8fafc"], [1, "#f8fafc"]],
        showscale=False,
    )
)
fig_quiver.update_layout(
    title="全网格最优动作流向图 (Policy Flow Arrows)",
    annotations=arrow_annotations,
    yaxis=dict(autorange="reversed", showticklabels=False),
    xaxis=dict(showticklabels=False),
    margin=dict(l=20, r=20, t=40, b=20),
    height=280,
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
)
st.plotly_chart(fig_quiver, use_container_width=True)

# ---------------------------------------------------------------------------
# [G] 核心可视化 4: 2026 DeepSeek-R1 GRPO 推理涌现演化
# ---------------------------------------------------------------------------
render_section_heading(
    "2026 DEEPSEEK-R1 GRPO REASONING // 组相对策略优化与长思维链顿悟涌现",
    icon_name="cpu",
    subtext="无需训练复杂的 Critic 模型，纯基于规则验证奖励即可自发涌现长思考链与‘Aha Moment’自我反思纠错",
)

st.markdown(
    f'<div id="region-g" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("G", "rose")} <span style="font-weight:800;color:#9f1239;font-size:0.86rem;">GRPO REASONING EVOLUTION // 纯强化学习推理演进仿真</span>'
    f"</div>",
    unsafe_allow_html=True,
)

r1_iters = st.slider(
    "GRPO 纯强化学习训练轮次 (Training Steps)",
    min_value=1,
    max_value=15,
    value=15,
    help="拖动滑块观察思考链 Token 长度如何自发爆发，模型如何学会‘等等，让我重新验证一下...’",
)

r1_data = GRPORunner.simulate_r1_reasoning_evolution(n_iterations=r1_iters)

col_r1_left, col_r1_right = st.columns([1.2, 1])

with col_r1_left:
    fig_r1 = go.Figure()
    fig_r1.add_trace(
        go.Scatter(
            x=r1_data["iterations"],
            y=r1_data["cot_lengths"],
            mode="lines+markers",
            name="思考链长度 (CoT Tokens)",
            line=dict(color="#be123c", width=2.5),
            marker=dict(size=6),
        )
    )
    fig_r1.add_trace(
        go.Scatter(
            x=r1_data["iterations"],
            y=[acc * 100 for acc in r1_data["accuracies"]],
            mode="lines+markers",
            name="复杂数学准确率 (%)",
            line=dict(color="#047857", width=2.5),
            marker=dict(size=6),
            yaxis="y2",
        )
    )
    fig_r1.update_layout(
        title="GRPO 训练轮数与思维链长度 / 准确率爆发曲线",
        xaxis_title="GRPO 优化轮次",
        yaxis=dict(title=dict(text="思考 Token 数", font=dict(color="#be123c"))),
        yaxis2=dict(
            title=dict(text="准确率 (%)", font=dict(color="#047857")),
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(x=0.05, y=0.95),
    )
    st.plotly_chart(fig_r1, use_container_width=True)

with col_r1_right:
    st.markdown("##### [REASONING TRACE // 推理轨迹]  演化阶段输出对比 (Reasoning Traces)")
    active_case = r1_data["cases"][-1] if r1_iters >= 10 else (r1_data["cases"][1] if r1_iters >= 5 else r1_data["cases"][0])
    
    st.markdown(
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #be123c;padding:0.8rem 1rem;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">'
        f'<div style="font-weight:700;color:#9f1239;font-size:0.85rem;margin-bottom:0.3rem;">{active_case["step"]} · {active_case["type"]}</div>'
        f'<div style="font-family:JetBrains Mono;font-size:0.82rem;background:#f8fafc;padding:0.6rem;border-radius:6px;color:#0f172a;line-height:1.6;white-space:pre-wrap;">{active_case["cot"]}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:0.5rem;font-size:0.8rem;color:#64748b;">'
        f'<span>Tokens: <b>{active_case["length"]}</b></span>'
        f'<span>Accuracy: <b style="color:#047857;">{active_case["accuracy"]}</b></span>'
        f'</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
