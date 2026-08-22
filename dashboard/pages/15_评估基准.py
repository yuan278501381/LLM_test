# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 15: 评估基准框架 (Evaluation Harness) - 零基础入门保姆级教学平台

解剖困惑度 (Perplexity)、标准化考场套件 (MMLU / HellaSwag / GSM8K / Safety)、多维能力雷达画像与 Open LLM Leaderboard 排行榜机制。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.evaluation import (
    EvaluationHarness,
    compute_perplexity,
    get_mini_gsm8k,
    get_mini_hellaswag,
    get_mini_mmlu,
    get_mini_safety,
)

st.set_page_config(
    page_title="Evaluation Harness · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="大模型评估基准框架 (Harness)",
    subtitle="从语言困惑度到高阶推理大考：解剖 Perplexity $\\text{PPL} = e^{-\\frac{1}{N}\\sum \\log p(x_i)}$、MMLU / GSM8K 客观考场、多维能力雷达图与排行榜",
    badge_text="MILESTONE 15 // EVALUATION HARNESS",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="大语言模型评估框架与能力画像入门",
    plain_intro=(
        "<b>大模型训练完了，怎么客观证明它比别的模型更聪明？</b><br>"
        "在学术界和工业界，我们绝不靠主观感觉，而是依赖一套严密的<b>标准化考试评估体系 (Evaluation Harness)</b>：<br>"
        "• <b>困惑度 (Perplexity / PPL)</b>：衡量模型在阅读文本时的'吃惊程度'。PPL 越低，说明模型对人类语言越胸有成竹；<br>"
        "• <b>Mini-MMLU (通识学科)</b>：覆盖数理化、文史哲的大综合单选题；<br>"
        "• <b>Mini-HellaSwag (常识推理)</b>：考察模型能否识破荒谬的常识陷阱；<br>"
        "• <b>Mini-GSM8K (小学数学)</b>：考验多步思维链推理能力；<br>"
        "• <b>Mini-Safety (安全合规)</b>：严防越狱与危险红线诱导！"
    ),
    hyperparams_desc=(
        "• <b>模拟模型智力等级</b>：切换'弱模型'（30% 猜对）、'中等模型'（65% 掌握）与'强模型'（90% 专家）。<br>"
        "• <b>评测考卷勾选</b>：自由组合多维度能力测试套件。"
    ),
    telemetry_desc=(
        "• <b>当前模型综合得分</b>：所有已勾选考卷的加权平均准确率。<br>"
        "• <b>困惑度 (PPL)</b>：语言模型对测试语料的交叉熵指数得分。<br>"
        "• <b>总答题正确率</b>：当前模拟模型在客观考场中的答题通过比例。"
    ),
    experiments=[
        "<b>第 1 步【体验全自动考试】</b>：在 Section 2 展开具体的试卷题目，观察模型是如何选出答案并与标准答案比对判分的！",
        "<b>第 2 步【观察雷达能力画像】</b>：在左侧切换【模拟模型等级】为'强模型 (Expert)'，观察 Section 3 中雷达图面积如何全面膨胀！",
        "<b>第 3 步【查阅工业级排行榜】</b>：在 Section 4 查阅模拟的 Open LLM Leaderboard，理解模型参数量与综合得分的正相关规律！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

model_tier = st.sidebar.radio(
    "模拟待测模型能力等级",
    options=["弱模型 (Base TinyGPT · 30% 准确率)", "中等模型 (SFT 7B · 65% 准确率)", "强模型 (Aligned 70B · 90% 准确率)"],
    index=2,
)

all_tasks_dict = {
    "Mini-MMLU (学科通识)": get_mini_mmlu(),
    "Mini-HellaSwag (常识推理)": get_mini_hellaswag(),
    "Mini-GSM8K (数学推理)": get_mini_gsm8k(),
    "Mini-Safety (安全合规)": get_mini_safety(),
}

selected_task_names = st.sidebar.multiselect(
    "参加评测的基准考卷",
    options=list(all_tasks_dict.keys()),
    default=list(all_tasks_dict.keys()),
)

# ---------------------------------------------------------------------------
# 模拟评测执行
# ---------------------------------------------------------------------------
accuracy_prob = 0.90 if "强模型" in model_tier else (0.65 if "中等模型" in model_tier else 0.30)
sim_ppl = 12.4 if "强模型" in model_tier else (38.6 if "中等模型" in model_tier else 145.2)

np.random.seed(42)


def mock_predict_fn(question: str, choices: list[str]) -> int:
    # 模拟真实能力水平的答题预测
    # 查找此题的真实答案
    for t in all_tasks_dict.values():
        for q in t.questions:
            if q.question == question:
                if np.random.rand() < accuracy_prob:
                    return q.answer_idx
                else:
                    # 随机选择一个错误选项
                    wrong_choices = [i for i in range(len(choices)) if i != q.answer_idx]
                    return int(np.random.choice(wrong_choices))
    return 0


harness = EvaluationHarness(tasks=[all_tasks_dict[name] for name in selected_task_names])
scores_dict = harness.run_all(mock_predict_fn)
avg_score = float(np.mean(list(scores_dict.values()))) if len(scores_dict) > 0 else 0.0

total_questions_count = sum([len(all_tasks_dict[name].questions) for name in selected_task_names])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "OVERALL BENCHMARK SCORE // 综合得分",
        f"{avg_score:.1f} / 100",
        delta="多维加权总评",
        delta_type="positive" if avg_score >= 80 else ("neutral" if avg_score >= 60 else "negative"),
        icon_name="target",
    )
    + render_metric_card(
        "PERPLEXITY (PPL) // 困惑度",
        f"{sim_ppl:.1f}",
        delta="越低预测越笃定 (↓)",
        delta_type="positive" if sim_ppl < 20 else "neutral",
        icon_name="activity",
    )
    + render_metric_card(
        "TESTED QUESTIONS // 总考题数",
        f"{total_questions_count} 题",
        delta=f"{len(selected_task_names)} 套综合考卷",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "MODEL INTELLECT TIER // 评测等级",
        model_tier.split(" ")[0],
        delta="全自动 Harness 考场",
        delta_type="positive",
        icon_name="layers",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: 语言模型困惑度 (PPL) 仪表盘与训练演化
# ---------------------------------------------------------------------------
render_section_heading("LANGUAGE MODEL PERPLEXITY (PPL) // 困惑度仪表盘与收敛演化", icon_name="activity")

col_ppl_gauge, col_ppl_curve = st.columns([1, 1.4])

with col_ppl_gauge:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sim_ppl,
            title={"text": "Perplexity (PPL 困惑度)", "font": {"size": 18, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [1, 200]},
                "bar": {"color": "#1d4ed8"},
                "steps": [
                    {"range": [1, 20], "color": "rgba(4, 120, 87, 0.2)"},
                    {"range": [20, 80], "color": "rgba(217, 119, 6, 0.2)"},
                    {"range": [80, 200], "color": "rgba(190, 18, 60, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#be123c", "width": 3},
                    "thickness": 0.75,
                    "value": 150,
                },
            },
        )
    )
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=280)
    fig_gauge = _apply_light_theme(fig_gauge, "当前模型 Perplexity 仪表盘")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_ppl_curve:
    # 模拟训练过程中 PPL 从 200 指数衰减至 10 的曲线
    steps_x = np.arange(1, 51)
    ppl_curve = 10.0 + 190.0 * np.exp(-0.08 * steps_x) + np.random.randn(50) * 1.5
    fig_ppl = go.Figure()
    fig_ppl.add_trace(
        go.Scatter(
            x=steps_x,
            y=ppl_curve,
            mode="lines",
            line=dict(color="#1d4ed8", width=2.5),
            name="PPL 演化曲线",
            hovertemplate="Epoch: %{x}<br>PPL: %{y:.1f}<extra></extra>",
        )
    )
    fig_ppl.update_layout(
        xaxis=dict(title="预训练迭代轮数 (Training Epochs)"),
        yaxis=dict(title="Perplexity (PPL)"),
        margin=dict(l=40, r=20, t=30, b=40),
        height=280,
    )
    fig_ppl = _apply_light_theme(fig_ppl, "预训练全周期困惑度 (PPL) 下降收敛曲线")
    st.plotly_chart(fig_ppl, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 2: Mini 客观基准模拟考场
# ---------------------------------------------------------------------------
render_section_heading("STANDARDIZED BENCHMARK EXAMS // 标准化客观基准模拟考场逐题透视", icon_name="target")

for t_name in selected_task_names:
    cur_task = all_tasks_dict[t_name]
    score = scores_dict.get(t_name, 0.0)
    with st.expander(f"📝 考卷：【{t_name}】— 当前得分：{score:.1f}% ({cur_task.description})", expanded=False):
        for q_idx, q in enumerate(cur_task.questions):
            pred_idx = mock_predict_fn(q.question, q.choices)
            is_correct = (pred_idx == q.answer_idx)
            
            status_badge = "🟢 正确 (PASSED)" if is_correct else "🔴 错误 (FAILED)"
            st.markdown(f"**第 {q_idx + 1} 题【{q.category}】**：{q.question}　`{status_badge}`")
            
            c_cols = st.columns(4)
            for c_i, c_text in enumerate(q.choices):
                with c_cols[c_i]:
                    if c_i == q.answer_idx and c_i == pred_idx:
                        st.success(f"✓ {c_text} (正确答案/模型选择)")
                    elif c_i == q.answer_idx:
                        st.info(f"✓ {c_text} (标准答案)")
                    elif c_i == pred_idx:
                        st.error(f"✗ {c_text} (模型误选)")
                    else:
                        st.markdown(f"○ {c_text}")
            st.divider()

# ---------------------------------------------------------------------------
# Section 3: 多维能力雷达图 (Radar Chart)
# ---------------------------------------------------------------------------
render_section_heading("MULTI-DIMENSIONAL RADAR // 大模型多维能力画像雷达对比", icon_name="cpu")

col_radar_plot, col_radar_desc = st.columns([1.3, 1])

with col_radar_plot:
    task_keys = [t.split(" ")[0] for t in selected_task_names]
    radar_fig = go.Figure()

    # 绘制当前测试模型的雷达线
    r_cur = [scores_dict[k] for k in selected_task_names]
    if len(r_cur) > 0:
        r_cur.append(r_cur[0])
        theta_labels = task_keys + [task_keys[0]]

        radar_fig.add_trace(
            go.Scatterpolar(
                r=r_cur,
                theta=theta_labels,
                fill="toself",
                fillcolor="rgba(29, 78, 216, 0.25)",
                line=dict(color="#1d4ed8", width=2.5),
                name=f"当前测试模型 ({model_tier.split(' ')[0]})",
            )
        )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    radar_fig = _apply_light_theme(radar_fig, "大模型基准考试能力画像雷达图")
    st.plotly_chart(radar_fig, use_container_width=True)

with col_radar_desc:
    with st.container(border=True):
        st.markdown(
            f"""
            #### [EVALUATION METRICS // 工业评测标准]
            - **MMLU**：综合通识考场，检验模型是否具备大学本科级别的跨学科知识储备；
            - **HellaSwag**：常识推理大考，杜绝模型在显而易见的物理常识面前'产生幻觉'；
            - **GSM8K**：小学多步数学应用题，衡量模型链式思考 (CoT) 与精确计算能力；
            - **Safety**：红线越狱与对齐检测，确保模型绝不输出危害人类安全的内容！
            """
        )

# ---------------------------------------------------------------------------
# Section 4: 开放模型排行榜模拟 (Open LLM Leaderboard)
# ---------------------------------------------------------------------------
render_section_heading("OPEN LLM LEADERBOARD // 2026 开放大语言模型权威天梯排行榜", icon_name="layers")

leaderboard_data = {
    "Rank": ["#1", "#2", "#3", "#4", "#5", "[CURRENT]"],
    "Model Name": [
        "Claude 3.5 Sonnet / Opus 4.6",
        "GPT-4o / GPT-5 Omni",
        "LLaMA-3.3-70B-Instruct",
        "DeepSeek-V3 / R1",
        "Qwen-2.5-72B-Chat",
        f"NN-Playground ({model_tier.split(' ')[0]})",
    ],
    "Params": ["Unknown", "Unknown", "70B", "671B (37B Act)", "72B", "Pure NumPy"],
    "MMLU": [88.7, 88.5, 82.3, 88.5, 85.0, scores_dict.get("Mini-MMLU (学科通识)", 0.0)],
    "HellaSwag": [90.2, 89.6, 88.0, 88.9, 87.5, scores_dict.get("Mini-HellaSwag (常识推理)", 0.0)],
    "GSM8K": [96.4, 95.8, 86.5, 95.2, 91.0, scores_dict.get("Mini-GSM8K (数学推理)", 0.0)],
    "Safety": [98.5, 98.2, 94.0, 96.0, 93.5, scores_dict.get("Mini-Safety (安全合规)", 0.0)],
}

df_lb = pd.DataFrame(leaderboard_data)
# 计算综合均分
numeric_cols = ["MMLU", "HellaSwag", "GSM8K", "Safety"]
df_lb["Average Score"] = df_lb[numeric_cols].mean(axis=1).round(1)

st.dataframe(
    df_lb.sort_values(by="Average Score", ascending=False).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# 零基础进阶：大模型评估基准核心公式拆解
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 困惑度 (PPL) 与 Elo 竞技场核心公式全解", expanded=True):
    st.markdown(
        r"""
        ### 0. 核心公式逐字拆解：语言模型困惑度 (Perplexity / PPL)
        $$\\text{PPL} = \\exp\\left(-\\frac{1}{N} \\sum_{i=1}^N \\log P(w_i | w_{<i})\\right) = \\exp(\\text{Cross-Entropy Loss})$$
        
        | 符号 | 中文名称 | 数值范围 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$P(w_i \| w_{<i})$** | **模型猜对真实单词的置信概率** | $0.0 \\sim 1.0$ | 模型每走一步接龙时，给人类真实标准答案打出的预测概率。 |
        | **$-\\log P$** | **单步交叉熵罚分** | $\\ge 0$ | 猜得越准罚分越少；猜得越离谱罚分越大。 |
        | **$\\text{PPL}$** | **困惑度数值 (Perplexity)** | $1.0 \\sim +\\infty$ | **“模型在选词时有多纠结、相当于几选一”**：<br>• $\\text{PPL} = 1.0$：完全成竹在胸（绝对神准）；<br>• $\\text{PPL} = 15$：相当于在 15 个等概率候选词里掷骰子（很聪明）；<br>• $\\text{PPL} = 50000$：完全懵圈，相当于在全词表盲猜（彻底失智）。 |
        
        ---
        
        ### 1. 为什么评测不能只看 MMLU 单项分？
        * **数据污染 (Data Contamination)**：很多模型在预训练时偷偷“背过了”MMLU 题库（泄题作弊）；
        * **Chatbot Arena (盲测竞技场)**：让人类真实用户盲测双盲对话并用 Elo 积分实时打分，是当前公认最难作弊、最权威的真实综合能力试金石！
        """
    )

