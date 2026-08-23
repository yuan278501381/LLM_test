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
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.charts import _apply_light_theme
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_floating_hud_navigator,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
)
from nn_core.evaluation import (
    EvaluationHarness,
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
render_lesson_evidence("M15", show_contract=True)
render_core_result_evidence("M15")

render_hero_header(
    title="大模型评估基准框架 (Harness)",
    subtitle="从困惑度计算到评估协议：区分 Mini 教学题集、概率模拟与正式 benchmark，并理解指标边界",
    badge_text="MILESTONE 15 // EVALUATION HARNESS",
    badge_type="blue",
)

render_floating_hud_navigator(
    [
        {
            "id": "A",
            "name": "考场配置控制台",
            "desc": "在左侧侧边栏切换模拟模型智力等级与参加评测的基准考卷",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解困惑度 PPL 吃惊程度、MMLU/GSM8K 考场与排行榜",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时评估遥测",
            "desc": "显示当前综合得分率、困惑度 PPL、总答题数与胜率统计",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "PPL 仪表盘与考场",
            "desc": "PPL 困惑度仪表盘与四大基准考卷逐题答题透视判分",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "模拟能力画像",
            "desc": "基于预设答对概率的教学雷达图与重复采样对比",
            "color": "blue",
            "target_id": "region-e",
        },
    ]
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="大模型评估框架与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "考场配置控制台",
            "desc": "在左侧侧边栏切换模拟模型智力等级与参加评测的基准考卷",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解困惑度 PPL 吃惊程度、MMLU/GSM8K 考场与排行榜",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时评估遥测",
            "desc": "显示当前综合得分率、困惑度 PPL、总答题数与胜率统计",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "PPL 仪表盘与考场",
            "desc": "PPL 困惑度仪表盘与四大基准考卷逐题答题透视判分",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "模拟能力画像",
            "desc": "基于预设答对概率的教学雷达图与重复采样对比",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        "<b>大模型训练完了，怎么客观证明它比别的模型更聪明？</b><br>"
        "可靠评估需要公开任务、数据、提示模板、评分规则和不确定性，而不能只看主观样例。<br>"
        "• <b>困惑度 (Perplexity / PPL)</b>：在固定 tokenizer、数据和协议下度量平均负对数似然；跨协议数值通常不可直接比较；<br>"
        "• <b>MMLU-style 自建教学题</b>：多学科单选练习，非 MMLU 子集或翻译；<br>"
        "• <b>HellaSwag-style 自建教学题</b>：常识续写形式练习，未使用原数据；<br>"
        "• <b>GSM8K-style 自建教学题</b>：多步算术练习，未使用原数据；<br>"
        "• <b>Safety-style 自建教学题</b>：仅用于演示评分流程，不构成安全评测。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>模拟模型智力等级</b>：切换'弱模型'（30% 猜对）、'中等模型'（65% 掌握）与'强模型'（90% 专家）。<br>"
        f"• <b>评测考卷勾选</b>：自由组合多维度能力测试套件。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[C. 评估遥测]', 'emerald', target_id='region-c')} 评估</b>：综合得分率与 PPL 困惑度。<br>"
        f"• <b>在 {anchor_badge('[D. 客观考场]', 'purple', target_id='region-d')} 透视</b>：逐题答题比对与 PPL 收敛曲线。<br>"
        f"• <b>在 {anchor_badge('[E. 模拟能力画像]', 'blue', target_id='region-e')} 查阅</b>：预设答对概率下的随机采样结果。"
    ),
    experiments=[
        f"<b>第 1 步【体验全自动考试】</b>：在 {anchor_badge('[D. 客观考场]', 'purple', target_id='region-d')} 展开具体的试卷题目，观察模型是如何选出答案并与标准答案比对判分的！",
        f"<b>第 2 步【观察雷达能力画像】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 切换【模拟模型等级】为'强模型 (Expert)'，观察 {anchor_badge('[E. 能力雷达]', 'blue', target_id='region-e')} 雷达图面积如何全面膨胀！",
        "<b>第 3 步【观察采样方差】</b>：比较预设能力概率与有限题量实测分数，理解小题集成绩会随随机样本波动。",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>EVALUATION CONTROLS // 评估控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 侧边栏参数控制
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

model_tier_hints = {
    "模拟配置 A (弱先验基准)": "教学模拟档位 · 知识与推理先验较弱，合成答题准确率 ~30%",
    "模拟配置 B (中等先验基准)": "教学模拟档位 · 基础指令与上下文遵循，合成答题准确率 ~65%",
    "模拟配置 C (强先验基准)": "教学模拟档位 · 高推理与对齐护栏，合成答题准确率 ~90%",
}

model_tier = st.sidebar.radio(
    "模拟待测模型能力等级",
    options=list(model_tier_hints.keys()),
    format_func=lambda o: f"**{o}**\n\n↳ *{model_tier_hints[o]}*",
    index=2,
)

all_tasks = (get_mini_mmlu(), get_mini_hellaswag(), get_mini_gsm8k(), get_mini_safety())
all_tasks_dict = {task.name: task for task in all_tasks}

selected_task_names = st.sidebar.multiselect(
    "参加评测的基准考卷",
    options=list(all_tasks_dict.keys()),
    default=list(all_tasks_dict.keys()),
)

# ---------------------------------------------------------------------------
# 模拟评测执行
# ---------------------------------------------------------------------------
accuracy_prob = 0.90 if "强先验" in model_tier else (0.65 if "中等先验" in model_tier else 0.30)
sim_ppl = 12.4 if "强先验" in model_tier else (38.6 if "中等先验" in model_tier else 145.2)

evaluation_rng = np.random.default_rng(42)
simulated_predictions: dict[str, int] = {}
for task in all_tasks:
    for question in task.questions:
        if evaluation_rng.random() < accuracy_prob:
            simulated_predictions[question.question] = question.answer_idx
        else:
            wrong_choices = [i for i in range(len(question.choices)) if i != question.answer_idx]
            simulated_predictions[question.question] = int(evaluation_rng.choice(wrong_choices))


def mock_predict_fn(question: str, choices: list[str]) -> int:
    del choices
    if question not in simulated_predictions:
        raise KeyError(f"未注册的教学题: {question}")
    return simulated_predictions[question]


harness = EvaluationHarness(tasks=[all_tasks_dict[name] for name in selected_task_names])
scores_dict = harness.run_all(mock_predict_fn)
avg_score = float(np.mean(list(scores_dict.values()))) if len(scores_dict) > 0 else 0.0

total_questions_count = sum([len(all_tasks_dict[name].questions) for name in selected_task_names])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">EVALUATION TELEMETRY // 实时评估遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "OVERALL BENCHMARK SCORE // 综合得分",
        f"{avg_score:.1f} / 100",
        delta="多维加权总评",
        delta_type="positive"
        if avg_score >= 80
        else ("neutral" if avg_score >= 60 else "negative"),
        icon_name="target",
    )
    + render_metric_card(
        "SIMULATED PPL // 模拟困惑度",
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
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">PPL DASHBOARD & BENCHMARK EXAMS // 困惑度与标准化考场</span>'
    f"</div>",
    unsafe_allow_html=True,
)

render_live_param_status_bar(
    title="EVALUATION HARNESS & BENCHMARK DYNAMICS // 评测套件与困惑度参数",
    badges=[
        {"label": "Model Tier", "value": f"{model_tier.split()[0]}", "color": "blue"},
        {"label": "Simulated PPL", "value": f"{sim_ppl:.1f}", "color": "amber"},
        {"label": "Pass Rate", "value": f"{accuracy_prob:.0%}", "color": "emerald"},
        {"label": "Overall Score", "value": f"{avg_score:.1f}/100", "color": "purple"},
    ],
    metrics=[
        ("参评科目数", f"{len(selected_task_names)} 科"),
        ("总题目数量", f"{total_questions_count} 题"),
        ("评估协议", "Zero-Shot Harness"),
    ],
    tag="BENCHMARK PASS" if avg_score >= 60 else "RE-ALIGNMENT NEEDED",
    tag_color="emerald" if avg_score >= 60 else "rose",
)

render_section_heading("SIMULATED PERPLEXITY // 困惑度计算示意（非模型实测）", icon_name="activity")

st.warning(
    "本页的答题结果由预设答对概率驱动，PPL 与训练曲线也由公式曲线模拟；所有数值均不是任何真实模型或正式 benchmark 的成绩。"
)

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
    st.plotly_chart(fig_gauge, width="stretch")

with col_ppl_curve:
    # 模拟训练过程中 PPL 从 200 指数衰减至 10 的曲线
    steps_x = np.arange(1, 51)
    ppl_rng = np.random.default_rng(42)
    ppl_curve = 10.0 + 190.0 * np.exp(-0.08 * steps_x) + ppl_rng.normal(0.0, 1.5, 50)
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
    fig_ppl = _apply_light_theme(fig_ppl, "模拟 PPL 下降曲线（非训练日志）")
    st.plotly_chart(fig_ppl, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 困惑度 (PPL) 仪表盘与收敛曲线", expanded=False):
    st.markdown(
        """
        * **左图【PPL 仪表盘】**：PPL 是平均负对数似然的指数；仅能在相同 tokenizer、语料与评估协议下公平比较，固定色带不是通用能力等级。
        * **右图【PPL 下降曲线】**：横轴【训练轮数】，纵轴【PPL 数值】。下降表示在该评估数据上的平均预测概率提高，但不能单独证明事实性、推理或安全性。
        """
    )

# ---------------------------------------------------------------------------
# Section 2: Mini 客观基准模拟考场
# ---------------------------------------------------------------------------
render_section_heading(
    "STANDARDIZED BENCHMARK EXAMS // 标准化客观基准模拟考场逐题透视", icon_name="target"
)

for t_name in selected_task_names:
    cur_task = all_tasks_dict[t_name]
    score = scores_dict.get(t_name, 0.0)
    with st.expander(
        f"[EVAL] 考卷：【{t_name}】— 当前得分：{score:.1f}% ({cur_task.description})",
        expanded=False,
    ):
        for q_idx, q in enumerate(cur_task.questions):
            pred_idx = mock_predict_fn(q.question, q.choices)
            is_correct = pred_idx == q.answer_idx

            status_badge = "[PASSED] 正确 (PASSED)" if is_correct else "[FAILED] 错误 (FAILED)"
            st.markdown(f"**第 {q_idx + 1} 题【{q.category}】**：{q.question}　`{status_badge}`")

            c_cols = st.columns(4)
            for c_i, c_text in enumerate(q.choices):
                with c_cols[c_i]:
                    if c_i == q.answer_idx and c_i == pred_idx:
                        st.success(f"[OK]  {c_text} (正确答案/模型选择)")
                    elif c_i == q.answer_idx:
                        st.info(f"[OK]  {c_text} (标准答案)")
                    elif c_i == pred_idx:
                        st.error(f" {c_text} (模型误选)")
                    else:
                        st.markdown(f"○ {c_text}")
            st.divider()

# ---------------------------------------------------------------------------
# Section 3: 多维能力雷达图 (Radar Chart)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1.2rem;">'
    f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">MODEL CAPABILITY PROFILE // 模拟能力画像</span>'
    f"</div>",
    unsafe_allow_html=True,
)
render_section_heading("MULTI-DIMENSIONAL RADAR // 大模型多维能力画像雷达对比", icon_name="cpu")

col_radar_plot, col_radar_desc = st.columns([1.3, 1])

with col_radar_plot:
    task_keys = [t.split(" ")[0] for t in selected_task_names]
    radar_fig = go.Figure()

    # 绘制当前测试模型的雷达线
    r_cur = [scores_dict[k] for k in selected_task_names]
    if len(r_cur) > 0:
        r_cur.append(r_cur[0])
        theta_labels = [*task_keys, task_keys[0]]

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
    st.plotly_chart(radar_fig, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 多维基准考试雷达图", expanded=False):
        st.markdown(
            """
            * **各轴顶点**：学科通识 (MMLU)、常识推理 (HellaSwag)、数学应用 (GSM8K)、安全合规 (Safety)。
            * **[比较边界]**：各轴分数分别描述对应任务；雷达面积依赖轴顺序与尺度，不是有统计意义的“通用能力总分”，应逐轴并结合置信区间比较。
            """
        )

with col_radar_desc, st.container(border=True):
    st.markdown(
        """
            #### [EVALUATION METRICS // 工业评测标准]
            - **MMLU**：综合通识考场，检验模型是否具备大学本科级别的跨学科知识储备；
            - **HellaSwag**：常识推理大考，杜绝模型在显而易见的物理常识面前'产生幻觉'；
            - **GSM8K**：小学多步数学应用题，衡量模型链式思考 (CoT) 与精确计算能力；
            - **Safety**：用有限测试样例探测部分安全风险；通过题集不能保证模型在所有输入下安全。
            """
    )

# ---------------------------------------------------------------------------
# Section 4: 预设概率与有限样本得分
# ---------------------------------------------------------------------------
render_section_heading("SAMPLING VARIANCE // 预设答对概率与有限题集得分", icon_name="layers")

st.dataframe(
    {
        "项目": ["预设答对概率", "本次 Mini 题集综合得分", "总题数", "证据性质"],
        "数值": [
            f"{accuracy_prob * 100:.0f}%",
            f"{avg_score:.1f}%",
            str(total_questions_count),
            "概率模拟",
        ],
        "可以说明": [
            "mock predictor 的生成规则",
            "本次随机样本结果",
            "有限样本规模",
            "评估流程教学",
        ],
        "不能说明": ["真实模型能力", "正式 benchmark 成绩", "总体能力置信区间", "任一生产模型排名"],
    },
    width="stretch",
    hide_index=True,
)

# ---------------------------------------------------------------------------
# 零基础进阶：大模型评估基准核心公式拆解
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 困惑度 (PPL) 与 Elo 竞技场核心公式全解", expanded=True
):
    st.markdown(
        r"""
        ### 0. 核心公式逐字拆解：语言模型困惑度 (Perplexity / PPL)
        $$\\text{PPL} = \\exp\\left(-\\frac{1}{N} \\sum_{i=1}^N \\log P(w_i | w_{<i})\\right) = \\exp(\\text{Cross-Entropy Loss})$$

        | 符号 | 中文名称 | 数值范围 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---:|:---|
        | **$P(w_i \| w_{<i})$** | **模型猜对真实单词的置信概率** | $0.0 \\sim 1.0$ | 模型每走一步接龙时，给人类真实标准答案打出的预测概率。 |
        | **$-\\log P$** | **单步交叉熵罚分** | $\\ge 0$ | 猜得越准罚分越少；猜得越离谱罚分越大。 |
        | **$\\text{PPL}$** | **困惑度数值 (Perplexity)** | $1.0 \\sim +\\infty$ | 在固定 tokenizer、语料和计算协议下，PPL 是平均负对数似然的指数。较低通常表示对该数据分布赋予更高概率，但不直接衡量事实正确性、推理、安全或综合智能。 |

        ---

        ### 1. 为什么评测不能只看 MMLU 单项分？
        * **数据污染 (Data Contamination)**：很多模型在预训练时偷偷“背过了”MMLU 题库（泄题作弊）；
        * **人类偏好盲测**：可以补充静态题集，但会受到参与者、提示分布、位置偏差和统计方法影响，也不是单一“权威真值”。
        """
    )
