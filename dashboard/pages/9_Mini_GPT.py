# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 9: Mini-GPT 文本生成 (Auto-Regressive Text Generation) - 零基础入门保姆级教学平台

解剖现代大语言模型 (ChatGPT/GPT-4) 的自回归生成原理：Next-Token 概率预测、Temperature 温度采样、Top-k 过滤与实时注意力追踪。
"""

import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib
import numpy as np
import streamlit as st

import dashboard.components.charts

importlib.reload(dashboard.components.charts)

from dashboard.components.charts import plot_attention_heatmap_nlp, plot_token_probabilities
from dashboard.components.pedagogy import render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
    render_text_stream_box,
)
from nn_core.embeddings import get_mini_vocab, get_pretrained_embeddings
from nn_core.gpt import TinyGPT

st.set_page_config(
    page_title="Mini-GPT · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M09")

render_hero_header(
    title="Mini-GPT 文本生成与自回归采样",
    subtitle="将所有积木拼成完全体：自回归逐 Token 采样、Next-Token 概率分布、Temperature 创造力调控与实时注意力追踪",
    badge_text="MILESTONE 09 // AUTO-REGRESSIVE GPT",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="Mini-GPT 文本生成与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "文本生成控制台",
            "desc": "在左侧侧边栏调节 Prompt、Temperature 温度与 Top-K 候选池",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解 ChatGPT 逐词猜下一个字机理与 KV-Cache",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时生成遥测",
            "desc": "显示当前置信度最高的下一词预测、创造力系数与序列长度",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "实时文本流展示",
            "desc": "打字机效果逐词呈现模型吐出的新词汇与上下文窗口",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "候选词概率排行榜",
            "desc": "柱状图实时揭秘模型心中的候选词概率分布与温度平滑度",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>终于到了见证奇迹的时刻——我们把前面的所有技术拼成了一个微型 ChatGPT！</b><br>"
        f"GPT 的本质其实非常纯粹：<b>它永远只做一件事——猜下一个词 (Next-Token Prediction)</b>。<br>"
        f"它给词表中每一个词打一个概率分（如'女王': 45%, '国王': 20%, '桌子': 0.1%）；<br>"
        f"然后根据你在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 设定的<b>温度 (Temperature)</b> 进行概率轮盘赌，抽中一个词拼到 {anchor_badge('[D. 文本流]', 'purple', target_id='region-d')}，再重复这个过程！<br><br>"
        f"<b>【2026 前沿拓展】：KV-Cache 推理加速</b><br>"
        f"每次生成新词时，模型不需要把前面的所有词都重新计算一遍，而是将历史词汇的 Key/Value 缓存起来。<br>"
        f"这使得 Transformer 的推理复杂度从 $O(N^2)$ 骤降为 $O(1)$，是实现流式实时响应的核心关键！"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>提示词 (Prompt)</b>：给 GPT 的开头引导语（如 <code>the king and</code>）。<br>"
        f"• <b>采样温度 (Temperature)</b>：控制创造力。T=0.1 确定保守，T=0.8 均衡灵动。<br>"
        f"• <b>Top-k 过滤</b>：只在概率最高的前 K 个词中采样，过滤长尾词。<br>"
        f"• <b>生成长度 (Max Tokens)</b>：让模型接续生成的词数。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 实时文本流]', 'purple', target_id='region-d')} 观测</b>：打字机效果逐词呈现 GPT 吐出的新词汇。<br>"
        f"• <b>在 {anchor_badge('[E. 概率排行榜]', 'blue', target_id='region-e')} 揭秘</b>：实时揭秘模型心中的'候选词排行榜'。<br>"
        f"• <b>在 {anchor_badge('[C. 实时指标]', 'emerald', target_id='region-c')} 评估</b>：下一词预测置信度与创造力状态。"
    ),
    experiments=[
        f"<b>第 1 步【体验确定性生成】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把【Temperature】设为 <code>0.1</code>，点击生成，观察 {anchor_badge('[E. 概率图]', 'blue', target_id='region-e')} 顶端出现尖锐的绝对优势词！",
        f"<b>第 2 步【体验高创造力】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 把【Temperature】调到 <code>1.5</code>，观察概率柱状图变得平坦，模型给出随机出人意料的词汇组合！",
        f"<b>第 3 步【观测显存暴涨】</b>：滚动到底部实验室，点击生成后，观察 KV-Cache Size 随着生成步数的增加呈现线性甚至几何级膨胀的现象！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>GENERATION CONTROLS // 文本生成控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 模型与词表初始化
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
inv_vocab = {v: k for k, v in raw_vocab.items()}
vocab_size = len(vocab_words)


# 实例化 TinyGPT 模型
@st.cache_resource
def load_tiny_gpt():
    gpt_model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=24,
        d_model=32,
        num_heads=4,
        num_layers=2,
    )
    # 注入具有语义特征的预训练嵌入
    pretrained_w = get_pretrained_embeddings(vocab_size, d_model=32)
    gpt_model.wte = pretrained_w
    return gpt_model


gpt = load_tiny_gpt()

# ---------------------------------------------------------------------------
# 侧边栏参数面板
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与控制")

prompt_presets = {
    "经典王族开头": "the king and queen",
    "动作与地点开头": "the cat went to paris",
    "动物伙伴开头": "the puppy and kitten run",
    "国家与首都开头": "the china and japan were",
    "自定义 Prompt...": "",
}

prompt_opts = list(prompt_presets.keys())
selected_prompt_key = st.sidebar.segmented_control(
    "Prompt 预设",
    options=prompt_opts,
    default=prompt_opts[0],
)
selected_prompt_key = selected_prompt_key or prompt_opts[0]

if "自定义" in selected_prompt_key:
    custom_prompt = st.sidebar.text_input("输入自定义英文 Prompt", "the king and")
    current_prompt_text = custom_prompt if custom_prompt.strip() else "the king and"
else:
    current_prompt_text = prompt_presets[selected_prompt_key]

temperature = st.sidebar.slider(
    "采样温度 (Temperature)",
    min_value=0.1,
    max_value=2.0,
    value=0.7,
    step=0.1,
    help="调节 Softmax 概率平滑度。越小越趋近贪婪选择（确定/保守），越大越随机平权（高创造力/不稳定）。",
)

top_k = st.sidebar.slider(
    "Top-k 截断采样",
    min_value=1,
    max_value=min(20, vocab_size),
    value=8,
    help="仅保留预测概率最高的前 k 个 Token 并重新归一化采样，彻底剔除低概率无关词汇。",
)

gen_tokens_count = st.sidebar.slider(
    "生成 Token 数量",
    min_value=1,
    max_value=10,
    value=4,
    help="模型连续自回归预测的新词数量。",
)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_generate_all = st.button("RUN // 一键自回归", type="primary", width="stretch")
with col_btn2:
    btn_step_one = st.button("STEP // 单步推演", width="stretch")

# ---------------------------------------------------------------------------
# 会话状态管理 (Session State)
# ---------------------------------------------------------------------------
if (
    "current_generated_tokens" not in st.session_state
    or st.session_state.get("last_prompt") != current_prompt_text
):
    init_tokens = [
        w.lower().strip(",.!?") for w in current_prompt_text.strip().split() if w.strip()
    ]
    st.session_state.current_generated_tokens = init_tokens
    st.session_state.last_prompt = current_prompt_text
    st.session_state.step_counter = 0


# ---------------------------------------------------------------------------
# 模型预测核心函数
# ---------------------------------------------------------------------------
def compute_next_token_distribution(tokens_list: list[str], temp: float, k: int):
    """根据当前 Token 列表计算下一个 Token 的概率分布"""
    # 截断到最大序列长度
    truncated = tokens_list[-20:]
    token_ids = [raw_vocab.get(w, 0) for w in truncated]
    context = np.array([token_ids], dtype=np.int32)

    # 前向传播得到 logits: (1, T, vocab_size)
    logits = gpt.forward(context)
    next_logits = logits[0, -1, :] / max(temp, 1e-4)

    # 减去最大值防止溢出
    exp_logits = np.exp(next_logits - np.max(next_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Top-K 截断
    top_indices = np.argsort(probs)[-k:]
    masked_probs = np.zeros_like(probs)
    masked_probs[top_indices] = probs[top_indices]
    masked_probs = masked_probs / (np.sum(masked_probs) + 1e-12)

    return masked_probs, truncated


# ---------------------------------------------------------------------------
# 触发生成逻辑
# ---------------------------------------------------------------------------
if btn_generate_all:
    tokens_so_far = [
        w.lower().strip(",.!?") for w in current_prompt_text.strip().split() if w.strip()
    ]
    for _ in range(gen_tokens_count):
        probs, _ = compute_next_token_distribution(tokens_so_far, temperature, top_k)
        sampled_id = np.random.choice(vocab_size, p=probs)
        sampled_word = inv_vocab.get(sampled_id, "the")
        tokens_so_far.append(sampled_word)
    st.session_state.current_generated_tokens = tokens_so_far
    st.session_state.step_counter += gen_tokens_count

elif btn_step_one:
    tokens_so_far = list(st.session_state.current_generated_tokens)
    probs, _ = compute_next_token_distribution(tokens_so_far, temperature, top_k)
    sampled_id = np.random.choice(vocab_size, p=probs)
    sampled_word = inv_vocab.get(sampled_id, "the")
    tokens_so_far.append(sampled_word)
    st.session_state.current_generated_tokens = tokens_so_far
    st.session_state.step_counter += 1

# 获取当前状态下的预测概率
current_tokens = st.session_state.current_generated_tokens
current_probs, current_context_tokens = compute_next_token_distribution(
    current_tokens, temperature, top_k
)

top_predicted_id = int(np.argmax(current_probs))
top_predicted_word = inv_vocab.get(top_predicted_id, "the")
top_predicted_prob = float(current_probs[top_predicted_id])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-c" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;">'
    f'{anchor_badge("C", "emerald")} <span style="font-weight:800;color:#065f46;font-size:0.86rem;">GENERATION TELEMETRY // 实时生成与采样遥测</span>'
    f"</div>",
    unsafe_allow_html=True,
)
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "TOP PREDICTION // 下一词预测之王",
        top_predicted_word.upper(),
        delta=f"置信度 {top_predicted_prob * 100:.1f}%",
        delta_type="positive" if top_predicted_prob > 0.4 else "neutral",
        icon_name="target",
    )
    + render_metric_card(
        "TEMPERATURE // 创造力系数",
        f"T = {temperature:.1f}",
        delta="确定模式"
        if temperature < 0.4
        else ("均衡创作" if temperature <= 1.0 else "发散混乱"),
        delta_type="positive" if 0.4 <= temperature <= 1.0 else "neutral",
        icon_name="zap",
    )
    + render_metric_card(
        "TOP-K CANDIDATES // 采样候选池",
        f"{top_k} TOKENS",
        delta="已过滤低频长尾",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "TOTAL TOKENS // 当前序列长度",
        f"{len(current_tokens)} TOKENS",
        delta=f"已接续生成 +{len(current_tokens) - len(current_prompt_text.split())} 词",
        delta_type="positive",
        icon_name="database",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 主视图区 1：自回归实时文本输出展示
# ---------------------------------------------------------------------------
st.markdown(
    f'<div id="region-d" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:1rem;">'
    f'{anchor_badge("D", "purple")} <span style="font-weight:800;color:#5b21b6;font-size:0.86rem;">LIVE TEXT STREAM // 实时自回归文本流 (打字机效果)</span>'
    f"</div>",
    unsafe_allow_html=True,
)
prompt_len = len(current_prompt_text.strip().split())
render_text_stream_box(current_tokens, prompt_len)

top_idx = int(np.argmax(current_probs))
top_token_name = vocab_words[top_idx]
top_token_prob = float(current_probs[top_idx])

render_live_param_status_bar(
    title="NEXT-TOKEN GENERATION DYNAMICS // 自回归生成与采样参数",
    badges=[
        {"label": "Temperature T", "value": f"{temperature:.2f}", "color": "blue"},
        {"label": "Top-K", "value": f"{top_k}", "color": "amber"},
        {"label": "Argmax Token", "value": f"'{top_token_name}'", "color": "emerald"},
        {"label": "Peak P", "value": f"{top_token_prob:.1%}", "color": "purple"},
    ],
    metrics=[
        ("序列长度 T", f"{len(current_tokens)} tokens"),
        ("词表规模 |V|", f"{len(vocab_words)}"),
        ("解码策略", "Greedy" if temperature < 0.05 else f"Top-{top_k} Sampling"),
    ],
    tag=f"GENERATING: +{len(current_tokens) - prompt_len} TOKENS",
    tag_color="emerald",
)

# ---------------------------------------------------------------------------
# 主视图区 2：下一词概率柱状图 & 实时注意力热力图
# ---------------------------------------------------------------------------
col_bar, col_attn = st.columns([1.2, 1])

with col_bar:
    st.markdown(
        f'<div id="region-e" class="interactive-region" style="padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;margin-bottom:0.5rem;margin-top:0.4rem;">'
        f'{anchor_badge("E", "blue")} <span style="font-weight:800;color:#1e40af;font-size:0.86rem;">NEXT-TOKEN PROBABILITIES // 候选词概率排行榜 (T={temperature})</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    fig_probs = plot_token_probabilities(
        token_probs=current_probs,
        vocab=vocab_words,
        top_k=top_k,
        title=f"PROBABILITY DISTRIBUTION // 下一词概率柱状图 (Top {top_k})",
    )
    st.plotly_chart(fig_probs, width="stretch")
    with st.expander("[HOW TO READ // 读图指南] 下一词概率预测柱状图", expanded=False):
        st.markdown(
            """
            * **横轴【预测候选单词】** 与 **纵轴【置信度概率 (0% ~ 100%)】**。
            * **柱子高度**：模型基于所有历史上下文，计算出下一个词是该候选词的几率。
            * **[OPTIMAL // 采样机制]**：当 Temperature=0 时强制选择最高的柱子 (Greedy)；开启 Top-K 后，仅在最高的 K 根柱子里按比例轮盘赌采样。
            """
        )

with col_attn:
    render_section_heading("STEP ATTENTION FOCUS // 当前步注意力聚光灯", icon_name="zap")
    # 提取多层 Transformer 的注意力权重
    all_attn = gpt.get_all_attention_weights()
    if all_attn:
        # 取最后一层第 0 头的注意力
        last_layer_attn = all_attn[-1][0, 0]
        fig_realtime_attn = plot_attention_heatmap_nlp(
            attention_weights=last_layer_attn,
            tokens_x=current_context_tokens,
            tokens_y=current_context_tokens,
            title="Transformer Layer 2 Attention",
        )
        st.plotly_chart(fig_realtime_attn, width="stretch")
        with st.expander("[HOW TO READ // 读图指南] 当前步因果注意力聚焦矩阵", expanded=False):
            st.markdown(
                """
                * **最底下一行**：最新生成的单词正在把“聚光灯”打在前面哪几个词上。
                * **高亮亮点**：说明最新词与历史哪个特定词产生了强烈的因果逻辑关联。
                """
            )
    else:
        st.info("模型正在初始化注意力矩阵...")

# ---------------------------------------------------------------------------
# 底部理论对比：Temperature 对概率分布的几何塑造
# ---------------------------------------------------------------------------
render_section_heading("TEMPERATURE DYNAMICS // 温度参数对概率分布的塑造原理", icon_name="activity")

col_t1, col_t2, col_t3 = st.columns(3)

# 构造一个基准的 Logits 分布演示
toy_logits = np.array([4.0, 3.2, 2.1, 1.5, 0.8, -0.5, -1.2])
toy_words = ["queen", "king", "princess", "prince", "woman", "cat", "dog"]


def toy_softmax(l_arr, t_val):
    scaled = l_arr / t_val
    exps = np.exp(scaled - np.max(scaled))
    return exps / np.sum(exps)


with col_t1:
    with st.container(border=True):
        st.markdown("#### [GREEDY // 绝对确定] T = 0.1")
        st.caption("最高分被无限放大，其余概率归零。每次点击生成的词绝对确定。")
        fig_t1 = plot_token_probabilities(
            toy_softmax(toy_logits, 0.1), toy_words, top_k=5, title="T=0.1 极度尖锐 (Greedy)"
        )
        st.plotly_chart(fig_t1, width="stretch")

with col_t2:
    with st.container(border=True):
        st.markdown("#### [CREATIVE // 均衡创作] T = 0.7")
        st.caption("高分词依然占优，但赋予次高分词适度机会，展现生动多样的表达。")
        fig_t2 = plot_token_probabilities(
            toy_softmax(toy_logits, 0.7), toy_words, top_k=5, title="T=0.7 经典推荐 (Balanced)"
        )
        st.plotly_chart(fig_t2, width="stretch")

with col_t3:
    with st.container(border=True):
        st.markdown("#### [CHAOTIC // 随机发散] T = 2.0")
        st.caption("差异被强行抹平，所有词概率趋同，模型极易产生乱码和幻觉。")
        fig_t3 = plot_token_probabilities(
            toy_softmax(toy_logits, 2.0), toy_words, top_k=5, title="T=2.0 极度平坦 (Uniform)"
        )
        st.plotly_chart(fig_t3, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 采样温度 Temperature 概率塑形对比", expanded=False):
    st.markdown(
        """
        * **横向对比三张图**：
          * **T=0.1 (左图，极度陡峭)**：第 1 个词独占近 100% 概率，模型回答严谨、确定但死板；
          * **T=0.7 (中图，经典平衡)**：保留首选词的明显优势，同时给 2~3 个同义词合理的微弱几率，生成生动富有文采；
          * **T=2.0 (右图，极度平缓)**：所有柱子高度几乎拉平，模型开始胡言乱语。
        """
    )

# ---------------------------------------------------------------------------
# 2026 前沿拓展：KV-Cache 自回归推理加速与显存占用实时监控
# ---------------------------------------------------------------------------
render_section_heading(
    "2026 INFERENCE ACCELERATION // 自回归推理加速：KV-Cache 显存与算力优化", icon_name="zap"
)

col_kv_info, col_kv_stat = st.columns([1.2, 1])

with col_kv_info:
    with st.container(border=True):
        st.markdown("#### [WHY KV-CACHE? // 为什么推理必须使用 KV 缓存？]")
        st.markdown(
            """
            - **无缓存推理的噩梦 ($O(N^2)$ 计算浪费)**：
              在生成第 $t$ 个词时，如果不存历史键值对，必须将前 $t-1$ 个历史 Token 重新做完整的矩阵乘法，生成 1000 个 Token 需要重复计算 $1000^2 / 2 = 50$ 万次！
            - **KV-Cache 破局 ($O(1)$ 单步增量推理)**：
              过去 Token 的 Key 和 Value 已经计算定型，直接就地缓存在显存中。每一步生成只需计算**当前单个 Token** 的 Q/K/V 投影，推理吞吐暴增数十倍！
            """
        )

with col_kv_stat:
    with st.container(border=True):
        st.markdown("#### [REAL-TIME TELEMETRY // 当前推理会话 KV 显存开销]")

        current_seq_tokens = len(current_tokens)
        from nn_core.kv_cache import KVCache

        kv_tracker = KVCache(
            num_layers=2, max_batch_size=1, max_seq_len=64, num_kv_heads=2, head_dim=16
        )
        # 模拟填入当前 token
        fake_kv = np.zeros((1, 2, current_seq_tokens, 16))
        kv_tracker.update(0, fake_kv, fake_kv)
        kv_tracker.update(1, fake_kv, fake_kv)
        kv_stats = kv_tracker.get_memory_stats()

        flops_saved_ratio = max(1.0, current_seq_tokens / 2.0)

        st.metric(
            label="当前 KV 缓存物理显存占用",
            value=f"{kv_stats['used_kb']:.2f} KB",
            delta=f"缓存槽位利用率 {kv_stats['utilization_percent']:.1f}% ({int(kv_stats['current_tokens'])}/{int(kv_stats['max_tokens'])} Tokens)",
            delta_color="normal",
        )
        st.metric(
            label="累计避免的重复计算开销 (FLOPs 节省)",
            value=f"{flops_saved_ratio:.1f}× 吞吐提速",
            delta=f"避免了过去 {current_seq_tokens} 步的冗余重算",
            delta_color="normal",
        )

# ---------------------------------------------------------------------------
# 零基础进阶：GPT 自回归生成与采样核心公式拆解
# ---------------------------------------------------------------------------
with st.expander(
    "[GROWTH GUIDE // 成长指南] 自回归生成与采样核心公式拆解（ChatGPT 说话的秘密）", expanded=True
):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：带温度调节的自回归概率分布
        $$P(w_i | w_{<t}) = \\frac{\\exp(z_i / T)}{\\sum_{j \\in \\text{Top-}K} \\exp(z_j / T)}$$
        
        | 符号 | 中文名称 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---|
        | **$w_{<t}$** | **历史上下文 (Context)** | 截至目前已经生成的所有上文词汇。 |
        | **$z_i$** | **第 $i$ 个词的原始得分 (Logit)** | 分类头未归一化的原始打分。打分越高，代表模型越倾向于选这个词。 |
        | **$T$** | **采样温度系数 (Temperature)** | **创造力调音旋钮**：<br>• $T \\to 0$：严谨保守（永远选得分最高的第 1 名，无任何废话与错别字）；<br>• $T = 0.7$：最佳平衡（既讲逻辑，又有生动的文学修辞）；<br>• $T > 1.5$：发散发癫（所有词机会均等，开始胡言乱语）。 |
        | **$\\text{Top-}K$** | **截断候选词池** | **长尾毒草过滤器**。只允许在概率最高的前 $K$ 个词里抽签，概率排在第 $K+1$ 名开外的冷门词直接被一刀切掉，防止生成生僻乱码。 |
        | **$P(w_i)$** | **最终轮盘赌抽签概率** | 掷骰子选出下一个词的依据。 |
        
        ---
        
        ### 1. 什么是【KV-Cache】？—— “大脑里的工作记忆草稿纸”
        * **生活比喻**：你写作文写到第 100 个字时，不需要把前 99 个字从头到尾重新在脑子里造一遍句，只需要顺着脑海里已经想好的上文线索，直接往后蹦出第 100 个字！
        * **本质机理**：把前面所有 Token 的 Key 和 Value 矩阵保存在显存缓存中，推理速度直接从龟速的 $O(N^2)$ 飙升到极速的 $O(1)$。
        """
    )
