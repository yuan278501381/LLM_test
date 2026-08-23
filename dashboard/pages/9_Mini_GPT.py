# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 9: Mini-GPT 文本生成 (Auto-Regressive Text Generation) - 零基础入门保姆级教学平台

解剖现代大语言模型 (ChatGPT/GPT-4) 的自回归生成原理：Next-Token 概率预测、Temperature 温度采样、Top-k 过滤与实时注意力追踪。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib

import numpy as np
import streamlit as st

import dashboard.components.charts

importlib.reload(dashboard.components.charts)

from dashboard.components.charts import plot_attention_heatmap_nlp, plot_token_probabilities
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
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
from nn_core.embeddings import get_mini_vocab, get_synthetic_demo_embeddings
from nn_core.gpt import TinyGPT
from nn_core.paged_kv_cache import PagedKVCache

st.set_page_config(
    page_title="Mini-GPT · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M09", show_contract=True)
render_core_result_evidence("M09")

render_hero_header(
    title="Mini-GPT 文本生成与自回归采样",
    subtitle="未训练 TinyGPT 结构演示：自回归逐 Token 采样、概率分布、Temperature 锐度与注意力数值",
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
            "desc": "显示未训练模型的最高概率 token、温度与序列长度",
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
        f"<b>这是一个未经语料训练的 TinyGPT 计算和采样演示，不是微型 ChatGPT，输出不代表语言能力。</b><br>"
        f"GPT 的本质其实非常纯粹：<b>它永远只做一件事——猜下一个词 (Next-Token Prediction)</b>。<br>"
        f"它给词表中每一个词打一个概率分（如'女王': 45%, '国王': 20%, '桌子': 0.1%）；<br>"
        f"然后根据你在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 设定的<b>温度 (Temperature)</b> 进行概率轮盘赌，抽中一个词拼到 {anchor_badge('[D. 文本流]', 'purple', target_id='region-d')}，再重复这个过程！<br><br>"
        f"<b>【2026 前沿拓展】：KV-Cache 推理加速</b><br>"
        f"每次生成新词时，模型不需要把前面的所有词都重新计算一遍，而是将历史词汇的 Key/Value 缓存起来。<br>"
        f"对单个新 token，缓存避免重算历史 K/V；但它仍需与历史缓存做注意力，计算和内存会随上下文长度增长，不是整体 $O(1)$。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>提示词 (Prompt)</b>：给 GPT 的开头引导语（如 <code>the king and</code>）。<br>"
        f"• <b>采样温度 (Temperature)</b>：对 logits 缩放并改变分布锐度/随机性；不直接保证创造力、正确性或质量。<br>"
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
        "<b>第 2 步【提高随机性】</b>：把 Temperature 调到 <code>1.5</code>，观察同一 logits 分布变平。这只说明采样不确定性增加，不是创造力评估。",
        "<b>第 3 步【观测显存暴涨】</b>：滚动到底部实验室，点击生成后，观察 KV-Cache Size 随着生成步数的增加呈现线性甚至几何级膨胀的现象！",
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
    pretrained_w = get_synthetic_demo_embeddings(vocab_size, d_model=32)
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
    help="调节 Softmax 分布锐度。越小越集中于高 logit token，越大越平坦；不直接表示创造力或质量。",
)

top_k = st.sidebar.slider(
    "Top-k 截断采样",
    min_value=1,
    max_value=min(20, vocab_size),
    value=8,
    help="仅保留当前分布中概率最高的前 k 个 Token 并重新归一化。被截断 token 不一定无关。",
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
    st.session_state.gpt_rng = np.random.default_rng(42)

if "gpt_rng" not in st.session_state:
    st.session_state.gpt_rng = np.random.default_rng(42)


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
        sampled_id = st.session_state.gpt_rng.choice(vocab_size, p=probs)
        sampled_word = inv_vocab.get(sampled_id, "the")
        tokens_so_far.append(sampled_word)
    st.session_state.current_generated_tokens = tokens_so_far
    st.session_state.step_counter += gen_tokens_count

elif btn_step_one:
    tokens_so_far = list(st.session_state.current_generated_tokens)
    probs, _ = compute_next_token_distribution(tokens_so_far, temperature, top_k)
    sampled_id = st.session_state.gpt_rng.choice(vocab_size, p=probs)
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
        "TEMPERATURE // 分布锐度系数",
        f"T = {temperature:.1f}",
        delta="高度集中" if temperature < 0.4 else ("中等锐度" if temperature <= 1.0 else "较平坦"),
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
            * **[采样机制]**：本实现把 Temperature=0 定义为选取最高 logit（greedy）；Top-K 先截断到 K 个最高 logit，再按重归一化后的概率采样。
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
                * **最底下一行**：最新生成的单词正在对前面哪些上下文 token 分配内部注意力加权系数。
                * **高亮亮点**：表示当前头在该步对历史特定 token 的加权权重较高；注意力权重是模型内部特征组合系数，不等于严格的因果推断或特征重要性证明（Jain & Wallace, 2019）。
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


with col_t1, st.container(border=True):
    st.markdown("#### [LOW TEMPERATURE // 分布较尖锐] T = 0.1")
    st.caption("高分 token 的相对概率通常会增大；只要仍在采样，就不保证每次结果相同。")
    fig_t1 = plot_token_probabilities(
        toy_softmax(toy_logits, 0.1), toy_words, top_k=5, title="T=0.1 极度尖锐 (Greedy)"
    )
    st.plotly_chart(fig_t1, width="stretch")

with col_t2, st.container(border=True):
    st.markdown("#### [MID TEMPERATURE // 中等锐度] T = 0.7")
    st.caption("高分词依然占优，但赋予次高分词适度机会，展现生动多样的表达。")
    fig_t2 = plot_token_probabilities(
        toy_softmax(toy_logits, 0.7), toy_words, top_k=5, title="T=0.7 经典推荐 (Balanced)"
    )
    st.plotly_chart(fig_t2, width="stretch")

with col_t3, st.container(border=True):
    st.markdown("#### [HIGH TEMPERATURE // 分布较平坦] T = 2.0")
    st.caption("差异被强行抹平，所有词概率趋同，模型极易产生乱码和幻觉。")
    fig_t3 = plot_token_probabilities(
        toy_softmax(toy_logits, 2.0), toy_words, top_k=5, title="T=2.0 极度平坦 (Uniform)"
    )
    st.plotly_chart(fig_t3, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 采样温度 Temperature 概率塑形对比", expanded=False):
    st.markdown(
        """
        * **横向对比三张图**：
          * **T=0.1 (左图，分布较尖锐)**：最高 logit 的 token 往往占据更大概率；这不直接表示回答更严谨、正确或确定；
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
# 2026 前沿服务架构：PagedAttention 显存分页与前缀共享 (vLLM 核心算法)
# ---------------------------------------------------------------------------
render_section_heading(
    "PAGEDATTENTION // 现代推理服务显存分页与前缀共享 (vLLM 核心算法)", icon_name="layers"
)

with st.container(border=True):
    col_pa_ctrl, col_pa_view = st.columns([1.2, 2.0])

    with col_pa_ctrl:
        st.markdown("#### [PAGE TABLE CONTROLLER // 分页与并发控制台]")
        pa_block_size = st.selectbox(
            "物理页块大小 (Tokens/Block)", [4, 8, 16], index=0, key="pa_bs_sel"
        )
        pa_enable_prefix = st.checkbox(
            "启用前缀缓存共享 (Prefix Caching)", value=True, key="pa_prefix_chk"
        )
        pa_append_steps = st.slider(
            "并发新生成 Token 步数", min_value=1, max_value=8, value=3, key="pa_gen_steps"
        )

        # 模拟运行 PagedKVCache
        paged_mgr = PagedKVCache(
            total_blocks=16, block_size=pa_block_size, num_kv_heads=2, head_dim=16
        )

        # 请求 A: 初始 6 个 token
        prompt_a_k = np.random.randn(2, 6, 16)
        prompt_a_v = np.random.randn(2, 6, 16)
        paged_mgr.allocate_sequence("Req-A (用户1)", prompt_a_k, prompt_a_v)

        if pa_enable_prefix:
            # 请求 B 共享请求 A 的前缀 (Prefix Caching)
            paged_mgr.fork_sequence_prefix("Req-A (用户1)", "Req-B (用户2 共享前缀)")
        else:
            # 独立分配
            paged_mgr.allocate_sequence(
                "Req-B (用户2 独立前缀)", prompt_a_k.copy(), prompt_a_v.copy()
            )

        # 模拟生成追加 token
        cow_count = 0
        for _ in range(pa_append_steps):
            tok_k = np.random.randn(2, 16)
            tok_v = np.random.randn(2, 16)
            _, cow_a = paged_mgr.append_token("Req-A (用户1)", tok_k, tok_v)
            target_b = "Req-B (用户2 共享前缀)" if pa_enable_prefix else "Req-B (用户2 独立前缀)"
            _, cow_b = paged_mgr.append_token(target_b, tok_k, tok_v)
            if cow_a:
                cow_count += 1
            if cow_b:
                cow_count += 1

        p_stats = paged_mgr.get_memory_stats()

        st.markdown(
            f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:0.8rem;margin-top:0.6rem;">
                <div style="font-size:0.75rem;font-weight:700;color:#64748b;">显存利用率对比</div>
                <div style="font-size:1.3rem;font-weight:800;color:#047857;margin:0.2rem 0;">
                    {p_stats["internal_fragmentation_rate"] * 100:.1f}% <span style="font-size:0.85rem;color:#475569;font-weight:500;">内部碎片率</span>
                </div>
                <div style="font-size:0.8rem;color:#475569;">
                    物理块总数: <b>16 Blocks</b> (空闲 <b>{p_stats["free_blocks"]}</b>)<br>
                    逻辑 Token 总量: <b>{p_stats["total_logical_tokens"]} Tokens</b><br>
                    物理存储 Token: <b>{p_stats["physical_stored_tokens"]} Tokens</b><br>
                    前缀共享节省: <b style="color:#1d4ed8;">{p_stats["shared_saved_tokens"]} Tokens</b><br>
                    传统预分配浪费: <b style="color:#be123c;">{p_stats["traditional_prealloc_waste_rate"] * 100:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pa_view:
        st.markdown("#### [VIRTUAL BLOCK TABLE MAPPING // 虚拟页表与显存池映射]")

        # 渲染虚拟页表映射
        table_rows = []
        for req_id, b_ids in paged_mgr.block_tables.items():
            blocks_badges = " ".join(
                [
                    f'<span style="background:#dbeafe;color:#1d4ed8;font-weight:700;padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.75rem;">Block #{bid} (ref={paged_mgr.physical_blocks[bid].ref_count})</span>'
                    for bid in b_ids
                ]
            )
            table_rows.append(
                f"<tr>"
                f'<td style="padding:6px 8px;font-weight:700;font-size:0.82rem;border-bottom:1px solid #f1f5f9;">{req_id}</td>'
                f'<td style="padding:6px 8px;font-size:0.82rem;border-bottom:1px solid #f1f5f9;">{blocks_badges}</td>'
                f'<td style="padding:6px 8px;font-size:0.82rem;text-align:right;font-weight:700;color:#0f172a;border-bottom:1px solid #f1f5f9;">{paged_mgr.seq_lengths[req_id]} Tokens</td>'
                f"</tr>"
            )

        st.markdown(
            f"""
            <table style="width:100%;border-collapse:collapse;margin-bottom:0.8rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;">
                <thead style="background:#f8fafc;font-size:0.75rem;color:#64748b;font-weight:700;text-transform:uppercase;">
                    <tr><th style="padding:6px 8px;text-align:left;">请求 ID</th><th style="padding:6px 8px;text-align:left;">映射物理页表 (Block IDs)</th><th style="padding:6px 8px;text-align:right;">序列长度</th></tr>
                </thead>
                <tbody>{"".join(table_rows)}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "核心机理：传统大模型推理会为每个请求按最长上下文（如 2048）一次性连续预分配显存，导致大量显存闲置浪费（碎片率 60%~80%）。"
            "PagedAttention 引入操作系统的虚拟内存分页思想，将 Key/Value 缓存切分为不连续的定长物理块，"
            "按需动态分配并通过页表索引，同时支持多个请求零拷贝共享 System Prompt 前缀，极大提升高并发吞吐。"
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
        | **$T$** | **采样温度系数 (Temperature)** | 用 $z/T$ 缩放 logits：$0<T<1$ 通常使分布更尖，$T>1$ 更平。它不保证逻辑、事实性、创造力或质量。 |
        | **$\\text{Top-}K$** | **截断候选词池** | 保留 logit/概率最高的 $K$ 项，其余设为零后重新归一化。它不保证被删除项无关，也不保证保留项正确。 |
        | **$P(w_i)$** | **最终轮盘赌抽签概率** | 掷骰子选出下一个词的依据。 |

        ---

        ### 1. 什么是【KV-Cache】？—— “自回归生成的工作记忆缓存”
        * **生活比喻**：写作文写到第 100 个字时，不需要把前 99 个字从头重新逐字构思，只需要顺着脑海里已经保留好的上文线索，直接生成第 100 个字！
        * **本质机理**：将历史 Token 已经计算过的 Key 和 Value 向量缓存在显存中，避免每一步重新对整个历史上下文进行重复投影；新 Token 仅需单步投影并与历史 $t$ 个缓存向量计算注意力（单步注意力计算随历史长度 $t$ 线性增长 $O(t)$，生成 $N$ 个 token 的总计算量由无缓存的 $O(N^3)$ 降至 $O(N^2)$）。
        """
    )
