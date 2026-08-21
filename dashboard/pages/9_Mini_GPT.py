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
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
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
    title="Mini-GPT 文本生成与自回归引擎",
    plain_intro=(
        "<b>终于到了见证奇迹的时刻——我们把前面的所有技术拼成了一个微型 ChatGPT！</b><br>"
        "GPT 的本质其实非常纯粹：<b>它永远只做一件事——猜下一个词 (Next-Token Prediction)</b>。<br>"
        "它给词表中每一个词打一个概率分（如'女王': 45%, '国王': 20%, '桌子': 0.1%）；<br>"
        "然后根据你设定的<b>温度 (Temperature)</b> 进行概率轮盘赌，抽中一个词拼到句子末尾，再重复这个过程！<br><br>"
        "<b>【2026 前沿拓展】：KV-Cache 推理加速</b><br>"
        "每次生成新词时，模型不需要把前面的所有词都重新计算一遍，而是将历史词汇的 Key/Value 缓存起来。<br>"
        "这使得 Transformer 的推理复杂度从 $O(N^2)$ 骤降为 $O(1)$，是实现流式实时响应的核心关键！"
    ),
    hyperparams_desc=(
        "• <b>提示词 (Prompt)</b>：给 GPT 的开头引导语（如 <code>the king and</code>）。<br>"
        "• <b>采样温度 (Temperature)</b>：控制创造力。T=0.1 确定保守，T=0.8 均衡灵动。<br>"
        "• <b>Top-k 过滤</b>：只在概率最高的前 K 个词中采样，过滤长尾词。<br>"
        "• <b>生成长度 (Max Tokens)</b>：让模型接续生成的词数。"
    ),
    telemetry_desc=(
        "• <b>实时生成文本流</b>：打字机效果逐词呈现 GPT 吐出的新词汇。<br>"
        "• <b>概率分布柱状图</b>：实时揭秘模型心中的'候选词排行榜'。<br>"
        "• <b>KV-Cache 显存开销遥测</b>：实时监控由于不断延长上下文而消耗的缓存显存池。"
    ),
    experiments=[
        "<b>第 1 步【体验确定性生成】</b>：把【Temperature】设为 <code>0.1</code>，点击生成，观察柱状图顶端出现尖锐的绝对优势词！",
        "<b>第 2 步【体验高创造力】</b>：把【Temperature】调到 <code>1.5</code>，观察概率柱状图变得平坦，模型给出随机出人意料的词汇组合！",
        "<b>第 3 步【观测显存暴涨】</b>：滚动到底部实验室，点击生成后，观察 KV-Cache Size 随着生成步数的增加呈现线性甚至几何级膨胀的现象！",
    ],
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

selected_prompt_key = st.sidebar.selectbox(
    "Prompt 预设",
    list(prompt_presets.keys()),
    index=0,
)

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
    btn_generate_all = st.button("RUN // 一键自回归", type="primary", use_container_width=True)
with col_btn2:
    btn_step_one = st.button("STEP // 单步推演", use_container_width=True)

# ---------------------------------------------------------------------------
# 会话状态管理 (Session State)
# ---------------------------------------------------------------------------
if "current_generated_tokens" not in st.session_state or st.session_state.get("last_prompt") != current_prompt_text:
    init_tokens = [w.lower().strip(",.!?") for w in current_prompt_text.strip().split() if w.strip()]
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
    tokens_so_far = [w.lower().strip(",.!?") for w in current_prompt_text.strip().split() if w.strip()]
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
current_probs, current_context_tokens = compute_next_token_distribution(current_tokens, temperature, top_k)

top_predicted_id = int(np.argmax(current_probs))
top_predicted_word = inv_vocab.get(top_predicted_id, "the")
top_predicted_prob = float(current_probs[top_predicted_id])

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
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
        delta="确定模式" if temperature < 0.4 else ("均衡创作" if temperature <= 1.0 else "发散混乱"),
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
render_section_heading("LIVE TEXT STREAM // 实时自回归文本流 (打字机效果)", icon_name="activity")
prompt_len = len(current_prompt_text.strip().split())
render_text_stream_box(current_tokens, prompt_len)

# ---------------------------------------------------------------------------
# 主视图区 2：下一词概率柱状图 & 实时注意力热力图
# ---------------------------------------------------------------------------
col_bar, col_attn = st.columns([1.2, 1])

with col_bar:
    render_section_heading(f"NEXT-TOKEN PROBABILITIES // 下一个词候选排行榜 (T={temperature})", icon_name="target")
    fig_probs = plot_token_probabilities(
        token_probs=current_probs,
        vocab=vocab_words,
        top_k=top_k,
        title=f"PROBABILITY DISTRIBUTION // 下一词概率柱状图 (Top {top_k})",
    )
    st.plotly_chart(fig_probs, use_container_width=True)

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
        st.plotly_chart(fig_realtime_attn, use_container_width=True)
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
        fig_t1 = plot_token_probabilities(toy_softmax(toy_logits, 0.1), toy_words, top_k=5, title="T=0.1 极度尖锐 (Greedy)")
        st.plotly_chart(fig_t1, use_container_width=True)

with col_t2:
    with st.container(border=True):
        st.markdown("#### [CREATIVE // 均衡创作] T = 0.7")
        st.caption("高分词依然占优，但赋予次高分词适度机会，展现生动多样的表达。")
        fig_t2 = plot_token_probabilities(toy_softmax(toy_logits, 0.7), toy_words, top_k=5, title="T=0.7 经典推荐 (Balanced)")
        st.plotly_chart(fig_t2, use_container_width=True)

with col_t3:
    with st.container(border=True):
        st.markdown("#### [CHAOTIC // 随机发散] T = 2.0")
        st.caption("差异被强行抹平，所有词概率趋同，模型极易产生乱码和幻觉。")
        fig_t3 = plot_token_probabilities(toy_softmax(toy_logits, 2.0), toy_words, top_k=5, title="T=2.0 极度平坦 (Uniform)")
        st.plotly_chart(fig_t3, use_container_width=True)

# ---------------------------------------------------------------------------
# 2026 前沿拓展：KV-Cache 自回归推理加速与显存占用实时监控
# ---------------------------------------------------------------------------
render_section_heading("2026 INFERENCE ACCELERATION // 自回归推理加速：KV-Cache 显存与算力优化", icon_name="zap")

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
        kv_tracker = KVCache(num_layers=2, max_batch_size=1, max_seq_len=64, num_kv_heads=2, head_dim=16)
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
