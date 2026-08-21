# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 9: Mini-GPT
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st
import time

from dashboard.styles.theme import apply_custom_theme, render_hero_header, render_page_guide
from dashboard.components.charts import plot_token_probabilities, plot_attention_heatmap_nlp

st.set_page_config(page_title="Mini-GPT", layout="wide")
apply_custom_theme()

render_hero_header(
    title="Mini-GPT 文本生成",
    subtitle="将积木拼成完全体：自回归逐词生成与采样",
    badge_text="MILESTONE 09 // AUTO-REGRESSIVE GEN",
    badge_type="brand-blue",
)

render_page_guide(
    title="M09 · Mini-GPT 文本生成 // Mini-GPT Text Generation",
    plain_intro="终于到了最激动人心的时刻！我们把所有零件——词嵌入、注意力机制、Transformer 积木——拼成一个完整的微型 GPT。虽然它的词汇量很小，但它能展示 ChatGPT 的核心工作原理：一个字一个字地'猜'下一个最可能的词！",
    hyperparams_desc="• <b>Temperature</b>：控制生成的创造力。越低越保守，越高越随机。<br>• <b>Top-k</b>：丢弃概率太小的词，防止生成无意义的乱码。<br>• <b>生成长度</b>：要让模型接续生成的词数。",
    telemetry_desc="• <b>文本实时显示</b>：打字机效果展示 GPT 逐词吐出的内容。<br>• <b>下一词概率分布</b>：模型心里的'候选项排行榜'。<br>• <b>T值效果对比</b>：直观理解 Temperature 是如何重塑概率分布的。",
    experiments=[
        "<b>第 1 步</b>：把 Temperature 调到 0.1，生成文本会非常确定，多次点击生成结果都一样。",
        "<b>第 2 步</b>：把 Temperature 调到 2.0，观察概率分布是否变得平缓，生成的句子是否开始跳脱。",
        "<b>第 3 步</b>：观察打字机生成时下方概率图的实时变化，体会自回归'Auto-Regressive'的含义。",
    ],
)

vocab = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", 
         "hello", "world", "AI", "is", "awesome", "today", "tomorrow", "good", "bad", "apple"]

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")
prompt = st.sidebar.selectbox("Prompt 输入", ["The quick brown", "AI is", "hello", "自定义..."])
if prompt == "自定义...":
    prompt = st.sidebar.text_input("自定义 Prompt", "The quick")

temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, step=0.1)
gen_len = st.sidebar.slider("生成长度", 1, 15, 5)
top_k = st.sidebar.slider("Top-k", 2, len(vocab), 5)
gen_button = st.sidebar.button("🎬 开始生成 // GENERATE")

# 文本生成展示区
st.markdown("#### 💬 生成过程")
out_container = st.empty()
prob_container = st.empty()
attn_container = st.empty()

def apply_temperature(probs, temp):
    logits = np.log(probs + 1e-9)
    scaled_logits = logits / temp
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

# 底部对比面板 (静态)
st.markdown("---")
st.markdown("#### 🌡️ Temperature 效果对比 (静态演示)")
col1, col2, col3 = st.columns(3)
base_probs = np.random.dirichlet(np.ones(len(vocab)) * 0.5)

with col1:
    fig1 = plot_token_probabilities(apply_temperature(base_probs, 0.1), vocab, top_k=5, title="T = 0.1 (绝对自信/保守)")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = plot_token_probabilities(apply_temperature(base_probs, 1.0), vocab, top_k=5, title="T = 1.0 (原始分布)")
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    fig3 = plot_token_probabilities(apply_temperature(base_probs, 2.0), vocab, top_k=5, title="T = 2.0 (混乱/高创造力)")
    st.plotly_chart(fig3, use_container_width=True)

if gen_button:
    current_text = prompt
    tokens_so_far = prompt.strip().split()
    
    for step in range(gen_len):
        out_container.markdown(f"> **{current_text}** `...`")
        
        # 模拟前向传播预测概率
        raw_probs = np.random.dirichlet(np.ones(len(vocab)))
        final_probs = apply_temperature(raw_probs, temperature)
        
        # top-k masking
        top_indices = np.argsort(final_probs)[-top_k:]
        masked_probs = np.zeros_like(final_probs)
        masked_probs[top_indices] = final_probs[top_indices]
        masked_probs = masked_probs / np.sum(masked_probs)
        
        # 绘图
        with prob_container:
            fig_prob = plot_token_probabilities(masked_probs, vocab, top_k=top_k, title=f"Step {step+1}: 预测分布")
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with attn_container:
            # 模拟最后一步的注意力
            curr_len = len(tokens_so_far)
            attn = np.random.rand(curr_len, curr_len)
            attn[np.triu_indices(curr_len, 1)] = 0
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
            fig_attn = plot_attention_heatmap_nlp(attn, tokens_so_far, tokens_so_far, title=f"Step {step+1}: 实时注意力")
            st.plotly_chart(fig_attn, use_container_width=True)
            
        # 采样 (为简单直接取最大或随机)
        # 按照分布采样
        next_word_idx = np.random.choice(len(vocab), p=masked_probs)
        next_word = vocab[next_word_idx]
        
        tokens_so_far.append(next_word)
        current_text += f" <span style='color:#1d4ed8; font-weight:bold;'>{next_word}</span>"
        
        out_container.markdown(f"> **{current_text}**", unsafe_allow_html=True)
        time.sleep(0.5)  # 打字机停顿
        
    st.balloons()
