# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 6: 序列记忆与遗忘瓶颈
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st

from dashboard.styles.theme import apply_custom_theme, render_hero_header, render_page_guide
from dashboard.components.charts import plot_memory_decay_heatmap

try:
    from nn_core.embeddings import Embedding
    from nn_core.rnn import RNNCell
except ImportError:
    class Embedding:
        def __init__(self, d_model): self.d_model = d_model
        def forward(self, token_id): return np.random.randn(self.d_model)
    class RNNCell:
        def __init__(self, input_dim, hidden_dim): self.hidden_dim = hidden_dim
        def forward(self, x, h_prev): return np.tanh(np.random.randn(self.hidden_dim) + h_prev * 0.5)

st.set_page_config(page_title="序列记忆", layout="wide")
apply_custom_theme()

render_hero_header(
    title="序列记忆与遗忘瓶颈",
    subtitle="RNN的状态传递与长期记忆衰减挑战",
    badge_text="MILESTONE 06 // SEQUENCE & FORGETTING",
    badge_type="purple",
)

render_page_guide(
    title="M06 · 序列记忆与遗忘瓶颈 // Sequence Memory & Forgetting",
    plain_intro="想象你正在听一段很长的故事。RNN就像一个记性不太好的听众——它能记住刚说过的几个词，但随着故事越来越长，开头说了什么就渐渐模糊了。这就是'遗忘瓶颈'！",
    hyperparams_desc="• <b>输入句子</b>：提供一段文字来测试 RNN 的记忆能力。<br>• <b>隐藏层维度</b>：增加维度可能略微缓解遗忘，但无法根治长距离依赖问题。",
    telemetry_desc="• <b>上：状态传递动画</b>：模拟 RNN 顺次处理每一个词。<br>• <b>中：记忆衰减热力图</b>：当前词的隐状态对之前词的保留强度，颜色越浅遗忘越严重。<br>• <b>下：对比卡片</b>：引出注意力机制的必要性。",
    experiments=[
        "<b>第 1 步</b>：输入一个长句子，观察右侧热力图右上角是否变白（遗忘）。",
        "<b>第 2 步</b>：调整隐藏层维度，观察衰减速度的微小变化。",
        "<b>第 3 步</b>：思考如何才能不被前面的内容遮蔽？(答案是：直接回头看所有内容)",
    ],
)

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")
text_input = st.sidebar.text_input("输入句子", "The cat sat on the mat and then the dog came and chased the cat away")
hidden_dim = st.sidebar.selectbox("隐藏层维度", [8, 16, 32], index=1)

# 处理句子
tokens = text_input.strip().split()
if not tokens:
    tokens = ["Empty"]

emb_dim = 8
emb = Embedding(d_model=emb_dim)
rnn = RNNCell(input_dim=emb_dim, hidden_dim=hidden_dim)

# 逐步处理
hidden_states = []
h_prev = np.zeros(hidden_dim)
for idx, token in enumerate(tokens):
    x = emb.forward(idx)  # dummy token id
    h_prev = rnn.forward(x, h_prev)
    hidden_states.append(h_prev.copy())

# 主区域
# 上半部：隐藏状态传递示意 (简化用 columns)
st.markdown("#### 隐藏状态顺次传递过程")
cols = st.columns(len(tokens))
for i, (col, token) in enumerate(zip(cols, tokens)):
    with col:
        # 用颜色深浅表示一点激活状态 (模拟)
        intensity = np.linalg.norm(hidden_states[i])
        bg_color = f"rgba(29, 78, 216, {min(1.0, 0.2 + intensity*0.1)})"
        html = f"""<div style="background:{bg_color}; padding:10px; border-radius:4px; text-align:center; color:#fff; font-size:12px; height:80px; display:flex; align-items:center; justify-content:center; flex-direction:column;">
        <b>{token}</b><br>t={i}</div>"""
        st.markdown(html, unsafe_allow_html=True)
        if i < len(tokens) - 1:
            st.markdown("<div style='text-align:center;'>⬇️</div>", unsafe_allow_html=True)

# 中部：记忆衰减热力图
st.markdown("---")
fig_decay = plot_memory_decay_heatmap(hidden_states, tokens)
st.plotly_chart(fig_decay, use_container_width=True)

# 底部：对比标注卡片
st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    st.error("#### ⚠️ RNN 的致命缺陷\n\n序列越长，早期信息的特征在隐藏状态向量中被稀释得越严重。所有的历史信息都被强行压缩到一个固定大小的向量中，成为**遗忘瓶颈**。")
with col_r:
    st.success("#### 💡 这就是为什么我们需要注意力机制\n\n如果每个词在处理时，都能直接查阅整个序列中所有的历史词汇，不受距离限制，那么遗忘瓶颈将被彻底打破。下一课，我们将引入 Attention！")
