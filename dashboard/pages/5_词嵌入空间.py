# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 5: 词嵌入语义空间
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import streamlit as st
import pandas as pd

from dashboard.styles.theme import apply_custom_theme, render_hero_header, render_page_guide, render_metric_card
from dashboard.components.charts import plot_embedding_space

try:
    from nn_core.embeddings import get_mini_vocab, get_pretrained_embeddings
except ImportError:
    # 模拟数据以防 nn_core 还没更新
    def get_mini_vocab():
        return ["国王", "女王", "男人", "女人", "猫", "狗", "桌子", "苹果", "香蕉", "北京", "中国", "巴黎", "法国"]
    def get_pretrained_embeddings():
        return np.random.randn(len(get_mini_vocab()), 16)

st.set_page_config(page_title="词嵌入空间", layout="wide")
apply_custom_theme()

render_hero_header(
    title="词嵌入空间",
    subtitle="探索高维向量中的语义距离与线性算术",
    badge_text="MILESTONE 05 // WORD EMBEDDING",
    badge_type="blue",
)

render_page_guide(
    title="M05 · 词嵌入语义空间 // Word Embedding Space",
    plain_intro="词嵌入就像给每个词配一个'性格坐标'。比如'猫'和'狗'的坐标很近（都是宠物），但'猫'和'桌子'的坐标就很远。更神奇的是，这些坐标还能做算术：国王-男人+女人≈女王！",
    hyperparams_desc="• <b>可视化维度选择</b>：在 2D 或 3D 空间中观察词汇簇。<br>• <b>高亮词组选择</b>：指定特定语义群体以观察其聚集效应。<br>• <b>语义算术输入</b>：自定义 A - B + C 以检验线性关系。",
    telemetry_desc="• <b>上：3D散点图</b>：词汇的高维空间降维映射。<br>• <b>下左：余弦相似度表格</b>：找出距离算术结果最近的Top词。<br>• <b>下右：语义算术结果</b>：验证经典语义代数方程。",
    experiments=[
        "<b>第 1 步</b>：选择'王族'高亮组，观察他们在空间中的聚集。",
        "<b>第 2 步</b>：尝试'国王'-'男人'+'女人'的算术，看看最近的词是不是'女王'。",
        "<b>第 3 步</b>：比较'One-Hot'和'Embedding'的区别，理解密集向量的优势。",
    ],
)

vocab = get_mini_vocab()
embeddings = get_pretrained_embeddings()

st.sidebar.markdown("#### HYPERPARAMETERS // 超参数")
dim_choice = st.sidebar.radio("可视化维度选择", ["3D", "2D"], index=0)
group_choice = st.sidebar.selectbox("预设语义组选择", ["全部", "王族", "动物", "国家城市"])

st.sidebar.markdown("#### 语义算术输入: A - B + C = ?")
word_A = st.sidebar.selectbox("词 A (如: 国王)", vocab, index=vocab.index("国王") if "国王" in vocab else 0)
word_B = st.sidebar.selectbox("词 B (如: 男人)", vocab, index=vocab.index("男人") if "男人" in vocab else 0)
word_C = st.sidebar.selectbox("词 C (如: 女人)", vocab, index=vocab.index("女人") if "女人" in vocab else 0)

# 计算算术结果
idx_A = vocab.index(word_A)
idx_B = vocab.index(word_B)
idx_C = vocab.index(word_C)

vec_A = embeddings[idx_A]
vec_B = embeddings[idx_B]
vec_C = embeddings[idx_C]
vec_R_ideal = vec_A - vec_B + vec_C

# 计算余弦相似度
def cosine_sim(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return np.dot(v1, v2) / (n1 * n2)

sims = [cosine_sim(vec_R_ideal, emb) for emb in embeddings]
top_indices = np.argsort(sims)[::-1]
# 排除输入的A, B, C，找到最接近的结果
result_word = ""
top_5_records = []
count = 0
for i in top_indices:
    word = vocab[i]
    if word not in [word_A, word_B, word_C] and count == 0:
        result_word = word
    if count < 5:
        top_5_records.append({"Word": word, "Cosine Similarity": f"{sims[i]:.4f}"})
        count += 1

highlight_words = []
if group_choice == "王族":
    highlight_words = [w for w in ["国王", "女王", "王子", "公主", "皇帝"] if w in vocab]
elif group_choice == "动物":
    highlight_words = [w for w in ["猫", "狗", "老虎", "狮子", "老鼠"] if w in vocab]
elif group_choice == "国家城市":
    highlight_words = [w for w in ["中国", "北京", "法国", "巴黎", "日本", "东京"] if w in vocab]
elif group_choice == "全部":
    highlight_words = vocab

arithmetic = {"A": word_A, "B": word_B, "C": word_C, "Result": result_word}

# 主区域
# 上半部：3D/2D 散点图
target_dim = 2 if dim_choice == "2D" else 3
if target_dim == 2 and embeddings.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    vectors_proj = pca.fit_transform(embeddings)
else:
    vectors_proj = embeddings

fig_space = plot_embedding_space(vocab, vectors_proj, highlight_words=highlight_words, arithmetic=arithmetic)
st.plotly_chart(fig_space, use_container_width=True)

# 下半部
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 余弦相似度 Top-5 (相对于理想结果)")
    st.table(pd.DataFrame(top_5_records))

with col2:
    st.markdown("#### 语义算术结果")
    res_html = (
        '<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:1.5rem; text-align:center;">'
        f'<span style="font-size:1.2rem; font-weight:bold; color:#1d4ed8;">{word_A}</span> '
        f'<span style="font-size:1.2rem; font-weight:bold;">-</span> '
        f'<span style="font-size:1.2rem; font-weight:bold; color:#b45309;">{word_B}</span> '
        f'<span style="font-size:1.2rem; font-weight:bold;">+</span> '
        f'<span style="font-size:1.2rem; font-weight:bold; color:#047857;">{word_C}</span> '
        f'<span style="font-size:1.5rem; font-weight:bold; margin:0 1rem;">≈</span> '
        f'<span style="font-size:1.5rem; font-weight:bold; color:#be123c;">{result_word}</span>'
        '</div>'
    )
    st.markdown(res_html, unsafe_allow_html=True)

# 最底部：One-Hot vs Embedding 对比卡片
st.markdown("---")
st.markdown("#### ⚠️ One-Hot vs Embedding (稀疏 vs 密集)")
col_o, col_e = st.columns(2)
with col_o:
    st.info("**One-Hot 编码**\n\n向量长度等于词表大小。只有一个1，其余全是0。词与词之间绝对正交，没有距离和相似度概念 (余弦相似度全为0)。")
with col_e:
    st.success("**Word Embedding 词嵌入**\n\n低维密集实数向量。语义相似的词在空间中距离相近，能通过线性组合表达复杂的概念关联。")
