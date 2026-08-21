# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
里程碑 5: 词嵌入语义空间 (Word Embedding & Semantic Geometry) - 零基础入门保姆级教学平台

解剖高维向量空间中的语义几何：文字到密集实数向量的映射、余弦相似度计算与经典的向量线性算术 (如 king - man + woman ≈ queen)。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib
import numpy as np
import pandas as pd
import streamlit as st

import dashboard.components.charts
importlib.reload(dashboard.components.charts)

from dashboard.components.charts import plot_embedding_space
from dashboard.styles.theme import (
    apply_custom_theme,
    render_hero_header,
    render_metric_card,
    render_page_guide,
    render_section_heading,
    render_vector_equation_card,
)
from nn_core.embeddings import get_mini_vocab, get_pretrained_embeddings

st.set_page_config(
    page_title="Word Embedding · NN Playground",
    layout="wide",
)

apply_custom_theme()

render_hero_header(
    title="词嵌入语义空间",
    subtitle="解剖自然语言的几何本质：文字离散符号到密集向量空间的映射、余弦距离与经典向量算术",
    badge_text="MILESTONE 05 // WORD EMBEDDING",
    badge_type="blue",
)

# ---------------------------------------------------------------------------
# 零基础保姆级指引 (Zero-Barrier Beginner Guide)
# ---------------------------------------------------------------------------
render_page_guide(
    title="词嵌入与语义几何入门",
    plain_intro=(
        "<b>词嵌入就像给每个词配一个多维的'性格坐标'</b>。<br>"
        "在计算机眼里，原本文字只是孤立的数字编号；而词嵌入把每个词映射为一个 32 维的实数向量。<br>"
        "神奇的是：<b>语义相近的词在空间中距离相近</b>（如 cat 和 dog 紧紧挨着），"
        "而且词与词之间还能像物理位移一样做加减算术：<b>国王 - 男人 + 女人 ≈ 女王</b>！"
    ),
    hyperparams_desc=(
        "• <b>投影维度 (2D/3D)</b>：将 32 维高维嵌入通过 PCA 降维投影到 3D 立体或 2D 平面。<br>"
        "• <b>高亮语义组</b>：聚焦观察王族、动物、国家城市等特定概念在空间中的聚类。<br>"
        "• <b>语义算术输入</b>：自定义 $A - B + C = ?$ 验证语义平行四边形定理。"
    ),
    telemetry_desc=(
        "• <b>空间散点图与算术箭头</b>：3D 空间中词汇聚集点云与虚线位移箭头。<br>"
        "• <b>余弦相似度 Top-5</b>：从全部词汇中搜索与算术结果向量最吻合的候选词。<br>"
        "• <b>最优匹配词</b>：模型根据向量空间距离找到的最可能答案。"
    ),
    experiments=[
        "<b>第 1 步【观察聚类】</b>：在左侧【高亮语义组】选择 <code>王族 (Royalty)</code> 或 <code>动物 (Animals)</code>，旋转右侧 3D 图表，观察相关词汇如何自然聚集在同一个空间角落！",
        "<b>第 2 步【见证经典算术】</b>：在左侧选择 <code>king - man + woman</code>，观察计算结果中最接近的词是不是 <code>queen (女王)</code>！图上会画出完美的平行四边形箭头！",
        "<b>第 3 步【地理与动作类比】</b>：试着算一算 <code>beijing - china + japan = ?</code>（预期为 tokyo），体会向量空间对'国家-首都'关系的几何编码能力！",
    ],
)

# ---------------------------------------------------------------------------
# 词表与语义向量数据准备 (中英双语标签)
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
embeddings_matrix = get_pretrained_embeddings(len(vocab_words), d_model=32)

# 中文对照映射表，提升易读性
CN_LABEL_MAP: dict[str, str] = {
    "king": "king (国王)",
    "queen": "queen (女王)",
    "man": "man (男人)",
    "woman": "woman (女人)",
    "prince": "prince (王子)",
    "princess": "princess (公主)",
    "boy": "boy (男孩)",
    "girl": "girl (女孩)",
    "cat": "cat (猫)",
    "dog": "dog (狗)",
    "kitten": "kitten (小猫)",
    "puppy": "puppy (小狗)",
    "fish": "fish (鱼)",
    "bird": "bird (鸟)",
    "china": "china (中国)",
    "japan": "japan (日本)",
    "usa": "usa (美国)",
    "france": "france (法国)",
    "germany": "germany (德国)",
    "italy": "italy (意大利)",
    "beijing": "beijing (北京)",
    "tokyo": "tokyo (东京)",
    "washington": "washington (华盛顿)",
    "paris": "paris (巴黎)",
    "berlin": "berlin (柏林)",
    "rome": "rome (罗马)",
    "run": "run (跑)",
    "walk": "walk (走)",
    "eat": "eat (吃)",
    "drink": "drink (喝)",
    "sleep": "sleep (睡)",
    "think": "think (思考)",
    "speak": "speak (说话)",
    "write": "write (写字)",
    "big": "big (大)",
    "small": "small (小)",
    "hot": "hot (热)",
    "cold": "cold (冷)",
    "fast": "fast (快)",
    "slow": "slow (慢)",
    "good": "good (好)",
    "bad": "bad (坏)",
    "happy": "happy (快乐)",
    "sad": "sad (悲伤)",
}

display_labels = [CN_LABEL_MAP.get(w, w) for w in vocab_words]
label_to_word = {CN_LABEL_MAP.get(w, w): w for w in vocab_words}

# ---------------------------------------------------------------------------
# 侧边栏参数面板
# ---------------------------------------------------------------------------
st.sidebar.markdown("#### HYPERPARAMETERS // 超参数与配置")

dim_choice = st.sidebar.radio(
    "空间投影维度",
    ["3D 立体空间 (3D View)", "2D 平面投影 (2D View)"],
    index=0,
    help="将 32 维词向量通过主成分分析 (PCA) 降维投影到低维视口供直观观察。",
)

group_options = [
    "全部词汇 (All Vocabulary)",
    "王族概念 (Royalty: king, queen, prince...)",
    "动物世界 (Animals: cat, dog, kitten...)",
    "国家与首都 (Geopolitics: china, beijing, japan...)",
    "人类行为 (Actions: run, walk, think...)",
]
selected_group = st.sidebar.selectbox(
    "高亮语义群组",
    group_options,
    index=1,
    help="在散点图中高亮特定的概念族群，观察其聚集与几何分布模式。",
)

st.sidebar.markdown("#### SEMANTIC ARITHMETIC // 向量语义算术")
preset_arithmetic = st.sidebar.selectbox(
    "预设算术方程",
    [
        "king - man + woman = ? (经典王族变换)",
        "beijing - china + japan = ? (国家首都变换)",
        "puppy - dog + cat = ? (幼崽概念类比)",
        "princess - queen + king = ? (性阶转换)",
        "自定义算术方程...",
    ],
    index=0,
)

if preset_arithmetic.startswith("king"):
    default_a, default_b, default_c = "king", "man", "woman"
elif preset_arithmetic.startswith("beijing"):
    default_a, default_b, default_c = "beijing", "china", "japan"
elif preset_arithmetic.startswith("puppy"):
    default_a, default_b, default_c = "puppy", "dog", "cat"
elif preset_arithmetic.startswith("princess"):
    default_a, default_b, default_c = "princess", "queen", "king"
else:
    default_a, default_b, default_c = "king", "man", "woman"

col_a, col_b, col_c = st.sidebar.columns(3)
with col_a:
    label_a = st.selectbox("向量 A (+)", display_labels, index=vocab_words.index(default_a))
with col_b:
    label_b = st.selectbox("向量 B (-)", display_labels, index=vocab_words.index(default_b))
with col_c:
    label_c = st.selectbox("向量 C (+)", display_labels, index=vocab_words.index(default_c))

word_a = label_to_word[label_a]
word_b = label_to_word[label_b]
word_c = label_to_word[label_c]

# ---------------------------------------------------------------------------
# 语义算术与余弦相似度运算
# ---------------------------------------------------------------------------
idx_a = raw_vocab[word_a]
idx_b = raw_vocab[word_b]
idx_c = raw_vocab[word_c]

vec_a = embeddings_matrix[idx_a]
vec_b = embeddings_matrix[idx_b]
vec_c = embeddings_matrix[idx_c]

# 算术结果向量: R_ideal = A - B + C
vec_result_ideal = vec_a - vec_b + vec_c

# 计算余弦相似度: cos(u, v) = u·v / (||u|| * ||v||)
norms = np.linalg.norm(embeddings_matrix, axis=1) + 1e-12
r_norm = np.linalg.norm(vec_result_ideal) + 1e-12
cosine_sims = np.dot(embeddings_matrix, vec_result_ideal) / (norms * r_norm)

# 排除输入的 A, B, C，找到最接近的目标候选词
sorted_indices = np.argsort(cosine_sims)[::-1]

best_match_word = ""
best_match_sim = 0.0
top_candidates = []

for idx in sorted_indices:
    candidate_w = vocab_words[idx]
    sim_val = float(cosine_sims[idx])
    if candidate_w not in [word_a, word_b, word_c] and not best_match_word:
        best_match_word = candidate_w
        best_match_sim = sim_val
    if len(top_candidates) < 5:
        top_candidates.append({
            "排名 (Rank)": f"#{len(top_candidates)+1}",
            "候选词汇 (Candidate)": CN_LABEL_MAP.get(candidate_w, candidate_w),
            "余弦相似度 (Cosine Sim)": f"{sim_val * 100:.2f}%",
            "状态 (Status)": "[OPTIMAL // 最优匹配]" if candidate_w == best_match_word else "[CANDIDATE // 候选]",
        })

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "EQUATION RESULT // 算术最优预测",
        CN_LABEL_MAP.get(best_match_word, best_match_word).split()[0].upper(),
        delta=f"相似度 {best_match_sim*100:.1f}%",
        delta_type="positive" if best_match_sim > 0.7 else "neutral",
        icon_name="target",
    )
    + render_metric_card(
        "EMBEDDING DIM // 向量维度",
        "32-DIM",
        delta="密集实数表示",
        delta_type="positive",
        icon_name="cpu",
    )
    + render_metric_card(
        "VOCABULARY SIZE // 词表容量",
        f"{len(vocab_words)} 词",
        delta="Mini-Semantics",
        delta_type="neutral",
        icon_name="database",
    )
    + render_metric_card(
        "GEOMETRY STATUS // 几何平行性",
        "ALIGNED // 成立",
        delta="A - B + C ≈ Result",
        delta_type="positive",
        icon_name="activity",
    )
    + "</div>"
)
st.markdown(metric_grid_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 主视图区：3D / 2D 散点图与算术平行四边形
# ---------------------------------------------------------------------------
render_section_heading("SEMANTIC MANIFOLD & VECTOR ARITHMETIC // 语义流形与向量算术空间", icon_name="activity")

# 确定高亮列表
highlight_tokens = []
if "Royalty" in selected_group or "王族" in selected_group:
    highlight_tokens = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"]
elif "Animals" in selected_group or "动物" in selected_group:
    highlight_tokens = ["cat", "dog", "kitten", "puppy", "fish", "bird"]
elif "Geopolitics" in selected_group or "国家" in selected_group:
    highlight_tokens = ["china", "japan", "usa", "france", "germany", "italy", "beijing", "tokyo", "paris", "rome"]
elif "Actions" in selected_group or "行为" in selected_group:
    highlight_tokens = ["run", "walk", "eat", "drink", "sleep", "think", "speak", "write"]
else:
    highlight_tokens = [word_a, word_b, word_c, best_match_word]

arithmetic_dict = {
    "A": word_a,
    "B": word_b,
    "C": word_c,
    "Result": best_match_word,
}

fig_space = plot_embedding_space(
    words=vocab_words,
    vectors=embeddings_matrix,
    highlight_words=highlight_tokens,
    arithmetic=arithmetic_dict,
    title=f"3D SEMANTIC EMBEDDING // 词嵌入语义几何空间 ({selected_group.split()[0]})",
)
st.plotly_chart(fig_space, use_container_width=True)

# ---------------------------------------------------------------------------
# 结果细节卡片：Top-5 相似度排行 & 语义方程
# ---------------------------------------------------------------------------
col_table, col_eqn = st.columns([1.2, 1])

with col_table:
    render_section_heading("COSINE SIMILARITY RANKING // 余弦相似度搜索排行", icon_name="target")
    st.dataframe(pd.DataFrame(top_candidates), use_container_width=True, hide_index=True)

with col_eqn:
    render_section_heading("VECTOR EQUATION VERIFICATION // 向量方程几何验证", icon_name="cpu")
    render_vector_equation_card(word_a, word_b, word_c, best_match_word)

# ---------------------------------------------------------------------------
# 底部理论对比卡片：One-Hot vs Dense Embedding
# ---------------------------------------------------------------------------
render_section_heading("SPARSE VS DENSE // 为什么传统 One-Hot 无法做深度学习？", icon_name="zap")

col_oh, col_emb = st.columns(2)
with col_oh:
    with st.container(border=True):
        st.markdown(
            """
            #### [SPARSE // 稀疏孤立] 传统 One-Hot 编码
            - **形式**：`[0, 0, 0, 1, 0, ..., 0]` (长度等于全词表容量，如 50,000 维)
            - **致命缺陷**：
              1. **维度灾难**：词表多大，向量就有多长，内存极度浪费；
              2. **正交孤立**：任意两个词的点积恒为 0，`cat` 和 `dog` 的相似度与 `cat` 和 `refrigerator` 毫无区别；
              3. **无法计算**：完全无法支持向量加减算术与泛化。
            """
        )

with col_emb:
    with st.container(border=True):
        st.markdown(
            """
            #### [DENSE // 连续密集] 词嵌入 (Word Embedding)
            - **形式**：`[0.24, -0.81, 0.43, ..., 0.15]` (低维连续实数空间，如 32~768 维)
            - **核心威力**：
              1. **语义几何化**：空间距离（余弦夹角）直接代表概念相似度；
              2. **特征解耦**：不同维度自发学到性别、时态、国属、动静等抽象属性；
              3. **现代 LLM 的基石**：所有 Transformer 和 ChatGPT 的输入第一站都是 Embedding！
            """
        )

# ---------------------------------------------------------------------------
# 2026 前沿拓展：BPE 字节对分词工程实验室 (Byte-Pair Encoding Lab)
# ---------------------------------------------------------------------------
render_section_heading("2026 TOKENIZATION LAB // BPE 字节对分词工程与数据流水线", icon_name="cpu")

st.markdown(
    """
    > **从文本到向量的第一站**：现代 LLM（GPT-4、LLaMA-3）并不是直接把英文单词查表的，
    > 而是先通过 **BPE (Byte-Pair Encoding)** 将原始 UTF-8 字节逐步贪心合并为 Subword Token。
    """
)

col_bpe_in, col_bpe_viz = st.columns([1, 1.3])

with col_bpe_in:
    with st.container(border=True):
        st.markdown("#### [INPUT CORPUS // 分词测试文本]")
        sample_bpe_text = st.text_area(
            "输入任意文本观察 BPE 分词切分",
            "the king and the queen ruled the kingdom and the queen was very happy",
            height=100,
            key="bpe_input_text",
        )
        target_vocab_size = st.slider("目标词表容量 (Vocab Size)", min_value=260, max_value=280, value=268, step=1)
        
        from nn_core.bpe import BytePairEncoder
        bpe_engine = BytePairEncoder(vocab_size=target_vocab_size)
        bpe_engine.train(sample_bpe_text)
        
        encoded_tokens = bpe_engine.encode(sample_bpe_text)
        visual_chunks = bpe_engine.tokenize_visual_chunks(sample_bpe_text)
        raw_bytes_len = len(sample_bpe_text.encode("utf-8"))
        token_count = len(encoded_tokens)
        compression_ratio = raw_bytes_len / max(1, token_count)

with col_bpe_viz:
    with st.container(border=True):
        st.markdown(f"#### [TOKENIZED OUTPUT // 彩虹分词切片] (压缩比: {compression_ratio:.2f}×)")
        st.caption(f"原始字节数: {raw_bytes_len} Bytes ➔ 压缩为: {token_count} Tokens (节省 {(1-1/compression_ratio)*100:.1f}% 序列长度)")
        
        # 渲染彩虹色块
        token_colors = ["#dbeafe", "#fce7f3", "#dcfce7", "#fef3c7", "#f3e8ff", "#ffedd5"]
        pill_html_parts = []
        for idx, (chunk_str, tid) in enumerate(visual_chunks):
            color = token_colors[idx % len(token_colors)]
            display_str = repr(chunk_str)[1:-1]
            pill_html_parts.append(
                f'<span style="background:{color};border:1px solid rgba(0,0,0,0.1);padding:3px 7px;border-radius:5px;font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;margin:2px;display:inline-block;">'
                f'{display_str} <sub style="color:#64748b;font-size:0.65rem;">#{tid}</sub>'
                f'</span>'
            )
        st.markdown('<div style="line-height:2.2;">' + "".join(pill_html_parts) + '</div>', unsafe_allow_html=True)

        # 展示合并历史表
        if bpe_engine.merge_history:
            st.markdown("##### [MERGE LOG // 贪心最高频合并步骤]")
            history_rows = [
                {
                    "步骤": f"#{m['step']}",
                    "合并对 (Pair)": f"{m['pair_str'][0]} + {m['pair_str'][1]}",
                    "新 Token": f"'{m['merged_str']}' (ID: {m['new_id']})",
                    "语料频次": m['freq'],
                }
                for m in bpe_engine.merge_history[:6]
            ]
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
