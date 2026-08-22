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
from dashboard.components.pedagogy import render_core_result_evidence, render_lesson_evidence
from dashboard.styles.theme import (
    anchor_badge,
    apply_custom_theme,
    render_hero_header,
    render_live_param_status_bar,
    render_metric_card,
    render_page_guide,
    render_section_heading,
    render_vector_equation_card,
)
from nn_core.embeddings import get_mini_vocab, get_synthetic_demo_embeddings

st.set_page_config(
    page_title="Word Embedding · NN Playground",
    layout="wide",
)

apply_custom_theme()
render_lesson_evidence("M05", show_contract=True)
render_core_result_evidence("M05")

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
    title="词嵌入与空间交互地图",
    blueprint_sections=[
        {
            "id": "A",
            "name": "语义算术控制台",
            "desc": "在左侧侧边栏配置经典向量算术公式与高亮词簇",
            "color": "amber",
            "target_id": "region-a",
        },
        {
            "id": "B",
            "name": "教学指引",
            "desc": "当前卡片：通俗理解高维空间向量距离与国王女王平行四边形",
            "color": "blue",
            "target_id": "region-b",
        },
        {
            "id": "C",
            "name": "实时算术遥测",
            "desc": "显示最优预测词、余弦相似度与几何对齐状态",
            "color": "emerald",
            "target_id": "region-c",
        },
        {
            "id": "D",
            "name": "3D 语义几何流形",
            "desc": "3D 空间直观旋转观测词汇聚类与向量加减位移虚线",
            "color": "purple",
            "target_id": "region-d",
        },
        {
            "id": "E",
            "name": "前沿 BPE 分词实验室",
            "desc": "探索教学级 BPE 分词的切片与合并过程",
            "color": "blue",
            "target_id": "region-e",
        },
    ],
    plain_intro=(
        f"<b>词嵌入将每个词汇映射为高维空间中的'方向坐标'</b>。<br>"
        f"在神经网络中，原本孤立的字符串，会被映射为一个 32 维的实数向量。<br>"
        f"下方 32 维向量是<b>代码人工注入关系轴的合成数据</b>，用来学习余弦相似度和类比运算，不是语料训练结果。"
        f"真实嵌入中的距离受训练目标、上下文和各向异性影响；<code>king - man + woman ≈ queen</code> 是某些词向量的经典现象，不等于逻辑推理证明。<br><br>"
        f"<b>【2026 前沿拓展】：BPE (字节对编码) 分词</b><br>"
        f"在查表前，现代大模型在 {anchor_badge('[E. BPE 分词实验室]', 'blue', target_id='region-e')} 采用 BPE 贪心合并策略，"
        f"将高频词缀合并为一个专属 Token，兼顾词汇量与未登录词（OOV）的处理。"
    ),
    hyperparams_desc=(
        f"• <b>在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 调节</b>：<br>"
        f"• <b>投影维度 (2D/3D)</b>：将 32 维的高维词嵌入通过 PCA 降维投影至 3D 或 2D 平面。<br>"
        f"• <b>高亮词簇</b>：直观观察动物、国家、王族等特定概念在空间中的聚集状态。<br>"
        f"• <b>向量代数运算</b>：自动计算 $A - B + C = ?$ 并验证平行四边形法则。"
    ),
    telemetry_desc=(
        f"• <b>在 {anchor_badge('[D. 3D 散点图]', 'purple', target_id='region-d')} 观测</b>：3D 空间内直观展示词汇聚落与加减法向量箭头。<br>"
        f"• <b>在 {anchor_badge('[C. 算术遥测]', 'emerald', target_id='region-c')} 揭晓</b>：全局词表中余弦夹角最小的候选词语。<br>"
        f"• <b>在 {anchor_badge('[E. BPE 实验室]', 'blue', target_id='region-e')} 验证</b>：实时展示输入文本的字符级切片与高频词块的贪心合并轨迹。"
    ),
    experiments=[
        f"<b>第 1 步【观察投影】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 选择高亮组并旋转 {anchor_badge('[D. 3D 图表]', 'purple', target_id='region-d')}；这是合成向量的 PCA 投影，低维距离可能失真，需与原 32 维余弦值对照。",
        f"<b>第 2 步【见证经典算术】</b>：在 {anchor_badge('[A. 控制台]', 'amber', target_id='region-a')} 选择 <code>king - man + woman</code>，观察 {anchor_badge('[C. 算术遥测]', 'emerald', target_id='region-c')} 计算结果中最接近的词是不是 <code>queen (女王)</code>！",
        f"<b>第 3 步【体验 BPE 分词合并】</b>：滚动到 {anchor_badge('[E. BPE 实验室]', 'blue', target_id='region-e')}，尝试输入长难句，观察系统如何将碎片的字母一步步合并为有意义的专属 Token！",
    ],
)

# ---------------------------------------------------------------------------
# 侧边栏控制面板
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div id="region-a" class="interactive-region" style="margin-bottom:0.6rem;padding:0.4rem 0.6rem;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">{anchor_badge("A", "amber")} <b>VECTOR CALCULATOR // 语义算术控制台</b></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 词表与语义向量数据准备 (中英双语标签)
# ---------------------------------------------------------------------------
raw_vocab = get_mini_vocab()
vocab_words = list(raw_vocab.keys())
embeddings_matrix = get_synthetic_demo_embeddings(len(vocab_words), d_model=32)

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

group_card_options = [
    ("全部词汇 (All Vocabulary)", "全局无遮挡展现全部词汇 2D/3D 流形空间"),
    ("王族概念 (Royalty)", "king, queen, prince, princess 性别与权阶流形"),
    ("动物世界 (Animals)", "cat, dog, kitten, puppy 物种与幼崽聚类"),
    ("国家与首都 (Geopolitics)", "china, beijing, japan, tokyo 地缘几何映射"),
    ("人类行为 (Actions)", "run, walk, think, eat 动态语义拓扑"),
]
selected_group_card = st.sidebar.radio(
    "高亮语义群组",
    options=group_card_options,
    format_func=lambda o: f"**{o[0]}**\n\n↳ *{o[1]}*",
    index=1,
    help="在散点图中高亮特定的概念族群，观察其聚集与几何分布模式。",
)
group_options = [
    "全部词汇 (All Vocabulary)",
    "王族概念 (Royalty: king, queen, prince...)",
    "动物世界 (Animals: cat, dog, kitten...)",
    "国家与首都 (Geopolitics: china, beijing, japan...)",
    "人类行为 (Actions: run, walk, think...)",
]
selected_group = group_options[group_card_options.index(selected_group_card)]

st.sidebar.markdown("#### SEMANTIC ARITHMETIC // 向量语义算术")
arithmetic_options = [
    ("king - man + woman = ? (经典王族变换)", "经典性别与王权关系向量平移"),
    ("beijing - china + japan = ? (国家首都变换)", "首都与国家地缘几何投影"),
    ("puppy - dog + cat = ? (幼崽概念类比)", "动物幼态概念语义平移"),
    ("princess - queen + king = ? (性阶转换)", "双重属性线性加减"),
    ("自定义算术方程...", "自由选择 3 个词向量进行代数计算"),
]
selected_arith_card = st.sidebar.radio(
    "预设算术方程",
    options=arithmetic_options,
    format_func=lambda o: f"**{o[0]}**\n\n↳ *{o[1]}*",
    index=0,
)
preset_arithmetic = selected_arith_card[0]

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
        top_candidates.append(
            {
                "排名 (Rank)": f"#{len(top_candidates) + 1}",
                "候选词汇 (Candidate)": CN_LABEL_MAP.get(candidate_w, candidate_w),
                "余弦相似度 (Cosine Sim)": f"{sim_val * 100:.2f}%",
                "状态 (Status)": "[CURRENT BEST // 当前候选最高]"
                if candidate_w == best_match_word
                else "[CANDIDATE // 候选]",
            }
        )

# ---------------------------------------------------------------------------
# 遥测指标卡
# ---------------------------------------------------------------------------
metric_grid_html = (
    '<div class="metric-grid">'
    + render_metric_card(
        "EQUATION RESULT // 算术最优预测",
        CN_LABEL_MAP.get(best_match_word, best_match_word).split()[0].upper(),
        delta=f"相似度 {best_match_sim * 100:.1f}%",
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
render_section_heading(
    "SEMANTIC MANIFOLD & VECTOR ARITHMETIC // 语义流形与向量算术空间", icon_name="activity"
)

render_live_param_status_bar(
    title="EMBEDDING GEOMETRY & COSINE TELEMETRY // 词向量几何与语义夹角",
    badges=[
        {"label": "Vector A", "value": f"{word_a}", "color": "blue"},
        {"label": "Vector B", "value": f"{word_b}", "color": "amber"},
        {"label": "Vector C", "value": f"{word_c}", "color": "purple"},
        {"label": "Target ≈", "value": f"{best_match_word}", "color": "emerald"},
    ],
    metrics=[
        ("最优余弦相似度", f"{best_match_sim * 100:.2f}%"),
        ("向量维度 d_model", f"{embeddings_matrix.shape[1]}D"),
        ("词表规模 |V|", f"{len(vocab_words)} words"),
    ],
    tag="SYNTHETIC GEOMETRY",
    tag_color="emerald",
)

# 确定高亮列表
highlight_tokens = []
if "Royalty" in selected_group or "王族" in selected_group:
    highlight_tokens = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"]
elif "Animals" in selected_group or "动物" in selected_group:
    highlight_tokens = ["cat", "dog", "kitten", "puppy", "fish", "bird"]
elif "Geopolitics" in selected_group or "国家" in selected_group:
    highlight_tokens = [
        "china",
        "japan",
        "usa",
        "france",
        "germany",
        "italy",
        "beijing",
        "tokyo",
        "paris",
        "rome",
    ]
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
st.plotly_chart(fig_space, width="stretch")

with st.expander("[HOW TO READ // 读图指南] 空间距离与向量语义算术", expanded=False):
    st.markdown(
        """
        * **1. 点的相对位置**：这是人工构造的 32 维向量的 PCA 投影。投影会压缩信息，因此图上距离不是原空间距离，更不能单独证明语义。
        * **2. 彩色虚线箭头（向量算术）**：
          * 红、蓝箭头展示两个向量差在投影中的方向；它们的关系由演示数据人工设定。
        * **结论边界**：图只说明“向量关系可用加减和相似度查询”，不证明模型具有逻辑类比能力。
        """
    )

# ---------------------------------------------------------------------------
# 结果细节卡片：Top-5 相似度排行 & 语义方程
# ---------------------------------------------------------------------------
col_table, col_eqn = st.columns([1.2, 1])

with col_table:
    render_section_heading("COSINE SIMILARITY RANKING // 余弦相似度搜索排行", icon_name="target")
    st.dataframe(pd.DataFrame(top_candidates), width="stretch", hide_index=True)

with col_eqn:
    render_section_heading("VECTOR EQUATION VERIFICATION // 向量方程几何验证", icon_name="cpu")
    render_vector_equation_card(word_a, word_b, word_c, best_match_word)

# ---------------------------------------------------------------------------
# 底部理论对比卡片：One-Hot vs Dense Embedding
# ---------------------------------------------------------------------------
render_section_heading("SPARSE VS DENSE // One-Hot 为何通常需要学习嵌入？", icon_name="zap")

col_oh, col_emb = st.columns(2)
with col_oh:
    with st.container(border=True):
        st.markdown(
            """
            #### [SPARSE // 稀疏孤立] 传统 One-Hot 编码
            - **形式**：`[0, 0, 0, 1, 0, ..., 0]` (长度等于全词表容量，如 50,000 维)
            - **局限**：
              1. **维度灾难**：词表多大，向量就有多长，内存极度浪费；
              2. **正交孤立**：任意两个词的点积恒为 0，`cat` 和 `dog` 的相似度与 `cat` 和 `refrigerator` 毫无区别；
              3. **缺少内生几何关系**：它仍可作为模型输入，但通常要通过嵌入层学习密集表示。
            """
        )

with col_emb:
    with st.container(border=True):
        st.markdown(
            """
            #### [DENSE // 连续密集] 词嵌入 (Word Embedding)
            - **形式**：`[0.24, -0.81, 0.43, ..., 0.15]` (低维连续实数空间，如 32~768 维)
            - **核心威力**：
              1. **可学习的几何表示**：训练目标可使某些几何关系与语言规律相关，但并非唯一或绝对语义尺度；
              2. **分布式特征**：概念通常分布在多个维度，不应预设单一维必然对应某个抽象属性；
              3. **适用范围**：许多神经语言模型使用 token embedding，具体 tokenizer、位置表示和多模态输入形式会不同。
            """
        )

# ---------------------------------------------------------------------------
# 零基础进阶：词嵌入与语义算术名词通俗速查 (含公式拆解)
# ---------------------------------------------------------------------------
with st.expander("[GROWTH GUIDE // 成长指南] 词嵌入与余弦相似度核心公式拆解", expanded=False):
    st.markdown(
        """
        ### 0. 核心公式逐字拆解：余弦相似度 (Cosine Similarity)
        $$\\text{Cosine}(u, v) = \\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\frac{\\sum_{i=1}^d u_i v_i}{\\sqrt{\\sum u_i^2} \\cdot \\sqrt{\\sum v_i^2}}$$

        | 符号 | 中文名称 | 通俗大白话解释（它是什么？起什么作用？） |
        |:---:|:---:|:---|
        | **$u, v$** | **待比较的两个词向量** | 比如 $u$ 代表“国王”的 32 维特征向量，$v$ 代表“女王”的 32 维特征向量。 |
        | **$u \\cdot v$** | **向量点积 (Dot Product)** | 对应维度数值相乘后累加。如果两个词在很多维度上都有相同的正负号，点积就会非常大。 |
        | **$\\|u\\|, \\|v\\|$** | **向量的长度模长 (L2 Norm)** | 消除词频或长短带来的干扰，只比较**纯粹的方向夹角**。 |
        | **$\\text{Cosine} \\in [-1, 1]$** | **几何方向的度量** | +1 表示同向，0 表示正交，-1 表示反向。它们是数学关系，不分别等同于“同义/无关/反义”；语义解释取决于训练与上下文。 |
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

with col_bpe_in, st.container(border=True):
    st.markdown("#### [INPUT CORPUS // 分词测试文本]")
    sample_bpe_text = st.text_area(
        "输入任意文本观察 BPE 分词切分",
        "the king and the queen ruled the kingdom and the queen was very happy",
        height=100,
        key="bpe_input_text",
    )
    target_vocab_size = st.slider(
        "目标词表容量 (Vocab Size)", min_value=260, max_value=280, value=268, step=1
    )

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
        st.caption(
            f"原始字节数: {raw_bytes_len} Bytes  压缩为: {token_count} Tokens (节省 {(1 - 1 / compression_ratio) * 100:.1f}% 序列长度)"
        )

        # 渲染彩虹色块
        token_colors = ["#dbeafe", "#fce7f3", "#dcfce7", "#fef3c7", "#f3e8ff", "#ffedd5"]
        pill_html_parts = []
        for idx, (chunk_str, tid) in enumerate(visual_chunks):
            color = token_colors[idx % len(token_colors)]
            display_str = repr(chunk_str)[1:-1]
            pill_html_parts.append(
                f"<span style=\"background:{color};border:1px solid rgba(0,0,0,0.1);padding:3px 7px;border-radius:5px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;margin:2px;display:inline-block;\">"
                f'{display_str} <sub style="color:#64748b;font-size:0.65rem;">#{tid}</sub>'
                f"</span>"
            )
        st.markdown(
            '<div style="line-height:2.2;">' + "".join(pill_html_parts) + "</div>",
            unsafe_allow_html=True,
        )

        # 展示合并历史表
        if bpe_engine.merge_history:
            st.markdown("##### [MERGE LOG // 贪心最高频合并步骤]")
            history_rows = [
                {
                    "步骤": f"#{m['step']}",
                    "合并对 (Pair)": f"{m['pair_str'][0]} + {m['pair_str'][1]}",
                    "新 Token": f"'{m['merged_str']}' (ID: {m['new_id']})",
                    "语料频次": m["freq"],
                }
                for m in bpe_engine.merge_history[:6]
            ]
            st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)
