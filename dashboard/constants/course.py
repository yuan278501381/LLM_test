# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""课程教学元数据、证据等级与权威参考资料的单一数据源。"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceLevel(StrEnum):
    """页面结果的证据性质；它描述证据，不评价内容难度。"""

    EXACT_COMPUTATION = "真实计算"
    TEACHING_SCALE = "教学缩小版"
    SYNTHETIC_DATA = "合成数据"
    SIMULATION = "概率模拟"
    ARCHITECTURE_ONLY = "架构示意"
    PAPER_REPRODUCTION = "论文复现"


class SourceType(StrEnum):
    """教学主张的来源类型。"""

    PRIMARY_PAPER = "原始论文"
    OFFICIAL_DOCUMENTATION = "官方文档"
    TEXTBOOK = "教材"
    REVIEW = "综述"


@dataclass(frozen=True)
class Reference:
    """直接支持课程结论的原始论文或权威资料。"""

    title: str
    url: str
    note: str
    source_type: SourceType
    author_or_organization: str
    year: int
    stable_identifier: str
    supports: str


class ClaimKind(StrEnum):
    CORE_FORMULA = "核心公式"
    CORE_RESULT = "核心图表/结果"
    HISTORY = "历史结论"
    FAILURE_MODE = "失败模式"


@dataclass(frozen=True)
class Claim:
    """可审计的页内教学主张。"""

    claim_id: str
    lesson_id: str
    kind: ClaimKind
    statement: str
    conditions: str
    evidence_level: EvidenceLevel
    sources: tuple[Reference, ...]
    result_id: str
    limitations: str
    last_verified: str


@dataclass(frozen=True)
class LessonMeta:
    """一个课程页面必须公开的教学契约。"""

    lesson_id: str
    title: str
    evidence: tuple[EvidenceLevel, ...]
    prerequisites: tuple[str, ...]
    objectives: tuple[str, ...]
    predecessor_problem: str
    controllable_parameters: tuple[str, ...]
    observations: tuple[str, ...]
    failure_cases: tuple[str, ...]
    conclusion_boundary: str
    historical_impact: str
    references: tuple[Reference, ...]


@dataclass(frozen=True)
class LearningLoop:
    """每课从诊断、实验到形成性评价的最小闭环。"""

    diagnostic_question: str
    minimum_experiment: str
    counterexample_experiment: str
    formative_assessment: str
    pass_criteria: str


@dataclass(frozen=True)
class FormativeQuiz:
    """不预先泄露反馈、支持诊断与重试的单题形成性测验。"""

    question: str
    options: tuple[str, ...]
    correct_index: int
    correct_explanation: str
    diagnostic_feedback: str


EVIDENCE_DESCRIPTIONS: dict[EvidenceLevel, str] = {
    EvidenceLevel.EXACT_COMPUTATION: "页面结果由当前代码按照展示公式实际计算。",
    EvidenceLevel.TEACHING_SCALE: "机制保真，但模型、数据或训练规模为便于观察而缩小。",
    EvidenceLevel.SYNTHETIC_DATA: "输入或标签由程序构造，不代表真实世界数据分布。",
    EvidenceLevel.SIMULATION: "结果由预设规则或概率生成，不代表模型真实推理或正式成绩。",
    EvidenceLevel.ARCHITECTURE_ONLY: "用于解释结构或数据流，未实现或训练完整工业模型。",
    EvidenceLevel.PAPER_REPRODUCTION: "按论文协议复现实验；必须同时公开数据、配置与指标。",
}


_REFERENCE_METADATA: dict[str, tuple[str, int, SourceType]] = {
    "The Matrix Calculus You Need For Deep Learning": (
        "Terence Parr; Jeremy Howard",
        2018,
        SourceType.REVIEW,
    ),
    "Numerical Optimization": ("Jorge Nocedal; Stephen J. Wright", 2006, SourceType.TEXTBOOK),
    "The Perceptron": ("Frank Rosenblatt", 1958, SourceType.PRIMARY_PAPER),
    "Learning representations by back-propagating errors": (
        "David Rumelhart; Geoffrey Hinton; Ronald Williams",
        1986,
        SourceType.PRIMARY_PAPER,
    ),
    "Adam: A Method for Stochastic Optimization": (
        "Diederik Kingma; Jimmy Ba",
        2014,
        SourceType.PRIMARY_PAPER,
    ),
    "Deep Learning": ("Ian Goodfellow; Yoshua Bengio; Aaron Courville", 2016, SourceType.TEXTBOOK),
    "Efficient Estimation of Word Representations": (
        "Tomas Mikolov et al.",
        2013,
        SourceType.PRIMARY_PAPER,
    ),
    "Long Short-Term Memory": (
        "Sepp Hochreiter; Jürgen Schmidhuber",
        1997,
        SourceType.PRIMARY_PAPER,
    ),
    "Neural Machine Translation by Jointly Learning to Align and Translate": (
        "Dzmitry Bahdanau; Kyunghyun Cho; Yoshua Bengio",
        2014,
        SourceType.PRIMARY_PAPER,
    ),
    "Attention Is All You Need": ("Ashish Vaswani et al.", 2017, SourceType.PRIMARY_PAPER),
    "Language Models are Unsupervised Multitask Learners": (
        "Alec Radford et al.; OpenAI",
        2019,
        SourceType.PRIMARY_PAPER,
    ),
    "An Image is Worth 16x16 Words": ("Alexey Dosovitskiy et al.", 2020, SourceType.PRIMARY_PAPER),
    "Learning Transferable Visual Models From Natural Language Supervision": (
        "Alec Radford et al.; OpenAI",
        2021,
        SourceType.PRIMARY_PAPER,
    ),
    "Robust Speech Recognition via Large-Scale Weak Supervision": (
        "Alec Radford et al.; OpenAI",
        2022,
        SourceType.PRIMARY_PAPER,
    ),
    "Denoising Diffusion Probabilistic Models": (
        "Jonathan Ho; Ajay Jain; Pieter Abbeel",
        2020,
        SourceType.PRIMARY_PAPER,
    ),
    "Scalable Diffusion Models with Transformers": (
        "William Peebles; Saining Xie",
        2022,
        SourceType.PRIMARY_PAPER,
    ),
    "Training Compute-Optimal Large Language Models": (
        "Jordan Hoffmann et al.; DeepMind",
        2022,
        SourceType.PRIMARY_PAPER,
    ),
    "Training language models to follow instructions with human feedback": (
        "Long Ouyang et al.; OpenAI",
        2022,
        SourceType.PRIMARY_PAPER,
    ),
    "Direct Preference Optimization": ("Rafael Rafailov et al.", 2023, SourceType.PRIMARY_PAPER),
    "Holistic Evaluation of Language Models": (
        "Percy Liang et al.; Stanford CRFM",
        2022,
        SourceType.PRIMARY_PAPER,
    ),
    "Reinforcement Learning: An Introduction": (
        "Richard Sutton; Andrew Barto",
        2018,
        SourceType.TEXTBOOK,
    ),
    "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning": (
        "DeepSeek-AI",
        2025,
        SourceType.PRIMARY_PAPER,
    ),
}


def _ref(
    title: str,
    url: str,
    note: str,
    *,
    source_type: SourceType | None = None,
    author_or_organization: str | None = None,
    year: int | None = None,
    supports: str | None = None,
) -> Reference:
    """创建结构化引用；stable identifier 保留 DOI/arXiv/官方 URL。"""
    metadata = _REFERENCE_METADATA.get(title)
    if metadata:
        default_author, default_year, default_type = metadata
        author_or_organization = author_or_organization or default_author
        year = year or default_year
        source_type = source_type or default_type
    if year is None:
        match = re.search(r"(?:19|20)\d{2}", f"{title} {note} {url}")
        year = int(match.group()) if match else 0
    return Reference(
        title=title,
        url=url,
        note=note,
        source_type=source_type or SourceType.PRIMARY_PAPER,
        author_or_organization=author_or_organization or "未标注",
        year=year,
        stable_identifier=url,
        supports=supports or note,
    )


LESSONS: dict[str, LessonMeta] = {
    "M00": LessonMeta(
        "M00",
        "数学与计算基础",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE),
        ("会使用 Python 基本语法",),
        ("理解 shape、矩阵乘法与广播", "用链式法则和有限差分核对梯度"),
        "直接记忆神经网络公式会掩盖维度、求导与数值稳定性错误。",
        ("有限差分步长", "输入值", "矩阵形状"),
        ("解析梯度与数值梯度相对误差", "中间张量 shape"),
        ("步长过大会产生截断误差", "步长过小会放大浮点舍入误差"),
        "有限差分是局部数值校验工具，不是训练网络的高效求导方法。",
        "线性代数、链式法则和概率建模构成后续所有里程碑的共同语言。",
        (
            _ref(
                "The Matrix Calculus You Need For Deep Learning",
                "https://arxiv.org/abs/1802.01528",
                "矩阵微积分教程",
            ),
            _ref(
                "Numerical Optimization",
                "https://doi.org/10.1007/978-0-387-40065-5",
                "数值优化权威教材",
            ),
        ),
    ),
    "M01": LessonMeta(
        "M01",
        "单神经元感知器",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SYNTHETIC_DATA),
        ("M00 的向量点积与导数",),
        ("解释线性决策边界", "观察学习率对优化轨迹的影响"),
        "固定规则无法从样本误差中自动调整决策边界。",
        ("学习率", "激活函数", "数据噪声"),
        ("损失", "权重轨迹", "决策边界"),
        ("线性不可分数据", "学习率过大导致震荡"),
        "单神经元只能表达线性边界；非线性激活不会单独创造弯曲边界。",
        "感知器把可学习参数与误差驱动更新结合，奠定监督学习基础。",
        (_ref("The Perceptron", "https://doi.org/10.1037/h0042519", "Rosenblatt 1958 原始论文"),),
    ),
    "M02": LessonMeta(
        "M02",
        "多层网络",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SYNTHETIC_DATA),
        ("M01", "链式法则"),
        ("理解隐藏层如何组合非线性", "诊断梯度消失与爆炸"),
        "线性模型不能拟合 XOR、双月等非线性决策边界。",
        ("深度", "宽度", "初始化", "激活函数"),
        ("逐层激活", "梯度分布", "泛化误差"),
        ("全零初始化的对称性", "饱和激活导致梯度衰减"),
        "更深或更宽会增加容量，但不保证优化更容易或测试性能更好。",
        "反向传播使多层表示能够端到端学习。",
        (
            _ref(
                "Learning representations by back-propagating errors",
                "https://doi.org/10.1038/323533a0",
                "Rumelhart 等 1986",
            ),
        ),
    ),
    "M03": LessonMeta(
        "M03",
        "优化器对比",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SYNTHETIC_DATA),
        ("梯度下降",),
        ("区分动量与自适应缩放", "用控制变量比较收敛轨迹"),
        "固定步长 SGD 在高曲率方向可能震荡，在平坦方向进展缓慢。",
        ("学习率", "动量系数", "二阶矩衰减"),
        ("损失曲线", "参数路径", "收敛稳定性"),
        ("Adam 不一定获得最佳泛化", "不同默认学习率使比较失真"),
        "单个合成任务的胜负不能证明某优化器普遍优于其他优化器。",
        "动量与自适应矩估计改变了深度网络训练动力学。",
        (
            _ref(
                "Adam: A Method for Stochastic Optimization",
                "https://arxiv.org/abs/1412.6980",
                "Adam 原始论文",
            ),
        ),
    ),
    "M04": LessonMeta(
        "M04",
        "参数实验室",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SYNTHETIC_DATA),
        ("M01-M03",),
        ("执行控制变量实验", "把训练异常映射到参数设置"),
        "只观察最终准确率无法解释训练过程为何成功或失败。",
        ("网络结构", "学习率", "正则化", "批大小"),
        ("损失", "梯度", "权重", "决策边界"),
        ("同时改变多个变量", "只报告单次随机运行"),
        "页面用于形成假设；可靠结论需要多随机种子与独立测试集。",
        "实验追踪与消融分析成为现代机器学习的基本研究方法。",
        (_ref("Deep Learning", "https://www.deeplearningbook.org/", "Goodfellow 等开放教材"),),
    ),
    "M05": LessonMeta(
        "M05",
        "词嵌入空间",
        (
            EvidenceLevel.EXACT_COMPUTATION,
            EvidenceLevel.TEACHING_SCALE,
            EvidenceLevel.SYNTHETIC_DATA,
        ),
        ("向量与余弦相似度",),
        ("理解离散 token 的连续表示", "区分展示用嵌入与语料训练嵌入"),
        "one-hot 向量不能直接表达词之间的相似关系。",
        ("嵌入维度", "BPE 词表大小"),
        ("余弦相似度", "降维投影"),
        ("把低维投影距离当作原空间精确距离", "把手工示例当作普遍语义规律"),
        "页面的小词表与预置向量用于说明几何关系，不代表大语料训练结果。",
        "分布式表示让语义关系可以通过向量运算被模型利用。",
        (
            _ref(
                "Efficient Estimation of Word Representations",
                "https://arxiv.org/abs/1301.3781",
                "word2vec 原始论文",
            ),
        ),
    ),
    "M06": LessonMeta(
        "M06",
        "序列记忆",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE),
        ("M02", "时间序列"),
        ("追踪循环状态", "解释长程梯度衰减"),
        "前馈网络没有跨时间步共享的内部状态。",
        ("序列长度", "隐藏维度", "循环权重尺度"),
        ("隐藏状态", "记忆衰减", "梯度路径长度"),
        ("长序列梯度消失/爆炸", "状态容量不足"),
        "Vanilla RNN 演示不能代表 LSTM/GRU 的门控记忆能力。",
        "LSTM 等门控结构针对普通 RNN 的长程依赖瓶颈提出改进。",
        (
            _ref(
                "Long Short-Term Memory",
                "https://doi.org/10.1162/neco.1997.9.8.1735",
                "LSTM 原始论文",
            ),
            _ref(
                "Learning long-term dependencies with gradient descent is difficult",
                "https://doi.org/10.1109/72.279181",
                "Vanilla RNN 长程梯度困难",
                author_or_organization="Yoshua Bengio; Patrice Simard; Paolo Frasconi",
                year=1994,
            ),
            _ref(
                "Learning Phrase Representations using RNN Encoder-Decoder",
                "https://arxiv.org/abs/1406.1078",
                "GRU 原始论文",
                author_or_organization="Kyunghyun Cho et al.",
                year=2014,
            ),
        ),
    ),
    "M07": LessonMeta(
        "M07",
        "注意力机制",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE),
        ("矩阵乘法", "softmax", "序列表示"),
        ("逐步计算 QKᵀ/√d", "验证掩码与权重归一化"),
        "固定长度循环状态会成为长序列信息瓶颈。",
        ("缩放因子", "因果掩码", "头数"),
        ("注意力 logits", "softmax 权重", "输出混合"),
        ("把注意力权重直接等同于因果解释", "忘记遮蔽未来 token"),
        "随机或教学权重只能解释计算机制，不能证明模型学到了语言关系。",
        "可微分内容寻址改善了序列对齐，并成为 Transformer 的核心算子。",
        (
            _ref(
                "Neural Machine Translation by Jointly Learning to Align and Translate",
                "https://arxiv.org/abs/1409.0473",
                "神经注意力早期代表论文",
            ),
            _ref(
                "Attention Is All You Need",
                "https://arxiv.org/abs/1706.03762",
                "缩放点积注意力与多头注意力",
                author_or_organization="Ashish Vaswani et al.",
                year=2017,
            ),
            _ref(
                "Attention is not Explanation",
                "https://arxiv.org/abs/1902.10186",
                "注意力权重的解释局限",
                author_or_organization="Sofia Serrano; Noah A. Smith",
                year=2019,
            ),
        ),
    ),
    "M08": LessonMeta(
        "M08",
        "Transformer Block",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.ARCHITECTURE_ONLY),
        ("M07", "残差连接", "归一化"),
        ("追踪 Pre-LN 残差流", "区分结构计算与训练后语义"),
        "循环计算限制并行性，且长路径不利于远距离信息交互。",
        ("层数", "头数", "前馈维度"),
        ("张量 shape", "残差范数", "未训练注意力权重"),
        ("把随机权重图解释成语义分工", "把残差连接视为稳定性保证"),
        "当前 Block 未经语料训练；图表展示结构与数值流，不能作为语义学习证据。",
        "Transformer 以自注意力和并行计算重塑了序列建模。",
        (
            _ref(
                "Attention Is All You Need",
                "https://arxiv.org/abs/1706.03762",
                "Transformer 原始论文",
            ),
            _ref(
                "Layer Normalization",
                "https://arxiv.org/abs/1607.06450",
                "LayerNorm 原始论文",
                author_or_organization="Jimmy Lei Ba; Jamie Ryan Kiros; Geoffrey Hinton",
                year=2016,
            ),
            _ref(
                "Deep Residual Learning for Image Recognition",
                "https://arxiv.org/abs/1512.03385",
                "残差连接原始论文",
                author_or_organization="Kaiming He et al.",
                year=2015,
            ),
            _ref(
                "On Layer Normalization in the Transformer Architecture",
                "https://arxiv.org/abs/2002.04745",
                "Pre-LN/Post-LN 训练动力学分析",
                author_or_organization="Ruibin Xiong et al.",
                year=2020,
            ),
        ),
    ),
    "M09": LessonMeta(
        "M09",
        "Mini-GPT",
        (
            EvidenceLevel.EXACT_COMPUTATION,
            EvidenceLevel.TEACHING_SCALE,
            EvidenceLevel.SYNTHETIC_DATA,
            EvidenceLevel.ARCHITECTURE_ONLY,
        ),
        ("M08", "自回归概率"),
        ("理解因果生成循环", "比较 temperature 与 top-k"),
        "固定输出规则无法根据上下文形成下一个 token 分布。",
        ("temperature", "top-k", "最大长度"),
        ("token 概率", "采样序列"),
        ("随机权重产生无意义文本", "把流畅度等同于事实正确性"),
        "TinyGPT 的规模与训练语料不能代表生产级大语言模型能力。",
        "decoder-only Transformer 与规模化预训练推动了通用生成模型。",
        (
            _ref(
                "Language Models are Unsupervised Multitask Learners",
                "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
                "GPT-2 技术报告",
            ),
        ),
    ),
    "M10": LessonMeta(
        "M10",
        "视觉感知",
        (
            EvidenceLevel.EXACT_COMPUTATION,
            EvidenceLevel.TEACHING_SCALE,
            EvidenceLevel.SYNTHETIC_DATA,
        ),
        ("二维卷积", "M07-M08"),
        ("理解局部卷积与图块 token", "比较 CNN 与 ViT 归纳偏置"),
        "全连接网络忽略图像局部结构且参数量随分辨率快速增长。",
        ("卷积核", "步幅", "patch 大小"),
        ("特征图", "patch token", "相似度"),
        ("未训练滤波器不等于语义特征", "小样本下忽略架构先验"),
        "页面展示核心算子和微型双塔，不是完整训练的视觉基础模型。",
        "CNN、ViT 与对比学习先后扩展了视觉和图文表示能力。",
        (
            _ref(
                "Gradient-Based Learning Applied to Document Recognition",
                "https://doi.org/10.1109/5.726791",
                "LeNet/CNN 经典原始资料",
                author_or_organization="Yann LeCun et al.",
                year=1998,
            ),
            _ref(
                "An Image is Worth 16x16 Words", "https://arxiv.org/abs/2010.11929", "ViT 原始论文"
            ),
            _ref(
                "Learning Transferable Visual Models From Natural Language Supervision",
                "https://arxiv.org/abs/2103.00020",
                "CLIP 原始论文",
            ),
        ),
    ),
    "M11": LessonMeta(
        "M11",
        "音频感知",
        (
            EvidenceLevel.EXACT_COMPUTATION,
            EvidenceLevel.SYNTHETIC_DATA,
            EvidenceLevel.ARCHITECTURE_ONLY,
        ),
        ("傅里叶变换", "矩阵表示"),
        ("从波形计算 log-Mel 特征", "区分连续帧切片与离散 tokenizer"),
        "原始波形难以直接呈现随时间变化的频率结构。",
        ("采样率", "FFT 窗长", "hop length", "Mel 频带数"),
        ("波形", "频谱", "log-Mel 特征"),
        ("混叠", "窗口泄漏", "把帧切片称为 Whisper tokenizer"),
        "当前实现不包含 Whisper 的卷积前端、Encoder-Decoder、文本 tokenizer 或训练权重。",
        "时频表示与 Transformer 让大规模弱监督语音识别成为可能。",
        (
            _ref(
                "A Tutorial on Short-Time Spectrum Analysis",
                "https://doi.org/10.1109/PROC.1977.10770",
                "STFT 窗函数与时频分析",
                author_or_organization="J. B. Allen; L. R. Rabiner",
                year=1977,
            ),
            _ref(
                "A Scale for the Measurement of the Psychological Magnitude Pitch",
                "https://doi.org/10.1121/1.1915893",
                "Mel 频率标度的心理声学来源",
                author_or_organization="Stanley Smith Stevens; John Volkmann; Edwin Newman",
                year=1937,
            ),
            _ref(
                "Robust Speech Recognition via Large-Scale Weak Supervision",
                "https://arxiv.org/abs/2212.04356",
                "Whisper 论文",
            ),
        ),
    ),
    "M12": LessonMeta(
        "M12",
        "视频与世界模型",
        (
            EvidenceLevel.EXACT_COMPUTATION,
            EvidenceLevel.SYNTHETIC_DATA,
            EvidenceLevel.ARCHITECTURE_ONLY,
        ),
        ("M08", "视频张量", "高斯噪声"),
        ("理解时空 patch", "验证 DDPM 前向加噪公式"),
        "逐帧独立处理会丢失运动与时间依赖。",
        ("帧数", "patch 大小", "扩散步数"),
        ("帧差", "时空相关性", "信号保留率"),
        ("把前向加噪误认为完整生成", "把两层预测头称为物理世界模拟器"),
        "当前实现没有反向去噪网络、视频生成训练或 Sora/DiT 复现。",
        "扩散模型和时空 token 化为视频生成及世界建模提供了通用组件。",
        (
            _ref(
                "Denoising Diffusion Probabilistic Models",
                "https://arxiv.org/abs/2006.11239",
                "DDPM 原始论文",
            ),
            _ref(
                "Scalable Diffusion Models with Transformers",
                "https://arxiv.org/abs/2212.09748",
                "DiT 原始论文",
            ),
        ),
    ),
    "M13": LessonMeta(
        "M13",
        "预训练范式",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE, EvidenceLevel.SIMULATION),
        ("M09-M12",),
        ("比较 MLM/CLM/对比学习/MAE 目标", "理解计算与数据规模的关系"),
        "针对单一标注任务训练难以利用海量未标注数据。",
        ("mask ratio", "数据配比", "token 预算"),
        ("目标函数", "重建误差", "规模计算器"),
        ("把经验 scaling law 外推到任意范围", "忽略数据质量与重复"),
        "规模定律计算器用于解释论文关系，不构成对未来模型性能的保证。",
        "自监督预训练把统一底座迁移到大量下游任务。",
        (
            _ref(
                "BERT: Pre-training of Deep Bidirectional Transformers",
                "https://arxiv.org/abs/1810.04805",
                "MLM 与 BERT 预训练",
                author_or_organization="Jacob Devlin et al.",
                year=2018,
            ),
            _ref(
                "Language Models are Unsupervised Multitask Learners",
                "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
                "GPT-2 因果语言建模",
                author_or_organization="Alec Radford et al.; OpenAI",
                year=2019,
            ),
            _ref(
                "Learning Transferable Visual Models From Natural Language Supervision",
                "https://arxiv.org/abs/2103.00020",
                "CLIP 对比预训练",
                author_or_organization="Alec Radford et al.; OpenAI",
                year=2021,
            ),
            _ref(
                "Masked Autoencoders Are Scalable Vision Learners",
                "https://arxiv.org/abs/2111.06377",
                "MAE 原始论文",
                author_or_organization="Kaiming He et al.",
                year=2021,
            ),
            _ref(
                "Neural Machine Translation of Rare Words with Subword Units",
                "https://arxiv.org/abs/1508.07909",
                "NMT 中 BPE subword 方法",
                author_or_organization="Rico Sennrich; Barry Haddow; Alexandra Birch",
                year=2015,
            ),
            _ref(
                "Training Compute-Optimal Large Language Models",
                "https://arxiv.org/abs/2203.15556",
                "Chinchilla 规模定律论文",
            ),
        ),
    ),
    "M14": LessonMeta(
        "M14",
        "后训练与对齐",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE, EvidenceLevel.SIMULATION),
        ("概率模型", "策略优化", "M13"),
        ("区分 SFT、偏好建模、PPO 与 DPO", "理解 LoRA 低秩更新"),
        "预训练目标不直接编码用户意图、帮助性与安全偏好。",
        ("KL 系数", "DPO beta", "LoRA rank"),
        ("偏好损失", "隐式奖励", "参数节省"),
        ("reward hacking", "偏好数据偏差", "把模板回答当作真实模型质变"),
        "页面计算教学目标并模拟部分轨迹，不是大模型 RLHF 训练复现。",
        "指令微调和偏好优化显著改变了基础模型的人机交互行为。",
        (
            _ref(
                "Proximal Policy Optimization Algorithms",
                "https://arxiv.org/abs/1707.06347",
                "PPO 原始论文",
                author_or_organization="John Schulman et al.; OpenAI",
                year=2017,
            ),
            _ref(
                "Training language models to follow instructions with human feedback",
                "https://arxiv.org/abs/2203.02155",
                "InstructGPT 论文",
            ),
            _ref(
                "Direct Preference Optimization", "https://arxiv.org/abs/2305.18290", "DPO 原始论文"
            ),
            _ref(
                "LoRA: Low-Rank Adaptation of Large Language Models",
                "https://arxiv.org/abs/2106.09685",
                "LoRA 原始论文",
                author_or_organization="Edward Hu et al.; Microsoft",
                year=2021,
            ),
        ),
    ),
    "M15": LessonMeta(
        "M15",
        "评估基准",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SIMULATION),
        ("分类指标", "语言模型概率"),
        ("计算 PPL、accuracy 与 macro-F1", "区分教学题集和正式 benchmark"),
        "主观样例不能稳定比较模型能力。",
        ("模拟答对概率", "教学题集组合"),
        ("计算指标", "模拟得分", "模拟 PPL"),
        ("数据污染", "提示格式敏感", "把概率模拟当作模型成绩"),
        "Mini 题集、mock predictor、模拟 PPL 和能力画像均不是正式评测结果。",
        "标准化评估促进模型比较，也暴露了覆盖度、污染与可复现性问题。",
        (
            _ref(
                "Speech and Language Processing",
                "https://web.stanford.edu/~jurafsky/slp3/",
                "语言模型交叉熵与困惑度",
                source_type=SourceType.TEXTBOOK,
                author_or_organization="Dan Jurafsky; James H. Martin",
                year=2026,
            ),
            _ref(
                "Measuring Massive Multitask Language Understanding",
                "https://arxiv.org/abs/2009.03300",
                "MMLU 原始论文与任务定义",
                author_or_organization="Dan Hendrycks et al.",
                year=2020,
            ),
            _ref(
                "HellaSwag: Can a Machine Really Finish Your Sentence?",
                "https://arxiv.org/abs/1905.07830",
                "HellaSwag 原始论文与任务定义",
                author_or_organization="Rowan Zellers et al.",
                year=2019,
            ),
            _ref(
                "Training Verifiers to Solve Math Word Problems",
                "https://arxiv.org/abs/2110.14168",
                "GSM8K 数据集与协议",
                author_or_organization="Karl Cobbe et al.; OpenAI",
                year=2021,
            ),
            _ref(
                "Holistic Evaluation of Language Models",
                "https://arxiv.org/abs/2211.09110",
                "HELM 评估框架论文",
            ),
        ),
    ),
    "M16": LessonMeta(
        "M16",
        "强化学习与自主智能体",
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.TEACHING_SCALE, EvidenceLevel.SIMULATION),
        ("马尔可夫决策过程", "贝尔曼最优方程", "时序差分 Q-Learning"),
        (
            "求解离散网格 MDP 最优价值函数",
            "观察时序差分探索与利用平衡",
            "理解 DeepSeek-R1 式 GRPO 组相对优化",
        ),
        "监督学习通常从固定数据集学习；强化学习则显式建模动作、环境反馈与长期回报。",
        ("学习率 alpha", "折扣因子 gamma", "探索率 epsilon", "GRPO 训练轮数"),
        ("累积回报", "TD-Error", "贝尔曼价值曲面", "策略箭头分布", "思考链 Token 长度"),
        (
            "把单次试错失败当作无法收敛",
            "忽略探索率衰减导致无法收敛到最优策略",
            "混淆离散网格 Q-Learning 与大模型 GRPO 的适用边界",
        ),
        "当前网格寻路使用离散 Q-Table；动态规划仅是当前有限、确定性、已知转移 MDP 的数值参考。GRPO 曲线是手工规则仿真，不是语言模型训练日志。",
        "贝尔曼方程、时序差分学习和策略梯度建立了强化学习的主要方法族；2025 年 DeepSeek-R1 报告展示了这类方法在推理后训练中的一种应用。",
        (
            _ref(
                "Reinforcement Learning: An Introduction",
                "https://incompleteideas.net/book/the-book-2nd.html",
                "Sutton & Barto 强化学习圣经经典教材",
            ),
            _ref(
                "Q-learning",
                "https://doi.org/10.1007/BF00992698",
                "Q-Learning 收敛性原始论文",
                author_or_organization="Christopher Watkins; Peter Dayan",
                year=1992,
            ),
            _ref(
                "Simple statistical gradient-following algorithms for connectionist reinforcement learning",
                "https://doi.org/10.1007/BF00992696",
                "REINFORCE 原始论文",
                author_or_organization="Ronald Williams",
                year=1992,
            ),
            _ref(
                "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
                "https://arxiv.org/abs/2402.03300",
                "GRPO 算法来源",
                author_or_organization="DeepSeek-AI",
                year=2024,
            ),
            _ref(
                "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
                "https://arxiv.org/abs/2501.12948",
                "DeepSeek-R1/R1-Zero 的不同训练流程与推理案例",
            ),
        ),
    ),
    "M17": LessonMeta(
        "M17",
        "工程陷阱与Harness",
        (EvidenceLevel.SIMULATION, EvidenceLevel.TEACHING_SCALE),
        ("M07 注意力机制", "M08 Transformer", "M13 预训练范式", "M15 评估基准"),
        (
            "理解 Attention Sink 经验现象与流式 KV 缓存保留方案的适用边界",
            "理解 Lost in the Middle U 型位置敏感现象与 Rerank 权衡",
            "对比滑动截断、级联摘要与 AST 骨架保留等上下文压缩策略及反例",
            "剖析 Anthropic 2026 官方复盘与 Agent 循环工程、确定性验证及沙箱防线",
        ),
        "误以为大模型具备全能泛化且外围脚手架无需架构设计，在长上下文、反向推理、分词或工具调用中频繁遭遇物理盲区与系统崩溃。",
        (
            "流式滑动窗口与 Sink Token 配置",
            "长文档关键信息相对深度",
            "是否开启重排 (Rerank)",
            "循环工程验证器与回滚开关",
            "Claude Code 官方事故推演节点",
        ),
        (
            "流式窗口困惑度演化模拟",
            "不同深度检索率与注意力稀释",
            "实体前向与逆向条件概率断裂",
            "工具重复与关键词过滤拦截状态",
            "确定性测试门禁下循环收敛轨迹",
        ),
        (
            "丢弃初始 Token 导致滑动窗口注意力激活畸变与困惑度上升",
            "长文本中间信息检索召回率下降且重排破坏时序依赖",
            "AST 骨架压缩丢失函数体与控制流导致关键信息不可用",
            "缺乏外部验证器导致智能体陷入幻觉假阳性或级联发散",
        ),
        "模型能力与外围系统架构（Prompt 编排、工具约束、上下文管理、状态回退、执行沙箱）共同决定实际系统表现；页面曲线为规则模拟，结论需在具体任务集与模型协议下严格评测。",
        "推动 AI 应用开发从单模型崇拜演进为涵盖确定性验证门禁、受限沙箱与状态机回退的 Harness 系统级工程架构。",
        (
            _ref(
                "Efficient Streaming Language Models with Attention Sinks",
                "https://arxiv.org/abs/2309.17453",
                "StreamingLLM 与 Attention Sink 理论原始论文",
                author_or_organization="Guangxuan Xiao et al.; MIT & Meta",
                year=2023,
            ),
            _ref(
                "Lost in the Middle: How Language Models Use Long Contexts",
                "https://aclanthology.org/2024.tacl-1.9/",
                "长上下文多段落检索 U 型注意力衰减与位置偏置论文",
                author_or_organization="Nelson F. Liu et al.; Stanford",
                year=2024,
            ),
            _ref(
                "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'",
                "https://arxiv.org/abs/2309.12288",
                "自回归单向条件概率导致的逆向知识泛化断裂",
                author_or_organization="Lukas Berglund et al.; NYU",
                year=2023,
            ),
            _ref(
                "April 23, 2026: Claude Code Postmortem",
                "https://www.anthropic.com/engineering/april-23-postmortem",
                "Anthropic 官方对 2026 年 3~4 月 Claude Code 三大工程事故与改进的复盘报告",
                source_type=SourceType.OFFICIAL_DOCUMENTATION,
                author_or_organization="Anthropic",
                year=2026,
            ),
        ),
    ),
}


def _claim_evidence(lesson: LessonMeta, kind: ClaimKind) -> EvidenceLevel:
    """为默认主张选择最窄的证据类型。"""
    if kind is ClaimKind.HISTORY:
        return EvidenceLevel.TEACHING_SCALE
    if kind is ClaimKind.CORE_RESULT:
        for level in (
            EvidenceLevel.SIMULATION,
            EvidenceLevel.ARCHITECTURE_ONLY,
            EvidenceLevel.SYNTHETIC_DATA,
            EvidenceLevel.EXACT_COMPUTATION,
        ):
            if level in lesson.evidence:
                return level
    if EvidenceLevel.EXACT_COMPUTATION in lesson.evidence:
        return EvidenceLevel.EXACT_COMPUTATION
    return lesson.evidence[0]


def _build_claims() -> dict[str, Claim]:
    """为每页建立公式、结果、历史和失败模式四类最小可追溯集。"""
    claims: dict[str, Claim] = {}
    for lesson_id, lesson in LESSONS.items():
        if lesson_id == "M17":
            # 为 M17 精准绑定每项主张与专属参考论文（严格一对一绑定）
            m17_rows = (
                (
                    ClaimKind.CORE_FORMULA,
                    "本页 Attention Sink 汇聚模拟用于解析 Softmax 归一化背景下初始 Token 吸收冗余注意力的机制与流式 KV 缓存保留方案。",
                    "formula",
                    "规则模拟展示流式生成机制，不代表真实模型的端到端实测日志；Softmax 归一化是注意力汇聚的背景而非全部充分条件，不同模型的最优 Sink 数量存在差异。",
                    lesson.references[0],
                    EvidenceLevel.SIMULATION,
                ),
                (
                    ClaimKind.CORE_RESULT,
                    "页内图表用于观察长上下文多文档/键值检索下的 U 型位置敏感性以及 Rerank 重排对单点检索与时序因果的权衡。",
                    "result",
                    "U 型衰减属于特定多段落检索评测协议下的经验现象，不代表所有模型在所有输入分布下必为固定 U 曲线；Rerank 重排可能破坏上下文的自然时序与跨段落证据引用。",
                    lesson.references[1],
                    EvidenceLevel.SIMULATION,
                ),
                (
                    ClaimKind.HISTORY,
                    "从 2026 年 Anthropic 官方复盘中审阅 Client/Harness 外围工程事故（推理预算调整、会话缓存清理缺陷与系统提示词过度约束）及其修复时间线。",
                    "history",
                    "事故复盘基于 Anthropic 官方公开报告；官方记录系统提示词简短改动在某一扩展编码评测集中观察到约 3% 性能下降，官方未公布全场景总体准确率绝对数值。",
                    lesson.references[3],
                    EvidenceLevel.TEACHING_SCALE,
                ),
                (
                    ClaimKind.FAILURE_MODE,
                    "本页必须检查的失败模式：自回归因果语言模型在无双向数据增强时面临的单向条件概率断裂（逆向诅咒）。",
                    "failure",
                    "逆向诅咒指预训练权重中的单向泛化缺失；若在当前 In-Context 提示词中显式提供了实体双向关系定义，模型具备在上下文中推理反向关系的能力。",
                    lesson.references[2],
                    EvidenceLevel.SIMULATION,
                ),
            )
            for kind, statement, suffix, limitations, source, ev_level in m17_rows:
                claim_id = f"m17-{suffix}"
                claims[claim_id] = Claim(
                    claim_id=claim_id,
                    lesson_id="M17",
                    kind=kind,
                    statement=statement,
                    conditions="适用于 M17 工程陷阱与Harness 页面所公开的规则模拟、参数和证据等级。",
                    evidence_level=ev_level,
                    sources=(source,),
                    result_id=f"m17-{suffix}",
                    limitations=limitations,
                    last_verified="2026-08-22",
                )
            continue

        rows = (
            (
                ClaimKind.CORE_FORMULA,
                f"本页核心计算用于{lesson.objectives[0]}。",
                "formula",
                lesson.conclusion_boundary,
            ),
            (
                ClaimKind.CORE_RESULT,
                f"页内图表与指标用于观察：{'、'.join(lesson.observations)}。",
                "result",
                lesson.conclusion_boundary,
            ),
            (
                ClaimKind.HISTORY,
                lesson.historical_impact,
                "history",
                "历史影响不表示后续方法在所有任务上更优。",
            ),
            (
                ClaimKind.FAILURE_MODE,
                f"本页必须检查的失败模式：{lesson.failure_cases[0]}。",
                "failure",
                lesson.conclusion_boundary,
            ),
        )
        for kind, statement, suffix, limitations in rows:
            claim_id = f"{lesson_id.lower()}-{suffix}"
            source = (
                lesson.references[-1]
                if kind in {ClaimKind.HISTORY, ClaimKind.FAILURE_MODE}
                else lesson.references[0]
            )
            claims[claim_id] = Claim(
                claim_id=claim_id,
                lesson_id=lesson_id,
                kind=kind,
                statement=statement,
                conditions=f"适用于 {lesson.title} 页面所公开的实现、参数和证据等级。",
                evidence_level=_claim_evidence(lesson, kind),
                sources=(source,),
                result_id=f"{lesson_id.lower()}-{suffix}",
                limitations=limitations,
                last_verified="2026-08-22",
            )
    return claims


CLAIMS: dict[str, Claim] = _build_claims()


CURRICULUM_DAG: dict[str, tuple[str, ...]] = {
    "M00": (),
    "M01": ("M00",),
    "M02": ("M01",),
    "M03": ("M02",),
    "M04": ("M01", "M02", "M03"),
    "M05": ("M00",),
    "M06": ("M02", "M05"),
    "M07": ("M00", "M06"),
    "M08": ("M07",),
    "M09": ("M08",),
    "M10": ("M02", "M07"),
    "M11": ("M00",),
    "M12": ("M08", "M10"),
    "M13": ("M09", "M10", "M11", "M12"),
    "M14": ("M13",),
    "M15": ("M00", "M09"),
    "M16": ("M00", "M02"),
    "M17": ("M07", "M08", "M13", "M15"),
}


def _build_learning_loops() -> dict[str, LearningLoop]:
    """从教学契约生成可执行、可判定且包含反例的 17 课学习闭环。"""
    loops: dict[str, LearningLoop] = {}
    for lesson_id, lesson in LESSONS.items():
        loops[lesson_id] = LearningLoop(
            diagnostic_question=f"在实验前说明：为什么“{lesson.predecessor_problem}”会限制当前方法？",
            minimum_experiment=(
                f"只改变“{lesson.controllable_parameters[0]}”，保持其他条件不变，"
                f"记录“{lesson.observations[0]}”。"
            ),
            counterexample_experiment=(
                f"主动构造或选择“{lesson.failure_cases[0]}”，比较它与正常设置下的结果。"
            ),
            formative_assessment=(
                f"用自己的话解释观察结果，并明确为何不能超出这条边界：{lesson.conclusion_boundary}"
            ),
            pass_criteria=(
                "能够给出参数、控制变量、观测量和失败案例；结论包含适用条件，"
                "且不把合成数据、模拟或架构示意误称为真实能力。"
            ),
        )
    return loops


LEARNING_LOOPS: dict[str, LearningLoop] = _build_learning_loops()


def _build_formative_quizzes() -> dict[str, FormativeQuiz]:
    """
    为 M00-M17 体系化构建专属高阶形成性测验题库。
    全面覆盖：数学推导、Shape 推理、梯度计算、复杂度度量、图表诊断、代码实现与架构取舍。
    """
    quiz_data: dict[str, dict[str, Any]] = {
        "M00": {
            "question": "【Shape 与广播推理】在 NumPy/PyTorch 中，张量 A 形状为 (B, 1, D)，张量 B 形状为 (1, T, D)，二者逐元素相加 A + B 后的输出形状是什么？若张量 C 形状为 (B, T)，能否直接执行 (A + B) + C？",
            "options": (
                "A + B 形状为 (B, T, D)；可以直接执行 (A + B) + C，因为尾部会自动填充。",
                "A + B 形状为 (B, T, D)；不能直接与 (B, T) 相加，因为尾部维度 D ≠ 1 且未显式扩维，会触发广播形状不匹配错误。",
                "A + B 形状为 (B, T, 2D)；可以直接相加。",
            ),
            "correct_index": 1,
            "correct_explanation": "正确！根据 NumPy 广播规则，尾部对齐且长度为 1 的轴会自动广播复制，故 (B, 1, D) 与 (1, T, D) 得到 (B, T, D)；而 (B, T) 尾部缺失 D 维度，必须通过 C[:, :, None] 显式扩维为 (B, T, 1) 方可合法广播。",
            "diagnostic_feedback": "请回顾广播原则：各张量从尾部维度对齐比较，每个维度必须相等或其中一个为 1。缺失维度需显式扩维。",
        },
        "M01": {
            "question": "【数学推导与梯度形式】单神经元二分类模型中，线性加权和 z = wᵀx + b，激活函数 ŷ = σ(z) = 1/(1 + e⁻ᶻ)，二元交叉熵损失 L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]。损失对未激活加权和的导数 ∂L/∂z 是什么？经典 Rosenblatt 感知机为何不能使用该导数？",
            "options": (
                "∂L/∂z = ŷ - y；经典 Rosenblatt 感知机采用非连续阶跃激活，导数在非零处恒为 0，无法提供微积分梯度信号。",
                "∂L/∂z = (ŷ - y) / [ŷ(1-ŷ)]；经典感知机因为参数过少无法求导。",
                "∂L/∂z = 2(ŷ - y)；经典感知机使用的是 MSE 损失。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！链式法则中 ∂L/∂ŷ = (ŷ-y)/[ŷ(1-ŷ)] 与 ∂ŷ/∂z = ŷ(1-ŷ) 相乘分母完美抵消，得到 ∂L/∂z = ŷ - y；Rosenblatt 感知机采用阶跃函数，导数处处为 0，只能依靠离散纠错规则更新。",
            "diagnostic_feedback": "请回顾 Sigmoid 与二元交叉熵求导链：Sigmoid 导数 σ'(z)=σ(z)(1-σ(z)) 恰好抵消 BCE 分母，使得逻辑回归梯度极其简洁稳定。",
        },
        "M02": {
            "question": "【反向传播与通用逼近】在全连接多层网络中，第 l 层误差项定义为 δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾。若已知后层误差 δ⁽ˡ⁺¹⁾ 与权重 W⁽ˡ⁺¹⁾，δ⁽ˡ⁾ 的递推公式是什么？通用逼近定理（UAT）的核心结论是什么？",
            "options": (
                "δ⁽ˡ⁾ = (δ⁽ˡ⁺¹⁾ W⁽ˡ⁺¹⁾ᵀ) ⊙ σ'(z⁽ˡ⁾)；通用逼近定理证明单隐层非线性网络在神经元充分时具备逼近紧集上任意连续函数的容量，但不保证一阶梯度能有效优化到全局最优。",
                "δ⁽ˡ⁾ = δ⁽ˡ⁺¹⁾ W⁽ˡ⁺¹⁾；通用逼近定理要求网络必须包含 3 层以上并加入 L2 正则化方可成立。",
                "δ⁽ˡ⁾ = δ⁽ˡ⁺¹⁾ ⊙ W⁽ˡ⁺¹⁾；通用逼近定理保证了多层网络在有限样本下必然收敛到全局最优解。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！反向传播通过权重矩阵转置反投影误差并逐元素乘以激活导数；通用逼近定理（Cybenko 1989）仅证明了函数容量的存在性，并未解决非凸优化的可达性与泛化性。",
            "diagnostic_feedback": "请区分定理的'表示容量存在性'与'梯度优化可达性'：网络有能力表示该函数，并不等于训练算法一定能找到它。",
        },
        "M03": {
            "question": "【算法机理与参数化】在标准物理动量法 vₜ = β vₜ₋₁ + η gₜ, W ← W - vₜ 中，当梯度恒定为 g 时，稳态下的单步实际有效更新步长是多少？Adam 优化器中的偏差校正 m̂ₜ = mₜ / (1 - β₁ᵗ) 解决了什么问题？",
            "options": (
                "稳态有效步长为 [η / (1 - β)] · g；偏差校正消除了由于初始一阶矩 m₀ = 0 导致的迭代初期向 0 严重偏置的问题。",
                "稳态有效步长为 η(1 - β)·g；偏差校正用于保证学习率永远单调递减。",
                "稳态有效步长恒等于 η·g；偏差校正用于防止梯度爆炸。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！等比级数求和 ∑ βᵏ = 1/(1-β)，故动量在同向累积时将有效步长放大 1/(1-β) 倍；由于 m₀ 初始化为 0，期望 E[mₜ]=(1-β₁ᵗ)E[g]，除以 (1-β₁ᵗ) 可在冷启动阶段恢复无偏估计。",
            "diagnostic_feedback": "请回顾动量累积动力学与指数移动平均的冷启动特性：早期 βᵗ 接近 1 导致未校正值接近 0，必须除以 (1-βᵗ) 进行放大校正。",
        },
        "M04": {
            "question": "【正则化数学与代码实现】关于 L1 正则化、L2 正则化与 Weight Decay 的区别，下列哪一项陈述在数学与工程实现上完全正确？",
            "options": (
                "L1 正则化通过恒定次梯度 λ·sign(W) 驱动不重要分量向 0 截断产生稀疏性；在 Adam 等自适应优化器中，L2 正则化因二阶矩累积导致有效衰减失真，必须采用 AdamW 进行解耦权重衰减。",
                "L1 与 L2 正则化在任何优化器下都与 Weight Decay 完全等价，可以直接互换使用。",
                "L2 正则化会使得权重的每个分量绝对值严格平均分配；Adam 优化器天然不需要权重衰减。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！L1 的恒定梯度在零点附近提供恒定惩罚力促使参数稀疏；L2 正则化将梯度变为 g + λW，在 Adam 中进入二阶矩 √v 导致权重越大的维度衰减反而被过度惩罚，AdamW（Loshchilov et al., 2017）通过在梯度更新后直接执行 W ← W(1 - ηλ) 实现了真正的解耦衰减。",
            "diagnostic_feedback": "请回顾 AdamW 原论文（ICLR 2019）：自适应梯度算法中的 L2 正则化会导致大梯度参数衰减不足而小梯度参数过度衰减，必须解耦。",
        },
        "M05": {
            "question": "【几何与线性代数】在大模型 Embedding 层中，输入为形状 (B, T) 的整数 Token ID 矩阵，词表大小为 V，嵌入维度为 D。Embedding 查找在数学矩阵乘法上等价于什么？为什么词向量语义相似度通常采用余弦相似度而非欧氏距离？",
            "options": (
                "数学上等价于输入 One-Hot 稀疏矩阵 (B, T, V) 与嵌入权重矩阵 W_emb ∈ ℝ^(V×D) 的矩阵乘法；余弦相似度通过模长归一化消除了高频词由于梯度频繁更新导致向量模长天然偏大的尺度干扰。",
                "数学上等价于哈希查找，无法表示为矩阵乘法；欧氏距离比余弦相似度更好因为包含了长度信息。",
                "数学上等价于 Softmax 运算；余弦相似度只能衡量正相关而不能衡量负相关。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！Embedding 查找即选择权重矩阵的特定行，等价于 X_onehot @ W_emb；在语义空间中，词频差异会显著影响向量的 L2 范数，余弦相似度仅关注方向夹角，能更纯粹地反映语义相关性。",
            "diagnostic_feedback": "请回顾词嵌入查找算子的线性代数本质与余弦相似度几何定义：方向代表语义特征分配，模长受频次等外部尺度干扰。",
        },
        "M06": {
            "question": "【动力学与稳定性】在标准循环神经网络（RNN）中，隐状态更新公式为 hₜ = tanh(W_hh hₜ₋₁ + W_xh xₜ + b)。在反向传播通过时间（BPTT）计算 ∂L_T/∂h₁ 时，导致梯度消失或爆炸的核心数学原因是什么？",
            "options": (
                "雅可比矩阵连乘项 ∏ (∂hₖ/∂hₖ₋₁) = ∏ diag(1 - hₖ²) W_hhᵀ，若 W_hh 的最大奇异值小于 1 或 tanh 导数小于 1，乘积随时间跨度 T 呈指数级衰减趋近于 0。",
                "因为 RNN 每一时刻都必须重新初始化参数，导致历史梯度无法传递。",
                "因为 W_xh 矩阵形状不匹配导致求导维度丢失。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！BPTT 的长程梯度包含 T-1 个转移雅可比矩阵连乘，由于 W_hhᵀ 的幂次效应与 tanh' 的饱和衰减，梯度随序列长度指数级趋近于 0（或在奇异值大于 1 时指数爆炸），这也是 LSTM 门控与 Transformer 自注意力取代原生 RNN 的根本动力。",
            "diagnostic_feedback": "请回顾 Pascanu et al. (2013) 关于 RNN 梯度消失与爆炸的数学分析：长程时间展开对应矩阵幂次效应。",
        },
        "M07": {
            "question": "【矩阵运算与数值稳定性】在缩放点积注意力 Attention(Q, K, V) = Softmax(QKᵀ / √d_k + M) V 中，缩放因子 1/√d_k 的核心数学作用是什么？因果掩码 M 是如何构造与生效的？",
            "options": (
                "若 q, k 的各分量为独立标准正态分布，点积 q·k 的方差为 d_k；除以 √d_k 将方差拉回 1，防止点积过大落入 Softmax 饱和区导致梯度极度微弱；因果掩码在上三角填入 -∞，经 Softmax 后使未来位置权重精确为 0。",
                "缩放因子是为了将注意力矩阵的秩压缩到 √d_k；因果掩码是将未来位置的值直接替换为 0。",
                "缩放因子是为了加速 GPU 矩阵乘法；因果掩码在反向传播时丢弃梯度。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！两个 d_k 维独立标准正态向量的点积方差为 ∑ Var(q_i k_i) = d_k，若不缩放，高维下极大值会导致 Softmax 输出趋近于 One-Hot（最大值接近 1，其余接近 0），导数接近 0 造成梯度消失；因果掩码加 -∞ 使得 exp(-∞)=0，严格屏蔽未来信息。",
            "diagnostic_feedback": "请回顾 Vaswani et al. (2017) 论文 3.2.1 节：在高维空间下未缩放的点积会使 Softmax 函数进入梯度极小的极值区域。",
        },
        "M08": {
            "question": "【架构取舍与深层训练】在现代大语言模型（如 LLaMA、GPT-3、DeepSeek）中，为什么普遍采用 Pre-LN（x + SubLayer(LN(x))）而非原始 Transformer 论文中的 Post-LN（LN(x + SubLayer(x)))？",
            "options": (
                "Pre-LN 在主干残差流上保留了一条未被 LayerNorm 缩放的恒等梯度直通路径（Identity Path），使得反向传播时浅层梯度不随深度增加而衰减，无需极其脆弱的学习率 Warmup 即可稳定训练超深层模型。",
                "Pre-LN 计算速度比 Post-LN 快两倍，并且能自动消除所有过拟合风险。",
                "Pre-LN 能让注意力权重矩阵变成对称矩阵，从而节省一半的显存。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！Xiong et al. (2020) 证明 Post-LN 在输出层的 LayerNorm 导致深层梯度期望严重依赖层数，不加 Warmup 初始步梯度直接发散；而 Pre-LN 的残差连接直接跨越子层，维持了梯度的无损高速公路。",
            "diagnostic_feedback": "请参考 Xiong et al. (2020) 'On Layer Normalization in the Transformer Architecture'：Pre-LN 的梯度范数对层数稳定。",
        },
        "M09": {
            "question": "【算法复杂度计算】在自回归解码生成长度为 N 的完整文本过程中，使用 KV-Cache 相比无缓存重新计算，单步生成的浮点计算量与全序列总计算量的时间复杂度分别如何变化？",
            "options": (
                "单步注意力计算由 O(t²) 降为 O(t)，全序列总计算量由 O(N³) 降为 O(N²)；代价是需要常驻显存保存历史 Token 的 Key/Value 状态（空间复杂度 O(t)）。",
                "单步注意力计算降为 O(1)，全序列总计算量降为 O(N)；显存占用为常数 O(1)。",
                "单步计算量不变，但显存占用由 O(N) 降为 O(1)。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！生成第 t 个 token 时只需计算当前 1 个 Query 与历史 t 个 Key 的点积（O(t) 复杂度）；若无缓存每次需对 t 个 Token 全部做全序列前向（O(t²)）。对 t=1…N 累加，总计算量分别为 ∑ O(t) = O(N²) 与 ∑ O(t²) = O(N³)。",
            "diagnostic_feedback": "请回顾生成解码过程中的复杂度递推：单步 Query 维度为 (1, d)，与历史 Key (t, d) 相乘为 O(t·d)；全序列累加为二次方复杂度。",
        },
        "M10": {
            "question": "【计算机视觉与表征学习】关于深度学习二维卷积与 CLIP 对比学习损失，下列哪一项陈述完全符合科学规范？",
            "options": (
                "深度学习中的 Conv2D 算子工程实现约定为互相关（Cross-Correlation），省略了数学严格卷积中的核中心 180 度翻转；CLIP 采用对称的 InfoNCE 对比损失，通过双向交叉熵最大化正样本图文对的余弦相似度并压制负样本对。",
                "深度学习 Conv2D 严格执行了数学卷积的核翻转；CLIP 损失只计算图像对文本的单向损失。",
                "Conv2D 卷积层无法通过矩阵乘法 (im2col) 实现加速；CLIP 模型不需要温度系数 τ。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！深度学习框架中的卷积核是自学习参数，无需人工翻转核即可直接学习到所需空间滤波器特征（二者等价）；CLIP（Radford et al., 2021）采用图像到文本与文本到图像的双向对称 InfoNCE 损失，配合可学习温度系数 τ 调节对比分布平滑度。",
            "diagnostic_feedback": "请参考 Goodfellow et al. 《Deep Learning》第 9 章：深度学习库普遍使用互相关算子并通称为卷积；CLIP 采用双向对称对比损失。",
        },
        "M11": {
            "question": "【信号处理与特征工程】在短时傅里叶变换（STFT）与梅尔滤波器组（Mel Filterbank）中，窗长（n_fft）的选择面临什么基本物理权衡？梅尔刻度的非线性映射依据是什么？",
            "options": (
                "窗长受到海森堡-加博尔不确定性原理制约：窗长越长，频率分辨率越高但时间分辨率越低；梅尔刻度依据人耳听觉系统对低频敏感度远高于高频的非线性感知特性，在低频密集采样而在高频稀疏分布。",
                "窗长越长，时间和频率分辨率同时单调提升；梅尔刻度是纯线性对数变换，与人耳生理结构无关。",
                "STFT 计算不需要窗函数；梅尔滤波器组必须使用矩形窗以消除频谱泄漏。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！时频分辨率存在物理互斥 Δt · Δf ≥ 1/(4π)；Mel 刻度公式 m = 2595 log₁₀(1 + f/700) 精确模拟了人类耳蜗基底膜的非线性频率响应特性。",
            "diagnostic_feedback": "请回顾时频分析基础与心理声学原理：长窗捕捉精细音高（频域准），短窗捕捉快速起振与瞬态（时域准）。",
        },
        "M12": {
            "question": "【生成模型与动力学模拟】关于 DDPM 扩散模型的前向加噪过程与视频生成世界模型，下列哪一项理解最为科学严谨？",
            "options": (
                "DDPM 前向加噪可由高斯分布解析推导一步直达：q(xₜ|x₀) = 𝒩(xₜ; √(ᾱₜ)x₀, (1-ᾱₜ)I)；视频生成模型学习的是时间序列的条件概率分布 ℙ(xₜ|x<ₜ)，时序上的强相关性不代表建立了物理世界的因果动力学模型。",
                "DDPM 加噪过程必须一步步顺序循环执行，无法直接跳步采样；视频模型由于生成了连贯动作，证明其已经完全理解了现实因果律。",
                "扩散模型在推理逆向去噪时不需要学习方差；3D 时空切片只在空间维度切片而在时间维度做平均。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！利用高斯变量可加性 αₜ 的累积乘积 ᾱₜ = ∏ αᵢ，前向可在任意时间步 t 一步闭式生成带噪图像；统计时序预测（如 Next-Frame Prediction）仅拟合像素转移联合概率，不能等同于掌握因果干预与反事实推理能力。",
            "diagnostic_feedback": "请回顾 Ho et al. (2020) DDPM 论文公式 (4)：高斯分布的马尔可夫链前向推导可闭式合并为单步解析采样；时间相关性 ≠ 物理因果性。",
        },
        "M13": {
            "question": "【预训练工程与数据科学】在 DeepMind Chinchilla（Hoffmann et al., 2022）扩展定律与 BERT 掩码语言模型（MLM）中，下列哪项结论具有权威实验支持？",
            "options": (
                "Chinchilla Approach 3 拟合表明由于数据与参数存在边际收益非对称（α=0.34, β=0.28），算力最优配比下数据增长速度略高于参数量（D ∝ C^0.54, N ∝ C^0.46）；BERT 采用 80% [MASK]、10% 随机词、10% 保持原词的掩码混合策略以缓解预训练与微调时的输入分布不一致。",
                "Chinchilla 证明所有大模型最优配比严格且唯一恒定为 20:1；BERT 必须对所有被选中位置 100% 替换为 [MASK]。",
                "预训练 Loss 降到 1.69 以下代表模型已经掌握通用人工智能；Kaplan 扩展定律在所有算力尺度上完全优于 Chinchilla。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！Chinchilla 论文 Approach 3 揭示了非对称幂律；BERT 的 80/10/10 策略让模型在下游微调阶段（输入绝无 [MASK] 标记）也能有效利用上下文表征。",
            "diagnostic_feedback": "请回顾 Hoffmann et al. (2022) Table A3 参数拟合及 Devlin et al. (2018) BERT 论文第 3.1 节关于 80/10/10 策略的设计动机。",
        },
        "M14": {
            "question": "【后训练微调与对齐】关于 LoRA 低秩微调的显存与参数机制，下列哪一项陈述完全正确？",
            "options": (
                "LoRA（h = xW₀ + (α/r)(xA)B）初始化为 A ~ 𝒩, B=0 确保微调起点与原模型无扰动一致；LoRA 仅大幅减少可训练参数、梯度与优化器状态显存，冻结的基座权重与前向激活仍占显存，单卡超大模型需结合 QLoRA 4-bit 量化。",
                "LoRA 将微调过程中的总显存整体减少 99%，单张 24G 显卡即可全精度 16-bit 微调 70B 模型。",
                "LoRA 的矩阵 A 和 B 必须全部初始化为全零；LoRA 微调后无法合并回原始权重，推理必须保留旁路。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！B=0 使得初始 ΔW = BA = 0；显存由权重、激活、梯度和优化器状态共同组成，LoRA 只压缩了后两者的体量；QLoRA（Dettmers et al., 2023）引入 NF4 量化才解决了静态基座权重的显存瓶颈。",
            "diagnostic_feedback": "请回顾 Hu et al. (2021) LoRA 原论文：初始化必须保证 ΔW=0；显存消耗需全面核算权重、激活与优化器状态。",
        },
        "M15": {
            "question": "【大模型评测与实验设计】在评估大语言模型的 Perplexity（PPL）与 MMLU 等多项选择题基准时，下列哪种现象属于已知必须防范的评测陷阱？",
            "options": (
                "PPL 严格受分词器词表粒度与分词碎片率影响，无法跨不同 Tokenizer 绝对比较；多项选择基准评测高度敏感于选项字母顺序（存在位置偏置）、Few-shot 样本选取顺序以及 Prompt 提示词模板的微小改动。",
                "PPL 是绝对客观指标，跨模型比较无需对齐词表；基准评测得分在任何提示词模板下都完全恒定不变。",
                "只要在测试集上准确率达到 90%，就足以证明模型彻底消除了幻觉并且具备真正的逻辑推理能力。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！分词更碎的模型单个 token 预测难度降低导致 PPL 虚低；大模型在 MCQ 评测中普遍存在选择 A/B 偏置及对 Prompt 格式的脆弱敏感性，严谨评测需使用统一的评测 Harness（如 EleutherAI lm-eval）并报告置信区间与鲁棒性方差。",
            "diagnostic_feedback": "请参考 Zheng et al. (2023) 与 Sclar et al. (2023) 关于评测格式敏感性与分词器对困惑度影响的研究。",
        },
        "M16": {
            "question": "【强化学习数学与对齐】在 PPO-Clip 算法中，目标函数为 L_CLIP(θ) = Êₜ[min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)]。该截断机制的核心设计意图是什么？",
            "options": (
                "当新旧策略概率比率 rₜ 偏离 1 过大时实施硬截断，防止单次更新策略变化幅度过大（Policy Collapse），通过悲观下界估计保障策略在置信区间内平稳单调改进。",
                "截断机制是为了将所有回答的奖励分数强行压缩到 [1-ε, 1+ε] 区间内。",
                "截断机制是为了彻底消除对价值函数 Critic 网络的依赖。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！Schulman et al. (2017) 提出 PPO-Clip 通过限制概率比率 rₜ(θ) = π_θ(aₜ|sₜ) / π_θold(aₜ|sₜ) 的偏移，以一阶优化方法近似 TRPO 的二阶自然梯度信赖域约束，兼顾了计算效率与更新稳定性。",
            "diagnostic_feedback": "请回顾 Schulman et al. (2017) PPO 原论文第 3 节：min 与 clip 的组合构成了策略更新优势的悲观下界估计。",
        },
        "M17": {
            "question": "【工程复现性与评估安全】在大模型评测与训练工程中，何谓数据污染（Data Contamination）？浮点数非确定性运算如何影响实验复现？",
            "options": (
                "数据污染指评估基准测试集的题目或其改写版本在预训练/微调阶段被无意泄露进训练语料，导致模型仅凭记忆答题造成能力虚高；GPU 异步并行归约（如 atomicAdd）中非结合律浮点舍入误差会导致不同批次或硬件间产生微小数值漂移，需固定种子并约束算子确定性。",
                "数据污染是指语料中含有拼写错误；浮点数运算在任何 GPU 架构下都绝对逐位一致。",
                "只要把测试题从英文翻译成中文就能完全防止数据污染；浮点误差会自动被反向传播抵消。",
            ),
            "correct_index": 0,
            "correct_explanation": "正确！测试集污染会导致天梯榜被'刷榜'污染而失去真实泛化代表性；GPU 多线程浮点求和顺序不确定（(a+b)+c ≠ a+(b+c)）会随训练步数逐步累积放大，必须通过确定性算子调度（Deterministic Mode）与环境快照固化复现实验。",
            "diagnostic_feedback": "请参考 OpenAI GPT-4 技术报告关于数据去重污染排查与 PyTorch 官方 torch.use_deterministic_algorithms 文档。",
        },
    }

    quizzes: dict[str, FormativeQuiz] = {}
    for lesson_id, item in quiz_data.items():
        quizzes[lesson_id] = FormativeQuiz(
            question=item["question"],
            options=tuple(item["options"]),
            correct_index=item["correct_index"],
            correct_explanation=item["correct_explanation"],
            diagnostic_feedback=item["diagnostic_feedback"],
        )
    return quizzes


FORMATIVE_QUIZZES: dict[str, FormativeQuiz] = _build_formative_quizzes()


def validate_course_registry() -> None:
    """在导入和测试时快速发现缺页、空字段或不合法引用。"""

    expected = {f"M{i:02d}" for i in range(18)}
    if set(LESSONS) != expected:
        missing = sorted(expected - set(LESSONS))
        extra = sorted(set(LESSONS) - expected)
        raise ValueError(f"课程注册表不完整: missing={missing}, extra={extra}")

    if (
        set(CURRICULUM_DAG) != expected
        or set(LEARNING_LOOPS) != expected
        or set(FORMATIVE_QUIZZES) != expected
    ):
        raise ValueError("课程依赖图、学习闭环或形成性测验未覆盖 M00-M17")
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(lesson_id: str) -> None:
        if lesson_id in visiting:
            raise ValueError(f"课程依赖图存在环: {lesson_id}")
        if lesson_id in visited:
            return
        visiting.add(lesson_id)
        for prerequisite in CURRICULUM_DAG[lesson_id]:
            if prerequisite not in expected:
                raise ValueError(f"{lesson_id} 存在未知依赖: {prerequisite}")
            visit(prerequisite)
        visiting.remove(lesson_id)
        visited.add(lesson_id)

    for lesson_id in expected:
        visit(lesson_id)
        loop = LEARNING_LOOPS[lesson_id]
        if not all(
            (
                loop.diagnostic_question,
                loop.minimum_experiment,
                loop.counterexample_experiment,
                loop.formative_assessment,
                loop.pass_criteria,
            )
        ):
            raise ValueError(f"{lesson_id} 的学习闭环不完整")
        quiz = FORMATIVE_QUIZZES[lesson_id]
        if (
            len(quiz.options) < 3
            or not 0 <= quiz.correct_index < len(quiz.options)
            or len(set(quiz.options)) != len(quiz.options)
            or not quiz.question
            or not quiz.correct_explanation
            or not quiz.diagnostic_feedback
        ):
            raise ValueError(f"{lesson_id} 的形成性测验不完整")

    for lesson_id, lesson in LESSONS.items():
        if lesson.lesson_id != lesson_id or not lesson.evidence:
            raise ValueError(f"{lesson_id} 的标识或证据等级无效")
        if EvidenceLevel.PAPER_REPRODUCTION in lesson.evidence:
            raise ValueError(f"{lesson_id} 尚无资格标注为论文复现")
        if not lesson.references:
            raise ValueError(f"{lesson_id} 缺少权威参考资料")
        if any(not ref.url.startswith("https://") for ref in lesson.references):
            raise ValueError(f"{lesson_id} 存在非 HTTPS 参考链接")
        for ref in lesson.references:
            if (
                not ref.author_or_organization
                or ref.year <= 0
                or not ref.stable_identifier
                or not ref.supports
            ):
                raise ValueError(f"{lesson_id} 存在不完整的结构化引用: {ref.title}")

    if len(CLAIMS) != 72:
        raise ValueError("每个页面必须注册 4 类核心主张")
    if len(CLAIMS) != len(set(CLAIMS)):
        raise ValueError("claim ID 必须唯一")
    result_ids: set[str] = set()
    for claim_id, claim in CLAIMS.items():
        if claim.claim_id != claim_id or claim.lesson_id not in LESSONS:
            raise ValueError(f"无效 claim: {claim_id}")
        if (
            not claim.statement
            or not claim.conditions
            or not claim.limitations
            or not claim.sources
        ):
            raise ValueError(f"{claim_id} 缺少主张边界或来源")
        if not claim.result_id or claim.result_id in result_ids:
            raise ValueError(f"{claim_id} 的 result ID 缺失或重复")
        result_ids.add(claim.result_id)
        if claim.evidence_level is EvidenceLevel.PAPER_REPRODUCTION:
            raise ValueError(f"{claim_id} 尚无论文复现证据")


validate_course_registry()
