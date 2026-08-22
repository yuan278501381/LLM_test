# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""课程教学元数据、证据等级与权威参考资料的单一数据源。"""

import re
from dataclasses import dataclass
from enum import StrEnum


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
        (EvidenceLevel.EXACT_COMPUTATION, EvidenceLevel.SIMULATION),
        ("M07 注意力机制", "M08 Transformer", "M13 预训练范式", "M15 评估基准"),
        (
            "掌握 Attention Sink 与滑动窗口困惑度稳定方案",
            "理解 Lost in the Middle U 型注意力衰减与 Rerank 优化",
            "掌握上下文压缩成功/失败案例与 AST 骨架保留最佳实践",
            "剖析 2026 Claude Code 事故根因与 Agent Harness 熔断防御",
        ),
        "误以为大模型具备全能泛化且外围脚手架无需架构设计，在长上下文、反向推理、分词或工具调用中频繁遭遇物理盲区与系统崩溃。",
        (
            "滑动窗口 Sink Token 数",
            "目标信息相对深度",
            "是否开启 Rerank",
            "工具调用重复阈值",
            "Claude Code 事故状态",
        ),
        (
            "滑动窗口困惑度",
            "U 型检索准确率",
            "前向与逆向因果概率",
            "工具死循环拦截状态",
            "事故修复前后准确率与延迟",
        ),
        (
            "丢弃初始 4 个 Sink Token 导致流式生成困惑度指数爆炸",
            "关键证据置于 128K 上下文正中间且未作 Rerank 导致检索失败",
            "上下文压缩采用简单滚动摘要导致关键行号与变量被幻觉抹杀",
            "工具调用缺少幂等守卫陷入数十轮循环震荡烧光 Token",
        ),
        "AI 系统的最终表现 80% 取决于外围 Harness 控制环的严谨性；通过保留 Attention Sinks、Rerank 置顶重排、AST 骨架压缩、工具调用幂等熔断与确定性 Evaluation Harness，方可构建工业级高可用系统。",
        "推动 AI 工程从单模型崇拜走向 Harness 系统级防御，奠定了 2026 智能体控制环、长文本检索与上下文压缩的工业级最佳实践。",
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
                "https://arxiv.org/abs/2307.03172",
                "长上下文 U 型注意力衰减与位置偏置论文",
                author_or_organization="Nelson F. Liu et al.; Stanford",
                year=2023,
            ),
            _ref(
                "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'",
                "https://arxiv.org/abs/2309.12288",
                "自回归单向条件概率导致的逆向知识检索断裂",
                author_or_organization="Lukas Berglund et al.; NYU",
                year=2023,
            ),
            _ref(
                "Claude Code Performance Postmortem and Systemic Improvements",
                "https://www.anthropic.com/news/claude-code-postmortem-2026",
                "Anthropic 官方对 2026 年 3~4 月 Claude Code 三大工程事故的深度复盘报告",
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
    quizzes: dict[str, FormativeQuiz] = {}
    for lesson_id, lesson in LESSONS.items():
        correct = lesson.conclusion_boundary
        distractors = (
            "当前页面已经证明该方法在所有任务、数据和参数下都最优。",
            "页面中的合成数据、规则模拟和架构示意都可视为生产模型的真实能力。",
        )
        correct_index = int(lesson_id[1:]) % 3
        options: list[str] = list(distractors)
        options.insert(correct_index, correct)
        quizzes[lesson_id] = FormativeQuiz(
            question=f"根据 {lesson.title} 的实验，哪一项结论最符合本页证据边界？",
            options=tuple(options),
            correct_index=correct_index,
            correct_explanation=f"正确。页面能够支持的边界是：{correct}",
            diagnostic_feedback=(
                f"请区分本页证据等级，并用失败案例“{lesson.failure_cases[0]}”检查这个结论。"
            ),
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
