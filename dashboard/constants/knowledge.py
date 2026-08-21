# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
dashboard.constants.knowledge - 神经网络核心概念知识库与元数据字典 (Zero Hardcoding)

统一收敛整个系统中所有：
- 激活函数 (Activations)
- 优化器 (Optimizers)
- 权重初始化器 (Initializers)
- 损失函数 (Losses)
- 正则化策略 (Regularizers)
- 数据集拓扑 (Datasets)
- 经典一键实验预设 (Presets)

每个条目均包含：
1. `name`: 纯英文标识
2. `label`: 中英双语显示名称
3. `formula`: 数学 LaTeX 公式
4. `desc`: 详细原理解析
5. `impact`: 对训练动态与梯度的实际影响
6. `example`: 经典真实应用与学习案例
7. `tip`: 针对 Streamlit 组件 `help` 的综合说明
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActivationMeta:
    id: str
    label: str
    formula: str
    desc: str
    impact: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[MATHEMATICS // 数学公式]**: {self.formula}\n\n"
            f"• **[DEFINITION // 原理解析]**: {self.desc}\n\n"
            f"• **[DYNAMICS // 梯度影响]**: {self.impact}\n\n"
            f"• **[BENCHMARK // 实战案例]**: {self.example}"
        )


@dataclass(frozen=True)
class OptimizerMeta:
    id: str
    label: str
    formula: str
    desc: str
    impact: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[MATHEMATICS // 更新公式]**: {self.formula}\n\n"
            f"• **[DEFINITION // 原理解析]**: {self.desc}\n\n"
            f"• **[DYNAMICS // 收敛特性]**: {self.impact}\n\n"
            f"• **[BENCHMARK // 实战案例]**: {self.example}"
        )


@dataclass(frozen=True)
class InitializerMeta:
    id: str
    label: str
    formula: str
    desc: str
    impact: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[MATHEMATICS // 分布公式]**: {self.formula}\n\n"
            f"• **[DEFINITION // 原理解析]**: {self.desc}\n\n"
            f"• **[DYNAMICS // 方差传播]**: {self.impact}\n\n"
            f"• **[BENCHMARK // 实战案例]**: {self.example}"
        )


@dataclass(frozen=True)
class DatasetMeta:
    id: str
    label: str
    desc: str
    difficulty: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[TOPOLOGY // 流形特征]**: {self.desc}\n\n"
            f"• **[COMPLEXITY // 拟合难度]**: {self.difficulty}\n\n"
            f"• **[BENCHMARK // 学习场景]**: {self.example}"
        )


@dataclass(frozen=True)
class RegularizerMeta:
    id: str
    label: str
    formula: str
    desc: str
    impact: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[MATHEMATICS // 惩罚项]**: {self.formula}\n\n"
            f"• **[DEFINITION // 核心原理]**: {self.desc}\n\n"
            f"• **[DYNAMICS // 几何收缩]**: {self.impact}\n\n"
            f"• **[BENCHMARK // 实战案例]**: {self.example}"
        )


# ---------------------------------------------------------------------------
# 1. 激活函数知识库 (Activation Registry)
# ---------------------------------------------------------------------------
ACTIVATIONS: dict[str, ActivationMeta] = {
    "ReLU": ActivationMeta(
        id="ReLU",
        label="ReLU (线性整流函数)",
        formula="f(z) = \\max(0, z)",
        desc="正向区域保持完全线性，负向区域直接置零。现代深度学习中最通用、计算最快的基准激活函数。",
        impact="彻底解决正向区域的梯度消失问题；但若学习率过大导致神经元输出长期落入负区，会出现导数恒为 0 的「神经元坏死 (Dying ReLU)」现象。",
        example="绝大多数卷积神经网络 (CNN) 与深层 MLP 的隐藏层默认首选；在双螺旋、半月形流形折叠中表现优异。",
    ),
    "Sigmoid": ActivationMeta(
        id="Sigmoid",
        label="Sigmoid (S型激活函数)",
        formula="\\sigma(z) = \\frac{1}{1 + e^{-z}}",
        desc="将任意实数平滑映射到 (0, 1) 区间，历史悠久，几何形状呈现优美的 S 型曲线。",
        impact="在输入绝对值较大（|z| > 5）时导数急剧趋近于 0，在多层网络中会导致严重的前端梯度消失；输出非零均值会减慢梯度更新速度。",
        example="二分类任务的最终输出层（将得分转化为 0~1 的置信概率）；单神经元感知器线性概率划分。",
    ),
    "Tanh": ActivationMeta(
        id="Tanh",
        label="Tanh (双曲正切函数)",
        formula="\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}",
        desc="将实数映射至 (-1, 1) 区间，具有零均值（Zero-Centered）对称特性，比 Sigmoid 具有更好的中心化统计分布。",
        impact="输出均值为 0，能显著减轻下一层权重更新时的锯齿状震荡；但在饱和区依然存在导数接近 0 的梯度消失问题。",
        example="异或 (XOR) 问题破解；经典循环神经网络 (RNN/LSTM) 的隐藏状态更新门控。",
    ),
    "LeakyReLU": ActivationMeta(
        id="LeakyReLU",
        label="LeakyReLU (带泄露线性整流)",
        formula="f(z) = \\max(\\alpha z, z), \\quad \\alpha = 0.01",
        desc="在负半轴保留微小的倾斜斜率（默认 0.01），允许负信号产生微弱响应，绝不彻底封死。",
        impact="彻底杜绝了 ReLU 的神经元死亡问题，使得原本休眠的神经元在反向传播中依然能接收微弱梯度并有机会被重新唤醒。",
        example="生成对抗网络 (GAN) 判别器；包含高噪声或负向特征密集的复杂拓扑拟合。",
    ),
    "Softmax": ActivationMeta(
        id="Softmax",
        label="Softmax (归一化指数函数)",
        formula="S(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}",
        desc="将多维向量归一化为和恒等于 1 的多元互斥概率分布，极大放大最大得分的相对优势。",
        impact="配合 CategoricalCrossEntropy 损失函数时，其复合导数为预测值与真实标签的直接残差 (p - y)，求导极其优雅。",
        example="多分类模型最终输出层；大语言模型 (LLM) 词表下一个 Token 的概率分布预测。",
    ),
}

# ---------------------------------------------------------------------------
# 2. 优化器知识库 (Optimizer Registry)
# ---------------------------------------------------------------------------
OPTIMIZERS: dict[str, OptimizerMeta] = {
    "Adam": OptimizerMeta(
        id="Adam",
        label="Adam (自适应矩估计)",
        formula="\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t",
        desc="结合了 Momentum 的一阶动量（方向惯性）与 RMSProp 的二阶未中心化方差（自适应步长），并包含初始冷启动偏差修正。",
        impact="对学习率超参数鲁棒，能自适应调节不同参数的更新步长，在极高曲率和鞍点地形中收敛速度远超传统优化器。",
        example="几乎所有现代深度学习架构（Transformer、GPT、ResNet、Diffusion）的工业级默认第一选择。",
    ),
    "SGD": OptimizerMeta(
        id="SGD",
        label="SGD (随机梯度下降)",
        formula="\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta)",
        desc="最经典原生的梯度下降算法，严格沿着当前 Batch 梯度的反方向以固定步长更新参数。",
        impact="无动量积累也无自适应学习率，在峡谷与鞍点地形容易产生横向剧烈振荡，在平坦损失地形前进极其缓慢。",
        example="凸优化教学演示基准；大 Batch Size 配合退火学习率时的精细泛化微调。",
    ),
    "Momentum": OptimizerMeta(
        id="Momentum",
        label="Momentum (动量加速梯度下降)",
        formula="v_t = \\beta v_{t-1} + \\eta \\nabla W, \\quad W = W - v_t",
        desc="引入经典力学动量概念，参数更新不仅取决于当前梯度，还累积了历史运动的惯性速度（默认保留系数 β=0.9，即保留 90% 的上一步速度）。",
        impact="有效抑制垂直于前进方向的高频震荡，在平坦梯度方向持续加速滚落，能帮助模型冲出浅局部极小值和鞍点。",
        example="目标检测网络 (YOLO) 与早期卷积神经网络 (AlexNet/VGG) 的经典高效训练配置。",
    ),
    "RMSProp": OptimizerMeta(
        id="RMSProp",
        label="RMSProp (均方根自适应学习率)",
        formula="s_t = \\beta s_{t-1} + (1-\\beta)g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{s_t + \\epsilon}} g_t",
        desc="通过指数加权移动平均跟踪历史梯度的平方和，为每个参数独立动态缩放其专属学习率。",
        impact="梯度剧烈变化的参数步长自动变小，梯度稀疏平缓的参数步长自动变大，大幅缓解峡谷震荡。",
        example="循环神经网络 (RNN) 序列训练；强化学习 (DQN) 智能体训练。",
    ),
}

# ---------------------------------------------------------------------------
# 3. 初始化策略知识库 (Initializer Registry)
# ---------------------------------------------------------------------------
INITIALIZERS: dict[str, InitializerMeta] = {
    "he": InitializerMeta(
        id="he",
        label="He / Kaiming (何恺明正态分布)",
        formula="W \\sim \\mathcal{N}\\left(0,\\; \\sigma^2 = \\frac{2}{n_{in}}\\right)",
        desc="何恺明针对 ReLU 类单边抑制激活函数推导的方差补偿初始化方案，补偿了负半轴归零导致的信号能量减半。",
        impact="保持深层网络中各层激活值与梯度的方差稳定恒定，彻底避免深层前向信号衰竭或反向梯度弥散。",
        example="所有搭载 ReLU / LeakyReLU 的深层网络（如 50~152 层 ResNet）的标准标配。",
    ),
    "xavier": InitializerMeta(
        id="xavier",
        label="Xavier / Glorot (正态分布)",
        formula="W \\sim \\mathcal{N}\\left(0,\\; \\sigma^2 = \\frac{2}{n_{in} + n_{out}}\\right)",
        desc="针对 Sigmoid / Tanh 等双边对称激活函数推导的方差匹配初始化，平衡输入与输出维度的信号能量。",
        impact="确保信号在多层线性加权传递中不会指数级放大也不会衰减为零，使浅层与深层网络协同学习。",
        example="早中期 MLP、自动编码器 (Autoencoder) 以及搭载 Sigmoid/Tanh 的经典神经网络架构。",
    ),
    "random": InitializerMeta(
        id="random",
        label="Random (小方差正态分布)",
        formula="W \\sim \\mathcal{N}(0, 0.01)",
        desc="最朴素的纯随机小正态分布初始化，不考虑前后层神经元维度的动态缩放关系。",
        impact="层数较深（>3层）时，前向激活值在乘法链条中迅速衰减至趋近于 0，极易引发严重的前端梯度消失。",
        example="教学对照试验，用于清晰演示为什么缺乏方差缩放的随机初始化会毁掉深层网络。",
    ),
    "zeros": InitializerMeta(
        id="zeros",
        label="Zeros (全零基准初始化)",
        formula="W = 0, \\quad b = 0",
        desc="将所有权重矩阵和偏置向量全部设定为绝对的 0。",
        impact="破坏了「对称性破缺 (Symmetry Breaking)」，导致同层所有神经元前向输出和反向梯度完全一致，网络退化为单节点。",
        example="经典教学反面教材，验证为什么神经网络权重必须引入随机打破对称。",
    ),
}

# ---------------------------------------------------------------------------
# 4. 数据集拓扑知识库 (Dataset Registry)
# ---------------------------------------------------------------------------
DATASETS: dict[str, DatasetMeta] = {
    "moons": DatasetMeta(
        id="moons",
        label="Moons (双月形非线性分布)",
        desc="两个交错穿插的半月形流形，决策边界为一条连续扭曲的 S 型非线性曲线。",
        difficulty="⭐⭐ (简单非线性)",
        example="验证单隐藏层 (4~8 神经元) 配合 ReLU 即可完成优雅折叠与无缝分割。",
    ),
    "circles": DatasetMeta(
        id="circles",
        label="Circles (同心圆径向分布)",
        desc="内圆与外圆呈嵌套同心环状分布，类别完全依赖于到原点的径向欧氏距离。",
        difficulty="⭐⭐⭐ (中等径向非线性)",
        example="线性超平面彻底失效，需要至少两个隐藏层或足够的神经元构建闭合环形决策面。",
    ),
    "xor": DatasetMeta(
        id="xor",
        label="XOR (正交经典异或分布)",
        desc="对角象限同类别分布，即 (1,1) 与 (-1,-1) 为一类，其余为另一类，为 AI 历史著名转折点。",
        difficulty="⭐⭐ (经典逻辑非线性)",
        example="1969 年明斯基证明单层感知机无法解决异或问题，直接开启了多层神经网络 (MLP) 的时代。",
    ),
    "spiral": DatasetMeta(
        id="spiral",
        label="Spiral (双螺旋高曲率分布)",
        desc="两条互相缠绕延伸的高曲率阿基米德螺旋臂，数据流形在 2D 空间高度密集缠绕。",
        difficulty="⭐⭐⭐⭐⭐ (极难高曲率流形)",
        example="检验多层深度网络非线性特征扭曲能力的终极基准，通常需要 3 层以上深度才能完全拟合。",
    ),
    "blobs": DatasetMeta(
        id="blobs",
        label="Blobs (高斯聚类线性分布)",
        desc="标准多元高斯分布聚类，两簇数据在欧氏空间中具备明显的线性间隔。",
        difficulty="⭐ (极简线性可分)",
        example="单神经元感知机 (Single Perceptron) 最佳验证场景，直接展示单个超平面 WX+b=0 的空间划分。",
    ),
}

# ---------------------------------------------------------------------------
# 5. 正则化策略知识库 (Regularizer Registry)
# ---------------------------------------------------------------------------
REGULARIZERS: dict[str, RegularizerMeta] = {
    "None": RegularizerMeta(
        id="None",
        label="None (无正则化约束)",
        formula="L_{total} = L_{data}",
        desc="模型仅以最小化训练样本误差为目标，不对权重参数的数值大小做任何惩罚。",
        impact="在样本量较小或噪声较大时，模型可能过度拟合样本噪声，导致决策边界过度扭曲破碎（过拟合）。",
        example="干净数据集基准拟合；需要验证模型最大拟合容量上限时使用。",
    ),
    "L2": RegularizerMeta(
        id="L2",
        label="L2 (Weight Decay / 权重衰减)",
        formula="L_{total} = L_{data} + \\frac{1}{2}\\lambda \\sum W^2",
        desc="在损失函数中增加权重向量的 L2 范数平方和，促使所有权重向 0 均匀平滑衰减。",
        impact="有效抑制个别超大权重异常突出，使决策边界更加平滑自然，极大提升模型在未知数据上的泛化能力。",
        example="深度学习工业界最通用的防过拟合策略；在高噪声 (Noise > 0.2) 场景下尤为关键。",
    ),
    "L1": RegularizerMeta(
        id="L1",
        label="L1 (Lasso / 权重稀疏化)",
        formula="L_{total} = L_{data} + \\lambda \\sum |W|",
        desc="在损失函数中增加权重向量的绝对值和，反向求导时对权重施加固定大小的恒定衰减。",
        impact="驱动次要或冗余特征的权重直接归零，产生天然的稀疏性 (Sparsity)，实现内置特征选择。",
        example="高维特征筛选；模型压缩与参数轻量化场景。",
    ),
}


# ---------------------------------------------------------------------------
# 6. 一键预设方案知识库 (Presets Registry)
# ---------------------------------------------------------------------------
PRESETS_REGISTRY: dict[str, dict[str, Any]] = {
    "自定义配置 (Custom)": {
        "desc": "自由配置所有超参数与网络架构，探索微观参数协同效应",
        "dataset": "moons",
        "n_samples": 250,
        "noise": 0.12,
        "n_layers": 2,
        "neurons": [8, 4],
        "activation": "ReLU",
        "initializer": "he",
        "optimizer": "Adam",
        "lr": 0.05,
        "epochs": 150,
    },
    "线性可分基准 (Linear Baseline)": {
        "desc": "单层感知机即可完美求解的经典线性分类问题，理解超平面 WX+b=0",
        "dataset": "blobs",
        "n_samples": 200,
        "noise": 0.10,
        "n_layers": 1,
        "neurons": [1],
        "activation": "Sigmoid",
        "initializer": "xavier",
        "optimizer": "SGD",
        "lr": 0.1,
        "epochs": 100,
    },
    "XOR 历史困境与破解 (XOR Problem)": {
        "desc": "明斯基异或难题：单层线性失效，双层隐藏层破解非线性对角决策",
        "dataset": "xor",
        "n_samples": 300,
        "noise": 0.08,
        "n_layers": 2,
        "neurons": [8, 4],
        "activation": "Tanh",
        "initializer": "xavier",
        "optimizer": "Adam",
        "lr": 0.05,
        "epochs": 200,
    },
    "双螺旋奇点挑战 (Spiral Singularity)": {
        "desc": "高曲率交缠流形，检验深度网络的非线性特征空间扭曲拟合极限",
        "dataset": "spiral",
        "n_samples": 400,
        "noise": 0.15,
        "n_layers": 3,
        "neurons": [16, 12, 8],
        "activation": "LeakyReLU",
        "initializer": "he",
        "optimizer": "Adam",
        "lr": 0.03,
        "epochs": 300,
    },
    "梯度消失复现与拯救 (Vanishing Gradient)": {
        "desc": "深层 Sigmoid + Random 初始化导致前端梯度归零 vs ReLU + He 救场",
        "dataset": "circles",
        "n_samples": 300,
        "noise": 0.10,
        "n_layers": 4,
        "neurons": [12, 12, 12, 12],
        "activation": "Sigmoid",
        "initializer": "random",
        "optimizer": "SGD",
        "lr": 0.05,
        "epochs": 200,
    },
}

# ---------------------------------------------------------------------------
# 7. 现代大模型架构核心组件知识库 (2026 Modern LLM Architecture)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchitectureMeta:
    id: str
    label: str
    formula: str
    desc: str
    impact: str
    example: str

    @property
    def tip(self) -> str:
        return (
            f"**{self.label}**\n\n"
            f"• **[MATHEMATICS // 核心公式]**: {self.formula}\n\n"
            f"• **[DEFINITION // 架构解析]**: {self.desc}\n\n"
            f"• **[DYNAMICS // 系统收益]**: {self.impact}\n\n"
            f"• **[BENCHMARK // 工业标准]**: {self.example}"
        )

MODERN_LLM_ARCH: dict[str, ArchitectureMeta] = {
    "BPE": ArchitectureMeta(
        id="BPE",
        label="BPE (Byte-Pair Encoding / 字节对分词)",
        formula="\\arg\\max_{(x,y)} \\text{freq}(x, y) \\rightarrow z",
        desc="通过贪心合并数据集中出现频率最高的一对连续字节或字符，自下而上地构建子词词表。在处理未登录词 (OOV) 时可优雅回退到字符级。",
        impact="平衡了词汇表大小与序列长度，解决了传统词表庞大且稀疏的维度灾难，是现代 LLM 理解人类语言的第一道核心防线。",
        example="GPT-4 的 tiktoken 分词器 (cl100k_base)、Llama-3 的 128K 超大容量多语种分词器。",
    ),
    "RoPE": ArchitectureMeta(
        id="RoPE",
        label="RoPE (Rotary Position Embedding / 旋转位置编码)",
        formula="f(q, m) = (q_0+iq_1)e^{im\\theta}, \\quad \\langle f(q,m), f(k,n)\\rangle = \\text{Re}\\left(q k^* e^{i(m-n)\\theta}\\right)",
        desc="将位置信息通过复数域的旋转矩阵注入到 Query 和 Key 中，使得其内积结果自然带入且仅依赖于相对距离 (m-n)。",
        impact="彻底取代了早期的绝对正弦位置编码，展现出极强的长度外推泛化能力 (Length Extrapolation)，模型能轻松处理比训练时更长的文本。",
        example="2026 几乎所有开源模型的绝对标配，如 LLaMA 全系列、Qwen、Mistral 等均采用 RoPE 及其改进变体 (YaRN/NTK)。",
    ),
    "GQA": ArchitectureMeta(
        id="GQA",
        label="GQA (Grouped-Query Attention / 分组查询注意力)",
        formula="KV_{heads} = Q_{heads} / G, \\quad G \\in \\{1, 2, 4, 8\\}",
        desc="介于 MHA (多头) 和 MQA (单头) 之间的折中方案。将所有的 Query 头划分为多个组，每组共享同一个 Key 和 Value 头。",
        impact="在几乎不损失模型多头表征能力的前提下，成倍数级地压缩了推理时的 KV-Cache 显存开销与访存带宽限制。",
        example="Llama-3-70B (8 个 KV 头服务 64 个 Q 头) 与大显存吞吐场景的标准架构。",
    ),
    "SwiGLU": ArchitectureMeta(
        id="SwiGLU",
        label="SwiGLU (Swish Gated Linear Unit / 门控前馈网络)",
        formula="\\text{FFN}(x) = (\\text{Swish}(x W) \\otimes (x V)) W_2",
        desc="用两个并行的线性投影矩阵代替单个矩阵，其中一路经过 Swish 激活后作为另一路的乘法门控 (Gate)，决定信息如何流转。",
        impact="门控乘法机制引入了更高级的非线性拟合能力，在参数量相近的情况下，其知识容量和训练收敛速度均显著碾压传统 GELU MLP。",
        example="PaLM, LLaMA-2/3 等现代底层引擎抛弃经典 Transformer MLP 后的标准非线性模块。",
    ),
    "KV-Cache": ArchitectureMeta(
        id="KV-Cache",
        label="KV-Cache (自回归推理缓存容器)",
        formula="K_{1:t} = [K_{1:t-1}; k_t], \\quad V_{1:t} = [V_{1:t-1}; v_t]",
        desc="在自回归 Next-Token 生成期间，将历史 Token 计算出的 Key 和 Value 张量保存于显存中，避免重复冗余的矩阵运算。",
        impact="将 Transformer 解码步的计算时间复杂度从 $O(N^2)$ 降维至 $O(1)$，是实现高速流式输出的工程基石，但其空间复杂度会随上下文长度线性增长。",
        example="vLLM, TensorRT-LLM, TGI 等工业级高性能推理引擎中的核心 PagedAttention 显存池化管理对象。",
    ),
}
