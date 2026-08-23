# 🧠 NN Playground — 从零手搓神经网络到理解多模态世界模型的交互式学习平台

> 从单个感知器神经元出发，用 **纯 NumPy** 构建从 MLP、Transformer、GPT 到 **多模态感知、世界模型与后训练对齐** 的白盒教学链路，
> 配合 **Streamlit + Plotly 交互式仪表板**，观察参数、梯度与激活响应如何影响模型行为。
> 通过可运行、可验证、可解释的实验，逐步打开深度学习与现代大模型的黑盒。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-pytest%20verified-brightgreen.svg)](#)

## 🎯 项目使命与教学原则

> **项目使命（确立于 2026-08-22）**：打造一套从零基础通向深入理解的神经网络理论与实践课程。学习者不仅能知道一个模型“怎么用”，还应能解释它“为什么有效、参数如何改变结果、它解决了前代架构的什么问题，以及又付出了什么代价”。

本项目的长期目标是：

1. **从零到深入，理论与实践闭环**：从线性代数、微积分、概率与优化的必要基础出发，完成公式推导、纯 NumPy 实现、可视化观察、参数实验、结果解释与知识检验的完整学习循环。
2. **理解架构，而不是背诵名词**：系统覆盖神经网络发展史上具有代表性和持续影响力的架构、训练方法与工程技术；逐一说明其结构、关键参数、归纳偏置、适用边界、计算代价与失败模式。
3. **让参数影响可观察、可比较、可解释**：通过控制变量实验展示学习率、初始化、深度、宽度、激活函数、正则化、归一化、注意力头数、上下文长度、采样策略等因素如何影响收敛、泛化、稳定性、效率与输出行为。
4. **沿历史问题链理解每次变革**：每个关键节点都采用“前代瓶颈 → 核心创新 → 实验证据 → 新增能力 → 代价与局限 → 后续演进”的方式讲解，建立从感知器、反向传播、CNN/RNN 到 Transformer、多模态、生成模型与对齐技术的因果脉络。
5. **以教学严谨性作为第一约束**：公式、术语、数据、图表和历史结论应可追溯；关键数值应可复现；明确区分事实、经验规律、教学类比和作者推断，并持续用单元测试、数值梯度校验及参考实现交叉验证。
6. **追求世界级交互与学习体验**：交互不只追求“好看”，还应减少认知负担，提供清晰的先修知识、学习目标、操作引导、即时反馈、对照实验、错误诊断、阶段测验、进度记录和无障碍支持。

### 内容证据与范围边界

- **白盒实现不等于工业复刻**：本项目优先以小规模、可运行、可观察的实现解释原理；教学缩小版、合成数据、模拟结果和真实训练结果必须明确标注，不把演示值包装成论文复现或真实基准成绩。
- **“完整”指知识链路完整，而非穷举一切**：神经网络架构仍在快速演进。项目以“代表性、影响力、教学价值、可验证性”为收录标准，并通过版本化路线图持续扩充，而不宣称在某一时刻已经覆盖所有架构。
- **每个主题的目标交付物**：概念地图、必要数学、从零实现、参数消融实验、常见误区、历史定位、原始资料/权威参考、自动化验证与可操作练习。

### 教学可信度审计状态（更新于 2026-08-23）

> **当前状态：M17 增量整改与全站 18 页面复验已全部通过。** M17 成功完成证据链重构，四类 Claims 严格一对一绑定至四份独立权威文献，官方事故事实（Anthropic 2026-04-23 Postmortem，事故 3 提示词变更致扩展评测集 3% 下降）与本项目工程推断严格分立，剔除所有未证实虚构数字，收窄 Softmax 与 CoT 描述，全站 18 页 100% 达成蓝图目标存在性与唯一性断言。

当前代码收集 **364 项测试并全部通过**，`nn_core + datasets` 分支覆盖率为 **95.03%**（远超 80.0% 门禁标准）。

本次整改建立了以下可审计基础：

- M00-M17 已进入机器可读无环课程依赖图；每课公开诊断题、最小实验、反例实验、形成性评价和通过标准。
- 18 页当前登记 72 个 result ID；M00-M17 全部完成与文献或计算协议的一一绑定。
- 核心契约补强了 attention/mask、RoPE 显式布局、PPL/Accuracy/F1、CLIP 对比损失、扩散调度、Q-Learning/GRPO 模拟边界、局部随机源与早停恢复最佳权重。
- Ruff 覆盖 dashboard（仅保留有理由的最小页面级 ignore），Pyright 覆盖核心、数据、公共组件和课程常量；本地 DevOps 统一控制台（Pre-commit + `scripts/devops.py gate`）统一调度 7 大阶段（Format、Lint、Typecheck、Diff、364 项单测 / 95.03% 分支覆盖率、真实 Chromium 浏览器 E2E 交互与部署幂等性验证门禁）。
- 2026-08-23 浏览器全量复验确认 18 页无 Streamlit 页面异常、导航锚点目标存在性与唯一性 100% 通过。

当前基线已更新为 **364 passed / 95.03%**。覆盖率只表示受执行代码比例，不能代替断言质量与教学事实审校。

因此，README、页面和发布物不得使用“完全正确”“全部架构”“100% 覆盖”“零缺陷”“工业级复刻”等未经证据支持的认证性措辞。教学质量是按版本持续审计的属性，不是一次测试后永久成立的标签。

第一阶段历史记录见 [原交接清单](HANDOFF_SOL.md)；第二阶段验收基准以 [2026-08-23 第二阶段清单](HANDOFF_SOL_2026-08-23.md) 为准；长期不变量见 [项目记忆](PROJECT_MEMORY.md)。

## ✨ 学习路径与 M00-M17 里程碑总览

本项目目前提供 **M00-M17 共 18 个里程碑**。完整状态见 [课程地图](CURRICULUM_MAP.md)：

| 篇章 | 里程碑 | 主题 | 核心可视化与教学亮点 |
|:----:|:------:|:-----|:-----------|
| **先修篇** | M00 | 数学与计算基础 | 张量 shape、矩阵乘法、链式法则、有限差分梯度检查 |
| **基础篇** | M01 | 单神经元感知器 | 决策边界动态平移、Loss 曲线、权重寻优轨迹 |
| | M02 | 多层网络 (MLP) | 网络拓扑图、神经元活性探针、梯度弥散/爆炸诊断 |
| | M03 | 优化器竞速对比 | SGD / Momentum / RMSProp / Adam 同屏竞速对比 |
| **微观篇** | M04 | 参数实验室 | 四宫格全维度监控、Step-by-Step 单步调试 |
| | M05 | 词嵌入空间 | 合成展示向量、原空间余弦相似度、PCA 投影失真与经典类比现象边界 |
| | M06 | 序列记忆 (RNN) | 记忆衰减热力图、长短程遗忘瓶颈对比 |
| **LLM 篇** | M07 | 注意力机制 | QKV 矩阵分解、缩放因子开关实验、因果掩码 |
| | M08 | Transformer Block | Pre-LN 残差流、GELU/SwiGLU 门控 FFN、多层堆叠 |
| | M09 | Mini-GPT | 自回归接龙生成、Temperature/Top-K 采样、打字机 |
| **多模态篇** | M10 | 卷积与视觉感知 | 2D 卷积、ViT Patch、随机双塔前向与合成相似度示例（非已训练 CLIP 对齐） |
| | M11 | 音频信号与语音 | 实时波形示波器、FFT 频谱分解、Mel 滤波器组、连续音频帧切片（非 Whisper 复刻） |
| | M12 | 视频与世界模型 | 合成视频、时空 Patch、未训练输出头结构示意、DDPM 前向加噪及残余信号/SNR |
| **训练篇** | M13 | 预训练与扩展定律 | 教学规模 MLM/CLM、Contrastive/MAE、Chinchilla 条件化经验关系、BPE 与数据流程 |
| | M14 | 后训练对齐工程 | SFT/RLHF/DPO/LoRA 目标片段、预置模板案例与被省略的完整训练协议 |
| **评估篇** | M15 | 评估基准框架 | PPL 契约、MMLU/HellaSwag/GSM8K-style 自建教学题、一次采样一致的模拟结果 |
| **强化学习篇** | M16 | MDP、Q-Learning 与推理 RL 边界 | GridWorld、值迭代、Q-Learning；GRPO/R1 部分当前仅为规则曲线模拟，尚非真实语言模型训练 |
| **工程可靠性篇** | M17 | 长上下文、Agent Harness 与工程失败模式 | Attention Sinks 机制、Lost in the Middle、上下文压缩策略、控制环震荡与 2026 Claude Code 官方复盘 |

## 🚀 快速开始与启停最佳实践

```bash
# 1. 安装依赖 (基于现代 uv 极速包管理器)
uv sync

# 2. 运行自动化回归测试套件（测试数量随课程增长，以命令输出为准）
uv run pytest tests/ -v
```

### 🖥️ 最佳工程实践：启动与退出仪表盘

为了避免在 Windows 环境下由于进程树逃逸而产生占用 `8501` 端口的"孤儿进程"，强烈建议您在终端会话中**手动接管**服务生命周期：

**1. 启动服务 (Start)**
在您的 VSCode 集成终端或独立 PowerShell 中执行：
```powershell
uv run streamlit run dashboard/app.py --server.port 8501
```
> 💡 *此时服务生命周期与您的终端会话直接绑定，您可实时观测访问日志与报错堆栈。*

**2. 优雅退出 (Stop/Exit)**
在运行该服务的**同一终端窗口**内，按下：
`Ctrl + C`
> 💡 *终端会向下游广播 `SIGINT` 中断信号，干净利落地关闭 WebSocket 链接并退出底层 Python 进程，绝无端口残留隐患。*

*(或者您也可以直接使用自动化脚本 `.\scripts\run_dashboard.ps1` 进行启动)*

## 🔧 技术栈与架构标准

| 组件 | 技术标准 | 架构说明 |
|:-----|:-----|:-----|
| 核心计算引擎 | **纯 NumPy (Zero External ML Libs)** | 手写基础算子的前向/反向传播、优化器、MHA/GQA、ViT、DDPM 前向加噪调度、DPO、LoRA |
| 前端交互呈现 | **Streamlit + Plotly (Linear / Stripe Light Theme)** | 统一亮色主题、响应式布局与参数联动图表 |
| 跨架构与跨平台 | **Python 3.11+ (Windows / macOS / Linux)** | 原生适配 x64 与 ARM64 指令集架构 |
| 测试与数学保障 | **pytest 单元测试 + Streamlit AppTest** | 双侧中心差分法梯度数值校验 ($\text{Error} < 10^{-4}$)，全页面启动仿真 |

## 📁 项目拓扑结构

```
├── nn_core/                  # 🔧 核心底层引擎（纯 NumPy 教学实现）
│   ├── tensor.py             # 数值稳定性工具 (safe_log, safe_exp, clip_gradients)
│   ├── activations.py        # Sigmoid, ReLU, Tanh, LeakyReLU, Softmax, SiLU
│   ├── gelu.py               # GELU 激活函数
│   ├── swiglu.py             # SwiGLU 门控前馈单元
│   ├── losses.py             # MSE, BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── layers.py             # Dense 全连接层, Inverted Dropout
│   ├── conv2d.py             # 2D 卷积层 (基于 im2col 向量化) 与 MaxPool2D
│   ├── layernorm.py          # Layer Normalization (全微分反向链)
│   ├── optimizers.py         # SGD, Momentum, RMSProp, Adam (基于 UID Hash 通用参数协议)
│   ├── initializers.py       # Zeros, Random, Xavier (Glorot), He (Kaiming)
│   ├── regularizers.py       # L1 (Lasso), L2 (Ridge)
│   ├── bpe.py                # BPE 字节对分词器
│   ├── embeddings.py         # 词嵌入查表层 + 正弦位置编码 + 迷你词表
│   ├── rope.py               # 旋转位置编码 (Rotary Positional Embedding)
│   ├── rnn.py                # Vanilla RNN Cell (时序记忆)
│   ├── attention.py          # 缩放点积注意力 + 多头注意力 (MHA) + 因果掩码
│   ├── gqa.py                # 分组查询注意力 (Grouped-Query Attention)
│   ├── kv_cache.py           # 自回归推理 KV-Cache 显存缓存池
│   ├── transformer.py        # Pre-LN Transformer Decoder Block + FFN
│   ├── gpt.py                # TinyGPT (自回归文本生成 + Temperature/Top-K 采样)
│   ├── vit.py                # Vision Transformer (PatchEmbedding 图像切片)
│   ├── clip.py               # CLIP 图文双塔对齐与 InfoNCE 对比学习
│   ├── audio.py              # STFT、梅尔滤波器组、连续帧切片、WAV 打包
│   ├── video.py              # 32x32 合成视频动力学、3D 时空 Patch 嵌入
│   ├── world_model.py        # 教学级下一帧预测头 + Diffusion 前向加噪调度器
│   ├── pretraining.py        # 预训练四大范式：MLM (BERT), CLM (GPT), 对比 (CLIP), MAE
│   ├── rlhf.py               # 奖励模型 (RewardModel), PPO-Clip 策略梯度, DPO 隐式损失
│   ├── lora.py               # LoRA 低秩旁路矩阵分解与权重合并
│   ├── posttraining.py       # 后训练阶段与模板案例（非模型训练结果）
│   ├── evaluation.py         # Perplexity 困惑度, Evaluation Harness 自动化考试框架
│   ├── model.py              # Sequential 模型容器 + Mini-batch 训练循环
│   └── callbacks.py          # 训练历史、早停机制、实验追踪
├── datasets/                 # 📦 合成数据集生成器
│   └── generators.py         # Circle, Moons, Blobs, Spirals, XOR, Wave
├── dashboard/                # 🎨 Streamlit + Plotly 交互式可视化实验室
│   ├── app.py                # M00-M17 课程导航主枢纽
│   ├── pages/                # 17 个渐进式实验页面 (M00 ~ M16)
│   ├── components/           # 可复用图表引擎、参数面板、网络拓扑图
│   ├── constants/            # 领域知识元数据库 (公式、教学隐喻、前沿架构)
│   ├── styles/               # 主题系统 (矢量 SVG 图标库、高对比度卡片)
│   └── utils/                # 状态管理与参数解析
├── tests/                    # ✅ 自动化回归测试矩阵（数量以 pytest 收集结果为准）
│   ├── test_activations.py   # 激活函数单元测试
│   ├── test_layers.py        # Dense, Dropout 层单元测试
│   ├── test_losses.py        # 损失函数单元测试
│   ├── test_optimizers.py    # 优化器收敛单元测试
│   ├── test_gradients.py     # 数值梯度中心差分校验 (err < 1e-5)
│   ├── test_llm_core.py      # LLM 核心算子测试
│   ├── test_vision.py        # Conv2D 梯度、ViT、CLIP 对齐测试
│   ├── test_audio.py         # STFT 变换、Mel 滤波器、WAV 打包测试
│   ├── test_video.py         # 视频动力学、时空切片、Diffusion 调度测试
│   ├── test_pretraining.py   # MLM, CLM, MAE, 对比学习预训练测试
│   ├── test_alignment.py     # RLHF, DPO, LoRA, 后训练测试
│   ├── test_evaluation.py    # Perplexity 困惑度与 Harness 考试测试
│   ├── test_devops_idempotent_deploy.py # DevOps 幂等部署生命周期测试
│   └── test_dashboard_ui.py  # 全局 SVG 图标、图表与 15 大页面 E2E AppTest 仿真
└── scripts/                  # 🚀 统一 DevOps 质量门禁 (CI) 与幂等部署 (CD) 控制台
    ├── devops.py             # 统一 DevOps 调度引擎 (gate / deploy / pipeline / status)
    ├── deploy.ps1            # PowerShell 一键部署与运维入口
    └── run_dashboard.ps1     # 仪表板一键幂等拉起入口
```

## 🧮 数学核验范围与梯度校验

多个关键反向传播算子在受控、可导、有限输入上通过**双侧中心差分法**交叉核对：

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \quad \text{相对误差} < 10^{-4}$$

当前覆盖 Dense、Conv2D、激活函数、L1/L2、LoRA、LayerNorm 与 SwiGLU 等路径。一次数值梯度通过只表示当前输入、步长、精度和容差下近似一致；不可导点、近零梯度、随机算子和未覆盖路径仍需分别处理。

## 📜 开源许可证与版权声明

本项目基于 **MIT License** 开源。

```
Copyright (c) 2026 Yy1 (yuan278501381)
GitHub: https://github.com/yuan278501381
```
