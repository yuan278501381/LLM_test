# 🧠 NN Playground — 从零手搓神经网络到理解多模态世界模型的交互式学习平台

> 从单个感知器神经元出发，用 **纯 NumPy** 白盒实现从 MLP、Transformer、GPT 到 **多模态视觉音频、视频理解、世界模型与后训练对齐** 的完整全生命周期闭环链路，
> 配合 **Streamlit + Plotly 工业级交互式仪表板** 实时解剖每个参数与激活响应对模型行为的影响。
> 一步步彻底击碎深度学习与现代通用大模型的全部黑盒。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-198%20passed-brightgreen.svg)](#)

## ✨ 学习路径与 15 大里程碑总览

本项目将"从零深入理解现代通用大模型"的学习之旅拆分为 **15 个渐进式里程碑**，覆盖从最基础的线性分类到视频世界模型与基准考试框架：

| 篇章 | 里程碑 | 主题 | 核心可视化与教学亮点 |
|:----:|:------:|:-----|:-----------|
| **基础篇** | M01 | 单神经元感知器 | 决策边界动态平移、Loss 曲线、权重寻优轨迹 |
| | M02 | 多层网络 (MLP) | 网络拓扑图、神经元活性探针、梯度弥散/爆炸诊断 |
| | M03 | 优化器竞速对比 | SGD / Momentum / RMSProp / Adam 同屏竞速对比 |
| **微观篇** | M04 | 参数实验室 | 四宫格全维度监控、Step-by-Step 单步调试 |
| | M05 | 词嵌入空间 | 3D 语义流形、余弦相似度、国王-女王向量平行四边形 |
| | M06 | 序列记忆 (RNN) | 记忆衰减热力图、长短程遗忘瓶颈对比 |
| **LLM 篇** | M07 | 注意力机制 | QKV 矩阵分解、缩放因子开关实验、因果掩码 |
| | M08 | Transformer Block | Pre-LN 残差流、GELU/SwiGLU 门控 FFN、多层堆叠 |
| | M09 | Mini-GPT | 自回归接龙生成、Temperature/Top-K 采样、打字机 |
| **多模态篇** | M10 | 卷积与视觉感知 | 2D 卷积滑动计算、特征图热力图、ViT Patch 切片、CLIP 图文对齐 |
| | M11 | 音频信号与语音 | 实时波形示波器、FFT 频谱分解、Mel 滤波器组、Whisper Token 化 |
| | M12 | 视频与世界模型 | 32×32 视频时序采样、空间 vs 时间注意力、下一帧物理推演、Sora 扩散加噪 |
| **训练篇** | M13 | 预训练范式全景 | MLM (BERT) / CLM (GPT) / 对比 (CLIP) / MAE、下游迁移得分画像 |
| | M14 | 后训练对齐工程 | SFT、RLHF、DPO、LoRA 矩阵分解、同一问题三阶段回答质变实录 |
| **评估篇** | M15 | 评估基准框架 | Perplexity (PPL) 仪表盘、MMLU / GSM8K 模拟考场、六维雷达图、天梯排行榜 |

## 🚀 快速开始与启停最佳实践

```bash
# 1. 安装依赖 (基于现代 uv 极速包管理器)
uv sync

# 2. 运行自动化回归测试套件 (198 项全通过 · 梯度误差 < 1e-4)
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
| 核心计算引擎 | **纯 NumPy (Zero External ML Libs)** | 手写前向传播、解析梯度反向传播、优化器、MHA/GQA、ViT、Sora 扩散调度、DPO、LoRA |
| 前端交互呈现 | **Streamlit + Plotly (Linear / Stripe Light Theme)** | 世界级纯净亮色主题，零文字重叠，全屏幕高分屏自适应 (High-DPI) |
| 跨架构与跨平台 | **Python 3.11+ (Windows / macOS / Linux)** | 原生适配 x64 与 ARM64 指令集架构 |
| 测试与数学保障 | **pytest (198 项单元 + E2E 自动化测试)** | 双侧中心差分法梯度数值校验 ($\text{Error} < 10^{-4}$)，全页面 AppTest 仿真 |

## 📁 项目拓扑结构

```
├── nn_core/                  # 🔧 核心底层引擎（纯 NumPy 100% 白盒实现）
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
│   ├── audio.py              # STFT 傅里叶变换、梅尔滤波器组、AudioTokenizer、WAV 打包
│   ├── video.py              # 32x32 合成视频动力学、3D 时空 Patch 嵌入
│   ├── world_model.py        # 下一帧自回归物理世界预测模型 + Diffusion 扩散去噪调度器
│   ├── pretraining.py        # 预训练四大范式：MLM (BERT), CLM (GPT), 对比 (CLIP), MAE
│   ├── rlhf.py               # 奖励模型 (RewardModel), PPO-Clip 策略梯度, DPO 隐式损失
│   ├── lora.py               # LoRA 低秩旁路矩阵分解与权重合并
│   ├── posttraining.py       # 后训练全生命周期流水线与真实案例实录
│   ├── evaluation.py         # Perplexity 困惑度, Evaluation Harness 自动化考试框架
│   ├── model.py              # Sequential 模型容器 + Mini-batch 训练循环
│   └── callbacks.py          # 训练历史、早停机制、实验追踪
├── datasets/                 # 📦 合成数据集生成器
│   └── generators.py         # Circle, Moons, Blobs, Spirals, XOR, Wave
├── dashboard/                # 🎨 Streamlit + Plotly 世界级交互式可视化实验室
│   ├── app.py                # 15 里程碑 Bento Grid 导航主枢纽
│   ├── pages/                # 15 个渐进式实验页面 (M01 ~ M15)
│   ├── components/           # 可复用图表引擎、参数面板、网络拓扑图
│   ├── constants/            # 领域知识元数据库 (公式、教学隐喻、前沿架构)
│   ├── styles/               # 主题系统 (矢量 SVG 图标库、高对比度卡片)
│   └── utils/                # 状态管理与参数解析
├── tests/                    # ✅ 自动化回归测试矩阵 (198 项全通过)
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
│   └── test_dashboard_ui.py  # 全局 SVG 图标、图表与 15 大页面 E2E AppTest 仿真
└── scripts/                  # 🚀 跨平台自动化运维与启动脚本
```

## 🧮 数学正确性保证与梯度校验

所有核心算子的反向传播梯度均通过**双侧中心差分法**严格校验：

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \quad \text{相对误差} < 10^{-4}$$

覆盖校验：Dense 全连接层、Conv2D 卷积权重与输入特征图、MaxPool2D 梯度路由、激活函数导数链、L1/L2 正则化项、LoRA 旁路微分、LayerNorm 全微分反向链。

## 📜 开源许可证与版权声明

本项目基于 **MIT License** 开源。

```
Copyright (c) 2026 Yy1 (yuan278501381)
GitHub: https://github.com/yuan278501381
```
