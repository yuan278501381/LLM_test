# 🧠 NN Playground — 从零手搓神经网络到理解 LLM 的交互式学习平台

> 从单个神经元出发，用 **纯 NumPy** 白盒实现从感知器到 GPT 的完整学习链路，
> 配合 **Streamlit 交互式仪表板** 实时观察每个参数对模型行为的影响。
> 一步步揭开大语言模型 (LLM) 的全部数学本质。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-96%20passed-brightgreen.svg)](#)

## ✨ 学习路径与里程碑总览

本项目将"从零理解 LLM"的学习之旅拆分为 **9 个渐进式里程碑**，覆盖从最基础的线性分类到完整的自回归文本生成：

| 篇章 | 里程碑 | 主题 | 核心可视化 |
|:----:|:------:|:-----|:-----------|
| **基础篇** | M01 | 单神经元感知器 | 决策边界、Loss 曲线、权重寻优轨迹 |
| | M02 | 多层网络 (MLP) | 网络拓扑图、神经元活性探针、梯度健康诊断 |
| | M03 | 优化器竞速对比 | 4 优化器同屏 Loss 竞速、决策边界并排透视 |
| **微观篇** | M04 | 参数实验室 | 四宫格全维度监控、Step-by-Step 单步调试 |
| | M05 | 词嵌入空间 | 3D 语义流形、余弦相似度、向量平行四边形 |
| | M06 | 序列记忆 (RNN) | 记忆衰减热力图、长短程遗忘瓶颈对比 |
| **LLM 篇** | M07 | 注意力机制 | QKV 矩阵、缩放因子开关实验、因果掩码 |
| | M08 | Transformer Block | Pre-LN 残差流、GELU FFN、多层堆叠 |
| | M09 | Mini-GPT | 自回归生成、Temperature/Top-K 采样、实时打字机 |

## 🚀 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 运行测试（验证数学正确性 — 96 项全通过）
uv run pytest tests/ -v

# 3. 启动可视化仪表板
uv run streamlit run dashboard/app.py
```

或使用 PowerShell 脚本：
```powershell
.\scripts\run_tests.ps1       # 运行测试
.\scripts\run_dashboard.ps1   # 启动仪表板
```

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|:-----|:-----|:-----|
| 核心引擎 | **纯 NumPy** | 手写前向传播、反向传播、优化器、注意力、Transformer、GPT |
| 可视化 | **Streamlit + Plotly** | 交互式仪表板 + 世界级数据可视化图表 |
| 数据集 | **sklearn** | 仅用于生成 2D 合成数据 (Circle, Moons, Spirals 等) |
| 测试 | **pytest** | 梯度数值校验 (中心差分法) + 单元测试 + E2E 页面测试 |

## 📁 项目结构

```
├── nn_core/                  # 🔧 核心引擎（纯 NumPy 白盒实现）
│   ├── tensor.py             # 数值稳定性工具 (safe_log, safe_exp, clip_gradients)
│   ├── activations.py        # Sigmoid, ReLU, Tanh, LeakyReLU, Softmax
│   ├── gelu.py               # GELU 激活函数 (tanh 近似)
│   ├── losses.py             # MSE, BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── layers.py             # Dense 全连接层, Dropout (Inverted)
│   ├── layernorm.py          # Layer Normalization (完整反向微分链)
│   ├── optimizers.py         # SGD, Momentum, RMSProp, Adam
│   ├── initializers.py       # Zeros, Random, Xavier (Glorot), He (Kaiming)
│   ├── regularizers.py       # L1 (Lasso), L2 (Ridge)
│   ├── embeddings.py         # 词嵌入查表层 + 正弦位置编码 + 迷你词表
│   ├── rnn.py                # Vanilla RNN Cell (序列记忆)
│   ├── attention.py          # 缩放点积注意力 + 多头注意力 (MHA)
│   ├── transformer.py        # Pre-LN Transformer Decoder Block + FFN
│   ├── gpt.py                # TinyGPT (自回归生成 + Temperature/Top-K 采样)
│   ├── model.py              # Sequential 模型容器 + Mini-batch 训练循环
│   └── callbacks.py          # 训练历史、早停、实验日志
├── datasets/                 # 📦 合成数据集生成器
│   └── generators.py         # Circle, Moons, Blobs, Spirals, XOR, Wave
├── dashboard/                # 🎨 Streamlit 交互式仪表板
│   ├── app.py                # 主入口 (3×3 Bento Grid 导航)
│   ├── pages/                # 9 个里程碑页面
│   ├── components/           # 可复用 UI 组件 (图表引擎, 参数面板, 网络拓扑)
│   ├── constants/            # 知识元数据库 (公式, 教学隐喻)
│   ├── styles/               # 主题系统 (SVG 图标, 指标卡片)
│   └── utils/                # 状态管理与参数解析
├── tests/                    # ✅ 自动化测试套件 (96 项)
│   ├── test_activations.py   # 激活函数单元测试
│   ├── test_layers.py        # Dense, Dropout 层单元测试
│   ├── test_losses.py        # 损失函数单元测试
│   ├── test_optimizers.py    # 优化器单元测试
│   ├── test_gradients.py     # 数值梯度校验 (中心差分法, err < 1e-5)
│   └── test_dashboard_ui.py  # 图表工厂 + E2E 页面渲染测试
└── scripts/                  # 🚀 启动与部署脚本
```

## 🧮 数学正确性保证

所有核心算子的反向传播梯度均通过**双侧中心差分法**严格校验：

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \quad \text{相对误差} < 10^{-5}$$

覆盖的梯度检验包括：Dense 权重/偏置、Sigmoid、ReLU、Tanh、LeakyReLU、MSE、BinaryCrossEntropy、L2 正则化、多层端到端网络链。

## 📜 License

MIT License · Copyright (c) 2026 [Yy1 (yuan278501381)](https://github.com/yuan278501381)
