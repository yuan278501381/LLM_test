# 🧠 NN Playground — 手搓神经网络可视化实验平台

> 从零用 **纯 NumPy** 手写神经网络，配合 **Streamlit 交互式仪表板**，
> 通过拖动滑块实时观察每个参数对训练结果的影响。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ 功能特色

| 里程碑 | 主题 | 可视化 |
|--------|------|--------|
| 🎯 M1 | 单神经元感知器 | 决策边界、Loss 曲线、权重轨迹 |
| 🧱 M2 | 多层网络 | 网络拓扑图、激活热力图、梯度直方图 |
| ⚙️ M3 | 优化器对比 | 多优化器 Loss 对比、决策边界并排 |
| 🔬 M4 | 参数实验室 | 四宫格全维度监控、逐步训练、A/B 对比 |

## 🚀 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 运行测试（验证数学正确性）
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
|------|------|------|
| 核心引擎 | **纯 NumPy** | 手写前向传播、反向传播、优化器 |
| 可视化 | **Streamlit + Plotly** | 交互式仪表板 + 高质量图表 |
| 数据集 | **sklearn** | 仅用于生成 2D 合成数据 |
| 测试 | **pytest** | 梯度数值校验 + 单元测试 |

## 📁 项目结构

```
├── nn_core/              # 🔧 核心引擎（纯 NumPy）
│   ├── tensor.py         # 数值稳定性工具
│   ├── activations.py    # Sigmoid, ReLU, Tanh, LeakyReLU, Softmax
│   ├── losses.py         # MSE, BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── layers.py         # Dense, Dropout
│   ├── optimizers.py     # SGD, Momentum, RMSProp, Adam
│   ├── initializers.py   # Zero, Random, Xavier, He
│   ├── regularizers.py   # L1, L2
│   ├── model.py          # Sequential 模型容器
│   └── callbacks.py      # 训练历史、早停、实验日志
├── datasets/             # 📦 数据集生成器
├── dashboard/            # 🎨 Streamlit 可视化仪表板
│   ├── app.py            # 主入口
│   ├── pages/            # 4 个里程碑页面
│   └── components/       # 可复用 UI 组件
├── tests/                # ✅ 单元测试 + 梯度校验
└── scripts/              # 🚀 启动脚本
```

## 📜 License

MIT License · Copyright (c) 2026 [Yy1 (yuan278501381)](https://github.com/yuan278501381)
