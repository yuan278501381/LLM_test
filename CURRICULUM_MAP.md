# NN Playground 版本化课程地图

> 基线日期：2026-08-23
>
> 状态定义：`已实现` 表示已有可运行核心实验；`部分实现` 表示只覆盖部分算子或教学缩小版；`仅规划` 表示尚未实现，不应在产品文案中宣称已覆盖。

本路线按“前代问题 → 核心创新 → 新代价 → 后续演进”组织，而不是按流行名词堆叠。

| 历史问题链 | 核心节点 | 当前状态 | 现有入口 | 后续补齐重点 |
|---|---|---|---|---|
| 数学与可学习系统 | Shape/矩阵 → 导数/梯度 → 概率/log → softmax/交叉熵 → 感知器 → MLP | 已实现核心链路 | M00-M02 | 增加更多概率统计与优化练习 |
| 优化与稳定训练 | SGD → Momentum/RMSProp/Adam → 初始化/正则化 | 已实现 | M03-M04 | 学习率调度、AdamW、BatchNorm |
| 卷积视觉 | LeNet → AlexNet → BatchNorm → ResNet | 部分实现 | M10 的 Conv2D/MaxPool | 经典架构逐代消融与真实小数据训练 |
| 序列记忆 | Vanilla RNN → LSTM/GRU → Seq2Seq | 部分实现 | M06 的 Vanilla RNN | 门控单元、teacher forcing、encoder-decoder |
| 内容寻址 | Seq2Seq 瓶颈 → Attention → Transformer | 已实现核心算子 | M07-M08 | 训练后 checkpoint、Encoder/Decoder 对照 |
| 自回归语言模型 | n-gram → decoder-only Transformer → GPT | 部分实现 | M09 | 完整训练循环、验证集与生成质量评估 |
| 表示学习 | one-hot → word2vec → BERT/MLM | 部分实现 | M05、M13 | 真实语料训练、上下文化表示对照 |
| 生成模型 | Autoencoder → VAE/GAN → U-Net/DDPM/DiT | 部分实现 | M12 的 DDPM 前向过程 | VAE/GAN、可训练去噪器与逆向采样 |
| 视觉 Transformer | CNN → ViT → CLIP | 部分实现 | M10 | 可训练微型 ViT/CLIP 与零样本协议 |
| 音频与语音 | STFT/Mel → encoder-decoder ASR → Whisper | 部分实现 | M11 的声学前端 | 卷积下采样、Encoder-Decoder；不伪称 Whisper 复现 |
| 视频与世界模型 | 帧模型 → 时空 token → 预测/生成世界模型 | 部分实现 | M12 | 可训练动力学模型、长时评估、动作条件 |
| 规模化预训练 | 任务监督 → 自监督 → scaling/data mixture/MoE | 部分实现 | M13 | 数据质量实验、稀疏 MoE、真实训练日志 |
| 后训练对齐 | SFT → RLHF/PPO → DPO/LoRA | 部分实现 | M14 | 数据协议、完整小型训练与安全评估 |
| 评估科学 | 单指标 → 多任务 harness → 人类偏好/安全 | 部分实现 | M15 | 正式数据适配、置信区间、污染与提示敏感性 |
| 强化学习 | MDP/动态规划 → TD/Q-Learning → Policy Gradient → 大模型推理 RL | 部分实现 | M16 的 GridWorld、值迭代、Q-Learning；GRPO 仅有优势计算与规则曲线模拟 | REINFORCE 实验闭环、真实最小 GRPO 更新、多种子统计；严格区分 R1-Zero 与 R1 |
| AI 工程可靠性 | 长上下文现象 → 上下文管理 → 工具权限与 Agent 控制环 | 部分实现 | M17 的规则模拟与案例卡 | 包含 Attention Sinks、Lost in the Middle、上下文管理与 2026 Claude 复盘分析 |
| 新序列架构 | 高二次复杂度 → 状态空间模型/Mamba 等 | 仅规划 | 无 | 先建立选择标准，再决定是否纳入白盒实现 |

## 收录标准

新增主题至少满足以下一项，并必须说明未收录的代价：

1. 改变了主流架构或训练范式；
2. 解决了前代方案可复现的关键瓶颈；
3. 对理解现代模型仍有基础教学价值；
4. 能设计可运行、可观察、可证伪的控制变量实验。

每个正式章节最终应交付：必要数学、历史问题链、纯 NumPy 实现、参数消融、失败案例、结论边界、权威参考、自动化测试与形成性测验。

## 机器可读课程契约

`dashboard/constants/course.py` 是课程架构的单一数据源：

- `CURRICULUM_DAG` 声明 M00-M17 的直接先修关系，导入时执行完整覆盖、未知依赖和环检测；
- `LEARNING_LOOPS` 为每课声明诊断问题、最小控制变量实验、反例实验、形成性评价和可判定通过标准；
- `CLAIMS` 为每课登记公式、结果、历史、失败模式四类主张，并绑定证据等级、result ID、条件、局限、日期和直接来源。

页面中的课程表和证据卡来自上述注册表；路线图中标为“部分实现/仅规划”的主题不得因出现在概念地图或参考文献中而被宣传为已实现。
