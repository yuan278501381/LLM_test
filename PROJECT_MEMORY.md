# NN Playground 项目记忆

> 记忆版本：2026-08-22 教学可信度审计基线
>
> 用途：为后续 Codex、Sol 和人工维护者保存不可丢失的项目目标、事实边界与验收规则。
>
> 本文记录的是当前审计状态，不是对项目质量的永久认证。

## 1. 项目使命

建立一套能让学习者从必要数学出发，逐步理解神经网络、现代大模型、多模态、后训练、评估和强化学习的理论与实践课程。每章必须回答：

1. 前代方法遇到了什么可复现的问题；
2. 新方法改变了哪个假设、结构或训练目标；
3. 公式、代码和图表之间如何一一对应；
4. 参数变化对优化、泛化、稳定性、效率和行为有什么影响；
5. 结论在哪些条件下成立，何时会失败；
6. 哪些结果是真实计算，哪些是合成数据、教学缩小版、规则模拟或架构示意；
7. 原始论文或权威资料具体支持了哪条结论。

## 2. 不可妥协的教学规则

- 不把“程序算出了一个数”自动称为真实实验、模型能力或论文复现。
- 不把手工构造向量称为训练得到的语义，不把随机注意力图解释为语言理解。
- 不把概率模拟曲线、模板回答或预置概率称为真实训练日志、benchmark 成绩或能力涌现证据。
- 不使用“绝不、彻底、完美、终极、100% 正确、零缺陷”等无条件结论；必须给出前提、范围和反例。
- 页面结论应当能被实验推翻；只展示成功案例不构成参数因果证据。
- 引用优先级：原始论文/官方规范/作者资料 > 权威教材或综述 > 高质量实现文档；博客和营销页面不能作为核心事实的唯一来源。
- 一个引用必须与一个或一组明确教学主张绑定；“页面底部有论文链接”不等于完成事实核验。
- 教学缩小实现必须明确列出相对论文或生产系统省略了什么。
- “从零开始”意味着诊断先修知识、补齐必要数学、给出逐步推导、常见错误、形成性反馈和掌握标准，而不只是增加白话类比。

## 3. 证据等级

继续使用 `EXACT_COMPUTATION`、`TEACHING_SCALE`、`SYNTHETIC_DATA`、`SIMULATION`、`ARCHITECTURE_ONLY`、`PAPER_REPRODUCTION`，但需遵守：

- 等级应标到具体结果或实验，不应只在页面顶部粗粒度标一次。
- 同一页面可以混合多个等级；每个图表、指标、动画和案例要能追溯到自己的证据等级。
- `EXACT_COMPUTATION` 只表示代码按某个公式计算，不表示公式选择、输入数据或教学解释必然正确。
- `PAPER_REPRODUCTION` 只有在数据、预处理、模型、训练配置、随机种子、指标和误差范围均满足复现协议时才能使用；当前所有课程均无此资格。

## 4. 2026-08-22 审计结论

当前结论为：**可作为持续开发中的交互式教学原型，不可作为已经完成全面事实审校的正式教材发布。**

已确认优势：

- M00-M16 共 17 个页面均能被 Streamlit AppTest 启动；
- 多数基础算子有性质测试或数值梯度测试；
- 页面已有统一证据元数据、结论边界和原始资料入口；
- 核心 NumPy 代码覆盖率达到 90.28%，基础层、注意力、RNN、Transformer、ViT 等模块覆盖较高；
- 模拟评估、音频帧切片、扩散前向过程等部分已开始主动声明非工业复刻边界。

阻断“教学级认证”的主要问题：

- M00 对有限差分的保证、FLOPs 计数和“从零数学”覆盖不准确或不完整；
- M05-M09 仍混有手工数据/随机权重被解释为学到语义或创造力的措辞；
- M12 的随机预测头、余弦扩散端点和“纯噪声”解释需要严格收窄；
- M13 把 Chinchilla 的经验关系说成普适严格证明；
- M15 的自建题目沿用 MMLU/HellaSwag/GSM8K 名称，且 `BenchmarkTask.metric` 当前未被执行器使用；
- M16 混淆 2025 年 DeepSeek-R1、R1-Zero 与纯 RL，`GRPORunner` 只生成预设曲线而非执行 GRPO；页面还会把未验证的贪心路径称为最优路径；
- RoPE 同时支持两种四维布局的自动推断存在歧义，且“相对距离越远必然单调衰减”不是普适性质；
- 测试和 CI 文案存在“全组合、100%、零缺陷、世界级”等与实际断言不符的宣传性描述。

## 5. 当前可复现工程基线

```text
pytest + branch coverage: 271 passed, total 90.28%
ruff check:              PASS，但 dashboard/ 被排除
ruff format --check:     FAIL，8 files would be reformatted
pyright:                 FAIL，nn_core/reinforcement.py 有 3 个错误
```

覆盖率不是充分条件。当前尤其需要补强 `callbacks.py`、`clip.py`、`model.py`、`video.py`，以及所有涉及输入校验、失败路径、统计波动和跨实现对照的测试。

## 6. 测试设计记忆

测试必须优先验证可观察语义，而不是实现细节或宣传文案：

- 数学算子：已知解析值、性质、不变量、数值梯度、参考实现交叉核对、异常输入。
- 随机算法：固定随机源之外，还要做多种子统计断言和置信范围，不能断言单次曲线必然单调。
- 教学内容：用结构化 claim/source/evidence 注册表校验，不依赖“某句话不存在”的脆弱字符串测试。
- UI：冒烟测试只证明不崩溃；导航、视觉反馈、控件组合、暂停/播放和无障碍需要浏览器行为断言。
- 性能：阈值要有硬件与环境基线，避免把易波动的墙钟测试称为算法正确性。
- CI 必须实际执行 format、lint、type、unit/property、coverage 和页面行为门禁；不得在成功输出中写“零缺陷认证”。

## 7. 版本化验收规则

Sol 实施以 `HANDOFF_SOL.md` 为唯一任务清单。完成后 Codex 必须独立执行：

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=nn_core --cov=datasets --cov-branch --cov-report=term-missing -q
git diff --check
```

此外必须人工抽查所有页面的结论边界、引用映射、图表证据标签和零基础教学路径。只有 P0 阻断项全部关闭，才能使用“本版本通过教学可信度审计”；仍不得使用“永久完全正确”或“零缺陷”。

## 8. 权威核验锚点

- [Attention Is All You Need（NeurIPS 2017）](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Chinchilla compute-optimal scaling](https://arxiv.org/abs/2203.15556)
- [DeepSeek-R1（2025）](https://arxiv.org/abs/2501.12948)
- [MMLU](https://arxiv.org/abs/2009.03300)、[HellaSwag](https://arxiv.org/abs/1905.07830)、[GSM8K](https://arxiv.org/abs/2110.14168)
- [HELM：透明、可复现、多指标评估](https://crfm.stanford.edu/helm/index.html)
- [PyTorch gradcheck 限制说明](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html)
