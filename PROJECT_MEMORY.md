# NN Playground 项目记忆

> 记忆版本：2026-08-23 M17 增量复审
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

## 4. 2026-08-23 审计结论

历史结论：M00-M16 在 2026-08-22 通过当时既定自动化门禁和 17 页浏览器抽查。

当前结论：**M17 增量整改与全站 18 页复验已通过。** M17 成功完成证据链重构，四类 Claims 严格一对一绑定至四份独立权威文献，官方事故事实（Anthropic 2026-04-23 Postmortem，事故 3 提示词变更致扩展评测集 3% 下降）与本项目工程推断严格分立，剔除所有未证实虚构数字，收窄 Softmax 与 CoT 描述，全站 18 页 100% 达成蓝图目标存在性与唯一性断言。

本轮已关闭的原阻断项：

- M00 已区分乘法次数与 FLOP 约定，收窄有限差分保证，补 shape/广播反例及概率、log-sum-exp、softmax、交叉熵最小链路。
- M05-M09 已区分合成嵌入、投影失真、隐藏状态相似度、未训练注意力/Transformer/GPT 与真实语义或语言能力；temperature/top-k 不再代称创造力或相关性。M05 蓝图目标（region-a~e）均已完整生成并能精准定位。
- M10-M14 已标注合成 CLIP 示例、未训练下一帧头、DDPM 残余信号、Chinchilla 经验适用范围、模板回答模拟及 PPO/DPO/LoRA 省略项。
- M15 自建题已改为 `*-style 教学题`，metric 调度与 PPL/Accuracy/F1 契约已实现，并保证同一区域分数与逐题答案来自同一次抽样。
- M16 已区分 R1-Zero 与 R1 多阶段流程；规则曲线不再称 GRPO 训练或能力涌现；Q-Learning 报告到达状态、Bellman residual 与多 seed 回报差距。
- M17 已实现四条 claim 严格 1-to-1 绑定四篇独立论文与官方文献，官方事实与本项目推断明确分离，为所有仿真引擎补齐 NaN/Inf/类型边界防御，全站 18 页面无任何锚点缺失或冲突。
- 全站 18 页共有 72 个独立结果级证据 ID，全部完成与文献或计算协议的一一绑定。

仍然明确不宣称：已穷举所有 AI 架构、已复现原论文、模拟结果代表模型能力、测试覆盖等于知识正确、或不存在尚未发现的问题。

## 5. 当前可复现工程基线

```text
pytest full run:         359 passed
branch coverage:         nn_core + datasets 95.03% (门禁阈值 80.0%)
ruff check/format:       PASS，117 个 Python 文件全量纳入检查并通过
pyright:                 PASS (0 errors, 0 warnings)
browser audit:           全站 18 页全量渲染与导航锚点目标存在性/唯一性 100% 验证通过
local quality gate:      Pre-commit 钩子 + scripts/devops.py gate 7 阶段自动化全流程保证 (含自包含真实 Chromium E2E 交互与部署幂等性门禁)
```

引用注册表当前共 47 个去重链接。M17 的 Anthropic 官方复盘文献链接为 `https://www.anthropic.com/engineering/april-23-postmortem`。官方确认评测集 3% 下降归因于 4 月 16 日的系统提示词改动（4 月 20 日已回退解决）。

覆盖率不是充分条件。本轮完整运行中 `callbacks.py 95.49%`、`clip.py 95.24%`、`model.py 95.98%`、`video.py 97.22%`、`harness_traps.py 95.38%`。后续新增功能仍必须优先测试失败路径、统计波动和跨实现对照。

浏览器控制台在含 `st.iframe`/组件的页面上仍可能记录 Streamlit 自动尺寸脚本的 `MutationObserver.observe(document.body, …)` 日志；该日志无项目源码 URL 或应用堆栈，页面与交互未抛出 Streamlit 异常。它作为框架层升级/回归观察项保留，不能被描述成“控制台永久零错误”。

## 5.1 2026-08-23 必须保留的验收记忆

- 高覆盖率不会验证页面事实；M17 已通过严格的一对一文献绑定与官方事实/工程推断隔离保证教学可信度。
- `test_harness_traps.py` 覆盖了边界防御（NaN/Inf/非法负数/未知枚举）与多维度变异测试。
- M17 必须持续区分论文经验、官方事实、规则模拟、作者推断和工具级安全示例。
- 浏览器实测确认：全站 18 页无 `stException`，所有蓝图目标存在且唯一，Section A/B/C/D/E 导航点击均能精准聚焦。

## 6. 测试设计记忆

测试必须优先验证可观察语义，而不是实现细节或宣传文案：

- 数学算子：已知解析值、性质、不变量、数值梯度、参考实现交叉核对、异常输入。
- 随机算法：固定随机源之外，还要做多种子统计断言和置信范围，不能断言单次曲线必然单调。
- 教学内容：用结构化 claim/source/evidence 注册表校验，不依赖“某句话不存在”的脆弱字符串测试。
- UI：冒烟测试只证明不崩溃；导航、视觉反馈、控件组合、暂停/播放和无障碍需要浏览器行为断言。
- 性能：阈值要有硬件与环境基线，避免把易波动的墙钟测试称为算法正确性。
- 本地 DevOps 质量门禁（Pre-commit + `scripts/devops.py gate`）必须实际执行 format、lint、type、unit/property、coverage 和页面行为门禁；不得在成功输出中写“零缺陷认证”。

## 7. 版本化验收规则

每次教学内容或核心实现变更后必须执行本地 DevOps 质量门禁：

```powershell
uv run python scripts/devops.py gate
```

或分步执行：

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
git diff --check
uv run pytest --cov=nn_core --cov=datasets --cov-branch --cov-report=term-missing -q
```

此外必须人工抽查结论边界、引用映射、图表证据标签和零基础教学路径。浏览器门禁至少验证导航目标、聚焦清理、重复点击、reduced-motion CSS、播放/暂停、暂停状态与布局稳定。通过后只能说“本版本通过既定教学可信度审计”；仍不得使用“永久完全正确”或“零缺陷”。

## 8. 权威核验锚点

- [Attention Is All You Need（NeurIPS 2017）](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Chinchilla compute-optimal scaling](https://arxiv.org/abs/2203.15556)
- [DeepSeek-R1（2025）](https://arxiv.org/abs/2501.12948)
- [MMLU](https://arxiv.org/abs/2009.03300)、[HellaSwag](https://arxiv.org/abs/1905.07830)、[GSM8K](https://arxiv.org/abs/2110.14168)
- [HELM：透明、可复现、多指标评估](https://crfm.stanford.edu/helm/index.html)
- [PyTorch gradcheck 限制说明](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html)
