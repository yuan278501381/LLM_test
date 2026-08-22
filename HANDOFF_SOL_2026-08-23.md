# NN Playground 第二阶段教学可信度整改清单（交给 Sol-中）

> 审计日期：2026-08-23
>
> 审计对象：当前 `main`（包含 M17「工程陷阱与 Harness」）
>
> 当前自动化基线：343 passed；`nn_core + datasets` 分支覆盖率 95.18%；Ruff、格式、Pyright、差异卫生通过。
>
> 发布结论：**工程门禁通过，但教学内容审计不通过。M17 属于发布阻断项。**

## 0. 执行原则

1. 不用换词掩盖问题。所有百分比、事故指标、因果解释和“最佳实践”都必须能追溯到直接来源或明确标成教学假设。
2. 规则生成的曲线只能叫 `SIMULATION`，不能写成实测 PPL、真实准确率、生产成功率或官方事故数据。
3. 论文观察不自动成为所有模型的结构定理；产品事故也不能被扩展成行业普遍比例。
4. 修复必须同时覆盖核心实现、页面、课程注册表、知识库、测验、测试和 README/项目记忆。
5. 未完成 P0 前，不得恢复“当前版本通过教学可信度审计”的状态。

## 1. 已确认的发布阻断事实

| ID | 当前问题 | 为什么会误导 | 正确处理方向 |
|---|---|---|---|
| F17-01 | 页面把“AI 系统 80% 取决于 Harness”设为标准答案 | 无权威来源，也不是可跨任务量化的定律 | 删除比例；改成“模型与外围系统共同决定结果”，列出可测维度 |
| F17-02 | `35% / 82% / 94%` 被展示为不同 Agent 循环的成功率 | 数字是代码常量，不是实验或论文结果 | 标成可调教学假设，或建立真实、可复现的任务集评测 |
| F17-03 | Claude Code 事故页展示 54.2→91.8 等准确率和延迟 | Anthropic 官方文章未公布这些数字；属于伪造实测数据 | 删除指标卡；只呈现官方确认的日期、影响范围、修复与已公开的 3% eval 下降 |
| F17-04 | Anthropic 引用 URL 错误 | 当前注册 URL 不是官方文章地址 | 改为 `https://www.anthropic.com/engineering/april-23-postmortem` |
| F17-05 | Attention Sink 曲线宣称无 Sink 时 PPL 必然指数爆炸至 `10^4+`，有 Sink 时“100% 平稳” | 曲线由手写指数函数生成；StreamingLLM 是特定模型/协议的经验结果 | 页面标为规则模拟；不得给出普遍阈值；增加模型、窗口、sink 数和数据协议边界 |
| F17-06 | 把 Attention Sink 归因为“Softmax 和为 1，所以初始 token 自然成为 sink” | 归一化是必要背景，不足以推出位置、数量和强度 | 改为论文观察及假说；区分经验现象、机制解释和未证明因果 |
| F17-07 | Lost-in-the-Middle 的 U 曲线与 Rerank 90%+ 是代码预设 | 原论文结果依赖模型、任务、上下文和位置；Rerank 不保证修复 | 使用论文数据需逐项注明协议；否则只显示无数值承诺的教学示意 |
| F17-08 | Reversal Curse 用固定 98.5%/6.2% 和真实人物模板模拟 | 不是模型推理；人物关系表还存在方向错误风险 | 使用虚构实体与明确 scripted 标签，或运行可复现模型实验；不得称因果概率 |
| F17-09 | “strawberry”解释声称字符被 BPE 物理消除、不开 CoT 就无法看见 | tokenizer 依模型而异，token 表征仍可编码字符信息，字符任务可通过多种策略完成 | 接入明确 tokenizer 或使用教学 tokenizer；展示 tokenization，不推断模型必然能力 |
| F17-10 | 前导空格 Token ID 和向量范数是任意常量 | 被呈现为 tokenizer 事实 | 删除伪造 ID/范数，或使用指定 tokenizer 的真实输出并记录版本 |
| F17-11 | 正则表达式黑名单被称为 Prompt Injection 防御与“安全沙盒校验” | 极易绕过；也没有权限、参数 schema、sandbox 或人工批准 | 降级为“玩具字符串过滤器”；补 threat model、允许列表、权限与执行隔离 |
| F17-12 | AST 骨架压缩被称为无损、Prompt cache 100% 命中 | 函数体、控制流和局部变量可能正是任务所需信息；缓存命中受产品协议影响 | 改为有损策略并展示反例；缓存仅说明稳定前缀的潜在收益 |
| F17-13 | M17 重复渲染学习契约和第二套测验 | 初学者看到两套通过标准，且自定义测验把无来源 80% 当正确答案 | 只保留统一形成性测验组件；答案必须可由本页证据推出 |
| F17-14 | M17 的四条 Claim 仍由通用模板自动生成 | 公式、曲线、事故和失败模式没有各自绑定直接来源 | 为 M17 手工建立 result-level claims，逐图绑定证据等级和来源 |
| F17-15 | 实际证据卡把规则模拟标成“真实计算”，并让 Anthropic 事故文章支持 Attention Sink 失败结论 | 证据等级与来源主题均错配，证据卡本身会增强错误可信感 | 逐卡重建 claim；测试 source 必须直接支持 statement，禁止按首/末引用自动分配 |

权威核验锚点：

- StreamingLLM / Attention Sinks：<https://arxiv.org/abs/2309.17453>
- Lost in the Middle（正式 TACL 版本）：<https://aclanthology.org/2024.tacl-1.9/>
- Reversal Curse：<https://arxiv.org/abs/2309.12288>
- Anthropic 2026-04-23 官方复盘：<https://www.anthropic.com/engineering/april-23-postmortem>

## 2. P0：M17 内容和实现整改

### P0-17A 事故资料逐句核对

- 以 Anthropic 官方文章为唯一事故事实主来源，建立 `official_fact / teaching_inference / synthetic_value` 三类字段。
- 准确保留官方确认事实：3 月 4 日 effort 默认值变化、4 月 7 日回退；3 月 26 日 idle-session thinking 清理 bug、4 月 10 日 v2.1.101 修复；4 月 16 日简短提示上线、4 月 20 日回退；全部问题在 v2.1.116 解决。
- 不写官方未公布的总体错误率、准确率和延迟。官方只报告某一扩展评测出现 3% 下降时，必须注明它不能代表所有编码任务。
- 测试必须断言页面不存在虚构的 54.2、91.8、28.5、89.4、42.0、93.5 等事故指标。

### P0-17B 重新定义所有模拟器契约

- `AttentionSinkSimulator`、`LostInTheMiddleSimulator`、`ReversalCurseEngine`、`LoopEngineeringEngine` 更名或 API 文档明确为规则模拟。
- 所有函数改用局部 `np.random.Generator`，不得调用 `np.random.seed` 污染全局状态。
- 增加有限值、正整数、范围、空输入、未知枚举和极端预算校验。
- `num_sink_tokens` 必须真正影响输出；否则删除该参数。
- 输出携带 `evidence_level="SIMULATION"`、`assumptions`、`not_measured` 和 `source_scope`。
- 页面不得把模拟输出字段命名为 `BUGGY ACC`、`真实 PPL` 或“官方准确率”。

### P0-17C Attention Sink / 长上下文

- 区分：论文观察到的 attention sink、StreamingLLM 保留初始 KV 的方法、页面人为生成的教学曲线。
- 不从 Softmax 归一化单独推出 sink；不保证固定 4 个 token 对所有模型最优。
- Lost-in-the-Middle 使用正式 TACL 引用，解释它是多模型、特定 QA/检索协议的经验结果，不是所有上下文必为 U 型。
- Rerank、首尾放置与上下文压缩都要给失败反例，例如重排损坏顺序依赖、关键证据跨段、检索器漏召回。

### P0-17D Tokenizer / Reversal Curse

- tokenizer 示例必须记录 tokenizer 名称、词表版本、实际 encode 输出；教学 tokenizer 需明确不可外推到 GPT/Claude/Llama。
- 字符计数实验区分“输入 token 化方式”与“模型行为”，不得声称字符信息被物理消除。
- Reversal Curse 使用虚构关系和明确训练协议；固定概率只能标为 UI 示例值，不能当模型概率。
- 解释原论文的重要边界：训练后反向泛化失败不等于给定上下文时不能推导反向关系。

### P0-17E Agent Harness 安全边界

- 将正则过滤器命名为 toy detector，并加入大小写、Unicode、拆分字符串、编码、嵌套对象和良性误报反例。
- 不返回 `EXECUTION_PERMITTED [HEALTHY]`；字符串未命中只能表示“此检测器未发现已知模式”。
- 教学架构补全：参数 schema、工具 allowlist、最小权限、路径约束、沙箱、超时、幂等键、人工批准、审计日志和输出验证。
- 禁止把 `git restore` 作为通用原子回滚；解释它可能覆盖用户改动，示例使用隔离工作树或快照。

### P0-17F 页面教学闭环

- 删除 80/20 标准答案、营销式“黄金架构”“物理盲区”“彻底防御”。
- 每张图紧邻显示：这是计算、模拟、官方事实还是作者推断；输入、公式、随机源和不可推出的结论。
- 统一使用课程注册表的测验与学习契约，删除重复组件。
- 初学者路径改为：观察现象 → 识别证据类型 → 找反例 → 设计真实评测 → 得出有限结论。

## 3. P0：全站残余高风险表述

- M01：移除“Blobs 必然 100%”“Minsky 证明所有非线性流形绝对无法”等扩大结论；区分 XOR 表示限制、特定数据几何与当前训练结果。
- M04：删除更新/权重比固定在 `10^-3` 才健康的通用规则。
- M07：Softmax 行和应写“在浮点容差内为 1”，不能写严格 100%；RoPE 相对位置结论写清使用的子空间与变换条件。
- M11：波形振幅不直接等于主观响度；补 RMS/声压、频率响应、响度模型与单位边界。
- M13：所有 Chinchilla “最优”均限定到论文拟合范围；MoE 不称普遍算力最优。
- M15/知识库：PPL 不是无偏、通用能力指标；Evaluation Harness 不能杜绝主观偏差，需讨论题集选择、污染、提示敏感性、置信区间和多次采样。
- 知识库：收紧 BERT/CLM/DPO/L1 等“赋予能力、最主流、显存减半、精确归零”无条件措辞。

## 4. P1：测试覆盖面与断言重构

当前测试的优点：已有数值梯度、非法输入、统计性质、sklearn 交叉核对、18 课注册契约、页面启动和控件代表值检查。

当前不足及任务：

1. `test_harness_traps.py` 主要断言手写规则自身成立，属于循环自证；改为验证证据标签、参数契约、局部 RNG、反例和“不作错误外推”。
2. 事故测试不得只断言 `fixed > buggy`；应对照官方事实表，并拒绝没有来源的 measurement 字段。
3. Attention/Lost-middle/Reversal 的科学结论不能由模拟器单测证明；新增 content test，保证页面明确写出 `SIMULATION` 和协议边界。
4. 为 `AgentHarnessGuard` 加变异测试：拆分/编码绕过、Unicode、嵌套参数、误报、不同工具相同副作用；测试必须证明 toy detector 不能授权执行。
5. 为所有新增 API 加无效范围、NaN/Inf、空值、未知枚举、不可变输入和全局 RNG 隔离测试。
6. 引入手工 mutation matrix：反转优化符号、移除 mask、伪造引用、删掉证据标签、把模拟改名为真实，测试必须失败。
7. Coverage 当前只统计 `nn_core/datasets`；增加 dashboard 课程注册与纯组件逻辑覆盖报告，但不要用覆盖率代替浏览器行为测试。
8. Pyright 当前排除 pages/tests/scripts 且为 basic；至少先纳入 M17、测试辅助和 CI 脚本，再逐步提高严格度。
9. 浏览器测试补键盘导航、焦点可见、reduced-motion、颜色非唯一编码、移动/窄屏、图表替代文本与控制台错误。
10. CI 增加引用 URL/稳定标识检查和文档数字同步检查；网络受限项需记录人工替代核验，不应直接判错。

## 5. P2：从零到深入的课程缺口

课程地图仍明确存在大量“部分实现/仅规划”，不能宣传为完整 AI 课程。按下列顺序建设：

1. 实验科学：训练/验证/测试拆分、数据泄漏、置信区间、统计功效、随机种子和消融。
2. 经典视觉链：LeNet → AlexNet → BatchNorm → ResNet，逐代瓶颈与同协议消融。
3. 序列链：Vanilla RNN → LSTM/GRU → Seq2Seq/teacher forcing → attention。
4. 生成模型：Autoencoder → VAE/GAN → U-Net/DDPM，补可训练去噪和逆向采样。
5. 训练工程：调度器、AdamW、BatchNorm、checkpoint、混合精度概念、可复现报告。
6. 现代 LLM：Encoder/Decoder、BERT/GPT、MoE、RAG、工具调用与 Agent 责任边界。
7. 评估与责任：校准、鲁棒性、公平性、隐私、安全、数据治理、污染与不确定性。

每个正式主题必须有先修、推导、最小实现、控制变量实验、失败案例、局限、直接来源、形成性测验和可判定通过标准；只有概念卡不算实现完成。

## 6. Sol-中提交材料

1. `F17-xx / P0-xx / P1-xx` 逐项完成表。
2. 每个修正结论的“旧说法 → 新说法 → 直接来源 → 适用条件 → 反例”。
3. 删除或保留每个模拟数字的理由；保留者必须公开生成公式和证据等级。
4. 新增测试说明：旧断言为何会放过错误，新断言可杀死什么变异。
5. 343 项基线之上的完整测试、分支覆盖率与关键模块覆盖率。
6. M00-M17 共 18 个课程页面的浏览器行为记录。
7. 47 个去重引用的可达性、稳定标识和 claim 支持关系报告。
8. 尚存风险与未实现课程清单，不得隐藏。

## 7. Codex 验收门槛

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=nn_core --cov=datasets --cov-branch --cov-report=term-missing -q
git diff --check
```

自动化通过后，Codex 仍会逐句抽查 M17、随机选择至少 6 个旧课程高风险解释、核对所有 M17 原始资料，并在真实浏览器验证 18 页。只有内容与工程同时通过，才能恢复“本版本通过既定教学可信度审计”。

## 8. 2026-08-23 浏览器复核记录

- M17 页面可启动，未出现 Streamlit `stException`。
- 页面确实渲染 4 张 result evidence 卡，但其中 2 张把规则内容标成“真实计算”，失败模式卡错误引用 Anthropic 事故文章。
- 页面同时出现 2 个 `LEARNING CONTRACT`、统一 `FORMATIVE CHECK` 和一套额外自定义测验，教学闭环重复。
- 页面可见“事故期准确率”指标，并出现 13 处包含 `80%` 的文本节点；这不是隐藏代码问题，而是学习者会直接看到的内容。
- 注册的错误 Anthropic URL 在实际页面生成 4 个链接。
- 控制台仍有既有 Streamlit iframe `MutationObserver` 框架日志；需在升级/验收时复查，但它不是本轮事实错误的来源。
