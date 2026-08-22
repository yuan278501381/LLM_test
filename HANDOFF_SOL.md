# NN Playground 教学可信度整改交接清单（Sol-中执行）

> 交接日期：2026-08-22
>
> 工作目录：`C:\repo\LLM_test`
>
> 执行者：Sol-中；完成后由 Codex 独立验收。
>
> 目标：把当前交互式教学原型收敛为“本版本可审计、可追溯、无已知误导”的教学项目。不要声称永久完全正确或零缺陷。

## 0. 执行边界

1. 先阅读 `README.md`、`PROJECT_MEMORY.md`、`CURRICULUM_MAP.md`、`dashboard/constants/course.py`。
2. 保留用户现有交互、播放/暂停、导航和视觉修改；不得通过回退功能换取测试通过。
3. 不引入 PyTorch/TensorFlow 作为运行时核心；可以在测试依赖中用 scikit-learn 或独立公式作交叉核对，但必须说明用途。
4. 不把未完成主题包装成已实现；课程地图必须区分 `已实现 / 部分实现 / 仅规划`。
5. 每完成一个任务，补测试、权威来源和结论边界；不要先批量改文案、最后再猜测试。
6. 不删除失败测试、不放宽正确性阈值、不扩大 lint/type 排除范围、不输出“ZERO-DEFECT CERTIFIED”。

## 1. 整改前审计基线（保留用于前后对照）

```text
pytest + branch coverage: 271 passed, total 90.28%
ruff check:              PASS，但 dashboard/ 被排除
ruff format --check:     FAIL，8 files would be reformatted
pyright:                 FAIL，nn_core/reinforcement.py 有 3 个返回类型错误
```

低覆盖模块：`callbacks.py 22.47%`、`clip.py 73.02%`、`model.py 75.36%`、`video.py 76.85%`。

当前测试的主要局限：

- AppTest 冒烟测试只断言页面不抛异常，不能证明教学解释、图表含义和交互正确。
- 所谓“控件全组合模糊遍历”实际上只测试首/末一个 slider 的极值和第一个 number input 的原值，没有覆盖 radio、selectbox、multiselect、按钮或组合状态。
- 教学回归多为“禁用字符串不存在”，可以被改写同义句绕过，也无法证明正确主张已经出现。
- API 契约测试只验证 `hasattr/callable`，不验证签名、返回值和行为。
- 性能测试是单机墙钟阈值，没有预热、重复统计或硬件基线，不应被称为算法正确性门禁。
- 覆盖率高不代表断言充分；部分测试只验证 shape、长度或单次趋势。

## 2. P0：发布阻断项（全部完成后才允许申请验收）

### P0-1 建立“教学主张—证据—来源”注册表

目标：让每个关键教学结论可追溯，而不是页面只挂一篇相关论文。

任务：

- 扩展 `dashboard/constants/course.py`，新增结构化 `Claim`：唯一 ID、主张正文、适用条件、证据等级、直接来源、页内结果 ID、反例/局限、最后核验日期。
- 将证据等级从“整页标签”细化到图表、指标、动画和案例；同页混合真实计算与模拟时必须分别标注。
- `Reference` 增加来源类型（原始论文、官方文档、教材、综述）、作者/组织、年份、稳定标识（DOI/arXiv/官方 URL）和具体支持内容。
- 页面顶部仍可显示摘要，但每个结果附近必须显示本地证据标签和可展开的“为什么能这样解释”。
- 所有 17 页至少登记核心公式、核心图表、历史结论、失败模式各一条主张。
- 当前 `PAPER_REPRODUCTION` 继续禁止使用。

验收断言：

- 所有 claim ID 唯一，来源非空，适用条件和结论边界非空；
- 页面引用的 claim/result ID 必须真实存在；
- `SIMULATION`/`ARCHITECTURE_ONLY` 结果不得使用“训练证明、真实能力、论文复现”等结论；
- 链接检查区分 `通过 / 重定向 / 站点拒绝自动访问 / 确认失效`，不得把 403 自动判为文献不存在。

### P0-2 修正 M00 数学基础

文件重点：`dashboard/pages/0_数学基础.py`、数学与损失相关测试。

必须修正：

- `N * Din * Dout` 不是完整 FLOPs；明确 MAC 与 FLOP 计数约定，并计入加法/偏置或改名为“乘法次数”。
- 有限差分只能提高发现导数错误的概率，不能“100% 验证无数学 Bug”；补充不可导点、步长、精度、随机性和局部检查限制。
- “相对误差 < 1e-5 说明理论导数完全正确”改为“在当前输入、精度和容差下与数值近似一致”；同时使用绝对误差与相对误差，处理近零梯度。
- 修复 M00 A 区导航目标缺失和 HUD A-D 与页面 A-E 不一致。
- 补齐 README 已承诺但页面缺失的最小概率基础：概率分布、log/exp、log-sum-exp、softmax、二元/多类交叉熵，并用可操作实验串到 M01/M07/M09。
- 增加 shape 错误示例、广播错误示例和一个需要学习者预测结果再揭示答案的小测。

权威核验参考：PyTorch `gradcheck` 官方文档明确说明精度、不可导点、重叠内存和非确定性限制；不能把一次通过视为证明。

### P0-3 修正表示学习、RNN、注意力和 GPT 的语义过度解释

#### M05 词嵌入

- 将手工预置向量明确标成 `SYNTHETIC_DATA`，禁止称为模型训练所得。
- 将“距离直接代表语义”“+1 是同义、0 毫无关联、-1 是反义”的绝对解释改为上下文、训练目标和各向异性相关的有限解释。
- `king - man + woman ≈ queen` 只能作为某些词向量中的经典现象；不能据此声称“真正逻辑类比能力”。
- 3D 投影与原空间距离分开标注，解释降维失真。

#### M06 序列记忆

- 核实热力图究竟来自 RNN 状态/梯度还是人为衰减公式；若为后者标成 `SIMULATION`。
- 把“RNN 必然彻底忘光”“完全无法利用 GPU”改为有条件结论：序列时间步依赖限制序列内并行，但 batch、矩阵运算和层级仍可并行。
- 补 LSTM/GRU 的核心门控公式或明确列为未实现，不要只引用 LSTM 论文来支撑 Vanilla RNN 的全部结论。

#### M07/M08 注意力与 Transformer

- 删除随机/未训练头“各自关注语法、语义、指代”和“深层完成语义理解”的解释。
- 关闭 `1/sqrt(d_k)` 会增加 logits 方差和饱和风险，但不会在任意输入下必然变成 0/1 或使梯度彻底消失；做维度和输入尺度控制实验。
- 明确注意力权重是计算中的混合权重，不自动等于因果解释；引用相关反例研究。
- 保留因果掩码的定义，但补充 fully-masked row 的数值策略。
- M08 当前“结论边界”和下方“深层语义理解”互相矛盾，必须消除。

#### M09 Mini-GPT

- 审计所谓“预训练嵌入”：若为人工注入或固定规则，重命名并标证据等级。
- Temperature 只改变分布锐度/随机性，不直接等同创造力、事实性或质量。
- Top-k 是截断并重归一化，不保证被删除 token 都“无关”。
- 明确模型是否真正训练、训练数据、损失、验证集和生成循环；未训练输出不得作为语言能力案例。

### P0-4 修正多模态、预训练、评估与强化学习边界

#### M10-M12

- `get_pretrained_clip_data` 实际为手工正交基底加噪，重命名为 synthetic demo；保留兼容别名时给弃用说明。
- CLIP 双塔随机权重前向不等于图文对齐；相似度演示与真实模型计算分开展示。
- `NextFramePredictor` 是未训练随机 MLP，不能称为会推演未来帧；只可作为输出头结构示意，或补真实小型训练与独立验证。
- 余弦 schedule 末端 `alpha_bar` 被裁到 `1e-5`，不是信号“完全衰减为纯高斯噪声”；展示残余信号系数和 SNR。
- 始终区分 DDPM 前向加噪、训练噪声预测器和反向采样三件事。

#### M13-M14

- Chinchilla 是特定模型族、数据和计算范围内拟合的经验 compute-optimal 关系，不是对任意模型“严格证明 D≈20N”。
- 为 MLM、CLM、BPE、MAE、对比学习、PPO、DPO、LoRA 分别补直接原始来源；一篇 Chinchilla 或 InstructGPT 不能覆盖整页。
- 模板回答、预设质量分和公式曲线标为 simulation，不称为模型真实质变。
- PPO/DPO/LoRA 若只实现目标函数片段，列出相对完整训练省略的采样、reference policy、KL、优化循环和数据协议。

#### M15 评估

- 自建中文题目不得直接命名 `Mini-MMLU`、`Mini-HellaSwag`、`Mini-GSM8K`，除非能给出原数据 item ID、许可证和翻译协议；建议改为 `MMLU-style teaching quiz` 等明确名称。
- `BenchmarkTask.metric` 当前被 `run_task` 忽略，必须按 `accuracy/f1` 调度，未知 metric 报错。
- `compute_perplexity` 增加形状、有限值、是否为 log-prob、mask/有效 token 约定；不得静默把正 log-prob 裁为 0 或把极低概率统一裁到 -50 而不声明。
- accuracy/F1 对长度不一致、空输入、非法类别提供清晰契约；与 sklearn 或手算基准交叉核对。
- 页面 PPL 曲线是公式+噪声模拟，不是模型 token log-prob；不得和 `compute_perplexity` 的真实计算混在一个结论中。
- 评估章节补提示模板、tokenizer、few-shot 设置、数据污染、置信区间、多随机种子、校准、公平性与安全多指标。

#### M16 强化学习

- DeepSeek-R1 论文发布日期是 2025；区分 R1-Zero 的纯 RL 与 R1 的 cold-start + 多阶段训练，不得把 R1 整体称为纯 RL。
- `GRPORunner` 目前只计算组内标准化优势并生成预设 sigmoid/幂函数曲线，不是 GRPO 优化器，更不是语言模型训练；重命名为 simulation 或实现真实最小策略更新。
- “CoT 自动暴涨、准确率到 94%、Aha Moment”是手工公式/模板，必须标 `SIMULATION`，不能说“见证自发涌现”。
- 删除“RL 是 AI 的终极武器”“核心基石”等宣传性断言。
- 页面 `OPTIMAL PATH` 来自有限轮 Q-Learning 后的贪心策略，未验证时只能叫 learned greedy path；检测循环、截断、陷阱和是否到达终点。
- 将 Bellman 解称为“当前有限、确定性、已知转移 GridWorld MDP 下的数值参考解”，不是一般环境“绝对真值”。
- 增加 Q-Learning 多随机种子成功率、Bellman residual、策略回报与动态规划最优回报差距。

### P0-5 修正核心实现契约

- `nn_core/rope.py`：禁止通过维度大小猜 `(B,S,H,D)` 或 `(B,H,S,D)`；显式指定布局或统一一个布局，并分别测试。保留相对位置内积恒等式；删除“任意向量内积随距离必然单调衰减”的说法。
- `nn_core/attention.py`：校验 Q/K/V 维度、mask 可广播性、空序列和 fully-masked row；定义并测试数值行为。
- `nn_core/clip.py`：校验 similarity matrix、temperature > 0、批大小与对角配对；增加已知 logits 的 loss 手算测试；移除函数内部全局 `np.random.seed`。
- `nn_core/world_model.py`：校验 `num_steps`、beta、t 边界、噪声 shape；不要把越界 t 静默截到末端，除非 API 明确如此。
- `nn_core/evaluation.py`：修复 metric 调度、输入校验、PPL contract 和数据集命名。
- `nn_core/reinforcement.py`：修复 Pyright 返回类型错误；使用局部 RNG；校验 action、超参数和环境类型，未知 `grid_type` 不得悄悄回退为 simple。
- 全项目将全局 `np.random.seed` 迁移为显式 `numpy.random.Generator` 或可注入 seed，避免页面/测试互相污染。

### P0-6 重建引用体系

每页至少满足：核心公式原始来源、历史节点原始来源、局限/反例来源、实现差异说明。重点补充：

- M06：Vanilla RNN/梯度困难与 LSTM/GRU 各自来源；
- M07：Bahdanau attention、Transformer scaled dot-product、注意力解释局限；
- M08：LayerNorm、残差网络、Pre-LN 的直接来源；
- M10：卷积/ViT/CLIP 分开；
- M11：DSP/STFT/Mel 来源与 Whisper 架构来源分开；
- M13：BERT、GPT、CLIP、MAE、BPE、Chinchilla 分开；
- M14：PPO、InstructGPT、DPO、LoRA 分开；
- M15：PPL 定义、MMLU、HellaSwag、GSM8K、HELM 分开；
- M16：Sutton & Barto、Q-Learning、REINFORCE、DeepSeekMath/GRPO、DeepSeek-R1 分开。

首选核验锚点：

- Transformer：[NeurIPS 2017 原文](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- RoPE：[RoFormer](https://arxiv.org/abs/2104.09864)
- GQA：[Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- 注意力解释边界：[Attention is not Explanation](https://arxiv.org/abs/1902.10186)
- Chinchilla：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- DeepSeek-R1：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- MMLU：[arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- HellaSwag：[arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
- GSM8K：[arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- HELM：[Stanford CRFM](https://crfm.stanford.edu/helm/index.html)

### P0-7 让质量门禁名实相符

- 修复所有 `ruff format --check`、Pyright 错误。
- 不再整体排除 `dashboard/`；若必须分阶段，引入最小、带原因和到期任务的 per-file ignore。
- Pyright 至少覆盖 `nn_core/`、`datasets/`、`dashboard/components/`、`dashboard/constants/` 和测试辅助代码；逐步纳入页面。
- `scripts/devops_ci_gate.py` 加入 `ruff format --check`、`pyright`、`git diff --check`，并与 GitHub Actions 使用同一命令。
- 删除脚本和测试 docstring 中“世界级、100%、全组合、零缺陷”等不实描述；名称必须准确描述实际覆盖。
- CI 不应重复运行同一重型页面套件；按 unit/property/content/UI 分层并保留失败定位。

## 3. P1：测试与断言重构

### P1-1 数学与核心算子

- 为 BCE/CCE、LayerNorm、Conv2D、LoRA、attention、RoPE、SwiGLU 增加有限差分或独立公式交叉核对；覆盖近零、极值、非法 shape、空输入、非有限值。
- 数值梯度同时使用 `atol/rtol`，随机采样多个点，避开或单独处理不可导点。
- 对 softmax、概率、mask、归一化、旋转正交性、causal invariance 建立性质测试。
- 对参考实现比较时固定输入和约定，不能只比 shape。

### P1-2 训练与随机算法

- 优化器测试增加二次函数之外的病态曲率、偏置修正已知值、状态隔离和非法超参数。
- 模型训练测试增加 train/validation 拆分、损失下降不是单调保证、早停恢复最佳权重、batch 尾部、shuffle/seed。
- Q-Learning 用多个 seed 报告达到目标的比例和回报差距，不要求每条随机曲线单调。
- 扩散测试验证 beta/alpha/alpha_bar 不变量、闭式均值方差的统计性质和固定噪声可复现性。

### P1-3 评估与数据

- Accuracy/F1 与手算及 sklearn 交叉核对，覆盖类别缺失、长度不一致、空输入、二/多分类。
- PPL 覆盖 mask、padding、不同 token 数、极低概率、非法正 log-prob、NaN/Inf。
- 题库测试校验名称、来源、许可证、题目唯一性和答案；自建题只能叫 teaching/style set。
- 模拟评估使用局部 RNG，页面同一结果区域不得二次抽样造成分数和逐题答案不一致。

### P1-4 UI 与教学行为

- 重写 widget 测试：按页面元数据覆盖每个 radio/selectbox/multiselect/slider/number input/button 的代表值、边界值和关键组合；不能称穷举全部组合。
- 用真实浏览器验证导航目标存在、主滚动容器变化、聚焦动画开始/结束、重复点击、reduced-motion、播放/暂停和图表不闪烁。
- AppTest 继续作为 smoke gate，但命名为“启动无异常”，不冒充内容正确性。
- 为形成性测验验证：未答不泄露答案、答错给诊断反馈、答对解释理由、重试状态正确。
- 增加可访问性检查：键盘焦点、语义标签、颜色非唯一编码、对比度、动画减弱偏好。

### P1-5 内容测试

- 以结构化 claims 校验高风险主张，不使用大量脆弱的全文字符串快照。
- 建立限定词规则：遇到“必然、证明、最优、真实、语义、涌现、复现”等词时，必须绑定适用条件或证据。
- 检查每页先修、目标、公式、参数、观察、失败案例、结论边界、练习、参考均非空且与页面组件对应。
- 自动检查 README 的页面数、课程编号、测试命令和审计状态不会过期。

### P1-6 覆盖率策略

- 保持总分支覆盖率不低于当前 90.28%，并设置关键模块最低线；优先让 `callbacks/clip/model/video` 达到有意义的 85% 分支覆盖。
- 覆盖率新增必须来自失败路径和语义断言，禁止为了数字执行代码而不验证结果。
- 可试行 mutation testing 或手工变异：反转更新符号、移除 mask、交换 label、跳过 bias correction，确认测试必然失败。

## 4. P2：从零到深入的知识架构补齐

P0/P1 完成后再实施，不允许借扩课逃避纠错。

- M00 拆出最小数学链：数与函数 → 向量/矩阵 → 导数/梯度 → 概率/log → softmax/交叉熵 → 数值稳定性。
- 补数据与实验科学：训练/验证/测试、数据泄漏、偏差—方差、置信区间、随机种子、消融实验。
- 补经典架构因果链：LeNet → AlexNet → BatchNorm → ResNet；Vanilla RNN → LSTM/GRU → Seq2Seq；Autoencoder → VAE/GAN → U-Net/DDPM。
- 补训练工程：学习率调度、AdamW、BatchNorm、checkpoint、混合精度概念、数据管线和可复现报告。
- 补现代 LLM 但不追逐名词：Encoder/Decoder、BERT/GPT、RoPE/GQA/KV cache、MoE、检索、工具使用与智能体边界。
- 补评估与责任：校准、鲁棒性、公平性、安全、隐私、数据治理、污染与可复现性。
- 每章设置诊断题、学习目标、最小实验、反例实验、形成性测验和通关标准；课程先后关系写入机器可读 DAG。

## 5. Sol-中交付格式

提交验收前提供：

1. 按 `P0-x / P1-x` 的逐项完成表，未完成项不得隐藏；
2. 修改文件清单及每个文件的教学/数学理由；
3. 每条被修正主张的“旧说法 → 新说法 → 来源 → 适用边界”；
4. 新增/修改测试清单，说明旧测试为什么不足、新断言能杀死什么错误；
5. 完整命令结果与覆盖率表；
6. 17 页浏览器人工检查记录和关键截图；
7. 引用链接检查结果及自动访问受限项的人工复核记录；
8. `git status --short`、`git diff --stat`、`git diff --check`；
9. 尚存风险和明确不宣称的能力。

## 6. Codex 最终验收门槛

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=nn_core --cov=datasets --cov-branch --cov-report=term-missing -q
git diff --check
```

人工验收将抽查：

- M00、M05-M09、M12-M16 的已知风险是否真正消失；
- 代码计算、页面图表、证据标签和解释是否一致；
- R1/R1-Zero、GRPO、Chinchilla、benchmark 名称是否准确；
- 引用是否直接支持主张，而非仅“主题相关”；
- 测试是否能在故意注入典型错误时失败；
- 初学者能否沿先修—实验—反馈—反例—总结形成闭环；
- 是否为通过门禁而隐藏问题、扩大排除、降低阈值或删除测试。

P0 全部关闭、自动化与人工审校同时通过后，只能声明“该版本通过既定教学可信度审计”，不能声明“所有知识永久完全正确”。

## 7. 2026-08-22 执行记录

| 清单 | 状态 | 可复核交付 |
|---|---|---|
| P0-1 主张—证据—来源 | 已完成 | 68 个结构化 claim/result ID；17 页各 4 个常显结果证据卡；形成条件、局限、日期、直接来源 |
| P0-2 M00 | 已完成 | 乘法次数/FLOP 边界、有限差分限制、shape/广播反例、概率/log-sum-exp/softmax/CE、小测 |
| P0-3 M05-M09 | 已完成 | 合成嵌入与投影边界、RNN 状态相似度、未训练注意力/Transformer/GPT、temperature/top-k 限定 |
| P0-4 M10-M16 | 已完成 | synthetic CLIP、未训练世界模型头、DDPM SNR、Chinchilla/PPO/DPO/LoRA、style 题集、R1/R1-Zero/GRPO 边界 |
| P0-5 核心契约 | 已完成 | attention、RoPE、CLIP、world model、evaluation、reinforcement、训练与回调失败路径/局部 RNG |
| P0-6 引用 | 已完成 | 43 个去重直接来源；自动访问分类与受限项替代锚点记录见项目记忆 |
| P0-7 质量门禁 | 已完成 | dashboard 纳入 Ruff；format、Pyright、diff、分支覆盖率统一进脚本；删除夸大门禁文案 |
| P1 数学/训练/评估 | 已完成 | 数值梯度、性质、sklearn 交叉核对、多 seed RL、扩散统计、尾批/验证/早停恢复、失败路径 |
| P1 UI/内容 | 已完成 | 控件代表值而非伪称穷举；结构化 claim 测试；形成性测验未答/错答/改答；浏览器导航与播放器审计 |
| P1 覆盖率 | 已完成 | 322 passed；`nn_core + datasets` 分支覆盖率 95.08%；`model.py` 95.98% |
| P2 知识架构 | 已完成当前 17 课范围 | 机器可读无环 DAG；每课诊断、最小实验、反例、形成性评价、通关标准；未实现主题继续标为部分实现/仅规划 |

不宣称项：并未把“仅规划”的架构包装成可运行课程；没有任何页面标记 `PAPER_REPRODUCTION`；自动化门禁不等于永久知识正确或专家共识终审。

框架观察项：浏览器控制台可记录 Streamlit iframe 自动尺寸脚本的无应用堆栈 `MutationObserver` 日志；页面无 Streamlit 异常，导航与播放器行为通过抽查。后续升级 Streamlit 时应重新核验，不能将当前状态宣传为“控制台永久零错误”。
