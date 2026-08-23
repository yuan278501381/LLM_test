# NN Playground AI 科学性综合审计与一次性整改清单

> 审计日期：2026-08-23
>
> 审计对象：`main` 提交 `1d29b33`
>
> 审计目的：本项目将被学习者用作“从零到深入理解 AI，并为转型 AI 工作做准备”的长期课程，因此验收标准不是“页面能运行”，而是公式、实现、图表、测验、引用、结论边界和课程覆盖均不得系统性误导。
>
> 当前结论：**工程质量门禁通过；AI 科学教学验收不通过。当前版本只能作为可视化辅助课程，不能作为唯一主教材。**

## 0. 本文档如何使用

1. 本文档列出提交 `1d29b33` 经多轮审计发现的**全部当前已知问题**，已剔除确认修复的问题。
2. “全部已知”不等于承诺不存在尚未发现的问题。科学审计必须允许后续发现新问题，禁止使用“永久完全正确、零缺陷”等措辞。
3. P0 是发布与教学阻断项；P1 是重要科学性、教学表达和测试问题；P2 是相对于“从零到 AI 就业”目标的课程建设缺口。
4. 执行者必须按 ID 提交“旧实现/旧说法 → 新实现/新说法 → 直接来源 → 适用条件 → 反例 → 测试”的闭环，不能只换同义词。
5. 自动化测试通过只能证明既有断言通过，不能证明断言本身科学正确。

## 1. 当前工程基线与审计事实

本轮实测结果：

```text
commit:                    1d29b33
branch:                    main，与 origin/main 同步
worktree:                  clean（写入本审计文档前）
ruff check:                PASS
ruff format --check:       PASS，115 个 Python 文件
pyright:                   PASS，0 errors / 0 warnings
pytest:                    364 passed
branch coverage:           95.01%（nn_core + datasets）
Chromium E2E:              PASS（当前仅覆盖 M17 导航链路与首页 HUD 清理）
deployment idempotency:    5 passed
seven-stage gate:          PASS，252.373s
```

额外数值反例：

```text
BCE(p=-5e-7, y=0):        forward≈1e-12，解析梯度≈0.5，中心差分梯度=0
BCE(p=0.9, y=2):          forward=0，解析梯度≈-12.22，中心差分梯度=0
BCE(p=NaN, y=1):          forward=0，backward=NaN
BCEWithLogits(z=Inf,y=1): forward=0
CCE(p=[2,-1], y=[1,0]):   forward=-0.693147（交叉熵不应为负）
```

## 2. 已关闭问题：禁止回归，不要重复改坏

1. M01 当前训练链已统一为 `Dense → Sigmoid → BinaryCrossEntropy → P>=0.5`，概率场和决策线使用同一模型定义。
2. BCE 已不再对明显越界概率静默截断，并新增稳定的 `BCEWithLogitsLoss`；但完整输入契约仍未完成，见 P0-CORE。
3. M13 已区分 Chinchilla Approach 3 动态比例与 70B/1.4T 的 20:1 论文基准点。
4. M14 LoRA 主指标已从“总显存节省”收窄为“矩阵可训练参数减少”。
5. KV-Cache 已不再宣称单步 O(1) 或全序列 O(N)。
6. LoRA 行向量约定公式已改为 `(xA)B`。
7. 注意力权重已不再直接等同因果解释。
8. PPL 已注明 tokenizer 依赖且不直接代表推理、事实性或对齐能力。
9. L1/L2/weight decay、数学卷积/互相关、ASR/视觉 Transformer 等上一轮关键混淆已有明显收窄。
10. M17 官方事故事实、规则模拟标签、Anthropic 官方引用、导航延迟挂载和 HUD 生命周期问题已完成上一阶段整改。

## 3. P0：核心数学与实现阻断项

### P0-CORE-01 BCE 概率输入契约仍不闭合

位置：`nn_core/losses.py:113-143`

- 文档声称严格接受 `[0,1]`，实现却设置 `eps_tol=1e-6`，轻微越界值会被接受并截断。
- 截断后缓存截断值，backward 返回截断点导数，但 forward 对原始输入在截断区是常数，解析梯度与数值梯度不一致。
- 未验证 shape、空数组、dtype、NaN/Inf 和二元标签。
- `max(0.0, loss)` 会把非法标签或 NaN 造成的异常损失伪装成 0。

整改：严格验证非空、有限值、shape、浮点类型、概率域和标签；删除 `max(0, loss)`；训练主路径优先使用 logits 稳定公式。新增轻微越界、端点、NaN/Inf、非法标签、广播、空数组和中心差分断言。

### P0-CORE-02 BCEWithLogitsLoss 仍会吞掉非法状态

位置：`nn_core/losses.py:146-192`

- `NaN/Inf` logits 可返回 0 或运行时警告，而不是报错。
- 未验证标签、shape 和有限值。
- 非法标签可产生负 loss，随后被 `max(0, ...)` 隐藏，但 backward 仍返回非零梯度。

整改：与概率版共用统一验证器；删除非负截断；对极大但有限 logits 验证稳定性，对非有限 logits 显式报错。

### P0-CORE-03 CategoricalCrossEntropy 缺少概率单纯形契约

位置：`nn_core/losses.py:195-234`

- 不检查非负概率、行和、标签、shape 和有限值。
- `safe_log` 对非法负概率做裁剪，backward 却按原值求导。
- 非法 `[2,-1]` 可产生负交叉熵。
- 注释声称 `model.py` 自动处理 Softmax+CCE 融合梯度，但训练循环没有专门融合分支。

整改：概率版严格验证概率单纯形；删除错误注释；新增明确的 logits 版交叉熵并用 log-sum-exp；与独立参考实现对照 forward/gradient。

### P0-CORE-04 二分类准确率把任何单输出都当概率

位置：`nn_core/model.py:381-395`

`_count_correct` 只凭标签 shape 使用 0.5 阈值，无法区分概率、logit、Tanh 输出或回归值。若使用 `BCEWithLogitsLoss`，logit 阈值应为 0。

整改：由任务/输出契约显式配置 `binary_probability`、`binary_logit`、`multiclass_probability`、`multiclass_logit`、`regression`，禁止仅靠 shape 猜语义。

## 4. P0：形成性测验的“正确答案”包含错误

当前 `tests/test_pedagogy.py` 只验证题目数量、选项唯一和交互流程，不验证答案事实。

### P0-QUIZ-01 M04：L1 稀疏性与 AdamW

位置：`dashboard/constants/course.py:1081-1089`

- 普通 `lambda*sign(W)` 次梯度下降不保证把权重精确截断为 0；精确稀疏通常需近端梯度/软阈值。
- “Adam 中必须采用 AdamW”过度绝对；正确结论是自适应优化器中耦合 L2 与解耦 weight decay 不等价。
- “大权重维度被过度惩罚”的方向性解释不是可无条件推出的结论。
- 本题与 M04 页面正文自相矛盾。

### P0-QUIZ-02 M06：RNN 梯度条件逻辑错误

位置：`dashboard/constants/course.py:1102-1111`

- `tanh'<1` 本身不足以推出梯度必然衰减；必须分析完整时间雅可比乘积。
- `sigma_max(W_hh)>1` 也不自动保证爆炸。
- 转置取决于行/列向量约定，题目应先声明约定。

### P0-QUIZ-03 M08：Pre-LN 被写成无损梯度保证

位置：`dashboard/constants/course.py:1124-1133`

- Xiong et al. 分析的是特定假设下初始化阶段的期望梯度与实验表现。
- 论文未证明所有深度、架构和训练阶段都有“无损高速公路”。
- “不加 warmup 初始步直接发散”强于论文的“不稳定”结论。

### P0-QUIZ-04 M11：Mel 公式不是精确耳蜗模型

位置：`dashboard/constants/course.py:1157-1166`

- Mel 是心理声学音高尺度；常用公式是近似映射，不是耳蜗基底膜的精确物理模拟。
- `n_fft` 与窗长在常见库中可独立设置。
- 时频不确定性常数需要频率单位、归一化和方差定义。

### P0-QUIZ-05 M15：PPL 与分词碎片方向被错误固定

位置：`dashboard/constants/course.py:1201-1210`

不同 tokenizer 的 PPL 不可直接比较是正确的，但“分词更碎必然使单 token 更容易、PPL 虚低”不是普遍定律；序列长度、词表和条件分布同时变化，方向不能预先固定。

### P0-QUIZ-06 M16：PPO-Clip 被写成单调改进保证

位置：`dashboard/constants/course.py:1212-1221`

- clipping 不硬性限制所有概率比或 KL。
- 它只移除部分继续推高 surrogate objective 的越界激励。
- “保障策略平稳单调改进”是错误理论保证。

整改：六道题重写答案、解释和反馈；每题增加直接来源 ID、条件与限制；建立人工语义审校表。

## 5. P0：证据链仍是结构绑定，不是语义绑定

### P0-EVIDENCE-01 通用 Claim 按引用位置机械分配

位置：`dashboard/constants/course.py:930-974`

formula/result 固定绑定 `references[0]`，history/failure 固定绑定 `references[-1]`，只能保证非空，不能保证直接支持。

已确认错配：

1. M14 的后训练历史和 reward hacking 绑定 LoRA 论文。
2. M05 的低维投影失真绑定 word2vec 论文。
3. M03 的 Adam 泛化失败绑定原始 Adam 论文。
4. M06 梯度消失/爆炸绑定 GRU 论文。
5. M08 随机权重不等于语义分工绑定 Pre-LN 论文。
6. M10 未训练滤波器不等于语义特征绑定 CLIP 论文。

整改：删除首/末引用分配；每条 Claim 显式声明 `source_ids`、支持段落/公式/表格、适用条件和不可推出结论。

### P0-EVIDENCE-02 “72 个 ID 一一绑定”是过度认证

位置：`README.md:32-42`、`PROJECT_MEMORY.md:46-55`

当前 72 个 ID 只证明来源对象存在，不证明语义正确。完成语义复核前应改成“72 个结构化 Claim 已登记，语义映射仍在审计”。

### P0-EVIDENCE-03 测验没有来源结构

位置：`dashboard/constants/course.py:97-104`、`1030-1248`

`FormativeQuiz` 没有 sources、conditions、limitations 或 last_verified。解释中出现论文名不等于可审计引用。

### P0-EVIDENCE-04 证据测试只验证非空

位置：`tests/test_pedagogy.py:40-95`

现有测试无法发现“LoRA 论文支持 reward hacking”等语义错配。需要独立、人工批准的 `claim_id -> source_id -> locator` 清单，禁止从同一模板生成期望值。

## 6. P0：模拟值仍被局部 UI 包装成真实能力

### P0-SIM-01 M14 奖励卡暗示测得价值观对齐

位置：`dashboard/pages/14_后训练工程.py:214-246`、`358-371`

`simulate_rlhf_trajectory()` 是规则轨迹，但卡片写 `RLHF REWARD SCORE`、`人类价值观对齐` 和“对齐奖励峰值”。必须直接标成模拟 objective，说明单位任意、不是人类评测、安全性或事实性指标。

### P0-SIM-02 后训练雷达是硬编码分数

位置：`nn_core/posttraining.py:16-55`、`dashboard/pages/14_后训练工程.py:375-416`

预训练/SFT/RLHF/DPO 的六维分数是作者预设，不能表现成普遍阶段规律，更不能暗示 DPO 必然全面优于 RLHF。应改为可编辑“假设情景雷达”并给出能力回归、对齐税、谄媚和 reward hacking 反例。

### P0-SIM-03 手写模板被称为真实效果

位置：`nn_core/posttraining.py:69-107`

回答是手写模板，不是同一模型经 SFT/RLHF 后的实测输出，docstring 的“真实效果演变”必须改为“教学模板对比”。

## 7. P1：逐模块残余问题

### P1-M01 单神经元

1. 页面名叫感知器，核心是 logistic neuron；开头应区分 Rosenblatt 阶跃感知器与逻辑神经元。
2. `dashboard/components/param_panel.py:31-36` 的“Blobs 可 100% 分类”不成立，高斯簇可重叠且有限样本依赖 seed。
3. “0.05~0.2 黄金区间、100% 满分、损失平滑单调”均为无条件承诺。
4. “Minsky 证明所有非线性流形绝对无法切开”扩大历史结论。
5. 固定 Sigmoid 后，`m1_act` 与不可达激活诊断分支应清理。

### P1-M02 多层网络

1. “轻松解决所有非线性难题”错误。
2. “3 层完全解开双螺旋”把一次配置写成结构保证。
3. “深度越大能力越强”缺少参数预算、优化、数据和泛化条件。
4. UAT 不保证有限宽度、训练可达性、样本效率或泛化；顶部引导与正文边界冲突。

### P1-M03 优化器

1. Adam 的 `v` 是未中心化二阶矩，不是“未中心化方差”。
2. “冲出鞍点/局部极小值”只能是类比，不是保证。
3. Momentum 两种递推只有同步重缩放定义时等价。
4. 同学习率、同轮数比较不同优化器只能叫固定协议轨迹，不能证明最优性能。

### P1-M04 正则化

1. L2 不保证边界“瞬间恢复平滑”，过强会欠拟合，平滑也非泛化充分条件。
2. L1 普通次梯度不保证精确零。
3. L2 penalty 与 weight decay 只在特定 SGD 参数化下等价。

### P1-M05 Embedding

1. “余弦优于欧氏因为高频词模长天然偏大”是过强单因果解释。
2. 低维投影距离不是原空间精确语义距离。
3. 合成 embedding 不能证明真实语义关系。

### P1-M06 RNN/LSTM

1. 修正 P0-QUIZ-02。
2. LSTM/GRU 只缓解长程路径问题，不解决所有长期依赖。
3. 课程缺少可训练 LSTM/GRU、Seq2Seq 和 teacher forcing，只能称部分覆盖。

### P1-M07 Attention

1. 若掩码实现用有限大负数，权重只能说在容差内接近 0，并测试全遮挡行。
2. 单头/随机权重图不代表训练模型语义分工。
3. “注意力不是因果解释”的修复不得回归。

### P1-M08 Transformer

1. 区分 Pre-LN、Post-LN 与 RMSNorm，并限制到具体初始化/训练协议。
2. “现代模型普遍采用”需逐模型来源，不是数学必然。
3. Pre-LN 不自动消除 warmup、梯度问题或深层表示退化。

### P1-M09 Mini GPT/KV-Cache

1. 复杂度结论需声明只聚焦注意力且把隐藏维度视为常数；完整层还有投影和 MLP 成本。
2. 区分 prefill 与逐 token decode；KV-Cache 不降低 prefill 全序列计算。
3. 随机权重 Mini GPT 输出不代表语言能力。

### P1-M10 视觉/CLIP

1. 数学卷积与互相关运算定义不同，只在可学习核参数化表达力意义上可转换。
2. 合成 CLIP 相似度不证明零样本识别能力。
3. 不对架构未公开的商业系统断言具体 CLIP 依赖。

### P1-M11 音频

1. 修正 P0-QUIZ-04。
2. 波形振幅、功率、声压级和主观响度不是同一概念。
3. Mel spectrogram 受窗、hop、谱类型、Mel 实现、log 压缩和采样率影响。
4. Whisper 使用 log-Mel 不表示所有 ASR 使用同一前端。

### P1-M12 扩散/世界模型

1. `knowledge.py:520` 的“扩散全面超越 GAN”过度概括，质量、速度、可控性和任务各有权衡。
2. 前向闭式加噪不等于逆向生成一步完成。
3. 时序一致性不证明因果世界模型；Sora 等示例需区分官方定位、研究推断和类比。

### P1-M13 预训练/Scaling Law

1. “10^23 FLOPs 约训练 7B~13B”缺 token 数、精度、架构和 FLOP 口径。
2. H100 GPU-days 的 989 TFLOPs、MFU=45% 是教学假设，需紧邻显示硬件、精度、稠密/稀疏和通信条件。
3. `predicted_loss` 仅在拟合范围内有意义，不是语言“聪明程度”。
4. “极限超训练、持续注水”应改为相对某 compute-optimal 目标的 token/parameter 权衡。
5. 历史参数/token 必须标来源、版本和约数。

### P1-M14 后训练

1. 修正全部 P0-SIM。
2. 行向量约定下增量是 `AB`，说明中不应写 `Delta W=BA`。
3. 冻结基座通常不维护其梯度/优化器状态；应区分基座权重、LoRA 状态、激活和临时缓冲。
4. QLoRA 不是唯一降内存方案；还有其他量化、offload 和分片。
5. DPO beta 不保证 KL 上限、事实性或安全性。
6. DPO 不能称所有 LLaMA/Qwen/Mistral 配方的“默认标准”。
7. SFT 不会自动把“只会续写废话”的基座变成准确理解意图的助手。

### P1-M15 评估

1. 修正 P0-QUIZ-05。
2. PPL、Accuracy、F1、奖励和人类偏好测量对象不同，不能合并为通用能力总分。
3. Harness 不能消除题集、污染、提示、顺序、解码和主观偏差。
4. 合成榜单必须始终标模拟，避免真实基准实测印象。

### P1-M16 强化学习

1. 修正 P0-QUIZ-06。
2. PPO clipping 不等于正式 trust region 或 policy collapse 防护保证。
3. GridWorld Q-Learning 不能外推语言模型 RL。
4. GRPO/R1 规则曲线必须保持 simulation 标签，不能叫能力涌现证据。

### P1-KNOWLEDGE 通用知识库措辞

位置：`dashboard/constants/knowledge.py`

1. Tanh 是值域与函数对称性以 0 为中心，不保证实际输出均值为 0。
2. Xavier 只在假设下帮助维持方差，不“确保”深层信号稳定。
3. 固定 `N(0,0.01)` 没有“>3 层必然毁掉网络”的统一阈值。
4. MLM/CLM 目标提供学习信号，不应说目标本身“赋予深层语义、ICL 或 CoT”。
5. Diffusion、SFT、DPO 的“全面超越、准确理解、默认标准”需条件化。
6. Softmax 对最大项的相对放大取决于 logit 差和温度。

## 8. P1：测试与门禁缺口

### P1-TEST-01 损失测试未覆盖真实失败路径

为 BCE/CE 增加 NaN/Inf、非法标签、广播、空输入、轻微越界、端点、概率和、非法 one-hot、极端有限 logits；合法域做中心差分或 PyTorch/JAX 对照。

### P1-TEST-02 测验测试不验证事实

`tests/test_pedagogy.py:90-113` 即使 PPO 答案错误也会通过。增加来源 ID、人工审校状态、最后核验日期；测试保证流程完整，不能自动声称科学正确。

### P1-TEST-03 Claim 测试循环自证

Claim 与期望来自同一注册模板。使用独立人工 fixture 验证映射，禁止“代码生成什么，测试断言什么”。

### P1-TEST-04 缺少跨实现数值 Oracle

核心 NumPy 算子与 PyTorch/JAX 对照 forward、gradient、shape 和极端数值，至少覆盖 BCE/CE、Softmax、LayerNorm、Attention mask、Conv2D、RNN、LoRA、DPO、PPO objective。

### P1-TEST-05 统计结论缺少多 seed

凡宣称某优化器、正则化、深度或方法更好，需多 seed、均值/方差或置信区间，并允许结果不显著。

### P1-TEST-06 浏览器门禁范围被文档夸大

当前 Stage 6 只真实访问 M17 与首页，没有逐页点击 M00-M17。需参数化访问 18 页，验证实际 HUD、目标聚焦、重复点击、pending 清理、控制台和页面切换。

### P1-TEST-07 播放测试只验证对象结构

现有测试主要检查 frames/源码字符串，未验证真实浏览器播放、暂停、指定 epoch、静态终帧、固定坐标轴、布局不跳、reduced-motion 和无闪烁。

### P1-TEST-08 文档数字无单一真理源

实测覆盖率 95.01%，文档写 95.03%；项目记忆写 Ruff 117 个文件，实际 115。动态数字应由 CI 生成或取消硬编码。

### P1-TEST-09 缺科学措辞 mutation 测试

建立 mutation matrix：模拟→实测、相关→因果、有助于→保证、交换引用、删除条件、反转梯度符号，门禁必须失败。

### P1-TEST-10 覆盖率范围不能代表全项目

95.01% 只统计 `nn_core + datasets`，不代表 dashboard、课程文字、文档、浏览器和事实正确率。README 必须紧邻说明分母。

## 9. P1：README 与项目记忆不一致

1. “复验已全部通过”与当前科学教学验收不通过冲突。
2. 95.03% 应改为实测 95.01% 或取消手写动态值。
3. “72 个证据 ID 全部一一绑定”需按 P0-EVIDENCE 收窄。
4. Ruff 117 个文件应改为 115 或自动生成。
5. “18 页真实浏览器全量复验”需区分历史人工记录与当前 CI 可重复保证。
6. README losses 列表未列新增 `BCEWithLogitsLoss`。
7. 项目状态应分别显示工程门禁、科学审计和课程覆盖，不能合并成单一“通过”。

## 10. P2：相对于“从零到 AI 就业”的课程缺口

以下不是当前代码 bug，但要承担用户目标就必须进入路线图：

1. 系统数学：向量空间、矩阵微积分、概率、极大似然、信息论、数值稳定、优化和统计推断。
2. Python/NumPy/PyTorch：数据管道、autograd、device、dtype、checkpoint、profiling、混合精度和调试。
3. 实验科学：数据拆分、泄漏、置信区间、统计功效、多 seed、消融和错误分析。
4. 经典视觉：LeNet → AlexNet → BatchNorm → ResNet，含真实小数据训练和同协议消融。
5. 序列：RNN → LSTM/GRU → Seq2Seq → teacher forcing → attention。
6. 生成模型：Autoencoder → VAE → GAN → U-Net/DDPM/DiT，含可训练去噪和逆向采样。
7. Transformer/LLM：完整 tokenizer、encoder/decoder、BERT/GPT 最小训练、数据、调度、checkpoint 和验证。
8. 规模工程：分布式并行、ZeRO/FSDP、通信、吞吐、MFU、量化、蒸馏、推理服务和成本。
9. 现代架构：MoE、GQA/MQA、长上下文、SSM/Mamba。
10. 后训练：真实最小 SFT、偏好数据、reward model、PPO/DPO 更新、KL、评估和失败案例。
11. RAG/Agent：检索评估、chunking、rerank、工具 schema、权限、沙箱、审批、幂等、观测和威胁模型。
12. 评估与责任：校准、鲁棒、公平、隐私、版权、治理、污染、安全和不确定性。
13. MLOps：实验追踪、注册、数据版本、CI/CD、监控、漂移、回滚、成本和事故响应。
14. 就业闭环：2~3 个真实数据项目、复现实验报告、部署服务、系统设计、代码评审和作品集标准。

## 11. 权威核验锚点

- AdamW：<https://arxiv.org/abs/1711.05101>
- RNN 梯度：<https://arxiv.org/abs/1211.5063>
- Pre-LN/Post-LN：<https://arxiv.org/abs/2002.04745>
- PPO：<https://arxiv.org/abs/1707.06347>
- Chinchilla：<https://arxiv.org/abs/2203.15556>
- LoRA：<https://arxiv.org/abs/2106.09685>
- QLoRA：<https://arxiv.org/abs/2305.14314>
- DPO：<https://arxiv.org/abs/2305.18290>
- Attention is not Explanation：<https://arxiv.org/abs/1902.10186>
- StreamingLLM：<https://arxiv.org/abs/2309.17453>
- Lost in the Middle：<https://aclanthology.org/2024.tacl-1.9/>
- Reversal Curse：<https://arxiv.org/abs/2309.12288>
- Anthropic 官方复盘：<https://www.anthropic.com/engineering/april-23-postmortem>

每个来源必须直接支持相邻主张，并写明公式、章节、表格或实验协议。把论文放在页面底部不算完成证据绑定。

## 12. 执行者必须提交

1. 每个 `P0-* / P1-*` ID 的完成状态与 commit。
2. 每条科学文字的“旧说法 → 新说法 → 来源定位 → 条件 → 反例”。
3. 每个模拟数字的公式、证据等级、UI 标签和不可推出结论；无法解释的删除。
4. 新测试说明：旧测试为何放过错误，新断言能杀死哪个 mutation。
5. 18 页真实 Chromium 报告：导航、播放、暂停、固定帧、布局稳定和控制台。
6. `claim_id -> source_id -> locator` 人工审校表。
7. 课程地图状态更新；未实现内容不得包装成已实现。
8. 完整质量门禁日志和未解决风险。

## 13. Codex 最终验收门槛

### 13.1 自动化

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
git diff --check
uv run pytest --cov=nn_core --cov=datasets --cov-branch --cov-report=term-missing -q
uv run python tests/test_browser_pending_navigation.py
uv run pytest tests/test_devops_idempotent_deploy.py -q
```

### 13.2 科学门禁

1. 损失函数非法输入矩阵和全合法域梯度对照通过。
2. 18 道测验逐题来源、公式和答案人工复核。
3. 72 条 Claim 显式语义来源，不再按 references 首/末分配。
4. 模拟值在图标题、指标卡和读图指南均可识别为模拟。
5. 随机抽查至少 12 个页面结论，原始来源可直接支持。
6. Chromium 遍历 M00-M17，验证播放/暂停/静态终帧和布局稳定。
7. README、PROJECT_MEMORY、CURRICULUM_MAP 与实测数字和审计结论一致。

### 13.3 发布判定

- P0 未清零：科学教学验收不通过。
- P0 清零、P1 未清零：可作受限辅助课程，不能作唯一主教材。
- P0/P1 清零且 P2 诚实标注：可声明“本版本通过既定范围内教学可信度审计”。
- 即使全部通过，也不得声明“完全正确、覆盖所有 AI、零缺陷或永久无需复审”。
