# 教学质量与工程收敛实施报告

> 完成日期：2026-08-22  
> 执行依据：`HANDOFF_SOL.md`  
> 验收原则：自动化门禁和重点页面人工检查必须同时通过。

## 1. 完成状态

| 交接项 | 状态 | 主要证据 |
|---|---|---|
| P0-1 统一教学证据标签 | 完成 | `dashboard/constants/course.py`、`dashboard/components/pedagogy.py`；M00-M15 共 16 页显式声明 |
| P0-2 第一轮内容准确性审校 | 完成 | M08/M11/M12/M15 边界修正；知识库绝对化措辞收紧；每页提供原始资料入口 |
| P0-3 工程质量门禁 | 完成 | Ruff、格式、Pyright、pytest、diff check 全部通过 |
| P0-4 教学质量自动化测试 | 完成 | `tests/test_pedagogy.py` 覆盖注册表、逐页声明、已知误导表述和模拟结果标签 |
| P1-1 M00 数学基础 | 完成 | shape、广播、链式法则、有限差分误差曲线、形成性测验和 AppTest |
| P1-2 版本化课程地图 | 完成 | `CURRICULUM_MAP.md` 明确已实现/部分实现/仅规划 |
| P1-3 统一页面教学结构 | 完成首轮 | M00、M08、M11、M12、M15 展示完整学习契约；其余页面接入统一 metadata 和参考入口 |
| P2 | 未实施（符合范围） | 未引入账户、LMS、GPU 训练、真实 Whisper/Sora 或技术栈重写 |

## 2. 主要实现

### 教学可信度

- 新增六级 `EvidenceLevel`，本轮实际使用五级；没有页面自称论文复现。
- 16 页均在顶部显示证据标签、结论边界和至少一个原始资料链接。
- 统一课程 metadata 包含先修知识、学习目标、前代瓶颈、可控参数、观察指标、失败案例、历史影响和参考资料。
- 参考资料优先使用原始论文、论文 DOI、arXiv 或权威开放教材。

### 已知准确性问题修复

- M08：明确 Transformer Block 未训练，随机注意力图只用于验证计算，不能解释为语义分工；残差与 Pre-LN 改为有条件结论。
- M11：新增准确名称 `SpectrogramFramePatcher`；明确输出是连续浮点特征块，不是 Whisper tokenizer。
- M12：明确只实现教学预测头和 DDPM 前向加噪；加入紧邻图表的警告，未声称实现逆向采样、DiT 或 Sora。
- M15：删除带真实模型名称和虚构分数的排行榜，改为“预设概率 vs 有限样本”教学表；模拟 PPL、答题和能力画像在结果附近明确标注。
- `knowledge.py`：修正 ReLU/LeakyReLU、Adam、He 初始化、RoPE、ViT、DPO、LoRA 等绝对化表述。

### 工程质量

- 修复此前 246 个 Ruff 问题并统一格式。
- 修复 Pyright 暴露的 forward/backward 缓存前置条件、scikit-learn 返回类型、数组/字典推断和返回类型问题。
- backward 在未执行 forward 时现在抛出明确 `RuntimeError`，而不是依赖空缓存产生难懂异常。
- 完成 Streamlit 1.62 API 迁移：宽度参数统一为 `width="stretch"`，内嵌交互脚本改用 `st.iframe`；真实服务复验无相关过期警告。

### 交互体验优化

- M01 连续演播改为纯浏览器端原位动画：播放期间不执行 Streamlit rerun，不会清空、折叠或重建页面区域。
- 播放、暂停、单步和进度状态由稳定 iframe 控制；暂停后参数、决策图和权重轨迹保持在同一 Step。
- 决策概率背景、分界线和权重轨迹由高 DPI Canvas 原位重绘；Loss/Accuracy 图保持静态，从根源消除上下跳动与组件闪现。
- 抽取可跨页面复用的浏览器端时间轴；M12 视频帧、帧号与运动能量游标已接入同一套连续播放、暂停与逐帧检查机制，并使用帧间插值避免闪烁。
- M02 记录真实 Epoch 训练检查点并同步播放：概率色块、P=0.5 决策边界、网络权重、探针神经元响应与预测概率均来自同一模型状态；不以淡入最终图伪装训练过程。
- 全站普通 Plotly 图表统一接入浏览器端读图播放器：折线/散点逐步绘制、柱形增长、直方图样本累积、热力/等高/饼图渐显、仪表数值演进；默认保留完整静态终态，并提供起点、播放、暂停、终点控制。
- 动画语义分层：M01/M02/M12 使用真实训练或序列状态；其余静态统计图明确作为“读图构建动画”，不将视觉渐显错误描述成训练过程。
- 教学导航统一为平滑滚动、柔和入场光晕和“正在查看这里”定位标签，避免强烈闪烁，并支持系统“减少动态效果”偏好。
- 浮动空间导航仅在超宽屏显示，避免遮挡教学图表；常规视口仍保留页面内空间导航。

## 3. 验证结果

最终独立执行：

```text
uv run ruff check .
All checks passed!

uv run ruff format --check .
57 files already formatted

uv run pyright nn_core/ datasets/
0 errors, 0 warnings, 0 informations

uv run pytest tests/ -q
211 passed

git diff --check
PASS（仅 Git 提示 Windows CRLF 转换策略，无空白错误）
```

### 页面人工检查

使用本地 Streamlit 服务检查首页以及 M00、M08、M11、M12、M15：

- 页面均无 Streamlit exception；
- Streamlit 1.62 运行日志无 `use_container_width` 或 `components.html` 过期警告；
- 当前桌面视口均无横向溢出；
- 证据标签、结论边界与原始资料链接可见；
- M12 的“仅前向加噪”警告紧邻相关内容；
- M15 的“所有数值非真实模型成绩”警告可见；
- M00 的梯度检查默认设置显示解析与数值梯度吻合。

## 4. 兼容性说明

- `AudioTokenizer` 保留为 `SpectrogramFramePatcher` 的兼容子类，旧导入和调用方式继续工作。
- `nn_core.__init__` 同时导出新旧名称。
- 未加入新的运行时依赖。
- README 原有使命和范围边界得到保留，并新增 M00 与课程地图入口。

## 5. 参考资料核对

课程注册表收录并核对了感知器、反向传播、Adam、word2vec、LSTM、神经注意力、Transformer、ViT、CLIP、Whisper、DDPM、DiT、Chinchilla、InstructGPT、DPO 和 HELM 等原始资料入口。部分出版商 DOI 页面可能受机器人访问限制，但 DOI 本身保留为稳定的出版物定位符。

## 6. 已知限制与后续工作

- 除 M00、M08、M11、M12、M15 外，其余页面尚未完整展开学习契约；已具备 metadata，可逐页迁移。
- Transformer、GPT、ViT、CLIP、后训练和世界模型仍以核心算子、缩小实现或模拟为主；证据标签已经明确该边界。
- M00 是“后续课程必需数学”的最小入口，不替代完整线性代数、微积分和概率课程。
- 当前人工检查覆盖重点桌面页面；后续可增加移动端视口、色盲对比度和截图视觉回归。
- 随机实验仍有部分使用模块级 NumPy 随机状态；后续应逐步迁移为显式 `Generator` 并记录每次实验配置。

## 7. 变更范围摘要

- 新增：课程注册表、教学证据组件、M00、课程地图、教学质量测试和本报告。
- 修改：README、首页、M01-M15、关键知识文案、音频/视频接口说明、类型契约与现有测试。
- Ruff 格式化覆盖了若干未发生语义变化的既有 Python 文件，因此 diff 文件数较多；核心行为由 211 项测试复验。
- 未提交、未推送、未发布、未部署外部服务。
