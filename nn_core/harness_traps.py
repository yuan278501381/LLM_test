# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.harness_traps - AI 真实工程陷阱、物理盲区与 Harness 控制环算法模块

涵盖现代大语言模型落地中六大核心现实问题与工程架构防御：
1. AttentionSinkSimulator: 初始 Token 注意力汇聚与滑动窗口困惑度演化模拟 (基于 Xiao et al., 2023 StreamingLLM)
2. LostInTheMiddleSimulator: 长上下文位置敏感性与 U 型衰减模拟 (基于 Liu et al., TACL 2024)
3. ReversalCurseEngine: 自回归单向条件概率下的逆向推理断裂模拟 (基于 Berglund et al., 2023)
4. TokenizerTrapInspector: BPE 切分、字符计数与前导空格表示分析
5. AgentHarnessGuard: 工具调用死循环熔断器 (Circuit Breaker) 与工具级关键词过滤对比生产纵深防御
6. ClaudeCode2026PostmortemRunner: 2026 年 Anthropic Claude Code 官方技术复盘与工程状态推演
7. LoopEngineeringEngine: 智能体循环工程 (Loop Engineering) 演进轨迹与确定性验证门禁推演
"""

import logging
import re
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 注意力汇聚与 StreamingLLM 演化模拟 (Attention Sinks)
# ---------------------------------------------------------------------------
class AttentionSinkSimulator:
    """
    自回归注意力汇聚效应 (Attention Sinks) 与滑动窗口困惑度演化模拟器。

    教学与证据说明:
        - 证据等级: SIMULATION (规则模拟)
        - 理论依据: Xiao et al. (ICLR 2024 / arXiv 2023) "Efficient Streaming Language Models with Attention Sinks"
        - 机制解析: Softmax 归一化要求每层注意力权重非负且和为 1，当后续 Token 缺乏强语义关联时，
          模型倾向于将冗余注意力分配到初始 Token，使初始 Token 成为汇聚锚点 (Attention Sink)。
        - 边界与局限: 本模块为基于论文观察构建的教学规则模型，不代表特定生产模型在任意长文本下的真实实测 PPL；
          不同模型架构与分词器对 Sink Token 数量的最优配置可能不同，不保证固定 4 个对所有模型均最优。
    """

    @staticmethod
    def simulate_streaming_perplexity(
        seq_length: int = 128,
        window_size: int = 32,
        num_sink_tokens: int = 4,
        base_perplexity: float = 18.5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        模拟在流式生成中，保留 vs 丢失初始 Sink Token 的困惑度对比。

        参数:
            seq_length: 总序列长度 (步数)
            window_size: 滑动窗口大小 (必须 > 0)
            num_sink_tokens: 保留的初始 Sink Token 数量 (>= 0)
            base_perplexity: 基础困惑度基线 (> 0)
            seed: 局部随机数发生器种子 (隔离全局随机状态)
        """
        if not (isinstance(seq_length, (int, np.integer)) and seq_length > 0):
            raise ValueError(f"seq_length 必须为正整数: {seq_length}")
        if not (isinstance(window_size, (int, np.integer)) and window_size > 0):
            raise ValueError(f"window_size 必须为正整数: {window_size}")
        if not (isinstance(num_sink_tokens, (int, np.integer)) and num_sink_tokens >= 0):
            raise ValueError(f"num_sink_tokens 必须为非负整数: {num_sink_tokens}")
        if not (
            isinstance(base_perplexity, (int, float, np.floating))
            and np.isfinite(base_perplexity)
            and base_perplexity > 0
        ):
            raise ValueError(f"base_perplexity 必须为有效正数 (非 NaN/Inf): {base_perplexity}")

        rng = np.random.default_rng(seed)
        steps = np.arange(1, seq_length + 1)
        ppl_with_sinks = []
        ppl_no_sinks = []
        sink_attention_weights = []

        has_effective_sinks = num_sink_tokens > 0

        for t in steps:
            noise = float(rng.normal(0, 0.35))
            # 基础平稳困惑度
            ppl_base = base_perplexity + noise + 0.02 * np.sin(t / 5.0)

            # 初始 Token 吸收的注意力权重 (模拟随步数增长汇聚到 25%~60%)
            sink_weight = min(0.65, 0.30 + 0.25 * (1.0 - np.exp(-t / 15.0)))
            sink_attention_weights.append(float(sink_weight))

            # 有 Sink 策略: 若配置了 num_sink_tokens > 0 则平稳，否则同样发生漂移
            if has_effective_sinks:
                ppl_with_sinks.append(max(float(ppl_base), 5.0))
            else:
                if t <= window_size:
                    ppl_with_sinks.append(max(float(ppl_base), 5.0))
                else:
                    overflow = t - window_size
                    ppl_decay = base_perplexity * np.exp(min(overflow / 8.0, 5.0)) + float(
                        rng.uniform(10, 50)
                    )
                    ppl_with_sinks.append(float(ppl_decay))

            # 无 Sink 策略: 超出滑动窗口后激活分布畸变，模拟困惑度剧烈上升
            if t <= window_size:
                ppl_no_sinks.append(max(float(ppl_base), 5.0))
            else:
                overflow = t - window_size
                ppl_broken = base_perplexity * np.exp(min(overflow / 6.0, 6.5)) + float(
                    rng.uniform(30, 120)
                )
                ppl_no_sinks.append(float(ppl_broken))

        return {
            "evidence_level": "SIMULATION",
            "is_rule_simulation": True,
            "steps": steps.tolist(),
            "ppl_with_sinks": ppl_with_sinks,
            "ppl_no_sinks": ppl_no_sinks,
            "sink_attention_weights": sink_attention_weights,
            "window_size": window_size,
            "num_sink_tokens": num_sink_tokens,
            "assumptions": [
                "基于 Xiao et al. (2023) StreamingLLM 论文观察到的注意力汇聚经验现象构建",
                "Softmax 归一化是注意力汇聚的背景，初始 Token 吸收冗余注意力，保留初始 KV 可稳定注意力激活分布",
                "曲线为教学规则模拟，用于直观展示滑动窗口机制，不代表真实模型的端到端实测日志",
            ],
            "boundary": (
                "注意力汇聚现象在不同模型（Llama / Mistral / Qwen）和不同训练目标下表现强弱不同；"
                "并非所有长文本任务固定 4 个 Sink Token 均最优，需结合具体模型架构和上下文长度进行评测。"
            ),
        }


# ---------------------------------------------------------------------------
# 2. 迷失在中间与长上下文位置敏感性 (Lost in the Middle)
# ---------------------------------------------------------------------------
class LostInTheMiddleSimulator:
    """
    长上下文多文档/键值检索中的位置敏感性与 U 型衰减模拟器。

    教学与证据说明:
        - 证据等级: SIMULATION (规则模拟)
        - 理论依据: Liu et al. (TACL 2024 / arXiv 2023) "Lost in the Middle: How Language Models Use Long Contexts"
        - 机制解析: 在多段落 QA 与长上下文检索中，关键信息位于文档开头或结尾时召回率较高，
          位于正中间 (相对深度 p ≈ 0.5) 时容易被注意力稀释与位置偏置影响而导致召回率下降。
        - 边界与反例: U 型现象依赖特定评测协议（如跨多段落检索/单目标定位）；
          Rerank 重排将高相关段落置顶虽能改善单点检索，但可能打乱上下文的时序连续性、破坏因果依赖或拆分跨段落证据。
    """

    @staticmethod
    def compute_u_curve(
        context_length_k: int = 32,
        num_points: int = 21,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        计算 0.0~1.0 相对深度下的原始检索率 vs Rerank 重排后的模拟对比。

        参数:
            context_length_k: 模拟上下文长度 (千 Tokens, > 0)
            num_points: 采样点数 (>= 3)
            seed: 局部随机数发生器种子
        """
        if not (
            isinstance(context_length_k, (int, float, np.floating))
            and np.isfinite(context_length_k)
            and context_length_k > 0
        ):
            raise ValueError(f"context_length_k 必须为有效正数 (非 NaN/Inf): {context_length_k}")
        if not (isinstance(num_points, (int, np.integer)) and num_points >= 3):
            raise ValueError(f"num_points 至少为 3 的整数: {num_points}")

        rng = np.random.default_rng(seed)
        depths = np.linspace(0.0, 1.0, num_points)

        # 模拟随上下文长度增加，中间位置的衰减幅度增强
        scale_penalty = min(0.70, 0.20 + 0.10 * np.log2(max(context_length_k / 4.0, 1.0)))

        raw_accuracies = []
        rerank_accuracies = []
        attention_densities = []

        for p in depths:
            # U 型曲线: 4 * p * (1 - p) 在 p=0.5 达到峰值 1.0，两端为 0
            u_decay = 4.0 * p * (1.0 - p)
            noise_raw = float(rng.normal(0, 0.015))
            acc_raw = max(0.05, 0.94 - scale_penalty * u_decay + noise_raw)
            raw_accuracies.append(float(round(acc_raw * 100.0, 1)))

            # Rerank 教学模拟: 重排后置于开头，但考虑重排器可能轻微失真
            noise_rerank = float(rng.normal(0, 0.01))
            acc_rerank = np.clip(0.92 - 0.04 * p + noise_rerank, 0.80, 0.98)
            rerank_accuracies.append(float(round(float(acc_rerank) * 100.0, 1)))

            # 中间注意力稀释因子
            att_density = float(np.clip(1.0 - 0.80 * u_decay, 0.10, 1.0))
            attention_densities.append(float(round(att_density, 3)))

        return {
            "evidence_level": "SIMULATION",
            "is_rule_simulation": True,
            "depths": [round(float(d), 2) for d in depths],
            "raw_accuracies": raw_accuracies,
            "rerank_accuracies": rerank_accuracies,
            "attention_densities": attention_densities,
            "context_length_k": context_length_k,
            "assumptions": [
                "基于 Liu et al. (TACL 2024) 多段落检索实验观察到的 U 型位置敏感现象",
                "模拟函数以二次抛物线 4*p*(1-p) 刻画相对深度 p 下注意力稀释趋势",
            ],
            "boundary": (
                "U 型曲线是特定长上下文评测协议下的经验现象，不代表所有现代大模型在所有输入分布下必为固定 U 曲线；"
                "Rerank 优化需权衡重排器自身的漏召回风险，且重排可能破坏上下文的自然时序或跨段落逻辑引用。"
            ),
        }


# ---------------------------------------------------------------------------
# 3. 逆向诅咒 (Reversal Curse)
# ---------------------------------------------------------------------------
class ReversalCurseEngine:
    """
    自回归单向条件概率下的逆向推理断裂模拟引擎。

    教学与证据说明:
        - 证据等级: SIMULATION (规则模拟)
        - 理论依据: Berglund et al. (ICLR 2024 / arXiv 2023) "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'"
        - 机制解析: 自回归因果语言模型以单向条件概率 P(Token_t | Token_<t) 训练。
          当模型仅在训练数据中学习 '实体 A 的 [关系 R] 是 实体 B' 时，并未自动建立双向对称知识图谱，
          在无双向数据增强或反向推理时，反向查询 P(实体 A | 实体 B, 逆向关系) 的条件概率极低。
        - 核心边界: 逆向诅咒指的是预训练权重中的单向泛化缺失；如果通过 Prompt 在当前上下文中明确提供了实体双向关系，
          模型具备在上下文中推理反向关系的能力。
    """

    SYNTHETIC_ENTITIES: ClassVar[list[dict[str, Any]]] = [
        {
            "id": "rel_01",
            "entity_a": "学者阿尔法 (Scholar Alpha, 合成实体)",
            "entity_b": "论文贝塔 (Paper Beta, 合成成果)",
            "forward_rel": "主要著作是",
            "reverse_rel": "的主要作者是",
            "context_text": "学者阿尔法的主要著作是论文贝塔。",
        },
        {
            "id": "rel_02",
            "entity_a": "虚构星系恒星X (Star-X, 合成实体)",
            "entity_b": "环绕行星Y (Planet-Y, 合成实体)",
            "forward_rel": "的伴星系统包含",
            "reverse_rel": "所围绕公转的主星是",
            "context_text": "虚构星系恒星X的伴星系统包含环绕行星Y。",
        },
        {
            "id": "rel_03",
            "entity_a": "虚拟机构伽马 (Lab Gamma, 合成实体)",
            "entity_b": "发明家德尔塔 (Inventor Delta, 合成实体)",
            "forward_rel": "的首席科学家是",
            "reverse_rel": "所领导的研究机构是",
            "context_text": "虚拟机构伽马的首席科学家是发明家德尔塔。",
        },
    ]

    @classmethod
    def query_relation(
        cls,
        item_index: int = 0,
        is_reverse_query: bool = False,
    ) -> dict[str, Any]:
        """
        模拟自回归单向训练下的前向查询 vs 逆向查询示例。

        参数:
            item_index: 合成实体条目索引
            is_reverse_query: 是否执行逆向查询
        """
        if not (
            isinstance(item_index, (int, np.integer))
            and 0 <= item_index < len(cls.SYNTHETIC_ENTITIES)
        ):
            raise ValueError(
                f"item_index 超出合法范围 [0, {len(cls.SYNTHETIC_ENTITIES) - 1}]: {item_index}"
            )
        item = cls.SYNTHETIC_ENTITIES[item_index]

        if not is_reverse_query:
            prompt = f"问：{item['entity_a']} 的 {item['forward_rel']} 谁/什么？"
            prob_correct = 0.95
            response = f"答：{item['entity_a']} 的 {item['forward_rel']} {item['entity_b']}。"
            direction = "前向因果查询 (Forward Causal Query) [MATCH]"
            reasoning = (
                "模型在训练序列中单向学习了 'A 的关系是 B' 这一序列分布，自回归前向预测概率较高。"
            )
        else:
            prompt = f"问：{item['entity_b']} 的 {item['reverse_rel']} 谁/什么？"
            prob_correct = 0.08
            response = f"答：抱歉，我不确定 {item['entity_b']} 的具体关联实体，或给出了无关猜测。"
            direction = "逆向因果查询 (Reversed Query - Reversal Curse) [FAIL]"
            reasoning = (
                "自回归模型学习的是 P(B | A, 关系)，并未在参数中建立无向图谱。"
                "在没有反向语料显式微调时，逆向条件概率 P(A | B, 逆向关系) 极度稀疏。"
            )

        return {
            "evidence_level": "SIMULATION",
            "is_rule_simulation": True,
            "entity_id": item["id"],
            "prompt": prompt,
            "response": response,
            "prob_correct": prob_correct,
            "direction": direction,
            "reasoning": reasoning,
            "is_reverse": is_reverse_query,
            "context_text": item["context_text"],
            "assumptions": [
                "采用合成实体展示自回归因果概率单向性，避免真实实体语料双向污染",
                "概率值为展示反向断裂现象的教学示例值，非实际模型 logits",
            ],
            "boundary": (
                "原论文重要边界：反向诅咒指预训练权重中的无向泛化缺失；"
                "若在 In-Context 提示词中提供了双向定义，模型可以在上下文推理中正确回答逆向问题。"
            ),
        }


# ---------------------------------------------------------------------------
# 4. 分词器切分与字符认知 (Tokenizer Traps)
# ---------------------------------------------------------------------------
class TokenizerTrapInspector:
    """
    分词器子词切分、前导空格敏感度与字符计数分析器。

    教学与证据说明:
        - 证据等级: TEACHING_SCALE (教学缩小版 / 结构化示例)
        - 机制解析: BPE (Byte-Pair Encoding) 等分词算法将文本贪心合并为子词 Token ID。
          模型在 Transformer 嵌入层接收的是离散的 Token 向量，而非每个字符的独立输入；
          因此在没有思维链 (Chain of Thought) 显式展开或字符级辅助任务时，直接在单步中统计字符较为困难。
        - 澄清纠偏: 字符信息并未在物理层面彻底消失；通过思维链（如让模型逐字母输出并计数）或字符级辅助表示，
          通常有助于提升字符计数的准确率，但并不保证在所有模型与任务中 100% 消除误差。
    """

    @staticmethod
    def inspect_strawberry(word: str = "strawberry", target_char: str = "r") -> dict[str, Any]:
        """
        分析草莓单词字符计数的教学手工切分示例与注意力机制分析。
        """
        clean_word = str(word).strip() if word else "strawberry"
        clean_target = str(target_char).strip() if target_char else "r"

        # 教学手工示意分词（以常见子词切分模式为例）
        if clean_word.lower() == "strawberry":
            subwords = ["straw", "berry"]
        else:
            mid = max(1, len(clean_word) // 2)
            subwords = [clean_word[:mid], clean_word[mid:]]

        actual_count = clean_word.lower().count(clean_target.lower())
        char_list = list(clean_word)

        return {
            "evidence_level": "TEACHING_SCALE",
            "is_toy_demonstration": True,
            "word": clean_word,
            "target_char": clean_target,
            "actual_count": actual_count,
            "subwords": subwords,
            "char_list": char_list,
            "explanation": (
                f"在教学手工示意中，单词 '{clean_word}' 被切分为子词单元：{subwords}。"
                f"模型在注意力计算中接收的是这两个 Token 的向量嵌入，而非逐个字符的独立位置编码。"
                f"当直接要求单步输出目标字符 '{clean_target}' 的出现次数时，模型容易产生计数偏差；"
                "通过思维链（CoT）显式让模型逐字符拼写展开，通常有助于提升字符计数的准确率，但并不保证在所有输入下 100% 消除计数误差。"
            ),
        }

    @staticmethod
    def inspect_whitespace_tokens(base_text: str = "123") -> list[dict[str, Any]]:
        """
        展示前导空格产生不同 Token 切分的现象。
        """
        clean_base = str(base_text).strip() if base_text else "123"
        variants = [
            (f"{clean_base}", "无前导空格 (Raw Text)", "['" + clean_base + "']"),
            (f" {clean_base}", "单前导空格 (Leading Space)", "[' " + clean_base + "']"),
            (f"  {clean_base}", "双前导空格 (Double Space)", "[' ', ' " + clean_base + "']"),
        ]
        results = []
        for text, desc, tok_repr in variants:
            results.append(
                {
                    "text_repr": repr(text),
                    "desc": desc,
                    "token_representation": tok_repr,
                    "char_length": len(text),
                }
            )
        return results


# ---------------------------------------------------------------------------
# 5. Agent Harness 纵深防御与工具过滤器对比 (Agent Harness Guard)
# ---------------------------------------------------------------------------
class AgentHarnessGuard:
    """
    智能体运行控制架: 死循环熔断器与工具级关键词过滤对比生产级纵深防御。

    教学与证据说明:
        - 证据等级: TEACHING_SCALE (教学原型)
        - 机制解析: 正则表达式黑名单仅能作为最基础的工具级字符串检查 (Toy String Filter)，
          极易被大小写变换、Unicode 混淆、编码拆分或 Base64 绕过，绝不能单独作为安全授权的依据。
        - 生产级纵深防御标准:
          1. Strict JSON Schema 结构化参数强校验
          2. 工具最小权限原则 (Least Privilege) 与只读/写入权限物理隔离
          3. 容器/轻量级微虚拟机 (MicroVM) 沙箱执行
          4. 幂等键与超时/重复调用熔断器
          5. 关键写/删除/执行操作的人工确认 (Human-in-the-Loop Approval)
          6. 全链路不可篡改的结构化审计日志
    """

    TOY_INJECTION_PATTERNS: ClassVar[list[str]] = [
        r"ignore\s+previous\s+instructions",
        r"system\s+override",
        r"rm\s+-rf",
        r"delete\s+all",
        r"curl\s+.*\s*\|\s*bash",
    ]

    @classmethod
    def inspect_tool_call(
        cls,
        call_history: list[dict[str, Any]],
        new_tool: str,
        new_args: dict[str, Any],
        max_repeats: int = 3,
    ) -> dict[str, Any]:
        """
        检查新工具调用是否触发重复死循环或匹配到工具关键词规则。

        参数:
            call_history: 历史工具调用记录
            new_tool: 待调用工具名称
            new_args: 工具调用参数
            max_repeats: 允许的最大连续重复调用阈值 (必须 > 0)
        """
        if max_repeats <= 0:
            raise ValueError(f"max_repeats 必须为正整数: {max_repeats}")

        # 1. 检查重复调用 (乒乓死循环熔断)
        recent_calls = [c for c in call_history[-max_repeats:] if c.get("tool") == new_tool]
        is_ping_pong = len(recent_calls) >= max_repeats

        # 2. 工具关键词过滤检查 (仅用于教学演示脆弱性)
        arg_str = str(new_args).lower()
        injection_found = any(re.search(pat, arg_str) for pat in cls.TOY_INJECTION_PATTERNS)

        if injection_found:
            status = "TOY_FILTER_MATCHED [SUSPICIOUS_KEYWORD_FLAGGED]"
            action = "工具关键词过滤器命中已知黑名单模式（注：黑名单极易被绕过，需结合上下文隔离与沙箱执行）。"
            color = "rose"
            is_blocked = True
        elif is_ping_pong:
            status = "PING_PONG_LOOP_INTERRUPTED [CIRCUIT_BREAKER]"
            action = f"检测到连续 {max_repeats} 次相同工具调用，触发 Harness 状态机熔断，阻止空转并注入反思引导。"
            color = "amber"
            is_blocked = True
        else:
            status = "TOY_FILTER_NO_MATCH [INCOMPLETE_CHECK_ONLY]"
            action = (
                "工具关键词过滤器未命中已知模式。（警告：此过滤器极易被大小写/编码绕过，"
                "严禁单独作为生产安全准入依据！生产环境必须依赖参数 Schema 强校验、权限隔离与沙箱）。"
            )
            color = "emerald"
            is_blocked = False

        return {
            "evidence_level": "TEACHING_SCALE",
            "status": status,
            "action": action,
            "color": color,
            "is_blocked": is_blocked,
            "production_defense_checklist": [
                "Strict JSON Schema 参数强类型校验与未知字段丢弃",
                "工具白名单与最小权限执行环境 (Least Privilege)",
                "容器/只读文件系统沙箱隔离 (Sandbox Isolation)",
                "幂等键与超时/死循环熔断器 (Idempotency & Circuit Breaker)",
                "关键破坏性操作人工二次确认 (Human-in-the-Loop Approval)",
                "全链路 TraceID 审计与日志透传 (Audit Logging)",
            ],
        }


# ---------------------------------------------------------------------------
# 6. 2026 Claude Code 事故官方复盘推演器 (Official Postmortem Runner)
# ---------------------------------------------------------------------------
class ClaudeCode2026PostmortemRunner:
    """
    2026 年 Anthropic Claude Code 官方技术复盘与工程状态推演器。

    教学与证据说明:
        - 证据等级: OFFICIAL_DOCUMENTATION / TEACHING_SCALE
        - 官方事实来源: Anthropic (2026-04-23) "An update on recent Claude Code quality reports"
          URL: https://www.anthropic.com/engineering/april-23-postmortem
        - 官方确认事实要点:
          1. 3 月 4 日: 默认 reasoning effort 从 high 调整为 medium；4 月 7 日全面回退。
          2. 3 月 26 日: 闲置会话内存清理逻辑 Bug，在每轮执行中误清除了历史 thinking 记录，导致遗忘、重复与异常工具选择；
             4 月 10 日发布 v2.1.101 修复。
          3. 4 月 16 日: 在系统提示词中追加简短要求（verbosity 提示词）损害了 coding quality，在某扩展编码评测集之一中观察到下降约 3%；
             4 月 20 日全面回退解决。
    """

    OFFICIAL_INCIDENTS: ClassVar[dict[str, dict[str, Any]]] = {
        "reasoning_downgrade": {
            "title": "事故 1: 思考预算默认值调整 (3月4日 ~ 4月7日)",
            "official_timeline": "3 月 4 日将默认 reasoning effort 从 high 调为 medium；4 月 7 日回退为默认 high/xhigh。",
            "root_cause": "官方确认根因：为降低 high effort 下偶发思考时间过长导致的界面冻结与过高延迟，将默认 effort 调为 medium，后认为权衡不当。",
            "official_finding": "官方确认事实：用户反馈调为 medium 后模型体验不如之前聪明（felt less intelligent），且内部评测显示 medium 智力表现略低（slightly lower intelligence）。",
            "resolution": "官方确认修复：4 月 7 日回退决定，默认 effort 恢复为 high（Opus 4.7 设为 xhigh），并支持用户通过 /effort 自行调节。",
            "course_inferred_recommendation": "【课程推断与工程建议】思考预算调低直接制约长复杂编码任务的推理深度；建议向用户提供客户端显式控制参数并进行充分长周期评测。",
        },
        "session_cache_wipe": {
            "title": "事故 2: 闲置会话缓存清理逻辑缺陷 (3月26日 ~ 4月10日)",
            "official_timeline": "3 月 26 日引入闲置超 1 小时会话的思考清理优化；4 月 10 日发布 v2.1.101 修复。",
            "root_cause": "官方确认根因：设计初衷为会话闲置超 1 小时后仅在恢复时清理一次旧思考（thinking）历史以降低成本，但实现 Bug 导致此后每轮执行都持续丢弃历史思考块。",
            "official_finding": "官方确认事实：导致模型丢失先前决策记忆，出现健忘（forgetful）、重复（repetitive）与异常工具选择（odd tool choices），且由于缓存未命中导致使用限额加速消耗。",
            "resolution": "官方确认修复：4 月 10 日在 v2.1.101 中修复该持续清理 Bug，恢复完整推理历史传递。",
            "course_inferred_recommendation": "【课程推断与工程建议】上下文内存优化必须具备端到端多轮一致性回归测试，防止内存清理逻辑破坏跨轮推理历史。",
        },
        "verbosity_clamp": {
            "title": "事故 3: 系统提示词简洁度调整 (4月16日 ~ 4月20日)",
            "official_timeline": "4 月 16 日在系统提示词中追加简洁度限制指令；4 月 20 日全面回退。",
            "root_cause": "官方确认根因：为控制输出篇幅在系统提示词中添加了简洁度（verbosity）字数限制指令，与其他提示词变动叠加后损害了代码质量（hurt coding quality）。",
            "official_finding": "官方确认事实：官方扩展评测集之一显示 Opus 4.6 与 Opus 4.7 的评测表现均出现约 3% 下降（One of these evaluations showed a 3% drop for both Opus 4.6 and 4.7）。",
            "resolution": "官方确认修复：4 月 20 日全面回退该系统提示词指令。",
            "course_inferred_recommendation": "【课程推断与工程建议】提示词字数硬性限制容易与复杂长代码生成的完整性需求产生冲突导致代码截断；建议设立防截断门禁与 AST 语法完整性断言。",
        },
    }

    @classmethod
    def get_incident_data(cls, incident_key: str = "reasoning_downgrade") -> dict[str, Any]:
        """
        获取特定事故的官方复盘核验数据。
        """
        if incident_key not in cls.OFFICIAL_INCIDENTS:
            raise ValueError(
                f"未知事故标识 '{incident_key}'。合法键值: {list(cls.OFFICIAL_INCIDENTS.keys())}"
            )
        data = cls.OFFICIAL_INCIDENTS[incident_key]
        return {
            "evidence_level": "OFFICIAL_DOCUMENTATION",
            "key": incident_key,
            **data,
            "official_source_url": "https://www.anthropic.com/engineering/april-23-postmortem",
        }


# ---------------------------------------------------------------------------
# 7. 循环工程 (Loop Engineering) 仿真推演引擎
# ---------------------------------------------------------------------------
class LoopEngineeringEngine:
    """
    智能体循环工程 (Loop Engineering) 教学仿真推演引擎。

    教学与证据说明:
        - 证据等级: SIMULATION (规则模拟)
        - 核心概念: 对比 Naive ReAct、Evaluator-Optimizer (生成-验证-优化) 与 Plan-Execute 循环在软件工程中的行为轨迹。
        - 机制解析: 缺乏外部确定性验证器 (如 pytest/编译器) 时，智能体容易基于自我幻觉提前终止 (False Positive)；
          具备确定性验证器与状态回退机制时，智能体能够将错误信息转为反思反馈，驱动闭环收敛。
        - 边界说明: 成功率与步数为教学规则模型下的模拟结果，不代表具体大模型在任意工业代码库中的生产胜率；
          教学示例中的 git restore 代表原子回滚概念，生产中应采用隔离工作树 (Worktree) 或快照以避免覆盖用户未提交修改。
    """

    LOOP_PATTERNS: ClassVar[dict[str, dict[str, Any]]] = {
        "naive_react": {
            "name": "朴素 ReAct 循环 (Naive ReAct Loop)",
            "description": "Thought -> Action -> Observe -> Output。缺乏外部确定性验证门禁，容易基于自我幻觉误判完成。",
            "assumed_success_rate": 0.35,
            "avg_tokens_per_step": 1200,
            "rollback_support": False,
        },
        "evaluator_optimizer": {
            "name": "生成-评估器-优化器循环 (Evaluator-Optimizer Loop)",
            "description": "Generator -> Deterministic Verifier (编译/测试) -> (Pass ? Done : 注入错误堆栈并重新生成)。",
            "assumed_success_rate": 0.80,
            "avg_tokens_per_step": 1600,
            "rollback_support": True,
        },
        "plan_and_execute": {
            "name": "分层规划-执行-动态重排循环 (Plan-and-Execute with Rollback)",
            "description": "Planner 拆解 DAG 任务 -> Step Executor 逐一执行 -> 确定性验证门禁 -> 失败原子回滚并动态重排。",
            "assumed_success_rate": 0.92,
            "avg_tokens_per_step": 2000,
            "rollback_support": True,
        },
    }

    @classmethod
    def simulate_agent_loop(
        cls,
        pattern: str = "evaluator_optimizer",
        task_difficulty: str = "medium",
        has_deterministic_verifier: bool = True,
        has_state_rollback: bool = True,
        max_iterations: int = 8,
        budget_token_limit: int = 15000,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """
        推演一次智能体循环执行轨迹与收敛状态。

        参数:
            pattern: 循环拓扑 ("naive_react", "evaluator_optimizer", "plan_and_execute")
            task_difficulty: 任务难度 ("simple", "medium", "complex_refactor")
            has_deterministic_verifier: 是否接入外部确定性测试/编译器验证器
            has_state_rollback: 失败时是否执行原子回滚
            max_iterations: 最大循环步数限制 (必须 > 0)
            budget_token_limit: Token 预算硬上限 (必须 > 0)
            random_seed: 局部随机种子
        """
        if pattern not in cls.LOOP_PATTERNS:
            raise ValueError(
                f"未知循环拓扑 '{pattern}'。合法选项: {list(cls.LOOP_PATTERNS.keys())}"
            )
        difficulty_target_steps = {
            "simple": 2,
            "medium": 4,
            "complex_refactor": 6,
        }
        if task_difficulty not in difficulty_target_steps:
            raise ValueError(
                f"未知任务难度 '{task_difficulty}'。合法选项: {list(difficulty_target_steps.keys())}"
            )
        if not (isinstance(max_iterations, (int, np.integer)) and max_iterations > 0):
            raise ValueError(f"max_iterations 必须为正整数: {max_iterations}")
        if not (isinstance(budget_token_limit, (int, np.integer)) and budget_token_limit > 0):
            raise ValueError(f"budget_token_limit 必须为正整数: {budget_token_limit}")

        rng = np.random.default_rng(random_seed)
        target_steps = difficulty_target_steps[task_difficulty]
        cfg = cls.LOOP_PATTERNS[pattern]
        avg_tokens = cfg["avg_tokens_per_step"]

        trace_steps: list[dict[str, Any]] = []
        cumulative_tokens = 0
        current_pass_pct = 0.0
        dirty_states_count = 0
        rollback_count = 0
        is_success = False
        terminal_status = "MAX_ITERATIONS_REACHED"

        for step in range(1, max_iterations + 1):
            step_token = int(avg_tokens * float(rng.uniform(0.85, 1.15)))
            cumulative_tokens += step_token

            # 预算超限熔断
            if cumulative_tokens > budget_token_limit:
                terminal_status = "BUDGET_EXCEEDED [COST CIRCUIT BREAKER]"
                trace_steps.append(
                    {
                        "step": step,
                        "phase": "BUDGET_GUARD_TRIGGERED",
                        "action": "Token 消耗突破设定预算上限，触发系统熔断中止空转！",
                        "verifier_status": "BLOCKED",
                        "test_pass_pct": current_pass_pct,
                        "step_tokens": step_token,
                        "cumulative_tokens": cumulative_tokens,
                        "state_clean": dirty_states_count == 0,
                    }
                )
                break

            # 无验证器时的假阳性自判完成模拟
            if pattern == "naive_react" and not has_deterministic_verifier and step >= 2:
                is_fake_pass = bool(rng.random() < 0.65)
                if is_fake_pass:
                    terminal_status = "FALSE_POSITIVE_TERMINATION"
                    current_pass_pct = 40.0 if task_difficulty == "simple" else 25.0
                    trace_steps.append(
                        {
                            "step": step,
                            "phase": "PREMATURE_EXIT",
                            "action": "模型基于自我幻觉断言'所有修复已完成'，主动结束循环（实际测试未通过）。",
                            "verifier_status": "UNVERIFIED [FALSE POSITIVE]",
                            "test_pass_pct": current_pass_pct,
                            "step_tokens": step_token,
                            "cumulative_tokens": cumulative_tokens,
                            "state_clean": True,
                        }
                    )
                    break

            # 正常推进模拟
            progress_delta = (100.0 / target_steps) * float(rng.uniform(0.75, 1.25))
            if dirty_states_count > 0:
                progress_delta -= dirty_states_count * 15.0

            tentative_pass = min(100.0, current_pass_pct + max(0.0, progress_delta))

            # 确定性验证器检查
            if has_deterministic_verifier:
                if tentative_pass >= 98.0:
                    current_pass_pct = 100.0
                    verifier_status = "PASS [DETERMINISTIC_VERIFIER_GREEN]"
                    is_success = True
                    terminal_status = "SUCCESS_CONVERGED"
                    action_desc = "执行代码修改并运行测试套件，外部确定性验证门禁全绿通过！"
                else:
                    verifier_status = (
                        f"FAIL [ASSERTION_ERROR: {100.0 - tentative_pass:.0f}% REMAINING]"
                    )
                    current_pass_pct = tentative_pass
                    if has_state_rollback:
                        rollback_count += 1
                        action_desc = (
                            "外部测试门禁拦截到失败用例，执行隔离环境原子回滚，"
                            "并将失败堆栈转化为反思提示词注入下一轮迭代。"
                        )
                    else:
                        dirty_states_count += 1
                        action_desc = (
                            f"测试失败但未开启状态回滚，未通过的代码残留在工作区，"
                            f"脏状态等级: {dirty_states_count}。"
                        )
            else:
                current_pass_pct = tentative_pass
                verifier_status = "SKIPPED [NO_VERIFIER_CONFIGURED]"
                action_desc = (
                    f"智能体根据自身判断继续编辑代码，累计推进度 {current_pass_pct:.0f}%。"
                )
                if step >= target_steps:
                    is_success = bool(rng.random() < cfg["assumed_success_rate"])
                    terminal_status = "SUCCESS_CONVERGED" if is_success else "SILENT_FAILURE"

            trace_steps.append(
                {
                    "step": step,
                    "phase": f"ITERATION_{step}",
                    "action": action_desc,
                    "verifier_status": verifier_status,
                    "test_pass_pct": current_pass_pct,
                    "step_tokens": step_token,
                    "cumulative_tokens": cumulative_tokens,
                    "state_clean": dirty_states_count == 0,
                }
            )

            if is_success:
                break

            # 脏状态过多级联崩溃
            if dirty_states_count >= 3:
                terminal_status = "CASCADE_ERROR_ABORT"
                trace_steps.append(
                    {
                        "step": step + 1,
                        "phase": "CASCADE_FAILURE",
                        "action": "未回滚的脏代码导致依赖破坏与语法错误级联发散，智能体陷入死循环崩溃！",
                        "verifier_status": "CRITICAL_ERROR",
                        "test_pass_pct": current_pass_pct,
                        "step_tokens": 0,
                        "cumulative_tokens": cumulative_tokens,
                        "state_clean": False,
                    }
                )
                break

        return {
            "evidence_level": "SIMULATION",
            "is_rule_simulation": True,
            "pattern": pattern,
            "pattern_name": cfg["name"],
            "task_difficulty": task_difficulty,
            "is_success": is_success,
            "terminal_status": terminal_status,
            "iterations_used": len(trace_steps),
            "total_tokens": cumulative_tokens,
            "rollback_count": rollback_count,
            "final_pass_pct": current_pass_pct,
            "trace_steps": trace_steps,
            "assumptions": [
                "基于软件工程 Agent 常见模式构建的教学离散仿真模型",
                "用于对比验证门禁、回滚机制与无验证器自判完成的系统收敛差异",
            ],
            "boundary": (
                "仿真结果不代表具体大模型在任意生产代码库中的实际胜率；"
                "生产环境中的状态回退应使用隔离工作树 (Worktree) 或容器快照，避免覆盖用户本地未提交修改。"
            ),
        }
