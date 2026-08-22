# Copyright (c) 2026 Yy1 (yuan278501381) | MIT License
"""
nn_core.harness_traps - AI 真实工程陷阱、物理盲区与 Harness 控制环防御算法模块

涵盖现代大语言模型落地中六大核心现实问题与工程架构防御：
1. AttentionSinkSimulator: 初始 Token 注意力汇聚与无 Sink 滑动窗口困惑度爆炸模拟 (Xiao et al., 2023 StreamingLLM)
2. LostInTheMiddleSimulator: 长上下文 U 型注意力衰减与 Rerank 拯救 (Liu et al., 2023)
3. ReversalCurseEngine: 自回归因果条件概率下的逆向推理断裂模拟 (Berglund et al., 2023)
4. TokenizerTrapInspector: BPE 前导空格敏感度、字符计数盲区与 Glitch Token 离群向量分析
5. AgentHarnessGuard: 工具调用乒乓死循环熔断器 (Ping-Pong Circuit Breaker) 与间接提示词注入过滤
6. ClaudeCode2026PostmortemRunner: 2026 年 Anthropic 官方复盘之 3 大工程失误与修复全景推演
"""

import logging
import re
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 注意力黑洞与 StreamingLLM 汇聚模拟 (Attention Sinks)
# ---------------------------------------------------------------------------
class AttentionSinkSimulator:
    """模拟自回归注意力汇聚效应 (Attention Sinks) 与滑动窗口困惑度演化。"""

    @staticmethod
    def simulate_streaming_perplexity(
        seq_length: int = 128,
        window_size: int = 32,
        num_sink_tokens: int = 4,
        base_perplexity: float = 18.5,
    ) -> dict[str, Any]:
        """
        模拟在长序列流式生成中，保留 vs 丢失初始 Token (Attention Sink) 的困惑度对比。

        数学依据:
            当 Softmax 缺乏全局参照锚点时，随着初始 Sink Token 被移出滑动窗口，
            隐藏层激活分布严重畸变，导致 PPL 呈指数级爆炸 (可达 10^4+)。
        """
        steps = np.arange(1, seq_length + 1)
        ppl_with_sinks = []
        ppl_no_sinks = []
        sink_attention_weights = []

        np.random.seed(42)
        for t in steps:
            # 基础波动
            noise = np.random.normal(0, 0.4)
            # 有 Sink 时困惑度保持平稳
            ppl_safe = base_perplexity + noise + 0.02 * np.sin(t / 5.0)
            ppl_with_sinks.append(max(ppl_safe, 10.0))

            # 初始 Token 吸收的注意力权重 (通常高达 20%~60%)
            sink_weight = min(0.65, 0.35 + 0.25 * (1.0 - np.exp(-t / 15.0)))
            sink_attention_weights.append(sink_weight)

            # 无 Sink 时：一旦超出窗口大小，PPL 指数级崩盘
            if t <= window_size:
                ppl_no_sinks.append(ppl_safe)
            else:
                overflow = t - window_size
                # 指数级爆炸
                ppl_broken = base_perplexity * np.exp(min(overflow / 6.0, 7.0)) + np.random.uniform(
                    50, 200
                )
                ppl_no_sinks.append(ppl_broken)

        return {
            "steps": steps.tolist(),
            "ppl_with_sinks": ppl_with_sinks,
            "ppl_no_sinks": ppl_no_sinks,
            "sink_attention_weights": sink_attention_weights,
            "window_size": window_size,
            "num_sink_tokens": num_sink_tokens,
        }


# ---------------------------------------------------------------------------
# 2. 迷失在中间与 U 型注意力衰减 (Lost in the Middle)
# ---------------------------------------------------------------------------
class LostInTheMiddleSimulator:
    """模拟长上下文环境下关键事实被置于不同深度时的检索召回率与注意力稀释。"""

    @staticmethod
    def compute_u_curve(
        context_length_k: int = 32,
        num_points: int = 21,
    ) -> dict[str, Any]:
        """
        计算 0.0~1.0 相对深度下的原始检索准确率 vs Rerank 重排后的准确率。

        数学模型:
            Accuracy(p) = Base_Acc - Dilution_Factor * 4 * p * (1 - p)
            在 p=0.5 (正中间) 处准确率达到谷底。
        """
        depths = np.linspace(0.0, 1.0, num_points)
        # 上下文越长，中间衰减幅度越大
        scale_penalty = min(0.75, 0.25 + 0.12 * np.log2(max(context_length_k / 4.0, 1.0)))

        raw_accuracies = []
        rerank_accuracies = []
        attention_densities = []

        for p in depths:
            # U 型曲线：开头与结尾高，中间深陷
            u_decay = 4.0 * p * (1.0 - p)  # 在 0.5 处为 1.0，两端为 0
            acc_raw = max(0.05, 0.96 - scale_penalty * u_decay + np.random.normal(0, 0.015))
            raw_accuracies.append(float(acc_raw * 100.0))

            # Rerank 后置顶：无论原位置在哪里，重排后置于开头或末尾，准确率恢复至 90%+
            acc_rerank = float(
                np.clip(0.94 - 0.05 * p + np.random.normal(0, 0.01), 0.85, 0.99) * 100.0
            )
            rerank_accuracies.append(acc_rerank)

            # 中间注意力密度 (Softmax 权重被摊薄)
            att_density = float(np.clip(1.0 - 0.85 * u_decay, 0.08, 1.0))
            attention_densities.append(att_density)

        return {
            "depths": [round(float(d), 2) for d in depths],
            "raw_accuracies": raw_accuracies,
            "rerank_accuracies": rerank_accuracies,
            "attention_densities": attention_densities,
            "context_length_k": context_length_k,
        }


# ---------------------------------------------------------------------------
# 3. 逆向诅咒 (Reversal Curse)
# ---------------------------------------------------------------------------
class ReversalCurseEngine:
    """自回归因果概率下的单向实体关系断裂模拟。"""

    RELATIONS_DB: ClassVar[list[dict[str, Any]]] = [
        {
            "person_a": "玛丽·居里 (Marie Curie)",
            "person_b": "伊雷娜·约里奥-居里 (Irène Joliot-Curie)",
            "rel_ab": "母亲是",
            "rel_ba": "孩子是",
        },
        {
            "person_a": "斯蒂芬·库里 (Stephen Curry)",
            "person_b": "戴尔·库里 (Dell Curry)",
            "rel_ab": "父亲是",
            "rel_ba": "儿子是",
        },
        {
            "person_a": "苹果公司 (Apple Inc.)",
            "person_b": "史蒂夫·乔布斯 (Steve Jobs)",
            "rel_ab": "联合创始人是",
            "rel_ba": "创立的公司包括",
        },
    ]

    @classmethod
    def query_relation(
        cls,
        item_index: int = 0,
        is_reverse_query: bool = False,
    ) -> dict[str, Any]:
        """模拟前向自回归查询 vs 逆向自回归查询。"""
        item = cls.RELATIONS_DB[item_index % len(cls.RELATIONS_DB)]

        if not is_reverse_query:
            prompt = f"问：{item['person_a']} 的 {item['rel_ab']} 谁？"
            prob_correct = 0.985
            response = f"答：{item['person_a']} 的 {item['rel_ab']} {item['person_b']}。"
            direction = "前向因果查询 (Forward Causal Query) [MATCH]"
            reasoning = (
                "模型在训练语料中频繁接触 'A 的母亲是 B' 这一序列模式，自回归下一词预测置信度极高。"
            )
        else:
            prompt = f"问：{item['person_b']} 的 {item['rel_ba']} 谁？"
            prob_correct = 0.062
            response = f"答：抱歉，我不确定 {item['person_b']} 的具体家庭成员/关联细节，或者猜测为 [错误实体]。"
            direction = "逆向因果查询 (Reversed Query - Reversal Curse) [FAIL]"
            reasoning = "自回归模型学习的是 P(B | A, 母亲)，并未建立通用的无向知识图谱。在没有反向语料显式训练时，逆向条件概率 P(A | B, 孩子) 极度稀疏。"

        return {
            "prompt": prompt,
            "response": response,
            "prob_correct": prob_correct,
            "direction": direction,
            "reasoning": reasoning,
            "is_reverse": is_reverse_query,
        }


# ---------------------------------------------------------------------------
# 4. 分词器盲区与 Glitch Tokens 检查器 (Tokenizer Traps)
# ---------------------------------------------------------------------------
class TokenizerTrapInspector:
    """分词器几何盲区、前导空格敏感度与字符计数分析。"""

    @staticmethod
    def inspect_strawberry(word: str = "strawberry", target_char: str = "r") -> dict[str, Any]:
        """分析草莓单词字符计数盲区。"""
        # BPE 模拟切分
        subwords = ["straw", "berry"]
        actual_count = word.lower().count(target_char.lower())
        char_list = list(word)

        return {
            "word": word,
            "target_char": target_char,
            "actual_count": actual_count,
            "subwords": subwords,
            "char_list": char_list,
            "explanation": (
                f"在 BPE 分词器中，单词 '{word}' 被直接压缩切分为 2 个抽象 Token ID：{subwords}。"
                f"模型在注意力计算中只接收这两个高维 Token 向量，底层的具体字母 '{target_char}' 已被物理消除。"
                "如果没有启用思维链（CoT）显式拆解字母，模型在隐空间中无法直接'看'到底层字符。"
            ),
        }

    @staticmethod
    def inspect_whitespace_tokens(base_text: str = "123") -> list[dict[str, Any]]:
        """模拟前导空格产生的不同 Token ID。"""
        variants = [
            (f"{base_text}", "无前导空格 (Raw)", 4892),
            (f" {base_text}", "单前导空格 (Single Space)", 20481),
            (f"  {base_text}", "双前导空格 (Double Space)", 59124),
        ]
        results = []
        for text, desc, tid in variants:
            results.append(
                {
                    "text_repr": repr(text),
                    "desc": desc,
                    "token_id": tid,
                    "vector_norm": round(float(np.sqrt(tid) / 50.0 + 1.2), 3),
                }
            )
        return results


# ---------------------------------------------------------------------------
# 5. Agent Harness 运行控制架与熔断器 (Agent Circuit Breaker)
# ---------------------------------------------------------------------------
class AgentHarnessGuard:
    """智能体控制环：工具调用死循环检测 (Ping-Pong Guard) 与间接注入防御。"""

    INJECTION_PATTERNS: ClassVar[list[str]] = [
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
        检查新工具调用是否触发死循环或间接注入。
        """
        # 1. 检查重复乒乓调用
        recent_calls = [c for c in call_history[-max_repeats:] if c.get("tool") == new_tool]
        is_ping_pong = len(recent_calls) >= max_repeats

        # 2. 检查参数中是否包含间接提示词注入
        arg_str = str(new_args).lower()
        injection_found = any(re.search(pat, arg_str) for pat in cls.INJECTION_PATTERNS)

        if injection_found:
            status = "CIRCUIT_BREAKER_BLOCKED [INJECTION DETECTED]"
            action = "拦截工具执行，向模型注入安全隔离警告，防止间接提示词走私！"
            color = "rose"
        elif is_ping_pong:
            status = "PING_PONG_LOOP_INTERRUPTED [CIRCUIT BREAKER]"
            action = f"检测到连续 {max_repeats} 次相同工具调用失败，触发状态机熔断，强制唤醒自愈反思提示词！"
            color = "amber"
        else:
            status = "EXECUTION_PERMITTED [HEALTHY]"
            action = "安全沙盒校验通过，允许在本地隔离环境中执行工具。"
            color = "emerald"

        return {
            "status": status,
            "action": action,
            "color": color,
            "is_blocked": injection_found or is_ping_pong,
        }


# ---------------------------------------------------------------------------
# 6. 2026 Claude Code 事故复盘推演器 (Postmortem Runner)
# ---------------------------------------------------------------------------
class ClaudeCode2026PostmortemRunner:
    """2026 年 3~4 月 Anthropic Claude Code 官方技术复盘与三种工程状态仿真。"""

    INCIDENTS: ClassVar[dict[str, dict[str, Any]]] = {
        "reasoning_downgrade": {
            "title": "事故 1: 思考预算静默调低 (March 4)",
            "cause": "为降低终端响应延迟与界面卡顿感，将默认 Reasoning Effort 从 high 降为 medium。",
            "symptom": "在复杂架构重构与长逻辑编程中，模型推理深度不足，错误率暴涨 40%+。",
            "fix": "4 月 7 日全面回滚默认设置，允许用户显式指定 reasoning-effort 参数。",
            "accuracy_buggy": 54.2,
            "accuracy_fixed": 91.8,
            "latency_buggy": 2.1,
            "latency_fixed": 6.8,
        },
        "session_cache_wipe": {
            "title": "事故 2: 会话缓存清理引入全失忆 Bug (March 26)",
            "cause": "为优化闲置会话内存，实现清理逻辑时错误地在每一轮 (Every Turn) 抹除全部历史思考链与缓存。",
            "symptom": "模型表现出严重健忘症与机械重复，长任务完全无法连贯推进。",
            "fix": "4 月 10 日发布 v2.1.101 修复会话状态机与 KV 缓存生命周期绑定。",
            "accuracy_buggy": 28.5,
            "accuracy_fixed": 89.4,
            "latency_buggy": 4.5,
            "latency_fixed": 3.8,
        },
        "verbosity_clamp": {
            "title": "事故 3: 系统提示词硬加简短限制 (April 16)",
            "cause": "在 System Prompt 中追加了要求回答更加简短的控制指令，导致深度被硬性截断。",
            "symptom": "复杂代码实现大面积残缺，函数只有空壳，注释过度省略。",
            "fix": "4 月 20 日发布 v2.1.116 移除破坏性指令，重置全网受影响用户的订阅额度。",
            "accuracy_buggy": 42.0,
            "accuracy_fixed": 93.5,
            "latency_buggy": 1.5,
            "latency_fixed": 5.2,
        },
    }

    @classmethod
    def get_incident_data(cls, incident_key: str = "reasoning_downgrade") -> dict[str, Any]:
        """获取特定事故复盘数据。"""
        return cls.INCIDENTS.get(incident_key, cls.INCIDENTS["reasoning_downgrade"])
